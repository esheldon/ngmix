"""
Fit an image with a gaussian mixture using the EM algorithm
"""
import logging
import numpy as np
from numba import njit

from .gexceptions import GMixRangeError, GMixMaxIterEM
from .priors import srandu
from .jacobian import Jacobian, UnitJacobian, DiagonalJacobian
from .observation import Observation

from .gmix import GMix, GMixModel
from .gmix_nb import (
    gauss2d_set,
    gmix_set_norms,
    gmix_get_cen,
    gmix_convolve_fill,
    gmix_eval_pixel_fast,
    GMIX_LOW_DETVAL,
)

from .fastexp import exp5

logger = logging.getLogger(__name__)


EM_RANGE_ERROR = 2 ** 0
EM_MAXITER = 2 ** 1


def fit_em(obs, guess, **keys):
    """
    fit the observation with EM
    """
    im, sky = prep_image(obs.image)
    newobs = Observation(im, jacobian=obs.jacobian)
    fitter = GMixEM(newobs)
    fitter.go(guess, sky, **keys)

    return fitter


def prep_image(im0):
    """
    Prep an image to fit with EM.  Make sure there are no pixels < 0

    parameters
    ----------
    image: ndarray
        2d image

    output
    ------
    new_image, sky:
        The image with new background level and the background level
    """
    im = im0.copy()

    # need no zero pixels and sky value
    im_min = im.min()
    im_max = im.max()

    desired_minval = 0.001 * (im_max - im_min)

    sky = desired_minval - im_min
    im += sky

    return im, sky


class GMixEM(object):
    """
    Fit an image with a gaussian mixture using the EM algorithm

    Parameters
    ----------
    obs: Observation
        An ngmix.Observation object

        The image should not have zero or negative pixels. You can
        use the ngmix.em.prep_image() function to ensure this.
    minimum: number, optional
        The minimum number of iterations, default 10
    maxiter: number, optional
        The maximum number of iterations, default 1000
    tol: number, optional
        The tolerance in the moments that implies convergence,
        default 0.001
    vary_sky: bool
        If True, fit for the sky level
    """
    def __init__(self,
                 obs,
                 miniter=40,
                 maxiter=500,
                 tol=0.001,
                 vary_sky=False):

        self._obs = obs

        self.miniter = miniter
        self.maxiter = maxiter
        self.tol = tol
        self.vary_sky = vary_sky

        self._sums = None
        self._result = None

        self._set_runner()

    def has_gmix(self):
        """
        returns True if a gmix is set
        """
        if hasattr(self, '_gm'):
            return True
        else:
            return False

    def get_gmix(self):
        """
        Get a copy of the gaussian mixture from the final iteration
        """
        if not self.has_gmix():
            raise RuntimeError('no gmix set')

        return self._gm.copy()

    def get_convolved_gmix(self):
        """
        Get the gaussian mixture from the final iteration
        """
        if not self.has_gmix():
            raise RuntimeError('no gmix set')

        return self._gm_conv.copy()

    def get_result(self):
        """
        Get some stats about the processing
        """
        return self._result

    def make_image(self):
        """
        Get an image of the best fit mixture
        """
        return self._gm.make_image(
            self._obs.image.shape,
            jacobian=self._obs.jacobian,
        )

    def go(self, gmix_guess, sky):
        """
        Run the em algorithm from the input starting guesses

        parameters
        ----------
        gmix_guess: GMix
            A gaussian mixture (GMix or child class) representing a starting
            guess for the algorithm.  This should be *before* psf convolution.
        sky: number
            The sky value added to the image
        """

        if hasattr(self, '_gm'):
            del self._gm
            del self._gm_conv

        obs = self._obs

        # makes a copy
        if not obs.has_psf() or not obs.psf.has_gmix():
            logger.debug('NO PSF SET')
            gmix_psf = GMixModel([0., 0., 0., 0., 0., 1.0], 'gauss')
        else:
            gmix_psf = obs.psf.gmix
            gmix_psf.set_flux(1.0)

        conf = self._make_conf()
        conf['sky'] = sky

        gm = gmix_guess.copy()
        gm_conv = gm.convolve(gmix_psf)

        sums = self._make_sums(len(gm))

        pixels = obs.pixels.copy()

        if np.any(pixels['ierr'] <= 0.0):
            fill_zero_weight = True
        else:
            fill_zero_weight = False

        flags = 0
        try:
            numiter, fdiff, sky = self._runner(
                conf,
                pixels,
                sums,
                gm.get_data(),
                gmix_psf.get_data(),
                gm_conv.get_data(),
                fill_zero_weight=fill_zero_weight,
            )

            pars = gm.get_full_pars()
            pars_conv = gm_conv.get_full_pars()
            self._gm = GMix(pars=pars)
            self._gm_conv = GMix(pars=pars_conv)

            if numiter >= self.maxiter:
                flags = EM_MAXITER
                message = 'maxit'
            else:
                message = 'OK'

            result = {
                'flags': flags,
                'numiter': numiter,
                'fdiff': fdiff,
                'sky': sky,
                'message': message,
            }

        except (GMixRangeError, ZeroDivisionError) as err:
            message = str(err)
            logger.info(message)
            result = {
                'flags': EM_RANGE_ERROR,
                'message': message,
            }

        self._result = result

    def _make_sums(self, ngauss):
        """
        make the sum structure
        """
        return np.zeros(ngauss, dtype=_sums_dtype)

    def _make_conf(self):
        """
        make the sum structure
        """
        conf = np.zeros(1, dtype=_em_conf_dtype)
        conf = conf[0]

        conf['tol'] = self.tol
        conf['miniter'] = self.miniter
        conf['maxiter'] = self.maxiter
        conf['pixel_scale'] = self._obs.jacobian.scale
        conf['vary_sky'] = self.vary_sky

        return conf

    def _set_runner(self):
        self._runner = em_run


class GMixEMFixCen(GMixEM):
    """
    Fit an image with a gaussian mixture using the EM algorithm

    Parameters
    ----------
    obs: Observation
        An ngmix.Observation object

        The image should not have zero or negative pixels. You can
        use the ngmix.em.prep_image() function to ensure this.
    minimum: number, optional
        The minimum number of iterations, default 10
    maxiter: number, optional
        The maximum number of iterations, default 1000
    tol: number, optional
        The tolerance in the moments that implies convergence,
        default 0.001
    vary_sky: bool
        If True, fit for the sky level
    """

    def _make_sums(self, ngauss):
        """
        make the sum structure
        """
        return np.zeros(ngauss, dtype=_sums_dtype_fixcen)

    def _set_runner(self):
        self._runner = em_run_fixcen


class GMixEMPOnly(GMixEMFixCen):
    """
    Fit an image with a gaussian mixture using the EM algorithm,
    allowing only the fluxes to vary

    Parameters
    ----------
    obs: Observation
        An Observation object, containing the image and possibly
        non-trivial jacobian.  see ngmix.observation.Observation

        The image should not have zero or negative pixels. You can
        use the prep_image() function to ensure this.
    minimum: number, optional
        The minimum number of iterations, default 10
    maxiter: number, optional
        The maximum number of iterations, default 1000
    tol: number, optional
        The tolerance in the fluxes that implies convergence,
        default 0.001
    """

    def __init__(self,
                 obs,
                 miniter=20,
                 maxiter=500,
                 tol=0.001,
                 vary_sky=False):
        """
        over-riding because we want a different default miniter
        """
        super(GMixEMPOnly, self).__init__(
            obs,
            miniter=miniter,
            maxiter=maxiter,
            tol=tol,
            vary_sky=vary_sky,
        )

    def _set_runner(self):
        self._runner = em_run_ponly

    def _make_sums(self, ngauss):
        """
        make the sum structure
        """
        return np.zeros(ngauss, dtype=_sums_dtype_ponly)


@njit
def em_run(conf,
           pixels,
           sums,
           gmix,
           gmix_psf,
           gmix_conv,
           fill_zero_weight=False):
    """
    run the EM algorithm, with fixed positions and a psf mixture to provide a
    shapred minimim resolution

    Parameters
    ----------
    conf: array
        Should have fields

            tol: tolerance for stopping
            miniter: minimum number of iterations
            maxiter: maximum number of iterations
            pixel_scale: pixel scale
            sky: the sky, or guess for sky if fitting for it
            vary_sky: True if fitting for the sky

    pixels: pixel array
        for the image/jacobian
    sums: array with fields
        The sums array, a type _sums_dtype_fixcen
    gmix: gauss2d array
        The initial mixture.  The final result is also stored in this array.
    gmix_psf: gauss2d array
        Single gaussian psf
    gmix_conv: gauss2d array
        Convolved gmix
    fill_zero_weight: bool
        If True, fill the zero weight pixels with the model on
        each iteration
    """

    gmix_set_norms(gmix_conv)
    ngauss_psf = gmix_psf.size

    taudata = np.zeros(gmix_conv.size, dtype=_tau_dtype)

    tol = conf['tol']

    pix_area = conf['pixel_scale']*conf['pixel_scale']
    npix = pixels.size

    sky = conf['sky']

    elogL_last = -9999.9e9

    for i in range(conf['maxiter']):

        elogL = 0.0
        skysum = 0.0

        clear_sums(sums)
        set_logtau_logdet(gmix_conv, taudata)

        if fill_zero_weight:
            fill_zero_weight_pixels(gmix_conv, pixels, sky)

        for pixel in pixels:

            gsum, tlogL = do_scratch_sums(
                pixel, gmix_conv, sums, ngauss_psf,
                taudata,
            )

            gtot = gsum + sky
            if gtot == 0.0:
                raise GMixRangeError('gtot == 0')

            elogL += tlogL

            skysum += sky*pixel['val']/gtot

            do_sums(sums, pixel, gtot)

        gmix_set_from_sums(
            gmix,
            gmix_psf,
            gmix_conv,
            sums,
            pix_area,
        )

        if conf['vary_sky']:
            sky = skysum/npix
            # print('sky_orig:', conf['sky'], 'sky:', sky)

        numiter = i+1
        if numiter >= conf['miniter']:

            if elogL == 0.0:
                raise GMixRangeError('elogL == 0')

            frac_diff = abs((elogL - elogL_last)/elogL)
            if frac_diff < tol:
                break

        elogL_last = elogL

    # we have modified the mixture and not set the norms, and we don't want to
    # set them for the pre-psf mixture

    gmix['norm_set'][:] = 0

    return numiter, frac_diff, sky


@njit
def clear_sums(sums):
    """
    set all sums to zero
    """
    sums['gi'][:] = 0.0

    sums['tusum'][:] = 0.0
    sums['tvsum'][:] = 0.0

    sums['tu2sum'][:] = 0.0
    sums['tuvsum'][:] = 0.0
    sums['tv2sum'][:] = 0.0

    # sums over all pixels
    sums['pnew'][:] = 0.0

    sums['usum'][:] = 0.0
    sums['vsum'][:] = 0.0

    sums['u2sum'][:] = 0.0
    sums['uvsum'][:] = 0.0
    sums['v2sum'][:] = 0.0


@njit
def do_scratch_sums(pixel, gmix_conv, sums, ngauss_psf, taudata):
    """
    do the basic sums for this pixel, using scratch space in the sums struct

    we may have multiple components per "object" so we update the
    sums accordingly
    """

    v = pixel['v']
    u = pixel['u']

    gsum = 0.0
    logL = 0.0

    ngauss = gmix_conv.size//ngauss_psf

    for ii in range(ngauss):
        tsums = sums[ii]
        tsums['gi'] = 0.0
        tsums['tvsum'] = 0.0
        tsums['tusum'] = 0.0
        tsums['tv2sum'] = 0.0
        tsums['tuvsum'] = 0.0
        tsums['tu2sum'] = 0.0

        start = ii*ngauss_psf
        end = (ii+1)*ngauss_psf

        for i in range(start, end):
            gauss = gmix_conv[i]
            ttau = taudata[i]

            # in practice we have co-centric gaussians even after psf
            # convolution, so we could move these to the outer loop

            vdiff = v - gauss['row']
            udiff = u - gauss['col']

            u2 = udiff*udiff
            v2 = vdiff*vdiff
            uv = udiff*vdiff

            chi2 = gauss['dcc']*v2 + gauss['drr']*u2 - 2.0*gauss['drc']*uv

            if chi2 < 25.0 and chi2 >= 0.0:
                val = gauss['pnorm']*exp5(-0.5*chi2)
            else:
                val = 0.0

            tsums['gi'] += val
            gsum += val

            tsums['tvsum'] += v*val
            tsums['tusum'] += u*val

            tsums['tv2sum'] += v2*val
            tsums['tuvsum'] += uv*val
            tsums['tu2sum'] += u2*val

            logL += val*(ttau['logtau'] - 0.5*ttau['logdet'] - 0.5*chi2)

    if gsum == 0.0:
        logL = 0.0
    else:
        logL *= 1.0/gsum

    return gsum, logL


@njit
def do_sums(sums, pixel, gtot):
    """
    do the sums based on the scratch values
    """

    factor = pixel['val']/gtot

    n_gauss = sums.size
    for i in range(n_gauss):
        tsums = sums[i]

        wtau = tsums['gi']*factor

        tsums['pnew'] += wtau

        # row*gi/gtot*imnorm

        tsums['usum'] += tsums['tusum']*factor
        tsums['vsum'] += tsums['tvsum']*factor

        tsums['u2sum'] += tsums['tu2sum']*factor
        tsums['uvsum'] += tsums['tuvsum']*factor
        tsums['v2sum'] += tsums['tv2sum']*factor


@njit
def gmix_set_from_sums(gmix,
                       gmix_psf,
                       gmix_conv,
                       sums,
                       pix_area):
    """
    fill the gaussian mixture from the em sums, requiring that the covariance
    matrix before psf deconvolution is not singular

    We may want to relax that
    """

    minval = 1.0e-4

    _, _, psf_irr, psf_irc, psf_icc, _ = gmix_get_moms(gmix_psf)

    n_gauss = gmix.size
    for i in range(n_gauss):

        tsums = sums[i]
        gauss = gmix[i]

        p = tsums['pnew']
        pinv = 1.0/p

        # update for convolved gaussian
        v = tsums['vsum']*pinv
        u = tsums['usum']*pinv
        irr = tsums['v2sum']*pinv
        irc = tsums['uvsum']*pinv
        icc = tsums['u2sum']*pinv

        # get pre-psf moments, only works if the psf gaussians
        # are all centered
        irr = irr - psf_irr
        irc = irc - psf_irc
        icc = icc - psf_icc

        # currently are forcing the sizes of pre-psf gaussians to
        # be positive.  We may be able to relax this

        if irr < 0.0 or icc < 0.0:
            irr, irc, icc = minval, 0.0, minval

        # this causes oscillations in likelihood
        det = irr*icc - irc**2
        if det < GMIX_LOW_DETVAL:
            T = irr + icc
            irr = icc = T/2
            irc = 0.0

        # ngmix works in surface brightness, so multiply p by area since it
        # has the actual image value in there

        gauss2d_set(
            gauss,
            p*pix_area,
            v,
            u,
            irr,
            irc,
            icc,
        )

    gmix_convolve_fill(gmix_conv, gmix, gmix_psf)
    gmix_set_norms(gmix_conv)


@njit
def em_run_fixcen(conf,
                  pixels,
                  sums,
                  gmix,
                  gmix_psf,
                  gmix_conv,
                  fill_zero_weight=False):
    """
    run the EM algorithm, with fixed positions and a psf mixture to provide a
    shapred minimim resolution

    Parameters
    ----------
    conf: array
        Should have fields

            tol: tolerance for stopping
            miniter: minimum number of iterations
            maxiter: maximum number of iterations
            pixel_scale: pixel scale
            sky: the sky, or guess for sky if fitting for it
            vary_sky: True if fitting for the sky

    pixels: pixel array
        for the image/jacobian
    sums: array with fields
        The sums array, a type _sums_dtype_fixcen
    gmix: gauss2d array
        The initial mixture.  The final result is also stored in this array.
    gmix_psf: gauss2d array
        Single gaussian psf
    gmix_conv: gauss2d array
        Convolved gmix
    fill_zero_weight: bool
        If True, fill the zero weight pixels with the model on
        each iteration
    """

    gmix_set_norms(gmix_conv)
    ngauss_psf = gmix_psf.size

    taudata = np.zeros(gmix_conv.size, dtype=_tau_dtype)

    tol = conf['tol']

    pix_area = conf['pixel_scale']*conf['pixel_scale']
    npix = pixels.size

    sky = conf['sky']

    elogL_last = -9999.9e9

    for i in range(conf['maxiter']):

        elogL = 0.0
        skysum = 0.0

        clear_sums_fixcen(sums)
        set_logtau_logdet(gmix_conv, taudata)

        if fill_zero_weight:
            fill_zero_weight_pixels(gmix_conv, pixels, sky)

        for pixel in pixels:

            gsum, tlogL = do_scratch_sums_fixcen(
                pixel, gmix_conv, sums, ngauss_psf,
                taudata,
            )

            gtot = gsum + sky
            if gtot == 0.0:
                raise GMixRangeError('gtot == 0')

            elogL += tlogL

            skysum += sky*pixel['val']/gtot

            do_sums_fixcen(sums, pixel, gtot)

        gmix_set_from_sums_fixcen(
            gmix,
            gmix_psf,
            gmix_conv,
            sums,
            pix_area,
        )

        if conf['vary_sky']:
            sky = skysum/npix
            # print('sky_orig:', conf['sky'], 'sky:', sky)

        numiter = i+1
        if numiter >= conf['miniter']:

            if elogL == 0.0:
                raise GMixRangeError('elogL == 0')

            frac_diff = abs((elogL - elogL_last)/elogL)
            if frac_diff < tol:
                break

        elogL_last = elogL

    # we have modified the mixture and not set the norms, and we don't want to
    # set them for the pre-psf mixture

    gmix['norm_set'][:] = 0

    return numiter, frac_diff, sky


@njit
def fill_zero_weight_pixels(gmix, pixels, sky):
    """
    fill zero weight pixels with the model
    """

    for pixel in pixels:
        if pixel['ierr'] <= 0.0:
            val = gmix_eval_pixel_fast(gmix, pixel)
            pixel['val'] = sky + val


@njit
def do_scratch_sums_fixcen(pixel, gmix_conv, sums, ngauss_psf, taudata):
    """
    do the basic sums for this pixel, using scratch space in the sums struct

    we may have multiple components per "object" so we update the
    sums accordingly
    """

    v = pixel['v']
    u = pixel['u']

    gsum = 0.0
    logL = 0.0

    ngauss = gmix_conv.size//ngauss_psf

    for ii in range(ngauss):
        tsums = sums[ii]
        tsums['gi'] = 0.0
        tsums['tv2sum'] = 0.0
        tsums['tuvsum'] = 0.0
        tsums['tu2sum'] = 0.0

        start = ii*ngauss_psf
        end = (ii+1)*ngauss_psf

        for i in range(start, end):
            gauss = gmix_conv[i]
            ttau = taudata[i]

            # in practice we have co-centric gaussians even after psf
            # convolution, so we could move these to the outer loop

            vdiff = v - gauss['row']
            udiff = u - gauss['col']

            u2 = udiff*udiff
            v2 = vdiff*vdiff
            uv = udiff*vdiff

            chi2 = gauss['dcc']*v2 + gauss['drr']*u2 - 2.0*gauss['drc']*uv

            if chi2 < 25.0 and chi2 >= 0.0:
                val = gauss['pnorm']*exp5(-0.5*chi2)
            else:
                val = 0.0

            tsums['gi'] += val
            gsum += val

            tsums['tv2sum'] += v2*val
            tsums['tuvsum'] += uv*val
            tsums['tu2sum'] += u2*val

            logL += val*(ttau['logtau'] - 0.5*ttau['logdet'] - 0.5*chi2)

    if gsum == 0.0:
        logL = 0.0
    else:
        logL *= 1.0/gsum

    return gsum, logL


@njit
def do_sums_fixcen(sums, pixel, gtot):
    """
    do the sums based on the scratch values
    """

    factor = pixel['val']/gtot

    n_gauss = sums.size
    for i in range(n_gauss):
        tsums = sums[i]

        wtau = tsums['gi']*factor

        tsums['pnew'] += wtau

        # row*gi/gtot*imnorm
        tsums['u2sum'] += tsums['tu2sum']*factor
        tsums['uvsum'] += tsums['tuvsum']*factor
        tsums['v2sum'] += tsums['tv2sum']*factor


@njit
def gmix_set_from_sums_fixcen(gmix,
                              gmix_psf,
                              gmix_conv,
                              sums,
                              pix_area):
    """
    fill the gaussian mixture from the em sums, requiring that the covariance
    matrix before psf deconvolution is not singular

    We may want to relax that
    """

    minval = 1.0e-4

    _, _, psf_irr, psf_irc, psf_icc, _ = gmix_get_moms(gmix_psf)

    n_gauss = gmix.size
    for i in range(n_gauss):

        tsums = sums[i]
        gauss = gmix[i]

        p = tsums['pnew']
        pinv = 1.0/p

        # update for convolved gaussian
        irr = tsums['v2sum']*pinv
        irc = tsums['uvsum']*pinv
        icc = tsums['u2sum']*pinv

        # get pre-psf moments, only works if the psf gaussians
        # are all centered
        irr = irr - psf_irr
        irc = irc - psf_irc
        icc = icc - psf_icc

        # currently are forcing the sizes of pre-psf gaussians to
        # be positive.  We may be able to relax this

        if irr < 0.0 or icc < 0.0:
            irr, irc, icc = minval, 0.0, minval

        # this causes oscillations in likelihood
        det = irr*icc - irc**2
        if det < GMIX_LOW_DETVAL:
            T = irr + icc
            irr = icc = T/2
            irc = 0.0

        # ngmix works in surface brightness, so multiply p by area since it
        # has the actual image value in there

        gauss2d_set(
            gauss,
            p*pix_area,
            gauss['row'],
            gauss['col'],
            irr,
            irc,
            icc,
        )

    gmix_convolve_fill(gmix_conv, gmix, gmix_psf)
    gmix_set_norms(gmix_conv)


@njit
def set_logtau_logdet(gmix, sums):
    """
    set log(tau) and log(det) for every gaussian
    """

    for i in range(gmix.size):
        gauss = gmix[i]
        tsums = sums[i]
        tsums['logtau'] = np.log(gauss['p'])
        tsums['logdet'] = np.log(gauss['det'])


@njit
def clear_sums_fixcen(sums):
    """
    set all sums to zero
    """
    sums['gi'][:] = 0.0

    sums['tu2sum'][:] = 0.0
    sums['tuvsum'][:] = 0.0
    sums['tv2sum'][:] = 0.0

    # sums over all pixels
    sums['pnew'][:] = 0.0
    sums['u2sum'][:] = 0.0
    sums['uvsum'][:] = 0.0
    sums['v2sum'][:] = 0.0


@njit
def em_run_ponly(conf,
                 pixels,
                 sums,
                 gmix,
                 gmix_psf,
                 gmix_conv,
                 fill_zero_weight=False):
    """
    run the EM algorithm, allowing only fluxes to vary

    Parameters
    ----------
    conf: array
        Should have fields
            tol: tolerance for stopping
            miniter: minimum number of iterations
            maxiter: maximum number of iterations
            pixel_scale: pixel scale
            sky: the sky, or guess for sky if fitting for it
            vary_sky: True if fitting for the sky

    pixels: pixel array
        for the image/jacobian
    sums: array with fields
        The sums array, a type _sums_dtype_ponly
    gmix: gauss2d array
        The initial mixture.  The final result is also stored in this array.
    gmix_psf: gauss2d array
        Single gaussian psf
    gmix_conv: gauss2d array
        Convolved gmix
    fill_zero_weight: bool
        If True, fill the zero weight pixels with the model on
        each iteration
    """

    gmix_set_norms(gmix_conv)
    ngauss_psf = gmix_psf.size

    tol = conf['tol']

    pix_area = conf['pixel_scale']*conf['pixel_scale']
    npix = pixels.size

    sky = conf['sky']

    p_last = gmix['p'].sum()

    for i in range(conf['maxiter']):
        skysum = 0.0
        clear_sums_ponly(sums)

        if fill_zero_weight:
            fill_zero_weight_pixels(gmix_conv, pixels, sky)

        for pixel in pixels:

            # this fills some fields of sums, as well as return
            gsum = do_scratch_sums_ponly(pixel, gmix_conv, sums, ngauss_psf)

            gtot = gsum + sky
            if gtot == 0.0:
                raise GMixRangeError("gtot == 0")

            skysum += sky*pixel['val']/gtot

            do_sums_ponly(sums, pixel, gtot)

        gmix_set_from_sums_ponly(
            gmix,
            gmix_psf,
            gmix_conv,
            sums,
            pix_area,
        )

        if conf['vary_sky']:
            sky = skysum/npix
            # print('sky_orig:', conf['sky'], 'sky:', sky)

        psum = gmix['p'].sum()

        numiter = i+1
        if numiter >= conf['miniter']:
            frac_diff = abs(psum/p_last-1)
            if frac_diff < tol:
                break

        p_last = psum

    # we have modified the mixture and not set the norms, and we don't want to
    # set them for the pre-psf mixture

    gmix['norm_set'][:] = 0

    return numiter, frac_diff, sky


@njit
def do_scratch_sums_ponly(pixel, gmix_conv, sums, ngauss_psf):
    """
    do the basic sums for this pixel, using
    scratch space in the sums struct
    """

    v = pixel['v']
    u = pixel['u']

    gsum = 0.0

    ngauss = gmix_conv.size//ngauss_psf

    for ii in range(ngauss):
        tsums = sums[ii]
        tsums['gi'] = 0.0

        start = ii*ngauss_psf
        end = (ii+1)*ngauss_psf

        for i in range(start, end):
            gauss = gmix_conv[i]

            # in practice we have co-centric gaussians even after psf
            # convolution, so we could move these to the outer loop

            vdiff = v-gauss['row']
            udiff = u-gauss['col']

            u2 = udiff*udiff
            v2 = vdiff*vdiff
            uv = udiff*vdiff

            chi2 = gauss['dcc']*v2 + gauss['drr']*u2 - 2.0*gauss['drc']*uv

            if chi2 < 25.0 and chi2 >= 0.0:
                val = gauss['pnorm']*exp5(-0.5*chi2)
            else:
                val = 0.0

            tsums['gi'] += val
            gsum += val

    return gsum


@njit
def do_sums_ponly(sums, pixel, gtot):
    """
    do the sums based on the scratch values
    """

    factor = pixel['val']/gtot

    n_gauss = sums.size
    for i in range(n_gauss):
        tsums = sums[i]

        # gi*image_val/sum(gi)
        wtau = tsums['gi']*factor

        tsums['pnew'] += wtau


@njit
def gmix_set_from_sums_ponly(gmix,
                             gmix_psf,
                             gmix_conv,
                             sums,
                             pix_area):
    """
    fill the gaussian mixture from the em sums
    """

    n_gauss = gmix.size
    for i in range(n_gauss):

        tsums = sums[i]
        gauss = gmix[i]

        # ngmix works in surface brightness
        p = tsums['pnew']*pix_area

        gauss2d_set(
            gauss,
            p,
            gauss['row'],
            gauss['col'],
            gauss['irr'],
            gauss['irc'],
            gauss['icc'],
        )

    gmix_convolve_fill(gmix_conv, gmix, gmix_psf)
    gmix_set_norms(gmix_conv)


@njit
def clear_sums_ponly(sums):
    """
    set all sums to zero
    """
    sums['gi'][:] = 0.0

    # sums over all pixels
    sums['pnew'][:] = 0.0


@njit
def em_convolve_1gauss(gmix, gmix_psf, fix=False):
    psf_irr = gmix_psf['irr'][0]
    psf_irc = gmix_psf['irc'][0]
    psf_icc = gmix_psf['icc'][0]

    for i in range(gmix.size):
        gauss = gmix[i]

        irr = gauss['irr'] + psf_irr
        irc = gauss['irc'] + psf_irc
        icc = gauss['icc'] + psf_icc

        if fix:
            det = irr*icc - irc**2
            if det < GMIX_LOW_DETVAL:
                T = irr + icc
                irr = icc = T/2
                irc = 0.0

        gauss2d_set(
            gauss,
            gauss['p'],
            gauss['row'],
            gauss['col'],
            irr,
            irc,
            icc,
        )


@njit
def em_deconvolve_1gauss(gmix, gmix_psf):
    psf_irr = gmix_psf['irr'][0]
    psf_irc = gmix_psf['irc'][0]
    psf_icc = gmix_psf['icc'][0]

    for i in range(gmix.size):
        gauss = gmix[i]

        print('p:', gauss['p'])
        print('T before:', gauss['irr'] + gauss['icc'])
        gauss2d_set(
            gauss,
            gauss['p'],
            gauss['row'],
            gauss['col'],
            gauss['irr'] - psf_irr,
            gauss['irc'] - psf_irc,
            gauss['icc'] - psf_icc,
        )
        print('T after:', gauss['irr'] + gauss['icc'])


@njit
def get_flux(gsum, g2sum, gIsum):
    """
    calculate the flux from the cross-correlation sums
    """
    return gsum*gIsum/g2sum


@njit
def gmix_get_moms(gmix):
    """
    get row, col, irr, irc, icc, psum
    """

    row, col, psum = gmix_get_cen(gmix)

    irr = irc = icc = 0.0

    for i in range(gmix.size):
        gauss = gmix[i]

        rowdiff = gauss['row'] - row
        coldiff = gauss['col'] - col

        p = gauss['p']
        irr += p*(gauss['irr'] + rowdiff**2)
        irc += p*(gauss['irc'] + rowdiff*coldiff)
        icc += p*(gauss['icc'] + coldiff**2)

    irr /= psum
    irc /= psum
    icc /= psum

    return row, col, irr, irc, icc, psum


_em_conf_dtype = [
    ('tol', 'f8'),
    ('maxiter', 'i4'),
    ('miniter', 'i4'),
    ('sky', 'f8'),
    ('vary_sky', 'bool'),
    ('pixel_scale', 'f8'),
]
_em_conf_dtype = np.dtype(_em_conf_dtype, align=True)

_sums_dtype = [
    ('gi', 'f8'),

    # used for convergence tests only
    ('logtau', 'f8'),
    ('logdet', 'f8'),

    # scratch on a given pixel
    ('tvsum', 'f8'),
    ('tusum', 'f8'),

    ('tu2sum', 'f8'),
    ('tuvsum', 'f8'),
    ('tv2sum', 'f8'),

    # sums over all pixels
    ('pnew', 'f8'),
    ('vsum', 'f8'),
    ('usum', 'f8'),
    ('u2sum', 'f8'),
    ('uvsum', 'f8'),
    ('v2sum', 'f8'),
]
_sums_dtype = np.dtype(_sums_dtype, align=True)


_sums_dtype_fixcen = [
    ('gi', 'f8'),

    # used for convergence tests only
    ('logtau', 'f8'),
    ('logdet', 'f8'),

    # scratch on a given pixel
    ('tu2sum', 'f8'),
    ('tuvsum', 'f8'),
    ('tv2sum', 'f8'),

    # sums over all pixels
    ('pnew', 'f8'),
    ('u2sum', 'f8'),
    ('uvsum', 'f8'),
    ('v2sum', 'f8'),
]
_sums_dtype_fixcen = np.dtype(_sums_dtype_fixcen, align=True)

_tau_dtype = [
    ('logtau', 'f8'),
    ('logdet', 'f8'),
]
_tau_dtype = np.dtype(_tau_dtype)

_sums_dtype_ponly = [
    ('gi', 'f8'),

    # sums over all pixels
    ('pnew', 'f8'),
]
_sums_dtype_ponly = np.dtype(_sums_dtype_ponly, align=True)


def test_1gauss(
    counts=100.0,
    noise=0.0,
    T=4.0,
    maxiter=4000,
    g1=0.0,
    g2=0.0,
    show=False,
    pad=False,
    verbose=True,
    seed=31415,
):
    import time

    rng = np.random.RandomState(seed)

    sigma = np.sqrt(T / 2)
    dim = int(2 * 5 * sigma)
    dims = [dim] * 2
    cen = [dims[0] / 2.0, dims[1] / 2.0]

    jacob = UnitJacobian(row=cen[0], col=cen[1])

    pars = [0.0, 0.0, g1, g2, T, counts]
    gm = GMixModel(pars, "gauss")

    im0 = gm.make_image(dims, jacobian=jacob)

    im = im0 + rng.normal(size=im0.shape, scale=noise)

    imsky, sky = prep_image(im)

    obs = Observation(imsky, jacobian=jacob)

    guess_pars = [
        srandu(rng=rng),
        srandu(rng=rng),
        0.05 * srandu(rng=rng),
        0.05 * srandu(rng=rng),
        T * (1.0 + 0.1 * srandu(rng=rng)),
        counts * (1.0 + 0.1 * srandu(rng=rng)),
    ]
    gm_guess = GMixModel(guess_pars, "gauss")

    print("gm:", gm)
    print("gm_guess:", gm_guess)

    # twice, first time numba compiles the code
    for i in range(2):
        tm0 = time.time()
        em = GMixEM(obs, maxiter=maxiter)
        em.go(gm_guess, sky)
        tm = time.time() - tm0

    gmfit = em.get_gmix()
    res = em.get_result()

    if verbose:
        print("dims:", dims)
        print("cen:", cen)
        print("guess:")
        print(gm_guess)

        print("time:", tm, "seconds")
        print()

        print()
        print("results")
        print(res)

        print()
        print("gmix true:")
        print(gm)
        print("best fit:")
        print(gmfit)

    if show:
        try:
            import images
        except ImportError:
            from espy import images

        imfit = em.make_image()

        images.compare_images(im, imfit)

    return gmfit


def test_1gauss_T_recovery(
    noise, T=8.0, counts=1.0, ntrial=100, show=True, png=None
):
    import biggles

    T_true = T

    T_meas = np.zeros(ntrial)
    for i in range(ntrial):
        while True:
            try:
                gm = test_1gauss(
                    noise=noise, T=T_true, counts=counts, verbose=False,
                )
                T = gm.get_T()
                T_meas[i] = T
                break
            except GMixRangeError:
                pass
            except GMixMaxIterEM:
                pass

    mean = T_meas.mean()
    std = T_meas.std()
    print("<T>:", mean, "sigma(T):", std)
    binsize = 0.2 * std
    plt = biggles.plot_hist(T_meas, binsize=binsize, visible=False)
    p = biggles.Point(T_true, 0.0, type="filled circle", size=2, color="red")
    plt.add(p)
    plt.title = "Flux: %g T: %g noise: %g" % (counts, T_true, noise)

    xmin = mean - 4.0 * std
    xmax = mean + 4.0 * std

    plt.xrange = [xmin, xmax]

    if show:
        plt.show()

    if png is not None:
        print(png)
        plt.write_img(800, 800, png)


def test_1gauss_jacob(
    counts_sky=100.0, noise_sky=0.0, maxiter=100, show=False
):
    import time

    rng = np.random.RandomState(42587)

    # import images
    dims = [25, 25]
    cen = [dims[0] / 2.0, dims[1] / 2.0]

    # j1,j2,j3,j4=0.26,0.02,-0.03,0.23
    dvdrow, dvdcol, dudrow, dudcol = -0.04, -0.915, 1.10, 0.12
    j = Jacobian(
        row=cen[0],
        col=cen[1],
        dvdrow=dvdrow,
        dvdcol=dvdcol,
        dudrow=dudrow,
        dudcol=dudcol,
    )

    jfac = j.get_scale()

    g1 = 0.1
    g2 = 0.05

    Tsky = 8.0 * jfac ** 2
    noise_pix = noise_sky / jfac ** 2

    pars = [0.0, 0.0, g1, g2, Tsky, counts_sky]
    gm = GMixModel(pars, "gauss")
    print("gmix true:")
    print(gm)

    im0 = gm.make_image(dims, jacobian=j)

    im = im0 + noise_pix * np.random.randn(im0.size).reshape(dims)

    imsky, sky = prep_image(im)

    obs = Observation(imsky, jacobian=j)

    gm_guess = gm.copy()
    gm_guess._data["p"] = 1.0
    gm_guess._data["row"] += srandu(rng=rng)
    gm_guess._data["col"] += srandu(rng=rng)
    gm_guess._data["irr"] += srandu(rng=rng)
    gm_guess._data["irc"] += srandu(rng=rng)
    gm_guess._data["icc"] += srandu(rng=rng)

    print("guess:")
    print(gm_guess)

    tm0 = time.time()
    em = GMixEM(obs)
    em.go(gm_guess, sky, maxiter=maxiter)
    tm = time.time() - tm0
    print("time:", tm, "seconds")

    gmfit = em.get_gmix()
    res = em.get_result()
    print("best fit:")
    print(gmfit)
    print("results")
    print(res)

    if show:
        import images

        imfit = gmfit.make_image(im.shape, jacobian=j)
        imfit *= im0.sum() / imfit.sum()

        images.compare_images(im, imfit)

    return gmfit


def test_2gauss(counts=100.0, noise=0.0, show=False, scale=1.0):
    import time

    rng = np.random.RandomState(42587)

    dims = [25, 25]

    cen = (np.array(dims) - 1.0) / 2.0
    jacob = UnitJacobian(row=cen[0], col=cen[1])

    cen1 = [-3.25, -3.25]
    cen2 = [3.0, 0.5]

    e1_1 = 0.1
    e2_1 = 0.05
    T_1 = 8.0
    counts_1 = 0.4 * counts
    irr_1 = T_1 / 2.0 * (1 - e1_1)
    irc_1 = T_1 / 2.0 * e2_1
    icc_1 = T_1 / 2.0 * (1 + e1_1)

    e1_2 = -0.2
    e2_2 = -0.1
    T_2 = 4.0
    counts_2 = 0.6 * counts
    irr_2 = T_2 / 2.0 * (1 - e1_2)
    irc_2 = T_2 / 2.0 * e2_2
    icc_2 = T_2 / 2.0 * (1 + e1_2)

    pars = [
        counts_1,
        cen1[0],
        cen1[1],
        irr_1,
        irc_1,
        icc_1,
        counts_2,
        cen2[0],
        cen2[1],
        irr_2,
        irc_2,
        icc_2,
    ]

    gm = GMix(pars=pars)
    print("gmix true:")
    print(gm)

    im0 = gm.make_image(dims, jacobian=jacob)
    im = im0 + noise * np.random.randn(im0.size).reshape(dims)

    imsky, sky = prep_image(im)
    print('sky:', sky)

    obs = Observation(imsky, jacobian=jacob)

    gm_guess = gm.copy()
    gm_guess._data["p"] += counts_1/10 * srandu(2, rng=rng)
    gm_guess._data["row"] += 4 * srandu(2, rng=rng)
    gm_guess._data["col"] += 4 * srandu(2, rng=rng)
    gm_guess._data["irr"] += 0.5 * srandu(2, rng=rng)
    gm_guess._data["irc"] += 0.5 * srandu(2, rng=rng)
    gm_guess._data["icc"] += 0.5 * srandu(2, rng=rng)

    print("guess:")
    print(gm_guess)

    for i in range(2):
        tm0 = time.time()
        em = GMixEM(obs)
        em.go(gm_guess, sky)
        tm = time.time() - tm0
    print("time:", tm, "seconds")

    gmfit = em.get_gmix()
    res = em.get_result()
    print("best fit:")
    print(gmfit)
    print("results")
    print(res)

    print('im sum:', im.sum())
    if show:
        try:
            import images
        except ImportError:
            from espy import images

        imfit = em.make_image()
        print('imfit sum:', imfit.sum())
        images.compare_images(im, imfit, colorbar=True)

    return tm


def test_2gauss_jacob(counts=100.0, noise=0.0, show=False, scale=0.263):
    import time

    rng = np.random.RandomState(3812)

    dims = [25, 25]

    cen = (np.array(dims) - 1.0) / 2.0
    jacob = DiagonalJacobian(scale=scale, row=cen[0], col=cen[1])

    cen1 = [-3.25*scale, -3.25*scale]
    cen2 = [3.0*scale, 0.5*scale]

    e1_1 = 0.1
    e2_1 = 0.05
    T_1 = 8.0 * scale**2
    counts_1 = 0.4 * counts
    irr_1 = T_1 / 2.0 * (1 - e1_1)
    irc_1 = T_1 / 2.0 * e2_1
    icc_1 = T_1 / 2.0 * (1 + e1_1)

    e1_2 = -0.2
    e2_2 = -0.1
    T_2 = 4.0 * scale**2
    counts_2 = 0.6 * counts
    irr_2 = T_2 / 2.0 * (1 - e1_2)
    irc_2 = T_2 / 2.0 * e2_2
    icc_2 = T_2 / 2.0 * (1 + e1_2)

    pars = [
        counts_1,
        cen1[0],
        cen1[1],
        irr_1,
        irc_1,
        icc_1,
        counts_2,
        cen2[0],
        cen2[1],
        irr_2,
        irc_2,
        icc_2,
    ]

    gm = GMix(pars=pars)
    print("gmix true:")
    print(gm)

    im0 = gm.make_image(dims, jacobian=jacob)
    im = im0 + noise * np.random.randn(im0.size).reshape(dims)

    imsky, sky = prep_image(im)
    print('sky:', sky)

    obs = Observation(imsky, jacobian=jacob)

    gm_guess = gm.copy()
    gm_guess._data["p"] += counts_1/10 * srandu(2, rng=rng)
    gm_guess._data["row"] += 4 * srandu(2, rng=rng)
    gm_guess._data["col"] += 4 * srandu(2, rng=rng)
    gm_guess._data["irr"] += 0.5 * srandu(2, rng=rng)
    gm_guess._data["irc"] += 0.5 * srandu(2, rng=rng)
    gm_guess._data["icc"] += 0.5 * srandu(2, rng=rng)

    print("guess:")
    print(gm_guess)

    for i in range(2):
        tm0 = time.time()
        em = GMixEM(obs)
        em.go(gm_guess, sky)
        tm = time.time() - tm0
    print("time:", tm, "seconds")

    gmfit = em.get_gmix()
    res = em.get_result()
    print("best fit:")
    print(gmfit)
    print("results")
    print(res)

    print('im sum:', im.sum())
    if show:
        try:
            import images
        except ImportError:
            from espy import images

        imfit = em.make_image()
        print('imfit sum:', imfit.sum())
        images.compare_images(im, imfit, colorbar=True)

    return tm
