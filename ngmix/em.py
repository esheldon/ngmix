"""
Fit an image with a gaussian mixture using the EM algorithm
"""
__all__ = [
    'run_em', 'prep_image', 'prep_obs', 'EMResult', 'EMFitter',
    'EMFitterFixCen', 'EMFitterFluxOnly',
]
import logging
import numpy as np
from numba import njit

from .gexceptions import GMixRangeError
from .observation import Observation

from .gmix import GMix, GMixModel
from .gmix.gmix_nb import (
    gauss2d_set,
    gmix_set_norms,
    gmix_get_cen,
    gmix_convolve_fill,
    gmix_eval_pixel_fast,
    GMIX_LOW_DETVAL,
)

from .fastexp_nb import fexp

logger = logging.getLogger(__name__)


EM_RANGE_ERROR = 2 ** 0
EM_MAXITER = 2 ** 1

DEFAULT_TOL = 1.0e-5


def run_em(obs, guess, sky=None, fixcen=False, fluxonly=False, **kws):
    """
    fit the observation with EM

    Parameters
    ----------
    obs: ngmix.Observation
        The observation to fit
    guess: ngmix.GMix
        The initial guess as a gaussian mixture
    sky: number, optional
        The sky value for the image.  If you don't send this, it is assumed the
        true sky in the image is zero, and the prep_obs code is used to set a
        sky such that there are no negative pixels.
    fixcen: bool, optional
        if True, use the fixed center fitter
    fluxonly: bool, optional
        if True, use the flxu only fitter
    minimum: number, optional
        The minimum number of iterations, default depends on the fitter
    maxiter: number, optional
        The maximum number of iterations, default depends on the fitter
    tol: number, optional
        The fractional change in the log likelihood that implies convergence,
        default 0.001
    vary_sky: bool
        If True, fit for the sky level

    Returns
    -------
    The EM fitter
    """

    if fixcen:
        fitter = EMFitterFixCen(**kws)
    elif fluxonly:
        fitter = EMFitterFluxOnly(**kws)
    else:
        fitter = EMFitter(**kws)

    return fitter.go(obs=obs, guess=guess, sky=sky)


# for backwards compatibility
fit_em = run_em


def prep_obs(obs):
    """
    Prep the image to fit with EM and produce a new observation.

    Parameters
    ----------
    obs: ngmix.Observation
        The observation to fit

    Returns
    ------
    new_obs, sky: ngmix.Observation and sky
        The obs with new background level and the applied background level
    """

    imsky, sky = prep_image(obs.image)
    if obs.has_psf():
        newobs = Observation(
            imsky,
            jacobian=obs.jacobian,
            psf=obs.psf,
        )
    else:
        newobs = Observation(
            imsky,
            jacobian=obs.jacobian,
        )

    return newobs, sky


def prep_image(im0):
    """
    Prep an image to fit with EM.  Make sure there are no pixels < 0

    Parameters
    ----------
    image: ndarray
        2d image

    Returns
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


class EMResult(dict):
    """
    Class to represent an EM model fit and generate images and gaussian
    mixtures for the best fit

    parameters
    ----------
    obs: Observation
        An ngmix.Observation object.  This must already have been run through
        prep_obs to make sure there are no pixels less than zero
    guess: GMix
        A gaussian mixture (GMix or child class) representing a starting
        guess for the algorithm.  This should be *before* psf convolution.
    """

    def __init__(self, obs, result, gm=None, gm_conv=None):
        self._obs = obs
        self.update(result)
        if gm is not None and gm_conv is not None:
            self._gm = gm
            self._gm_conv = gm_conv

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

        Returns
        ----------
        gm: ngmix.GMix
            The best fit mixture.  It will not be convolved by the PSF if a psf
            was present
        """
        if not self.has_gmix():
            raise RuntimeError('no gmix set')

        return self._gm.copy()

    def get_convolved_gmix(self):
        """
        Get the gaussian mixture from the final iteration

        Returns
        ----------
        gm: ngmix.GMix
            The best fit mixture.  It will be convolved by the
            PSF if a psf was present
        """
        if not self.has_gmix():
            raise RuntimeError('no gmix set')

        return self._gm_conv.copy()

    def make_image(self):
        """
        Get an image of the best fit mixture

        Returns
        -------
        image: array
            Image of the model, including the PSF if a psf was sent
        """
        gm = self.get_convolved_gmix()
        return gm.make_image(
            self._obs.image.shape,
            jacobian=self._obs.jacobian,
        )


class EMFitter(object):
    """
    Fit an image with a gaussian mixture using the EM algorithm

    Parameters
    ----------
    miniter: number, optional
        The minimum number of iterations, default 40
    maxiter: number, optional
        The maximum number of iterations, default 500
    tol: number, optional
        The fractional change in the log likelihood that implies convergence,
        default 0.001
    vary_sky: bool
        If True, fit for the sky level
    """
    def __init__(self,
                 miniter=40,
                 maxiter=500,
                 tol=DEFAULT_TOL,
                 vary_sky=False):

        self.miniter = miniter
        self.maxiter = maxiter
        self.tol = tol
        self.vary_sky = vary_sky
        self._set_runner()

    def go(self, obs, guess, sky=None):
        """
        Run the em algorithm from the input starting guesses

        parameters
        ----------
        obs: Observation
            An ngmix.Observation object
        guess: GMix
            A gaussian mixture (GMix or child class) representing a starting
            guess for the algorithm.  This should be *before* psf convolution.
        sky: number, optional
            The sky value for the image.  If you don't send this, it is assumed the
            true sky in the image is zero, and the prep_obs code is used to set a
            sky such that there are no negative pixels.
        """

        if not isinstance(obs, Observation):
            raise ValueError('input obs must be an instance of Observation')

        if sky is None:
            obs_sky, sky = prep_obs(obs)
        else:
            obs_sky = obs

        # makes a copy
        if not obs_sky.has_psf() or not obs_sky.psf.has_gmix():
            logger.debug('NO PSF SET')
            gmix_psf = GMixModel([0., 0., 0., 0., 0., 1.0], 'gauss')
        else:
            gmix_psf = obs_sky.psf.gmix
            gmix_psf.set_flux(1.0)

        conf = self._make_conf(obs_sky)
        conf['sky'] = sky

        gm_to_fit = guess.copy()
        gm_conv_to_fit = gm_to_fit.convolve(gmix_psf)

        sums = self._make_sums(len(gm_to_fit))

        pixels = obs_sky.pixels.copy()

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
                gm_to_fit.get_data(),
                gmix_psf.get_data(),
                gm_conv_to_fit.get_data(),
                fill_zero_weight=fill_zero_weight,
            )

            pars = gm_to_fit.get_full_pars()
            pars_conv = gm_conv_to_fit.get_full_pars()
            gm = GMix(pars=pars)
            gm_conv = GMix(pars=pars_conv)

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
            gm = None
            gm_conv = None
            message = str(err)
            logger.info(message)
            result = {
                'flags': EM_RANGE_ERROR,
                'message': message,
            }

        return EMResult(obs=obs, result=result, gm=gm, gm_conv=gm_conv)

    def _make_sums(self, ngauss):
        """
        make the sum structure
        """
        return np.zeros(ngauss, dtype=_sums_dtype)

    def _make_conf(self, obs):
        """
        make the sum structure
        """
        conf = np.zeros(1, dtype=_em_conf_dtype)
        conf = conf[0]

        conf['tol'] = self.tol
        conf['miniter'] = self.miniter
        conf['maxiter'] = self.maxiter
        conf['vary_sky'] = self.vary_sky

        return conf

    def _set_runner(self):
        self._runner = em_run


class EMFitterFixCen(EMFitter):
    """
    Fit an image with a gaussian mixture using the EM algorithm

    Parameters
    ----------
    miniter: number, optional
        The minimum number of iterations, default 40
    maxiter: number, optional
        The maximum number of iterations, default 500
    tol: number, optional
        The fractional change in the log likelihood that implies convergence,
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


class EMFitterFluxOnly(EMFitterFixCen):
    """
    Fit an image with a gaussian mixture using the EM algorithm,
    allowing only the fluxes to vary

    Parameters
    ----------
    miniter: number, optional
        The minimum number of iterations, default 20
    maxiter: number, optional
        The maximum number of iterations, default 500
    tol: number, optional
        The fractional change in the log likelihood that implies convergence,
        default 0.001
    vary_sky: bool
        If True, fit for the sky level
    """

    def __init__(self,
                 miniter=20,
                 maxiter=500,
                 tol=DEFAULT_TOL,
                 vary_sky=False):
        """
        over-riding because we want a different default miniter
        """
        super().__init__(
            miniter=miniter,
            maxiter=maxiter,
            tol=tol,
            vary_sky=vary_sky,
        )

    def _set_runner(self):
        self._runner = em_run_fluxonly

    def _make_sums(self, ngauss):
        """
        make the sum structure
        """
        return np.zeros(ngauss, dtype=_sums_dtype_fluxonly)


@njit
def em_run(conf,
           pixels,
           sums,
           gmix,
           gmix_psf,
           gmix_conv,
           fill_zero_weight=False):
    """
    run the EM algorithm

    Parameters
    ----------
    conf: array
        Should have fields

            tol: The fractional change in the log likelihood that implies
                convergence
            miniter: minimum number of iterations
            maxiter: maximum number of iterations
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

    Returns
    -------
    numiter, frac_diff, sky
        number of iterations, fractional difference in log likelihood between
        last two steps, and sky.  The sky may have been fit for.
    """

    gmix_set_norms(gmix_conv)
    ngauss_psf = gmix_psf.size

    taudata = np.zeros(gmix_conv.size, dtype=_tau_dtype)

    tol = conf['tol']

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
        )

        if conf['vary_sky']:
            sky = skysum/npix

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

    Parameters
    ----------
    sums: array
        Array with dtype _sums_dtype
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

    Parameters
    ----------
    pixel: pixel structure
        Dtype should be ngmix.pixels._pixels_dtype
    gmix_conv: ngmix.GMix
        The current mixture, convolved by the PSF
    sums: sums structure
        With dtype _sums_dtype
    ngauss_psf: int
        Number of gaussians in psf
    taudata: tau data struct
        With dtype _tau_dtype

    Returns
    -------
    gsum, logL:
        The total of the gaussians evaluated in the pixel and the
        log likelihood
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
                val = gauss['pnorm']*fexp(-0.5*chi2) * pixel['area']
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

    Parameters
    ----------
    sums: sums structure
        With dtype _sums_dtype_fixcen
    pixel: pixel structure
        Dtype should be ngmix.pixels._pixels_dtype
    gtot: float
        The sum over gaussian values
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
                       sums):
    """
    fill the gaussian mixture from the em sums

    Parameters
    ----------
    gmix: ngmix.GMix
        The gaussian mixture before psf convolution
    gmix_psf: ngmix.GMix
        The gaussian mixture of the PSF (can be zero size for no psf)
    gmix_conv: ngmix.GMix
        The final convolved mixture
    sums: sums structure
        With dtype _sums_dtype
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

        gauss2d_set(
            gauss,
            p,
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
    run the EM algorithm with fixed positions

    Parameters
    ----------
    conf: array
        Should have fields

            tol: The fractional change in the log likelihood that implies
                convergence
            miniter: minimum number of iterations
            maxiter: maximum number of iterations
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

    Returns
    -------
    numiter, frac_diff, sky
        number of iterations, fractional difference in log likelihood between
        last two steps, and sky.  The sky may have been fit for.
    """

    gmix_set_norms(gmix_conv)
    ngauss_psf = gmix_psf.size

    taudata = np.zeros(gmix_conv.size, dtype=_tau_dtype)

    tol = conf['tol']

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
        )

        if conf['vary_sky']:
            sky = skysum/npix

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
def do_scratch_sums_fixcen(pixel, gmix_conv, sums, ngauss_psf, taudata):
    """
    do the basic sums for this pixel, using scratch space in the sums struct

    we may have multiple components per "object" so we update the
    sums accordingly

    Parameters
    ----------
    pixel: pixel structure
        Dtype should be ngmix.pixels._pixels_dtype
    gmix_conv: ngmix.GMix
        The current mixture, convolved by the PSF
    sums: sums structure
        With dtype _sums_dtype_fixcen
    ngauss_psf: int
        Number of gaussians in psf
    taudata: tau data struct
        With dtype _tau_dtype

    Returns
    -------
    gsum, logL:
        The total of the gaussians evaluated in the pixel and the
        log likelihood
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
                val = gauss['pnorm']*fexp(-0.5*chi2) * pixel['area']
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

    Parameters
    ----------
    sums: sums structure
        With dtype _sums_dtype_fixcen
    pixel: pixel structure
        Dtype should be ngmix.pixels._pixels_dtype
    gtot: float
        The sum over gaussian values
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
                              sums):
    """
    fill the gaussian mixture from the em sums

    Parameters
    ----------
    gmix: ngmix.GMix
        The gaussian mixture before psf convolution
    gmix_psf: ngmix.GMix
        The gaussian mixture of the PSF (can be zero size for no psf)
    gmix_conv: ngmix.GMix
        The final convolved mixture
    sums: sums structure
        With dtype _sums_dtype_fixcen
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

        gauss2d_set(
            gauss,
            p,
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

    Parameters
    -----------
    gmix: ngmix.Gmix
        gaussian Mixture
    sums: array
        Array with sums
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

    Parameters
    ----------
    sums: array
        Array with dtype _sums_dtype_fixcen
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
def em_run_fluxonly(
    conf,
    pixels,
    sums,
    gmix,
    gmix_psf,
    gmix_conv,
    fill_zero_weight=False,
):
    """
    run the EM algorithm, allowing only fluxes to vary

    Parameters
    ----------
    conf: array
        Should have fields
            tol: The fractional change in the log likelihood that implies
                convergence
            miniter: minimum number of iterations
            maxiter: maximum number of iterations
            sky: the sky, or guess for sky if fitting for it
            vary_sky: True if fitting for the sky

    pixels: pixel array
        for the image/jacobian
    sums: array with fields
        The sums array, a type _sums_dtype_fluxonly
    gmix: gauss2d array
        The initial mixture.  The final result is also stored in this array.
    gmix_psf: gauss2d array
        Single gaussian psf
    gmix_conv: gauss2d array
        Convolved gmix
    fill_zero_weight: bool
        If True, fill the zero weight pixels with the model on
        each iteration

    Returns
    -------
    numiter, frac_diff, sky
        number of iterations, fractional difference in log likelihood between
        last two steps, and sky.  The sky may have been fit for.
    """

    gmix_set_norms(gmix_conv)
    ngauss_psf = gmix_psf.size

    tol = conf['tol']

    npix = pixels.size

    sky = conf['sky']

    p_last = gmix['p'].sum()

    for i in range(conf['maxiter']):
        skysum = 0.0
        clear_sums_fluxonly(sums)

        if fill_zero_weight:
            fill_zero_weight_pixels(gmix_conv, pixels, sky)

        for pixel in pixels:

            # this fills some fields of sums, as well as return
            gsum = do_scratch_sums_fluxonly(pixel, gmix_conv, sums, ngauss_psf)

            gtot = gsum + sky
            if gtot == 0.0:
                raise GMixRangeError("gtot == 0")

            skysum += sky*pixel['val']/gtot

            do_sums_fluxonly(sums, pixel, gtot)

        gmix_set_from_sums_fluxonly(
            gmix,
            gmix_psf,
            gmix_conv,
            sums,
        )

        if conf['vary_sky']:
            sky = skysum/npix

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
def do_scratch_sums_fluxonly(pixel, gmix_conv, sums, ngauss_psf):
    """
    do the basic sums for this pixel, using
    scratch space in the sums struct

    Parameters
    ----------
    pixel: pixel structure
        Dtype should be ngmix.pixels._pixels_dtype
    gmix_conv: ngmix.GMix
        The current mixture, convolved by the PSF
    sums: sums structure
        With dtype _sums_dtype_fluxonly
    ngauss_psf: int
        Number of gaussians in psf

    Returns
    -------
    gsum, logL:
        The total of the gaussians evaluated in the pixel and the
        log likelihood
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
                val = gauss['pnorm']*fexp(-0.5*chi2) * pixel['area']
            else:
                val = 0.0

            tsums['gi'] += val
            gsum += val

    return gsum


@njit
def do_sums_fluxonly(sums, pixel, gtot):
    """
    do the sums based on the scratch values

    Parameters
    ----------
    sums: sums structure
        With dtype _sums_dtype_fluxonly
    pixel: pixel structure
        Dtype should be ngmix.pixels._pixels_dtype
    gtot: float
        The sum over gaussian values
    """

    factor = pixel['val']/gtot

    n_gauss = sums.size
    for i in range(n_gauss):
        tsums = sums[i]

        # gi*image_val/sum(gi)
        wtau = tsums['gi']*factor

        tsums['pnew'] += wtau


@njit
def gmix_set_from_sums_fluxonly(
    gmix,
    gmix_psf,
    gmix_conv,
    sums,
):
    """
    fill the gaussian mixture from the em sums

    Parameters
    ----------
    gmix: ngmix.GMix
        The gaussian mixture before psf convolution
    gmix_psf: ngmix.GMix
        The gaussian mixture of the PSF (can be zero size for no psf)
    gmix_conv: ngmix.GMix
        The final convolved mixture
    sums: sums structure
        With dtype _sums_dtype_fluxonly
    """

    n_gauss = gmix.size
    for i in range(n_gauss):

        tsums = sums[i]
        gauss = gmix[i]

        p = tsums['pnew']

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
def clear_sums_fluxonly(sums):
    """
    set all sums to zero

    Parameters
    ----------
    sums: array
        Array with dtype _sums_dtype_fluxonly
    """
    sums['gi'][:] = 0.0

    # sums over all pixels
    sums['pnew'][:] = 0.0


@njit
def gmix_get_moms(gmix):
    """
    get row, col, irr, irc, icc, psum

    Parameters
    ----------
    gmix: ngmix.Gmix
        Gaussian mixture

    Returns
    -------
    return row, col, irr, irc, icc, psum
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


@njit
def fill_zero_weight_pixels(gmix, pixels, sky):
    """
    fill zero weight pixels with the model

    Paramters
    ---------
    gmix: ngmix.Gmix
        The mixture with which to evaluate in the zero weight pixels
    pixels: array of pixels
        The pixels to be modified
    sky: float
        Value of the sky in the pixels
    """

    for pixel in pixels:
        if pixel['ierr'] <= 0.0:
            val = gmix_eval_pixel_fast(gmix, pixel)
            pixel['val'] = sky + val


_em_conf_dtype = [
    ('tol', 'f8'),
    ('maxiter', 'i4'),
    ('miniter', 'i4'),
    ('sky', 'f8'),
    ('vary_sky', 'bool'),
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

_sums_dtype_fluxonly = [
    ('gi', 'f8'),

    # sums over all pixels
    ('pnew', 'f8'),
]
_sums_dtype_fluxonly = np.dtype(_sums_dtype_fluxonly, align=True)
