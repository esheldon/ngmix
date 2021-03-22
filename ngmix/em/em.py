"""
Fit an image with a gaussian mixture using the EM algorithm
"""
__all__ = [
    'run_em', 'prep_image', 'prep_obs', 'EMResult', 'EMFitter',
    'EMFitterFixCen', 'EMFitterFluxOnly',
]
import logging
import numpy as np

from ..gexceptions import GMixRangeError
from ..observation import Observation
from ..gmix import GMix, GMixModel
from ..flags import EM_RANGE_ERROR, EM_MAXITER
from .em_nb import em_run, em_run_fixcen, em_run_fluxonly

logger = logging.getLogger(__name__)


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

_sums_dtype_fluxonly = [
    ('gi', 'f8'),

    # sums over all pixels
    ('pnew', 'f8'),
]
_sums_dtype_fluxonly = np.dtype(_sums_dtype_fluxonly, align=True)
