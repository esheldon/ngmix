import numpy as np
from .gmix import GMix, GMixModel, get_coellip_npars
from .util import print_pars
from .gexceptions import GMixRangeError, PSFFluxFailure
from .priors import srandu
from .defaults import LOWVAL
from .shape import Shape
from . import moments
import logging

LOGGER = logging.getLogger(__name__)


class TFluxGuesser(object):
    """
    get full guesses from just T, fluxes

    parameters
    ----------
    rng: numpy.random.RandomState
        Random state for generating guesses
    T: float
        Central value for T guesses
    flux: float or sequence
        Central value for flux guesses.  Can be a single float or a sequence/array
        for multiple bands
    prior: optional
        If sent, "fix-up" guesses if they are not allowed by the prior
    """

    def __init__(self, rng, T, flux, prior=None):
        self.rng = rng
        self.T = T

        self.fluxes = np.array(flux, dtype="f8", ndmin=1)
        self.prior = prior

    def __call__(self, nrand=1, obs=None):
        """
        Generate a guess.  The center, shape are distributed tightly around
        zero

        Parameters
        ----------
        nrand: int, optional
            Number of samples to draw.  Default 1
        obs: ignored
            This keyword is here to conform to the interface
        """

        rng = self.rng

        fluxes = self.fluxes
        nband = fluxes.size
        npars = 5 + nband

        guess = np.zeros((nrand, npars))
        guess[:, 0] = rng.uniform(low=-0.01, high=0.01, size=nrand)
        guess[:, 1] = rng.uniform(low=-0.01, high=0.01, size=nrand)
        guess[:, 2] = rng.uniform(low=-0.02, high=0.02, size=nrand)
        guess[:, 3] = rng.uniform(low=-0.02, high=0.02, size=nrand)
        guess[:, 4] = self.T * rng.uniform(low=0.9, high=1.1, size=nrand)

        for band in range(nband):
            guess[:, 5 + band] = (
                fluxes[band] * rng.uniform(low=0.9, high=1.1, size=nrand)
            )

        if self.prior is not None:
            _fix_guess(guess, self.prior)

        if nrand == 1:
            guess = guess[0, :]

        return guess


class TPSFFluxGuesser(object):
    """
    get full guesses from just the input T and fluxes based on psf fluxes

    parameters
    ----------
    rng: numpy.random.RandomState
        Random state for generating guesses
    T: float
        Central value for T guesses
    prior: optional
        If sent, "fix-up" guesses if they are not allowed by the prior
    """

    def __init__(self, rng, T, prior=None):
        self.rng = rng
        self.T = T
        self.prior = prior
        self._id_last = None
        self._psf_fluxes = None

    def _get_psf_fluxes(self, obs):
        oid = id(obs)
        if oid != self._id_last:
            self._id_last = oid
            fdict = _get_psf_fluxes(rng=self.rng, obs=obs)
            self._psf_fluxes = fdict['flux']
        return self._psf_fluxes

    def __call__(self, obs, nrand=1):
        """
        Generate a guess.

        Parameters
        ----------
        obs: Observation
            The observation(s) used for psf fluxes
        nrand: int, optional
            Number of samples to draw.  Default 1
        """

        rng = self.rng

        fluxes = self._get_psf_fluxes(obs=obs)

        nband = fluxes.size
        npars = 5 + nband

        guess = np.zeros((nrand, npars))
        guess[:, 0] = rng.uniform(low=-0.01, high=0.01, size=nrand)
        guess[:, 1] = rng.uniform(low=-0.01, high=0.01, size=nrand)
        guess[:, 2] = rng.uniform(low=-0.02, high=0.02, size=nrand)
        guess[:, 3] = rng.uniform(low=-0.02, high=0.02, size=nrand)
        guess[:, 4] = self.T * rng.uniform(low=0.9, high=1.1, size=nrand)

        for band in range(nband):
            guess[:, 5 + band] = (
                fluxes[band] * rng.uniform(low=0.9, high=1.1, size=nrand)
            )

        if self.prior is not None:
            _fix_guess(guess, self.prior)

        if nrand == 1:
            guess = guess[0, :]

        return guess


class TPSFFluxAndPriorGuesser(TPSFFluxGuesser):
    """
    get full guesses from just the T guess, psf fluxes and the prior

    parameters
    ----------
    rng: numpy.random.RandomState
        Random state for generating guesses
    T: float
        Central value for T guesses
    prior: joint prior
        cen, g drawn from this prior
    """

    def __init__(self, rng, T, prior):
        self.rng = rng
        self.T = T
        self.prior = prior
        self._id_last = None
        self._psf_fluxes = None

    def __call__(self, obs, nrand=1):
        """
        Generate a guess.

        Parameters
        ----------
        obs: Observation
            The observation(s) used for psf fluxes
        nrand: int, optional
            Number of samples to draw.  Default 1
        """

        rng = self.rng

        fluxes = self._get_psf_fluxes(obs=obs)

        nband = fluxes.size

        guess = self.prior.sample(nrand)

        r = rng.uniform(low=-0.1, high=0.1, size=nrand)
        guess[:, 4] = self.T * (1.0 + r)

        for band in range(nband):
            guess[:, 5 + band] = (
                fluxes[band] * rng.uniform(low=0.9, high=1.1, size=nrand)
            )

        if self.prior is not None:
            _fix_guess_TFlux(guess, self.prior)

        if nrand == 1:
            guess = guess[0, :]

        return guess


def _get_psf_fluxes(rng, obs):
    """
    Get psf fluxes for the input observations

    The result is cached with cache size 64

    Parameters
    ----------
    rng: numpy.random.RandomState
        Random state for generating guesses
    obs: Observation, Obslist, MultiBandObsList
        The observations

    Returns
    -------
    a dict with 'flags' 'flux' 'flux_err' for each band
    """
    from .fitting import PSFFluxFitter
    from .observation import get_mb_obs

    mbobs = get_mb_obs(obs)

    nband = len(mbobs)
    flux = np.zeros(nband)
    flux_err = np.zeros(nband)
    flags = np.zeros(nband, dtype='i4')

    fitter = PSFFluxFitter()

    for iband, obslist in enumerate(mbobs):
        res = fitter.go(obs=obslist)

        # flags are set for all zero weight pixels, so there isn't anything we
        # can do, we'll have to fix it up
        flags[iband] = res['flags']
        flux[iband] = res['flux']
        flux_err[iband] = res['flux_err']

    logic = (flags == 0) & np.isfinite(flux)
    wgood, = np.where(logic)
    if wgood.size != nband:
        if wgood.size == 0:
            # no good fluxes due to flags. This means we have no information
            # to use for a fit or measurement, so there is no point in
            # generating a guess
            raise PSFFluxFailure("no good psf fluxes")
        else:
            # make a guess based on the good ones
            wbad, = np.where(~logic)
            fac = 1.0 + rng.uniform(low=-0.1, high=0.1, size=wbad.size)
            flux[wbad] = flux[wgood].mean()*fac

    return {
        'flags': flags,
        'flux': flux,
        'flux_err': flux_err,
    }


class TFluxAndPriorGuesser(object):
    """
    Make guesses from the input T, flux and prior

    parameters
    ----------
    T: float
        Center for T guesses
    flux: float or sequences
        Center for flux guesses
    prior:
        cen, g drawn from this prior
    """

    def __init__(self, rng, T, flux, prior):

        fluxes = np.array(flux, dtype="f8", ndmin=1)

        self.T = T
        self.fluxes = fluxes
        self.prior = prior

        lfluxes = self.fluxes.copy()
        (w,) = np.where(self.fluxes < 0.0)
        if w.size > 0:
            lfluxes[w[:]] = 1.0e-10

    def __call__(self, nrand=1, obs=None):
        """
        Generate a guess, with center and shape drawn from the prior.

        Parameters
        ----------
        nrand: int, optional
            Number of samples to draw.  Default 1
        obs: ignored
            This keyword is here to conform to the interface
        """

        rng = self.prior.cen_prior.rng

        fluxes = self.fluxes

        nband = fluxes.size

        guess = self.prior.sample(nrand)

        r = rng.uniform(low=-0.1, high=0.1, size=nrand)
        guess[:, 4] = self.T * (1.0 + r)

        for band in range(nband):
            r = rng.uniform(low=-0.1, high=0.1, size=nrand)
            guess[:, 5 + band] = fluxes[band] * (1.0 + r)

        _fix_guess_TFlux(guess, self.prior)

        if nrand == 1:
            guess = guess[0, :]
        return guess


class BDFGuesser(object):
    """
    Make BDF guesses from the input T, flux and prior

    parameters
    ----------
    T: float
        Center for T guesses
    flux: float or sequences
        Center for flux guesses
    prior:
        cen, g drawn from this prior
    """

    def __init__(self, T, flux, prior):
        self.T = T
        self.fluxes = np.array(flux, ndmin=1)
        self.prior = prior

    def __call__(self, nrand=1, obs=None):
        """
        center, shape are just distributed around zero

        nrand: int, optional
            Number of samples to draw.  Default 1
        obs: ignored
            This keyword is here to conform to the interface
        """
        rng = self.prior.cen_prior.rng

        fluxes = self.fluxes

        guess = self.prior.sample(nrand)

        nband = fluxes.size

        r = rng.uniform(low=-0.1, high=0.1, size=nrand)
        guess[:, 4] = self.T * (1.0 + r)

        # fracdev prior
        guess[:, 5] = rng.uniform(low=0.4, high=0.6, size=nrand)

        for band in range(nband):
            r = rng.uniform(low=-0.1, high=0.1, size=nrand)
            guess[:, 6 + band] = fluxes[band] * (1.0 + r)

        if self.prior is not None:
            _fix_guess(guess, self.prior)

        if nrand == 1:
            guess = guess[0, :]
        return guess


class BDGuesser(object):
    """
    Make BD guesses from the input T, flux and prior

    parameters
    ----------
    T: float
        Center for T guesses
    flux: float or sequences
        Center for flux guesses
    prior:
        cen, g drawn from this prior
    """

    def __init__(self, T, flux, prior):
        self.T = T
        self.fluxes = np.array(flux, ndmin=1)
        self.prior = prior

    def __call__(self, nrand=1, obs=None):
        """
        Generate a guess from the T, flux and prior for the rest

        Parameters
        ----------
        nrand: int, optional
            Number of samples to draw.  Default 1
        obs: ignored
            This keyword is here to conform to the interface
        """

        rng = self.prior.cen_prior.rng

        fluxes = self.fluxes

        guess = self.prior.sample(nrand)

        nband = fluxes.size

        r = rng.uniform(low=-0.1, high=0.1, size=nrand)
        guess[:, 4] = self.T * (1.0 + r)

        # fracdev prior
        guess[:, 5] = rng.uniform(low=0.4, high=0.6, size=nrand)

        for band in range(nband):
            r = rng.uniform(low=-0.1, high=0.1, size=nrand)
            guess[:, 7 + band] = fluxes[band] * (1.0 + r)

        if self.prior is not None:
            _fix_guess(guess, self.prior)

        if nrand == 1:
            guess = guess[0, :]
        return guess


class ParsGuesser(object):
    """
    Make guess based on an input set of parameters

    parameters
    ----------
    rng: numpy.random.RandomState
        Random state for generating guesses
    pars: array/sequence
        Parameters upon which to base the guess
    widths: array/sequence
        Widths for random guesses, default is 0.1, absolute
        for cen, g but relative for T, flux
    prior: joint prior, optional
        If sent, "fix-up" guesses if they are not allowed by the prior
    """

    def __init__(self, rng, pars, prior=None, widths=None):
        self.rng = rng
        self.pars = np.array(pars)
        self.prior = prior

        self.np = self.pars.size

        if widths is None:
            self.widths = self.pars * 0 + 0.1
            self.widths[0:0+2] = 0.02
        else:
            self.widths = widths

    def __call__(self, nrand=None, obs=None):
        """
        Generate a guess

        Parameters
        ----------
        nrand: int, optional
            Number of samples to draw.  Default 1
        obs: ignored
            This keyword is here to conform to the interface
        """

        rng = self.rng

        if nrand is None:
            is_scalar = True
            nrand = 1
        else:
            is_scalar = False

        pars = self.pars
        widths = self.widths

        guess = np.zeros((nrand, self.np))
        guess[:, 0] = pars[0] + widths[0] * srandu(nrand, rng=rng)
        guess[:, 1] = pars[1] + widths[1] * srandu(nrand, rng=rng)

        # prevent from getting too large
        guess_shape = get_shape_guess(
            rng=rng,
            g1=pars[2],
            g2=pars[3],
            nrand=nrand,
            width=widths[2:2+2],
            max=0.8
        )
        guess[:, 2] = guess_shape[:, 0]
        guess[:, 3] = guess_shape[:, 1]

        for i in range(4, self.np):
            guess[:, i] = pars[i] * (1.0 + widths[i] * srandu(nrand, rng=rng))

        if self.prior is not None:
            _fix_guess(guess, self.prior)

        if is_scalar:
            guess = guess[0, :]

        return guess


def get_shape_guess(rng, g1, g2, nrand, width, max=0.99):
    """
    Get guess, making sure the range is OK
    """

    g = np.sqrt(g1 ** 2 + g2 ** 2)
    if g > max:
        fac = max / g

        g1 = g1 * fac
        g2 = g2 * fac

    guess = np.zeros((nrand, 2))
    shape = Shape(g1, g2)

    for i in range(nrand):

        while True:
            try:
                g1_offset = width[0] * srandu(rng=rng)
                g2_offset = width[1] * srandu(rng=rng)
                shape_new = shape.get_sheared(g1_offset, g2_offset)
                break
            except GMixRangeError:
                pass

        guess[i, 0] = shape_new.g1
        guess[i, 1] = shape_new.g2

    return guess


class R50FluxGuesser(object):
    """
    get full guesses from just r50 and fluxes

    parameters
    ----------
    r50: float
        Center for r50 (half light radius )guesses
    flux: float or sequences
        Center for flux guesses
    prior: optional
        If sent, "fix-up" guesses if they are not allowed by the prior
    """

    def __init__(self, rng, r50, flux, prior=None):

        self.rng = rng
        if r50 < 0.0:
            raise GMixRangeError("r50 <= 0: %g" % r50)

        self.r50 = r50

        self.fluxes = np.array(flux, dtype="f8", ndmin=1)
        self.prior = prior

    def __call__(self, nrand=1, obs=None):
        """
        Generate a guess.

        Parameters
        ----------
        nrand: int, optional
            Number of samples to draw.  Default 1
        obs: ignored
            This keyword is here to conform to the interface
        """

        rng = self.rng

        fluxes = self.fluxes
        nband = fluxes.size
        npars = 5 + nband

        guess = np.zeros((nrand, npars))
        guess[:, 0] = 0.01 * srandu(nrand, rng=rng)
        guess[:, 1] = 0.01 * srandu(nrand, rng=rng)
        guess[:, 2] = 0.02 * srandu(nrand, rng=rng)
        guess[:, 3] = 0.02 * srandu(nrand, rng=rng)

        guess[:, 4] = self.r50 * (1.0 + 0.1 * srandu(nrand, rng=rng))

        fluxes = self.fluxes
        for band in range(nband):
            guess[:, 5 + band] = fluxes[band] * (
                1.0 + 0.1 * srandu(nrand, rng=rng)
            )

        if self.prior is not None:
            _fix_guess(guess, self.prior)

        if nrand == 1:
            guess = guess[0, :]
        return guess


class PriorGuesser(object):
    """
    get guesses simply sampling from a prior

    Parameters
    ----------
    prior: joint prior
        A joint prior for all parameters
    """
    def __init__(self, prior):
        self.prior = prior

    def __call__(self, obs=None, nrand=None):
        """
        nrand: int, optional
            Number of samples to draw.  Default 1
        obs: ignored
            This keyword is here to conform to the interface
        """
        return self.prior.sample(nrand)


class R50NuFluxGuesser(R50FluxGuesser):
    """
    get full guesses from just r50 spergel nu and fluxes

    parameters
    ----------
    r50: float
        Center for r50 (half light radius )guesses
    nu: float
        Index for the spergel function
    flux: float or sequences
        Center for flux guesses
    prior: optional
        If sent, "fix-up" guesses if they are not allowed by the prior
    """

    NUMIN = -0.99
    NUMAX = 3.5

    def __init__(self, rng, r50, nu, flux, prior=None):
        super(R50NuFluxGuesser, self).__init__(
            rng=rng, r50=r50, flux=flux, prior=prior,
        )

        if nu < self.NUMIN:
            nu = self.NUMIN
        elif nu > self.NUMAX:
            nu = self.NUMAX

        self.nu = nu

    def __call__(self, nrand=1, obs=None):
        """
        Generate a guess

        Parameters
        ----------
        nrand: int, optional
            Number of samples to draw.  Default 1
        obs: ignored
            This keyword is here to conform to the interface
        """

        rng = self.rng

        fluxes = self.fluxes
        nband = fluxes.size
        npars = 6 + nband

        guess = np.zeros((nrand, npars))
        guess[:, 0] = 0.01 * srandu(nrand, rng=rng)
        guess[:, 1] = 0.01 * srandu(nrand, rng=rng)
        guess[:, 2] = 0.02 * srandu(nrand, rng=rng)
        guess[:, 3] = 0.02 * srandu(nrand, rng=rng)

        guess[:, 4] = self.r50 * (1.0 + 0.1 * srandu(nrand, rng=rng))

        for i in range(nrand):
            while True:
                nuguess = self.nu * (1.0 + 0.1 * srandu(rng=rng))
                if nuguess > self.NUMIN and nuguess < self.NUMAX:
                    break
            guess[i, 5] = nuguess

        fluxes = self.fluxes
        for band in range(nband):
            guess[:, 6 + band] = fluxes[band] * (
                1.0 + 0.1 * srandu(nrand, rng=rng)
            )

        if self.prior is not None:
            _fix_guess(guess, self.prior)

        if nrand == 1:
            guess = guess[0, :]
        return guess


class GMixPSFGuesser(object):
    """
    Generate a full gaussian mixture for a psf fit.  Useful for EM and admom

    Parameters
    ----------
    rng: numpy.random.RandomState
        Random state for generating guesses
    ngauss: int
        number of gaussians
    guess_from_moms: bool, optional
        If set to True, use weighted moments to generate the starting flux and
        T for the guess.  If set to False, the starting flux is gotten from
        summing the image and the fwhm of the guess is set to 3.5 times the
        pixel scale
    """
    def __init__(self, rng, ngauss, guess_from_moms=False):

        self.rng = rng
        self.ngauss = ngauss
        self.guess_from_moms = guess_from_moms

        if self.ngauss == 1:
            self._guess_func = self._get_guess_1gauss
        elif self.ngauss == 2:
            self._guess_func = self._get_guess_2gauss
        elif self.ngauss == 3:
            self._guess_func = self._get_guess_3gauss
        elif self.ngauss == 4:
            self._guess_func = self._get_guess_4gauss
        elif self.ngauss == 5:
            self._guess_func = self._get_guess_5gauss
        else:
            raise ValueError("bad ngauss: %d" % self.ngauss)

    def __call__(self, obs):
        """
        Get a guess for the EM algorithm

        Parameters
        ----------
        obs: Observation
            Starting flux and T for the overall mixture are derived from the
            input observation.  How depends on the gauss_from_moms constructor
            argument

        Returns
        -------
        ngmix.GMix
            The guess mixture, with the number of gaussians as specified in the
            constructor
        """
        return self._get_guess(obs=obs)

    def _get_guess(self, obs):
        T, flux = self._get_T_flux(obs=obs)
        return self._guess_func(flux=flux, T=T)

    def _get_T_flux(self, obs):
        if self.guess_from_moms:
            T, flux = self._get_T_flux_from_moms(obs=obs)
        else:
            T, flux = self._get_T_flux_default(obs=obs)

        return T, flux

    def _get_T_flux_default(self, obs):
        """
        get starting T and flux from a multiple of the pixel scale and the sum
        of the image
        """
        scale = obs.jacobian.scale
        flux = obs.image.sum()
        # for DES 0.9/0.263 = 3.42
        fwhm = scale * 3.5
        T = moments.fwhm_to_T(fwhm)
        return T, flux

    def _get_T_flux_from_moms(self, obs):
        """
        get starting T and flux from weighted moments, deweighted

        if the measured T is too small, fall back to the _get_T_flux method
        """
        scale = obs.jacobian.scale

        # 0.9/0.263 = 3.42
        fwhm = scale * 3.5

        Tweight = moments.fwhm_to_T(fwhm)
        wt = GMixModel([0.0, 0.0, 0.0, 0.0, Tweight, 1.0], "gauss")

        res = wt.get_weighted_moments(obs=obs, maxrad=1.0e9)
        if res['flags'] != 0:
            return self._get_T_flux_default(obs=obs)

        # ngmix is in flux units, need to divide by area
        area = scale**2

        Tmeas = res['T']

        fwhm_meas = moments.T_to_fwhm(Tmeas)
        if fwhm_meas < scale:
            # something probably went wrong
            T, flux = self._get_T_flux_default(obs=obs)
        else:
            # deweight assuming true profile is a gaussian
            T = 1.0/(1/Tmeas - 1/Tweight)
            flux = res['flux'] * np.pi * (Tweight + T) / area

        return T, flux

    def _get_guess_1gauss(self, flux, T):
        rng = self.rng

        sigma2 = T/2

        pars = np.array([
            flux * rng.uniform(low=0.9, high=1.1),
            rng.uniform(low=-0.1, high=0.1),
            rng.uniform(low=-0.1, high=0.1),
            sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.2 * sigma2, high=0.2 * sigma2),
            sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
        ])

        return GMix(pars=pars)

    def _get_guess_2gauss(self, flux, T):
        rng = self.rng

        sigma2 = T/2

        pars = np.array([
            _em2_pguess[0] * flux,
            rng.uniform(low=-0.1, high=0.1),
            rng.uniform(low=-0.1, high=0.1),
            _em2_fguess[0] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            0.0,
            _em2_fguess[0] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),

            _em2_pguess[1] * flux,
            rng.uniform(low=-0.1, high=0.1),
            rng.uniform(low=-0.1, high=0.1),
            _em2_fguess[1] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            0.0,
            _em2_fguess[1] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
        ])

        return GMix(pars=pars)

    def _get_guess_3gauss(self, flux, T):
        rng = self.rng

        sigma2 = T/2

        pars = np.array([
            flux * _em3_pguess[0] * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.1, high=0.1),
            rng.uniform(low=-0.1, high=0.1),
            _em3_fguess[0] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.01, high=0.01),
            _em3_fguess[0] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),

            flux * _em3_pguess[1] * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.1, high=0.1),
            rng.uniform(low=-0.1, high=0.1),
            _em3_fguess[1] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.01, high=0.01),
            _em3_fguess[1] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),

            flux * _em3_pguess[2] * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.1, high=0.1),
            rng.uniform(low=-0.1, high=0.1),
            _em3_fguess[2] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.01, high=0.01),
            _em3_fguess[2] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
        ])

        return GMix(pars=pars)

    def _get_guess_4gauss(self, flux, T):
        rng = self.rng

        sigma2 = T/2

        pars = np.array([
            flux * _em4_pguess[0] * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.1, high=0.1),
            rng.uniform(low=-0.1, high=0.1),
            _em4_fguess[0] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.01, high=0.01),
            _em4_fguess[0] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),

            flux * _em4_pguess[1] * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.1, high=0.1),
            rng.uniform(low=-0.1, high=0.1),
            _em4_fguess[1] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.01, high=0.01),
            _em4_fguess[1] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),

            flux * _em4_pguess[2] * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.1, high=0.1),
            rng.uniform(low=-0.1, high=0.1),
            _em4_fguess[2] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.01, high=0.01),
            _em4_fguess[2] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),

            flux * _em4_pguess[2] * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.1, high=0.1),
            rng.uniform(low=-0.1, high=0.1),
            _em4_fguess[2] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.01, high=0.01),
            _em4_fguess[2] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
        ])

        return GMix(pars=pars)

    def _get_guess_5gauss(self, flux, T):
        rng = self.rng

        sigma2 = T/2

        pars = np.array([
            flux * _em5_pguess[0] * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.1, high=0.1),
            rng.uniform(low=-0.1, high=0.1),
            _em5_fguess[0] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.01, high=0.01),
            _em5_fguess[0] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),

            flux * _em5_pguess[1] * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.1, high=0.1),
            rng.uniform(low=-0.1, high=0.1),
            _em5_fguess[1] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.01, high=0.01),
            _em5_fguess[1] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),

            flux * _em5_pguess[2] * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.1, high=0.1),
            rng.uniform(low=-0.1, high=0.1),
            _em5_fguess[2] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.01, high=0.01),
            _em5_fguess[2] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),

            flux * _em5_pguess[2] * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.1, high=0.1),
            rng.uniform(low=-0.1, high=0.1),
            _em5_fguess[2] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.01, high=0.01),
            _em5_fguess[2] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),

            flux * _em5_pguess[2] * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.1, high=0.1),
            rng.uniform(low=-0.1, high=0.1),
            _em5_fguess[2] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
            rng.uniform(low=-0.01, high=0.01),
            _em5_fguess[2] * sigma2 * (1.0 + rng.uniform(low=-0.1, high=0.1)),
        ])

        return GMix(pars=pars)


_em2_pguess = np.array([0.596510042804182, 0.4034898268889178])
_em2_fguess = np.array([0.5793612389470884, 1.621860687127999])

_em3_pguess = np.array(
    [0.596510042804182, 0.4034898268889178, 1.303069003078001e-07]
)
_em3_fguess = np.array([0.5793612389470884, 1.621860687127999, 7.019347162356363])

_em4_pguess = np.array(
    [0.596510042804182, 0.4034898268889178, 1.303069003078001e-07, 1.0e-8]
)
_em4_fguess = np.array(
    [0.5793612389470884, 1.621860687127999, 7.019347162356363, 16.0]
)

_em5_pguess = np.array(
    [0.59453032, 0.35671819, 0.03567182, 0.01189061, 0.00118906]
)
_em5_fguess = np.array([0.5, 1.0, 3.0, 10.0, 20.0])


class SimplePSFGuesser(GMixPSFGuesser):
    """
    guesser for simple psf fitting

    Parameters
    ----------
    rng: numpy.random.RandomState
        Random state for generating guesses
    guess_from_moms: bool, optional
        If set to True, use weighted moments to generate the starting flux and
        T for the guess.  If set to False, the starting flux is gotten from
        summing the image and the fwhm of the guess isset to 3.5 times the
        pixel scale
    """
    def __init__(self, rng, guess_from_moms=False):

        self.rng = rng
        self.guess_from_moms = guess_from_moms
        self.npars = 6

    def __call__(self, obs):
        """
        Get a guess for the simple psf

        Parameters
        ----------
        obs: Observation
            Starting flux and T for the overall mixture are derived from the
            input observation.  How depends on the gauss_from_moms constructor
            argument

        Returns
        -------
        guess: array
            The guess array [cen1, cen2, g1, g2, T, flux]
        """
        return self._get_guess(obs=obs)

    def _get_guess(self, obs):
        rng = self.rng
        T, flux = self._get_T_flux(obs=obs)

        guess = np.zeros(self.npars)

        guess[0:0 + 2] += rng.uniform(low=-0.01, high=0.01, size=2)
        guess[2:2 + 2] += rng.uniform(low=-0.05, high=0.05, size=2)

        guess[4] = T * rng.uniform(low=0.9, high=1.1)
        guess[5] = flux * rng.uniform(low=0.9, high=1.1)
        return guess


class CoellipPSFGuesser(GMixPSFGuesser):
    """
    guesser for coelliptical psf fitting

    Parameters
    ----------
    rng: numpy.random.RandomState
        Random state for generating guesses
    ngauss: int
        number of gaussians
    guess_from_moms: bool, optional
        If set to True, use weighted moments to generate the starting flux and
        T for the guess.  If set to False, the starting flux is gotten from
        summing the image and the fwhm of the guess isset to 3.5 times the
        pixel scale
    """
    def __init__(self, rng, ngauss, guess_from_moms=False):
        super().__init__(
            rng=rng,
            ngauss=ngauss,
            guess_from_moms=guess_from_moms,
        )
        self.npars = get_coellip_npars(ngauss)

    def __call__(self, obs):
        """
        Get a guess for the EM algorithm

        Parameters
        ----------
        obs: Observation
            Starting flux and T for the overall mixture are derived from the
            input observation.  How depends on the gauss_from_moms constructor
            argument

        Returns
        -------
        guess: array
            The guess array, [cen1, cen2, g1, g2, T1, T2, ..., F1, F2, ...]
        """
        return self._get_guess(obs=obs)

    def _make_guess_array(self):
        rng = self.rng
        guess = np.zeros(self.npars)

        guess[0:0 + 2] += rng.uniform(low=-0.01, high=0.01, size=2)
        guess[2:2 + 2] += rng.uniform(low=-0.05, high=0.05, size=2)
        return guess

    def _get_guess_1gauss(self, flux, T):
        rng = self.rng
        guess = self._make_guess_array()

        guess[4] = T * rng.uniform(low=0.9, high=1.1)
        guess[5] = flux * rng.uniform(low=0.9, high=1.1)
        return guess

    def _get_guess_2gauss(self, flux, T):
        rng = self.rng
        guess = self._make_guess_array()

        low, high = 0.99, 1.01
        guess[4] = T * _moffat2_fguess[0] * rng.uniform(low=low, high=high)
        guess[5] = T * _moffat2_fguess[1] * rng.uniform(low=low, high=high)
        guess[6] = flux * _moffat2_pguess[0] * rng.uniform(low=low, high=high)
        guess[7] = flux * _moffat2_pguess[1] * rng.uniform(low=low, high=high)

        return guess

    def _get_guess_3gauss(self, flux, T):
        rng = self.rng
        guess = self._make_guess_array()

        low, high = 0.99, 1.01
        guess[4] = T * _moffat3_fguess[0] * rng.uniform(low=low, high=high)
        guess[5] = T * _moffat3_fguess[1] * rng.uniform(low=low, high=high)
        guess[6] = T * _moffat3_fguess[2] * rng.uniform(low=low, high=high)

        guess[7] = flux * _moffat3_pguess[0] * rng.uniform(low=low, high=high)
        guess[8] = flux * _moffat3_pguess[1] * rng.uniform(low=low, high=high)
        guess[9] = flux * _moffat3_pguess[2] * rng.uniform(low=low, high=high)
        return guess

    def _get_guess_4gauss(self, flux, T):
        rng = self.rng
        guess = self._make_guess_array()

        low, high = 0.99, 1.01
        guess[4] = T * _moffat4_fguess[0] * rng.uniform(low=low, high=high)
        guess[5] = T * _moffat4_fguess[1] * rng.uniform(low=low, high=high)
        guess[6] = T * _moffat4_fguess[2] * rng.uniform(low=low, high=high)
        guess[7] = T * _moffat4_fguess[3] * rng.uniform(low=low, high=high)

        guess[8] = flux * _moffat4_pguess[0] * rng.uniform(low=low, high=high)
        guess[9] = flux * _moffat4_pguess[1] * rng.uniform(low=low, high=high)
        guess[10] = flux * _moffat4_pguess[2] * rng.uniform(low=low, high=high)
        guess[11] = flux * _moffat4_pguess[3] * rng.uniform(low=low, high=high)
        return guess

    def _get_guess_5gauss(self, flux, T):
        rng = self.rng
        guess = self._make_guess_array()

        low, high = 0.99, 1.01
        guess[4] = T * _moffat5_fguess[0] * rng.uniform(low=low, high=high)
        guess[5] = T * _moffat5_fguess[1] * rng.uniform(low=low, high=high)
        guess[6] = T * _moffat5_fguess[2] * rng.uniform(low=low, high=high)
        guess[7] = T * _moffat5_fguess[3] * rng.uniform(low=low, high=high)
        guess[8] = T * _moffat5_fguess[4] * rng.uniform(low=low, high=high)

        guess[9] = flux * _moffat5_pguess[0] * rng.uniform(low=low, high=high)
        guess[10] = flux * _moffat5_pguess[1] * rng.uniform(low=low, high=high)
        guess[11] = flux * _moffat5_pguess[2] * rng.uniform(low=low, high=high)
        guess[12] = flux * _moffat5_pguess[3] * rng.uniform(low=low, high=high)
        guess[13] = flux * _moffat5_pguess[4] * rng.uniform(low=low, high=high)
        return guess


_moffat2_pguess = np.array([0.5, 0.5])
_moffat2_fguess = np.array([0.48955064, 1.50658978])

_moffat3_pguess = np.array([0.27559669, 0.55817131, 0.166232])
_moffat3_fguess = np.array([0.36123609, 0.8426139, 2.58747785])

_moffat4_pguess = np.array([0.44534, 0.366951, 0.10506, 0.0826497])
_moffat4_fguess = np.array([0.541019, 1.19701, 0.282176, 3.51086])

_moffat5_pguess = np.array([0.45, 0.25, 0.15, 0.1, 0.05])
_moffat5_fguess = np.array([0.541019, 1.19701, 0.282176, 3.51086])

_moffat5_pguess = np.array(
    [0.57874897, 0.32273483, 0.03327272, 0.0341253, 0.03111819]
)
_moffat5_fguess = np.array(
    [0.27831284, 0.9959897, 5.86989779, 5.63590429, 4.17285878]
)


def _fix_guess_TFlux(guess, prior, ntry=4):
    """
    just fix T and flux
    """

    n = guess.shape[0]
    for j in range(n):
        for itry in range(ntry):
            try:
                lnp = prior.get_lnprob_scalar(guess[j, :])

                if lnp <= LOWVAL:
                    dosample = True
                else:
                    dosample = False
            except GMixRangeError:
                dosample = True

            if dosample:
                print_pars(guess[j, :], front="bad guess:", logger=LOGGER)
                if itry < ntry:
                    tguess = prior.sample()
                    guess[j, 4:] = tguess[4:]
                else:
                    # give up and just drawn a sample
                    guess[j, :] = prior.sample()
            else:
                break


def _fix_guess(guess, prior, ntry=4):
    """
    Fix a guess for out-of-bounds values according the the input prior

    Bad guesses are replaced by a sample from the prior
    """

    n = guess.shape[0]
    for j in range(n):
        for itry in range(ntry):
            try:
                lnp = prior.get_lnprob_scalar(guess[j, :])

                if lnp <= LOWVAL:
                    dosample = True
                else:
                    dosample = False
            except GMixRangeError:
                dosample = True

            if dosample:
                print_pars(guess[j, :], front="bad guess:", logger=LOGGER)
                guess[j, :] = prior.sample()
            else:
                break
