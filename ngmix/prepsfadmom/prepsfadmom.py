"""
Adaptive moments in Fourier space, measuring pre-PSF moments.

The usual adaptive moments iteration is performed, but the weighted
real-space moment sums are evaluated in k-space against the
PSF-deconvolved FFT of the image, using derivatives of the elliptical
gaussian weight

    W(k) = exp(-k^T wt_cov k / 2)

where wt_cov is the real-space covariance matrix of the weight.  Because
the moment sums are linear in the image, multiple epochs and bands can
be accumulated for a joint fit with a common center and covariance; the
per-band fluxes are then measured with the common converged weight,
providing consistent pre-seeing apertures across bands (colors are
independent of the per-band PSFs).

The PSF deconvolution amplifies noise at high k.  To stabilize the fit,
the deconvolved image is smoothed by a common round gaussian with fwhm
fwhm_smooth, which should be at least as large as the largest PSF among
the input images.  The smoothing covariance is subtracted from the
converged weight to produce the galaxy covariance; this correction is
exact for gaussian profiles.  Without smoothing the iteration is
unstable in the presence of noise unless the object is very well
resolved.
"""
__all__ = [
    'run_prepsf_admom', 'PAdmomFitter', 'PAdmomResult',
    'get_phase_angles', 'deweight',
]

import logging

import numpy as np

from ..observation import get_mb_obs, MultiBandObsList
from ..gmix import GMix, GMixModel
from ..gmix.gmix_nb import GMIX_LOW_DETVAL
from ..moments import fwhm_to_T, e2mom
from ..shape import e1e2_to_g1g2
from ..util import get_ratio_error
from .prepsfadmom_nb import admom_ksums, admom_finalize
from ..moments import cov_from_e, mom_from_cov, cov_from_mom
from .models import model_ksums, mixture_model_valid, det2
from .errors import (
    flux_cov_delta, model_sandwich, joint_flux_s2n,
)
from .prep import choose_fwhm_smooth, prep_epoch, DEFAULT_SMOOTH_FAC
import ngmix.flags

logger = logging.getLogger(__name__)

DEFAULT_MAXITER = 200
DEFAULT_SHIFTMAX = 5.0  # pixels for scale=1
DEFAULT_ETOL = 1.0e-5
DEFAULT_CENTOL = 1.0e-4  # pixels for scale=1


def run_prepsf_admom(
    obs, guess=None,
    model='gauss',
    fwhm_smooth=None,
    smooth_fac=DEFAULT_SMOOTH_FAC,
    pad_factor=4,
    ap_rad=1.5,
    maxiter=DEFAULT_MAXITER,
    shiftmax=DEFAULT_SHIFTMAX,
    etol=DEFAULT_ETOL,
    cen_tol=DEFAULT_CENTOL,
    no_psf=False,
    use_noise_image=False,
    fixcen=False,
    full_errors=False,
    noise_autocov=None,
    rng=None,
):
    """
    Run pre-PSF adaptive moments on the observation(s)

    Parameters
    ----------
    obs: Observation, ObsList, or MultiBandObsList
        The observation(s) to fit.  All epochs and bands are fit jointly
        with a common center and covariance; fluxes are measured per
        band.
    guess: ngmix.GMix or a float, optional
        A guess for the fitter.  Can be a full gaussian mixture or a
        single value for the pre-PSF T, in which case the rest of the
        parameters are generated.  If not sent, a default guess is used.
    model: str, optional
        The object model: 'gauss' (default), 'exp', 'dev' or
        'star'.  With 'gauss' the fit is the standard adaptive
        moments fixed point.  With
        'exp' (or 'dev') the ngmix
        6-gaussian exponential (10-gaussian de Vaucouleurs)
        expansion is fit
        by matching its weighted moments to the measured ones; T, e1
        and e2 are then the exponential family parameters and the
        fluxes are total model fluxes rather than gaussian aperture
        fluxes.  The family size is not clipped and can scatter
        through zero for marginally resolved objects, keeping
        ensemble averages unbiased; the shape is then undefined and
        NONPOS_SIZE is set while T and the fluxes remain usable.
        With 'star' the object is a pre-psf delta function:
        the weight is the smoothing gaussian, only the center and the
        per band fluxes are fit, and fwhm_smooth must be positive.
    fwhm_smooth: float, optional
        The fwhm of the common round gaussian smoothing applied to the
        deconvolved images.  If not sent, it is chosen automatically as
        smooth_fac times the largest PSF fwhm among the input images.
        Send 0 to disable smoothing; this is only stable for noiseless
        or very well resolved data.
    smooth_fac: float, optional
        Factor by which to multiply the largest PSF fwhm when choosing
        the smoothing automatically.  Default 1.05.
    pad_factor: int, optional
        The factor by which to pad the FFTs used for the image.
        Default is 4.
    ap_rad: float, optional
        The apodization radius for the stamp in pixels, default 1.5.
    maxiter: integer, optional
        Maximum number of iterations, default 200
    shiftmax: float, optional
        Largest allowed shift in the centroid, relative to the initial
        guess.  Default 5.0 (5 pixels if the jacobian scale is 1)
    etol: float, optional
        relative tolerance on the fixed point residual to determine
        convergence: the largest element of the covariance update
        step (the weight for 'gauss', the family covariance for
        'exp'/'dev') must fall below etol times the trace of the
        updated weight.  Default 1.0e-5
    cen_tol: float, optional
        absolute tolerance in the center to determine convergence,
        default 1.0e-4 (1.0e-4 pixels if the jacobian scale is 1)
    no_psf: bool, optional
        If True, allow inputs without a PSF observation; only the pixel
        window function is deconvolved.  fwhm_smooth= must be sent in
        this case.  Defaults to False.
    use_noise_image: bool, optional
        If True, estimate the per-mode noise power for the error
        estimates from the noise image attached to each observation
        (obs.noise), instead of assuming white noise at the level set
        by the weight map.  This makes the errors correct for
        stationary correlated noise, such as that induced by metacal.
        The noise image must be an independent realization of the
        noise, in the same frame as the image (e.g. as maintained by
        metacal).  The measured moments are unchanged.  Default False.
    fixcen: bool, optional
        If True, the center is held fixed at the guess instead of
        being fit; the moments are measured about that center.
        Default False.
    full_errors: bool, optional
        If True, replace the flux, T and shape errors, including
        those of the gauss_* entries, with the full fixed-point
        errors (see full_errors.py), which stay calibrated under
        model mismatch, and set the cross-band flux_cov.
        Supported for the 'gauss', 'exp' and 'dev'
        models.  The fit takes about three times longer.  When the
        full errors cannot be evaluated the default errors are
        kept.  The noise is taken from the weight maps, treating
        the pixels as independent, or with use_noise_image=True
        from the noise images, as stationary correlated noise; in
        that case the weight maps only mark the good pixels
        (weight > 0).  Default False.
    noise_autocov: array or sequence of arrays, optional
        The noise autocovariance for the full_errors=True errors,
        as a lag array from ngmix.prepsfadmom.full_errors
        noise_autocov: one array for every band, or a sequence
        with one per band.  Used in place of the autocovariance
        measured from the noise image of each observation, which
        is noisy for a small stamp; measure it on a larger region
        with the same noise.  The weight maps only mark the good
        pixels.  It does not change the default errors.
    rng: np.random.RandomState, optional
        Random state used to generate guesses from a T guess and for
        PSF fits when choosing the smoothing automatically.

    Returns
    -------
    PAdmomResult
    """
    fitter = PAdmomFitter(
        model=model,
        fwhm_smooth=fwhm_smooth,
        smooth_fac=smooth_fac,
        pad_factor=pad_factor,
        ap_rad=ap_rad,
        maxiter=maxiter,
        shiftmax=shiftmax,
        etol=etol,
        cen_tol=cen_tol,
        use_noise_image=use_noise_image,
        fixcen=fixcen,
        full_errors=full_errors,
        noise_autocov=noise_autocov,
        rng=rng,
    )
    return fitter.go(obs=obs, guess=guess, no_psf=no_psf)


class PAdmomResult(dict):
    """
    Represent a pre-PSF adaptive moments fit.  This inherits from dict
    and has entries for the results of the fit.

    Parameters
    ----------
    obs: observation(s)
        Observation, ObsList, or MultiBandObsList
    result: dict
        the basic fit result, to be added to this object's keys

    The entries will include, but not be limited to
    -----------------------------------------------
    flags: int
        flags for processing
    flagstr: str
        Explanation of flags
    numiter: int
        number of iterations in the adaptive moments algorithm
    cen: array
        Offset of the common center from the image jacobian centers
    e1, e2: float
        Pre-PSF ellipticity parameters, with errors e1err, e2err
    e_cov: array (2, 2)
        The covariance matrix of e1 and e2, including their
        covariance
    T: float
        Pre-PSF T, with error T_err.  This is the T of the converged
        weight with the smoothing subtracted, and can be non-positive
        for marginally resolved objects.
    flux: float or array
        Flux per band, measured with the common converged weight.  This
        is an array if the input was a MultiBandObsList, otherwise a
        scalar.  Errors are in flux_err.
    s2n: float
        The total flux s/n.  Where the cross-band flux covariance
        is available (flux_cov) this is the covariance-aware
        joint value sqrt(F^T C^-1 F), pricing the cross-band
        correlations from the shared structure response;
        otherwise the independent-band quadrature sum, which is
        exact on the diagonal paths (e.g. stars).
    fwhm_smooth: float
        The smoothing fwhm that was used
    pars: array
        Array of gaussian pars [v, u, M1, M2, T, flux], where flux is
        summed over bands
    """

    def __init__(self, obs, result):
        self._obs = obs
        self.update(result)

    def get_gmix(self):
        """
        get a gmix representing the best fit pre-PSF model, normalized
        to unit flux
        """
        if self['flags'] != 0:
            raise RuntimeError('cannot create gmix, fit failed')

        model = self.get('model', 'gauss')

        pars = self['pars'].copy()
        pars[5] = 1.0

        if model == 'star':
            # a T=0 gaussian representing the pre-psf delta function;
            # this is allowed so it can be convolved with a psf later,
            # and only errors if evaluated directly
            pars[2] = 0.0
            pars[3] = 0.0
            return GMixModel(pars, 'gauss')

        e1 = pars[2] / pars[4]
        e2 = pars[3] / pars[4]

        g1, g2 = e1e2_to_g1g2(e1, e2)
        pars[2] = g1
        pars[3] = g2

        return GMixModel(pars, model)


class PAdmomFitter(object):
    """
    Measure pre-PSF adaptive moments in Fourier space

    Parameters
    ----------
    model: str, optional
        The object model: 'gauss' (default), 'exp', 'dev' or
        'star'.  With 'gauss' the fit is the standard adaptive
        moments fixed point.  With 'exp' (or 'dev') the ngmix
        6-gaussian exponential (10-gaussian de Vaucouleurs)
        expansion is fit
        by matching its weighted moments to the measured ones; T, e1
        and e2 are then the exponential family parameters and the
        fluxes are total model fluxes rather than gaussian aperture
        fluxes.  The family size is not clipped and can scatter
        through zero for marginally resolved objects, keeping
        ensemble averages unbiased; the shape is then undefined and
        NONPOS_SIZE is set while T and the fluxes remain usable.
        With 'star' the object is a pre-psf delta function:
        the weight is the smoothing gaussian, only the center and the
        per band fluxes are fit, and fwhm_smooth must be positive.
    fwhm_smooth: float, optional
        The fwhm of the common round gaussian smoothing applied to the
        deconvolved images.  If not sent, it is chosen automatically as
        smooth_fac times the largest PSF fwhm among the input images.
        Send 0 to disable smoothing; this is only stable for noiseless
        or very well resolved data.
    smooth_fac: float, optional
        Factor by which to multiply the largest PSF fwhm when choosing
        the smoothing automatically.  Default 1.05.
    pad_factor: int, optional
        The factor by which to pad the FFTs used for the image.
        Default is 4.
    ap_rad: float, optional
        The apodization radius for the stamp in pixels, default 1.5.
    maxiter: integer, optional
        Maximum number of iterations, default 200
    shiftmax: float, optional
        Largest allowed shift in the centroid, relative to the initial
        guess.  Default 5.0 (5 pixels if the jacobian scale is 1)
    etol: float, optional
        relative tolerance on the fixed point residual to determine
        convergence: the largest element of the covariance update
        step (the weight for 'gauss', the family covariance for
        'exp'/'dev') must fall below etol times the trace of the
        updated weight.  Default 1.0e-5
    cen_tol: float, optional
        absolute tolerance in the center to determine convergence,
        default 1.0e-4 (1.0e-4 pixels if the jacobian scale is 1)
    use_noise_image: bool, optional
        If True, estimate the per-mode noise power for the error
        estimates from the noise image attached to each observation
        (obs.noise), instead of assuming white noise at the level set
        by the weight map.  This makes the errors correct for
        stationary correlated noise, such as that induced by metacal.
        The noise image must be an independent realization of the
        noise, in the same frame as the image (e.g. as maintained by
        metacal).  The measured moments are unchanged.  Default False.
    fixcen: bool, optional
        If True, the center is held fixed at the guess instead of
        being fit; the moments are measured about that center.
        Default False.
    full_errors: bool, optional
        If True, replace the flux, T and shape errors, including
        those of the gauss_* entries, with the full fixed-point
        errors (see full_errors.py), which stay calibrated under
        model mismatch, and set the cross-band flux_cov.
        Supported for the 'gauss', 'exp' and 'dev'
        models.  The fit takes about three times longer.  When the
        full errors cannot be evaluated the default errors are
        kept.  The noise is taken from the weight maps, treating
        the pixels as independent, or with use_noise_image=True
        from the noise images, as stationary correlated noise; in
        that case the weight maps only mark the good pixels
        (weight > 0).  Default False.
    noise_autocov: array or sequence of arrays, optional
        The noise autocovariance for the full_errors=True errors,
        as a lag array from ngmix.prepsfadmom.full_errors
        noise_autocov: one array for every band, or a sequence
        with one per band.  Used in place of the autocovariance
        measured from the noise image of each observation, which
        is noisy for a small stamp; measure it on a larger region
        with the same noise.  The weight maps only mark the good
        pixels.  It does not change the default errors.
    rng: np.random.RandomState, optional
        Random state used to generate guesses from a T guess and for
        PSF fits when choosing the smoothing automatically.

    Notes
    -----
    The reported errors include the first order response of the
    adaptive weight to the noise, propagated through the fixed point
    conditions of the iteration (the delta method), with the
    derivatives evaluated analytically for the gaussian implied by
    the converged weight.  For the flux this doubles the naive fixed
    weight variance of a gaussian object; the T and shape errors
    carry the same response through the deweighting factors.  The
    errors are first order and remain approximate at very low s/n or
    for strongly non-gaussian profiles.

    For model='exp' the flux, T and shape errors come from the full
    sandwich over the moment matching conditions.  "Sandwich" is the
    standard covariance of an estimator defined by conditions
    g(theta; data) = 0: at first order Cov(theta) = J^-1 Cov(g) J^-T
    with J = dg/dtheta.  The model derivatives in J are
    evaluated in closed form for the mixture (see model_sandwich);
    the only approximation beyond first order is replacing the data
    by the converged model, which is second order at the matched
    moments.  The errors match the observed scatter at the ~5
    percent level at s/n 20, improving with s/n.  For model='star'
    the weight is frozen and the fixed weight flux errors are exact.

    By default the noise is assumed white, with each Fourier mode
    assigned the same power, set by the weight map.  With
    use_noise_image=True the power is instead measured per mode from
    the attached noise realization, which is exact in the mean for any
    stationary noise; the single-realization scatter self-averages in
    the error sums over many modes.

    Bad pixels cannot be masked in Fourier space; images should have
    defects interpolated before fitting.  The weight maps set the
    overall noise level of each epoch and the relative weights of
    epochs in the joint fit; the latter holds even when
    use_noise_image=True.
    """

    kind = 'pam'

    def __init__(
        self,
        model='gauss',
        fwhm_smooth=None,
        smooth_fac=DEFAULT_SMOOTH_FAC,
        pad_factor=4,
        ap_rad=1.5,
        maxiter=DEFAULT_MAXITER,
        shiftmax=DEFAULT_SHIFTMAX,
        etol=DEFAULT_ETOL,
        cen_tol=DEFAULT_CENTOL,
        use_noise_image=False,
        fixcen=False,
        full_errors=False,
        noise_autocov=None,
        rng=None,
    ):

        self.model = _check_model(model)
        self.fixcen = fixcen
        self.full_errors = full_errors
        if full_errors:
            if self.model not in ('gauss', 'exp', 'dev'):
                raise ValueError(
                    'full_errors supports the gauss, exp and '
                    f'dev models, got {self.model!r}'
                )
        self.fwhm_smooth = fwhm_smooth
        self.smooth_fac = smooth_fac
        self.pad_factor = pad_factor
        self.ap_rad = ap_rad
        self.maxiter = maxiter
        self.shiftmax = shiftmax
        self.etol = etol
        self.cen_tol = cen_tol
        self.use_noise_image = use_noise_image
        self.noise_autocov = noise_autocov
        self.rng = rng

    def go(self, obs, guess=None, no_psf=False):
        """
        run the pre-PSF adaptive moments

        Parameters
        ----------
        obs: Observation, ObsList, or MultiBandObsList
            The observation(s) to fit.  All epochs and bands are fit
            jointly with a common center and covariance; fluxes are
            measured per band.
        guess: ngmix.GMix or a float, optional
            A guess for the fitter.  Can be a full gaussian mixture or a
            single value for the pre-PSF T, in which case the rest of
            the parameters are generated.  If not sent, a default guess
            is used.
        no_psf: bool, optional
            If True, allow inputs without a PSF observation; only the
            pixel window function is deconvolved.  fwhm_smooth= must be
            sent in this case.  Defaults to False.

        Returns
        -------
        PAdmomResult
        """
        is_mb = isinstance(obs, MultiBandObsList)
        mb_obs = get_mb_obs(obs)

        fwhm_smooth = choose_fwhm_smooth(
            mb_obs, fwhm_smooth=self.fwhm_smooth,
            smooth_fac=self.smooth_fac, no_psf=no_psf,
            rng=self._get_rng(),
        )
        Tsmooth = fwhm_to_T(fwhm_smooth) if fwhm_smooth > 0 else 0.0

        if self.model == 'star' and Tsmooth <= 0:
            raise ValueError(
                'the star model requires positive fwhm_smooth'
            )

        autocovs = self._get_noise_autocovs(len(mb_obs))

        epochs = []
        for band, obslist in enumerate(mb_obs):
            for tobs in obslist:
                ep = prep_epoch(
                    tobs, band=band, fwhm_smooth=fwhm_smooth,
                    pad_factor=self.pad_factor, ap_rad=self.ap_rad,
                    use_noise_image=self.use_noise_image,
                    no_psf=no_psf,
                    store_transfer=self.full_errors,
                )
                if self.full_errors:
                    # the influence-kernel covariance needs the
                    # per-pixel variances, or with use_noise_image
                    # the noise image and the good pixels
                    ep['obs_weight'] = tobs.weight
                    if self.use_noise_image:
                        ep['obs_noise'] = tobs.noise
                    if autocovs is not None:
                        # takes the place of the noise image
                        ep['noise_autocov'] = autocovs[band]
                epochs.append(ep)

        if len(epochs) == 0:
            raise ValueError('no epochs sent')

        guess_gmix = self._get_guess(mb_obs, guess, Tsmooth)

        result = self._run_admom(
            epochs=epochs,
            nband=len(mb_obs),
            guess_gmix=guess_gmix,
            Tsmooth=Tsmooth,
        )
        result['fwhm_smooth'] = fwhm_smooth
        result['Tsmooth'] = Tsmooth

        if not is_mb:
            result['flux'] = result['flux'][0]
            result['flux_err'] = result['flux_err'][0]

        return PAdmomResult(obs=obs, result=result)

    def _run_admom(self, epochs, nband, guess_gmix, Tsmooth):
        """
        the adaptive moments iteration, accumulating the k-space sums
        over all epochs

        Parameters
        ----------
        epochs: list of dict
            The prep_epoch dicts, one per epoch
        nband: int
            The number of bands
        guess_gmix: ngmix.GMix
            The guess, providing the starting center, shape and T
        Tsmooth: float
            The T of the gaussian smoothing
        """
        if self.model in ('exp', 'dev'):
            return self._run_admom_mixture(
                epochs, nband, guess_gmix, Tsmooth,
            )
        elif self.model == 'star':
            return self._run_admom_star(epochs, nband, guess_gmix, Tsmooth)
        else:
            return self._run_admom_gauss(epochs, nband, guess_gmix, Tsmooth)

    def _measure_step(self, epochs, wt_cov, v0, u0, vorig, uorig):
        """
        one measurement pass under the current weight, shared by the
        gauss and mixture iterations: validate the weight, accumulate
        the k-space sums, update the center, and assemble the measured
        moment matrix.  Returns (flags, v0, u0, dv, du, M); the
        iteration cannot continue when flags is nonzero, and M is None
        in that case

        The center update refers the second moments to the updated
        center with the exact centroid correction
        <(x - d)_i (x - d)_j> = <x_i x_j> - d_i d_j.  The weight
        center lags by one iteration, which vanishes at the fixed
        point

        Parameters
        ----------
        epochs: list of dict
            The prep_epoch dicts, one per epoch
        wt_cov: (2, 2) array
            The weight covariance
        v0, u0: float
            The current center
        vorig, uorig: float
            The starting center, for the shiftmax check
        """
        if (wt_cov[0, 0] <= 0 or wt_cov[1, 1] <= 0
                or det2(wt_cov) < GMIX_LOW_DETVAL):
            return ngmix.flags.LOW_DET, v0, u0, 0.0, 0.0, None

        sums = self._accumulate(epochs, wt_cov, v0, u0)

        if sums[5] <= 0:
            return ngmix.flags.NONPOS_FLUX, v0, u0, 0.0, 0.0, None

        finv = 1.0 / sums[5]
        if self.fixcen:
            # the center is not updated and the moments are
            # measured about it directly
            dv = 0.0
            du = 0.0
            M1 = sums[2] * finv
            M2 = sums[3] * finv
            T = sums[4] * finv
        else:
            dv = sums[0] * finv
            du = sums[1] * finv
            v0 += dv
            u0 += du

            if (abs(v0 - vorig) > self.shiftmax
                    or abs(u0 - uorig) > self.shiftmax):
                return ngmix.flags.CEN_SHIFT, v0, u0, dv, du, None

            M1 = sums[2] * finv - (du * du - dv * dv)
            M2 = sums[3] * finv - 2 * dv * du
            T = sums[4] * finv - (dv * dv + du * du)

        if T <= 0:
            # the measured moment matrix is not positive definite
            # and cannot be deweighted into a valid next weight
            return ngmix.flags.NONPOS_SIZE, v0, u0, dv, du, None

        M = cov_from_mom(M1, M2, T)
        return 0, v0, u0, dv, du, M

    def _run_admom_gauss(self, epochs, nband, guess_gmix, Tsmooth):
        """
        the standard adaptive moments fixed point: deweight the
        measured moments into the next weight until the weight stops
        changing

        Parameters
        ----------
        epochs: list of dict
            The prep_epoch dicts, one per epoch
        nband: int
            The number of bands
        guess_gmix: ngmix.GMix
            The guess, providing the starting center, shape and T
        Tsmooth: float
            The T of the gaussian smoothing
        """
        cen_guess = guess_gmix.get_cen()
        v0 = cen_guess[0]
        u0 = cen_guess[1]
        e1g, e2g, Tg = guess_gmix.get_e1e2T()
        irr, irc, icc = e2mom(e1g, e2g, Tg + Tsmooth)
        wt_cov = np.array([[irr, irc], [irc, icc]])

        vorig = v0
        uorig = u0

        flags = 0
        numiter = 0

        for i in range(self.maxiter):
            numiter = i + 1

            flags, v0, u0, dv, du, M = self._measure_step(
                epochs, wt_cov, v0, u0, vorig, uorig,
            )
            if flags != 0:
                break

            new_wt_cov, flags = deweight(M, wt_cov)
            if flags != 0:
                break

            # converge on the weight fixed point residual, the same
            # test as the mixture loop.  The centroid correction
            # decouples the center from the moments, so the center
            # must be tested explicitly
            shift = new_wt_cov - wt_cov
            scale = new_wt_cov[0, 0] + new_wt_cov[1, 1]
            converged = (
                np.abs(shift).max() < self.etol * scale
                and abs(dv) < self.cen_tol
                and abs(du) < self.cen_tol
            )
            wt_cov = new_wt_cov

            if converged:
                break
        else:
            flags = ngmix.flags.MAXITER

        return self._get_result(
            epochs=epochs, nband=nband, flags=flags, numiter=numiter,
            wt_cov=wt_cov, v0=v0, u0=u0, Tsmooth=Tsmooth,
        )

    def _run_admom_mixture(self, epochs, nband, guess_gmix, Tsmooth):
        """
        fit a fixed-ratio gaussian mixture family ('exp' or 'dev')
        by moment matching.
        The weight follows the standard deweight iteration on the
        measured moments, and the family covariance is shifted by the
        difference of the deweight-mapped measured and predicted
        moments, which has near unit gain; for a single-gaussian
        family this reduces exactly to the standard deweight update.
        The smoothing covariance cancels in the difference.

        The family state is the covariance matrix fam_cov, which is not
        constrained to be positive definite: the model only enters
        the sums through the smoothed component covariances, so the
        family size can scatter through zero and the ellipticity is
        not clipped.  A proposed update is accepted when every
        smoothed component gives a positive definite total covariance
        (mixture_model_valid); otherwise the step is damped, and the fit
        is flagged if no valid step is found.  The etol convergence
        criterion applies to the raw family covariance shift relative
        to the weight T.

        The linear convergence of the coupled weight/family iteration
        is accelerated with guarded Steffensen extrapolation: pairs of
        plain steps estimate the contraction ratio and the remaining
        geometric series is applied in a single boosted step.  This is
        the vector form of Aitken's delta-squared process iterated as
        in Steffensen's method [1] [2]; the ratio estimate from the
        inner product of successive differences is the rank-one case
        of the vector extrapolation methods reviewed in [3].

        [1] A. C. Aitken (1926), "On Bernoulli's numerical solution
            of algebraic equations", Proc. Roy. Soc. Edinburgh 46,
            289
        [2] J. F. Steffensen (1933), "Remarks on iteration", Skand.
            Aktuarietidskr. 16, 64
        [3] A. Sidi (2017), "Vector Extrapolation Methods with
            Applications", SIAM

        Parameters
        ----------
        epochs: list of dict
            The prep_epoch dicts, one per epoch
        nband: int
            The number of bands
        guess_gmix: ngmix.GMix
            The guess, providing the starting center, shape and T
        Tsmooth: float
            The T of the gaussian smoothing
        """
        cen_guess = guess_gmix.get_cen()
        v0 = cen_guess[0]
        u0 = cen_guess[1]
        e1g, e2g, Tg = guess_gmix.get_e1e2T()
        irr, irc, icc = e2mom(e1g, e2g, Tg + Tsmooth)
        wt_cov = np.array([[irr, irc], [irc, icc]])
        fam_cov = cov_from_e(e1g, e2g, Tg)

        vorig = v0
        uorig = u0

        flags = 0
        numiter = 0

        # last accepted plain (unboosted, undamped) shift, for the
        # extrapolation ratio; None whenever a fresh pair is needed
        prev_shift = None

        for i in range(self.maxiter):
            numiter = i + 1

            flags, v0, u0, dv, du, M = self._measure_step(
                epochs, wt_cov, v0, u0, vorig, uorig,
            )
            if flags != 0:
                break

            new_wt_cov, flags = deweight(M, wt_cov)
            if flags != 0:
                break

            raw_shift, pflags = self._model_shift(
                fam_cov, M, wt_cov, new_wt_cov, Tsmooth,
            )
            shift, boosted = self._steffensen_boost(
                raw_shift, prev_shift, pflags,
            )
            prop, idamp, accepted = self._damped_step(
                fam_cov, shift, new_wt_cov, Tsmooth,
            )
            if not accepted:
                flags = ngmix.flags.LOW_DET
                break

            # the ratio estimate needs a fresh pair of plain steps
            # after a boost, a damped step, or a fallback update
            if idamp == 0 and not boosted and pflags == 0:
                prev_shift = raw_shift
            else:
                prev_shift = None

            scale = new_wt_cov[0, 0] + new_wt_cov[1, 1]
            # convergence requires an undamped step and is tested on
            # the raw fixed-point residual, not the boosted step: a
            # damped step can be small only because it was shortened
            # at the validity boundary, not because the fit has
            # settled
            converged = (
                idamp == 0
                and np.abs(raw_shift).max() < self.etol * scale
                and abs(dv) < self.cen_tol
                and abs(du) < self.cen_tol
            )
            fam_cov = prop
            wt_cov = new_wt_cov

            if converged:
                break
        else:
            flags = ngmix.flags.MAXITER

        model_state = {
            'type': self.model, 'cov': fam_cov, 'F': np.ones(nband),
        }
        return self._get_result(
            epochs=epochs, nband=nband, flags=flags, numiter=numiter,
            wt_cov=wt_cov, v0=v0, u0=u0, Tsmooth=Tsmooth,
            model_state=model_state,
        )

    def _model_shift(self, fam_cov, M, wt_cov, new_wt_cov, Tsmooth):
        """
        the raw fixed point shift of the family covariance: the
        deweight-mapped difference of the measured and model
        predicted moments.  Returns (shift, pflags); when the
        prediction cannot be deweighted (pflags nonzero) the shift
        is the gain-1 fallback on the weighted moment ratios

        Parameters
        ----------
        fam_cov: (2, 2) array
            The family covariance
        M: (2, 2) array
            The measured weighted moment matrix
        wt_cov: (2, 2) array
            The weight covariance under which M was measured
        new_wt_cov: (2, 2) array
            The deweight of M
        Tsmooth: float
            The T of the gaussian smoothing
        """
        # the model predicted moment ratios; with a common center
        # and common structure across epochs these are the same
        # for every epoch, so one closed-form evaluation suffices
        unit_state = self._unit_state(fam_cov)
        psums = model_ksums(
            unit_state, 0, 0.0, 0.0, wt_cov, 1.0, Tsmooth,
        )
        pinv = 1.0 / psums[5]
        Mp1 = psums[2] * pinv
        Mp2 = psums[3] * pinv
        Tp = psums[4] * pinv
        Mpred = cov_from_mom(Mp1, Mp2, Tp)
        # Sp: the family covariance implied by the predicted
        # moments (the deweight image of Mpred)
        Sp, pflags = deweight(Mpred, wt_cov)

        if pflags == 0:
            # shift the family covariance by the deweight-mapped
            # difference of the measured and predicted moments
            shift = new_wt_cov - Sp
        else:
            # gain-1 fallback on the weighted moment ratios,
            # composed in matrix form: scale by the T ratio and
            # shift the anisotropy by the ratio differences
            M1, M2, Tm = mom_from_cov(M)
            Tf = fam_cov[0, 0] + fam_cov[1, 1]
            fac = Tm / Tp
            de1 = M1 / Tm - Mp1 / Tp
            de2 = M2 / Tm - Mp2 / Tp
            shift = (fac - 1) * fam_cov + 0.5 * fac * Tf * np.array([
                [-de1, de2],
                [de2, de1],
            ])
        return shift, pflags

    def _steffensen_boost(self, raw_shift, prev_shift, pflags):
        """
        Steffensen-style extrapolation: the iteration converges
        linearly with a steady ratio, so two successive plain steps
        give the ratio and the remaining geometric series can be
        summed in one boosted step.  Guarded to the primary update
        branch and a stable ratio range.  Returns (shift, boosted)

        Parameters
        ----------
        raw_shift: (2, 2) array
            The plain shift of this iteration, from _model_shift
        prev_shift: (2, 2) array or None
            The plain shift of the previous iteration; None when no
            plain step is available to form the ratio
        pflags: int
            The flags from _model_shift; nonzero on the fallback branch
        """
        if pflags == 0 and prev_shift is not None:
            denom = np.sum(prev_shift * prev_shift)
            if denom > 0:
                rho = np.sum(raw_shift * prev_shift) / denom
                # the upper limit must admit the slowly
                # contracting T mode of the dev family at large
                # size (rho approaches 1); overshoots from a
                # noisy ratio estimate are caught by the
                # validity damping
                if 0.2 < rho < 0.99:
                    return raw_shift / (1 - rho), True
        return raw_shift, False

    def _damped_step(self, fam_cov, shift, new_wt_cov, Tsmooth):
        """
        accept the largest step, damping if needed, for which the
        smoothed model components stay positive definite.  Returns
        (prop, idamp, accepted)

        Parameters
        ----------
        fam_cov: (2, 2) array
            The family covariance
        shift: (2, 2) array
            The proposed shift of the family covariance
        new_wt_cov: (2, 2) array
            The updated weight covariance, under which validity is
            checked
        Tsmooth: float
            The T of the gaussian smoothing
        """
        accepted = False
        for idamp in range(10):
            prop = fam_cov + shift
            if mixture_model_valid(
                    self.model, prop, new_wt_cov, Tsmooth):
                accepted = True
                break
            shift = 0.5 * shift
        return prop, idamp, accepted

    def _unit_state(self, fam_cov):
        """
        a unit-flux model state dict at the given family covariance

        Parameters
        ----------
        fam_cov: (2, 2) array
            The family covariance
        """
        return {'type': self.model, 'cov': fam_cov, 'F': np.ones(1)}

    def _run_admom_star(self, epochs, nband, guess_gmix, Tsmooth):
        """
        star model: a pre-psf delta function.  In the smoothed plane
        both the object and the matched weight are the smoothing
        gaussian; the weight is frozen and only the center is
        iterated

        Parameters
        ----------
        epochs: list of dict
            The prep_epoch dicts, one per epoch
        nband: int
            The number of bands
        guess_gmix: ngmix.GMix
            The guess, providing the starting center, shape and T
        Tsmooth: float
            The T of the gaussian smoothing
        """
        cen_guess = guess_gmix.get_cen()
        v0 = cen_guess[0]
        u0 = cen_guess[1]
        vorig = v0
        uorig = u0
        wt_cov = np.diag([Tsmooth / 2, Tsmooth / 2])

        flags = 0
        numiter = 0
        for i in range(self.maxiter):
            numiter = i + 1

            sums = self._accumulate(epochs, wt_cov, v0, u0)

            if sums[5] <= 0:
                flags = ngmix.flags.NONPOS_FLUX
                break

            if self.fixcen:
                break

            finv = 1.0 / sums[5]
            dv = sums[0] * finv
            du = sums[1] * finv
            v0 += dv
            u0 += du

            if (abs(v0 - vorig) > self.shiftmax
                    or abs(u0 - uorig) > self.shiftmax):
                flags = ngmix.flags.CEN_SHIFT
                break

            if abs(dv) < self.cen_tol and abs(du) < self.cen_tol:
                break
        else:
            flags = ngmix.flags.MAXITER

        model_state = {
            'type': 'star',
            'cov_sm': np.diag([Tsmooth / 2, Tsmooth / 2]),
            'F': np.ones(nband),
        }
        return self._get_result(
            epochs=epochs, nband=nband, flags=flags, numiter=numiter,
            wt_cov=wt_cov, v0=v0, u0=u0, Tsmooth=Tsmooth,
            model_state=model_state,
        )

    def _get_result(
        self, epochs, nband, flags, numiter, wt_cov, v0, u0, Tsmooth,
        model_state=None,
    ):
        """
        package the result, measuring the per-band fluxes and the
        errors with the converged weight.  model_state, sent for the
        non-gauss models, holds the converged family parameters and
        sets the flux normalization

        Parameters
        ----------
        epochs: list of dict
            The prep_epoch dicts, one per epoch
        nband: int
            The number of bands
        flags: int
            The flags from the iteration
        numiter: int
            The number of iterations taken
        wt_cov: (2, 2) array
            The converged weight covariance
        v0, u0: float
            The converged center
        Tsmooth: float
            The T of the gaussian smoothing
        model_state: dict, optional
            The converged model dict (see the models.py module
            docstring); not sent for the gauss path
        """

        res = {
            'flags': flags,
            'numiter': numiter,
            'nband': nband,
            'model': self.model,
            'gauss_e1': np.nan,
            'gauss_e2': np.nan,
            'gauss_e1err': np.nan,
            'gauss_e2err': np.nan,
            'gauss_T': np.nan,
            'gauss_T_err': np.nan,
            'gauss_e_flags': 0,
            'cen': np.array([v0, u0]),
            'pars': np.zeros(6) + np.nan,
            'sums': np.zeros(6) + np.nan,
            'sums_cov': np.zeros((6, 6)) + np.nan,
            'e1': np.nan,
            'e2': np.nan,
            'e1err': np.nan,
            'e2err': np.nan,
            'e': np.array([np.nan, np.nan]),
            'e_err': np.array([np.nan, np.nan]),
            'e_cov': np.diag([np.nan, np.nan]),
            'e_flags': 0,
            'T': np.nan,
            'T_err': np.nan,
            'T_flags': 0,
            'flux': np.zeros(nband) + np.nan,
            'flux_err': np.zeros(nband) + np.nan,
            'flux_flags': 0,
            's2n': np.nan,
            'wt_irr': wt_cov[0, 0],
            'wt_irc': wt_cov[0, 1],
            'wt_icc': wt_cov[1, 1],
        }

        if flags == 0:
            (sums, sums_cov, fluxes, flux_vars, fam_err_cov,
             flux_cov) = self._finalize(
                epochs, nband, wt_cov, v0, u0, Tsmooth,
                model_state=model_state,
            )
            res['sums'] = sums
            res['sums_cov'] = sums_cov

            M1, M2, Tgal, shape_ok = self._shape_state(
                model_state, wt_cov, Tsmooth,
            )
            res['T'] = Tgal

            res['flux'] = fluxes
            if flux_cov is not None:
                # the full cross-band flux covariance from the
                # shared family response; what color errors need
                res['flux_cov'] = flux_cov
            if np.all(flux_vars > 0):
                res['flux_err'] = np.sqrt(flux_vars)
                # covariance-aware total s/n where the cross-band
                # covariance exists; the quadrature sum is the
                # fallback (and is exact on the diagonal paths,
                # e.g. stars)
                s2n = None
                if flux_cov is not None:
                    s2n = joint_flux_s2n(fluxes, flux_cov)
                if s2n is None:
                    s2n = np.sqrt(
                        np.sum((res['flux'] / res['flux_err']) ** 2)
                    )
                res['s2n'] = s2n
            else:
                res['flux_flags'] |= ngmix.flags.NONPOS_VAR

            self._set_T_err(res, model_state, fam_err_cov, sums, sums_cov)

            res['pars'] = np.array([
                v0, u0, M1, M2, Tgal, np.sum(fluxes),
            ])

            self._set_shape(
                res, model_state, shape_ok, M1, M2, Tgal, wt_cov,
                fam_err_cov, sums, sums_cov,
            )

            self._set_gauss_entries(
                res, wt_cov, Tsmooth, sums, sums_cov,
            )

        if (
            self.full_errors and flags == 0
            and (
                (model_state is None and self.model == 'gauss')
                or (
                    model_state is not None
                    and model_state['type'] in ('exp', 'dev')
                )
            )
        ):
            # the full (fixed-point) errors: differentiate the
            # actual update at the actual data, staying calibrated
            # under model mismatch where the model_sandwich errors
            # under-predict (T by ~17 percent and flux by ~11
            # percent for dev truth fit with exp).  On a guarded
            # branch the sandwich errors are kept
            from .full_errors import padmom_full_covariance

            fe = padmom_full_covariance(
                self, epochs, nband, model_state, wt_cov,
                v0, u0, Tsmooth,
            )
            if fe is not None:
                res['flux_err'] = fe['flux_err']
                res['flux_cov'] = fe['flux_cov']
                res['s2n'] = fe['s2n']
                fam_err_cov = fe['fam_err_cov']
                if fam_err_cov[2, 2] > 0:
                    res['T_err'] = np.sqrt(fam_err_cov[2, 2])
                Tgal = res['T']
                if (
                    np.isfinite(res['e1']) and Tgal > 0
                    and fam_err_cov[2, 2] > 0
                ):
                    ev1, ev2, e12 = _e_cov_from_mom_cov(
                        res['e1'], res['e2'], Tgal, fam_err_cov,
                    )
                    if ev1 > 0 and ev2 > 0:
                        res['e1err'] = np.sqrt(ev1)
                        res['e2err'] = np.sqrt(ev2)
                        res['e_err'] = np.array([
                            res['e1err'], res['e2err'],
                        ])
                        res['e_cov'] = np.array([
                            [ev1, e12], [e12, ev2],
                        ])

                # the gauss (converged weight) estimator entries,
                # from the weight block of the same covariance
                gauss_err_cov = fe['gauss_err_cov']
                gauss_T = res['gauss_T']
                if gauss_err_cov[2, 2] > 0:
                    res['gauss_T_err'] = np.sqrt(gauss_err_cov[2, 2])
                if (
                    np.isfinite(res['gauss_e1']) and gauss_T > 0
                    and gauss_err_cov[2, 2] > 0
                ):
                    ev1, ev2, _ = _e_cov_from_mom_cov(
                        res['gauss_e1'], res['gauss_e2'], gauss_T,
                        gauss_err_cov,
                    )
                    if ev1 > 0 and ev2 > 0:
                        res['gauss_e1err'] = np.sqrt(ev1)
                        res['gauss_e2err'] = np.sqrt(ev2)

        # propagate fitting failures, but not the post-hoc NONPOS_SIZE
        # or NONPOS_SHAPE_VAR set above, for which T and flux are still
        # usable; e_flags == 0 iff the ellipticities and their errors
        # are usable
        res['T_flags'] |= flags
        res['flux_flags'] |= flags
        res['e_flags'] |= flags

        res['flagstr'] = ngmix.flags.get_flags_str(res['flags'])
        res['T_flagstr'] = ngmix.flags.get_flags_str(res['T_flags'])
        res['flux_flagstr'] = ngmix.flags.get_flags_str(res['flux_flags'])
        res['e_flagstr'] = ngmix.flags.get_flags_str(res['e_flags'])

        return res

    def _shape_state(self, model_state, wt_cov, Tsmooth):
        """
        the galaxy moments (M1, M2, Tgal) and whether the shape M/Tgal
        is defined, per model type.  det > 0 with positive trace is
        |e| < 1: a non positive definite galaxy covariance has no
        defined shape

        Parameters
        ----------
        model_state: dict or None
            The converged model dict (see the models.py module docstring);
            None for the plain gauss path
        wt_cov: (2, 2) array
            The weight covariance
        Tsmooth: float
            The T of the gaussian smoothing
        """
        if model_state is None:
            # gauss: the converged weight with the smoothing
            # subtracted.  The round smoothing shifts only the
            # diagonal, so it drops out of M1 and M2 and only T
            # needs the subtraction
            Sgal = wt_cov - np.diag([Tsmooth / 2, Tsmooth / 2])
            M1, M2, Twt = mom_from_cov(wt_cov)
            Tgal = Twt - Tsmooth
            shape_ok = Tgal > 0 and det2(Sgal) > 0
        elif model_state['type'] in ('exp', 'dev'):
            # the family covariance can scatter out of positive
            # definite, where the shape is undefined
            fam_cov = model_state['cov']
            M1, M2, Tgal = mom_from_cov(fam_cov)
            shape_ok = Tgal > 0 and det2(fam_cov) > 0
        else:
            # star: a delta function has no size or shape
            M1 = 0.0
            M2 = 0.0
            Tgal = 0.0
            shape_ok = False
        return M1, M2, Tgal, shape_ok

    def _set_T_err(self, res, model_state, fam_err_cov, sums, sums_cov):
        """
        set the T error in the result, or NONPOS_VAR in T_flags

        Parameters
        ----------
        res: dict
            The result dict, modified in place
        model_state: dict or None
            The converged model dict (see the models.py module docstring);
            None for the plain gauss path
        fam_err_cov: (3, 3) array or None
            The covariance of the family (M1, M2, T) from model_sandwich;
            None for the gauss and star paths, or when the sandwich could
            not be evaluated
        sums: array of size 6
            The joint [v, u, M1, M2, T, flux] sums at the converged weight
        sums_cov: (6, 6) array
            The covariance of the sums
        """
        if model_state is not None and model_state['type'] == 'star':
            # T is fixed at zero by the model, not measured
            pass
        elif model_state is not None:
            # exp: from the family covariance sandwich
            if fam_err_cov is not None and fam_err_cov[2, 2] > 0:
                res['T_err'] = np.sqrt(fam_err_cov[2, 2])
            else:
                res['T_flags'] |= ngmix.flags.NONPOS_VAR
        elif sums_cov[4, 4] > 0 and sums_cov[5, 5] > 0:
            # the sums include the weight, so need a factor of two
            # to correct, as in real-space adaptive moments
            res['T_err'] = 4 * get_ratio_error(
                sums[4], sums[5],
                sums_cov[4, 4], sums_cov[5, 5], sums_cov[4, 5],
            )
        else:
            res['T_flags'] |= ngmix.flags.NONPOS_VAR

    def _set_gauss_entries(self, res, wt_cov, Tsmooth, sums, sums_cov):
        """
        the gauss (converged-weight) shape estimator entries,
        available for free with every model: the weight iteration
        is model independent, so the smoothing-subtracted weight
        is the standard adaptive moments shape regardless of the
        family being fit.  The errors are the weighted moment
        ratio errors, as in the gauss model fit.  For the star
        model the weight is frozen at the smoothing and the
        entries stay flagged

        Parameters
        ----------
        res: dict
            The result dict, modified in place
        wt_cov: (2, 2) array
            The weight covariance
        Tsmooth: float
            The T of the gaussian smoothing
        sums: array of size 6
            The joint [v, u, M1, M2, T, flux] sums at the converged weight
        sums_cov: (6, 6) array
            The covariance of the sums
        """
        Sgal = wt_cov - np.diag([Tsmooth / 2, Tsmooth / 2])
        Twt = wt_cov[0, 0] + wt_cov[1, 1]
        Tgal = Twt - Tsmooth
        res['gauss_T'] = Tgal

        if sums_cov[4, 4] > 0 and sums_cov[5, 5] > 0:
            res['gauss_T_err'] = 4 * get_ratio_error(
                sums[4], sums[5],
                sums_cov[4, 4], sums_cov[5, 5], sums_cov[4, 5],
            )

        if not (Tgal > 0 and det2(Sgal) > 0):
            res['gauss_e_flags'] |= ngmix.flags.NONPOS_SIZE
            return

        res['gauss_e1'] = (Sgal[1, 1] - Sgal[0, 0]) / Tgal
        res['gauss_e2'] = 2 * Sgal[0, 1] / Tgal

        if sums_cov[2, 2] > 0 and sums_cov[3, 3] > 0:
            lever = Twt / Tgal
            res['gauss_e1err'] = lever * 2 * get_ratio_error(
                sums[2], sums[4],
                sums_cov[2, 2], sums_cov[4, 4], sums_cov[2, 4],
            )
            res['gauss_e2err'] = lever * 2 * get_ratio_error(
                sums[3], sums[4],
                sums_cov[3, 3], sums_cov[4, 4], sums_cov[3, 4],
            )
        else:
            res['gauss_e_flags'] |= ngmix.flags.NONPOS_SHAPE_VAR

    def _set_shape(
        self, res, model_state, shape_ok, M1, M2, Tgal, wt_cov,
        fam_err_cov, sums, sums_cov,
    ):
        """
        set the ellipticities and their errors in the result, or the
        flag bits recording why they are unusable

        Parameters
        ----------
        res: dict
            The result dict, modified in place
        model_state: dict or None
            The converged model dict (see the models.py module docstring);
            None for the plain gauss path
        shape_ok: bool
            Whether the shape is defined, from _shape_state
        M1, M2, Tgal: float
            The galaxy moments, from _shape_state
        wt_cov: (2, 2) array
            The weight covariance
        fam_err_cov: (3, 3) array or None
            The covariance of the family (M1, M2, T) from model_sandwich;
            None for the gauss and star paths, or when the sandwich could
            not be evaluated
        sums: array of size 6
            The joint [v, u, M1, M2, T, flux] sums at the converged weight
        sums_cov: (6, 6) array
            The covariance of the sums
        """
        if model_state is not None and model_state['type'] == 'star':
            # a delta function has no shape; this is by
            # construction, not a failure, so the overall flags
            # stay clean, but e_flags still marks the
            # ellipticities unusable
            res['e_flags'] |= ngmix.flags.NONPOS_SIZE
        elif shape_ok:
            res['e1'] = M1 / Tgal
            res['e2'] = M2 / Tgal
            res['e'] = np.array([res['e1'], res['e2']])

            e1err = np.nan
            e2err = np.nan
            e12cov = np.nan
            if model_state is not None and fam_err_cov is None:
                # the family sandwich failed; the errors stay nan
                # and the flags are set below
                pass
            elif model_state is not None:
                # exp: linearize e = M/T over the family
                # covariance sandwich
                ev1 = (
                    fam_err_cov[0, 0]
                    - 2 * res['e1'] * fam_err_cov[0, 2]
                    + res['e1'] ** 2 * fam_err_cov[2, 2]
                ) / Tgal ** 2
                ev2 = (
                    fam_err_cov[1, 1]
                    - 2 * res['e2'] * fam_err_cov[1, 2]
                    + res['e2'] ** 2 * fam_err_cov[2, 2]
                ) / Tgal ** 2
                if ev1 > 0 and ev2 > 0:
                    e1err = np.sqrt(ev1)
                    e2err = np.sqrt(ev2)
                    e12cov = (
                        fam_err_cov[0, 1]
                        - res['e1'] * fam_err_cov[1, 2]
                        - res['e2'] * fam_err_cov[0, 2]
                        + res['e1'] * res['e2'] * fam_err_cov[2, 2]
                    ) / Tgal ** 2
            elif sums_cov[2, 2] > 0 and sums_cov[3, 3] > 0:
                # the lever arm Twt/Tgal accounts for the smoothing
                # subtraction in the denominator of e = M1/Tgal
                Twt = wt_cov[0, 0] + wt_cov[1, 1]
                lever = Twt / Tgal
                e1err = lever * 2 * get_ratio_error(
                    sums[2], sums[4],
                    sums_cov[2, 2], sums_cov[4, 4], sums_cov[2, 4],
                )
                e2err = lever * 2 * get_ratio_error(
                    sums[3], sums[4],
                    sums_cov[3, 3], sums_cov[4, 4], sums_cov[3, 4],
                )
                # the covariance of the two moment ratios with the
                # common denominator, consistent with get_ratio_var
                r1 = sums[2] / sums[4]
                r2 = sums[3] / sums[4]
                e12cov = (lever * 2) ** 2 * (
                    sums_cov[2, 3] - r1 * sums_cov[3, 4]
                    - r2 * sums_cov[2, 4] + r1 * r2 * sums_cov[4, 4]
                ) / sums[4] ** 2

            if (np.isfinite(e1err) and np.isfinite(e2err)
                    and np.isfinite(e12cov)):
                res['e1err'] = e1err
                res['e2err'] = e2err
                res['e_err'] = np.array([e1err, e2err])
                res['e_cov'] = np.array([
                    [e1err ** 2, e12cov],
                    [e12cov, e2err ** 2],
                ])
            else:
                res['flags'] |= ngmix.flags.NONPOS_SHAPE_VAR
                res['e_flags'] |= ngmix.flags.NONPOS_SHAPE_VAR
        else:
            # a converged fit can have non-positive size after the
            # smoothing is subtracted; the fluxes and T are still
            # usable but the shape is undefined
            res['flags'] |= ngmix.flags.NONPOS_SIZE
            res['e_flags'] |= ngmix.flags.NONPOS_SIZE

    def _accumulate(self, epochs, wt_cov, v0, u0):
        """
        accumulate the k-space moment sums over epochs

        returns sums [v, u, M1, M2, T, flux], weighted over epochs and
        normalized to sky units

        Parameters
        ----------
        epochs: list of dict
            The prep_epoch dicts, one per epoch
        wt_cov: (2, 2) array
            The weight covariance
        v0, u0: float
            The center, as sky offsets from the jacobian center
        """
        sums = np.zeros(6)
        esums = np.zeros(6)
        for epoch in epochs:
            alpha, beta = get_phase_angles(epoch, v0, u0)
            admom_ksums(
                epoch['kim'], epoch['iy'], epoch['ix'], epoch['dim'],
                alpha, beta, epoch['kv'], epoch['ku'],
                wt_cov[0, 0], wt_cov[0, 1], wt_cov[1, 1], epoch['df2'],
                esums,
            )
            sums += epoch['weight'] * epoch['detAtinv'] * esums
        return sums

    def _finalize(self, epochs, nband, wt_cov, v0, u0, Tsmooth,
                  model_state=None):
        """
        get the final accumulated sums, their covariance, and the
        per-band fluxes and variances, using the converged weight.
        With a model_state the fluxes are normalized by the unit flux
        model predictions instead of the gaussian fixed point factor;
        for the gauss model at the fixed point the two agree.

        The covariance treats each Fourier mode as independent with
        variance given by the noise power carried in the per-epoch
        err_fac2, either white at the level set by the weight map or
        measured per mode from the attached noise image.  The flux
        variances include the response of the adaptive weight to the
        noise; see flux_var_delta.

        The flux normalization: the raw flux sum corresponds to a
        weighted flux with an effective real-space kernel of peak value
        1/(2 pi sqrt(det(wt_cov)) detAtinv).  We normalize to a unit peak
        kernel, and the factor of 2 converts the gaussian-weighted flux
        to the total flux at the fixed point of the iteration.  The
        smoothing preserves flux, so this also holds when smoothing.

        Parameters
        ----------
        epochs: list of dict
            The prep_epoch dicts, one per epoch
        nband: int
            The number of bands
        wt_cov: (2, 2) array
            The converged weight covariance
        v0, u0: float
            The converged center
        Tsmooth: float
            The T of the gaussian smoothing
        model_state: dict, optional
            The converged model dict (see the models.py module
            docstring); not sent for the gauss path

        Returns
        -------
        sums: array of size 6
            The joint [v, u, M1, M2, T, flux] sums
        cov: (6, 6) array
            The covariance of the sums
        fluxes: array of size nband
            The per-band fluxes
        flux_vars: array of size nband
            The flux variances
        fam_err_cov: (3, 3) array or None
            The covariance of the family (M1, M2, T); None for the gauss
            and star paths or when the sandwich could not be evaluated
        flux_cov: (nband, nband) array or None
            The cross-band flux covariance; None when the sandwich could
            not be evaluated
        """
        knrm = 2 * np.pi * np.sqrt(det2(wt_cov))

        sums = np.zeros(6)
        cov = np.zeros((6, 6))
        esums = np.zeros(6)
        ecov = np.zeros((6, 6))

        fsums = np.zeros(nband)
        fvars = np.zeros(nband)
        # cov of the joint (M1, M2, T) sums with the band flux sums,
        # for the weight response term in the flux errors
        fmcovs = np.zeros((nband, 3))
        wsums = np.zeros(nband)

        for epoch in epochs:
            alpha, beta = get_phase_angles(epoch, v0, u0)
            admom_finalize(
                epoch['kim'], epoch['iy'], epoch['ix'], epoch['dim'],
                alpha, beta, epoch['kv'], epoch['ku'],
                wt_cov[0, 0], wt_cov[0, 1], wt_cov[1, 1], epoch['df2'],
                epoch['err_fac2'],
                esums, ecov,
            )
            fac = epoch['weight'] * epoch['detAtinv']
            nfac = epoch['df2'] ** 2

            sums += fac * esums
            cov += fac ** 2 * nfac * ecov

            band = epoch['band']
            fsums[band] += fac * esums[5]
            fvars[band] += fac ** 2 * nfac * ecov[5, 5]
            fmcovs[band] += fac ** 2 * nfac * ecov[2:5, 5]
            wsums[band] += epoch['weight']

        fam_err_cov = None
        if model_state is None:
            fluxes = 2 * knrm * fsums / wsums
            # the analytic delta method, with the cross-band
            # covariance from the shared weight response assembled
            # in closed form; the diagonal is the flux_var_delta
            # variances unchanged
            fcov_raw = flux_cov_delta(
                wt_cov, sums, cov, fsums, fvars, fmcovs,
            )
            uband = 2 * knrm / wsums
            flux_vars = uband ** 2 * np.diag(fcov_raw)
            flux_cov = fcov_raw * np.outer(uband, uband)
        else:
            upred = np.zeros(nband)
            for epoch in epochs:
                fac = epoch['weight'] * epoch['detAtinv']
                upred[epoch['band']] += fac * model_ksums(
                    model_state, epoch['band'], 0.0, 0.0, wt_cov,
                    epoch['detAtinv'], Tsmooth,
                )[5]
            fluxes = fsums / upred
            fcov_raw = None
            if model_state['type'] == 'star':
                # the weight is frozen for stars, so the fixed weight
                # variance is exact and the bands are independent
                rawvars = fvars
                fcov_raw = np.diag(fvars)
            else:
                rawvars, fam_err_cov, fcov_raw = model_sandwich(
                    model_state['type'], model_state['cov'], wt_cov,
                    Tsmooth, sums, cov, fsums, fvars, fmcovs,
                )
                if rawvars is None:
                    # the sandwich could not be evaluated; fall
                    # back to the fixed weight variances, with the
                    # structure errors flagged downstream
                    rawvars = fvars
                    fam_err_cov = None
                    fcov_raw = None
            flux_vars = rawvars / upred ** 2
            if fcov_raw is not None:
                uinv = 1.0 / upred
                flux_cov = fcov_raw * np.outer(uinv, uinv)
            else:
                flux_cov = None
        return sums, cov, fluxes, flux_vars, fam_err_cov, flux_cov

    def _get_guess(self, mb_obs, guess, Tsmooth):
        """
        get the guess as a gaussian mixture

        Parameters
        ----------
        mb_obs: MultiBandObsList
            The observations, used for the pixel scale
        guess: ngmix.GMix, float or None
            A gmix is returned as sent.  A float is a guess for the
            pre-PSF T, with the center and shape drawn randomly.
            For None the guess is round and centered, with T the
            larger of Tsmooth and that of a two pixel fwhm
        Tsmooth: float
            The T of the gaussian smoothing

        Returns
        -------
        guess_gmix: ngmix.GMix
        """
        if isinstance(guess, GMix):
            return guess

        if guess is None:
            scale = mb_obs[0][0].jacobian.get_scale()
            Tguess = max(Tsmooth, fwhm_to_T(2 * scale))
            pars = [0.0, 0.0, 0.0, 0.0, Tguess, 1.0]
            return GMixModel(pars, 'gauss')

        Tguess = guess
        rng = self._get_rng()
        scale = mb_obs[0][0].jacobian.get_scale()
        pars = np.zeros(6)
        pars[0:2] = rng.uniform(low=-0.5 * scale, high=0.5 * scale, size=2)
        pars[2:4] = rng.uniform(low=-0.3, high=0.3, size=2)
        pars[4] = Tguess * (1.0 + rng.uniform(low=-0.1, high=0.1))
        pars[5] = 1.0
        return GMixModel(pars, 'gauss')

    def _get_noise_autocovs(self, nband):
        """
        the noise autocovariance of each band, or None when not
        sent: one array is used for every band
        """
        acov = self.noise_autocov
        if acov is None:
            return None
        if isinstance(acov, np.ndarray) and acov.ndim == 2:
            return [acov] * nband
        if len(acov) != nband:
            raise ValueError(
                f'got {len(acov)} noise_autocov arrays for '
                f'{nband} bands'
            )
        return list(acov)

    def _get_rng(self):
        """
        get the random state, creating one if none was sent
        """
        if self.rng is None:
            self.rng = np.random.RandomState()
        return self.rng


def _e_cov_from_mom_cov(e1, e2, T, mom_cov):
    """
    the covariance of the ellipticities e = (M1, M2) / T, linearized
    over the covariance of the moments

    With de_i = (dM_i - e_i dT) / T,

        Var(e_i) = (C_ii - 2 e_i C_iT + e_i^2 C_TT) / T^2
        Cov(e_1, e_2) = (C_12 - e_1 C_2T - e_2 C_1T
                         + e_1 e_2 C_TT) / T^2

    Parameters
    ----------
    e1, e2: float
        The ellipticities
    T: float
        The size
    mom_cov: (3, 3) array
        The covariance C of (M1, M2, T)

    Returns
    -------
    var_e1, var_e2, cov_e1e2: float
        The variances are not checked for positivity
    """
    Tinv2 = 1.0 / T ** 2
    var_e1 = (
        mom_cov[0, 0]
        - 2 * e1 * mom_cov[0, 2]
        + e1 ** 2 * mom_cov[2, 2]
    ) * Tinv2
    var_e2 = (
        mom_cov[1, 1]
        - 2 * e2 * mom_cov[1, 2]
        + e2 ** 2 * mom_cov[2, 2]
    ) * Tinv2
    cov_e1e2 = (
        mom_cov[0, 1]
        - e1 * mom_cov[1, 2]
        - e2 * mom_cov[0, 2]
        + e1 * e2 * mom_cov[2, 2]
    ) * Tinv2
    return var_e1, var_e2, cov_e1e2


def _check_model(model):
    """
    check the model name, returning it.  Unknown names raise

    Parameters
    ----------
    model: str
        The model name
    """
    if model not in ('gauss', 'exp', 'dev', 'star'):
        raise ValueError(
            f"bad model {model!r}, expected 'gauss', 'exp', 'dev' "
            "or 'star'"
        )
    return model


def get_phase_angles(epoch, v0, u0):
    """
    get the centering phase angles per unit row/col frequency

    The base phase (drow, dcol) centers the fft at the image jacobian
    center; the (v0, u0) sky offsets are converted to pixel offsets
    with the jacobian.  The phase for a mode is then
    exp(i (f1d[iy] alpha + f1d[ix] beta)) which is applied inside the
    numba kernels using two 1d phasor arrays.

    Parameters
    ----------
    epoch: dict
        A prep_epoch dict
    v0, u0: float
        The center, as sky offsets from the jacobian center

    Returns
    -------
    alpha, beta: float
        The phase angles for the row and col frequencies, in pixel
        units
    """
    A = epoch['Atinv']
    alpha = epoch['drow'] + A[0, 0] * v0 + A[1, 0] * u0
    beta = epoch['dcol'] + A[0, 1] * v0 + A[1, 1] * u0
    return alpha, beta


def deweight(M, wt_cov):
    """
    deweight the measured moments, returning the new weight covariance

    wt_cov_new = (M^{-1} - wt_cov^{-1})^{-1}

    Parameters
    ----------
    M: (2, 2) array
        The measured weighted moment matrix
    wt_cov: (2, 2) array
        The weight covariance under which M was measured

    Returns
    -------
    new_wt_cov: (2, 2) array
        The deweighted covariance; the input wt_cov when flagged
    flags: int
        Zero on success, LOW_DET when either input or the result is
        not positive definite
    """
    detm = det2(M)
    if detm <= GMIX_LOW_DETVAL:
        return wt_cov, ngmix.flags.LOW_DET

    detw = det2(wt_cov)
    if detw <= GMIX_LOW_DETVAL:
        return wt_cov, ngmix.flags.LOW_DET

    idetm = 1.0 / detm
    idetw = 1.0 / detw

    # inverse of a 2x2 [[a, b], [b, c]] is [[c, -b], [-b, a]]/det
    Nvv = M[1, 1] * idetm - wt_cov[1, 1] * idetw
    Nuu = M[0, 0] * idetm - wt_cov[0, 0] * idetw
    Nvu = -M[0, 1] * idetm + wt_cov[0, 1] * idetw

    # a positive determinant is not sufficient for a 2x2 symmetric
    # matrix: both eigenvalues negative also gives det > 0.  That
    # happens when the measured moments exceed the weight in both
    # eigendirections (e.g. heavy neighbor contamination), and the
    # inverse would be a negative definite weight
    detn = Nvv * Nuu - Nvu * Nvu
    if detn <= GMIX_LOW_DETVAL or Nvv <= 0 or Nuu <= 0:
        return wt_cov, ngmix.flags.LOW_DET

    idetn = 1.0 / detn
    new_wt_cov = np.array([
        [Nuu * idetn, -Nvu * idetn],
        [-Nvu * idetn, Nvv * idetn],
    ])
    return new_wt_cov, 0
