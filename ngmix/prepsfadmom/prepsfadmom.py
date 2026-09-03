"""
Adaptive moments in Fourier space, measuring pre-PSF moments.

The usual adaptive moments iteration is performed, but the weighted
real-space moment sums are evaluated in k-space against the
PSF-deconvolved FFT of the image, using derivatives of the elliptical
gaussian weight

    W(k) = exp(-k^T Sigma k / 2)

where Sigma is the real-space covariance matrix of the weight.  Because
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
from numpy import fft
from .models import mom_from_cov, det2
from .errors import (
    flux_cov_delta,
    joint_flux_s2n,
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
    model: str or dict, optional
        The object model: 'gauss' (default), 'exp', 'dev', 'star'
        or 'bdf'.  A dict form is also accepted: {'type': name},
        with the 'bdf' model requiring the additional 'TdByTe'
        entry (the dev to exp size ratio), e.g.
        model={'type': 'bdf', 'TdByTe': 1.0}.  With 'gauss' the
        fit is the standard adaptive moments fixed point.  With
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
        With 'bdf' the object is the composite exp plus dev model:
        shared center and ellipticity, the dev size TdByTe times
        the exp size, with the per band flux split between the
        components (fracdev) fit by a per-sweep two-template GLS
        solve on the retained modes, interleaved with the family
        adaptive step.  The result gains fracdev, fracdev_gls,
        fracdev_gls_err, flux_exp, flux_dev and flux_gls_cov
        entries; the flux entries are the band-summed GLS
        component fluxes (the split is per band, so bulge and disk
        colors differ), and the flux and structure errors are
        conditional on the converged split.  The optional
        'fracdev0' and 'fracdev_sigma0' entries (sent together)
        regularize the split that builds the composite model: the
        inverse-variance blend of the measured split with the
        prior, using the conditional GLS split variance, which is
        deterministic given the structure.  Only the model split
        (the reported fracdev) is regularized; the reported
        component fluxes and fracdev_gls stay the raw linear
        solutions.  fracdev_sigma0=0 freezes the model split at
        fracdev0.
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


class PAdmomFitter:
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
        rng=None,
    ):

        self.model, self.TdByTe, self.fracdev_shrink = (
            _parse_model_spec(model)
        )
        self.fixcen = fixcen
        self.full_errors = full_errors
        if full_errors and self.model not in ('gauss', 'exp', 'dev'):
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
                    # per-pixel variances
                    ep['obs_weight'] = tobs.weight
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
            if 'flux_exp' in result:
                result['flux_exp'] = result['flux_exp'][0]
                result['flux_dev'] = result['flux_dev'][0]
                result['flux_gls_cov'] = result['flux_gls_cov'][0]

        return PAdmomResult(obs=obs, result=result)

    def _run_admom(self, epochs, nband, guess_gmix, Tsmooth):
        """
        the adaptive moments iteration, accumulating the k-space sums
        over all epochs
        """
        if self.model in ('exp', 'dev'):
            raise RuntimeError("exp and dev models are not currently supported for pre-psf admom!")
            # return self._run_admom_mixture(
            #     epochs, nband, guess_gmix, Tsmooth,
            # )
        elif self.model == 'bdf':
            raise RuntimeError("bdf models are not currently supported for pre-psf admom!")
            # return self._run_admom_bdf(
            #     epochs, nband, guess_gmix, Tsmooth,
            # )
        elif self.model == 'star':
            raise RuntimeError("star models are not currently supported for pre-psf admom!")
            # return self._run_admom_star(epochs, nband, guess_gmix, Tsmooth)
        else:
            return self._run_admom_gauss(epochs, nband, guess_gmix, Tsmooth)

    def _measure_step(self, epochs, Sigma, v0, u0, vorig, uorig):
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
        """
        if (Sigma[0, 0] <= 0 or Sigma[1, 1] <= 0
                or det2(Sigma) < GMIX_LOW_DETVAL):
            return ngmix.flags.LOW_DET, v0, u0, 0.0, 0.0, None

        sums = self._accumulate(epochs, Sigma, v0, u0)

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

        M = np.array([
            [0.5 * (T - M1), 0.5 * M2],
            [0.5 * M2, 0.5 * (T + M1)],
        ])
        return 0, v0, u0, dv, du, M

    def _run_admom_gauss(self, epochs, nband, guess_gmix, Tsmooth):
        """
        the standard adaptive moments fixed point: deweight the
        measured moments into the next weight until the weight stops
        changing
        """
        cen_guess = guess_gmix.get_cen()
        v0 = cen_guess[0]
        u0 = cen_guess[1]
        e1g, e2g, Tg = guess_gmix.get_e1e2T()
        irr, irc, icc = e2mom(e1g, e2g, Tg + Tsmooth)
        Sigma = np.array([[irr, irc], [irc, icc]])

        vorig = v0
        uorig = u0

        flags = 0
        numiter = 0

        for i in range(self.maxiter):
            numiter = i + 1

            flags, v0, u0, dv, du, M = self._measure_step(
                epochs, Sigma, v0, u0, vorig, uorig,
            )
            if flags != 0:
                break

            newSigma, flags = deweight(M, Sigma)
            if flags != 0:
                break

            # converge on the weight fixed point residual, the same
            # test as the mixture loop.  The centroid correction
            # decouples the center from the moments, so the center
            # must be tested explicitly
            shift = newSigma - Sigma
            scale = newSigma[0, 0] + newSigma[1, 1]
            converged = (
                np.abs(shift).max() < self.etol * scale
                and abs(dv) < self.cen_tol
                and abs(du) < self.cen_tol
            )
            Sigma = newSigma

            if converged:
                break
        else:
            flags = ngmix.flags.MAXITER

        return self._get_result(
            epochs=epochs, nband=nband, flags=flags, numiter=numiter,
            Sigma=Sigma, v0=v0, u0=u0, Tsmooth=Tsmooth,
        )

    def _get_result(
        self, epochs, nband, flags, numiter, Sigma, v0, u0, Tsmooth,
        model_state=None,
    ):
        """
        package the result, measuring the per-band fluxes and the
        errors with the converged weight.  model_state, sent for the
        non-gauss models, holds the converged family parameters and
        sets the flux normalization
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
            'wt_irr': Sigma[0, 0],
            'wt_irc': Sigma[0, 1],
            'wt_icc': Sigma[1, 1],
        }

        if flags == 0:
            (sums, sums_cov, fluxes, flux_vars, fam_cov,
             flux_cov) = self._finalize(
                epochs, nband, Sigma, v0, u0, Tsmooth,
                model_state=model_state,
            )
            res['sums'] = sums
            res['sums_cov'] = sums_cov

            M1, M2, Tgal, shape_ok = self._shape_state(
                model_state, Sigma, Tsmooth,
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

            self._set_T_err(res, model_state, fam_cov, sums, sums_cov)

            res['pars'] = np.array([
                v0, u0, M1, M2, Tgal, np.sum(fluxes),
            ])

            self._set_shape(
                res, model_state, shape_ok, M1, M2, Tgal, Sigma,
                fam_cov, sums, sums_cov,
            )

            self._set_gauss_entries(
                res, Sigma, Tsmooth, sums, sums_cov,
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
                self, epochs, nband, model_state, Sigma,
                v0, u0, Tsmooth,
            )
            if fe is not None:
                res['flux_err'] = fe['flux_err']
                res['flux_cov'] = fe['flux_cov']
                res['s2n'] = fe['s2n']
                fam_cov = fe['fam_cov']
                if fam_cov[2, 2] > 0:
                    res['T_err'] = np.sqrt(fam_cov[2, 2])
                Tgal = res['T']
                if (
                    np.isfinite(res['e1']) and Tgal > 0
                    and fam_cov[2, 2] > 0
                ):
                    ev1 = (
                        fam_cov[0, 0]
                        - 2 * res['e1'] * fam_cov[0, 2]
                        + res['e1'] ** 2 * fam_cov[2, 2]
                    ) / Tgal ** 2
                    ev2 = (
                        fam_cov[1, 1]
                        - 2 * res['e2'] * fam_cov[1, 2]
                        + res['e2'] ** 2 * fam_cov[2, 2]
                    ) / Tgal ** 2
                    if ev1 > 0 and ev2 > 0:
                        e12 = (
                            fam_cov[0, 1]
                            - res['e1'] * fam_cov[1, 2]
                            - res['e2'] * fam_cov[0, 2]
                            + res['e1'] * res['e2']
                            * fam_cov[2, 2]
                        ) / Tgal ** 2
                        res['e1err'] = np.sqrt(ev1)
                        res['e2err'] = np.sqrt(ev2)
                        res['e_cov'] = np.array([
                            [ev1, e12], [e12, ev2],
                        ])

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

    def _shape_state(self, model_state, Sigma, Tsmooth):
        """
        the galaxy moments (M1, M2, Tgal) and whether the shape M/Tgal
        is defined, per model type.  det > 0 with positive trace is
        |e| < 1: a non positive definite galaxy covariance has no
        defined shape
        """
        if model_state is None:
            # gauss: the converged weight with the smoothing
            # subtracted.  The round smoothing shifts only the
            # diagonal, so it drops out of M1 and M2 and only T
            # needs the subtraction
            Sgal = Sigma - np.diag([Tsmooth / 2, Tsmooth / 2])
            M1, M2, Twt = mom_from_cov(Sigma)
            Tgal = Twt - Tsmooth
            shape_ok = Tgal > 0 and det2(Sgal) > 0
        # TODO: Turn back on for other models
        # elif model_state['type'] in ('exp', 'dev', 'bdf'):
        #     # the family covariance can scatter out of positive
        #     # definite, where the shape is undefined
        #     Sfam = model_state['cov']
        #     M1, M2, Tgal = mom_from_cov(Sfam)
        #     shape_ok = Tgal > 0 and det2(Sfam) > 0
        # else:
        #     # star: a delta function has no size or shape
        #     M1 = 0.0
        #     M2 = 0.0
        #     Tgal = 0.0
        #     shape_ok = False
        return M1, M2, Tgal, shape_ok

    def _set_T_err(self, res, model_state, fam_cov, sums, sums_cov):
        """
        set the T error in the result, or NONPOS_VAR in T_flags
        """
        # TODO: Turn back on for other models
        # if model_state is not None and model_state['type'] == 'star':
        #     # T is fixed at zero by the model, not measured
        #     pass
        # elif model_state is not None:
        #     # exp: from the family covariance sandwich
        #     if fam_cov is not None and fam_cov[2, 2] > 0:
        #         res['T_err'] = np.sqrt(fam_cov[2, 2])
        #     else:
        #         res['T_flags'] |= ngmix.flags.NONPOS_VAR
        # elif sums_cov[4, 4] > 0 and sums_cov[5, 5] > 0:
        # remove this line...
        if sums_cov[4, 4] > 0 and sums_cov[5, 5] > 0:
            # the sums include the weight, so need a factor of two
            # to correct, as in real-space adaptive moments
            res['T_err'] = 4 * get_ratio_error(
                sums[4], sums[5],
                sums_cov[4, 4], sums_cov[5, 5], sums_cov[4, 5],
            )
        else:
            res['T_flags'] |= ngmix.flags.NONPOS_VAR

    def _set_gauss_entries(self, res, Sigma, Tsmooth, sums, sums_cov):
        """
        the gauss (converged-weight) shape estimator entries,
        available for free with every model: the weight iteration
        is model independent, so the smoothing-subtracted weight
        is the standard adaptive moments shape regardless of the
        family being fit.  The errors are the weighted moment
        ratio errors, as in the gauss model fit.  For the star
        model the weight is frozen at the smoothing and the
        entries stay flagged
        """
        Sgal = Sigma - np.diag([Tsmooth / 2, Tsmooth / 2])
        Twt = Sigma[0, 0] + Sigma[1, 1]
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
        self, res, model_state, shape_ok, M1, M2, Tgal, Sigma,
        fam_cov, sums, sums_cov,
    ):
        """
        set the ellipticities and their errors in the result, or the
        flag bits recording why they are unusable
        """
        # TODO: Turn back on for other models
        # if model_state is not None and model_state['type'] == 'star':
        #     # a delta function has no shape; this is by
        #     # construction, not a failure, so the overall flags
        #     # stay clean, but e_flags still marks the
        #     # ellipticities unusable
        #     res['e_flags'] |= ngmix.flags.NONPOS_SIZE
        # elif shape_ok:
        if shape_ok:
            res['e1'] = M1 / Tgal
            res['e2'] = M2 / Tgal
            res['e'] = np.array([res['e1'], res['e2']])

            e1err = np.nan
            e2err = np.nan
            e12cov = np.nan
            # TODO: Turn back on for other models
            # if model_state is not None and fam_cov is None:
            #     # the family sandwich failed; the errors stay nan
            #     # and the flags are set below
            #     pass
            # elif model_state is not None:
            #     # exp: linearize e = M/T over the family
            #     # covariance sandwich
            #     ev1 = (
            #         fam_cov[0, 0]
            #         - 2 * res['e1'] * fam_cov[0, 2]
            #         + res['e1'] ** 2 * fam_cov[2, 2]
            #     ) / Tgal ** 2
            #     ev2 = (
            #         fam_cov[1, 1]
            #         - 2 * res['e2'] * fam_cov[1, 2]
            #         + res['e2'] ** 2 * fam_cov[2, 2]
            #     ) / Tgal ** 2
            #     if ev1 > 0 and ev2 > 0:
            #         e1err = np.sqrt(ev1)
            #         e2err = np.sqrt(ev2)
            #         e12cov = (
            #             fam_cov[0, 1]
            #             - res['e1'] * fam_cov[1, 2]
            #             - res['e2'] * fam_cov[0, 2]
            #             + res['e1'] * res['e2'] * fam_cov[2, 2]
            #         ) / Tgal ** 2
            # elif sums_cov[2, 2] > 0 and sums_cov[3, 3] > 0:
            if sums_cov[2, 2] > 0 and sums_cov[3, 3] > 0:
                # the lever arm Twt/Tgal accounts for the smoothing
                # subtraction in the denominator of e = M1/Tgal
                Twt = Sigma[0, 0] + Sigma[1, 1]
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

    def _accumulate(self, epochs, Sigma, v0, u0):
        """
        accumulate the k-space moment sums over epochs

        returns sums [v, u, M1, M2, T, flux], weighted over epochs and
        normalized to sky units
        """
        sums = np.zeros(6)
        esums = np.zeros(6)
        for epoch in epochs:
            alpha, beta = get_phase_angles(epoch, v0, u0)
            admom_ksums(
                epoch['kim'], epoch['iy'], epoch['ix'], epoch['dim'],
                alpha, beta, epoch['kv'], epoch['ku'],
                Sigma[0, 0], Sigma[0, 1], Sigma[1, 1], epoch['df2'],
                esums,
            )
            sums += epoch['weight'] * epoch['detAtinv'] * esums
        return sums

    def _finalize(self, epochs, nband, Sigma, v0, u0, Tsmooth,
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
        1/(2 pi sqrt(det(Sigma)) detAtinv).  We normalize to a unit peak
        kernel, and the factor of 2 converts the gaussian-weighted flux
        to the total flux at the fixed point of the iteration.  The
        smoothing preserves flux, so this also holds when smoothing.
        """
        knrm = 2 * np.pi * np.sqrt(det2(Sigma))

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
                Sigma[0, 0], Sigma[0, 1], Sigma[1, 1], epoch['df2'],
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

        fam_cov = None
        if model_state is None:
            fluxes = 2 * knrm * fsums / wsums
            # the analytic delta method, with the cross-band
            # covariance from the shared weight response assembled
            # in closed form; the diagonal is the flux_var_delta
            # variances unchanged
            fcov_raw = flux_cov_delta(
                Sigma, sums, cov, fsums, fvars, fmcovs,
            )
            uband = 2 * knrm / wsums
            flux_vars = uband ** 2 * np.diag(fcov_raw)
            flux_cov = fcov_raw * np.outer(uband, uband)
        # TODO: Turn back on for other models
        # else:
        #     upred = np.zeros(nband)
        #     for epoch in epochs:
        #         fac = epoch['weight'] * epoch['detAtinv']
        #         upred[epoch['band']] += fac * model_ksums(
        #             model_state, epoch['band'], 0.0, 0.0, Sigma,
        #             epoch['detAtinv'], Tsmooth,
        #         )[5]
        #     fluxes = fsums / upred
        #     fcov_raw = None
        #     if model_state['type'] == 'star':
        #         # the weight is frozen for stars, so the fixed weight
        #         # variance is exact and the bands are independent
        #         rawvars = fvars
        #         fcov_raw = np.diag(fvars)
        #     else:
        #         if model_state['type'] == 'bdf':
        #             spec = model_state
        #         else:
        #             spec = model_state['type']
        #         rawvars = None
        #         fam_cov = None
        #         if (model_state['type'] == 'bdf'
        #                 and getattr(self, '_bdf_err', None)
        #                 is not None):
        #             G, k, fdv, eta_scov, eta_fcovs = self._bdf_err
        #             rawvars, fam_cov, fd_var_tot = (
        #                 bdf_joint_sandwich(
        #                     model_state, Sigma, Tsmooth, sums,
        #                     cov, fsums, fvars, fmcovs,
        #                     split_grad=G, shrink_k=k,
        #                     fd_var_data=fdv,
        #                     eta_scov=eta_scov,
        #                     eta_fcovs=eta_fcovs,
        #                 )
        #             )
        #             self._bdf_fd_var_total = fd_var_tot
        #         if rawvars is None:
        #             rawvars, fam_cov, fcov_raw = model_sandwich(
        #                 spec, model_state['cov'], Sigma,
        #                 Tsmooth, sums, cov, fsums, fvars, fmcovs,
        #             )
        #         if rawvars is None:
        #             # the sandwich could not be evaluated; fall
        #             # back to the fixed weight variances, with the
        #             # structure errors flagged downstream
        #             rawvars = fvars
        #             fam_cov = None
        #             fcov_raw = None
        #     flux_vars = rawvars / upred ** 2
        #     if fcov_raw is not None:
        #         uinv = 1.0 / upred
        #         flux_cov = fcov_raw * np.outer(uinv, uinv)
        #     else:
        #         flux_cov = None
        return sums, cov, fluxes, flux_vars, fam_cov, flux_cov

    def _get_guess(self, mb_obs, guess, Tsmooth):
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

    def _get_rng(self):
        if self.rng is None:
            self.rng = np.random.RandomState()
        return self.rng


def _parse_model_spec(model):
    """
    normalize a model specification to (type, TdByTe, shrink).
    A string names the type; the dict form is {'type': name} plus,
    for 'bdf' only, the required 'TdByTe' entry (the dev to exp
    size ratio) and the optional shrinkage pair 'fracdev0' and
    'fracdev_sigma0' (see run_prepsf_admom).  shrink is
    (fracdev0, sigma0) or None.  Unknown types and unexpected
    entries raise
    """
    if isinstance(model, str):
        model = {'type': model}
    else:
        model = dict(model)

    if 'type' not in model:
        raise ValueError("model dict must have a 'type' entry")
    mtype = model['type']
    if mtype not in ('gauss', 'exp', 'dev', 'star', 'bdf'):
        raise ValueError(
            f"bad model '{mtype}', expected 'gauss', 'exp', 'dev', "
            "'star' or 'bdf'"
        )

    shrink = None
    if mtype == 'bdf':
        if 'TdByTe' not in model:
            raise ValueError(
                "the bdf model requires a 'TdByTe' entry, e.g. "
                "model={'type': 'bdf', 'TdByTe': 1.0}"
            )
        TdByTe = float(model['TdByTe'])
        if TdByTe <= 0:
            raise ValueError(f'TdByTe must be positive, got {TdByTe}')
        allowed = {'type', 'TdByTe', 'fracdev0', 'fracdev_sigma0'}

        has0 = 'fracdev0' in model
        hass = 'fracdev_sigma0' in model
        if has0 != hass:
            raise ValueError(
                "the bdf shrinkage requires both 'fracdev0' and "
                "'fracdev_sigma0' (or neither)"
            )
        if has0:
            fracdev0 = float(model['fracdev0'])
            sigma0 = float(model['fracdev_sigma0'])
            if sigma0 < 0:
                raise ValueError(
                    f'fracdev_sigma0 must be non-negative, got '
                    f'{sigma0}'
                )
            shrink = (fracdev0, sigma0)
    else:
        TdByTe = None
        allowed = {'type'}

    extra = set(model) - allowed
    if extra:
        raise ValueError(
            f"unexpected model entries {sorted(extra)} for "
            f"'{mtype}'"
        )
    return mtype, TdByTe, shrink


def get_phase_angles(epoch, v0, u0):
    """
    get the centering phase angles per unit row/col frequency

    The base phase (drow, dcol) centers the fft at the image jacobian
    center; the (v0, u0) sky offsets are converted to pixel offsets
    with the jacobian.  The phase for a mode is then
    exp(i (f1d[iy] alpha + f1d[ix] beta)) which is applied inside the
    numba kernels using two 1d phasor arrays.
    """
    A = epoch['Atinv']
    alpha = epoch['drow'] + A[0, 0] * v0 + A[1, 0] * u0
    beta = epoch['dcol'] + A[0, 1] * v0 + A[1, 1] * u0
    return alpha, beta


def deweight(M, Sigma):
    """
    deweight the measured moments, returning the new weight covariance

    Sigma_new = (M^{-1} - Sigma^{-1})^{-1}
    """
    detm = det2(M)
    if detm <= GMIX_LOW_DETVAL:
        return Sigma, ngmix.flags.LOW_DET

    detw = det2(Sigma)
    if detw <= GMIX_LOW_DETVAL:
        return Sigma, ngmix.flags.LOW_DET

    idetm = 1.0 / detm
    idetw = 1.0 / detw

    # inverse of a 2x2 [[a, b], [b, c]] is [[c, -b], [-b, a]]/det
    Nvv = M[1, 1] * idetm - Sigma[1, 1] * idetw
    Nuu = M[0, 0] * idetm - Sigma[0, 0] * idetw
    Nvu = -M[0, 1] * idetm + Sigma[0, 1] * idetw

    # a positive determinant is not sufficient for a 2x2 symmetric
    # matrix: both eigenvalues negative also gives det > 0.  That
    # happens when the measured moments exceed the weight in both
    # eigendirections (e.g. heavy neighbor contamination), and the
    # inverse would be a negative definite weight
    detn = Nvv * Nuu - Nvu * Nvu
    if detn <= GMIX_LOW_DETVAL or Nvv <= 0 or Nuu <= 0:
        return Sigma, ngmix.flags.LOW_DET

    idetn = 1.0 / detn
    newSigma = np.array([
        [Nuu * idetn, -Nvu * idetn],
        [-Nvu * idetn, Nvv * idetn],
    ])
    return newSigma, 0
