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
from .models import model_ksums, cov_from_e, exp_model_valid, det2
from .errors import flux_var_delta, model_sandwich
from .prep import choose_fwhm_smooth, prep_epoch, DEFAULT_SMOOTH_FAC
import ngmix.flags

logger = logging.getLogger(__name__)

DEFAULT_MAXITER = 200
DEFAULT_SHIFTMAX = 5.0  # pixels for scale=1
DEFAULT_ETOL = 1.0e-5
DEFAULT_TTOL = 1.0e-3
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
    Ttol=DEFAULT_TTOL,
    cen_tol=DEFAULT_CENTOL,
    no_psf=False,
    use_noise_image=False,
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
        The object model: 'gauss' (default), 'exp' or 'star'.  With
        'gauss' the fit is the standard adaptive moments fixed point.
        With 'exp' the ngmix 6-gaussian exponential expansion is fit
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
        absolute tolerance in e1 or e2 to determine convergence,
        default 1.0e-5
    Ttol: float, optional
        relative tolerance in T to determine convergence, default 1.0e-3
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
        Ttol=Ttol,
        cen_tol=cen_tol,
        use_noise_image=use_noise_image,
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
        The flux s/n, combined over bands in quadrature.
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
        The object model: 'gauss' (default), 'exp' or 'star'.  With
        'gauss' the fit is the standard adaptive moments fixed point.
        With 'exp' the ngmix 6-gaussian exponential expansion is fit
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
        absolute tolerance in e1 or e2 to determine convergence,
        default 1.0e-5
    Ttol: float, optional
        relative tolerance in T to determine convergence, default 1.0e-3
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
        Ttol=DEFAULT_TTOL,
        cen_tol=DEFAULT_CENTOL,
        use_noise_image=False,
        rng=None,
    ):

        if model not in ('gauss', 'exp', 'star'):
            raise ValueError(
                f"bad model '{model}', expected 'gauss', 'exp' "
                "or 'star'"
            )
        self.model = model
        self.fwhm_smooth = fwhm_smooth
        self.smooth_fac = smooth_fac
        self.pad_factor = pad_factor
        self.ap_rad = ap_rad
        self.maxiter = maxiter
        self.shiftmax = shiftmax
        self.etol = etol
        self.Ttol = Ttol
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
                epochs.append(prep_epoch(
                    tobs, band=band, fwhm_smooth=fwhm_smooth,
                    pad_factor=self.pad_factor, ap_rad=self.ap_rad,
                    use_noise_image=self.use_noise_image,
                    no_psf=no_psf,
                ))

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
        """
        if self.model == 'exp':
            return self._run_admom_exp(epochs, nband, guess_gmix, Tsmooth)
        elif self.model == 'star':
            return self._run_admom_star(epochs, nband, guess_gmix, Tsmooth)

        cen_guess = guess_gmix.get_cen()
        v0 = cen_guess[0]
        u0 = cen_guess[1]
        e1g, e2g, Tg = guess_gmix.get_e1e2T()
        irr, irc, icc = e2mom(e1g, e2g, Tg + Tsmooth)
        Sigma = np.array([[irr, irc], [irc, icc]])

        vorig = v0
        uorig = u0

        flags = 0
        e1old = e2old = Told = np.nan
        numiter = 0

        for i in range(self.maxiter):
            numiter = i + 1

            if (Sigma[0, 0] <= 0 or Sigma[1, 1] <= 0
                    or det2(Sigma) < GMIX_LOW_DETVAL):
                flags = ngmix.flags.LOW_DET
                break

            sums = self._accumulate(epochs, Sigma, v0, u0)

            if sums[5] <= 0:
                flags = ngmix.flags.NONPOS_FLUX
                break

            # update the center, and refer the second moments to the
            # updated center with the exact centroid correction
            # <(x - d)_i (x - d)_j> = <x_i x_j> - d_i d_j.  The weight
            # center lags by one iteration, which vanishes at the fixed
            # point
            finv = 1.0 / sums[5]
            dv = sums[0] * finv
            du = sums[1] * finv
            v0 += dv
            u0 += du

            if (abs(v0 - vorig) > self.shiftmax
                    or abs(u0 - uorig) > self.shiftmax):
                flags = ngmix.flags.CEN_SHIFT
                break

            M1 = sums[2] * finv - (du * du - dv * dv)
            M2 = sums[3] * finv - 2 * dv * du
            T = sums[4] * finv - (dv * dv + du * du)

            if T <= 0:
                flags = ngmix.flags.NONPOS_SIZE
                break

            e1 = M1 / T
            e2 = M2 / T

            # the centroid correction decouples the center from the
            # moments, so the center must be tested explicitly
            if (abs(e1 - e1old) < self.etol
                    and abs(e2 - e2old) < self.etol
                    and abs(T / Told - 1) < self.Ttol
                    and abs(dv) < self.cen_tol
                    and abs(du) < self.cen_tol):
                break

            M = np.array([
                [0.5 * (T - M1), 0.5 * M2],
                [0.5 * M2, 0.5 * (T + M1)],
            ])
            Sigma, flags = deweight(M, Sigma)
            if flags != 0:
                break

            e1old = e1
            e2old = e2
            Told = T
        else:
            flags = ngmix.flags.MAXITER

        return self._get_result(
            epochs=epochs, nband=nband, flags=flags, numiter=numiter,
            Sigma=Sigma, v0=v0, u0=u0, Tsmooth=Tsmooth,
        )

    def _run_admom_exp(self, epochs, nband, guess_gmix, Tsmooth):
        """
        fit the 6-gaussian exponential expansion by moment matching.
        The weight follows the standard deweight iteration on the
        measured moments, and the family covariance is shifted by the
        difference of the deweight-mapped measured and predicted
        moments, which has near unit gain; for a single-gaussian
        family this reduces exactly to the standard deweight update.
        The smoothing covariance cancels in the difference.

        The family state is the covariance matrix Sfam, which is not
        constrained to be positive definite: the model only enters
        the sums through the smoothed component covariances, so the
        family size can scatter through zero and the ellipticity is
        not clipped.  A proposed update is accepted when every
        smoothed component gives a positive definite total covariance
        (exp_model_valid); otherwise the step is damped, and the fit
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
        """
        cen_guess = guess_gmix.get_cen()
        v0 = cen_guess[0]
        u0 = cen_guess[1]
        e1g, e2g, Tg = guess_gmix.get_e1e2T()
        irr, irc, icc = e2mom(e1g, e2g, Tg + Tsmooth)
        Sigma = np.array([[irr, irc], [irc, icc]])
        Sfam = cov_from_e(e1g, e2g, Tg)

        vorig = v0
        uorig = u0

        flags = 0
        numiter = 0
        unit_state = {'type': 'exp', 'cov': Sfam, 'F': np.ones(1)}

        # last accepted plain (unboosted, undamped) shift, for the
        # extrapolation ratio; None whenever a fresh pair is needed
        prev_shift = None

        for i in range(self.maxiter):
            numiter = i + 1

            if (Sigma[0, 0] <= 0 or Sigma[1, 1] <= 0
                    or det2(Sigma) < GMIX_LOW_DETVAL):
                flags = ngmix.flags.LOW_DET
                break

            sums = self._accumulate(epochs, Sigma, v0, u0)

            if sums[5] <= 0:
                flags = ngmix.flags.NONPOS_FLUX
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

            M1 = sums[2] * finv - (du * du - dv * dv)
            M2 = sums[3] * finv - 2 * dv * du
            Tm = sums[4] * finv - (dv * dv + du * du)

            if Tm <= 0:
                flags = ngmix.flags.NONPOS_SIZE
                break

            M = np.array([
                [0.5 * (Tm - M1), 0.5 * M2],
                [0.5 * M2, 0.5 * (Tm + M1)],
            ])
            newSigma, dflags = deweight(M, Sigma)
            if dflags != 0:
                flags = dflags
                break

            # the model predicted moment ratios; with a common center
            # and common structure across epochs these are the same
            # for every epoch, so one closed-form evaluation suffices
            unit_state['cov'] = Sfam
            psums = model_ksums(
                unit_state, 0, 0.0, 0.0, Sigma, 1.0, Tsmooth,
            )
            pinv = 1.0 / psums[5]
            Mp1 = psums[2] * pinv
            Mp2 = psums[3] * pinv
            Tp = psums[4] * pinv
            Mpred = np.array([
                [0.5 * (Tp - Mp1), 0.5 * Mp2],
                [0.5 * Mp2, 0.5 * (Tp + Mp1)],
            ])
            Sp, pflags = deweight(Mpred, Sigma)

            if pflags == 0:
                # shift the family covariance by the deweight-mapped
                # difference of the measured and predicted moments
                shift = newSigma - Sp
            else:
                # gain-1 fallback on the weighted moment ratios,
                # composed in matrix form: scale by the T ratio and
                # shift the anisotropy by the ratio differences
                Tf = Sfam[0, 0] + Sfam[1, 1]
                fac = Tm / Tp
                de1 = M1 / Tm - Mp1 / Tp
                de2 = M2 / Tm - Mp2 / Tp
                shift = (fac - 1) * Sfam + 0.5 * fac * Tf * np.array([
                    [-de1, de2],
                    [de2, de1],
                ])

            # Steffensen-style extrapolation: the iteration converges
            # linearly with a steady ratio, so two successive plain
            # steps give the ratio and the remaining geometric series
            # can be summed in one boosted step.  Guarded to the
            # primary update branch and a stable ratio range
            raw_shift = shift
            boosted = False
            if pflags == 0 and prev_shift is not None:
                denom = np.sum(prev_shift * prev_shift)
                if denom > 0:
                    rho = np.sum(raw_shift * prev_shift) / denom
                    if 0.2 < rho < 0.9:
                        shift = raw_shift / (1 - rho)
                        boosted = True

            # accept the largest step, damping if needed, for which
            # the smoothed model components stay positive definite
            accepted = False
            for idamp in range(10):
                prop = Sfam + shift
                if exp_model_valid(prop, newSigma, Tsmooth):
                    accepted = True
                    break
                shift = 0.5 * shift
            if not accepted:
                flags = ngmix.flags.LOW_DET
                break

            # the ratio estimate needs a fresh pair of plain steps
            # after a boost, a damped step, or a fallback update
            if idamp == 0 and not boosted and pflags == 0:
                prev_shift = raw_shift
            else:
                prev_shift = None

            scale = newSigma[0, 0] + newSigma[1, 1]
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
            Sfam = prop
            Sigma = newSigma

            if converged:
                break
        else:
            flags = ngmix.flags.MAXITER

        model_state = {
            'type': 'exp', 'cov': Sfam, 'F': np.ones(nband),
        }
        return self._get_result(
            epochs=epochs, nband=nband, flags=flags, numiter=numiter,
            Sigma=Sigma, v0=v0, u0=u0, Tsmooth=Tsmooth,
            model_state=model_state,
        )

    def _run_admom_star(self, epochs, nband, guess_gmix, Tsmooth):
        """
        star model: a pre-psf delta function.  In the smoothed plane
        both the object and the matched weight are the smoothing
        gaussian; the weight is frozen and only the center is
        iterated
        """
        cen_guess = guess_gmix.get_cen()
        v0 = cen_guess[0]
        u0 = cen_guess[1]
        vorig = v0
        uorig = u0
        Sigma = np.diag([Tsmooth / 2, Tsmooth / 2])

        flags = 0
        numiter = 0
        for i in range(self.maxiter):
            numiter = i + 1

            sums = self._accumulate(epochs, Sigma, v0, u0)

            if sums[5] <= 0:
                flags = ngmix.flags.NONPOS_FLUX
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
            Sigma=Sigma, v0=v0, u0=u0, Tsmooth=Tsmooth,
            model_state=model_state,
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
            sums, sums_cov, fluxes, flux_vars, fam_cov = self._finalize(
                epochs, nband, Sigma, v0, u0, Tsmooth,
                model_state=model_state,
            )
            res['sums'] = sums
            res['sums_cov'] = sums_cov

            Twt = Sigma[0, 0] + Sigma[1, 1]
            if model_state is None:
                Tgal = Twt - Tsmooth
                M1 = Sigma[1, 1] - Sigma[0, 0]
                M2 = 2 * Sigma[0, 1]
                shape_ok = Tgal > 0
            elif model_state['type'] == 'exp':
                Sfam = model_state['cov']
                Tgal = Sfam[0, 0] + Sfam[1, 1]
                M1 = Sfam[1, 1] - Sfam[0, 0]
                M2 = 2 * Sfam[0, 1]
                # the family covariance can scatter out of positive
                # definite, where the shape is undefined
                shape_ok = Tgal > 0 and det2(Sfam) > 0
            else:
                # star: a delta function has no size or shape
                Tgal = 0.0
                M1 = 0.0
                M2 = 0.0
                shape_ok = False

            res['T'] = Tgal

            res['flux'] = fluxes
            if np.all(flux_vars > 0):
                res['flux_err'] = np.sqrt(flux_vars)
                res['s2n'] = np.sqrt(
                    np.sum((res['flux'] / res['flux_err']) ** 2)
                )
            else:
                res['flux_flags'] |= ngmix.flags.NONPOS_VAR

            if model_state is not None and model_state['type'] == 'star':
                # T is fixed at zero by the model, not measured
                pass
            elif model_state is not None:
                # exp: from the family covariance sandwich
                if fam_cov[2, 2] > 0:
                    res['T_err'] = np.sqrt(fam_cov[2, 2])
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

            res['pars'] = np.array([
                v0, u0, M1, M2, Tgal, np.sum(fluxes),
            ])

            if model_state is not None and model_state['type'] == 'star':
                # a delta function has no shape; this is by
                # construction, not a failure
                pass
            elif shape_ok:
                res['e1'] = M1 / Tgal
                res['e2'] = M2 / Tgal
                res['e'] = np.array([res['e1'], res['e2']])

                e1err = np.nan
                e2err = np.nan
                e12cov = np.nan
                if model_state is not None:
                    # exp: linearize e = M/T over the family
                    # covariance sandwich
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
                        e1err = np.sqrt(ev1)
                        e2err = np.sqrt(ev2)
                        e12cov = (
                            fam_cov[0, 1]
                            - res['e1'] * fam_cov[1, 2]
                            - res['e2'] * fam_cov[0, 2]
                            + res['e1'] * res['e2'] * fam_cov[2, 2]
                        ) / Tgal ** 2
                elif sums_cov[2, 2] > 0 and sums_cov[3, 3] > 0:
                    # the lever arm Twt/Tgal accounts for the smoothing
                    # subtraction in the denominator of e = M1/Tgal
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
            else:
                # a converged fit can have non-positive size after the
                # smoothing is subtracted; the fluxes and T are still
                # usable but the shape is undefined
                res['flags'] |= ngmix.flags.NONPOS_SIZE

        # propagate fitting failures, but not the post-hoc NONPOS_SIZE
        # or NONPOS_SHAPE_VAR set above, for which T and flux are still
        # usable
        res['T_flags'] |= flags
        res['flux_flags'] |= flags

        res['flagstr'] = ngmix.flags.get_flags_str(res['flags'])
        res['T_flagstr'] = ngmix.flags.get_flags_str(res['T_flags'])
        res['flux_flagstr'] = ngmix.flags.get_flags_str(res['flux_flags'])

        return res

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
            flux_vars = (2 * knrm / wsums) ** 2 * flux_var_delta(
                Sigma, sums, cov, fsums, fvars, fmcovs,
            )
        else:
            upred = np.zeros(nband)
            for epoch in epochs:
                fac = epoch['weight'] * epoch['detAtinv']
                upred[epoch['band']] += fac * model_ksums(
                    model_state, epoch['band'], 0.0, 0.0, Sigma,
                    epoch['detAtinv'], Tsmooth,
                )[5]
            fluxes = fsums / upred
            if model_state['type'] == 'star':
                # the weight is frozen for stars, so the fixed weight
                # variance is exact
                rawvars = fvars
            else:
                rawvars, fam_cov = model_sandwich(
                    'exp', model_state['cov'], Sigma, Tsmooth,
                    sums, cov, fsums, fvars, fmcovs,
                )
            flux_vars = rawvars / upred ** 2
        return sums, cov, fluxes, flux_vars, fam_cov

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

    detn = Nvv * Nuu - Nvu * Nvu
    if detn <= GMIX_LOW_DETVAL:
        return Sigma, ngmix.flags.LOW_DET

    idetn = 1.0 / detn
    newSigma = np.array([
        [Nuu * idetn, -Nvu * idetn],
        [-Nvu * idetn, Nvv * idetn],
    ])
    return newSigma, 0
