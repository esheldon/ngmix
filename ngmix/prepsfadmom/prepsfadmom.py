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
from ..moments import fwhm_to_T
from ..shape import e1e2_to_g1g2
from ..util import get_ratio_error
from .prepsfadmom_nb import admom_ksums, admom_finalize
from .models import det2, deweight
from .model import get_padmom_model
from .prep import (
    choose_fwhm_smooth, prep_epoch, get_phase_angles,
    DEFAULT_SMOOTH_FAC,
)
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
    model: str, dict or PAdmomModel, optional
        The object model: a model object such as GaussModel(),
        ExpModel(), DevModel(), BDFModel(TdByTe=1.0) or StarModel()
        (see ngmix.prepsfadmom.model for the model descriptions and
        their parameters), or a string naming one: 'gauss'
        (default), 'exp', 'dev' or 'star'.  The equivalent dict
        form {'type': name} is also accepted, and is required for
        'bdf' if not using the object, to carry the 'TdByTe' entry
        (the dev to exp size ratio) and the optional 'fracdev0' and
        'fracdev_sigma0' shrinkage entries, e.g.
        model={'type': 'bdf', 'TdByTe': 1.0}.  See get_padmom_model
        for the accepted specifications.
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
    model: str, dict or PAdmomModel, optional
        The object model: a model object such as GaussModel(),
        ExpModel(), DevModel(), BDFModel(TdByTe=1.0) or StarModel()
        (see ngmix.prepsfadmom.model for the model descriptions and
        their parameters), a string naming one ('gauss' is the
        default), or the equivalent dict spec; see
        get_padmom_model.
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

        self.model = get_padmom_model(model)
        self.fixcen = fixcen
        self.full_errors = full_errors
        if full_errors:
            if not self.model.supports_full_errors:
                raise ValueError(
                    'full_errors supports the gauss, exp and '
                    f'dev models, got {self.model.name!r}'
                )
            if ap_rad != 0:
                raise ValueError(
                    'full_errors requires ap_rad=0: the '
                    'influence-kernel transfer assumes no '
                    'apodization'
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

        self.model.validate_fit(Tsmooth)

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
        over all epochs; the driver lives on the model object
        """
        return self.model.run_admom(
            self, epochs, nband, guess_gmix, Tsmooth,
        )

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
            'model': self.model.name,
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

            M1, M2, Tgal, shape_ok = self.model.shape_state(
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
                res['s2n'] = np.sqrt(
                    np.sum((res['flux'] / res['flux_err']) ** 2)
                )
            else:
                res['flux_flags'] |= ngmix.flags.NONPOS_VAR

            self.model.set_T_err(
                res, model_state, fam_cov, sums, sums_cov,
            )

            res['pars'] = np.array([
                v0, u0, M1, M2, Tgal, np.sum(fluxes),
            ])

            self.model.set_shape(
                res, model_state, shape_ok, M1, M2, Tgal, Sigma,
                fam_cov, sums, sums_cov,
            )

            self._set_gauss_entries(
                res, Sigma, Tsmooth, sums, sums_cov,
            )

        # full_errors is only constructible for the gauss, exp and
        # dev models (see __init__), so no model gating is needed
        if self.full_errors and flags == 0:
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
        The flux normalization and errors are the model's
        (flux_results): the model paths normalize by the unit flux
        model predictions, the gauss path by the gaussian fixed
        point factor; at the fixed point the two agree for gauss.

        The covariance treats each Fourier mode as independent with
        variance given by the noise power carried in the per-epoch
        err_fac2, either white at the level set by the weight map or
        measured per mode from the attached noise image.  The flux
        variances include the response of the adaptive weight to the
        noise; see flux_var_delta.
        """
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

        fluxes, flux_vars, fam_cov, flux_cov = self.model.flux_results(
            state=model_state, epochs=epochs, nband=nband,
            Sigma=Sigma, Tsmooth=Tsmooth, sums=sums, cov=cov,
            fsums=fsums, fvars=fvars, fmcovs=fmcovs, wsums=wsums,
        )
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
