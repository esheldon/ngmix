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
__all__ = ['run_prepsf_admom', 'PrePSFAdmomFitter', 'PrePSFAdmomResult']

import logging
import functools

import numpy as np
import scipy.fft as fft

from ..observation import get_mb_obs, MultiBandObsList
from ..gmix import GMix, GMixModel
from ..gmix.gmix_nb import GMIX_LOW_DETVAL
from ..moments import fwhm_to_T, T_to_fwhm, e2mom
from ..shape import e1e2_to_g1g2
from ..util import get_ratio_error
from ..gexceptions import FFTRangeError
from ..fastexp_nb import FASTEXP_MAX_CHI2
from .. import prepsfmom
from ..prepsfmom import (
    _zero_pad_image,
    _build_square_apodization_mask,
    _deconvolve_im_psf_inplace,
    _check_obs_and_get_psf_obs,
    _pixel_fft,
)
from .prepsfadmom_nb import admom_ksums, admom_finalize
import ngmix.flags

logger = logging.getLogger(__name__)

DEFAULT_MAXITER = 200
DEFAULT_SHIFTMAX = 5.0  # pixels for scale=1
DEFAULT_ETOL = 1.0e-5
DEFAULT_TTOL = 1.0e-3
DEFAULT_CENTOL = 1.0e-4  # pixels for scale=1
DEFAULT_SMOOTH_FAC = 1.05


def run_prepsf_admom(
    obs, guess=None,
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
    rng: np.random.RandomState, optional
        Random state used to generate guesses from a T guess and for
        PSF fits when choosing the smoothing automatically.

    Returns
    -------
    PrePSFAdmomResult
    """
    fitter = PrePSFAdmomFitter(
        fwhm_smooth=fwhm_smooth,
        smooth_fac=smooth_fac,
        pad_factor=pad_factor,
        ap_rad=ap_rad,
        maxiter=maxiter,
        shiftmax=shiftmax,
        etol=etol,
        Ttol=Ttol,
        cen_tol=cen_tol,
        rng=rng,
    )
    return fitter.go(obs=obs, guess=guess, no_psf=no_psf)


class PrePSFAdmomResult(dict):
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
        get a gmix representing the best fit pre-PSF gaussian, normalized
        to unit flux
        """
        if self['flags'] != 0:
            raise RuntimeError('cannot create gmix, fit failed')

        pars = self['pars'].copy()
        pars[5] = 1.0

        e1 = pars[2] / pars[4]
        e2 = pars[3] / pars[4]

        g1, g2 = e1e2_to_g1g2(e1, e2)
        pars[2] = g1
        pars[3] = g2

        return GMixModel(pars, "gauss")


class PrePSFAdmomFitter(object):
    """
    Measure pre-PSF adaptive moments in Fourier space

    Parameters
    ----------
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
    rng: np.random.RandomState, optional
        Random state used to generate guesses from a T guess and for
        PSF fits when choosing the smoothing automatically.

    Notes
    -----
    The reported errors are derived assuming a fixed weight function, as
    in real-space adaptive moments, and are approximate.  In particular
    the flux errors underestimate the observed scatter because the flux
    couples to the fitted size; the level of underestimation matches
    that of real-space adaptive moments.

    Bad pixels cannot be masked in Fourier space; images should have
    defects interpolated before fitting.  The weight maps are only used
    to set the overall noise level of each epoch.
    """

    kind = 'pam'

    def __init__(
        self,
        fwhm_smooth=None,
        smooth_fac=DEFAULT_SMOOTH_FAC,
        pad_factor=4,
        ap_rad=1.5,
        maxiter=DEFAULT_MAXITER,
        shiftmax=DEFAULT_SHIFTMAX,
        etol=DEFAULT_ETOL,
        Ttol=DEFAULT_TTOL,
        cen_tol=DEFAULT_CENTOL,
        rng=None,
    ):

        self.fwhm_smooth = fwhm_smooth
        self.smooth_fac = smooth_fac
        self.pad_factor = pad_factor
        self.ap_rad = ap_rad
        self.maxiter = maxiter
        self.shiftmax = shiftmax
        self.etol = etol
        self.Ttol = Ttol
        self.cen_tol = cen_tol
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
        PrePSFAdmomResult
        """
        is_mb = isinstance(obs, MultiBandObsList)
        mb_obs = get_mb_obs(obs)

        fwhm_smooth = self._get_fwhm_smooth(mb_obs, no_psf)
        Tsmooth = fwhm_to_T(fwhm_smooth) if fwhm_smooth > 0 else 0.0

        epochs = []
        for band, obslist in enumerate(mb_obs):
            for tobs in obslist:
                epochs.append(
                    self._prep_epoch(tobs, band, Tsmooth, no_psf)
                )

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

        return PrePSFAdmomResult(obs=obs, result=result)

    def _run_admom(self, epochs, nband, guess_gmix, Tsmooth):
        """
        the adaptive moments iteration, accumulating the k-space sums
        over all epochs
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
        e1old = e2old = Told = np.nan
        numiter = 0

        for i in range(self.maxiter):
            numiter = i + 1

            if (Sigma[0, 0] <= 0 or Sigma[1, 1] <= 0
                    or _det2(Sigma) < GMIX_LOW_DETVAL):
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
            Sigma, flags = _deweight(M, Sigma)
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

    def _get_result(
        self, epochs, nband, flags, numiter, Sigma, v0, u0, Tsmooth,
    ):
        """
        package the result, measuring the per-band fluxes and the
        errors with the converged weight
        """

        res = {
            'flags': flags,
            'numiter': numiter,
            'nband': nband,
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
            sums, sums_cov, fluxes, flux_vars = self._finalize(
                epochs, nband, Sigma, v0, u0,
            )
            res['sums'] = sums
            res['sums_cov'] = sums_cov

            Twt = Sigma[0, 0] + Sigma[1, 1]
            Tgal = Twt - Tsmooth
            M1 = Sigma[1, 1] - Sigma[0, 0]
            M2 = 2 * Sigma[0, 1]

            res['T'] = Tgal

            res['flux'] = fluxes
            if np.all(flux_vars > 0):
                res['flux_err'] = np.sqrt(flux_vars)
                res['s2n'] = np.sqrt(
                    np.sum((res['flux'] / res['flux_err']) ** 2)
                )
            else:
                res['flux_flags'] |= ngmix.flags.NONPOS_VAR

            if sums_cov[4, 4] > 0 and sums_cov[5, 5] > 0:
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

            if Tgal > 0:
                res['e1'] = M1 / Tgal
                res['e2'] = M2 / Tgal
                res['e'] = np.array([res['e1'], res['e2']])

                if sums_cov[2, 2] > 0 and sums_cov[3, 3] > 0:
                    # the lever arm Twt/Tgal accounts for the smoothing
                    # subtraction in the denominator of e = M1/Tgal
                    lever = Twt / Tgal
                    res['e1err'] = lever * 2 * get_ratio_error(
                        sums[2], sums[4],
                        sums_cov[2, 2], sums_cov[4, 4], sums_cov[2, 4],
                    )
                    res['e2err'] = lever * 2 * get_ratio_error(
                        sums[3], sums[4],
                        sums_cov[3, 3], sums_cov[4, 4], sums_cov[3, 4],
                    )
                    if (not np.isfinite(res['e1err'])
                            or not np.isfinite(res['e2err'])):
                        res['e1err'] = np.nan
                        res['e2err'] = np.nan
                        res['flags'] |= ngmix.flags.NONPOS_SHAPE_VAR
                    else:
                        res['e_err'] = np.array([res['e1err'], res['e2err']])
                        res['e_cov'] = np.diag(
                            [res['e1err'] ** 2, res['e2err'] ** 2]
                        )
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
            alpha, beta = _get_phase_angles(epoch, v0, u0)
            admom_ksums(
                epoch['kim'], epoch['iy'], epoch['ix'], epoch['dim'],
                alpha, beta, epoch['kv'], epoch['ku'],
                Sigma[0, 0], Sigma[0, 1], Sigma[1, 1], epoch['df2'],
                esums,
            )
            sums += epoch['weight'] * epoch['detAtinv'] * esums
        return sums

    def _finalize(self, epochs, nband, Sigma, v0, u0):
        """
        get the final accumulated sums, their covariance, and the
        per-band fluxes and variances, using the converged weight

        The covariance follows the prepsfmom convention: each Fourier
        mode is treated as independent with variance equal to the total
        variance of the input image.

        The flux normalization: the raw flux sum corresponds to a
        weighted flux with an effective real-space kernel of peak value
        1/(2 pi sqrt(det(Sigma)) detAtinv).  We normalize to a unit peak
        kernel, and the factor of 2 converts the gaussian-weighted flux
        to the total flux at the fixed point of the iteration.  The
        smoothing preserves flux, so this also holds when smoothing.
        """
        knrm = 2 * np.pi * np.sqrt(_det2(Sigma))

        sums = np.zeros(6)
        cov = np.zeros((6, 6))
        esums = np.zeros(6)
        ecov = np.zeros((6, 6))

        fsums = np.zeros(nband)
        fvars = np.zeros(nband)
        wsums = np.zeros(nband)

        for epoch in epochs:
            alpha, beta = _get_phase_angles(epoch, v0, u0)
            admom_finalize(
                epoch['kim'], epoch['iy'], epoch['ix'], epoch['dim'],
                alpha, beta, epoch['kv'], epoch['ku'],
                Sigma[0, 0], Sigma[0, 1], Sigma[1, 1], epoch['df2'],
                epoch['err_fac2'],
                esums, ecov,
            )
            fac = epoch['weight'] * epoch['detAtinv']
            nfac = epoch['tot_var'] * epoch['df2'] ** 2

            sums += fac * esums
            cov += fac ** 2 * nfac * ecov

            band = epoch['band']
            fsums[band] += fac * esums[5]
            fvars[band] += fac ** 2 * nfac * ecov[5, 5]
            wsums[band] += epoch['weight']

        fluxes = 2 * knrm * fsums / wsums
        flux_vars = (2 * knrm / wsums) ** 2 * fvars
        return sums, cov, fluxes, flux_vars

    def _prep_epoch(self, obs, band, Tsmooth, no_psf):
        """
        FFT the image and psf, deconvolve the psf, apply the smoothing,
        and store the k grids and noise information
        """
        psf_obs = _check_obs_and_get_psf_obs(obs, no_psf)

        wmsk = obs.weight > 0
        if not np.any(wmsk):
            raise ValueError('no positive weight pixels in observation')
        tot_var = np.sum(1.0 / obs.weight[wmsk])

        if psf_obs is not None:
            max_dim = max(obs.image.shape + psf_obs.image.shape)
        else:
            max_dim = max(obs.image.shape)
        target_dim = int(max_dim * self.pad_factor)
        # the square of this is the ratio of padded to unpadded pixel
        # counts, used to scale the noise
        eff_pad_factor = target_dim / np.sqrt(
            obs.image.shape[0] * obs.image.shape[1]
        )

        # the image is real, so we work with the rfft half plane;
        # conjugate modes are folded in with the symmetry weights
        kim, im_row, im_col = _zero_pad_and_compute_rfft(
            obs.image, obs.jacobian.row0, obs.jacobian.col0, target_dim,
            self.ap_rad,
        )
        dim = kim.shape[0]

        if psf_obs is not None:
            kpsf, psf_row, psf_col = _zero_pad_and_compute_rfft(
                psf_obs.image,
                psf_obs.jacobian.row0, psf_obs.jacobian.col0,
                target_dim,
                0,  # we do not apodize PSF stamps
            )
        else:
            kpsf = _pixel_fft(dim)[:, :dim // 2 + 1]
            psf_row = 0.0
            psf_col = 0.0

        max_amp = np.abs(kpsf[0, 0])

        # the k grids, mode selection and smoothing profile are shared
        # between fits with the same stamp geometry, so they are cached
        jac = obs.jacobian
        grids = _get_kspace_grids(
            dim, jac.dvdrow, jac.dvdcol, jac.dudrow, jac.dudcol,
            float(Tsmooth),
        )

        # extract the masked modes
        kim = kim[grids['iy'], grids['ix']]
        kpsf = kpsf[grids['iy'], grids['ix']]

        kim, kpsf, _ = _deconvolve_im_psf_inplace(kim, kpsf, max_amp)

        # fold holds the smoothing profile times the conjugate symmetry
        # weights; fold2 has the squared smoothing for the noise, where
        # the symmetry weight enters once
        kim *= grids['fold']
        # factor for noise propagation: the effective kernels act on
        # the raw image fft, so include the smoothing and deconvolution
        err_fac2 = grids['fold2'] / np.abs(kpsf) ** 2

        return {
            'band': band,
            'kim': kim,
            'iy': grids['iy'],
            'ix': grids['ix'],
            'dim': dim,
            'kv': grids['kv'],
            'ku': grids['ku'],
            'Atinv': grids['Atinv'],
            # the phase shift putting the effective center at the image
            # jacobian center; the psf centering phase cancels in the
            # deconvolution except for this difference
            'drow': im_row - psf_row,
            'dcol': im_col - psf_col,
            'detAtinv': grids['detAtinv'],
            'df2': 1.0 / dim ** 2,
            'tot_var': tot_var * eff_pad_factor ** 2,
            'err_fac2': err_fac2,
            'weight': 1.0 / (tot_var * eff_pad_factor ** 2),
        }

    def _get_fwhm_smooth(self, mb_obs, no_psf):
        """
        get the smoothing fwhm, fitting the PSFs if it was not sent
        """
        if self.fwhm_smooth is not None:
            return self.fwhm_smooth

        if no_psf:
            raise ValueError(
                'send fwhm_smooth= (0 to disable) when no_psf=True'
            )

        from ..admom import run_admom

        Tmax = 0.0
        for obslist in mb_obs:
            for obs in obslist:
                if not obs.has_psf():
                    raise RuntimeError(
                        "The PSF must be set to measure a pre-PSF moment!"
                    )
                psf_obs = obs.psf
                if psf_obs.has_gmix():
                    T = psf_obs.gmix.get_T()
                else:
                    scale = psf_obs.jacobian.get_scale()
                    T = None
                    for fac in [3.0, 1.5, 6.0]:
                        Tguess = fwhm_to_T(fac * scale)
                        pres = run_admom(
                            psf_obs, guess=Tguess, rng=self._get_rng(),
                        )
                        if pres['flags'] == 0:
                            T = pres['T']
                            break
                    if T is None:
                        raise RuntimeError(
                            'could not fit PSF to choose the smoothing; '
                            'send fwhm_smooth= explicitly'
                        )
                Tmax = max(Tmax, T)

        return self.smooth_fac * T_to_fwhm(Tmax)

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


def _get_phase_angles(epoch, v0, u0):
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


@functools.lru_cache(maxsize=16)
def _get_kspace_grids(dim, dvdrow, dvdcol, dudrow, dudcol, Tsmooth):
    """
    get the k grids in sky coordinates over the rfft half plane, the
    selection of modes where the smoothing kernel is significant, and
    the smoothing profile folded with the conjugate symmetry weights.
    These only depend on the stamp geometry and smoothing, so they are
    cached and shared between fits; the returned arrays are marked
    read-only.
    """
    half = dim // 2 + 1
    f1d = fft.fftfreq(dim) * (2.0 * np.pi)
    fx = f1d[:half].reshape(1, -1)
    fy = f1d.reshape(-1, 1)
    Atinv = np.linalg.inv(
        [[dvdrow, dvdcol], [dudrow, dudcol]]
    ).T
    kv = Atinv[0, 0] * fy + Atinv[0, 1] * fx
    ku = Atinv[1, 0] * fy + Atinv[1, 1] * fx
    detAtinv = np.abs(np.linalg.det(Atinv))

    kmag2 = kv * kv + ku * ku

    # we only keep modes where the smoothing kernel is significant
    if Tsmooth > 0:
        chi2_2 = 0.25 * Tsmooth * kmag2
        msk = chi2_2 < FASTEXP_MAX_CHI2 / 2
    else:
        msk = np.ones((dim, half), dtype=bool)

    iy, ix = np.where(msk)

    kmag2 = kmag2[msk]
    kv = kv[msk]
    ku = ku[msk]

    # conjugate symmetry weights: modes with 0 < kx < nyquist appear
    # twice in the full plane
    wsym = np.ones(ix.size)
    if dim % 2 == 0:
        wsym[(ix > 0) & (ix < dim // 2)] = 2.0
    else:
        wsym[ix > 0] = 2.0

    if Tsmooth > 0:
        smooth = np.exp(-0.25 * Tsmooth * kmag2)

        # check that the smoothing kernel is contained in the FFT
        # region; the converged weight is at least this large so this
        # covers the worst case.  the tolerance is loose since small
        # truncations only produce correspondingly small biases in the
        # moments
        nrm = np.sum(wsym * smooth) * detAtinv * np.pi * Tsmooth / dim**2
        if not np.allclose(nrm, 1.0, atol=1e-3, rtol=0):
            raise FFTRangeError(
                'FFT size appears too small for smoothing fwhm %g: '
                'norm = %f (should be 1)' % (T_to_fwhm(Tsmooth), nrm)
            )

        fold = wsym * smooth
        fold2 = wsym * smooth * smooth
    else:
        fold = wsym
        fold2 = wsym

    grids = {
        'kv': kv,
        'ku': ku,
        'iy': iy,
        'ix': ix,
        'fold': fold,
        'fold2': fold2,
        'Atinv': Atinv,
        'detAtinv': detAtinv,
    }
    for key in ['kv', 'ku', 'iy', 'ix', 'fold', 'fold2', 'Atinv']:
        grids[key].flags.writeable = False
    return grids


def _zero_pad_and_compute_rfft_impl(im, cen_row, cen_col, target_dim, ap_rad):
    """
    zero pad and compute the real fft, returning the fft and the center
    location in the padded image
    """
    if ap_rad > 0:
        ap_mask = np.ones_like(im)
        _build_square_apodization_mask(ap_rad, ap_mask)
        im = im * ap_mask

    pim, pad_row_before, pad_col_before = _zero_pad_image(im, target_dim)
    pad_cen_row = cen_row + pad_row_before
    pad_cen_col = cen_col + pad_col_before
    kpim = fft.rfftn(pim)
    return kpim, pad_cen_row, pad_cen_col


@functools.lru_cache(maxsize=128)
def _zero_pad_and_compute_rfft_cached(
    im_tuple, cen_row, cen_col, target_dim, ap_rad
):
    return _zero_pad_and_compute_rfft_impl(
        np.array(im_tuple), cen_row, cen_col, target_dim, ap_rad
    )


@functools.wraps(_zero_pad_and_compute_rfft_impl)
def _zero_pad_and_compute_rfft(im, cen_row, cen_col, target_dim, ap_rad):
    # respect the fft caching switch in prepsfmom
    if prepsfmom.USE_FFT_CACHE:
        return _zero_pad_and_compute_rfft_cached(
            tuple(tuple(ii) for ii in im),
            float(cen_row), float(cen_col), int(target_dim), float(ap_rad),
        )
    else:
        return _zero_pad_and_compute_rfft_impl(
            im, cen_row, cen_col, target_dim, ap_rad,
        )


def _det2(M):
    return M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]


def _deweight(M, Sigma):
    """
    deweight the measured moments, returning the new weight covariance

    Sigma_new = (M^{-1} - Sigma^{-1})^{-1}
    """
    detm = _det2(M)
    if detm <= GMIX_LOW_DETVAL:
        return Sigma, ngmix.flags.LOW_DET

    detw = _det2(Sigma)
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
