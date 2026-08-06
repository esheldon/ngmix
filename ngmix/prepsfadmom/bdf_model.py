"""
the composite exp plus dev ('bdf') model for the pre-PSF adaptive
moments fitter
"""
__all__ = ['BDFModel']

import numpy as np
from numpy import fft

from ..fastexp_nb import FASTEXP_MAX_CHI2
from ..moments import e2mom
import ngmix.flags

from .models import cov_from_e, get_profile_comps, deweight
from .errors import model_sandwich, bdf_joint_sandwich, _mbasis_cov
from .prep import get_phase_angles
from .family_model import FamilyModel


class BDFModel(FamilyModel):
    """
    the composite exp plus dev model: shared center and ellipticity,
    the dev size TdByTe times the exp size, with the per band flux
    split between the components (fracdev) fit by a per-sweep
    two-template GLS solve on the retained modes, interleaved with
    the family adaptive step.  The result gains fracdev,
    fracdev_gls, fracdev_gls_err, flux_exp, flux_dev and
    flux_gls_cov entries; the flux entries are the band-summed GLS
    component fluxes (the split is per band, so bulge and disk
    colors differ), and the flux and structure errors are
    conditional on the converged split.

    Parameters
    ----------
    TdByTe: float
        The dev to exp size ratio of the composite
    fracdev0, fracdev_sigma0: float, optional
        Sent together, regularize the split that builds the
        composite model: the inverse-variance blend of the measured
        split with the prior, using the conditional GLS split
        variance, which is deterministic given the structure.  Only
        the model split (the reported fracdev) is regularized; the
        reported component fluxes and fracdev_gls stay the raw
        linear solutions.  fracdev_sigma0=0 freezes the model split
        at fracdev0
    """

    name = 'bdf'
    supports_full_errors = False

    def __init__(self, TdByTe, fracdev0=None, fracdev_sigma0=None):
        TdByTe = float(TdByTe)
        if TdByTe <= 0:
            raise ValueError(f'TdByTe must be positive, got {TdByTe}')

        has0 = fracdev0 is not None
        hass = fracdev_sigma0 is not None
        if has0 != hass:
            raise ValueError(
                "the bdf shrinkage requires both 'fracdev0' and "
                "'fracdev_sigma0' (or neither)"
            )
        if has0:
            fracdev0 = float(fracdev0)
            sigma0 = float(fracdev_sigma0)
            if sigma0 < 0:
                raise ValueError(
                    f'fracdev_sigma0 must be non-negative, got '
                    f'{sigma0}'
                )
            self.fracdev_shrink = (fracdev0, sigma0)
        else:
            self.fracdev_shrink = None

        self.TdByTe = TdByTe

    def model_spec(self, state=None):
        """
        the model spec dict carrying the split state, for validity
        checks
        """
        return {
            'type': 'bdf',
            'fracdev': state['fracdev'],
            'TdByTe': self.TdByTe,
        }

    def unit_state(self, Sfam, state=None):
        """
        a unit-flux model state dict at the given family covariance
        """
        return {
            'type': 'bdf', 'cov': Sfam, 'F': np.ones(1),
            'fracdev': state['fracdev'], 'TdByTe': self.TdByTe,
        }

    def run_admom(self, fitter, epochs, nband, guess_gmix, Tsmooth):
        """
        the composite exp plus dev fit: per sweep, one family
        adaptive step at the current flux split (the composite at
        fixed fracdev is a fixed-ratio 16-gaussian mixture, so the
        mixture machinery applies unchanged), then a per-band
        two-template GLS flux solve at the updated structure, whose
        band-summed split becomes the next sweep's fracdev.  The
        interleaved schedule keeps the fixed point map stationary
        for the Steffensen extrapolation and converges much faster
        than alternating converged blocks.

        The split is clipped to [-0.5, 1.5] rather than [0, 1]:
        mildly out-of-range noise excursions stay linear (no
        selection-like clipping nonlinearity) while runaway values
        cannot destabilize the composite table
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

        state = {
            'type': 'bdf', 'fracdev': 0.5, 'TdByTe': self.TdByTe,
        }
        F2 = None
        fcovs = None

        flags = 0
        numiter = 0
        prev_shift = None

        for i in range(fitter.maxiter):
            numiter = i + 1

            flags, v0, u0, dv, du, M = fitter._measure_step(
                epochs, Sigma, v0, u0, vorig, uorig,
            )
            if flags != 0:
                break

            newSigma, flags = deweight(M, Sigma)
            if flags != 0:
                break

            raw_shift, pflags = self.model_shift(
                Sfam, M, Sigma, newSigma, Tsmooth, state=state,
            )
            shift, boosted = self.steffensen_boost(
                raw_shift, prev_shift, pflags,
            )
            prop, idamp, accepted = self.damped_step(
                Sfam, shift, newSigma, Tsmooth, state=state,
            )
            if not accepted:
                flags = ngmix.flags.LOW_DET
                break

            if idamp == 0 and not boosted and pflags == 0:
                prev_shift = raw_shift
            else:
                prev_shift = None

            Sfam = prop
            Sigma = newSigma

            # the flux reassignment at the updated structure; the
            # band-summed split drives the composite table
            F2, fcovs = self._solve_fluxes(
                epochs, nband, Sfam, v0, u0,
            )
            fd_gls, fd_var = _combined_split(F2, fcovs)
            if fd_gls is None:
                newfd = state['fracdev']
            else:
                newfd = np.clip(
                    self._shrink_split(fd_gls, fd_var), -0.5, 1.5,
                )
            dfd = abs(newfd - state['fracdev'])
            state['fracdev'] = newfd

            scale = newSigma[0, 0] + newSigma[1, 1]
            converged = (
                idamp == 0
                and np.abs(raw_shift).max() < fitter.etol * scale
                and abs(dv) < fitter.cen_tol
                and abs(du) < fitter.cen_tol
                and dfd < fitter.etol
            )
            if converged:
                break
        else:
            flags = ngmix.flags.MAXITER

        state['cov'] = Sfam
        state['F'] = np.ones(nband)

        # the joint-sandwich inputs: the split estimator's
        # structure response, the shrinkage factor, the
        # conditional split variance and the cross covariance of
        # the split noise with the moment and flux sums
        state['bdf_err'] = None
        if flags == 0 and F2 is not None:
            fd_gls, fd_var = _combined_split(F2, fcovs)
            if fd_var is not None and fd_var > 0:
                G = self._split_response(
                    epochs, nband, Sfam, F2, v0, u0,
                )
                eta_scov, eta_fcovs = self._noise_cross(
                    epochs, nband, Sfam, Sigma, F2, fcovs,
                )
                if G is not None and eta_scov is not None:
                    state['bdf_err'] = (
                        G, self._shrink_k(fd_var), fd_var,
                        eta_scov, eta_fcovs,
                    )

        state['fd_var_total'] = None
        res = fitter._get_result(
            epochs=epochs, nband=nband, flags=flags, numiter=numiter,
            Sigma=Sigma, v0=v0, u0=u0, Tsmooth=Tsmooth,
            model_state=state,
        )
        res['fracdev_err'] = np.nan
        if state['fd_var_total'] is not None:
            if state['fd_var_total'] > 0:
                res['fracdev_err'] = np.sqrt(
                    state['fd_var_total'],
                )
        res['fracdev'] = state['fracdev']
        res['TdByTe'] = self.TdByTe
        res['fracdev_gls'] = np.nan
        res['fracdev_gls_err'] = np.nan
        if F2 is not None:
            fd_gls, fd_var = _combined_split(F2, fcovs)
            if fd_gls is not None:
                res['fracdev_gls'] = fd_gls
                if fd_var > 0:
                    res['fracdev_gls_err'] = np.sqrt(fd_var)
        if F2 is not None:
            res['flux_exp'] = F2[:, 0].copy()
            res['flux_dev'] = F2[:, 1].copy()
            res['flux_gls_cov'] = fcovs.copy()
            if res['flags'] == 0:
                # the per-band totals come from the per-band GLS
                # component fluxes: the family normalization uses
                # the single shared fracdev, which biases bands
                # whose split differs from the combined one.  The
                # sandwich flux errors are kept: as error
                # estimates the split mismatch is negligible
                res['flux'] = F2.sum(axis=1)
                if np.all(res['flux_err'] > 0):
                    res['s2n'] = np.sqrt(
                        np.sum((res['flux'] / res['flux_err']) ** 2)
                    )
        else:
            res['flux_exp'] = np.zeros(nband) + np.nan
            res['flux_dev'] = np.zeros(nband) + np.nan
            res['flux_gls_cov'] = np.zeros((nband, 2, 2)) + np.nan
        return res

    def _shrink_split(self, fd_gls, fd_var):
        """
        the regularized flux split: the inverse-variance blend of
        the measured split with the prior (fracdev0, sigma0) from
        the model spec.  sigma0 = 0 freezes the split at fracdev0;
        without a prior the measured split passes through.  The
        blend uses the conditional GLS variance, which is set by
        the templates and the noise power (not the pixel noise),
        so the shrinkage weight is deterministic given the
        structure
        """
        if self.fracdev_shrink is None:
            return fd_gls
        fracdev0, sigma0 = self.fracdev_shrink
        if sigma0 == 0:
            return fracdev0
        if fd_var <= 0:
            return fd_gls
        w = 1.0 / fd_var
        w0 = 1.0 / sigma0 ** 2
        return (fd_gls * w + fracdev0 * w0) / (w + w0)

    def _templates(self, epoch, Sfam, phase):
        """
        the exp and dev unit-flux templates on the epoch's
        retained modes at the given family covariance, with the
        fold factor and centering phase applied
        """
        kv = epoch['kv']
        ku = epoch['ku']
        ts = []
        for spec, S in (
            ('exp', Sfam), ('dev', self.TdByTe * Sfam),
        ):
            amp = np.zeros(kv.size)
            for frac, cT in get_profile_comps(spec):
                q = (
                    cT * S[0, 0] * kv * kv
                    + 2 * cT * S[0, 1] * kv * ku
                    + cT * S[1, 1] * ku * ku
                )
                amp += frac * np.exp(-0.5 * q)
            ts.append(amp * epoch['fold'] * phase)
        return ts

    def _split_response(self, epochs, nband, Sfam, F2, v0, u0):
        """
        d fd_gls / d (M1, M2, T) of the family covariance at the
        model consistent point: the GLS split of the converged
        model itself, re-solved with templates at perturbed
        structure.  Central differences over the mbasis
        """
        Tw = Sfam[0, 0] + Sfam[1, 1] + self.TdByTe * (
            Sfam[0, 0] + Sfam[1, 1]
        )
        h = 1.0e-6 * max(Tw, 1.0e-3)
        fam0 = np.array([
            Sfam[1, 1] - Sfam[0, 0], 2 * Sfam[0, 1],
            Sfam[0, 0] + Sfam[1, 1],
        ])

        def split_at(famvec):
            Sp = _mbasis_cov(*famvec)
            A = np.zeros((nband, 2, 2))
            b = np.zeros((nband, 2))
            for epoch in epochs:
                alpha, beta = get_phase_angles(epoch, v0, u0)
                f1d = fft.fftfreq(epoch['dim']) * (2.0 * np.pi)
                ang = (
                    f1d[epoch['iy']] * alpha
                    + f1d[epoch['ix']] * beta
                )
                phase = np.cos(ang) - 1j * np.sin(ang)
                tp = self._templates(epoch, Sp, phase)
                t0 = self._templates(epoch, Sfam, phase)
                band = epoch['band']
                model = F2[band, 0] * t0[0] + F2[band, 1] * t0[1]
                g = 1.0 / epoch['err_fac2']
                for a in range(2):
                    b[band, a] += np.sum(
                        (np.conj(tp[a]) * model).real * g,
                    )
                    for cc in range(2):
                        A[band, a, cc] += np.sum(
                            (np.conj(tp[a]) * tp[cc]).real * g,
                        )
            E = 0.0
            D = 0.0
            for band in range(nband):
                Fb = np.linalg.solve(A[band], b[band])
                E += Fb[0]
                D += Fb[1]
            S = E + D
            return D / S if S != 0 else None

        G = np.zeros(3)
        for i in range(3):
            famp = fam0.copy()
            famm = fam0.copy()
            famp[i] += h
            famm[i] -= h
            fp = split_at(famp)
            fm = split_at(famm)
            if fp is None or fm is None:
                return None
            G[i] = (fp - fm) / (2 * h)
        return G

    def _noise_cross(self, epochs, nband, Sfam, Sigma, F2, fcovs):
        """
        the analytic cross covariance of the split noise eta with
        the accumulated moment and flux sums.

        Both are linear functionals of the same retained modes:
        the sums use the adaptive-weight moment kernels, the split
        the GLS of the templates with weights 1/err_fac2.  With
        Var(mode) = err_fac2 the noise power cancels in the cross
        term, leaving the pure template-times-kernel overlap

            Cov(b_a, s_c) = sum_epochs fac df2 sum_i t_a(i) kern_c(i)

        mapped through the per-band GLS inverse and the combined
        split derivative.  These cross terms are essential in the
        joint sandwich: the T kernel and the split direction are
        strongly anti-correlated and the coupled amplified paths
        nearly cancel.

        Returns
        -------
        eta_scov: size 3, Cov(eta, s[2:5]) in sums normalization
        eta_fcovs: size nband, Cov(eta, fs_band)
        """
        X = np.zeros((nband, 2, 4))
        for epoch in epochs:
            kv = epoch['kv']
            ku = epoch['ku']
            Sv = Sigma[0, 0] * kv + Sigma[0, 1] * ku
            Su = Sigma[0, 1] * kv + Sigma[1, 1] * ku
            chi2 = kv * Sv + ku * Su
            wk = np.exp(-0.5 * np.clip(chi2, 0, FASTEXP_MAX_CHI2))
            wk[(chi2 > FASTEXP_MAX_CHI2) | (chi2 < 0)] = 0.0
            vvk = (Sigma[0, 0] - Sv * Sv) * wk
            vuk = (Sigma[0, 1] - Sv * Su) * wk
            uuk = (Sigma[1, 1] - Su * Su) * wk
            kern = (uuk - vvk, 2 * vuk, uuk + vvk, wk)

            ts = self._templates(epoch, Sfam, 1.0)
            fac = epoch['weight'] * epoch['detAtinv'] * epoch['df2']
            band = epoch['band']
            for a in range(2):
                for c in range(4):
                    X[band, a, c] += fac * np.sum(ts[a] * kern[c])

        D = F2[:, 1].sum()
        S = F2.sum()
        if S == 0:
            return None, None
        fd = D / S

        eta_scov = np.zeros(3)
        eta_fcovs = np.zeros(nband)
        for band in range(nband):
            CF = fcovs[band] @ X[band]
            w = ((1.0 - fd) * CF[1] - fd * CF[0]) / S
            eta_scov += w[:3]
            eta_fcovs[band] = w[3]
        return eta_scov, eta_fcovs

    def _shrink_k(self, fd_var):
        """
        the shrinkage factor d fd_used / d fd_gls
        """
        if self.fracdev_shrink is None:
            return 1.0
        _, sigma0 = self.fracdev_shrink
        if sigma0 == 0:
            return 0.0
        if fd_var is None or fd_var <= 0:
            return 1.0
        return sigma0 ** 2 / (sigma0 ** 2 + fd_var)

    def _solve_fluxes(self, epochs, nband, Sfam, v0, u0):
        """
        per-band GLS fit of the exp and dev templates (dev at
        TdByTe times the family covariance) to the retained modes,
        accumulated over each band's epochs.  The per-mode noise
        weighting makes this the exact convolved-space GLS even
        though it runs on the deconvolved smoothed modes.  The
        templates carry the epoch fold factor, matching the kim
        construction, and the object's centering phase (the
        conjugate of the kernel phasors, which translate by +ang).

        Returns
        -------
        F2: (nband, 2) array of (F_exp, F_dev)
        fcovs: (nband, 2, 2) flux covariances, conditional on the
            structure
        """
        A = np.zeros((nband, 2, 2))
        b = np.zeros((nband, 2))

        for epoch in epochs:
            alpha, beta = get_phase_angles(epoch, v0, u0)
            f1d = fft.fftfreq(epoch['dim']) * (2.0 * np.pi)
            ang = f1d[epoch['iy']] * alpha + f1d[epoch['ix']] * beta
            phase = np.cos(ang) - 1j * np.sin(ang)

            ts = self._templates(epoch, Sfam, phase)

            g = 1.0 / epoch['err_fac2']
            kim = epoch['kim']
            band = epoch['band']
            for a in range(2):
                b[band, a] += np.sum((np.conj(ts[a]) * kim).real * g)
                for c in range(a, 2):
                    v = np.sum((np.conj(ts[a]) * ts[c]).real * g)
                    A[band, a, c] += v
                    if c != a:
                        A[band, c, a] += v

        F2 = np.zeros((nband, 2))
        fcovs = np.zeros((nband, 2, 2))
        for band in range(nband):
            F2[band] = np.linalg.solve(A[band], b[band])
            fcovs[band] = np.linalg.inv(A[band])
        return F2, fcovs

    def flux_results(
        self, state, epochs, nband, Sigma, Tsmooth, sums, cov,
        fsums, fvars, fmcovs, wsums,
    ):
        """
        the per-band fluxes normalized by the unit flux model
        predictions, with the joint sandwich errors when the split
        response ingredients are available, else the conditional
        sandwich
        """
        upred = self._unit_flux_preds(
            state, epochs, nband, Sigma, Tsmooth,
        )
        fluxes = fsums / upred

        rawvars = None
        fam_cov = None
        fcov_raw = None
        if state['bdf_err'] is not None:
            G, k, fdv, eta_scov, eta_fcovs = state['bdf_err']
            rawvars, fam_cov, fd_var_tot = bdf_joint_sandwich(
                state, Sigma, Tsmooth, sums,
                cov, fsums, fvars, fmcovs,
                split_grad=G, shrink_k=k,
                fd_var_data=fdv,
                eta_scov=eta_scov,
                eta_fcovs=eta_fcovs,
            )
            state['fd_var_total'] = fd_var_tot
        if rawvars is None:
            rawvars, fam_cov, fcov_raw = model_sandwich(
                state, state['cov'], Sigma,
                Tsmooth, sums, cov, fsums, fvars, fmcovs,
            )
        if rawvars is None:
            # the sandwich could not be evaluated; fall
            # back to the fixed weight variances, with the
            # structure errors flagged downstream
            rawvars = fvars
            fam_cov = None
            fcov_raw = None
        flux_vars = rawvars / upred ** 2
        if fcov_raw is not None:
            uinv = 1.0 / upred
            flux_cov = fcov_raw * np.outer(uinv, uinv)
        else:
            flux_cov = None
        return fluxes, flux_vars, fam_cov, flux_cov


def _combined_split(F2, fcovs):
    """
    the band-combined flux split fd = sum(F_dev) / sum(F_tot) and
    its variance from the per-band GLS flux covariances (delta
    method).  Returns (None, None) for zero total flux
    """
    E = F2[:, 0].sum()
    D = F2[:, 1].sum()
    S = E + D
    if S == 0:
        return None, None
    fd = D / S
    gE = -D / S ** 2
    gD = E / S ** 2
    grad = np.array([gE, gD])
    var = 0.0
    for band in range(F2.shape[0]):
        var += grad @ fcovs[band] @ grad
    return fd, var
