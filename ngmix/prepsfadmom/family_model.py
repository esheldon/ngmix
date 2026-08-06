"""
fixed-ratio gaussian mixture family models for the pre-PSF adaptive
moments fitter, fit by moment matching: FamilyModel and its
ExpModel and DevModel subclasses
"""
__all__ = ['FamilyModel', 'ExpModel', 'DevModel']

import numpy as np

from ..moments import e2mom
import ngmix.flags

from .models import (
    model_ksums, cov_from_e, mixture_model_valid, det2, deweight,
)
from .errors import model_sandwich
from .base_model import PAdmomModel


class FamilyModel(PAdmomModel):
    """
    a fixed-ratio gaussian mixture family fit by moment matching:
    the ngmix 6-gaussian exponential ('exp') or 10-gaussian de
    Vaucouleurs ('dev') expansion.  T, e1 and e2 are the family
    parameters and the fluxes are total model fluxes rather than
    gaussian aperture fluxes.  The family size is not clipped and
    can scatter through zero for marginally resolved objects,
    keeping ensemble averages unbiased; the shape is then undefined
    and NONPOS_SIZE is set while T and the fluxes remain usable.

    Parameters
    ----------
    name: str
        'exp' or 'dev'
    """

    supports_full_errors = True

    def __init__(self, name):
        if name not in ('exp', 'dev'):
            raise ValueError(
                f"bad family '{name}', expected 'exp' or 'dev'"
            )
        self.name = name

    def model_spec(self, state=None):
        """
        the model specification for validity checks
        """
        return self.name

    def unit_state(self, Sfam, state=None):
        """
        a unit-flux model state dict at the given family covariance
        """
        return {'type': self.name, 'cov': Sfam, 'F': np.ones(1)}

    def run_admom(self, fitter, epochs, nband, guess_gmix, Tsmooth):
        """
        fit a fixed-ratio gaussian mixture family ('exp' or 'dev')
        by moment matching.
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

        # last accepted plain (unboosted, undamped) shift, for the
        # extrapolation ratio; None whenever a fresh pair is needed
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
                Sfam, M, Sigma, newSigma, Tsmooth,
            )
            shift, boosted = self.steffensen_boost(
                raw_shift, prev_shift, pflags,
            )
            prop, idamp, accepted = self.damped_step(
                Sfam, shift, newSigma, Tsmooth,
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

            scale = newSigma[0, 0] + newSigma[1, 1]
            # convergence requires an undamped step and is tested on
            # the raw fixed-point residual, not the boosted step: a
            # damped step can be small only because it was shortened
            # at the validity boundary, not because the fit has
            # settled
            converged = (
                idamp == 0
                and np.abs(raw_shift).max() < fitter.etol * scale
                and abs(dv) < fitter.cen_tol
                and abs(du) < fitter.cen_tol
            )
            Sfam = prop
            Sigma = newSigma

            if converged:
                break
        else:
            flags = ngmix.flags.MAXITER

        model_state = {
            'type': self.name, 'cov': Sfam, 'F': np.ones(nband),
        }
        return fitter._get_result(
            epochs=epochs, nband=nband, flags=flags, numiter=numiter,
            Sigma=Sigma, v0=v0, u0=u0, Tsmooth=Tsmooth,
            model_state=model_state,
        )

    def model_shift(self, Sfam, M, Sigma, newSigma, Tsmooth,
                    state=None):
        """
        the raw fixed point shift of the family covariance: the
        deweight-mapped difference of the measured and model
        predicted moments.  Returns (shift, pflags); when the
        prediction cannot be deweighted (pflags nonzero) the shift
        is the gain-1 fallback on the weighted moment ratios
        """
        # the model predicted moment ratios; with a common center
        # and common structure across epochs these are the same
        # for every epoch, so one closed-form evaluation suffices
        unit_state = self.unit_state(Sfam, state=state)
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
            M1 = M[1, 1] - M[0, 0]
            M2 = 2 * M[0, 1]
            Tm = M[0, 0] + M[1, 1]
            Tf = Sfam[0, 0] + Sfam[1, 1]
            fac = Tm / Tp
            de1 = M1 / Tm - Mp1 / Tp
            de2 = M2 / Tm - Mp2 / Tp
            shift = (fac - 1) * Sfam + 0.5 * fac * Tf * np.array([
                [-de1, de2],
                [de2, de1],
            ])
        return shift, pflags

    def steffensen_boost(self, raw_shift, prev_shift, pflags):
        """
        Steffensen-style extrapolation: the iteration converges
        linearly with a steady ratio, so two successive plain steps
        give the ratio and the remaining geometric series can be
        summed in one boosted step.  Guarded to the primary update
        branch and a stable ratio range.  Returns (shift, boosted)
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

    def damped_step(self, Sfam, shift, newSigma, Tsmooth, state=None):
        """
        accept the largest step, damping if needed, for which the
        smoothed model components stay positive definite.  Returns
        (prop, idamp, accepted)
        """
        accepted = False
        for idamp in range(10):
            prop = Sfam + shift
            if mixture_model_valid(
                    self.model_spec(state), prop, newSigma, Tsmooth):
                accepted = True
                break
            shift = 0.5 * shift
        return prop, idamp, accepted

    def shape_state(self, model_state, Sigma, Tsmooth):
        """
        the galaxy moments (M1, M2, Tgal) and whether the shape
        M/Tgal is defined: the family covariance can scatter out of
        positive definite, where the shape is undefined
        """
        Sfam = model_state['cov']
        M1 = Sfam[1, 1] - Sfam[0, 0]
        M2 = 2 * Sfam[0, 1]
        Tgal = Sfam[0, 0] + Sfam[1, 1]
        shape_ok = Tgal > 0 and det2(Sfam) > 0
        return M1, M2, Tgal, shape_ok

    def set_T_err(self, res, model_state, fam_cov, sums, sums_cov):
        """
        set the T error in the result from the family covariance
        sandwich, or NONPOS_VAR in T_flags
        """
        if fam_cov is not None and fam_cov[2, 2] > 0:
            res['T_err'] = np.sqrt(fam_cov[2, 2])
        else:
            res['T_flags'] |= ngmix.flags.NONPOS_VAR

    def _shape_errors(self, res, fam_cov, Sigma, sums, sums_cov, Tgal):
        """
        linearize e = M/T over the family covariance sandwich
        """
        e1err = np.nan
        e2err = np.nan
        e12cov = np.nan
        if fam_cov is None:
            # the family sandwich failed; the errors stay nan
            # and the flags are set by the caller
            pass
        else:
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
        return e1err, e2err, e12cov

    def flux_results(
        self, state, epochs, nband, Sigma, Tsmooth, sums, cov,
        fsums, fvars, fmcovs, wsums,
    ):
        """
        the per-band fluxes normalized by the unit flux model
        predictions, with the sandwich errors over the moment
        matching conditions
        """
        upred = self._unit_flux_preds(
            state, epochs, nband, Sigma, Tsmooth,
        )
        fluxes = fsums / upred
        rawvars, fam_cov, fcov_raw = model_sandwich(
            self.name, state['cov'], Sigma, Tsmooth, sums, cov,
            fsums, fvars, fmcovs,
        )
        if rawvars is None:
            # the sandwich could not be evaluated; fall back to
            # the fixed weight variances, with the structure
            # errors flagged downstream
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


class ExpModel(FamilyModel):
    """
    the ngmix 6-gaussian exponential expansion, fit by moment
    matching; see FamilyModel
    """

    def __init__(self):
        super().__init__('exp')


class DevModel(FamilyModel):
    """
    the ngmix 10-gaussian de Vaucouleurs expansion, fit by moment
    matching; see FamilyModel
    """

    def __init__(self):
        super().__init__('dev')
