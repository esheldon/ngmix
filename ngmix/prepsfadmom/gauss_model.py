"""
the gauss model for the pre-PSF adaptive moments fitter: the
standard adaptive moments fixed point
"""
__all__ = ['GaussModel']

import numpy as np

from ..moments import e2mom
from ..util import get_ratio_error
import ngmix.flags

from .models import det2, deweight
from .errors import flux_var_delta
from .base_model import PAdmomModel


class GaussModel(PAdmomModel):
    """
    the standard adaptive moments fixed point: deweight the measured
    moments into the next weight until the weight stops changing.
    T, e1 and e2 are the smoothing-subtracted moments of the
    converged weight and the fluxes are gaussian weighted aperture
    fluxes
    """

    name = 'gauss'
    supports_full_errors = True

    def run_admom(self, fitter, epochs, nband, guess_gmix, Tsmooth):
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

            # converge on the weight fixed point residual, the same
            # test as the mixture loop.  The centroid correction
            # decouples the center from the moments, so the center
            # must be tested explicitly
            shift = newSigma - Sigma
            scale = newSigma[0, 0] + newSigma[1, 1]
            converged = (
                np.abs(shift).max() < fitter.etol * scale
                and abs(dv) < fitter.cen_tol
                and abs(du) < fitter.cen_tol
            )
            Sigma = newSigma

            if converged:
                break
        else:
            flags = ngmix.flags.MAXITER

        return fitter._get_result(
            epochs=epochs, nband=nband, flags=flags, numiter=numiter,
            Sigma=Sigma, v0=v0, u0=u0, Tsmooth=Tsmooth,
        )

    def shape_state(self, model_state, Sigma, Tsmooth):
        """
        the galaxy moments (M1, M2, Tgal) and whether the shape
        M/Tgal is defined: the converged weight with the smoothing
        subtracted.  The round smoothing shifts only the diagonal,
        so it drops out of M1 and M2 and only T needs the
        subtraction
        """
        Sgal = Sigma - np.diag([Tsmooth / 2, Tsmooth / 2])
        M1 = Sigma[1, 1] - Sigma[0, 0]
        M2 = 2 * Sigma[0, 1]
        Twt = Sigma[0, 0] + Sigma[1, 1]
        Tgal = Twt - Tsmooth
        shape_ok = Tgal > 0 and det2(Sgal) > 0
        return M1, M2, Tgal, shape_ok

    def set_T_err(self, res, model_state, fam_cov, sums, sums_cov):
        """
        set the T error in the result, or NONPOS_VAR in T_flags
        """
        if sums_cov[4, 4] > 0 and sums_cov[5, 5] > 0:
            # the sums include the weight, so need a factor of two
            # to correct, as in real-space adaptive moments
            res['T_err'] = 4 * get_ratio_error(
                sums[4], sums[5],
                sums_cov[4, 4], sums_cov[5, 5], sums_cov[4, 5],
            )
        else:
            res['T_flags'] |= ngmix.flags.NONPOS_VAR

    def _shape_errors(self, res, fam_cov, Sigma, sums, sums_cov, Tgal):
        """
        the weighted moment ratio errors of e = M/Tgal
        """
        e1err = np.nan
        e2err = np.nan
        e12cov = np.nan
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
        return e1err, e2err, e12cov

    def flux_results(
        self, state, epochs, nband, Sigma, Tsmooth, sums, cov,
        fsums, fvars, fmcovs, wsums,
    ):
        """
        the per-band fluxes and variances at the gaussian fixed
        point.

        The flux normalization: the raw flux sum corresponds to a
        weighted flux with an effective real-space kernel of peak
        value 1/(2 pi sqrt(det(Sigma)) detAtinv).  We normalize to a
        unit peak kernel, and the factor of 2 converts the
        gaussian-weighted flux to the total flux at the fixed point
        of the iteration.  The smoothing preserves flux, so this
        also holds when smoothing.
        """
        knrm = 2 * np.pi * np.sqrt(det2(Sigma))
        fluxes = 2 * knrm * fsums / wsums
        flux_vars = (2 * knrm / wsums) ** 2 * flux_var_delta(
            Sigma, sums, cov, fsums, fvars, fmcovs,
        )
        # the cross-band covariance is only assembled on the
        # model paths (model_sandwich); the plain gauss delta
        # path reports per-band variances only
        return fluxes, flux_vars, None, None
