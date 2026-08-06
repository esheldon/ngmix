"""
the star model for the pre-PSF adaptive moments fitter: a pre-psf
delta function
"""
__all__ = ['StarModel']

import numpy as np

import ngmix.flags

from .base_model import PAdmomModel


class StarModel(PAdmomModel):
    """
    a pre-psf delta function: the weight is the smoothing gaussian,
    only the center and the per band fluxes are fit, and
    fwhm_smooth must be positive
    """

    name = 'star'

    def validate_fit(self, Tsmooth):
        if Tsmooth <= 0:
            raise ValueError(
                'the star model requires positive fwhm_smooth'
            )

    def run_admom(self, fitter, epochs, nband, guess_gmix, Tsmooth):
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
        for i in range(fitter.maxiter):
            numiter = i + 1

            sums = fitter._accumulate(epochs, Sigma, v0, u0)

            if sums[5] <= 0:
                flags = ngmix.flags.NONPOS_FLUX
                break

            if fitter.fixcen:
                break

            finv = 1.0 / sums[5]
            dv = sums[0] * finv
            du = sums[1] * finv
            v0 += dv
            u0 += du

            if (abs(v0 - vorig) > fitter.shiftmax
                    or abs(u0 - uorig) > fitter.shiftmax):
                flags = ngmix.flags.CEN_SHIFT
                break

            if abs(dv) < fitter.cen_tol and abs(du) < fitter.cen_tol:
                break
        else:
            flags = ngmix.flags.MAXITER

        model_state = {
            'type': 'star',
            'cov_sm': np.diag([Tsmooth / 2, Tsmooth / 2]),
            'F': np.ones(nband),
        }
        return fitter._get_result(
            epochs=epochs, nband=nband, flags=flags, numiter=numiter,
            Sigma=Sigma, v0=v0, u0=u0, Tsmooth=Tsmooth,
            model_state=model_state,
        )

    def shape_state(self, model_state, Sigma, Tsmooth):
        """
        a delta function has no size or shape
        """
        return 0.0, 0.0, 0.0, False

    def set_T_err(self, res, model_state, fam_cov, sums, sums_cov):
        """
        T is fixed at zero by the model, not measured
        """
        pass

    def set_shape(
        self, res, model_state, shape_ok, M1, M2, Tgal, Sigma,
        fam_cov, sums, sums_cov,
    ):
        """
        a delta function has no shape; this is by construction, not
        a failure, so the overall flags stay clean, but e_flags
        still marks the ellipticities unusable
        """
        res['e_flags'] |= ngmix.flags.NONPOS_SIZE

    def flux_results(
        self, state, epochs, nband, Sigma, Tsmooth, sums, cov,
        fsums, fvars, fmcovs, wsums,
    ):
        """
        the per-band fluxes normalized by the unit flux model
        predictions.  The weight is frozen for stars, so the fixed
        weight variance is exact and the bands are independent
        """
        upred = self._unit_flux_preds(
            state, epochs, nband, Sigma, Tsmooth,
        )
        fluxes = fsums / upred
        rawvars = fvars
        fcov_raw = np.diag(fvars)
        flux_vars = rawvars / upred ** 2
        uinv = 1.0 / upred
        flux_cov = fcov_raw * np.outer(uinv, uinv)
        return fluxes, flux_vars, None, flux_cov
