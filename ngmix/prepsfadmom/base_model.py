"""
base class for the pre-PSF adaptive moments model objects; see
model.py for the overview and get_padmom_model for the accepted
model specifications
"""
__all__ = ['PAdmomModel']

import numpy as np

import ngmix.flags

from .models import model_ksums


class PAdmomModel(object):
    """
    base class for the pre-PSF adaptive moments model objects

    Subclasses own the model specific pieces of the fit: the
    adaptive moments driver (run_admom), the size and shape
    extraction from the converged state (shape_state, set_T_err,
    set_shape), the flux normalization and error sandwich
    (flux_results), and any per-model validation (validate_fit).

    Instances are immutable configuration; all per-run state lives
    in the model state dict created by run_admom
    """

    name = None
    supports_full_errors = False

    def validate_fit(self, Tsmooth):
        """
        validate the fit configuration; called by the fitter after
        the smoothing is chosen
        """
        pass

    def run_admom(self, fitter, epochs, nband, guess_gmix, Tsmooth):
        """
        run the adaptive moments iteration and return the result
        dict; fitter supplies the model independent machinery
        (_measure_step, _accumulate, _get_result) and the iteration
        controls (maxiter, etol, cen_tol, shiftmax, fixcen)
        """
        raise NotImplementedError('use a concrete model class')

    def set_shape(
        self, res, model_state, shape_ok, M1, M2, Tgal, Sigma,
        fam_cov, sums, sums_cov,
    ):
        """
        set the ellipticities and their errors in the result, or the
        flag bits recording why they are unusable
        """
        if shape_ok:
            res['e1'] = M1 / Tgal
            res['e2'] = M2 / Tgal
            res['e'] = np.array([res['e1'], res['e2']])

            e1err, e2err, e12cov = self._shape_errors(
                res, fam_cov, Sigma, sums, sums_cov, Tgal,
            )

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

    def _unit_flux_preds(self, state, epochs, nband, Sigma, Tsmooth):
        """
        the accumulated unit flux model predictions per band, with
        the converged weight; the flux normalization for the model
        fits
        """
        upred = np.zeros(nband)
        for epoch in epochs:
            fac = epoch['weight'] * epoch['detAtinv']
            upred[epoch['band']] += fac * model_ksums(
                state, epoch['band'], 0.0, 0.0, Sigma,
                epoch['detAtinv'], Tsmooth,
            )[5]
        return upred
