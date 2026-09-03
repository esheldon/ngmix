"""
numba kernel for the closed-form model sums

The weighted moment sums of a set of gaussian components under a
gaussian weight are closed form via the product-gaussian identities
(see models.py).  The kernel evaluates all components in one call,
which matters in deblending where the neighbor subtraction evaluates
O(ngroup^2) models per sweep, each with several components.
"""
import numpy as np
from numba import njit

# a total covariance with det <= DET_REL_TOL * C00 * C11 is treated as
# singular: the determinant is a difference of two products and at
# that relative size it is rounding noise (a runaway weight with
# entries of 1e20 produced determinants that were exact multiples of
# 2**65, including zero)
DET_REL_TOL = 1.0e-12


@njit
def gauss_comps_ksums(
    F, So00, So01, So11, dv, du, sw00, sw01, sw11, detAtinv, sums,
):
    """
    weighted moment sums of a set of gaussian components

    Non positive definite total covariances produce nan sums, which
    callers treat as a failed evaluation.  This includes a total
    covariance that is singular or numerically singular (see
    DET_REL_TOL); numba raises ZeroDivisionError on 1/0, so the
    check is explicit rather than left to the arithmetic.

    Parameters
    ----------
    F: array
        per component total fluxes
    So00, So01, So11: arrays
        per component covariances in the smoothed plane
    dv, du: arrays
        per component offsets of the component center from the
        weight center
    sw00, sw01, sw11: float
        the weight covariance
    detAtinv: float
        the k-space area factor
    sums: array of size 6
        output [v, u, M1, M2, T, flux] sums, overwritten
    """
    for k in range(6):
        sums[k] = 0.0

    nrm = 2 * np.pi * detAtinv

    for k in range(F.size):
        C00 = sw00 + So00[k]
        C01 = sw01 + So01[k]
        C11 = sw11 + So11[k]
        det = C00 * C11 - C01 * C01

        if not (C00 > 0.0 and det > DET_REL_TOL * C00 * C11):
            for j in range(6):
                sums[j] = np.nan
            return

        idet = 1.0 / det
        Ci00 = C11 * idet
        Ci01 = -C01 * idet
        Ci11 = C00 * idet

        Cd0 = Ci00 * dv[k] + Ci01 * du[k]
        Cd1 = Ci01 * dv[k] + Ci11 * du[k]

        sflux = F[k] * np.exp(
            -0.5 * (dv[k] * Cd0 + du[k] * Cd1)
        ) / (nrm * np.sqrt(det))

        mu0 = sw00 * Cd0 + sw01 * Cd1
        mu1 = sw01 * Cd0 + sw11 * Cd1

        # Sp = Sw C^-1 So
        A00 = sw00 * Ci00 + sw01 * Ci01
        A01 = sw00 * Ci01 + sw01 * Ci11
        A10 = sw01 * Ci00 + sw11 * Ci01
        A11 = sw01 * Ci01 + sw11 * Ci11
        Sp00 = A00 * So00[k] + A01 * So01[k]
        Sp01 = A00 * So01[k] + A01 * So11[k]
        Sp11 = A10 * So01[k] + A11 * So11[k]

        vv = Sp00 + mu0 * mu0
        vu = Sp01 + mu0 * mu1
        uu = Sp11 + mu1 * mu1

        sums[0] += sflux * mu0
        sums[1] += sflux * mu1
        sums[2] += sflux * (uu - vv)
        sums[3] += sflux * 2 * vu
        sums[4] += sflux * (uu + vv)
        sums[5] += sflux


def model_comps(model, Tsmooth):
    """
    the gaussian components of a model in the smoothed plane

    Parameters
    ----------
    model: dict
        model dict, see the module docstring
    Tsmooth: float
        T of the gaussian smoothing

    Returns
    -------
    fracs, So00, So01, So11: arrays
        per component flux fractions and covariances in the smoothed
        plane
    """
    # TODO: Turn back on for other models
    # if model['type'] in ('gauss', 'star'):
    if model['type'] in ('gauss',):
        c = model['cov_sm']
        return (
            np.ones(1),
            np.array([c[0, 0]]),
            np.array([c[0, 1]]),
            np.array([c[1, 1]]),
        )
    # TODO: Turn back on for other models
    # elif model['type'] in ('exp', 'dev', 'bdf'):
    #     if 'cov' in model:
    #         Sfam = model['cov']
    #     else:
    #         Sfam = cov_from_e(model['e1'], model['e2'], model['T'])
    #     smooth = Tsmooth / 2
    #     comps = _mixture_comps(model)
    #     n = len(comps)
    #     fracs = np.zeros(n)
    #     So00 = np.zeros(n)
    #     So01 = np.zeros(n)
    #     So11 = np.zeros(n)
    #     for k, (frac, cT) in enumerate(comps):
    #         fracs[k] = frac
    #         So00[k] = cT * Sfam[0, 0] + smooth
    #         So01[k] = cT * Sfam[0, 1]
    #         So11[k] = cT * Sfam[1, 1] + smooth
    #     return fracs, So00, So01, So11
    else:
        raise ValueError(f"bad model type: '{model['type']}'")


def model_ksums(model, band, dv, du, Sw, detAtinv, Tsmooth):
    """
    closed-form weighted moment sums of an object model, dispatching
    on the model type, evaluated via the numba kernel
    """
    from .models_nb import gauss_comps_ksums

    fracs, So00, So01, So11 = model_comps(model, Tsmooth)
    F = model['F'][band] * fracs
    n = F.size
    sums = np.zeros(6)
    gauss_comps_ksums(
        F, So00, So01, So11,
        np.full(n, float(dv)), np.full(n, float(du)),
        Sw[0, 0], Sw[0, 1], Sw[1, 1], float(detAtinv), sums,
    )
    return sums
