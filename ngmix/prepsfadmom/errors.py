"""
first order error propagation for the pre-PSF adaptive moments fits

The fits are M-estimators: the parameters solve moment conditions
that are ratios of weighted moment sums, linear in the data.  The
covariances here propagate the noise of those sums through the fixed
point conditions (the delta method / sandwich covariance), so they
include the response of the adaptively fitted weight to the noise,
not just the fixed weight term.

flux_var_delta evaluates the response analytically for the gaussian
implied by the converged weight (the model='gauss' fitter and the
deblender's gauss objects); model_sandwich evaluates the full
sandwich for the moment matched model families with closed form
model derivatives, and reduces exactly to flux_var_delta for a
single gaussian family.
"""
import numpy as np
from numba import njit

from .models import model_ksums, get_profile_comps
from .models_nb import gauss_comps_ksums

__all__ = [
    'flux_cov_delta',
    'joint_flux_s2n',
]


def joint_flux_s2n(F, fcov):
    """
    the covariance-aware total flux s/n sqrt(F^T C^-1 F) from the
    fluxes and their cross-band covariance (the Wald significance
    of the flux vector).  The statistic is invariant to per-band
    rescaling, so raw flux sums with the raw covariance equal
    physical fluxes with the physical covariance.  Returns None
    when the covariance is not positive definite or the form is
    not finite; the caller falls back to the independent-band
    quadrature sum

    Parameters
    ----------
    F: (nband,) array
        The fluxes (or raw flux sums)
    fcov: (nband, nband) array
        Their covariance, in matching units

    Returns
    -------
    float or None
    """
    try:
        L = np.linalg.cholesky(fcov)
    except np.linalg.LinAlgError:
        return None
    z = np.linalg.solve(L, F)
    q = z @ z
    if not np.isfinite(q):
        return None
    return np.sqrt(q)


def flux_cov_delta(Sigma, sums, cov, fsums, fvars, fmcovs):
    """
    full cross-band covariance of the raw per band flux sums,
    including the first order response of the adaptive weight to
    the noise (the delta method); the diagonal is flux_var_delta.

    The fluctuation dF_b is proportional to
    dS_Fb - r_b dS_F + r_b b . dS_M (see flux_var_delta): the
    joint flux and moment sum responses are shared by all bands,
    and only the band's own flux sum dS_Fb is band-specific, so
    the cross-band covariance assembles in closed form from the
    same scalars as the variances -- the outer products of the
    band shares with the shared response, plus each band's own
    cross terms.  No inversions beyond the 2x2 weight

    Parameters
    ----------
    As for flux_var_delta

    Returns
    -------
    (nband, nband) covariance of the raw per band flux sums;
    scale by the same normalization as the raw flux sums
    """
    Winv = np.linalg.inv(Sigma)
    # tr(Winv dS_M) = b . dS in the (M1, M2, T) sums basis
    b = np.array([
        0.5 * (Winv[1, 1] - Winv[0, 0]),
        Winv[0, 1],
        0.5 * (Winv[0, 0] + Winv[1, 1]),
    ])

    cmm = cov[2:5, 2:5]
    cmf = cov[2:5, 5]
    cff = cov[5, 5]

    # dF_b is proportional to dS_Fb - r dS_F + r b . dS_M with
    # r = S_Fb / S_F the band share of the joint flux sum
    r = fsums / sums[5]
    bcb = b @ cmm @ b
    s = cff + bcb - 2 * (b @ cmf)
    g = fmcovs @ b - fvars

    fcov = np.outer(r, r) * s + np.outer(r, g) + np.outer(g, r)
    # the diagonal via the original variance expression, keeping
    # the per-band variances bitwise identical to flux_var_delta
    # as it stood before the cross-band assembly
    idx = np.arange(fsums.size)
    fcov[idx, idx] = fvars + r ** 2 * s + 2 * r * g
    return fcov


def _mbasis_cov(M1, M2, T):
    """covariance matrix from (M1, M2, T) moment components"""
    return 0.5 * np.array([
        [T - M1, M2],
        [M2, T + M1],
    ])


def _model_pred_mbasis(model_type, Sfam, Sigma, Tsmooth):
    """
    the predicted weighted moment ratios (M1, M2, T) and the log of
    the unit flux prediction for a model family covariance, at unit
    detAtinv; the ratios and the log derivative are the same for
    every epoch and band.  model_type is a family name or a model
    spec dict (required for 'bdf', which needs fracdev and TdByTe)
    """
    if isinstance(model_type, dict):
        state = dict(model_type)
        state['cov'] = Sfam
        state['F'] = np.ones(1)
    elif model_type in ('exp', 'dev'):
        state = {'type': model_type, 'cov': Sfam, 'F': np.ones(1)}
    else:
        # single gaussian family, used to validate against the
        # analytic gauss delta method
        state = {
            'type': 'gauss',
            'cov_sm': Sfam + np.diag([Tsmooth / 2, Tsmooth / 2]),
            'F': np.ones(1),
        }
    s = model_ksums(state, 0, 0.0, 0.0, Sigma, 1.0, Tsmooth)
    if not s[5] > 0:
        # a state at or past the validity boundary (e.g. a
        # perturbed family covariance whose smoothed components
        # are indefinite) has no usable prediction; nan compares
        # false so it lands here too
        return None, None
    return s[2:5] / s[5], np.log(s[5])


_FAMILY_COMP_CACHE = {}


def _family_comp_arrays(model_type):
    """
    the (fracs, cT) component arrays for the plain string families,
    for the compiled sandwich derivative path; None for model spec
    dicts (bdf), which keep the general path.  A gauss family is the
    single component (1, 1): with cT = 1 the smoothed component
    covariance cT * Sfam + Tsmooth/2 reproduces the gauss branch of
    _model_pred_mbasis exactly (multiplication by 1 is exact)
    """
    if not isinstance(model_type, str):
        return None
    hit = _FAMILY_COMP_CACHE.get(model_type)
    if hit is None:
        if model_type == 'gauss':
            hit = (np.ones(1), np.ones(1))
        else:
            comps = get_profile_comps(model_type)
            hit = (
                np.array([c[0] for c in comps]),
                np.array([c[1] for c in comps]),
            )
        _FAMILY_COMP_CACHE[model_type] = hit
    return hit


@njit
def _sandwich_preds_nb(
    fracs, cts, fam0, h, sw00, sw01, sw11, Tsmooth, preds,
):
    """
    the six perturbed model predictions of the model_sandwich
    central differences, fused: per (axis i, sign si) the family
    covariance from the perturbed (M1, M2, T), the smoothed
    components, and the closed-form weighted sums.  Fills
    preds[i, si] = (M1, M2, T ratios, S_F); the logs and the
    difference quotients stay in the caller so the libm calls and
    the arithmetic order match the general path exactly.  Returns 0
    when any prediction is invalid (S_F not positive)
    """
    n = fracs.size
    smooth = Tsmooth / 2
    So00 = np.zeros(n)
    So01 = np.zeros(n)
    So11 = np.zeros(n)
    dvz = np.zeros(n)
    duz = np.zeros(n)
    sums = np.zeros(6)
    for i in range(3):
        for si in range(2):
            m1 = fam0[0]
            m2 = fam0[1]
            tt = fam0[2]
            d = h if si == 0 else -h
            if i == 0:
                m1 = m1 + d
            elif i == 1:
                m2 = m2 + d
            else:
                tt = tt + d
            sf00 = 0.5 * (tt - m1)
            sf01 = 0.5 * m2
            sf11 = 0.5 * (tt + m1)
            for k in range(n):
                cT = cts[k]
                So00[k] = cT * sf00 + smooth
                So01[k] = cT * sf01
                So11[k] = cT * sf11 + smooth
            for k in range(6):
                sums[k] = 0.0
            gauss_comps_ksums(
                fracs, So00, So01, So11, dvz, duz,
                sw00, sw01, sw11, 1.0, sums,
            )
            if not sums[5] > 0:
                return 0
            preds[i, si, 0] = sums[2] / sums[5]
            preds[i, si, 1] = sums[3] / sums[5]
            preds[i, si, 2] = sums[4] / sums[5]
            preds[i, si, 3] = sums[5]
    return 1
