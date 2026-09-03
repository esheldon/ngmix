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
    the noise (the delta method).

    docs for flux_var_delta (diagonal of flux_cov_delta)
    ----------------------------------------------------

    The converged weight satisfies the fixed point conditions
    M(Sigma) = Sigma / 2, where M is the measured weighted covariance,
    a ratio of the joint moment sums.  By the implicit function
    theorem the weight fluctuation is dSigma = -J^-1 dM with
    J = dM/dSigma - 1/2.  For the gaussian implied by the converged
    weight, dM/dSigma = 1/4 exactly at the fixed point, for any
    ellipticity and smoothing, so dSigma = 4 dM.  The flux
    normalization (proportional to sqrt(det Sigma)) and the kernel
    response of the flux sum combine to dln F = tr(Sigma^-1 dM), and
    for a single band the fixed weight flux fluctuation cancels
    exactly, leaving

        dF / F = tr(Sigma^-1 dS_M) / S_F

    in terms of the second moment sum fluctuations alone.  For a
    round gaussian with matched weight and white noise this doubles
    the fixed weight variance.  With multiple bands the band flux sum
    and the joint conditions are distinct and their cross covariances
    enter; the band flux sums covary only with their own epochs'
    contribution to the joint sums.

    docs for flux_cov_delta (this function)
    ---------------------------------------

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
    Sigma: 2x2 array
        The converged weight covariance
    sums: array of size 6
        The accumulated joint sums
    cov: array (6, 6)
        The covariance of the joint sums
    fsums: array of size nband
        The accumulated per band flux sums
    fvars: array of size nband
        The variances of the per band flux sums
    fmcovs: array (nband, 3)
        The covariance of the joint (M1, M2, T) sums with the per
        band flux sums

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
