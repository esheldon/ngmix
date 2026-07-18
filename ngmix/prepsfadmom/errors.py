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

from .models import model_ksums

__all__ = ['flux_var_delta', 'model_sandwich']


def flux_var_delta(Sigma, sums, cov, fsums, fvars, fmcovs):
    """
    variance of the raw per-band flux sums, including the first order
    response of the adaptive weight to the noise (the delta method)

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
    The variance of the per band flux sums including the weight
    response; scale by the same normalization as the raw flux sums.
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
    return (
        fvars
        + r ** 2 * (cff + bcb - 2 * (b @ cmf))
        + 2 * r * (fmcovs @ b - fvars)
    )


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
    every epoch and band
    """
    if model_type == 'exp':
        state = {'type': 'exp', 'cov': Sfam, 'F': np.ones(1)}
    else:
        # single gaussian family, used to validate against the
        # analytic gauss delta method
        state = {
            'type': 'gauss',
            'cov_sm': Sfam + np.diag([Tsmooth / 2, Tsmooth / 2]),
            'F': np.ones(1),
        }
    s = model_ksums(state, 0, 0.0, 0.0, Sigma, 1.0, Tsmooth)
    return s[2:5] / s[5], np.log(s[5])


def model_sandwich(
    model_type, Sfam, Sigma, Tsmooth, sums, cov, fsums, fvars, fmcovs,
):
    """
    first order (sandwich) errors for the moment matched model fits

    The estimating equations are the weight condition
    M_meas(Sigma) = Sigma/2, the family condition
    M_pred(Sigma, Sfam) = M_meas, and per band F_b N_b = fs_b with
    N_b the unit flux model prediction.  Evaluated with the data
    replaced by the converged model, the Jacobian is block
    triangular: the family and flux conditions lose their explicit
    weight dependence, leaving

        dSfam = B^-1 dM,    B = dM_pred/dSfam
        dF_b/F_b = dfs_b/fs_b - (dln N/dSfam) . dSfam

    where dM is the noise fluctuation of the measured moment ratios.
    B and the normalization derivative are evaluated in closed form
    (by central differences on the analytic predictions), so unlike
    the gauss delta method there is no gaussian approximation beyond
    replacing the data by the converged model; for a single gaussian
    family this reduces exactly to flux_var_delta.

    Parameters
    ----------
    model_type: str
        'exp', or 'gauss' for validation
    Sfam: (2, 2) array
        The converged family covariance
    Sigma: (2, 2) array
        The converged weight covariance
    Tsmooth: float
        The smoothing T
    sums, cov, fsums, fvars, fmcovs:
        As for flux_var_delta

    Returns
    -------
    var_raw: array of size nband
        The variance of the per band flux sums including the family
        response; scale by the same normalization as the raw sums
    fam_cov: (3, 3) array
        The covariance of the family (M1, M2, T) parameters
    """
    Tw = Sigma[0, 0] + Sigma[1, 1]
    h = 1.0e-6 * Tw
    fam0 = np.array([
        Sfam[1, 1] - Sfam[0, 0], 2 * Sfam[0, 1],
        Sfam[0, 0] + Sfam[1, 1],
    ])

    B = np.zeros((3, 3))
    dlnu = np.zeros(3)
    for i in range(3):
        famp = fam0.copy()
        famm = fam0.copy()
        famp[i] += h
        famm[i] -= h
        rp, lp = _model_pred_mbasis(
            model_type, _mbasis_cov(*famp), Sigma, Tsmooth,
        )
        rm, lm = _model_pred_mbasis(
            model_type, _mbasis_cov(*famm), Sigma, Tsmooth,
        )
        B[:, i] = (rp - rm) / (2 * h)
        dlnu[i] = (lp - lm) / (2 * h)

    # the noise fluctuation of the measured ratios,
    # dM = (dS_M - mvec dS_F) / S_F, in terms of the raw sums
    sfj = sums[5]
    mvec = np.array([
        Sigma[1, 1] - Sigma[0, 0], 2 * Sigma[0, 1], Tw,
    ]) / 2
    css = cov[2:5, 2:5]
    csf = cov[2:5, 5]
    cff = cov[5, 5]
    cmm = (
        css - np.outer(mvec, csf) - np.outer(csf, mvec)
        + np.outer(mvec, mvec) * cff
    ) / sfj ** 2

    Binv = np.linalg.inv(B)
    fam_cov = Binv @ cmm @ Binv.T
    a = np.linalg.solve(B.T, dlnu)

    # dF_b is proportional to dS_Fb - fs_b a . dM; the band flux
    # sums covary only with their own epochs' part of the joint sums
    aca = a @ cmm @ a
    var_raw = np.zeros(fsums.size)
    for band in range(fsums.size):
        acmf = (a @ fmcovs[band] - (a @ mvec) * fvars[band]) / sfj
        var_raw[band] = (
            fvars[band] + fsums[band] ** 2 * aca
            - 2 * fsums[band] * acmf
        )
    return var_raw, fam_cov
