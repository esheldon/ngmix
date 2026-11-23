"""
first order error propagation for the pre-PSF adaptive moments fits

The fits are M-estimators: the parameters solve moment conditions
that are ratios of weighted moment sums, linear in the data.  The
covariances here propagate the noise of those sums through the fixed
point conditions using the "sandwich" covariance, the standard first
order covariance of an estimator theta defined by estimating
equations g(theta; data) = 0:

    Cov(theta) = J^-1 Cov(g) J^-T,    J = dg/dtheta

(the noise of the conditions bracketed by the inverse response,
hence the name; Huber 1967; White 1982; see Stefanski & Boos 2002
for the general recipe).  Because the conditions include the weight
fixed point, these covariances carry the response of the adaptive
weight (Bernstein & Jarvis 2002; Hirata & Seljak 2003) to the
noise, not just the fixed weight term.

flux_var_delta and flux_cov_delta evaluate the response analytically
for the gaussian implied by the converged weight (the model='gauss'
fitter and the deblender's gauss objects); model_sandwich evaluates
the full sandwich for the moment matched model families with closed
form model derivatives, reducing exactly to flux_var_delta for a
single gaussian family; joint_flux_s2n is the covariance-aware total
flux significance.  The sandwich formalism is standard; the closed
forms here (the exact gaussian fixed point response and the
cross-band assembly) were derived for this code, not taken from the
literature.

References
----------
Huber (1967), "The behavior of maximum likelihood estimates under
    nonstandard conditions", Proc. Fifth Berkeley Symp. Math.
    Statist. Prob. 1, 221
White (1982), "Maximum likelihood estimation of misspecified
    models", Econometrica 50, 1
Stefanski & Boos (2002), "The calculus of M-estimation", Amer.
    Statist. 56, 29
Bernstein & Jarvis (2002), "Shapes and shears, stars and smears:
    optimal measurements for weak lensing", AJ 123, 583
Hirata & Seljak (2003), "Shear calibration biases in weak-lensing
    surveys", MNRAS 343, 459
Wald (1943), "Tests of statistical hypotheses concerning several
    parameters when the number of observations is large", Trans.
    Amer. Math. Soc. 54, 426

Research for the errors was aided significantly by Claude Code
"""
import numpy as np
from numba import njit

from .models import get_profile_comps
from .models_nb import gauss_comps_ksums

__all__ = [
    'flux_var_delta', 'flux_cov_delta', 'model_sandwich',
    'joint_flux_s2n',
]


def joint_flux_s2n(F, fcov):
    """
    the covariance-aware total flux s/n sqrt(F^T C^-1 F)

    The Wald significance (Wald 1943) of the flux vector given its
    cross-band covariance.  The statistic is invariant to per-band
    rescaling, so raw flux sums with the raw covariance give the
    same result as physical fluxes with the physical covariance.

    Parameters
    ----------
    F: (nband,) array
        The fluxes (or raw flux sums)
    fcov: (nband, nband) array
        Their covariance, in matching units

    Returns
    -------
    s2n: float or None
        The joint s/n, or None when the covariance is not positive
        definite or the form is not finite; callers fall back to
        the independent-band quadrature sum
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


def flux_var_delta(wt_cov, sums, cov, fsums, fvars, fmcovs):
    """
    variance of the raw per-band flux sums, including the adaptive
    weight response; the diagonal of flux_cov_delta

    Notation: W = wt_cov is the weight covariance and S the
    covariance of the object in the smoothed plane (cov_sm in
    models.py).  S_F is the joint flux sum and S_M the 2x2 matrix
    of joint second moment sums, so the measured weighted
    covariance is the ratio M = S_M / S_F.  d denotes a noise
    fluctuation.

    The fixed point.  The converged weight satisfies the adaptive
    moments condition M(W) = W / 2 (Bernstein & Jarvis 2002; Hirata
    & Seljak 2003).  Noise shifts M at fixed weight by dM, and the
    weight moves by dW to restore the condition:

        dM + (dM/dW) dW = dW / 2

    which is the implicit function theorem, dW = -J^-1 dM with
    J = dM/dW - 1/2.

    The response dM/dW.  It is evaluated for the gaussian implied
    by the converged weight.  For a gaussian object the weighted
    covariance is that of the product of the weight and object
    gaussians (see models.py),

        M^-1 = W^-1 + S^-1

    The fixed point M = W / 2 then requires S = W: the weight
    matches the object.  Varying W at fixed S,

        d(M^-1) = -W^-1 dW W^-1
        dM = -M d(M^-1) M = M W^-1 dW W^-1 M = dW / 4

    using M W^-1 = 1/2 at the fixed point.  This is a matrix
    identity, so it holds for any ellipticity, and the smoothing is
    part of S, so for any smoothing (the identity was derived for
    this code).  Thus J = 1/4 - 1/2 = -1/4 and

        dW = 4 dM

    The flux.  The flux sum of a gaussian object is the overlap
    integral S_F = F / (2 pi sqrt(det(W + S))) (see models.py), so
    the flux of the gaussian implied by the weight, S = W, is
    F = 4 pi sqrt(det W) S_F.  Its fluctuation has three parts: the
    noise in S_F at fixed weight, the change of S_F as the weight
    moves over the fixed data, and the change of the normalization
    sqrt(det W):

        dln F = dS_F / S_F
                - tr((W + S)^-1 dW) / 2
                + tr(W^-1 dW) / 2

    With S = W the two weight terms combine to tr(W^-1 dW) / 4,
    which is tr(W^-1 dM) by dW = 4 dM.  The noise in the ratio
    M = S_M / S_F is dM = (dS_M - M dS_F) / S_F, and
    tr(W^-1 M) = 1 at the fixed point, so

        dln F = dS_F / S_F + tr(W^-1 dS_M) / S_F - dS_F / S_F

    For a single band the fixed weight flux fluctuation cancels
    exactly, leaving

        dF / F = tr(W^-1 dS_M) / S_F

    in terms of the second moment sum fluctuations alone.  For a
    round gaussian with matched weight and white noise this doubles
    the fixed weight variance.  With multiple bands the band flux sum
    and the joint conditions are distinct and their cross covariances
    enter; the band flux sums covary only with their own epochs'
    contribution to the joint sums (see flux_cov_delta).

    Parameters
    ----------
    wt_cov: 2x2 array
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
    var: array of size nband
        The variance of the per band flux sums including the
        weight response; scale by the same normalization as the
        raw flux sums
    """
    return np.diag(flux_cov_delta(
        wt_cov, sums, cov, fsums, fvars, fmcovs,
    )).copy()


def flux_cov_delta(wt_cov, sums, cov, fsums, fvars, fmcovs):
    """
    full cross-band covariance of the raw per band flux sums,
    including the adaptive weight response; the diagonal is
    flux_var_delta

    Notation: admom_ksums accumulates the weighted moment sums
    over all bands and epochs (the joint sums); S_M denotes the
    triplet of second-moment sums (S_M1, S_M2, S_T), S_F the
    joint flux sum, and S_Fb band b's own flux sum.  M is the
    2x2 measured weighted covariance, built from the moment
    ratios S_M / S_F.  d denotes the noise fluctuation of a
    quantity: dS_M is the noise in the moment sums, dM the
    resulting noise in M.

    The band flux fluctuation follows from the weight response
    dln F = tr(wt_cov^-1 dM) derived in flux_var_delta by adding
    the band's own fixed-weight noise dS_Fb / S_Fb, expanding
    dM = (dS_M - M dS_F) / S_F, and using tr(wt_cov^-1 M) = 1 at
    the adaptive moments fixed point M = wt_cov / 2 (Bernstein &
    Jarvis 2002; Hirata & Seljak 2003):

        dF_b  propto  dS_Fb - r_b dS_F + r_b b . dS_M

    Here r_b = S_Fb / S_F, the fraction of the joint flux sum
    contributed by band b, scales the shared joint-sum noise
    down to the band.  b is the constant 3-vector

        b = ((Winv_uu - Winv_vv) / 2,
             Winv_vu,
             (Winv_vv + Winv_uu) / 2),    Winv = wt_cov^-1

    which writes the trace as a dot product over the (M1, M2, T)
    components: tr(wt_cov^-1 dS_M) = b . dS_M, with dS_M mapped
    to its 2x2 matrix form as in cov_from_mom.  The joint noises
    dS_F and dS_M are shared by all bands and only dS_Fb is
    band-specific, so the cross-band covariance assembles in
    closed form from the same scalars as the variances: the
    outer products of the band shares with the shared response,
    plus each band's own cross terms.  No inversions beyond the
    2x2 weight are needed.

    Parameters
    ----------
    wt_cov, sums, cov, fsums, fvars, fmcovs:
        As for flux_var_delta

    Returns
    -------
    fcov: (nband, nband) array
        The covariance of the raw per band flux sums; scale by
        the same normalization as the raw flux sums
    """
    Winv = np.linalg.inv(wt_cov)
    # tr(Winv dS_M) = b . dS in the (M1, M2, T) sums basis
    b = np.array([
        0.5 * (Winv[1, 1] - Winv[0, 0]),
        Winv[0, 1],
        0.5 * (Winv[0, 0] + Winv[1, 1]),
    ])

    # blocks of the joint-sum covariance: css = Cov(dS_M, dS_M),
    # csf = Cov(dS_M, dS_F) and cff = Var(dS_F)
    css = cov[2:5, 2:5]
    csf = cov[2:5, 5]
    cff = cov[5, 5]

    # dF_b propto dS_Fb + r_b (b . dS_M - dS_F), see the
    # docstring.  s is the variance of the shared response,
    #   s = Var(b . dS_M - dS_F)
    # and g_b its covariance with the band's own flux-sum noise,
    #   g_b = Cov(dS_Fb, b . dS_M - dS_F)
    # which uses Cov(dS_Fb, dS_F) = Var(dS_Fb) = fvars_b: the
    # band flux noise enters the joint flux sum exactly once
    r = fsums / sums[5]
    bcb = b @ css @ b
    s = cff + bcb - 2 * (b @ csf)
    g = fmcovs @ b - fvars

    fcov = np.outer(r, r) * s + np.outer(r, g) + np.outer(g, r)
    # the band flux-sum noises are independent across bands, so
    # their variances enter only on the diagonal: the outer
    # products above already carry the shared joint-sum noise and
    # the band cross terms for every pair, while
    # Cov(dS_Fbi, dS_Fbj) vanishes for i != j
    idx = np.arange(fsums.size)
    fcov[idx, idx] += fvars
    return fcov


_FAMILY_COMP_CACHE = {}


def _family_comp_arrays(model_type):
    """
    the (fracs, cT) component arrays for the model families

    Used by the compiled sandwich derivative path.  A gauss family
    is the single component (1, 1): with cT = 1 the smoothed
    component covariance is fam_cov + Tsmooth/2, a single gaussian
    (multiplication by 1 is exact)

    Parameters
    ----------
    model_type: str
        'exp', 'dev' or 'gauss'
    """
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
    central differences, fused into one compiled pass

    Per (axis i, sign si) this builds the family covariance from
    the perturbed (M1, M2, T), the smoothed components, and the
    closed-form weighted sums, filling
    preds[i, si] = (M1, M2, T ratios, S_F).  The logs and the
    difference quotients are taken by the caller.  Returns 1, or
    0 when any prediction is invalid (S_F not positive)

    Parameters
    ----------
    fracs: array
        Per component flux fractions of the family
    cts: array
        Per component T factors relative to the total T
    fam0: array of size 3
        The family (M1, M2, T) about which to perturb
    h: float
        The central difference step
    sw00, sw01, sw11: float
        The weight covariance
    Tsmooth: float
        The T of the gaussian smoothing
    preds: (3, 2, 4) array
        Output, indexed by perturbed axis, sign (0 for +h, 1 for -h)
        and the entries (M1, M2, T ratios, S_F)
    """
    n = fracs.size
    smooth = Tsmooth / 2
    cov_sm00 = np.zeros(n)
    cov_sm01 = np.zeros(n)
    cov_sm11 = np.zeros(n)
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
                cov_sm00[k] = cT * sf00 + smooth
                cov_sm01[k] = cT * sf01
                cov_sm11[k] = cT * sf11 + smooth
            for k in range(6):
                sums[k] = 0.0
            gauss_comps_ksums(
                fracs, cov_sm00, cov_sm01, cov_sm11, dvz, duz,
                sw00, sw01, sw11, 1.0, sums,
            )
            if not sums[5] > 0:
                return 0
            preds[i, si, 0] = sums[2] / sums[5]
            preds[i, si, 1] = sums[3] / sums[5]
            preds[i, si, 2] = sums[4] / sums[5]
            preds[i, si, 3] = sums[5]
    return 1


def model_sandwich(
    model_type, fam_cov, wt_cov, Tsmooth, sums, cov, fsums, fvars, fmcovs,
):
    """
    first order (sandwich) errors for the moment matched model fits

    The estimating equations are the weight condition
    M_meas(wt_cov) = wt_cov/2 (the adaptive moments fixed point;
    Bernstein & Jarvis 2002; Hirata & Seljak 2003), the family
    condition
    M_pred(wt_cov, fam_cov) = M_meas, and per band the flux
    condition F_b N_b = S_Fb, where S_Fb is the band's measured
    raw flux sum (the fsums argument) and N_b(wt_cov, fam_cov)
    is the model prediction of that sum at unit flux, so
    F_b = S_Fb / N_b.  The notation is that of flux_cov_delta:
    S_M, S_F and S_Fb are the joint moment sums, the joint flux
    sum and the band flux sum, and d denotes a noise fluctuation,
    so dM is the noise in the measured moment ratios M_meas and
    dS_Fb the noise in the band's flux sum.

    Evaluated with the data replaced by the converged model, the
    Jacobian is block triangular: the family and flux conditions
    lose their explicit weight dependence, leaving

        d(fam_cov) = B^-1 dM,    B = dM_pred/d(fam_cov)
        dF_b/F_b = dS_Fb/S_Fb - (dln N/d(fam_cov)) . d(fam_cov)

    the second line being the fluctuation of F_b = S_Fb / N_b:
    the direct flux-sum noise plus the normalization's response
    to the fitted family covariance.  B and the normalization
    derivative are evaluated by central differences on the closed
    form model predictions, so unlike the gauss delta method
    there is no gaussian approximation beyond replacing the data
    by the converged model; for a single gaussian family this
    reduces exactly to flux_var_delta.

    Parameters
    ----------
    model_type: str
        'exp' or 'dev', or 'gauss' for validation
    fam_cov: (2, 2) array
        The converged family covariance
    wt_cov: (2, 2) array
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
    fam_err_cov: (3, 3) array
        The covariance of the family (M1, M2, T) parameters
    fcov_raw: (nband, nband) array
        The full covariance of the per band flux sums, whose
        diagonal is var_raw.  The off diagonal is generated by the
        shared family response (the band sum noises are
        independent across bands) and is what honest color errors
        require: the independence combination of the per band
        errors over-predicts color variances by the shared term.
        Scale by the same normalization as var_raw

    Returns (None, None, None) when the sandwich cannot be
    evaluated (a model prediction at or past the validity
    boundary, or a singular response matrix); callers should fall
    back to the fixed weight variances and flag the structure
    errors
    """
    Tw = wt_cov[0, 0] + wt_cov[1, 1]
    h = 1.0e-6 * Tw
    fam0 = np.array([
        fam_cov[1, 1] - fam_cov[0, 0], 2 * fam_cov[0, 1],
        fam_cov[0, 0] + fam_cov[1, 1],
    ])

    B = np.zeros((3, 3))
    # dlnN: dln N/d(fam_cov), the same for every band
    dlnN = np.zeros(3)
    # the fused compiled evaluation of the six perturbed
    # predictions
    fracs, cts = _family_comp_arrays(model_type)
    preds = np.zeros((3, 2, 4))
    ok = _sandwich_preds_nb(
        fracs, cts, fam0, h,
        wt_cov[0, 0], wt_cov[0, 1], wt_cov[1, 1], Tsmooth, preds,
    )
    if not ok:
        return None, None, None
    for i in range(3):
        B[:, i] = (preds[i, 0, :3] - preds[i, 1, :3]) / (2 * h)
        dlnN[i] = (
            np.log(preds[i, 0, 3]) - np.log(preds[i, 1, 3])
        ) / (2 * h)

    # the noise fluctuation of the measured ratios,
    # dM = (dS_M - mvec dS_F) / S_F, in terms of the raw sums.
    # css, csf and cff are the blocks of the joint-sum covariance
    # as in flux_cov_delta, and cmm = Cov(dM, dM)
    sf = sums[5]
    # mvec: the measured moment ratios S_M / S_F in the
    # (M1, M2, T) basis.  At the converged weight these equal the
    # fixed point value wt_cov / 2, which is used here
    mvec = np.array([
        wt_cov[1, 1] - wt_cov[0, 0], 2 * wt_cov[0, 1], Tw,
    ]) / 2
    css = cov[2:5, 2:5]
    csf = cov[2:5, 5]
    cff = cov[5, 5]
    cmm = (
        css - np.outer(mvec, csf) - np.outer(csf, mvec)
        + np.outer(mvec, mvec) * cff
    ) / sf ** 2

    try:
        Binv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        return None, None, None
    # d(fam_cov) = B^-1 dM, so Cov(fam_cov) = B^-1 cmm B^-T
    fam_err_cov = Binv @ cmm @ Binv.T

    # the normalization response in terms of the moment noise:
    #   (dln N/d(fam_cov)) . d(fam_cov) = dlnN . B^-1 dM = a . dM
    # with a = B^-T dlnN
    a = np.linalg.solve(B.T, dlnN)

    # multiplying dF_b/F_b = dS_Fb/S_Fb - a . dM by S_Fb,
    #   dF_b  propto  dS_Fb - S_Fb a . dM
    # The family response a . dM is shared by all bands and only
    # dS_Fb is band-specific, so as in flux_cov_delta the
    # cross-band covariance assembles from
    #   aca    = Var(a . dM)
    #   acmf_b = Cov(dS_Fb, a . dM)
    #          = (a . Cov(dS_Fb, dS_M) - (a . mvec) Var(dS_Fb)) / S_F
    # where the second line expands dM = (dS_M - mvec dS_F) / S_F
    # and uses Cov(dS_Fb, dS_F) = Var(dS_Fb) = fvars_b: the band
    # flux noise enters the joint flux sum exactly once.  Then
    #   Cov(dF_b, dF_c) = S_Fb S_Fc aca - S_Fb acmf_c - acmf_b S_Fc
    #                     + delta_bc fvars_b
    # the last term on the diagonal only because the band flux-sum
    # noises are independent across bands
    aca = a @ cmm @ a
    acmf = np.array([
        (a @ fmcovs[band] - (a @ mvec) * fvars[band]) / sf
        for band in range(fsums.size)
    ])
    fcov_raw = (
        np.outer(fsums, fsums) * aca
        - np.outer(fsums, acmf) - np.outer(acmf, fsums)
        + np.diag(fvars)
    )
    var_raw = np.diag(fcov_raw).copy()
    return var_raw, fam_err_cov, fcov_raw
