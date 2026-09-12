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
single gaussian family; bdf_joint_sandwich extends it with the flux
split as an estimated parameter; joint_flux_s2n is the
covariance-aware total flux significance.  The sandwich formalism is
standard; the closed forms here (the exact gaussian fixed point
response, the cross-band assembly and the split cross terms) were
derived for this code, not taken from the literature.

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

from ..moments import cov_from_mom
from .models import model_ksums, get_profile_comps
from .models_nb import gauss_comps_ksums

__all__ = [
    'flux_var_delta', 'flux_cov_delta', 'model_sandwich',
    'bdf_joint_sandwich', 'joint_flux_s2n',
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

    The converged weight satisfies the adaptive moments fixed point
    conditions M(wt_cov) = wt_cov / 2 (Bernstein & Jarvis 2002; Hirata
    & Seljak 2003), where M is the measured weighted covariance, a
    ratio of the joint moment sums.  By the implicit function
    theorem the weight fluctuation is d(wt_cov) = -J^-1 dM with
    J = dM/d(wt_cov) - 1/2.  For the gaussian implied by the converged
    weight, dM/d(wt_cov) = 1/4 exactly at the fixed point, for any
    ellipticity and smoothing (an identity derived for this code),
    so d(wt_cov) = 4 dM.  The flux
    normalization (proportional to sqrt(det wt_cov)) and the kernel
    response of the flux sum combine to dln F = tr(wt_cov^-1 dM), and
    for a single band the fixed weight flux fluctuation cancels
    exactly, leaving

        dF / F = tr(wt_cov^-1 dS_M) / S_F

    in terms of the second moment sum fluctuations alone.  For a
    round gaussian with matched weight and white noise this doubles
    the fixed weight variance.  With multiple bands the band flux sum
    and the joint conditions are distinct and their cross covariances
    enter; the band flux sums covary only with their own epochs'
    contribution to the joint sums.

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

    # blocks of the joint-sum covariance: moment-moment,
    # moment-flux and flux-flux
    cmm = cov[2:5, 2:5]
    cmf = cov[2:5, 5]
    cff = cov[5, 5]

    # dF_b propto dS_Fb + r_b (b . dS_M - dS_F), see the
    # docstring.  s is the variance of the shared response,
    #   s = Var(b . dS_M - dS_F)
    # and g_b its covariance with the band's own flux-sum noise,
    #   g_b = Cov(dS_Fb, b . dS_M - dS_F)
    # which uses Cov(dS_Fb, dS_F) = Var(dS_Fb) = fvars_b: the
    # band flux noise enters the joint flux sum exactly once
    r = fsums / sums[5]
    bcb = b @ cmm @ b
    s = cff + bcb - 2 * (b @ cmf)
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


def _model_pred_mbasis(model_type, fam_cov, wt_cov, Tsmooth):
    """
    the predicted weighted moment ratios (M1, M2, T) and the log
    of the unit flux prediction for a model family covariance

    Evaluated at unit detAtinv; the ratios and the log derivative
    are the same for every epoch and band.  model_type is a family
    name or a model spec dict (required for 'bdf', which needs
    fracdev and TdByTe).  Returns (None, None) for a state with no
    usable prediction
    """
    if isinstance(model_type, dict):
        state = dict(model_type)
        state['cov'] = fam_cov
        state['F'] = np.ones(1)
    elif model_type in ('exp', 'dev'):
        state = {'type': model_type, 'cov': fam_cov, 'F': np.ones(1)}
    else:
        # single gaussian family, used to validate against the
        # analytic gauss delta method
        state = {
            'type': 'gauss',
            'cov_sm': fam_cov + np.diag([Tsmooth / 2, Tsmooth / 2]),
            'F': np.ones(1),
        }
    s = model_ksums(state, 0, 0.0, 0.0, wt_cov, 1.0, Tsmooth)
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
    the (fracs, cT) component arrays for the plain string families

    Used by the compiled sandwich derivative path; None for model
    spec dicts (bdf), which keep the general path.  A gauss family
    is the single component (1, 1): with cT = 1 the smoothed
    component covariance cT * fam_cov + Tsmooth/2 reproduces the
    gauss branch of _model_pred_mbasis exactly (multiplication by
    1 is exact)
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
    central differences, fused into one compiled pass

    Per (axis i, sign si) this builds the family covariance from
    the perturbed (M1, M2, T), the smoothed components, and the
    closed-form weighted sums, filling
    preds[i, si] = (M1, M2, T ratios, S_F).  The logs and the
    difference quotients stay in the caller so the libm calls and
    the arithmetic order match the general path exactly.  Returns
    1, or 0 when any prediction is invalid (S_F not positive)
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
    condition F_b N_b = fs_b, where fs_b is the band's measured
    raw flux sum (the fsums argument) and N_b(wt_cov, fam_cov)
    is the model prediction of that sum at unit flux, so
    F_b = fs_b / N_b.  As in flux_cov_delta, d denotes a noise
    fluctuation: dM is the noise in the measured moment ratios
    M_meas, dfs_b the noise in the band's flux sum.

    Evaluated with the data replaced by the converged model, the
    Jacobian is block triangular: the family and flux conditions
    lose their explicit weight dependence, leaving

        d(fam_cov) = B^-1 dM,    B = dM_pred/d(fam_cov)
        dF_b/F_b = dfs_b/fs_b - (dln N/d(fam_cov)) . d(fam_cov)

    the second line being the fluctuation of F_b = fs_b / N_b:
    the direct flux-sum noise plus the normalization's response
    to the fitted family covariance.  B and the normalization
    derivative are evaluated by central differences on the closed
    form model predictions, so unlike the gauss delta method
    there is no gaussian approximation beyond replacing the data
    by the converged model; for a single gaussian family this
    reduces exactly to flux_var_delta.

    Parameters
    ----------
    model_type: str or dict
        'exp' or 'dev', or 'gauss' for validation; a model spec
        dict for 'bdf' (the sandwich then treats fracdev and
        TdByTe as fixed, so the errors are conditional on the
        flux split)
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
    dlnu = np.zeros(3)
    comp_arrays = _family_comp_arrays(model_type)
    if comp_arrays is not None:
        # plain string family: the fused compiled evaluation of the
        # six perturbed predictions (identical arithmetic)
        fracs, cts = comp_arrays
        preds = np.zeros((3, 2, 4))
        ok = _sandwich_preds_nb(
            fracs, cts, fam0, h,
            wt_cov[0, 0], wt_cov[0, 1], wt_cov[1, 1], Tsmooth, preds,
        )
        if not ok:
            return None, None, None
        for i in range(3):
            B[:, i] = (preds[i, 0, :3] - preds[i, 1, :3]) / (2 * h)
            dlnu[i] = (
                np.log(preds[i, 0, 3]) - np.log(preds[i, 1, 3])
            ) / (2 * h)
    else:
        # model spec dict (bdf): the general path
        for i in range(3):
            famp = fam0.copy()
            famm = fam0.copy()
            famp[i] += h
            famm[i] -= h
            rp, lp = _model_pred_mbasis(
                model_type, cov_from_mom(*famp), wt_cov, Tsmooth,
            )
            rm, lm = _model_pred_mbasis(
                model_type, cov_from_mom(*famm), wt_cov, Tsmooth,
            )
            if rp is None or rm is None:
                return None, None, None
            B[:, i] = (rp - rm) / (2 * h)
            dlnu[i] = (lp - lm) / (2 * h)

    # the noise fluctuation of the measured ratios,
    # dM = (dS_M - mvec dS_F) / S_F, in terms of the raw sums
    sfj = sums[5]
    mvec = np.array([
        wt_cov[1, 1] - wt_cov[0, 0], 2 * wt_cov[0, 1], Tw,
    ]) / 2
    css = cov[2:5, 2:5]
    csf = cov[2:5, 5]
    cff = cov[5, 5]
    cmm = (
        css - np.outer(mvec, csf) - np.outer(csf, mvec)
        + np.outer(mvec, mvec) * cff
    ) / sfj ** 2

    try:
        Binv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        return None, None, None
    fam_err_cov = Binv @ cmm @ Binv.T
    a = np.linalg.solve(B.T, dlnu)

    # dF_b is proportional to dS_Fb - fs_b a . dM; the band flux
    # sums covary only with their own epochs' part of the joint
    # sums, and are independent across bands, so the full
    # cross-band covariance is generated by the shared family
    # response a . dM plus each band's own cross term
    aca = a @ cmm @ a
    acmf = np.array([
        (a @ fmcovs[band] - (a @ mvec) * fvars[band]) / sfj
        for band in range(fsums.size)
    ])
    fcov_raw = (
        np.outer(fsums, fsums) * aca
        - np.outer(fsums, acmf) - np.outer(acmf, fsums)
        + np.diag(fvars)
    )
    var_raw = np.diag(fcov_raw).copy()
    return var_raw, fam_err_cov, fcov_raw


def bdf_joint_sandwich(
    model_state, wt_cov, Tsmooth, sums, cov, fsums, fvars, fmcovs,
    split_grad, shrink_k, fd_var_data,
    eta_scov=None, eta_fcovs=None,
):
    """
    the sandwich for the bdf model with the flux split treated as
    an estimated parameter, removing the conditional-on-the-split
    underprediction of the errors.

    Notation follows flux_cov_delta and model_sandwich: d denotes
    a noise fluctuation and dM the noise in the measured moment
    ratios (M1, M2, T).  The estimating equations are the family
    condition M_pred(fam_cov, fd) = M_meas of model_sandwich plus
    the split condition fd = k g(fam_cov; data), with fd the dev
    flux fraction, k the (deterministic) shrinkage factor of the
    fitter's split prior and g the split estimator.
    Differentiating both at the model consistent point,

        B d(fam_cov) + c dfd = dM
        dfd = k (G . d(fam_cov) + eta)

    with B = dM_pred/d(fam_cov) and c = dM_pred/dfd (central
    differences of the closed form predictions, as in
    model_sandwich), G = dg/d(fam_cov) (supplied by the caller,
    since it depends on the estimator: the GLS in the
    single-object fitter, the two-aperture solve in the
    deblender), and eta the split estimator's own data noise,
    with conditional variance fd_var_data.  Substituting the
    second equation into the first gives the coupled response

        d(fam_cov) = Pinv (dM - k c eta),   P = B + k outer(c, G)
        dfd   = k (G . d(fam_cov) + eta)

    which reduces to model_sandwich at k = 0 (split frozen).  The
    flux rows follow model_sandwich, with the normalization now
    responding to both channels:
    dF_b/F_b = dfs_b/fs_b - dlnN . (d(fam_cov), dfd).

    The cross covariance of eta with the moment and flux sums is
    essential: both are linear functionals of the same noise and
    the T component is strongly anti-correlated with the split
    (the coupled loop amplifies each input several fold but the
    two amplified paths nearly cancel).  Without the cross terms
    the errors are overpredicted by factors of a few.  The caller
    supplies them via eta_scov/eta_fcovs; when omitted they are
    taken as zero, which should only be used as a fallback.

    Parameters
    ----------
    model_state: dict
        the converged bdf model state ('cov', 'fracdev', 'TdByTe')
    wt_cov, Tsmooth, sums, cov, fsums, fvars, fmcovs:
        as for model_sandwich
    split_grad: array of size 3
        d fd_gls / d (M1, M2, T) of the family covariance, at the
        model consistent point
    shrink_k: float
        the shrinkage factor (1 with no prior, 0 frozen)
    fd_var_data: float
        the conditional variance of the measured split
    eta_scov: array of size 3, optional
        Cov(eta, s[2:5]) of the accumulated (M1, M2, T) sums, in
        the same normalization as sums/cov
    eta_fcovs: array of size nband, optional
        Cov(eta, fs_band) of the per-band flux sums

    Returns
    -------
    var_raw: array of size nband
        The per band flux sum variances; scale as for
        model_sandwich
    fam_err_cov: (3, 3) array
        The covariance of the family (M1, M2, T) parameters
    fd_var: float
        The total variance of the split

    All three are None when the sandwich cannot be evaluated
    """
    fam_cov = model_state['cov']
    fd = model_state['fracdev']

    Tw = wt_cov[0, 0] + wt_cov[1, 1]
    h = 1.0e-6 * Tw
    fam0 = np.array([
        fam_cov[1, 1] - fam_cov[0, 0], 2 * fam_cov[0, 1],
        fam_cov[0, 0] + fam_cov[1, 1],
    ])

    def pred(famvec, fdval):
        state = dict(model_state)
        state['fracdev'] = fdval
        return _model_pred_mbasis(
            state, cov_from_mom(*famvec), wt_cov, Tsmooth,
        )

    # B, c: the predicted-ratio derivatives of the docstring, and
    # the flux normalization responses dlnu, dlnu_fd, by central
    # differences of the closed form predictions
    B = np.zeros((3, 3))
    dlnu = np.zeros(3)
    for i in range(3):
        famp = fam0.copy()
        famm = fam0.copy()
        famp[i] += h
        famm[i] -= h
        rp, lp = pred(famp, fd)
        rm, lm = pred(famm, fd)
        if rp is None or rm is None:
            return None, None, None
        B[:, i] = (rp - rm) / (2 * h)
        dlnu[i] = (lp - lm) / (2 * h)

    hfd = 1.0e-4
    rp, lp = pred(fam0, fd + hfd)
    rm, lm = pred(fam0, fd - hfd)
    if rp is None or rm is None:
        return None, None, None
    c = (rp - rm) / (2 * hfd)
    dlnu_fd = (lp - lm) / (2 * hfd)

    # Cov(dM): expand dM = (dS_M - mvec dS_F) / S_F as in
    # flux_cov_delta, with mvec the (M1, M2, T) components of the
    # fixed point M = wt_cov / 2
    sfj = sums[5]
    mvec = np.array([
        wt_cov[1, 1] - wt_cov[0, 0], 2 * wt_cov[0, 1], Tw,
    ]) / 2
    css = cov[2:5, 2:5]
    csf = cov[2:5, 5]
    cff = cov[5, 5]
    cmm = (
        css - np.outer(mvec, csf) - np.outer(csf, mvec)
        + np.outer(mvec, mvec) * cff
    ) / sfj ** 2

    # the coupled response operator of the docstring
    G = np.asarray(split_grad, dtype='f8')
    k = float(shrink_k)
    P = B + k * np.outer(c, G)
    try:
        Pinv = np.linalg.inv(P)
    except np.linalg.LinAlgError:
        return None, None, None

    # the coupled response written as linear maps on the two
    # noise inputs, [d(fam_cov); dfd] = LM dM + Le eta
    LM = np.zeros((4, 3))
    Le = np.zeros(4)
    LM[:3] = Pinv
    Le[:3] = -k * (Pinv @ c)
    LM[3] = k * (G @ Pinv)
    Le[3] = k * (1.0 - k * (G @ (Pinv @ c)))

    fdv = max(float(fd_var_data), 0.0)

    # the full input covariance of (dM, eta); the eta crosses come
    # from the same template-times-kernel overlap sums as the flux
    # covariances and are supplied by the caller
    if eta_fcovs is None:
        efc = np.zeros(fsums.size)
    else:
        efc = np.asarray(eta_fcovs, dtype='f8')
    if eta_scov is None:
        cme = np.zeros(3)
    else:
        # Cov(eta, dM) from the supplied sum crosses, using the
        # same expansion of dM; Cov(eta, dS_F) is the sum of the
        # band crosses since each band's flux noise enters the
        # joint flux sum exactly once
        ceta_f = efc.sum()
        cme = (np.asarray(eta_scov, dtype='f8') - mvec * ceta_f) / sfj

    Cx = np.zeros((4, 4))
    Cx[:3, :3] = cmm
    Cx[3, 3] = fdv
    Cx[:3, 3] = cme
    Cx[3, :3] = cme

    Amap = np.hstack([LM, Le[:, None]])
    joint = Amap @ Cx @ Amap.T
    fam_err_cov = joint[:3, :3]
    fd_var = joint[3, 3]

    # flux response: dF_b/F_b = dfs_b/fs_b - q . (d(fam_cov), dfd)
    q = np.concatenate([dlnu, [dlnu_fd]])
    a = q @ Amap     # coefficients on (dM, eta)
    u = a[:3]
    ue = a[3]
    aca = a @ Cx @ a

    # Var of the raw flux fluctuation dfs_b - fs_b a . (dM, eta):
    # fvars is the direct term, aca the amplified joint response,
    # and acmf = Cov(dfs_b, a . (dM, eta)) via the dM expansion
    # and Cov(dfs_b, dS_F) = Var(dfs_b), as in flux_cov_delta
    var_raw = np.zeros(fsums.size)
    for band in range(fsums.size):
        acmf = (
            (u @ fmcovs[band] - (u @ mvec) * fvars[band]) / sfj
            + ue * efc[band]
        )
        var_raw[band] = (
            fvars[band] + fsums[band] ** 2 * aca
            - 2 * fsums[band] * acmf
        )
    return var_raw, fam_err_cov, fd_var
