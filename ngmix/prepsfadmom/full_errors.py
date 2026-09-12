"""
Full (fixed-point) error propagation for the pre-PSF adaptive
moments machinery.

The weighted moment sums are linear in the prepped k-space image
with closed-form kernels (the admom_ksums accumulators), so their
exact covariance under the pixel noise follows from real-space
influence kernels (influence in the sense of Hampel 1974): each
sum's kernel is assembled on the k grid, carried to the image
plane through the stored image->kim transfer (epoch['ktransfer'])
and contracted with the per-pixel variance from the weight map.
Apodization enters exactly: the masked prep's image->kim map
factorizes as ktransfer(k) e^{-ik.x} m(x), so influence_kernels
applies the mask in pixel space, where it is diagonal.  The
kernel derivatives with respect to the weight covariance and
center are also closed form, allowing the data sums to be
linearized about a solution so that fixed-point Jacobians never
touch the data modes.  This construction, including the exact
apodization treatment, was derived for this code.

The building blocks are consumed by deblenders (kdeblend
full_errors); padmom_full_covariance assembles them into the full
covariance of the single-object fitter (full_errors=True).
Unlike the model_sandwich error path (see errors.py for the
sandwich definition and references), covariances built from them
make no model-consistency substitution, so they remain calibrated
under model mismatch, where the sandwich structure errors
under-predict (T by ~12 percent for real morphologies fit with
exp).

References
----------
Hampel (1974), "The influence curve and its role in robust
    estimation", JASA 69, 383

Research for the errors was aided significantly by Claude Code
"""
import numpy as np

from .prepsfadmom import get_phase_angles
from .errors import joint_flux_s2n

__all__ = [
    'moment_kernels', 'dsums_dtheta', 'influence_kernels',
    'sums_cov',
]

# the kernel weight support, matching FASTEXP_MAX_CHI2
WK_CHI2_MAX = 25.0


def kernel_ingredients(epoch, wt_cov, v0, u0):
    """
    the shared per-epoch pieces of the kernel expressions

    Exported for use in kdeblend.
    """
    alpha, beta = get_phase_angles(epoch, v0, u0)
    dim = epoch['dim']
    iy = epoch['iy'].astype(np.int64)
    ix = epoch['ix'].astype(np.int64)
    kv, ku = epoch['kv'], epoch['ku']
    wvv, wvu, wuu = wt_cov[0, 0], wt_cov[0, 1], wt_cov[1, 1]

    Sv = wvv * kv + wvu * ku
    Su = wvu * kv + wuu * ku
    chi2 = kv * Sv + ku * Su
    wk = np.where(
        chi2 < WK_CHI2_MAX, np.exp(-0.5 * chi2), 0.0,
    )
    yf = np.where(iy < (dim + 1) // 2, iy, iy - dim)
    xf = np.where(ix < (dim + 1) // 2, ix, ix - dim)
    phase = np.exp(
        2j * np.pi / dim * (alpha * yf + beta * xf),
    )
    base = wk * phase * epoch['df2']
    return Sv, Su, base, yf, xf, dim, (wvv, wvu, wuu)


def moment_kernels(epoch, wt_cov, v0, u0):
    """
    the (6, nmodes) complex moment kernels G with
    sums = Re(G @ kim)

    The admom_ksums accumulators in closed form.

    Parameters
    ----------
    epoch: dict
        A prep_epoch dict
    wt_cov: (2, 2) array
        The weight covariance
    v0, u0: float
        The center offset in the epoch's phase convention (as for
        get_phase_angles)

    Returns
    -------
    G: (6, nmodes) complex array
        The kernels for the [v, u, M1, M2, T, flux] sums
    """
    Sv, Su, base, _, _, _, (wvv, wvu, wuu) = kernel_ingredients(
        epoch, wt_cov, v0, u0,
    )
    G = np.empty((6, Sv.size), dtype=complex)
    G[0] = 1j * Sv * base
    G[1] = 1j * Su * base
    G[2] = ((wuu - wvv) - (Su ** 2 - Sv ** 2)) * base
    G[3] = 2.0 * (wvu - Sv * Su) * base
    G[4] = ((wuu + wvv) - (Su ** 2 + Sv ** 2)) * base
    G[5] = base
    return G


def dsums_dtheta(epoch, wt_cov, v0, u0):
    """
    the analytic (6, 5) derivative of the data sums with respect
    to theta = (wt_cov[0, 0], wt_cov[0, 1], wt_cov[1, 1], v, u)

    Contracted with the epoch's kim.  The kernel derivatives are
    exact; finite differences against admom_ksums itself only
    agree to ~1e-5 because of its polynomial fastexp
    approximation.

    Parameters
    ----------
    epoch: dict
        A prep_epoch dict
    wt_cov: (2, 2) array
        The weight covariance
    v0, u0: float
        The center offset (as for get_phase_angles)

    Returns
    -------
    D: (6, 5) array
        d(sums)/d(theta)
    """
    Sv, Su, base, yf, xf, dim, (wvv, wvu, wuu) = kernel_ingredients(
        epoch, wt_cov, v0, u0,
    )
    kv, ku = epoch['kv'], epoch['ku']
    kim = epoch['kim']

    c = [
        1j * Sv, 1j * Su,
        (wuu - wvv) - (Su ** 2 - Sv ** 2),
        2.0 * (wvu - Sv * Su),
        (wuu + wvv) - (Su ** 2 + Sv ** 2),
        np.ones_like(Sv),
    ]
    zero = np.zeros_like(Sv)
    dc = {
        0: [1j * kv, 1j * ku, zero],
        1: [zero, 1j * kv, 1j * ku],
        2: [
            -1.0 + 2 * Sv * kv,
            -2 * Su * kv + 2 * Sv * ku,
            1.0 - 2 * Su * ku,
        ],
        3: [
            -2 * kv * Su,
            2 * (1.0 - ku * Su - kv * Sv),
            -2 * Sv * ku,
        ],
        4: [
            1.0 - 2 * Sv * kv,
            -2 * (Sv * ku + Su * kv),
            1.0 - 2 * Su * ku,
        ],
        5: [zero, zero, zero],
    }
    dchi = [kv * kv, 2 * kv * ku, ku * ku]

    # d(alpha, beta)/d(v, u) is linear; measure it exactly
    a0, b0 = get_phase_angles(epoch, v0, u0)
    av, bv = get_phase_angles(epoch, v0 + 1.0, u0)
    au, bu = get_phase_angles(epoch, v0, u0 + 1.0)
    pj = np.array([
        [av - a0, au - a0],
        [bv - b0, bu - b0],
    ])

    D = np.zeros((6, 5))
    for a in range(6):
        for w in range(3):
            dG = (dc[a][w] - 0.5 * c[a] * dchi[w]) * base
            D[a, w] = (dG @ kim).real
        Ga = c[a] * base
        dSa = ((2j * np.pi / dim) * yf * Ga @ kim).real
        dSb = ((2j * np.pi / dim) * xf * Ga @ kim).real
        D[a, 3] = dSa * pj[0, 0] + dSb * pj[1, 0]
        D[a, 4] = dSa * pj[0, 1] + dSb * pj[1, 1]
    return D


def influence_kernels(epoch, G, shape, reference=False):
    """
    real-space influence kernels for a stacked set of k-space
    kernels

    h[a](x) = d(sum_a)/d(image[x]) on the image support of the
    given shape, via the epoch's stored image->kim transfer.  The
    half-plane mode set is Hermitian-completed and inverted with
    irfft2.

    With an apodized prep (epoch['ap_rad'] > 0) the stored
    transfer maps the masked image to kim, so the image->kim map
    factorizes as ktransfer(k) e^{-ik.x} m(x); the apodization
    mask is diagonal in pixel space and is applied here, making
    the kernels exact under apodization.

    Parameters
    ----------
    epoch: dict
        A prep_epoch dict with ktransfer set
        (store_transfer=True)
    G: (nk, nmodes) complex array
        Stacked kernels, e.g. from moment_kernels
    shape: (ny, nx)
        The image shape to restrict to; must be the shape the
        prep apodized (the observation's image shape)
    reference: bool, optional
        If True, use the brute full-grid fft2 construction
        instead of the rfft path (for validation).  Default False

    Returns
    -------
    h: (nk, ny, nx) array
        The influence kernels
    """
    if epoch.get('ktransfer') is None:
        raise ValueError(
            'epoch has no ktransfer: influence kernels require '
            'a prep with store_transfer=True'
        )
    dim = epoch['dim']
    iy, ix = epoch['iy'], epoch['ix']
    ny, nx = shape
    A = G * epoch['ktransfer']

    ap_rad = float(epoch.get('ap_rad', 0.0))
    if ap_rad > 0:
        from ..prepsfmom import _build_square_apodization_mask

        mask = np.ones(shape)
        _build_square_apodization_mask(ap_rad, mask)
    else:
        mask = None

    if reference:
        full = np.zeros(
            (G.shape[0], dim, dim), dtype=complex,
        )
        full[:, iy, ix] = A
        h = np.fft.fft2(full, axes=(-2, -1)).real[:, :ny, :nx]
        return h * mask if mask is not None else h

    # Re[fft2(A)] = fft2(C) with the Hermitian completion
    # C(k) = (A(k) + conj(A(-k))) / 2; on the rfft half plane the
    # generic columns carry A/2 (their conjugates are implied),
    # while the self-conjugate columns (kx = 0 and, for even dim,
    # kx = dim/2) contain both members of each +-ky pair
    half = dim // 2 + 1
    C = np.zeros((G.shape[0], dim, half), dtype=complex)
    C[:, iy, ix] = 0.5 * A
    sc = (ix == 0) | (ix == dim // 2)
    if np.any(sc):
        np.add.at(
            C, (slice(None), (dim - iy[sc]) % dim, ix[sc]),
            0.5 * np.conj(A[:, sc]),
        )
    h = np.fft.irfft2(
        np.conj(C), s=(dim, dim), axes=(-2, -1),
    ) * dim ** 2
    h = h[:, :ny, :nx]
    return h * mask if mask is not None else h


def sums_cov(hs, weight):
    """
    the covariance of the stacked sums

    From the real-space influence kernels and the per-pixel
    variance of the weight map; zero-weight pixels carry no
    noise.

    Parameters
    ----------
    hs: (nk, ny, nx) array
        Influence kernels, e.g. from influence_kernels
    weight: (ny, nx) array
        The inverse variance weight map

    Returns
    -------
    cov: (nk, nk) array
        The covariance of the sums
    """
    var = np.where(
        weight > 0, 1.0 / np.clip(weight, 1.0e-300, None), 0.0,
    )
    hv = hs * var
    return np.tensordot(hv, hs, axes=([1, 2], [1, 2]))


# ---------------------------------------------------------------
# shared symmetric-matrix differentials for the fixed-point
# update algebra (used here and by the kdeblend full errors)
# ---------------------------------------------------------------

def _sym_sandwich(A):
    """the 3x3 matrix S with vec(A X A) = S vec(X) for symmetric
    2x2 X in the (00, 01, 11) component basis"""
    a, b, c = A[0, 0], A[0, 1], A[1, 1]
    return np.array([
        [a * a, 2 * a * b, b * b],
        [a * b, a * c + b * b, b * c],
        [b * b, 2 * b * c, c * c],
    ])


def _sym_inv(A):
    """inverse of a symmetric 2x2 matrix"""
    det = A[0, 0] * A[1, 1] - A[0, 1] ** 2
    return np.array([
        [A[1, 1], -A[0, 1]], [-A[0, 1], A[0, 0]],
    ]) / det


def sym3_mat(v):
    """the symmetric 2x2 matrix from its (00, 01, 11) components"""
    return np.array([[v[0], v[1]], [v[1], v[2]]])


def dw_derivs(M, wt_cov):
    """
    the deweight map and its derivatives

    new_wt_cov = (M^-1 - wt_cov^-1)^-1 with
    d(new_wt_cov) = S(new)[S(M^-1) dM - S(wt_cov^-1) d(wt_cov)] in the
    sym3 basis.

    Returns
    -------
    new_wt_cov: (2, 2) array or None
        The deweighted covariance
    DW_M: (3, 3) array or None
        d(new_wt_cov)/dM in the sym3 basis
    DW_S: (3, 3) array or None
        d(new_wt_cov)/d(wt_cov) in the sym3 basis
    ok: bool
        False when the deweight is flagged; the others are then
        None
    """
    from .prepsfadmom import deweight

    newS, flags = deweight(M, wt_cov)
    if flags != 0:
        return None, None, None, False
    Snew = _sym_sandwich(newS)
    return (
        newS,
        Snew @ _sym_sandwich(_sym_inv(M)),
        -(Snew @ _sym_sandwich(_sym_inv(wt_cov))),
        True,
    )


# map from (M1, M2, T) moment components to the sym3 covariance
# components (c00, c01, c11) and back
_MBASIS_TO_SYM = 0.5 * np.array([
    [-1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
])
_SYM_TO_MBASIS = np.array([
    [-1.0, 0.0, 1.0],
    [0.0, 2.0, 0.0],
    [1.0, 0.0, 1.0],
])


def padmom_full_covariance(
    fitter, epochs, nband, model_state, wt_cov, v0, u0, Tsmooth,
):
    """
    the full (fixed-point) covariance for a converged
    PAdmomFitter state

    The single-object case of the coupled estimating equations
    (the multi-object version lives in kdeblend full_errors).
    The actual plain update step is differentiated at the actual
    data with no model-consistency substitution, so unlike
    model_sandwich this stays calibrated under model mismatch,
    where the sandwich under-predicts T errors by ~17 percent and
    flux errors by ~11 percent for dev truth fit with exp
    (MC 2026-07-29).

    The dense products run under single_core_blas: ngmix is a
    single core library, and they are large enough for a threaded
    BLAS to fan out.

    Requires epochs prepped with store_transfer=True and an
    'obs_weight' entry per epoch (attached by the fitter when
    full_errors is set).

    Parameters
    ----------
    fitter: PAdmomFitter
        The fitter, for the fixcen setting and the step
        validation
    epochs: list of dict
        The prep_epoch dicts
    nband: int
        The number of bands
    model_state: dict or None
        The converged family state; None for the plain gauss
        path
    wt_cov: (2, 2) array
        The converged weight covariance
    v0, u0: float
        The converged center
    Tsmooth: float
        The smoothing T

    Returns
    -------
    res: dict or None
        Entries flux_cov (nband, nband), flux_err, fam_err_cov (the
        (M1, M2, T) family covariance for the structure errors)
        and s2n; None when the model is unsupported or the
        evaluation at the solution takes a guarded branch (the
        caller keeps the model_sandwich errors)
    """
    from ..util import single_core_blas

    with single_core_blas():
        return _padmom_full_covariance(
            fitter=fitter, epochs=epochs, nband=nband,
            model_state=model_state, wt_cov=wt_cov, v0=v0, u0=u0,
            Tsmooth=Tsmooth,
        )


def _padmom_full_covariance(
    fitter, epochs, nband, model_state, wt_cov, v0, u0, Tsmooth,
):
    """the body of padmom_full_covariance; see that docstring"""
    if model_state is None:
        # the plain gauss path passes no model state
        mtype = 'gauss'
    else:
        mtype = model_state['type']
        if mtype not in ('exp', 'dev'):
            return None

    # the packed state: [center (unless fixcen)], [family
    # covariance for the mixture families], weight covariance,
    # each symmetric 2x2 carried as 3 components
    nep = len(epochs)
    fixcen = fitter.fixcen
    ncen = 0 if fixcen else 2
    is_mix = mtype in ('exp', 'dev')
    nfam = 3 if is_mix else 0
    npar = ncen + nfam + 3
    icen = 0
    ifam = ncen
    isw = ncen + nfam

    # the data sums at the converged state are the linearization
    # point; everything downstream is ratios, so the joint flux
    # sum must be positive
    esums, Ds, facs, sums = _epoch_sums_and_derivs(
        epochs, wt_cov, v0, u0,
    )
    if not sums[5] > 0:
        return None

    # the update consumes the measured moment ratios, not the
    # raw sums; a non-positive measured T has no deweight
    ddv, ddu, m3, dm3 = _measured_moment_response(sums, fixcen)
    if m3[2] <= 0:
        return None

    # the weight update is the deweight of the measured moments;
    # DWm_M, DWm_S are its derivatives with respect to the
    # measured matrix and the weight
    Mmat = sym3_mat(_MBASIS_TO_SYM @ m3)
    new_wt_cov, DWm_M, DWm_S, ok = dw_derivs(Mmat, wt_cov)
    if not ok:
        return None
    dsym_dsums = _MBASIS_TO_SYM @ dm3

    # Phi_s: d(new state)/d(joint sums); Phi_x: the direct state
    # dependence.  The sums' own state dependence enters through
    # the analytic kernel derivatives in the J assembly
    Phi_s = np.zeros((npar, 6))
    Phi_x = np.zeros((npar, npar))

    if not fixcen:
        # the center update is the measured centroid ratio itself
        Phi_s[icen] = ddv
        Phi_s[icen + 1] = ddu
        Phi_x[icen, icen] = 1.0
        Phi_x[icen + 1, icen + 1] = 1.0

    # weight rows: the deweight chained through the measured
    # ratios (the sums channel) and its direct weight dependence
    Phi_s[isw:isw + 3] = DWm_M @ dsym_dsums
    Phi_x[isw:isw + 3, isw:isw + 3] = DWm_S

    if is_mix:
        # family rows: the model-matching condition couples the
        # family covariance to the measured moments through the
        # model prediction; see _fill_family_rows
        ok = _fill_family_rows(
            Phi_s=Phi_s, Phi_x=Phi_x, ifam=ifam, isw=isw,
            model_state=model_state, wt_cov=wt_cov,
            new_wt_cov=new_wt_cov, Tsmooth=Tsmooth, m3=m3,
            dm3=dm3, dsym_dsums=dsym_dsums, DWm_M=DWm_M,
            DWm_S=DWm_S, fitter=fitter,
        )
        if not ok:
            return None

    # J = Phi_x + Phi_s dS/dx: the direct state dependence plus
    # the sums channel, with dS/dx from the analytic kernel
    # derivatives; D columns are (sw3, v, u) -> state order
    Dstate = np.zeros((6, npar))
    for D, fac in zip(Ds, facs):
        if not fixcen:
            Dstate[:, icen] += fac * D[:, 3]
            Dstate[:, icen + 1] += fac * D[:, 4]
        Dstate[:, isw:isw + 3] += fac * D[:, 0:3]
    J = Phi_x + Phi_s @ Dstate

    # A = B_Phi: the update's direct response to each epoch's
    # sums, Phi_s scaled by the epoch accumulation factor
    A = np.zeros((npar, 6 * nep))
    for iep, fac in enumerate(facs):
        A[:, iep * 6:(iep + 1) * 6] = Phi_s * fac

    # the exact noise covariance of the data sums; epochs carry
    # independent noise, so it is block diagonal
    covS = _data_sums_cov(epochs, wt_cov, v0, u0)
    if covS is None:
        return None

    # the fixed point response dx = (I - J)^-1 B_Phi dS by the
    # implicit function theorem, and its covariance
    Mm = np.eye(npar) - J
    Tx = np.linalg.solve(Mm, A)
    covX = Tx @ covS @ Tx.T

    # the family covariance in the (M1, M2, T) basis for the
    # structure errors; for gauss the family is the weight minus
    # the constant smoothing
    if is_mix:
        cblock = covX[ifam:ifam + 3, ifam:ifam + 3]
    else:
        cblock = covX[isw:isw + 3, isw:isw + 3]
    fam_err_cov = _SYM_TO_MBASIS @ cblock @ _SYM_TO_MBASIS.T

    # fluxes, following the _finalize normalization F_b = fs_b / N_b:
    # the plain gauss path has N_b = ws_b / (4 pi sqrt(det wt_cov))
    # and the model paths N_b = upred_b, the predicted unit flux sum
    fs = np.zeros(nband)
    for ep, s, fac in zip(epochs, esums, facs):
        fs[ep['band']] += fac * s[5]

    if is_mix:
        norm = _model_flux_norm(
            epochs=epochs, model_state=model_state, wt_cov=wt_cov,
            Tsmooth=Tsmooth, nband=nband, npar=npar, ifam=ifam,
            isw=isw,
        )
    else:
        norm = _gauss_flux_norm(
            epochs=epochs, wt_cov=wt_cov, nband=nband, npar=npar,
            isw=isw,
        )
    if norm is None:
        return None
    N, dlnN = norm
    F = fs / N

    # the fluxes are not state coordinates: derived rows
    # dq = (dq/dS + dq/dx R) dS, assembled in _flux_covariance
    flux_cov = _flux_covariance(
        F=F, N=N, dlnN=dlnN, epochs=epochs, Ds=Ds, facs=facs,
        Tx=Tx, covS=covS, fixcen=fixcen, icen=icen, isw=isw,
        npar=npar, nband=nband,
    )

    # a non-positive variance or non-finite family covariance
    # marks a failed evaluation; the s/n falls back to the
    # quadrature sum when the covariance is not positive definite
    var = np.diag(flux_cov)
    if np.any(var <= 0) or not np.all(np.isfinite(fam_err_cov)):
        return None
    s2n = joint_flux_s2n(F, flux_cov)
    if s2n is None:
        s2n = np.sqrt(np.sum(F ** 2 / var))
    return {
        'flux_cov': flux_cov,
        'flux_err': np.sqrt(var),
        'fam_err_cov': fam_err_cov,
        's2n': s2n,
    }


def _epoch_sums_and_derivs(epochs, wt_cov, v0, u0):
    """
    the per-epoch weighted data sums, their analytic derivatives
    with respect to (wt_cov, v, u), the accumulation factors
    weight * detAtinv, and the accumulated joint sums
    """
    from .prepsfadmom_nb import admom_ksums

    esums = []
    Ds = []
    facs = []
    for ep in epochs:
        alpha, beta = get_phase_angles(ep, v0, u0)
        s = np.zeros(6)
        admom_ksums(
            ep['kim'], ep['iy'], ep['ix'], ep['dim'],
            alpha, beta, ep['kv'], ep['ku'],
            wt_cov[0, 0], wt_cov[0, 1], wt_cov[1, 1], ep['df2'],
            s,
        )
        esums.append(s)
        Ds.append(dsums_dtheta(ep, wt_cov, v0, u0))
        facs.append(ep['weight'] * ep['detAtinv'])

    sums = np.zeros(6)
    for s, fac in zip(esums, facs):
        sums += fac * s
    return esums, Ds, facs, sums


def _measured_moment_response(sums, fixcen):
    """
    the centroid corrected measured moment ratios m3 = (M1, M2, T)
    and their derivative dm3 = d(m3)/d(sums), including the
    centroid correction chain; also the centroid ratio derivatives
    ddv, ddu for the center rows.  The corrections dv, du are tiny
    at the converged point but kept exactly
    """
    finv = 1.0 / sums[5]
    if fixcen:
        dv = du = 0.0
    else:
        dv = sums[0] * finv
        du = sums[1] * finv
    m3 = np.array([
        sums[2] * finv - (du * du - dv * dv),
        sums[3] * finv - 2 * dv * du,
        sums[4] * finv - (dv * dv + du * du),
    ])

    def dratio(j):
        out = np.zeros(6)
        out[j] = finv
        out[5] = -sums[j] * finv ** 2
        return out

    ddv = dratio(0)
    ddu = dratio(1)
    dm3 = np.zeros((3, 6))
    dm3[0] = dratio(2) - 2 * du * ddu + 2 * dv * ddv
    dm3[1] = dratio(3) - 2 * (dv * ddu + du * ddv)
    dm3[2] = dratio(4) - 2 * (dv * ddv + du * ddu)
    return ddv, ddu, m3, dm3


def _fill_family_rows(
    Phi_s, Phi_x, ifam, isw, model_state, wt_cov, new_wt_cov,
    Tsmooth, m3, dm3, dsym_dsums, DWm_M, DWm_S, fitter,
):
    """
    fill the family rows of Phi_s and Phi_x for the mixture
    families: the responses of the family condition to the data
    sums, the family covariance and the weight, on the main
    (deweighted prediction) branch or the gain-1 fallback branch,
    followed by the plain-step health check.  Returns False when
    a model prediction is invalid or the converged evaluation is
    on a guarded branch the analytic derivatives do not cover;
    the caller then keeps the model_sandwich errors
    """
    from .models import model_ksums

    fam_cov = model_state['cov']
    sfam3 = np.array([
        fam_cov[0, 0], fam_cov[0, 1], fam_cov[1, 1],
    ])
    hs = 1.0e-6 * max(wt_cov[0, 0] + wt_cov[1, 1], 0.1)

    def mp3_of(sf3, sw3):
        """the predicted (M1, M2, T) moment ratios of the family
        under the given weight"""
        st = dict(model_state)
        st['cov'] = sym3_mat(sf3)
        st['F'] = np.ones(1)
        ps = model_ksums(
            st, 0, 0.0, 0.0, sym3_mat(sw3), 1.0, Tsmooth,
        )
        if not ps[5] > 0:
            return None
        pinv = 1.0 / ps[5]
        return np.array([
            ps[2] * pinv, ps[3] * pinv, ps[4] * pinv,
        ])

    sw3 = np.array([
        wt_cov[0, 0], wt_cov[0, 1], wt_cov[1, 1],
    ])
    mp0 = mp3_of(sfam3, sw3)
    if mp0 is None:
        return False

    # d(predicted ratios)/d(fam_cov) and /d(wt_cov) by micro
    # central differences of the closed form predictions
    dmp_dsfam = np.zeros((3, 3))
    dmp_dsw = np.zeros((3, 3))
    for c in range(3):
        for target, dout in (
            (sfam3, dmp_dsfam), (sw3, dmp_dsw),
        ):
            tp = target.copy()
            tm = target.copy()
            tp[c] += hs
            tm[c] -= hs
            if target is sfam3:
                p = mp3_of(tp, sw3)
                m = mp3_of(tm, sw3)
            else:
                p = mp3_of(sfam3, tp)
                m = mp3_of(sfam3, tm)
            if p is None or m is None:
                return False
            dout[:, c] = (p - m) / (2 * hs)

    Mpred = sym3_mat(_MBASIS_TO_SYM @ mp0)
    # Sp: the deweight image of the predicted moments, as in
    # _model_shift
    Sp, DWp_M, DWp_S, pok = dw_derivs(Mpred, wt_cov)
    if pok:
        # shift = new_wt_cov - Sp; fam_cov' = fam_cov + shift
        Phi_s[ifam:ifam + 3] = DWm_M @ dsym_dsums
        Phi_x[ifam:ifam + 3, ifam:ifam + 3] = (
            np.eye(3)
            - DWp_M @ _MBASIS_TO_SYM @ dmp_dsfam
        )
        Phi_x[ifam:ifam + 3, isw:isw + 3] = (
            DWm_S
            - DWp_M @ _MBASIS_TO_SYM @ dmp_dsw - DWp_S
        )
        shift = np.array([
            new_wt_cov[0, 0] - Sp[0, 0], new_wt_cov[0, 1] - Sp[0, 1],
            new_wt_cov[1, 1] - Sp[1, 1],
        ])
    else:
        shift, dsh_dm, dsh_dp, dsh_dsfam = _gain1_shift_derivs(
            m3, mp0, sfam3,
        )
        Phi_s[ifam:ifam + 3] = dsh_dm @ dm3
        Phi_x[ifam:ifam + 3, ifam:ifam + 3] = (
            np.eye(3)
            + dsh_dsfam + dsh_dp @ dmp_dsfam
        )
        Phi_x[ifam:ifam + 3, isw:isw + 3] = (
            dsh_dp @ dmp_dsw
        )

    # health: the plain step must be undamped and accepted
    prop, idamp, accepted = fitter._damped_step(
        fam_cov, sym3_mat(shift), new_wt_cov, Tsmooth,
    )
    if not accepted or idamp > 0:
        return False
    return True


def _gain1_shift_derivs(m3, mp0, sfam3):
    """
    the family shift and its derivatives on the gain-1 fallback
    branch, taken when the deweight of the predicted moments is
    not evaluable.  The fallback matches the measured and
    predicted moment ratios directly: with Tm = m3[2],
    Tp = mp0[2] and Tf the family T,

        fac   = Tm / Tp
        de_i  = m3_i / Tm - mp0_i / Tp
        base  = fac Tf / 2
        shift = ((fac - 1) sfam3_0 - base de1,
                 (fac - 1) sfam3_1 + base de2,
                 (fac - 1) sfam3_2 + base de1)

    as in the _model_shift fallback of prepsfadmom.py.  Returns
    the shift and its derivatives with respect to the measured
    ratios (dsh_dm), the predicted ratios (dsh_dp) and the packed
    family covariance (dsh_dsfam), by direct differentiation of
    the expressions above
    """
    Tm = m3[2]
    Tp = mp0[2]
    Tf = sfam3[0] + sfam3[2]
    fac = Tm / Tp
    de1 = m3[0] / Tm - mp0[0] / Tp
    de2 = m3[1] / Tm - mp0[1] / Tp
    base = 0.5 * fac * Tf
    shift = np.array([
        (fac - 1) * sfam3[0] - base * de1,
        (fac - 1) * sfam3[1] + base * de2,
        (fac - 1) * sfam3[2] + base * de1,
    ])
    sgn = np.array([-1.0, 1.0, 1.0])
    dfac_dm = np.array([0.0, 0.0, 1.0 / Tp])
    dfac_dp = np.array([0.0, 0.0, -fac / Tp])
    dde1_dm = np.array([
        1.0 / Tm, 0.0, -m3[0] / Tm ** 2,
    ])
    dde2_dm = np.array([
        0.0, 1.0 / Tm, -m3[1] / Tm ** 2,
    ])
    dde1_dp = np.array([
        -1.0 / Tp, 0.0, mp0[0] / Tp ** 2,
    ])
    dde2_dp = np.array([
        0.0, -1.0 / Tp, mp0[1] / Tp ** 2,
    ])
    dde_dm = [dde1_dm, dde2_dm, dde1_dm]
    dde_dp = [dde1_dp, dde2_dp, dde1_dp]
    dsh_dm = np.zeros((3, 3))
    dsh_dp = np.zeros((3, 3))
    for r in range(3):
        de_r = de1 if r != 1 else de2
        dsh_dm[r] = (
            sfam3[r] * dfac_dm
            + sgn[r] * 0.5 * Tf * de_r * dfac_dm
            + sgn[r] * base * dde_dm[r]
        )
        dsh_dp[r] = (
            sfam3[r] * dfac_dp
            + sgn[r] * 0.5 * Tf * de_r * dfac_dp
            + sgn[r] * base * dde_dp[r]
        )
    dsh_dsfam = np.zeros((3, 3))
    tr = np.array([1.0, 0.0, 1.0])
    for r in range(3):
        dsh_dsfam[r, r] += fac - 1.0
        de_r = de1 if r != 1 else de2
        dsh_dsfam[r] += sgn[r] * 0.5 * fac * de_r * tr
    return shift, dsh_dm, dsh_dp, dsh_dsfam


def _data_sums_cov(epochs, wt_cov, v0, u0):
    """
    the exact covariance of the per-epoch data sums from the
    real-space influence kernels and the pixel weight maps;
    epoch-block-diagonal (independent noise per epoch).  None
    when an epoch is missing the stored transfer or the weight
    map
    """
    nep = len(epochs)
    covS = np.zeros((6 * nep, 6 * nep))
    for iep, ep in enumerate(epochs):
        if ep.get('ktransfer') is None:
            return None
        w = ep.get('obs_weight')
        if w is None:
            return None
        G = moment_kernels(ep, wt_cov, v0, u0)
        h = influence_kernels(ep, G, w.shape)
        covS[
            iep * 6:(iep + 1) * 6, iep * 6:(iep + 1) * 6,
        ] = sums_cov(h, w)
    return covS


def _gauss_flux_norm(epochs, wt_cov, nband, npar, isw):
    """
    the per-band flux normalization of the plain gauss path,
    N_b = ws_b / (4 pi sqrt(det wt_cov)) so that F_b = fs_b / N_b
    matches the _finalize normalization, and d(ln N)/d(state),
    which enters only through the det factor: the epoch weights
    are constants
    """
    ws = np.zeros(nband)
    for ep in epochs:
        ws[ep['band']] += ep['weight']
    detS = wt_cov[0, 0] * wt_cov[1, 1] - wt_cov[0, 1] ** 2
    cnorm = 4.0 * np.pi * np.sqrt(detS)
    N = ws / cnorm
    # d ln cnorm / d(sw3); d ln N = -d ln cnorm
    dlnc = np.array([
        wt_cov[1, 1], -2.0 * wt_cov[0, 1], wt_cov[0, 0],
    ]) / (2.0 * detS)
    dlnN = np.zeros((nband, npar))
    dlnN[:, isw:isw + 3] = -dlnc[None, :]
    return N, dlnN


def _model_flux_norm(
    epochs, model_state, wt_cov, Tsmooth, nband, npar, ifam, isw,
):
    """
    the per-band flux normalization of the mixture model paths,
    N_b = upred_b, the model predicted unit flux sum exactly as
    _finalize computes it, and d(ln N)/d(state) by micro central
    differences of the closed form predictions.  None when a
    prediction is zero
    """
    from .models import model_ksums

    upred = np.zeros(nband)
    for ep in epochs:
        fac = ep['weight'] * ep['detAtinv']
        upred[ep['band']] += fac * model_ksums(
            model_state, ep['band'], 0.0, 0.0, wt_cov,
            ep['detAtinv'], Tsmooth,
        )[5]
    if np.any(upred == 0):
        return None

    hs = 1.0e-6 * max(wt_cov[0, 0] + wt_cov[1, 1], 0.1)

    def upred_of(sf3, sw3v):
        st = dict(model_state)
        st['cov'] = sym3_mat(sf3)
        out = np.zeros(nband)
        for ep in epochs:
            fac = ep['weight'] * ep['detAtinv']
            ps = model_ksums(
                st, ep['band'], 0.0, 0.0, sym3_mat(sw3v),
                ep['detAtinv'], Tsmooth,
            )
            out[ep['band']] += fac * ps[5]
        return out

    sw3 = np.array([
        wt_cov[0, 0], wt_cov[0, 1], wt_cov[1, 1],
    ])
    sf3 = np.array([
        model_state['cov'][0, 0], model_state['cov'][0, 1],
        model_state['cov'][1, 1],
    ])
    dup_dstate = np.zeros((nband, npar))
    for c in range(3):
        swp = sw3.copy()
        swm = sw3.copy()
        swp[c] += hs
        swm[c] -= hs
        dup_dstate[:, isw + c] = (
            upred_of(sf3, swp) - upred_of(sf3, swm)
        ) / (2 * hs)
        sfp = sf3.copy()
        sfm = sf3.copy()
        sfp[c] += hs
        sfm[c] -= hs
        dup_dstate[:, ifam + c] = (
            upred_of(sfp, sw3) - upred_of(sfm, sw3)
        ) / (2 * hs)

    dlnN = dup_dstate / upred[:, None]
    return upred, dlnN


def _flux_covariance(
    F, N, dlnN, epochs, Ds, facs, Tx, covS, fixcen, icen, isw,
    npar, nband,
):
    """
    covariance of the fluxes F_b = fs_b / N_b: the direct
    flux-sum channel plus the chain through the state (the own
    kernel and center responses of the flux sums via Tx, minus
    the normalization response F dlnN), contracted with the
    data-sum covariance
    """
    nep = len(epochs)
    RF = np.zeros((nband, 6 * nep))
    for iep, ep in enumerate(epochs):
        b = ep['band']
        RF[b, iep * 6 + 5] += facs[iep] / N[b]
    gx = np.zeros((nband, npar))
    for iep, ep in enumerate(epochs):
        b = ep['band']
        D = Ds[iep]
        row = np.zeros(npar)
        if not fixcen:
            row[icen] = D[5, 3]
            row[icen + 1] = D[5, 4]
        row[isw:isw + 3] = D[5, 0:3]
        gx[b] += facs[iep] * row / N[b]
    gx -= F[:, None] * dlnN
    RF = RF + gx @ Tx
    return RF @ covS @ RF.T
