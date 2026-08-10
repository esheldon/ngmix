"""
Building blocks for full (fixed-point) error propagation of the
pre-PSF adaptive moments machinery.

The weighted moment sums are linear in the prepped k-space image
with closed-form kernels (the admom_ksums accumulators), so their
exact covariance under the pixel noise follows from real-space
influence kernels: each sum's kernel is assembled on the k grid,
carried to the image plane through the stored image->kim transfer
(epoch['ktransfer']) and contracted with the per-pixel variance
from the weight map.  Apodization enters exactly: the masked
prep's image->kim map factorizes as ktransfer(k) e^{-ik.x} m(x),
so influence_kernels applies the mask in pixel space, where it
is diagonal.  The
kernel derivatives with respect to the weight covariance and
center are also closed form, allowing the data sums to be
linearized about a solution so that fixed-point Jacobians never
touch the data modes.

These pieces are consumed by deblenders (kdeblend full_errors)
and are the basis for full errors of the single-object fitter.
Unlike the model_sandwich error path, covariances built from
them make no model-consistency substitution, so they remain
calibrated under model mismatch, where the sandwich structure
errors under-predict (T by ~12 percent for real morphologies
fit with exp).
"""
import numpy as np

from .prepsfadmom import get_phase_angles

__all__ = [
    'moment_kernels', 'dsums_dtheta', 'influence_kernels',
    'sums_cov',
]

# the kernel weight support, matching FASTEXP_MAX_CHI2
WK_CHI2_MAX = 25.0


def _ingredients(epoch, Sw, v0, u0):
    alpha, beta = get_phase_angles(epoch, v0, u0)
    dim = epoch['dim']
    iy = epoch['iy'].astype(np.int64)
    ix = epoch['ix'].astype(np.int64)
    kv, ku = epoch['kv'], epoch['ku']
    wvv, wvu, wuu = Sw[0, 0], Sw[0, 1], Sw[1, 1]

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


def moment_kernels(epoch, Sw, v0, u0):
    """
    the (6, nmodes) complex kernels G with sums = Re(G @ kim):
    the admom_ksums accumulators in closed form, at weight
    covariance Sw and center offset (v0, u0) in the epoch's
    phase convention (as for get_phase_angles)
    """
    Sv, Su, base, _, _, _, (wvv, wvu, wuu) = _ingredients(
        epoch, Sw, v0, u0,
    )
    G = np.empty((6, Sv.size), dtype=complex)
    G[0] = 1j * Sv * base
    G[1] = 1j * Su * base
    G[2] = ((wuu - wvv) - (Su ** 2 - Sv ** 2)) * base
    G[3] = 2.0 * (wvu - Sv * Su) * base
    G[4] = ((wuu + wvv) - (Su ** 2 + Sv ** 2)) * base
    G[5] = base
    return G


def dsums_dtheta(epoch, Sw, v0, u0):
    """
    the analytic (6, 5) derivative of the data sums with respect
    to theta = (Sw00, Sw01, Sw11, v, u), contracted with the
    epoch's kim.  The kernel derivatives are exact; finite
    differences against admom_ksums itself only agree to ~1e-3
    because of its table exponential
    """
    Sv, Su, base, yf, xf, dim, (wvv, wvu, wuu) = _ingredients(
        epoch, Sw, v0, u0,
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
    kernels: h[a](x) = d(sum_a)/d(image[x]) on the image support
    of the given shape, via the epoch's stored image->kim
    transfer.  The half-plane mode set is Hermitian-completed and
    inverted with irfft2; reference=True uses the brute full-grid
    fft2 construction instead (for validation).

    With an apodized prep (epoch['ap_rad'] > 0) the stored
    transfer maps the masked image to kim, so the image->kim map
    factorizes as ktransfer(k) e^{-ik.x} m(x): the apodization
    mask is diagonal in pixel space and is applied here, making
    the kernels exact under apodization

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
    the covariance of the stacked sums from their real-space
    influence kernels and the per-pixel variance of the weight
    map (zero-weight pixels carry no noise)
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

def sym_sandwich(A):
    """the 3x3 matrix S with vec(A X A) = S vec(X) for symmetric
    2x2 X in the (00, 01, 11) component basis"""
    a, b, c = A[0, 0], A[0, 1], A[1, 1]
    return np.array([
        [a * a, 2 * a * b, b * b],
        [a * b, a * c + b * b, b * c],
        [b * b, 2 * b * c, c * c],
    ])


def sym_inv(A):
    det = A[0, 0] * A[1, 1] - A[0, 1] ** 2
    return np.array([
        [A[1, 1], -A[0, 1]], [-A[0, 1], A[0, 0]],
    ]) / det


def sym3_mat(v):
    return np.array([[v[0], v[1]], [v[1], v[2]]])


def dw_derivs(M, Sigma):
    """deweight and its derivatives: newSigma = (M^-1 -
    Sigma^-1)^-1 with d(newSigma) = S(new)[S(M^-1) dM -
    S(Sigma^-1) dSigma] in the sym3 basis.  Returns (newSigma,
    DW_M, DW_S, ok)"""
    from .prepsfadmom import deweight

    newS, flags = deweight(M, Sigma)
    if flags != 0:
        return None, None, None, False
    Snew = sym_sandwich(newS)
    return (
        newS,
        Snew @ sym_sandwich(sym_inv(M)),
        -(Snew @ sym_sandwich(sym_inv(Sigma))),
        True,
    )


# map from (M1, M2, T) moment components to the sym3 covariance
# components (c00, c01, c11) and back
MBASIS_TO_SYM = 0.5 * np.array([
    [-1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
])
SYM_TO_MBASIS = np.array([
    [-1.0, 0.0, 1.0],
    [0.0, 2.0, 0.0],
    [1.0, 0.0, 1.0],
])


def padmom_full_covariance(
    fitter, epochs, nband, model_state, Sigma, v0, u0, Tsmooth,
):
    """
    the full (fixed-point) covariance for a converged
    PAdmomFitter state: the m=1 case of the coupled estimating
    equations, differentiating the actual plain update step at
    the actual data with no model-consistency substitution.
    The dense products run under single_core_blas: ngmix is a
    single core library, and they are large enough for a
    threaded BLAS to fan out.
    Unlike model_sandwich this stays calibrated under model
    mismatch, where the sandwich under-predicts T errors by ~17
    percent and flux errors by ~11 percent for dev truth fit
    with exp (MC 2026-07-29).

    Requires epochs prepped with store_transfer=True and an
    'obs_weight' entry per epoch (attached by the fitter when
    full_errors is set).

    Returns a dict with flux_cov (nband, nband), fam_cov (the
    (M1, M2, T) family covariance for the structure errors) and
    s2n, or None when the evaluation at the solution takes a
    guarded branch (the caller keeps the model_sandwich errors)
    """
    from ..util import single_core_blas

    with single_core_blas():
        return _padmom_full_covariance(
            fitter=fitter, epochs=epochs, nband=nband,
            model_state=model_state, Sigma=Sigma, v0=v0, u0=u0,
            Tsmooth=Tsmooth,
        )


def _padmom_full_covariance(
    fitter, epochs, nband, model_state, Sigma, v0, u0, Tsmooth,
):
    from .prepsfadmom import model_ksums
    from .prepsfadmom_nb import admom_ksums

    if model_state is None:
        # the plain gauss path passes no model state
        mtype = 'gauss'
    else:
        mtype = model_state['type']
        if mtype not in ('exp', 'dev'):
            return None

    nep = len(epochs)
    fixcen = fitter.fixcen
    ncen = 0 if fixcen else 2
    is_mix = mtype in ('exp', 'dev')
    nfam = 3 if is_mix else 0
    npar = ncen + nfam + 3
    icen = 0
    ifam = ncen
    isw = ncen + nfam

    # per-epoch data sums, derivatives and accumulation factors
    esums = []
    Ds = []
    facs = []
    for ep in epochs:
        from .prepsfadmom import get_phase_angles

        alpha, beta = get_phase_angles(ep, v0, u0)
        s = np.zeros(6)
        admom_ksums(
            ep['kim'], ep['iy'], ep['ix'], ep['dim'],
            alpha, beta, ep['kv'], ep['ku'],
            Sigma[0, 0], Sigma[0, 1], Sigma[1, 1], ep['df2'],
            s,
        )
        esums.append(s)
        Ds.append(dsums_dtheta(ep, Sigma, v0, u0))
        facs.append(ep['weight'] * ep['detAtinv'])

    sums = np.zeros(6)
    for s, fac in zip(esums, facs):
        sums += fac * s
    if not sums[5] > 0:
        return None
    finv = 1.0 / sums[5]

    # the measured moments with the centroid correction; dv, du
    # are tiny at the converged point but kept exactly
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
    if m3[2] <= 0:
        return None

    # d(m3)/d(sums) including the centroid correction chain
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

    Mmat = sym3_mat(MBASIS_TO_SYM @ m3)
    newSw, DWm_M, DWm_S, ok = dw_derivs(Mmat, Sigma)
    if not ok:
        return None
    dsym_dsums = MBASIS_TO_SYM @ dm3

    # Phi_s: d(new state)/d(joint sums); Phi_x: the direct state
    # dependence.  The sums' own state dependence enters through
    # the analytic kernel derivatives in the J assembly
    Phi_s = np.zeros((npar, 6))
    Phi_x = np.zeros((npar, npar))

    if not fixcen:
        Phi_s[icen] = ddv
        Phi_s[icen + 1] = ddu
        Phi_x[icen, icen] = 1.0
        Phi_x[icen + 1, icen + 1] = 1.0

    Phi_s[isw:isw + 3] = DWm_M @ dsym_dsums
    Phi_x[isw:isw + 3, isw:isw + 3] = DWm_S

    if is_mix:
        Sfam = model_state['cov']
        sfam3 = np.array([
            Sfam[0, 0], Sfam[0, 1], Sfam[1, 1],
        ])
        hs = 1.0e-6 * max(Sigma[0, 0] + Sigma[1, 1], 0.1)

        def mp3_of(sf3, sw3):
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
            Sigma[0, 0], Sigma[0, 1], Sigma[1, 1],
        ])
        mp0 = mp3_of(sfam3, sw3)
        if mp0 is None:
            return None
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
                    return None
                dout[:, c] = (p - m) / (2 * hs)

        Mpred = sym3_mat(MBASIS_TO_SYM @ mp0)
        Sp, DWp_M, DWp_S, pok = dw_derivs(Mpred, Sigma)
        if pok:
            # shift = newSw - Sp; Sfam' = Sfam + shift
            Phi_s[ifam:ifam + 3] = DWm_M @ dsym_dsums
            Phi_x[ifam:ifam + 3, ifam:ifam + 3] = (
                np.eye(3)
                - DWp_M @ MBASIS_TO_SYM @ dmp_dsfam
            )
            Phi_x[ifam:ifam + 3, isw:isw + 3] = (
                DWm_S
                - DWp_M @ MBASIS_TO_SYM @ dmp_dsw - DWp_S
            )
            shift = np.array([
                newSw[0, 0] - Sp[0, 0], newSw[0, 1] - Sp[0, 1],
                newSw[1, 1] - Sp[1, 1],
            ])
        else:
            # gain-1 fallback on the ratios
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
            Sfam, sym3_mat(shift), newSw, Tsmooth,
        )
        if not accepted or idamp > 0:
            return None

    # J: the state dependence of the sums through the analytic
    # kernel derivatives, D columns (sw3, v, u) -> state order
    Dstate = np.zeros((6, npar))
    for D, fac in zip(Ds, facs):
        if not fixcen:
            Dstate[:, icen] += fac * D[:, 3]
            Dstate[:, icen + 1] += fac * D[:, 4]
        Dstate[:, isw:isw + 3] += fac * D[:, 0:3]
    J = Phi_x + Phi_s @ Dstate

    # A: the data response per epoch
    A = np.zeros((npar, 6 * nep))
    for iep, fac in enumerate(facs):
        A[:, iep * 6:(iep + 1) * 6] = Phi_s * fac

    # Cov(S) from the influence kernels and the weight maps
    covS = np.zeros((6 * nep, 6 * nep))
    for iep, ep in enumerate(epochs):
        if ep.get('ktransfer') is None:
            return None
        w = ep.get('obs_weight')
        if w is None:
            return None
        G = moment_kernels(ep, Sigma, v0, u0)
        h = influence_kernels(ep, G, w.shape)
        covS[
            iep * 6:(iep + 1) * 6, iep * 6:(iep + 1) * 6,
        ] = sums_cov(h, w)

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
    fam_cov = SYM_TO_MBASIS @ cblock @ SYM_TO_MBASIS.T

    # fluxes, following the _finalize normalization: the model
    # paths use F_b = fs_b / upred_b; the plain gauss path uses
    # F_b = 2 knrm fs_b / ws_b with knrm = 2 pi sqrt(det Sigma)
    fs = np.zeros(nband)
    for ep, s, fac in zip(epochs, esums, facs):
        fs[ep['band']] += fac * s[5]

    if not is_mix:
        ws = np.zeros(nband)
        for ep in epochs:
            ws[ep['band']] += ep['weight']
        detS = Sigma[0, 0] * Sigma[1, 1] - Sigma[0, 1] ** 2
        cnorm = 4.0 * np.pi * np.sqrt(detS)
        F = cnorm * fs / ws
        # d ln cnorm / d(sw3)
        dlnc = np.array([
            Sigma[1, 1], -2.0 * Sigma[0, 1], Sigma[0, 0],
        ]) / (2.0 * detS)

        RF = np.zeros((nband, 6 * nep))
        for iep, ep in enumerate(epochs):
            b = ep['band']
            RF[b, iep * 6 + 5] += cnorm * facs[iep] / ws[b]
        gx = np.zeros((nband, npar))
        for iep, ep in enumerate(epochs):
            b = ep['band']
            D = Ds[iep]
            row = np.zeros(npar)
            if not fixcen:
                row[icen] = D[5, 3]
                row[icen + 1] = D[5, 4]
            row[isw:isw + 3] = D[5, 0:3]
            gx[b] += cnorm * facs[iep] * row / ws[b]
        gx[:, isw:isw + 3] += F[:, None] * dlnc[None, :]
        RF = RF + gx @ Tx
        flux_cov = RF @ covS @ RF.T

        var = np.diag(flux_cov)
        if np.any(var <= 0) or not np.all(np.isfinite(fam_cov)):
            return None
        s2n = np.sqrt(np.sum(F ** 2 / var))
        return {
            'flux_cov': flux_cov,
            'flux_err': np.sqrt(var),
            'fam_cov': fam_cov,
            's2n': s2n,
        }

    # upred exactly as _finalize computes it
    upred = np.zeros(nband)
    for ep in epochs:
        fac = ep['weight'] * ep['detAtinv']
        upred[ep['band']] += fac * model_ksums(
            model_state, ep['band'], 0.0, 0.0, Sigma,
            ep['detAtinv'], Tsmooth,
        )[5]
    if np.any(upred == 0):
        return None
    F = fs / upred

    # d(upred)/d(state): family and weight channels by micro-FD
    hs2 = 1.0e-6 * max(Sigma[0, 0] + Sigma[1, 1], 0.1)

    def upred_of(sf3, sw3v):
        st = dict(model_state)
        if is_mix:
            st['cov'] = sym3_mat(sf3)
        else:
            st['cov_sm'] = sym3_mat(sw3v)
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
        Sigma[0, 0], Sigma[0, 1], Sigma[1, 1],
    ])
    if is_mix:
        sf3 = np.array([
            model_state['cov'][0, 0], model_state['cov'][0, 1],
            model_state['cov'][1, 1],
        ])
    else:
        sf3 = None
    dup_dstate = np.zeros((nband, npar))
    for c in range(3):
        swp = sw3.copy()
        swm = sw3.copy()
        swp[c] += hs2
        swm[c] -= hs2
        dup_dstate[:, isw + c] = (
            upred_of(sf3, swp) - upred_of(sf3, swm)
        ) / (2 * hs2)
        if is_mix:
            sfp = sf3.copy()
            sfm = sf3.copy()
            sfp[c] += hs2
            sfm[c] -= hs2
            dup_dstate[:, ifam + c] = (
                upred_of(sfp, sw3) - upred_of(sfm, sw3)
            ) / (2 * hs2)

    # flux response over the data-sum space
    RF = np.zeros((nband, 6 * nep))
    for iep, ep in enumerate(epochs):
        b = ep['band']
        RF[b, iep * 6 + 5] += facs[iep] / upred[b]
    gx = np.zeros((nband, npar))
    for iep, ep in enumerate(epochs):
        b = ep['band']
        D = Ds[iep]
        row = np.zeros(npar)
        if not fixcen:
            row[icen] = D[5, 3]
            row[icen + 1] = D[5, 4]
        row[isw:isw + 3] = D[5, 0:3]
        gx[b] += facs[iep] * row / upred[b]
    gx -= (F / upred)[:, None] * dup_dstate
    RF = RF + gx @ Tx
    flux_cov = RF @ covS @ RF.T

    var = np.diag(flux_cov)
    if np.any(var <= 0) or not np.all(np.isfinite(fam_cov)):
        return None
    s2n = np.sqrt(np.sum(F ** 2 / var))
    return {
        'flux_cov': flux_cov,
        'flux_err': np.sqrt(var),
        'fam_cov': fam_cov,
        's2n': s2n,
    }
