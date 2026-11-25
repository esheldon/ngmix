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
under model mismatch, where the sandwich errors under-predict
(T by ~17 percent and flux by ~11 percent for dev truth fit with
exp).

Derivation
----------
Notation

    d       the pixel data
    S       the data sums, linear in d: S = H d, with the rows of
            H the influence kernels (influence_kernels)
    x       the packed state: [center], [family covariance],
            weight covariance
    Phi     one update step, new state = Phi(x, S)

The kernels depend on the state through the weight covariance and
center, so the sums are S(x) = H(x) d.

The estimating equation.  The fitter iterates the update to
convergence, so the reported state is a fixed point

    x = Phi(x, S(x))

The response to noise.  Noise perturbs the data, d -> d + dd,
which moves the sums at fixed kernel by dS = H dd, and the fixed
point moves to x + dx.  Expanding the estimating equation to
first order,

    dx = Phi_x dx + Phi_S (dS/dx dx + dS)

where

    Phi_x   dPhi/dx at fixed sums, the direct state dependence
    Phi_S   dPhi/dS at fixed state
    dS/dx   the motion of the sums under a moving kernel at fixed
            data (dsums_dtheta)

Collecting the dx terms,

    J = Phi_x + Phi_S dS/dx       total Jacobian of one update
    (I - J) dx = Phi_S dS
    dx = R dS,   R = (I - J)^-1 Phi_S

I - J is invertible when the iteration contracts, which is also
the condition for it to converge.

The covariance.  The estimator is linear in the sums to first
order, so

    Cov(x) = R Cov(S) R^T
    Cov(S) = H diag(var) H^T      (sums_cov)

with var the per-pixel variance from the weight map, the pixels
being independent.  For stationary correlated noise, with
autocovariance C estimated from a noise image,

    Cov(S) = H C H^T              (sums_cov_noise)

which is used when the fitter is run with use_noise_image.  The
noise image of a small stamp gives a noisy C: the errors of a
faint isolated object scatter by about 20 percent from the
estimate alone.  C can instead be measured once from a larger
region with the same noise (noise_autocov) and sent to the fitter
as noise_autocov.

Both Phi_x and Phi_S are evaluated at the observed sums, not at
sums predicted from the converged model; this is what the
sandwich path substitutes.

Derived quantities.  A quantity q(x, S) that is not a state
coordinate, such as a flux (a flux sum over a state-dependent
normalization), responds as

    dq = (dq/dS + dq/dx R) dS

In padmom_full_covariance the variables are Phi_s (Phi_S), Phi_x,
Dstate (dS/dx), J, A (Phi_S per epoch, the B_Phi of Sheldon 2026),
Tx (R), covS and covX.

References
----------
Hampel (1974), "The influence curve and its role in robust
    estimation", JASA 69, 383
Sheldon (2026), "Measurement covariances under correlated
    noise", in preparation

Research for the errors was aided significantly by Claude Code
"""
import numpy as np
import scipy.fft

from .prepsfadmom import get_phase_angles
from .errors import joint_flux_s2n

__all__ = [
    'WK_CHI2_MAX', 'kernel_ingredients', 'moment_kernels',
    'dsums_dtheta', 'influence_kernels', 'sums_cov',
    'noise_autocov_power', 'noise_autocov', 'sums_cov_noise',
    'sym3_mat',
    'dw_derivs', 'padmom_full_covariance',
]

# the kernel weight support, matching FASTEXP_MAX_CHI2
WK_CHI2_MAX = 25.0

# the noise autocovariance is estimated at the lags for which at
# least this fraction of the good pixels have a good partner
NOISE_LAG_MIN_PAIR_FRAC = 0.5

# the default largest lag, in pixels, of noise_autocov
NOISE_AUTOCOV_MAX_LAG = 32

# kernels per fft call in sums_cov_noise, bounding the memory
SUMS_COV_NOISE_CHUNK = 16


def kernel_ingredients(epoch, wt_cov, v0, u0):
    """
    the shared per-epoch pieces of the kernel expressions

    Exported for use in kdeblend.

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
    Sv, Su: arrays
        The components of wt_cov @ k at each mode
    base: complex array
        The weight times the centering phase times df2
    yf, xf: int arrays
        The signed row and col frequency indices of each mode
    dim: int
        The fft dimension
    (wvv, wvu, wuu): floats
        The weight covariance elements
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

    Derivation
    ----------
    Write W for wt_cov, k = (kv, ku) for a mode, and

        S = W k                (the vector (Sv, Su))
        chi2 = k^T W k
        wk = exp(-chi2 / 2)

    wk is the Fourier transform of the real-space gaussian weight
    of covariance W.  Multiplying the weight by a coordinate in
    real space is a k derivative in Fourier space, x_i -> i d/dk_i
    in the sign convention of the centering phase, and

        d(wk)/dk_i = -S_i wk
        d^2(wk)/dk_i dk_j = (S_i S_j - W_ij) wk

    so the weighted first and second moments have kernels

        x_i      ->  i S_i wk
        x_i x_j  ->  (W_ij - S_i S_j) wk

    The six sums are these in the [v, u, M1, M2, T, flux] basis,
    with M1 = uu - vv, M2 = 2 vu, T = uu + vv:

        v:    i Sv
        u:    i Su
        M1:   (W_uu - W_vv) - (Su^2 - Sv^2)
        M2:   2 (W_vu - Sv Su)
        T:    (W_uu + W_vv) - (Su^2 + Sv^2)
        flux: 1

    each times base = wk * phase * df2, where phase centers the
    weight on (v0, u0) and df2 is the fft normalization (see
    kernel_ingredients).

    The modes are the rfft half plane.  The image is real, so
    the omitted half plane contributes the complex conjugate of
    each term and the full-plane sum is twice the real part; that
    factor is already folded into kim as the conjugate symmetry
    weights (get_kspace_grids), leaving sums = Re(G @ kim).

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

    Derivation
    ----------
    Each kernel is G_a = c_a(S, W) wk phase df2 in the notation of
    moment_kernels, and the sum is Re(G_a @ kim), so the
    derivative of a sum is the derivative of its kernel
    contracted with the same kim.

    Weight covariance.  Only c_a and wk depend on W.  With
    S = W k and chi2 = k^T W k,

        dSv/dW_vv = kv    dSv/dW_vu = ku    dSv/dW_uu = 0
        dSu/dW_vv = 0     dSu/dW_vu = kv    dSu/dW_uu = ku
        dchi2/d(W_vv, W_vu, W_uu) = (kv^2, 2 kv ku, ku^2)

    and by the product rule

        dG_a/dW = (dc_a/dW - c_a dchi2/dW / 2) wk phase df2

    The dc table below is dc_a/dW for the c_a of moment_kernels,
    using the dS/dW above; e.g. for T,
    c = (W_uu + W_vv) - (Su^2 + Sv^2) gives
    dc/dW_vv = 1 - 2 Sv kv.  The chi2 < WK_CHI2_MAX support cut
    is held fixed (wk is exp(-12.5) there).

    Center.  Only the phase depends on the center:
    phase = exp(2 pi i (alpha yf + beta xf) / dim), so

        dG_a/dalpha = (2 pi i yf / dim) G_a
        dG_a/dbeta  = (2 pi i xf / dim) G_a

    and (alpha, beta) are linear in (v, u) through the jacobian
    (get_phase_angles), giving the chain rule with the constant
    2x2 matrix pj = d(alpha, beta)/d(v, u).

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

    # the kernel coefficients c_a of moment_kernels
    c = [
        1j * Sv, 1j * Su,
        (wuu - wvv) - (Su ** 2 - Sv ** 2),
        2.0 * (wvu - Sv * Su),
        (wuu + wvv) - (Su ** 2 + Sv ** 2),
        np.ones_like(Sv),
    ]
    zero = np.zeros_like(Sv)
    # dc[a] = dc_a/d(W_vv, W_vu, W_uu)
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
    # dchi2/d(W_vv, W_vu, W_uu)
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
        # weight covariance columns: product rule on c_a * wk
        for w in range(3):
            dG = (dc[a][w] - 0.5 * c[a] * dchi[w]) * base
            D[a, w] = (dG @ kim).real
        # center columns: d/d(alpha, beta) then the chain to (v, u)
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


def noise_autocov_power(noise, weight=None):
    """
    the noise power spectrum from the autocovariance of a noise
    image, for sums_cov_noise

    The noise is taken to be stationary, with covariance
    Cov(n_p, n_q) = C(p - q) between pixels p and q.  C is
    estimated at each lag d by the average over the pairs of
    good pixels separated by d,

        C(d) = sum_p g_p g_{p+d} n_p n_{p+d} / N(d)
        N(d) = sum_p g_p g_{p+d}

    with g the good pixel indicator.  Both sums are
    autocorrelations, evaluated by fft on a zero padded grid so
    that no pair wraps around the image.  Dividing by the pair
    count N(d) rather than the number of pixels makes the
    estimate unbiased on a finite image with bad pixels; the raw
    periodogram is the same sum divided by the number of pixels,
    which is low by N(d) / npix.

    Lags with few pairs are noisy and are dropped (C set to
    zero): those with N(d) less than NOISE_LAG_MIN_PAIR_FRAC
    times the number of good pixels, which for an unmasked image
    keeps lags out to a fraction of the image size.  Correlations
    on longer scales cannot be measured from the image.

    The returned power is the fft of C on the padded grid; it is
    real since C is symmetric, but not guaranteed non-negative.

    Parameters
    ----------
    noise: (ny, nx) array
        A realization of the noise in the frame of the image
    weight: (ny, nx) array, optional
        The weight map, used only to mark the good pixels,
        weight > 0.  All pixels are good if not sent

    Returns
    -------
    power: (py, px // 2 + 1) array or None
        The power on the rfft half plane of the padded grid;
        None when there are no good pixels
    shape: (py, px)
        The shape of the padded grid
    good: (ny, nx) bool array
        The good pixel indicator
    """
    ny, nx = noise.shape
    if weight is None:
        good = np.ones((ny, nx), dtype=bool)
    else:
        good = weight > 0

    # room for lags out to half the image without wrapping
    shape = (
        scipy.fft.next_fast_len(ny + ny // 2, real=True),
        scipy.fft.next_fast_len(nx + nx // 2, real=True),
    )
    ngood = int(good.sum())
    if ngood == 0:
        return None, shape, good

    gf = good.astype(float)
    # bad pixels can hold anything, including non finite values
    gnoise = np.where(good, noise, 0.0)

    num = scipy.fft.irfft2(
        np.abs(scipy.fft.rfft2(gnoise, shape)) ** 2, shape,
    )
    # the pair counts are integers
    npair = np.rint(scipy.fft.irfft2(
        np.abs(scipy.fft.rfft2(gf, shape)) ** 2, shape,
    ))

    # lags beyond half the image would wrap on the padded grid
    ly = np.fft.fftfreq(shape[0], d=1.0 / shape[0])[:, None]
    lx = np.fft.fftfreq(shape[1], d=1.0 / shape[1])[None, :]
    use = (
        (np.abs(ly) <= ny // 2) & (np.abs(lx) <= nx // 2)
        & (npair >= max(1.0, NOISE_LAG_MIN_PAIR_FRAC * ngood))
    )
    acov = np.zeros(shape)
    acov[use] = num[use] / npair[use]

    power = scipy.fft.rfft2(acov).real
    return power, shape, good


def noise_autocov(noise, weight=None, max_lag=NOISE_AUTOCOV_MAX_LAG):
    """
    the noise autocovariance from a noise image, as a lag array
    for sums_cov_noise

    The estimator is that of noise_autocov_power: at each lag the
    average of the products over the pairs of good pixels, which
    is unbiased on a finite image with bad pixels.  The result
    does not depend on the size of the image it is applied to, so
    it can be measured once on a large region, where it is well
    determined, and used for every stamp cut from that region.

    Lags with fewer good pairs than NOISE_LAG_MIN_PAIR_FRAC times
    the number of good pixels are set to zero, as are lags beyond
    the image.

    Parameters
    ----------
    noise: (ny, nx) array
        A realization of the noise
    weight: (ny, nx) array, optional
        The weight map, used only to mark the good pixels,
        weight > 0.  All pixels are good if not sent
    max_lag: int, optional
        The largest lag kept in each direction, in pixels.  The
        correlations beyond it are taken to be zero.  Default
        NOISE_AUTOCOV_MAX_LAG

    Returns
    -------
    autocov: (2 max_lag + 1, 2 max_lag + 1) array
        The autocovariance C(dy, dx) at autocov[max_lag + dy,
        max_lag + dx]; symmetric under (dy, dx) -> (-dy, -dx).
        Zero when there are no good pixels
    """
    max_lag = int(max_lag)
    if max_lag < 0:
        raise ValueError(f'max_lag must be >= 0, got {max_lag}')

    ny, nx = noise.shape
    autocov = np.zeros((2 * max_lag + 1, 2 * max_lag + 1))
    if weight is None:
        good = np.ones((ny, nx), dtype=bool)
    else:
        good = weight > 0
    ngood = int(good.sum())
    if ngood == 0:
        return autocov

    # room for the lags kept without wrapping
    ly, lx = min(max_lag, ny - 1), min(max_lag, nx - 1)
    shape = (
        scipy.fft.next_fast_len(ny + ly, real=True),
        scipy.fft.next_fast_len(nx + lx, real=True),
    )
    gf = good.astype(float)
    gnoise = np.where(good, noise, 0.0)
    num = scipy.fft.irfft2(
        np.abs(scipy.fft.rfft2(gnoise, shape)) ** 2, shape,
    )
    npair = np.rint(scipy.fft.irfft2(
        np.abs(scipy.fft.rfft2(gf, shape)) ** 2, shape,
    ))

    iy = np.arange(-ly, ly + 1)
    ix = np.arange(-lx, lx + 1)
    grid = np.ix_(iy % shape[0], ix % shape[1])
    num, npair = num[grid], npair[grid]
    use = npair >= max(1.0, NOISE_LAG_MIN_PAIR_FRAC * ngood)
    sub = np.zeros(num.shape)
    sub[use] = num[use] / npair[use]
    autocov[
        max_lag - ly:max_lag + ly + 1, max_lag - lx:max_lag + lx + 1,
    ] = sub
    return autocov


def _autocov_power(autocov, ny, nx):
    """
    the power of a lag array (noise_autocov) on a padded grid for
    an (ny, nx) image, with room for its lags without wrapping

    Returns
    -------
    power: (py, px // 2 + 1) array
    shape: (py, px)
    """
    autocov = np.asarray(autocov, dtype=float)
    if (autocov.ndim != 2 or autocov.shape[0] % 2 != 1
            or autocov.shape[1] % 2 != 1):
        raise ValueError(
            'autocov must be a 2d array with odd dimensions, '
            f'got shape {autocov.shape}'
        )
    my, mx = (autocov.shape[0] - 1) // 2, (autocov.shape[1] - 1) // 2
    # lags beyond the image pair no pixels
    ly, lx = min(my, ny - 1), min(mx, nx - 1)
    shape = (
        scipy.fft.next_fast_len(ny + ly, real=True),
        scipy.fft.next_fast_len(nx + lx, real=True),
    )
    iy = np.arange(-ly, ly + 1)
    ix = np.arange(-lx, lx + 1)
    acov = np.zeros(shape)
    acov[np.ix_(iy % shape[0], ix % shape[1])] = autocov[
        my - ly:my + ly + 1, mx - lx:mx + lx + 1,
    ]
    return scipy.fft.rfft2(acov).real, shape


def sums_cov_noise(hs, noise=None, weight=None, autocov=None):
    """
    the covariance of the stacked sums for stationary correlated
    noise, from a noise image or a noise autocovariance

    With C the noise autocovariance (see noise_autocov_power),

        Cov(S_a, S_b) = sum_pq h_a(p) C(p - q) h_b(q)

    which is sums_cov with the product h * var replaced by the
    convolution of h with C.  The convolution is evaluated by fft
    on a zero padded grid, so nothing wraps around the image.

    C is estimated from the noise image, or taken from autocov
    when that is sent (see noise_autocov), in which case the
    noise image is not used.  The estimate from the noise image
    of a small stamp is noisy; an autocov measured on a larger
    region with the same noise is not.

    The same quantity is a sum over Fourier modes of the
    transformed kernels times the power.  Summing the modes
    directly saves the inverse fft, but needs the transforms of
    all the kernels at once, which on the padded grid take more
    than twice the memory of hs; here only one chunk of kernels
    is in k space at a time.

    Bad pixels (weight <= 0) carry no noise, as in sums_cov: the
    kernels are set to zero there, and those pixels of the noise
    image are not used to estimate C.  The weight map is used
    only for this; the noise level comes from the noise image or
    autocov.

    Parameters
    ----------
    hs: (nk, ny, nx) array
        Influence kernels, e.g. from influence_kernels
    noise: (ny, nx) array, optional
        A realization of the noise in the frame of the image.
        Required unless autocov is sent
    weight: (ny, nx) array, optional
        The weight map, used only to mark the good pixels,
        weight > 0.  All pixels are good if not sent
    autocov: 2d array with odd dimensions, optional
        The noise autocovariance as a lag array centered on zero
        lag, from noise_autocov.  Used in place of the estimate
        from the noise image

    Returns
    -------
    cov: (nk, nk) array
        The covariance of the sums.  Zero when there are no good
        pixels
    """
    nk, ny, nx = hs.shape
    cov = np.zeros((nk, nk))

    if autocov is not None:
        if weight is None:
            good = np.ones((ny, nx), dtype=bool)
        else:
            good = weight > 0
        if not good.any():
            return cov
        power, shape = _autocov_power(autocov, ny, nx)
    else:
        if noise is None:
            raise ValueError('send noise or autocov')
        power, shape, good = noise_autocov_power(noise, weight=weight)
        if power is None:
            return cov

    hg = hs * good
    for i0 in range(0, nk, SUMS_COV_NOISE_CHUNK):
        rows = slice(i0, i0 + SUMS_COV_NOISE_CHUNK)
        # the convolution of the kernels with C
        hc = scipy.fft.irfft2(
            scipy.fft.rfft2(hg[rows], shape, axes=(-2, -1)) * power,
            shape, axes=(-2, -1),
        )[:, :ny, :nx]
        cov[:, rows] = np.tensordot(hg, hc, axes=([1, 2], [1, 2]))

    # symmetric up to rounding
    return 0.5 * (cov + cov.T)


# ---------------------------------------------------------------
# shared symmetric-matrix differentials for the fixed-point
# update algebra (used here and by the kdeblend full errors)
# ---------------------------------------------------------------

def _sym_sandwich(A):
    """
    the 3x3 matrix S with vec(A X A) = S vec(X) for symmetric
    2x2 A, X in the (00, 01, 11) component basis

    Each row is the corresponding component of A X A written
    out; with A = [[a, b], [b, c]] and X = [[x0, x1], [x1, x2]],

        (A X A)_00 = a^2 x0 + 2 a b x1 + b^2 x2
        (A X A)_01 = a b x0 + (a c + b^2) x1 + b c x2
        (A X A)_11 = b^2 x0 + 2 b c x1 + c^2 x2

    Parameters
    ----------
    A: (2, 2) array
        A symmetric matrix
    """
    a, b, c = A[0, 0], A[0, 1], A[1, 1]
    return np.array([
        [a * a, 2 * a * b, b * b],
        [a * b, a * c + b * b, b * c],
        [b * b, 2 * b * c, c * c],
    ])


def _sym_inv(A):
    """
    inverse of a symmetric 2x2 matrix

    Parameters
    ----------
    A: (2, 2) array
        A symmetric matrix
    """
    det = A[0, 0] * A[1, 1] - A[0, 1] ** 2
    return np.array([
        [A[1, 1], -A[0, 1]], [-A[0, 1], A[0, 0]],
    ]) / det


def sym3_mat(v):
    """
    the symmetric 2x2 matrix from its (00, 01, 11) components

    Parameters
    ----------
    v: array of size 3
        The (00, 01, 11) components
    """
    return np.array([[v[0], v[1]], [v[1], v[2]]])


def dw_derivs(M, wt_cov):
    """
    the deweight map and its derivatives

    new_wt_cov = (M^-1 - wt_cov^-1)^-1 with
    d(new_wt_cov) = S(new)[S(M^-1) dM - S(wt_cov^-1) d(wt_cov)] in the
    sym3 basis.

    Derivation
    ----------
    Write N for new_wt_cov and W for wt_cov, so
    N^-1 = M^-1 - W^-1.  The differential of a matrix inverse is
    d(A^-1) = -A^-1 dA A^-1.  Applying it to each inverse,

        d(N^-1) = -M^-1 dM M^-1 + W^-1 dW W^-1
        dN = -N d(N^-1) N
           = N M^-1 dM M^-1 N - N W^-1 dW W^-1 N

    Every term has the form A X A with A symmetric, which in the
    sym3 basis is the matrix S(A) of _sym_sandwich, giving

        dN/dM = S(N) S(M^-1)
        dN/dW = -S(N) S(W^-1)

    Parameters
    ----------
    M: (2, 2) array
        The measured weighted moment matrix
    wt_cov: (2, 2) array
        The weight covariance under which M was measured

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
    full_errors is set).  The noise is taken from these weight
    maps with the pixels treated as independent, unless the
    epochs also carry an 'obs_noise' entry (attached with
    use_noise_image), in which case it is the stationary
    correlated noise measured from those noise images; see
    sums_cov_noise.

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
        (M1, M2, T) family covariance for the structure errors),
        gauss_err_cov (the (M1, M2, T) covariance of the converged
        weight, for the errors of the gauss estimator entries; the
        same as fam_err_cov for the gauss model) and s2n; None
        when the model is unsupported or the evaluation at the
        solution takes a guarded branch (the caller keeps the
        model_sandwich errors)
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
    """
    the body of padmom_full_covariance; see that docstring

    Parameters
    ----------
    fitter, epochs, nband, model_state, wt_cov, v0, u0, Tsmooth:
        As for padmom_full_covariance
    """
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
        # center rows.  The update is v' = v + S_v / S_F (and the
        # same for u): the direct state dependence is the identity
        # and the sums channel is the centroid ratio derivative
        Phi_s[icen] = ddv
        Phi_s[icen + 1] = ddu
        Phi_x[icen, icen] = 1.0
        Phi_x[icen + 1, icen + 1] = 1.0

    # weight rows.  The update is W' = N(M(S), W), the deweight
    # of the measured moments M under the current weight W, so
    #     dW' = dN/dM dM/dS dS + dN/dW dW
    # the first term is the sums channel (M in the sym3 basis,
    # chained through the measured ratios) and the second the
    # direct weight dependence
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

    # the weight covariance in the (M1, M2, T) basis, for the
    # errors of the gauss (converged weight) estimator.  The
    # estimator is the weight minus the constant smoothing, so
    # its covariance is that of the weight
    wblock = covX[isw:isw + 3, isw:isw + 3]
    gauss_err_cov = _SYM_TO_MBASIS @ wblock @ _SYM_TO_MBASIS.T

    # the family covariance in the (M1, M2, T) basis for the
    # structure errors; for gauss the family is the weight minus
    # the constant smoothing
    if is_mix:
        cblock = covX[ifam:ifam + 3, ifam:ifam + 3]
        fam_err_cov = _SYM_TO_MBASIS @ cblock @ _SYM_TO_MBASIS.T
    else:
        fam_err_cov = gauss_err_cov

    # fluxes, following the _finalize normalization
    # F_b = S_Fb / N_b, with S_Fb the band flux sum (fs below):
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
        'gauss_err_cov': gauss_err_cov,
        's2n': s2n,
    }


def _epoch_sums_and_derivs(epochs, wt_cov, v0, u0):
    """
    the per-epoch weighted data sums, their analytic derivatives
    with respect to (wt_cov, v, u), the accumulation factors
    weight * detAtinv, and the accumulated joint sums

    Parameters
    ----------
    epochs: list of dict
        The prep_epoch dicts, one per epoch
    wt_cov: (2, 2) array
        The weight covariance
    v0, u0: float
        The center, as sky offsets from the jacobian center

    Returns
    -------
    esums: list of arrays of size 6
        The per-epoch sums
    Ds: list of (6, 5) arrays
        The per-epoch dsums_dtheta
    facs: list of float
        The per-epoch accumulation factors
    sums: array of size 6
        The joint sums, sum_e facs[e] * esums[e]
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

    With S_j the joint sums in the [v, u, M1, M2, T, flux] order
    and S_F = S_5 the flux sum,

        dv = S_0 / S_F,  du = S_1 / S_F
        M1 = S_2 / S_F - (du^2 - dv^2)
        M2 = S_3 / S_F - 2 dv du
        T  = S_4 / S_F - (dv^2 + du^2)

    the second terms moving the second moments from the weight
    center to the measured centroid.  Each ratio r_j = S_j / S_F
    has

        dr_j = dS_j / S_F - S_j dS_F / S_F^2

    and the rows of dm3 are these plus the product rule on the
    dv, du terms.

    Parameters
    ----------
    sums: array of size 6
        The joint [v, u, M1, M2, T, flux] sums
    fixcen: bool
        If True the center is not updated and the centroid
        corrections are zero

    Returns
    -------
    ddv, ddu: arrays of size 6
        d(dv)/d(sums) and d(du)/d(sums)
    m3: array of size 3
        The measured (M1, M2, T) ratios
    dm3: (3, 6) array
        d(m3)/d(sums)
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
        """d(S_j / S_F)/d(sums) for the sum with index j"""
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

    Derivation
    ----------
    Notation

        C       the family covariance (fam_cov)
        W       the weight covariance (wt_cov)
        M       the measured moments, a function of the sums S
        P       the moments the model predicts under the weight,
                a function P(C, W) of the state only
        N(., W) the deweight map, N(M, W) = (M^-1 - W^-1)^-1

    The update (_model_shift) moves the family covariance by a
    shift that vanishes when the measured and predicted moments
    agree, C' = C + shift.

    Main branch.  The shift is the difference of the deweighted
    measured and predicted moments,

        shift = N(M, W) - N(P, W)

    N(M, W) is the weight update W' of the caller, so its
    derivatives are the weight rows (DWm_M, DWm_S).  Writing
    N_M, N_W for the derivatives of N with respect to its two
    arguments (dw_derivs), evaluated at M or at P as marked,

        dC' = N_M|M dM/dS dS                           (sums)
            + [I - N_M|P dP/dC] dC                     (family)
            + [N_W|M - N_M|P dP/dW - N_W|P] dW         (weight)

    The family term is the identity from C' = C + ... minus the
    response of the prediction to the family; the weight term has
    the weight entering the measured deweight directly, and the
    predicted deweight both through P and directly.

    Fallback branch.  When N(P, W) is not evaluable the shift is
    built from the moment ratios, shift(m, p, C) with m, p the
    measured and predicted (M1, M2, T) (see _gain1_shift_derivs),
    so

        dC' = dshift/dm dm/dS dS                       (sums)
            + [I + dshift/dC + dshift/dp dp/dC] dC     (family)
            + [dshift/dp dp/dW] dW                     (weight)

    Here the weight enters only through the prediction.

    dP/dC and dP/dW (dmp_dsfam, dmp_dsw, in the (M1, M2, T)
    basis) are central differences of the closed-form prediction;
    _MBASIS_TO_SYM carries them to the sym3 basis that N_M acts
    on.

    Parameters
    ----------
    Phi_s: (npar, 6) array
        d(new state)/d(joint sums); the family rows are filled in
        place
    Phi_x: (npar, npar) array
        The direct state dependence; the family rows are filled in
        place
    ifam: int
        The index of the first family covariance entry in the packed
        state
    isw: int
        The index of the first weight covariance entry in the packed
        state
    model_state: dict
        The converged model dict
    wt_cov: (2, 2) array
        The converged weight covariance
    new_wt_cov: (2, 2) array
        The deweight of the measured moments under wt_cov
    Tsmooth: float
        The T of the gaussian smoothing
    m3: array of size 3
        The measured (M1, M2, T) ratios
    dm3: (3, 6) array
        d(m3)/d(sums)
    dsym_dsums: (3, 6) array
        dm3 carried to the sym3 basis
    DWm_M, DWm_S: (3, 3) arrays
        The derivatives of the deweight of the measured moments with
        respect to the moments and the weight, from dw_derivs
    fitter: PAdmomFitter
        The fitter, for the step validation
    """
    from .models import model_ksums

    fam_cov = model_state['cov']
    sfam3 = np.array([
        fam_cov[0, 0], fam_cov[0, 1], fam_cov[1, 1],
    ])
    hs = 1.0e-6 * max(wt_cov[0, 0] + wt_cov[1, 1], 0.1)

    def mp3_of(sf3, sw3):
        """
        the predicted (M1, M2, T) moment ratios of the family
        under the given weight

        Parameters
        ----------
        sf3, sw3: arrays of size 3
            The family and weight covariances in the sym3 basis
        """
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
        # main branch: shift = new_wt_cov - Sp = N(M, W) - N(P, W)
        # and fam_cov' = fam_cov + shift
        # sums: N_M|M dM/dS
        Phi_s[ifam:ifam + 3] = DWm_M @ dsym_dsums
        # family: I - N_M|P dP/dC
        Phi_x[ifam:ifam + 3, ifam:ifam + 3] = (
            np.eye(3)
            - DWp_M @ _MBASIS_TO_SYM @ dmp_dsfam
        )
        # weight: N_W|M - N_M|P dP/dW - N_W|P
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
        # fallback branch: shift(m, p, C) on the moment ratios
        # sums: dshift/dm dm/dS
        Phi_s[ifam:ifam + 3] = dsh_dm @ dm3
        # family: I + dshift/dC + dshift/dp dp/dC
        Phi_x[ifam:ifam + 3, ifam:ifam + 3] = (
            np.eye(3)
            + dsh_dsfam + dsh_dp @ dmp_dsfam
        )
        # weight: dshift/dp dp/dW
        Phi_x[ifam:ifam + 3, isw:isw + 3] = (
            dsh_dp @ dmp_dsw
        )

    # health: the plain step must be undamped and accepted; a
    # damped or rejected step is a different update map than the
    # one differentiated above
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

    Row r of the shift is

        shift_r = (fac - 1) sfam3_r + sgn_r (fac Tf / 2) de_r

    with sgn = (-1, +1, +1) and de_r = (de1, de2, de1).  fac and
    de depend on the measured and predicted ratios, Tf and sfam3
    on the family covariance, so by the product rule

        dshift_r/dm = sfam3_r dfac/dm
                      + sgn_r (Tf / 2) de_r dfac/dm
                      + sgn_r (fac Tf / 2) dde_r/dm

    and the same with p for m, while

        dshift_r/dsfam3_c = (fac - 1) delta_rc
                            + sgn_r (fac / 2) de_r dTf/dsfam3_c

    with dTf/dsfam3 = (1, 0, 1).

    Parameters
    ----------
    m3: array of size 3
        The measured (M1, M2, T) ratios
    mp0: array of size 3
        The predicted (M1, M2, T) ratios
    sfam3: array of size 3
        The family covariance in the sym3 basis
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
    # fac = Tm / Tp depends only on the T components
    dfac_dm = np.array([0.0, 0.0, 1.0 / Tp])
    dfac_dp = np.array([0.0, 0.0, -fac / Tp])
    # de_i = m3_i / Tm - mp0_i / Tp: quotient rule in each ratio
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
    # dTf/dsfam3
    tr = np.array([1.0, 0.0, 1.0])
    for r in range(3):
        dsh_dsfam[r, r] += fac - 1.0
        de_r = de1 if r != 1 else de2
        dsh_dsfam[r] += sgn[r] * 0.5 * fac * de_r * tr
    return shift, dsh_dm, dsh_dp, dsh_dsfam


def _data_sums_cov(epochs, wt_cov, v0, u0):
    """
    the covariance of the per-epoch data sums from the
    real-space influence kernels; epoch-block-diagonal
    (independent noise per epoch).

    The noise comes from the weight map of each epoch, with the
    pixels independent (sums_cov), unless the epoch carries a
    noise image in 'obs_noise', in which case it is the
    stationary correlated noise measured from that image
    (sums_cov_noise) and the weight map only marks the good
    pixels.  An epoch carrying a 'noise_autocov' lag array uses
    that autocovariance in place of the one measured from its
    noise image.

    None when an epoch is missing the stored transfer or the
    weight map, or when the flux sum of an epoch has no noise:
    there are no good pixels under its kernel (or the noise
    image is zero), and the epoch would enter the errors as
    noiseless data.  Also None when a variance from the noise
    image is negative; that estimate is not guaranteed positive
    (see noise_autocov_power)

    Parameters
    ----------
    epochs: list of dict
        The prep_epoch dicts, one per epoch
    wt_cov: (2, 2) array
        The weight covariance
    v0, u0: float
        The center, as sky offsets from the jacobian center

    Returns
    -------
    covS: (6 nep, 6 nep) array or None
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
        noise = ep.get('obs_noise')
        autocov = ep.get('noise_autocov')
        if autocov is not None:
            ecov = sums_cov_noise(h, weight=w, autocov=autocov)
        elif noise is not None:
            ecov = sums_cov_noise(h, noise, weight=w)
        else:
            ecov = sums_cov(h, w)
        if np.any(np.diag(ecov) < 0) or not ecov[5, 5] > 0:
            return None
        covS[
            iep * 6:(iep + 1) * 6, iep * 6:(iep + 1) * 6,
        ] = ecov
    return covS


def _gauss_flux_norm(epochs, wt_cov, nband, npar, isw):
    """
    the per-band flux normalization of the plain gauss path,
    N_b = ws_b / (4 pi sqrt(det wt_cov)) so that F_b = S_Fb / N_b
    matches the _finalize normalization, and d(ln N)/d(state),
    which enters only through the det factor: the epoch weights
    are constants

    With W = wt_cov, ln N_b = const - ln(det W) / 2, and
    det W = W_00 W_11 - W_01^2 gives

        d(det W)/d(W_00, W_01, W_11) = (W_11, -2 W_01, W_00)
        dlnN_b/dW = -(W_11, -2 W_01, W_00) / (2 det W)

    the same for every band.

    Parameters
    ----------
    epochs: list of dict
        The prep_epoch dicts, one per epoch
    wt_cov: (2, 2) array
        The weight covariance
    nband: int
        The number of bands
    npar: int
        The size of the packed state
    isw: int
        The index of the first weight covariance entry in the packed
        state

    Returns
    -------
    N: array of size nband
        The normalizations
    dlnN: (nband, npar) array
        d(ln N)/d(state)
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

    Parameters
    ----------
    epochs: list of dict
        The prep_epoch dicts, one per epoch
    model_state: dict
        The converged model dict
    wt_cov: (2, 2) array
        The weight covariance
    Tsmooth: float
        The T of the gaussian smoothing
    nband: int
        The number of bands
    npar: int
        The size of the packed state
    ifam: int
        The index of the first family covariance entry in the packed
        state
    isw: int
        The index of the first weight covariance entry in the packed
        state

    Returns
    -------
    N: array of size nband
        The normalizations
    dlnN: (nband, npar) array
        d(ln N)/d(state)
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
        """the predicted unit flux sums per band for the family and
        weight covariances sf3, sw3v in the sym3 basis"""
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
    covariance of the fluxes F_b = S_Fb / N_b: the direct
    flux-sum channel plus the chain through the state (the own
    kernel and center responses of the flux sums via Tx, minus
    the normalization response F dlnN), contracted with the
    data-sum covariance

    Derivation
    ----------
    S_Fb = sum_e fac_e S_e[flux] over the epochs e of band b is
    the band flux sum, and N_b(x) the normalization.  The flux is
    a derived quantity q(x, S) in the sense of the module
    docstring:

        dF_b = dS_Fb / N_b - F_b dlnN_b/dx dx

    The flux sum moves with the noise at fixed kernel and with
    the kernel at fixed data,

        dS_Fb = sum_e fac_e (dS_e[flux] + dS_e[flux]/dx dx)

    so with dx = R dS (R is Tx),

        dF = (RF0 + gx R) dS
        RF0[b, (e, flux)] = fac_e / N_b                   dq/dS
        gx[b] = sum_e fac_e dS_e[flux]/dx / N_b
                - F_b dlnN_b/dx                           dq/dx

    and Cov(F) = RF Cov(S) RF^T with RF = RF0 + gx R.

    Parameters
    ----------
    F: array of size nband
        The fluxes
    N: array of size nband
        The flux normalizations
    dlnN: (nband, npar) array
        d(ln N)/d(state)
    epochs: list of dict
        The prep_epoch dicts, one per epoch
    Ds: list of (6, 5) arrays
        The per-epoch dsums_dtheta
    facs: list of float
        The per-epoch accumulation factors
    Tx: (npar, 6 nep) array
        The state response to the per-epoch sums, R in the module
        docstring
    covS: (6 nep, 6 nep) array
        The covariance of the per-epoch sums
    fixcen: bool
        If True the center is not part of the state
    icen: int
        The index of the first center entry in the packed state
    isw: int
        The index of the first weight covariance entry in the packed
        state
    npar: int
        The size of the packed state
    nband: int
        The number of bands

    Returns
    -------
    flux_cov: (nband, nband) array
    """
    nep = len(epochs)
    # dq/dS: the flux sum of each epoch feeds its band's flux
    RF = np.zeros((nband, 6 * nep))
    for iep, ep in enumerate(epochs):
        b = ep['band']
        RF[b, iep * 6 + 5] += facs[iep] / N[b]
    # dq/dx: the flux sums under a moving kernel; row 5 of D is
    # the flux sum, columns (sw3, v, u) -> state order
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
    # ... minus the response of the normalization
    gx -= F[:, None] * dlnN
    # the total response dF = (dq/dS + dq/dx R) dS
    RF = RF + gx @ Tx
    return RF @ covS @ RF.T
