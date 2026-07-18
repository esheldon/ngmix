"""
numba kernels for pre-PSF adaptive moments

The k-space images are stored as 1d arrays of the modes where the
kernels are non-negligible, along with the (row, col) frequency indices
of each mode.

The centering phase exp(i k . cen) is separable in the row and col
frequencies of the FFT grid, so it is applied inside the loops as
products of two 1d phasor arrays, avoiding per-mode trigonometry.
"""
import numpy as np
from numba import njit

from ..fastexp_nb import fexp, FASTEXP_MAX_CHI2


@njit
def _fill_phasors(dim, ang, pre, pim):
    """
    fill the 1d phasors exp(i * f1d * ang) where f1d = 2 pi fftfreq(dim)
    """
    fac = 2.0 * np.pi * ang / dim
    half = (dim + 1) // 2
    for j in range(dim):
        if j < half:
            a = fac * j
        else:
            a = fac * (j - dim)
        pre[j] = np.cos(a)
        pim[j] = np.sin(a)


@njit
def admom_ksums(
    kim, iy, ix, dim, alpha, beta,
    kv, ku, wvv, wvu, wuu, df2,
    sums,
):
    """
    weighted real-space moment sums evaluated in k-space

    Parameters
    ----------
    kim: complex 1d array
        deconvolved, smoothed k-space image at the masked modes, with
        no centering phases applied
    iy, ix: integer 1d arrays
        row/col frequency indices of each mode
    dim: int
        the fft dimension
    alpha, beta: float
        the centering phase angles in pixel units; the phase for a mode
        is exp(i (f1d[iy] alpha + f1d[ix] beta))
    kv, ku: 1d arrays
        the k grids in sky coordinates at the masked modes
    wvv, wvu, wuu: float
        the weight covariance matrix
    df2: float
        1/dim^2 fft normalization
    sums: array of size 6
        output [v, u, M1, M2, T, flux] sums
    """
    pyre = np.empty(dim)
    pyim = np.empty(dim)
    pxre = np.empty(dim)
    pxim = np.empty(dim)
    _fill_phasors(dim, alpha, pyre, pyim)
    _fill_phasors(dim, beta, pxre, pxim)

    s0 = 0.0
    sv = 0.0
    su = 0.0
    svv = 0.0
    svu = 0.0
    suu = 0.0

    nmodes = kim.size
    for i in range(nmodes):
        kvi = kv[i]
        kui = ku[i]

        Sv = wvv * kvi + wvu * kui
        Su = wvu * kvi + wuu * kui
        chi2 = kvi * Sv + kui * Su

        if chi2 > FASTEXP_MAX_CHI2:
            continue
        wk = fexp(-0.5 * chi2)

        y = iy[i]
        x = ix[i]
        pr = pyre[y] * pxre[x] - pyim[y] * pxim[x]
        pi = pyre[y] * pxim[x] + pyim[y] * pxre[x]

        val = kim[i]
        re = val.real * pr - val.imag * pi
        im = val.imag * pr + val.real * pi

        wre = wk * re
        wim = wk * im

        s0 += wre
        sv -= Sv * wim
        su -= Su * wim
        svv += Sv * Sv * wre
        svu += Sv * Su * wre
        suu += Su * Su * wre

    vv = wvv * s0 - svv
    vu = wvu * s0 - svu
    uu = wuu * s0 - suu

    sums[0] = sv * df2
    sums[1] = su * df2
    sums[2] = (uu - vv) * df2
    sums[3] = 2 * vu * df2
    sums[4] = (uu + vv) * df2
    sums[5] = s0 * df2


@njit
def admom_finalize(
    kim, iy, ix, dim, alpha, beta,
    kv, ku, wvv, wvu, wuu, df2,
    err_fac2,
    sums, cov_raw,
):
    """
    compute the moment sums and the raw kernel cross sums for the noise
    covariance at the converged weight

    Parameters
    ----------
    Same as admom_ksums, plus

    err_fac2: 1d array
        The noise power times |smooth/kpsf|^2 at the masked modes: the
        per-mode variance of the raw image fft times the squared
        magnitude of the factors relating the effective kernels to it
    sums: array of size 6
        output [v, u, M1, M2, T, flux] sums
    cov_raw: array (6, 6)
        output kernel cross sums; multiply by df2^2 to get the
        covariance of the unnormalized sums.  The center sums come from
        the imaginary part and do not covary with the even sums.
    """
    pyre = np.empty(dim)
    pyim = np.empty(dim)
    pxre = np.empty(dim)
    pxim = np.empty(dim)
    _fill_phasors(dim, alpha, pyre, pyim)
    _fill_phasors(dim, beta, pxre, pxim)

    s0 = 0.0
    sv = 0.0
    su = 0.0
    svv = 0.0
    svu = 0.0
    suu = 0.0

    kern = np.zeros(6)
    cov_raw[:, :] = 0.0

    nmodes = kim.size
    for i in range(nmodes):
        kvi = kv[i]
        kui = ku[i]

        Sv = wvv * kvi + wvu * kui
        Su = wvu * kvi + wuu * kui
        chi2 = kvi * Sv + kui * Su

        if chi2 > FASTEXP_MAX_CHI2:
            continue
        wk = fexp(-0.5 * chi2)

        y = iy[i]
        x = ix[i]
        pr = pyre[y] * pxre[x] - pyim[y] * pxim[x]
        pi = pyre[y] * pxim[x] + pyim[y] * pxre[x]

        val = kim[i]
        re = val.real * pr - val.imag * pi
        im = val.imag * pr + val.real * pi

        wre = wk * re
        wim = wk * im

        s0 += wre
        sv -= Sv * wim
        su -= Su * wim
        svv += Sv * Sv * wre
        svu += Sv * Su * wre
        suu += Su * Su * wre

        # the effective kernels for noise propagation
        kern[0] = Sv * wk
        kern[1] = Su * wk
        vvk = (wvv - Sv * Sv) * wk
        vuk = (wvu - Sv * Su) * wk
        uuk = (wuu - Su * Su) * wk
        kern[2] = uuk - vvk
        kern[3] = 2 * vuk
        kern[4] = uuk + vvk
        kern[5] = wk

        ef2 = err_fac2[i]
        for a in range(6):
            for b in range(a, 6):
                # odd (center) and even sums do not covary
                if (a < 2) != (b < 2):
                    continue
                cov_raw[a, b] += kern[a] * kern[b] * ef2

    for a in range(6):
        for b in range(a + 1, 6):
            cov_raw[b, a] = cov_raw[a, b]

    vv = wvv * s0 - svv
    vu = wvu * s0 - svu
    uu = wuu * s0 - suu

    sums[0] = sv * df2
    sums[1] = su * df2
    sums[2] = (uu - vv) * df2
    sums[3] = 2 * vu * df2
    sums[4] = (uu + vv) * df2
    sums[5] = s0 * df2
