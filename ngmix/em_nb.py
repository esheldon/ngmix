from numba import njit

from .gexceptions import GMixRangeError
from .gmix_nb import (
    gauss2d_set,
    gmix_set_norms,
    gauss2d_set_norm,
    gmix_get_e1e2T,
)
from .fastexp import fexp, FASTEXP_MAX_CHI2


@njit
def em_run(conf, pixels, sums, gmix):
    """
    run the EM algorithm

    parameters
    ----------
    conf: array
        Should have fields

            sky_guess: guess for the sky
            counts: counts in the image
            tol: tolerance for stopping
            maxiter: maximum number of iterations
            pixel_scale: pixel scale

    pixels: pixel array
        for the image/jacobian
    gmix: gaussian mixture
        Initialized to the starting guess
    """

    gmix_set_norms(gmix)
    tol = conf["tol"]
    counts = conf["counts"]

    area = pixels.size * conf["pixel_scale"] * conf["pixel_scale"]

    nsky = conf["sky_guess"] / counts
    psky = conf["sky_guess"] / (counts / area)

    T_last = e1_last = e2_last = -9999.0

    for i in range(conf["maxiter"]):
        skysum = 0.0
        clear_sums(sums)

        for pixel in pixels:

            # this fills some fields of sums, as well as return
            gtot = do_scratch_sums(pixel, gmix, sums)

            gtot += nsky
            if gtot == 0.0:
                raise GMixRangeError("gtot == 0")

            imnorm = pixel["val"] / counts
            skysum += nsky * imnorm / gtot
            igrat = imnorm / gtot

            # multiply sums by igrat, among other things
            do_sums(sums, igrat)

        gmix_set_from_sums(gmix, sums)

        psky = skysum
        nsky = psky / area

        e1, e2, T = gmix_get_e1e2T(gmix)

        frac_diff = abs((T - T_last) / T)
        e1diff = abs(e1 - e1_last)
        e2diff = abs(e2 - e2_last)

        if frac_diff < tol and e1diff < tol and e2diff < tol:
            break

        T_last, e1_last, e2_last = T, e1, e2

    numiter = i + 1
    return numiter, frac_diff


@njit
def do_scratch_sums(pixel, gmix, sums):
    """
    do the basic sums for this pixel, using
    scratch space in the sums struct
    """

    gtot = 0.0

    n_gauss = gmix.size
    for i in range(n_gauss):
        gauss = gmix[i]
        tsums = sums[i]

        v = pixel["v"]
        u = pixel["u"]

        vdiff = v - gauss["row"]
        udiff = u - gauss["col"]

        u2 = udiff * udiff
        v2 = vdiff * vdiff
        uv = udiff * vdiff

        chi2 = gauss["dcc"] * v2 + gauss["drr"] * u2 - 2.0 * gauss["drc"] * uv

        if chi2 < FASTEXP_MAX_CHI2 and chi2 >= 0.0:
            tsums["gi"] = gauss["pnorm"] * fexp(-0.5 * chi2)
        else:
            tsums["gi"] = 0.0

        gtot += tsums["gi"]

        tsums["trowsum"] = v * tsums["gi"]
        tsums["tcolsum"] = u * tsums["gi"]
        tsums["tv2sum"] = v2 * tsums["gi"]
        tsums["tuvsum"] = uv * tsums["gi"]
        tsums["tu2sum"] = u2 * tsums["gi"]

    return gtot


@njit
def do_sums(sums, igrat):
    """
    do the sums based on the scratch values
    """

    n_gauss = sums.size
    for i in range(n_gauss):
        tsums = sums[i]

        # wtau is gi[pix]/gtot[pix]*imnorm[pix]
        # which is Dave's tau*imnorm = wtau
        wtau = tsums["gi"] * igrat

        tsums["pnew"] += wtau

        # row*gi/gtot*imnorm
        tsums["rowsum"] += tsums["trowsum"] * igrat
        tsums["colsum"] += tsums["tcolsum"] * igrat
        tsums["u2sum"] += tsums["tu2sum"] * igrat
        tsums["uvsum"] += tsums["tuvsum"] * igrat
        tsums["v2sum"] += tsums["tv2sum"] * igrat


@njit
def gmix_set_from_sums(gmix, sums):
    """
    fill the gaussian mixture from the em sums
    """

    n_gauss = gmix.size
    for i in range(n_gauss):

        tsums = sums[i]
        gauss = gmix[i]

        p = tsums["pnew"]
        pinv = 1.0 / p

        gauss2d_set(
            gauss,
            p,
            tsums["rowsum"] * pinv,
            tsums["colsum"] * pinv,
            tsums["v2sum"] * pinv,
            tsums["uvsum"] * pinv,
            tsums["u2sum"] * pinv,
        )

        gauss2d_set_norm(gauss)


@njit
def clear_sums(sums):
    """
    set all sums to zero
    """
    sums["gi"][:] = 0.0

    sums["trowsum"][:] = 0.0
    sums["tcolsum"][:] = 0.0
    sums["tu2sum"][:] = 0.0
    sums["tuvsum"][:] = 0.0
    sums["tv2sum"][:] = 0.0

    # sums over all pixels
    sums["pnew"][:] = 0.0
    sums["rowsum"][:] = 0.0
    sums["colsum"][:] = 0.0
    sums["u2sum"][:] = 0.0
    sums["uvsum"][:] = 0.0
    sums["v2sum"][:] = 0.0
