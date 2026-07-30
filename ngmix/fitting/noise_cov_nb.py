"""
numba kernel for the analytic derivative images of a
psf-convolved gaussian mixture model with respect to the simple
model parameters [cen1, cen2, g1, g2, T, flux].

Every derivative of a rendered gaussian shares the render's
exponential:

    d val / d mu      = val * (Q delta)          Q = Sigma^-1
    d val / d Sigma_a = val * ((Q delta)^T dSigma_a (Q delta)
                               - tr(Q dSigma_a)) / 2

so the six derivative images cost one pixel pass at a small
multiple of a single render, replacing the twelve renders of
the central-difference evaluation.  The flux derivative is the
value image over the flux and is formed by the caller.
"""
import numpy as np
from numba import njit

TWO_PI = 2.0 * np.pi


@njit(cache=True)
def deriv_images(gpars, dcov, vv, uu, area, out):
    """
    accumulate the value image and its derivatives with respect
    to [cen1 (v), cen2 (u), g1, g2, T].

    Parameters
    ----------
    gpars: (ngauss, 6) array
        The composed (model convolved with psf) gaussians as
        [p, v, u, irr, irc, icc] in sky coordinates
    dcov: (ngauss, 3, 3) array
        d(irr, irc, icc) of each composed gaussian with respect
        to the model parameters [g1, g2, T]
    vv, uu: (npix,) arrays
        The pixel sky coordinates
    area: float
        The pixel area, so the value image matches
        GMix.make_image
    out: (6, npix) array, zeroed by the caller
        Filled with the value image in row 0 and the
        [cen1, cen2, g1, g2, T] derivative images in rows 1-5
    """
    ngauss = gpars.shape[0]
    npix = vv.size

    for ig in range(ngauss):
        p = gpars[ig, 0]
        vcen = gpars[ig, 1]
        ucen = gpars[ig, 2]
        irr = gpars[ig, 3]
        irc = gpars[ig, 4]
        icc = gpars[ig, 5]

        det = irr * icc - irc * irc
        if det <= 0.0:
            continue
        norm = p * area / (TWO_PI * np.sqrt(det))

        for ipix in range(npix):
            dv = vv[ipix] - vcen
            du = uu[ipix] - ucen
            # Q delta with Q = Sigma^-1
            qv = (icc * dv - irc * du) / det
            qu = (-irc * dv + irr * du) / det
            chi2 = dv * qv + du * qu
            if chi2 > 50.0 or chi2 < 0.0:
                continue
            val = norm * np.exp(-0.5 * chi2)

            out[0, ipix] += val
            out[1, ipix] += val * qv
            out[2, ipix] += val * qu
            for a in range(3):
                da = dcov[ig, a, 0]
                db = dcov[ig, a, 1]
                dc = dcov[ig, a, 2]
                quad = qv * qv * da + 2.0 * qv * qu * db \
                    + qu * qu * dc
                tr = (icc * da - 2.0 * irc * db + irr * dc) / det
                out[3 + a, ipix] += 0.5 * val * (quad - tr)
