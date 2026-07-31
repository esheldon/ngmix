"""
numba kernel for the analytic derivatives of a psf-convolved
gaussian mixture model with respect to the simple model
parameters [cen1, cen2, g1, g2, T, flux], shared by the least
squares jacobian (FitModel.calc_jacobian) and the noise-power
sandwich covariance (noise_cov).

Every derivative of a rendered gaussian shares the render's
exponential:

    d val / d mu      = val * (Q delta)          Q = Sigma^-1
    d val / d Sigma_a = val * ((Q delta)^T dSigma_a (Q delta)
                               - tr(Q dSigma_a)) / 2

so the six derivative images cost one pixel pass at a small
multiple of a single render, replacing the many renders of the
central-difference evaluation.  The same fast exponential and
apodized chi^2 truncation as the fdiff renders are used, so
these are the derivatives of the actual fit objective.  In the
apodization band the render carries the window W(chi2), and
every term that flows through chi^2 picks up the window slope
as well:

    d [W E] / d p = E (W - 2 W') d(-chi^2/2)/dp

The flux derivative is the value image over the flux and is
formed by the caller.
"""
import numpy as np
from numba import njit

from ..fastexp_nb import (
    fexp, FASTEXP_MAX_CHI2, FASTEXP_APOD_CHI2,
    apod_window, apod_window_deriv,
)

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
    area: (npix,) array
        The pixel areas, so the value image matches
        GMix.make_image
    out: (6, npix) array, zeroed by the caller
        Filled with the value image in row 0 and the
        [cen1, cen2, g1, g2, T] derivative images in rows 1-5
    """
    ngauss = gpars.shape[0]
    npix = vv.size
    trs = np.zeros(3)

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
        norm = p / (TWO_PI * np.sqrt(det))

        # Q = Sigma^-1, and the trace terms tr(Q dSigma_a),
        # constant over the pixels
        w11 = icc / det
        w12 = -irc / det
        w22 = irr / det
        for a in range(3):
            trs[a] = (
                w11 * dcov[ig, a, 0]
                + 2.0 * w12 * dcov[ig, a, 1]
                + w22 * dcov[ig, a, 2]
            )

        for ipix in range(npix):
            dv = vv[ipix] - vcen
            du = uu[ipix] - ucen

            # Q delta
            qv = w11 * dv + w12 * du
            qu = w12 * dv + w22 * du
            chi2 = dv * qv + du * qu

            # the same fast exp and apodized truncation as the
            # fdiff renders: the fit's objective is the apodized
            # fastexp model, so its response derivatives are too
            if chi2 >= FASTEXP_MAX_CHI2 or chi2 < 0.0:
                continue
            val = norm * fexp(-0.5 * chi2) * area[ipix]
            if chi2 > FASTEXP_APOD_CHI2:
                w = apod_window(chi2)
                # terms that flow through chi^2 carry the window
                # slope: d[W E]/dp = E (W - 2 W') d(-chi^2/2)/dp
                valc = val * (w - 2.0 * apod_window_deriv(chi2))
                val *= w
            else:
                valc = val

            out[0, ipix] += val
            out[1, ipix] += valc * qv
            out[2, ipix] += valc * qu

            for a in range(3):
                quad = (
                    qv * qv * dcov[ig, a, 0]
                    + 2.0 * qv * qu * dcov[ig, a, 1]
                    + qu * qu * dcov[ig, a, 2]
                )

                out[3 + a, ipix] += 0.5 * (valc * quad - val * trs[a])
