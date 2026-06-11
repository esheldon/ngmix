"""
Noise-robust derivation of the round gaussian target reconvolution PSF
for metacal.

Drop-in replacement for metacal._get_gauss_target_psf (originally from galsim
tests/test_metacal.py).  That version finds the pinning scale with a pixelwise
min over the k image; the min is an extreme value statistic over the noisy k
pixels, so the derived kernel grows rapidly as the psf image S/N drops (e.g.
+60% in fwhm at S/N ~ 100 for a Moffat).

Here the threshold crossing is instead found on the azimuthally averaged
profile, which supresses the noise by the sqrt of the number of modes per
annulus, and the crossing is interpolated in log between annuli to remove the
bin quantization.  The result is independent of the psf image noise level at
the percent level for S/N >~ 100, and no knowledge of the noise level is
needed.
"""
import numpy as np
import galsim

# the threshold pair; the ratio SMALLER_KVAL/SMALL_KVAL = 0.3 is the built-in
# suppression margin (same as original _get_gauss_target_psf).  3e-2 was chosen
# from scans of small_kval compared to the original 1e-2 it reduces the mid-k
# power amplification to <~1.5 on real cell PSFs and moves the crossing to
# smaller k where the annular profile S/N is higher (noise immune at psf image
# S/N ~ 100), for a ~10% larger kernel.  Much smaller values (~<3e-3) are
# dangerous: the threshold approaches the wing plateau of real coadd PSF
# profiles

SMALL_KVAL = 3.0e-2    # find the k where the given psf hits this kvalue
SMALLER_KVAL = 9.0e-3  # target PSF will have this kvalue at the same k


def get_gauss_target_psf(
    psf,
    flux,
    small_kval=SMALL_KVAL,
    smaller_kval=SMALLER_KVAL,
):
    """
    get a round gaussian target reconvolution psf, pinned below the
    input psf profile at the scale where the psf profile falls to
    small_kval of its flux

    assumes the psf is centered

    Parameters
    ----------
    psf: galsim object
        the psf, e.g. a galsim.InterpolatedImage
    flux: float
        flux of the output gaussian
    small_kval: float
        find the k where the psf profile falls to this fraction of flux
    smaller_kval: float
        the target gaussian has this kvalue at that k

    Returns
    -------
    galsim.Gaussian
    """
    dk, prof = get_annular_kprofile(psf)
    k_cross = get_interpolated_crossing(dk, prof, small_kval * psf.flux)
    ksq_max = k_cross**2

    # exp(-0.5 * ksq_max * sigma_sq) = smaller_kval
    sigma_sq = -2.0 * np.log(smaller_kval) / ksq_max

    return galsim.Gaussian(sigma=np.sqrt(sigma_sq), flux=flux)


def get_annular_kprofile(psf):
    """
    azimuthally average the real part of the psf k image in annuli of
    width dk = stepk/4

    empty annuli beyond the corners of the k image are set to +inf so
    they can never trigger a threshold crossing

    Parameters
    ----------
    psf: galsim object
        the psf, e.g. a galsim.InterpolatedImage

    Returns
    -------
    dk, prof
    """
    dk = psf.stepk / 4.0

    kim = psf.drawKImage(scale=dk)
    karr_r = kim.real.array

    nk = karr_r.shape[0]
    kx, ky = np.meshgrid(
        np.arange(-nk / 2, nk / 2), np.arange(-nk / 2, nk / 2),
    )
    kmag = np.sqrt(kx**2 + ky**2) * dk

    ibin = np.rint(kmag / dk).astype(int).ravel()
    nbin = ibin.max() + 1
    num = np.bincount(ibin, minlength=nbin)

    prof = np.full(nbin, np.inf)
    np.divide(
        np.bincount(ibin, weights=karr_r.ravel(), minlength=nbin),
        num, out=prof, where=num > 0,
    )

    return dk, prof


def get_interpolated_crossing(dk, prof, thresh):
    """
    the k where the annular profile crosses below the threshold, interpolated
    between the first annulus below the threshold and the one before it.
    Interpolate in log(prof) since the profile decays roughly exponentially
    there
    """
    i = int(np.where(prof < thresh)[0].min())
    if i == 0:
        return 0.0

    p0, p1 = prof[i - 1], prof[i]
    if p0 > 0 and p1 > 0:
        frac = (np.log(thresh) - np.log(p0)) / (np.log(p1) - np.log(p0))
    else:
        frac = (thresh - p0) / (p1 - p0)

    return (i - 1 + frac) * dk
