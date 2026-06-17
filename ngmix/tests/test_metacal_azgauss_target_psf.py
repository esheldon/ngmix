"""
Tests of metacal.azgauss_target_psf.get_azgauss_target_psf, the noise-robust
derivation of the round gaussian target reconvolution psf used by
psf='azgauss'.

The key property is stability against noise in the psf image: the
threshold crossing is found on the azimuthally averaged k profile, so
the derived kernel size is flat with psf image S/N.  The old pixelwise
min derivation (_get_gauss_target_psf) is an extreme value statistic
and the kernel grows rapidly as the S/N drops.
"""
import galsim
import numpy as np
import pytest

from ngmix.metacal import get_azgauss_target_psf
from ngmix.metacal.azgauss_target_psf import SMALL_KVAL, SMALLER_KVAL
from ngmix.metacal.metacal import _get_gauss_target_psf

SCALE = 0.2


def _get_moffat_psf_image():
    psf = galsim.Moffat(beta=2.5, fwhm=0.9).shear(g1=0.05, g2=0.0)
    return psf.drawImage(nx=48, ny=48, scale=SCALE).array


def _get_interpolated_image(im):
    return galsim.InterpolatedImage(galsim.Image(im, scale=SCALE))


def test_gauss_target_psf_gaussian_analytic():
    """
    for a gaussian input psf the target sigma is known analytically:
    the crossing of exp(-0.5 k^2 sigma^2) = small_kval gives
    sigma_target^2 = sigma^2 * log(smaller_kval) / log(small_kval)
    """
    psf = galsim.Gaussian(fwhm=0.9)
    im = psf.drawImage(nx=48, ny=48, scale=SCALE).array

    gauss = get_azgauss_target_psf(_get_interpolated_image(im), flux=1.0)

    expected_sigma = psf.sigma * np.sqrt(
        np.log(SMALLER_KVAL) / np.log(SMALL_KVAL)
    )
    assert abs(gauss.sigma / expected_sigma - 1) < 0.005
    assert gauss.flux == 1.0


@pytest.mark.parametrize('s2n,mean_tol,trial_tol', [
    (100, 0.03, 0.15),
    (1000, 0.01, 0.05),
    (10000, 0.01, 0.05),
])
def test_gauss_target_psf_noise_stability(s2n, mean_tol, trial_tol):
    """
    the derived kernel fwhm is flat with psf image S/N: the mean over
    noise realizations stays within a few percent of the noiseless
    value even at S/N = 100, with no knowledge of the noise level
    """
    rng = np.random.RandomState(8312)

    psf_im = _get_moffat_psf_image()
    ref_fwhm = get_azgauss_target_psf(
        _get_interpolated_image(psf_im), flux=1.0,
    ).fwhm

    noise_sigma = np.sqrt((psf_im**2).sum()) / s2n

    ntrial = 20
    fwhms = np.zeros(ntrial)
    for i in range(ntrial):
        noisy_im = psf_im + rng.normal(scale=noise_sigma, size=psf_im.shape)
        fwhms[i] = get_azgauss_target_psf(
            _get_interpolated_image(noisy_im), flux=1.0,
        ).fwhm

    fracdev = fwhms / ref_fwhm - 1
    assert abs(fracdev.mean()) < mean_tol
    assert np.abs(fracdev).max() < trial_tol


def test_gauss_target_psf_beats_pixelwise_min():
    """
    at psf image S/N = 100 the old pixelwise min derivation inflates
    the kernel fwhm by tens of percent while the azimuthal average
    derivation stays flat at the percent level
    """
    rng = np.random.RandomState(8312)

    psf_im = _get_moffat_psf_image()
    ii = _get_interpolated_image(psf_im)
    ref_new = get_azgauss_target_psf(ii, flux=1.0).fwhm
    ref_old = _get_gauss_target_psf(ii, flux=1.0).fwhm

    noise_sigma = np.sqrt((psf_im**2).sum()) / 100

    ntrial = 20
    fwhms_new = np.zeros(ntrial)
    fwhms_old = np.zeros(ntrial)
    for i in range(ntrial):
        noisy_im = psf_im + rng.normal(scale=noise_sigma, size=psf_im.shape)
        ii = _get_interpolated_image(noisy_im)
        fwhms_new[i] = get_azgauss_target_psf(ii, flux=1.0).fwhm
        fwhms_old[i] = _get_gauss_target_psf(ii, flux=1.0).fwhm

    fracdev_new = fwhms_new.mean() / ref_new - 1
    fracdev_old = fwhms_old.mean() / ref_old - 1

    assert abs(fracdev_new) < 0.03
    assert fracdev_old > 0.25
