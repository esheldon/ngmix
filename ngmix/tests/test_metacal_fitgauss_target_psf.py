"""
Tests of metacal.fitgauss_target_psf.get_fitgauss_target_psf
"""
import galsim
import numpy as np
import pytest
import ngmix

from ngmix.metacal import get_fitgauss_target_psf

SCALE = 0.2


def _get_moffat_psf_image():
    psf = galsim.Moffat(beta=2.5, fwhm=0.9).shear(g1=0.05, g2=0.0)
    return psf.drawImage(nx=48, ny=48, scale=SCALE).array


def _get_interpolated_image(im):
    return galsim.InterpolatedImage(galsim.Image(im, scale=SCALE))


def test_fitgauss_target_psf_gaussian_analytic():
    """
    Test the fitting returns a matching gaussian in the zero noise, round case
    """

    rng = np.random.RandomState(8282)

    psf = galsim.Gaussian(fwhm=0.9)
    im = psf.drawImage(nx=48, ny=48, scale=SCALE, method='no_pixel').array

    cen = (np.array(im.shape) - 1.0) / 2.0
    jacobian = ngmix.DiagonalJacobian(
        row=cen[0],
        col=cen[1],
        scale=0.2,
    )
    psf_obs = ngmix.Observation(
        im,
        jacobian=jacobian,
    )

    gauss = get_fitgauss_target_psf(psf_obs, rng=rng, flux=1)

    assert abs(gauss.sigma / psf.sigma - 1) < 0.005
    assert gauss.flux == 1.0


@pytest.mark.parametrize('s2n,mean_tol,trial_tol', [
    (100, 0.03, 0.15),
    (1000, 0.01, 0.05),
    (10000, 0.01, 0.05),
])
def test_fitgauss_target_psf_noise_stability(s2n, mean_tol, trial_tol):
    """
    Test that the derived kernel fwhm is flat with psf image S/N: the mean over
    noise realizations stays within a few percent of the noiseless value even
    at S/N = 100, with no knowledge of the noise level
    """
    rng = np.random.RandomState(3232)

    psf_im = _get_moffat_psf_image()

    cen = (np.array(psf_im.shape) - 1.0) / 2.0
    jacobian = ngmix.DiagonalJacobian(
        row=cen[0],
        col=cen[1],
        scale=0.2,
    )
    psf_obs = ngmix.Observation(
        psf_im,
        jacobian=jacobian,
    )
    ref_fwhm = get_fitgauss_target_psf(psf_obs, rng=rng).fwhm

    noise_sigma = np.sqrt((psf_im**2).sum()) / s2n

    ntrial = 20
    fwhms = np.zeros(ntrial)
    for i in range(ntrial):
        noisy_im = psf_im + rng.normal(scale=noise_sigma, size=psf_im.shape)
        noisy_psf_obs = ngmix.Observation(
            noisy_im,
            jacobian=jacobian,
        )

        fwhms[i] = get_fitgauss_target_psf(noisy_psf_obs, rng=rng).fwhm

    fracdev = fwhms / ref_fwhm - 1
    assert abs(fracdev.mean()) < mean_tol
    assert np.abs(fracdev).max() < trial_tol
