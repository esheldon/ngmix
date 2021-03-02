import galsim
import numpy as np
import ngmix
import pytest


def _get_obs(rng, set_noise_image=False):

    noise = 0.005
    psf_noise = 1.0e-6

    scale = 0.263

    psf_fwhm = 0.9
    gal_fwhm = 0.7

    psf = galsim.Gaussian(fwhm=psf_fwhm)
    obj0 = galsim.Gaussian(fwhm=gal_fwhm)

    obj = galsim.Convolve(psf, obj0)

    psf_im = psf.drawImage(scale=scale).array
    im = obj.drawImage(scale=scale).array

    psf_im += rng.normal(scale=psf_noise, size=psf_im.shape)
    im += rng.normal(scale=noise, size=im.shape)

    cen = (np.array(im.shape)-1.0)/2.0
    psf_cen = (np.array(psf_im.shape)-1.0)/2.0

    j = ngmix.DiagonalJacobian(row=cen[0], col=cen[1], scale=scale)
    pj = ngmix.DiagonalJacobian(row=psf_cen[0], col=psf_cen[1], scale=scale)

    wt = im*0 + 1.0/noise**2
    psf_wt = psf_im*0 + 1.0/psf_noise**2

    psf_obs = ngmix.Observation(
        psf_im,
        weight=psf_wt,
        jacobian=pj,
    )

    if set_noise_image:
        nim = rng.normal(scale=noise, size=im.shape)
    else:
        nim = None

    obs = ngmix.Observation(
        im,
        weight=wt,
        noise=nim,
        jacobian=j,
        psf=psf_obs,
    )

    return obs


@pytest.mark.parametrize('psf', [None, 'gauss', 'fitgauss', 'galsim_obj'])
def test_metacal_smoke(psf):
    rng = np.random.RandomState(seed=100)

    obs = _get_obs(rng)

    if psf == 'galsim_obj':
        psf = galsim.Gaussian(fwhm=0.9)

    mpars = {'psf': psf}
    ngmix.metacal.get_all_metacal(obs, rng=rng, **mpars)


@pytest.mark.parametrize('fixnoise', [True, False])
def test_metacal_fixnoise(fixnoise):
    rng = np.random.RandomState(seed=100)

    obs = _get_obs(rng)

    mpars = {
        'psf': 'fitgauss',
        'fixnoise': fixnoise,
    }
    mdict = ngmix.metacal.get_all_metacal(obs, rng=rng, **mpars)

    for key, mobs in mdict.items():
        if fixnoise:
            assert mobs.weight[0, 0] == obs.weight[0, 0]/2
            assert mobs.pixels[0]['ierr'] == np.sqrt(obs.weight[0, 0]/2)
        else:
            assert mobs.weight[0, 0] == obs.weight[0, 0]
            assert mobs.pixels[0]['ierr'] == np.sqrt(obs.weight[0, 0])


def test_metacal_fixnoise_noise_image():
    rng = np.random.RandomState(seed=100)

    obs = _get_obs(rng, set_noise_image=True)

    mpars = {
        'psf': 'fitgauss',
        'use_noise_image': True,
    }
    mdict = ngmix.metacal.get_all_metacal(obs, rng=rng, **mpars)

    for key, mobs in mdict.items():
        assert mobs.weight[0, 0] == obs.weight[0, 0]/2
        assert mobs.pixels[0]['ierr'] == np.sqrt(obs.weight[0, 0]/2)
