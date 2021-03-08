import galsim
import numpy as np
import ngmix
import pytest
from ._galsim_sims import _get_obs


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
        assert np.all(mobs.image != obs.image)
        assert np.all(mobs.psf.image != obs.psf.image)
        if fixnoise:
            assert mobs.weight[0, 0] == obs.weight[0, 0]/2
            assert mobs.pixels[0]['ierr'] == np.sqrt(obs.weight[0, 0]/2)
        else:
            assert mobs.weight[0, 0] == obs.weight[0, 0]
            assert mobs.pixels[0]['ierr'] == np.sqrt(obs.weight[0, 0])


def test_metacal_fixnoise_noise_image():
    rng = np.random.RandomState(seed=100)

    obs = _get_obs(rng, set_noise_image=True)
    assert obs.has_noise()

    mpars = {
        'psf': 'fitgauss',
        'use_noise_image': True,
    }
    mdict = ngmix.metacal.get_all_metacal(obs, rng=rng, **mpars)

    for key, mobs in mdict.items():
        assert np.all(mobs.image != obs.image)
        assert np.all(mobs.psf.image != obs.psf.image)

        assert mobs.weight[0, 0] == obs.weight[0, 0]/2
        assert mobs.pixels[0]['ierr'] == np.sqrt(obs.weight[0, 0]/2)
