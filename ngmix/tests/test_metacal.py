import galsim
import numpy as np
import ngmix
import pytest
from ._galsim_sims import _get_obs


@pytest.mark.parametrize('psf', [None, 'gauss', 'fitgauss', 'galsim_obj'])
def test_metacal_smoke(psf):
    rng = np.random.RandomState(seed=100)

    obs = _get_obs(rng, noise=0.005)

    if psf == 'galsim_obj':
        psf = galsim.Gaussian(fwhm=0.9)

    if psf is None:
        expected_types = ngmix.metacal.METACAL_TYPES
    else:
        expected_types = ngmix.metacal.METACAL_MINIMAL_TYPES

    mpars = {'psf': psf}
    obs_dict = ngmix.metacal.get_all_metacal(obs, rng=rng, **mpars)

    assert len(obs_dict) == len(expected_types)
    for type in expected_types:
        assert type in obs_dict


@pytest.mark.parametrize('psf', [None, 'gauss', 'fitgauss', 'galsim_obj'])
def test_metacal_types_smoke(psf):
    rng = np.random.RandomState(seed=100)

    obs = _get_obs(rng, noise=0.005)

    if psf == 'galsim_obj':
        psf = galsim.Gaussian(fwhm=0.9)

    types = ['noshear', '1p']
    mpars = {'psf': psf, 'types': types}
    obs_dict = ngmix.metacal.get_all_metacal(obs, rng=rng, **mpars)

    assert len(obs_dict) == len(types)
    for type in types:
        assert type in obs_dict


@pytest.mark.parametrize('fixnoise', [True, False])
def test_metacal_fixnoise(fixnoise):
    rng = np.random.RandomState(seed=100)

    obs = _get_obs(rng, noise=0.005)

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

    obs = _get_obs(rng, noise=0.005, set_noise_image=True)
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
