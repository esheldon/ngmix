import galsim
import numpy as np
import ngmix
import ngmix.metacal.metacal
import pytest
from ._galsim_sims import _get_obs


@pytest.mark.parametrize('psf', ['gauss', 'fitgauss', 'galsim_obj', 'dilate'])
def test_metacal_smoke(psf):
    rng = np.random.RandomState(seed=100)

    obs = _get_obs(rng, noise=0.005)

    if psf == 'galsim_obj':
        psf = galsim.Gaussian(fwhm=0.9)

    if psf == 'dilate':
        expected_types = ngmix.metacal.METACAL_TYPES
    else:
        expected_types = ngmix.metacal.METACAL_MINIMAL_TYPES

    obs_dict = ngmix.metacal.get_all_metacal(obs, rng=rng, psf=psf)

    assert len(obs_dict) == len(expected_types)
    for type in expected_types:
        assert type in obs_dict

        mobs = obs_dict[type]
        assert mobs.image.shape == obs.image.shape
        assert np.all(mobs.image != obs.image)
        assert mobs.psf.image.shape == obs.psf.image.shape
        assert np.all(mobs.psf.image != obs.psf.image)

    assert np.all(obs_dict['1p'].image != obs_dict['1m'].image)
    assert np.all(obs_dict['2p'].image != obs_dict['2m'].image)
    assert np.all(obs_dict['noshear'].image != obs_dict['1p'].image)

    if psf == 'dilate':
        assert np.all(obs_dict['1p_psf'].image != obs_dict['1m_psf'].image)
        assert np.all(obs_dict['2p_psf'].image != obs_dict['2m_psf'].image)
        assert np.all(obs_dict['noshear'].image != obs_dict['2m_psf'].image)


@pytest.mark.parametrize('psf', ['gauss', 'fitgauss', 'galsim_obj'])
@pytest.mark.parametrize('send_rng', [True, False])
def test_metacal_send_rng(psf, send_rng):

    rng = np.random.RandomState(seed=100)
    obs = _get_obs(rng, noise=0.005)

    if psf == 'galsim_obj':
        psf = galsim.Gaussian(fwhm=0.9)

    kw = {
        'psf': psf, 'fixnoise': False, 'types': ['noshear', '1p', '1m'],
    }
    if send_rng:
        kw['rng'] = rng

    if psf == 'fitgauss' and not send_rng:
        with pytest.raises(ValueError):
            obs_dict1 = ngmix.metacal.get_all_metacal(obs, **kw)
        return
    else:
        obs_dict1 = ngmix.metacal.get_all_metacal(obs, **kw)
        obs_dict2 = ngmix.metacal.get_all_metacal(obs, **kw)

    for mtype in kw['types']:
        psf_image1 = obs_dict1[mtype].psf.image
        psf_image2 = obs_dict2[mtype].psf.image

        if send_rng:
            assert not np.allclose(psf_image1, psf_image2)
        else:
            assert np.allclose(psf_image1, psf_image2)


@pytest.mark.parametrize('psf', ['gauss', 'fitgauss', 'galsim_obj', 'dilate'])
def test_metacal_types_smoke(psf):
    rng = np.random.RandomState(seed=100)

    obs = _get_obs(rng, noise=0.005)

    if psf == 'galsim_obj':
        psf = galsim.Gaussian(fwhm=0.9)

    # shuld automatically add 1p
    types = ['noshear']
    obs_dict = ngmix.metacal.get_all_metacal(
        obs, rng=rng, psf=psf, types=types,
    )

    assert len(obs_dict) == len(types)
    for type in types + ['1p']:
        assert type in obs_dict

        mobs = obs_dict[type]
        assert mobs.image.shape == obs.image.shape
        assert np.all(mobs.image != obs.image)
        assert mobs.psf.image.shape == obs.psf.image.shape
        assert np.all(mobs.psf.image != obs.psf.image)


@pytest.mark.parametrize('otype', ['obs', 'obslist', 'mbobs'])
@pytest.mark.parametrize('set_noise_image', [True, False])
def test_metacal_fixnoise_smoke(otype, set_noise_image):
    rng = np.random.RandomState(seed=100)

    obs = _get_obs(rng, noise=0.005, set_noise_image=set_noise_image)

    if otype == 'obslist':
        oobs = obs
        obs = ngmix.ObsList()
        obs.append(oobs)
        check_type = ngmix.ObsList
    elif otype == 'mbobs':
        oobs = obs
        obslist = ngmix.ObsList()
        obslist.append(oobs)

        obs = ngmix.MultiBandObsList()
        obs.append(obslist)
        check_type = ngmix.MultiBandObsList
    else:
        check_type = ngmix.Observation

    resdict = ngmix.metacal.get_all_metacal(
        obs, rng=rng, use_noise_image=set_noise_image,
    )
    assert isinstance(resdict['noshear'], check_type)


@pytest.mark.parametrize('fixnoise', [True, False])
def test_metacal_fixnoise(fixnoise):
    rng = np.random.RandomState(seed=100)

    obs = _get_obs(rng, noise=0.005)

    mdict = ngmix.metacal.get_all_metacal(obs, rng=rng, fixnoise=fixnoise)

    for key, mobs in mdict.items():
        assert mobs.image.shape == obs.image.shape
        assert np.all(mobs.image != obs.image)
        assert mobs.psf.image.shape == obs.psf.image.shape
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

    mdict = ngmix.metacal.get_all_metacal(obs, rng=rng, use_noise_image=True)

    for key, mobs in mdict.items():
        assert mobs.image.shape == obs.image.shape
        assert np.all(mobs.image != obs.image)

        assert mobs.psf.image.shape == obs.psf.image.shape
        assert np.all(mobs.psf.image != obs.psf.image)

        assert mobs.weight[0, 0] == obs.weight[0, 0]/2
        assert mobs.pixels[0]['ierr'] == np.sqrt(obs.weight[0, 0]/2)


def test_metacal_errors():
    rng = np.random.RandomState(seed=100)
    obs = _get_obs(rng, noise=0.005, set_noise_image=True)

    with pytest.raises(ValueError):
        ngmix.metacal.get_all_metacal(obs=None, rng=rng)

    with pytest.raises(ValueError):
        ngmix.metacal.get_all_metacal(obs=obs, rng=None)

    with pytest.raises(ValueError):
        ngmix.metacal.MetacalFitGaussPSF(obs=obs, rng=None)

    with pytest.raises(TypeError):
        ngmix.metacal.metacal._check_shape(None)

    obs.set_psf(None)
    with pytest.raises(ValueError):
        ngmix.metacal.get_all_metacal(obs=obs, rng=rng)


def _do_test_low_psf_s2n():
    rng = np.random.RandomState(seed=100)
    noise = 1000

    for i in range(1000):
        obs = _get_obs(rng, noise=0.005, set_noise_image=True)

        with obs.psf.writeable():
            psf_im = obs.psf.image
            psf_wt = obs.psf.weight
            psf_im += rng.normal(scale=noise, size=psf_im.shape)
            psf_wt[:, :] = 1.0/noise**2

        ngmix.metacal.get_all_metacal(obs=obs, rng=rng, psf='fitgauss')


def test_low_psf_s2n():
    with pytest.raises(ngmix.BootPSFFailure):
        _do_test_low_psf_s2n()
