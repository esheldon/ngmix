import numpy as np
import pytest

from ngmix.observation import Observation
from ngmix.jacobian import DiagonalJacobian
from ngmix.gmix import GMix
from ngmix.pixels import make_pixels


@pytest.fixture()
def image_data():
    dims = (13, 15)
    rng = np.random.RandomState(seed=10)

    data = {}
    for key in ['image', 'weight', 'bmask', 'ormask', 'noise']:
        data[key] = rng.normal(size=dims)

        if key == 'weight':
            data[key] = np.exp(data[key])
        elif key in ['bmask', 'ormask']:
            data[key] = np.clip(data[key] * 100, 0, np.inf).astype(np.int32)
    data['jacobian'] = DiagonalJacobian(x=7, y=6, scale=0.25)
    data['meta'] = {'pi': 3.14}
    data['psf'] = Observation(
        image=rng.normal(size=dims), meta={'ispsf': True})
    data['psf_gmix'] = Observation(
        image=rng.normal(size=dims), meta={'ispsf': True},
        gmix=GMix(pars=rng.uniform(size=6)))
    data['gmix'] = GMix(pars=rng.uniform(size=6))
    return data


def test_observation_get_has(image_data):
    obs = Observation(image=image_data['image'])
    assert np.all(obs.image == image_data['image'])
    assert not obs.has_bmask()
    assert not obs.has_ormask()
    assert not obs.has_noise()
    assert not obs.has_psf()
    assert not obs.has_psf_gmix()
    assert not obs.has_gmix()

    obs = Observation(
        image=image_data['image'],
        weight=image_data['weight'])
    assert np.all(obs.weight == image_data['weight'])

    obs = Observation(
        image=image_data['image'],
        bmask=image_data['bmask'])
    assert np.all(obs.bmask == image_data['bmask'])
    assert obs.has_bmask()

    obs = Observation(
        image=image_data['image'],
        ormask=image_data['ormask'])
    assert np.all(obs.ormask == image_data['ormask'])
    assert obs.has_ormask()

    obs = Observation(
        image=image_data['image'],
        noise=image_data['noise'])
    assert np.all(obs.noise == image_data['noise'])
    assert obs.has_noise()

    obs = Observation(
        image=image_data['image'],
        psf=image_data['psf'])
    assert np.all(obs.psf.image == image_data['psf'].image)
    assert obs.has_psf()
    assert not obs.has_psf_gmix()

    obs = Observation(
        image=image_data['image'],
        gmix=image_data['gmix'])
    assert np.all(
        obs.gmix.get_full_pars() == image_data['gmix'].get_full_pars())
    assert obs.has_gmix()

    obs = Observation(
        image=image_data['image'],
        psf=image_data['psf_gmix'])
    assert np.all(obs.psf.image == image_data['psf_gmix'].image)
    assert obs.has_psf()
    assert obs.has_psf_gmix()
    assert np.all(obs.psf.gmix.get_full_pars() ==
                  obs.get_psf_gmix().get_full_pars())

    obs = Observation(
        image=image_data['image'],
        jacobian=image_data['jacobian'])
    assert (
        obs.jacobian.get_galsim_wcs() ==
        image_data['jacobian'].get_galsim_wcs())

    obs = Observation(
        image=image_data['image'],
        meta=image_data['meta'])
    assert obs.meta == image_data['meta']


def test_observation_set(image_data):
    obs = Observation(
        image=image_data['image'],
        weight=image_data['weight'],
        bmask=image_data['bmask'],
        ormask=image_data['ormask'],
        noise=image_data['noise'],
        jacobian=image_data['jacobian'],
        gmix=image_data['gmix'],
        psf=image_data['psf'],
        meta=image_data['meta'])

    rng = np.random.RandomState(seed=11)

    new_arr = rng.normal(size=image_data['image'].shape)
    assert np.all(obs.image != new_arr)
    obs.image = new_arr
    assert np.all(obs.image == new_arr)

    new_arr = np.exp(rng.normal(size=image_data['image'].shape))
    assert np.all(obs.weight != new_arr)
    obs.weight = new_arr
    assert np.all(obs.weight == new_arr)

    new_arr = (np.exp(rng.normal(size=image_data['image'].shape)) *
               100).astype(np.int32)
    assert np.all(obs.bmask != new_arr)
    obs.bmask = new_arr
    assert np.all(obs.bmask == new_arr)

    new_arr = (np.exp(rng.normal(size=image_data['image'].shape)) *
               100).astype(np.int32)
    assert np.all(obs.ormask != new_arr)
    obs.ormask = new_arr
    assert np.all(obs.ormask == new_arr)

    new_arr = rng.normal(size=image_data['image'].shape)
    assert np.all(obs.noise != new_arr)
    obs.noise = new_arr
    assert np.all(obs.noise == new_arr)

    new_jac = DiagonalJacobian(x=8, y=13, scale=1.2)
    assert new_jac.get_galsim_wcs() != obs.jacobian.get_galsim_wcs()
    obs.jacobian = new_jac
    assert new_jac.get_galsim_wcs() == obs.jacobian.get_galsim_wcs()

    new_meta = {'new': 5}
    assert obs.meta != new_meta
    obs.meta = new_meta
    assert obs.meta == new_meta

    new_meta = {'blue': 10}
    new_meta.update(obs.meta)
    assert obs.meta != new_meta
    obs.update_meta_data({'blue': 10})
    assert obs.meta == new_meta

    new_gmix = GMix(pars=rng.uniform(size=6))
    assert np.all(obs.gmix.get_full_pars() != new_gmix.get_full_pars())
    obs.gmix = new_gmix
    assert np.all(obs.gmix.get_full_pars() == new_gmix.get_full_pars())

    new_psf = Observation(
        image=rng.normal(size=obs.psf.image.shape), meta={'ispsf': True})
    assert np.all(obs.psf.image != new_psf.image)
    obs.psf = new_psf
    assert np.all(obs.psf.image == new_psf.image)


def test_observation_s2n(image_data):
    obs = Observation(
        image=image_data['image'],
        weight=image_data['weight'])
    s2n = obs.get_s2n()

    s2n_true = (
        np.sum(image_data['image']) /
        np.sqrt(np.sum(1.0/image_data['weight'])))

    assert np.allclose(s2n, s2n_true)


def test_observation_pixels(image_data):
    obs = Observation(
        image=image_data['image'],
        weight=image_data['weight'],
        jacobian=image_data['jacobian'])
    pixels = obs.pixels

    my_pixels = make_pixels(
        image_data['image'],
        image_data['weight'],
        image_data['jacobian'])

    assert np.all(pixels == my_pixels)


def test_observation_pixels_update_image(image_data):
    rng = np.random.RandomState(seed=11)
    obs = Observation(
        image=image_data['image'],
        weight=image_data['weight'],
        jacobian=image_data['jacobian'])

    new_arr = rng.normal(size=obs.image.shape)
    my_pixels = make_pixels(
        new_arr,
        image_data['weight'],
        image_data['jacobian'])

    assert np.all(obs.pixels != my_pixels)
    obs.image = new_arr
    assert np.all(obs.pixels == my_pixels)


def test_observation_pixels_update_weight(image_data):
    rng = np.random.RandomState(seed=11)
    obs = Observation(
        image=image_data['image'],
        weight=image_data['weight'],
        jacobian=image_data['jacobian'])

    new_arr = np.exp(rng.normal(size=obs.image.shape))
    my_pixels = make_pixels(
        image_data['image'],
        new_arr,
        image_data['jacobian'])

    assert np.all(obs.pixels != my_pixels)
    obs.weight = new_arr
    assert np.all(obs.pixels == my_pixels)


def test_observation_pixels_update_jacobian(image_data):
    obs = Observation(
        image=image_data['image'],
        weight=image_data['weight'],
        jacobian=image_data['jacobian'])
    new_jac = DiagonalJacobian(x=8, y=13, scale=1.2)

    my_pixels = make_pixels(
        image_data['image'],
        image_data['weight'],
        new_jac)

    assert np.all(obs.pixels != my_pixels)
    obs.jacobian = new_jac
    assert np.all(obs.pixels == my_pixels)


def test_observation_copy(image_data):
    obs = Observation(
        image=image_data['image'],
        weight=image_data['weight'],
        bmask=image_data['bmask'],
        ormask=image_data['ormask'],
        noise=image_data['noise'],
        jacobian=image_data['jacobian'],
        gmix=image_data['gmix'],
        psf=image_data['psf'],
        meta=image_data['meta'])

    new_obs = obs.copy()

    rng = np.random.RandomState(seed=11)

    new_arr = rng.normal(size=image_data['image'].shape)
    assert np.all(obs.image == new_obs.image)
    new_obs.image = new_arr
    assert np.all(obs.image != new_obs.image)

    new_arr = np.exp(rng.normal(size=image_data['image'].shape))
    assert np.all(obs.weight == new_obs.weight)
    new_obs.weight = new_arr
    assert np.all(obs.weight != new_obs.weight)

    new_arr = (np.exp(rng.normal(size=image_data['image'].shape)) *
               100).astype(np.int32)
    assert np.all(obs.bmask == new_obs.bmask)
    new_obs.bmask = new_arr
    assert np.all(obs.bmask != new_obs.bmask)

    new_arr = (np.exp(rng.normal(size=image_data['image'].shape)) *
               100).astype(np.int32)
    assert np.all(obs.ormask == new_obs.ormask)
    new_obs.ormask = new_arr
    assert np.all(obs.ormask != new_obs.ormask)

    new_arr = rng.normal(size=image_data['image'].shape)
    assert np.all(obs.noise == new_obs.noise)
    new_obs.noise = new_arr
    assert np.all(obs.noise != new_obs.noise)

    new_jac = DiagonalJacobian(x=8, y=13, scale=1.2)
    assert new_obs.jacobian.get_galsim_wcs() == obs.jacobian.get_galsim_wcs()
    new_obs.jacobian = new_jac
    assert new_obs.jacobian.get_galsim_wcs() != obs.jacobian.get_galsim_wcs()

    new_meta = {'new': 5}
    assert obs.meta == new_obs.meta
    new_obs.meta = new_meta
    assert obs.meta != new_obs.meta

    new_meta = {'blue': 10}
    new_meta.update(new_obs.meta)
    obs.update_meta_data({'blue': 10})
    assert obs.meta != new_obs.meta

    new_gmix = GMix(pars=rng.uniform(size=6))
    assert np.all(obs.gmix.get_full_pars() == new_obs.gmix.get_full_pars())
    new_obs.gmix = new_gmix
    assert np.all(obs.gmix.get_full_pars() != new_obs.gmix.get_full_pars())

    new_psf = Observation(
        image=rng.normal(size=obs.psf.image.shape), meta={'ispsf': True})
    assert np.all(obs.psf.image == new_obs.psf.image)
    new_obs.psf = new_psf
    assert np.all(obs.psf.image != new_obs.psf.image)
