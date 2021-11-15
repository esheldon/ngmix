import copy
import numpy as np
import pytest

from ngmix.observation import Observation
from ngmix.pixels import make_pixels
from ngmix.jacobian import DiagonalJacobian
from ngmix.gmix import GMix


@pytest.fixture()
def image_data():
    dims = (13, 15)
    rng = np.random.RandomState(seed=10)

    data = {}
    for key in ['image', 'weight', 'bmask', 'ormask', 'noise', 'mfrac']:
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
    assert not obs.has_mfrac()

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
        mfrac=image_data['mfrac'])
    assert np.all(obs.mfrac == image_data['mfrac'])
    assert obs.has_mfrac()

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
        meta=image_data['meta'],
        mfrac=image_data['mfrac'],
    )

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
    obs.bmask = None
    assert not obs.has_bmask()

    new_arr = rng.normal(size=image_data['image'].shape)
    assert np.all(obs.mfrac != new_arr)
    obs.mfrac = new_arr
    assert np.all(obs.mfrac == new_arr)

    new_arr = (np.exp(rng.normal(size=image_data['image'].shape)) *
               100).astype(np.int32)
    assert np.all(obs.ormask != new_arr)
    obs.ormask = new_arr
    assert np.all(obs.ormask == new_arr)
    obs.ormask = None
    assert not obs.has_ormask()

    new_arr = rng.normal(size=image_data['image'].shape)
    assert np.all(obs.noise != new_arr)
    obs.noise = new_arr
    assert np.all(obs.noise == new_arr)
    obs.noise = None
    assert not obs.has_noise()

    new_jac = DiagonalJacobian(x=8, y=13, scale=1.2)
    assert new_jac.get_galsim_wcs() != obs.jacobian.get_galsim_wcs()
    obs.jacobian = new_jac
    assert new_jac.get_galsim_wcs() == obs.jacobian.get_galsim_wcs()

    new_meta = {'new': 5}
    assert obs.meta != new_meta
    obs.meta = new_meta
    assert obs.meta == new_meta
    with pytest.raises(TypeError):
        obs.meta = [10]
    obs.meta = None
    assert len(obs.meta) == 0

    new_meta = {'blue': 10}
    new_meta.update(obs.meta)
    assert obs.meta != new_meta
    obs.update_meta_data({'blue': 10})
    assert obs.meta == new_meta
    with pytest.raises(TypeError):
        obs.update_meta_data([10])

    new_gmix = GMix(pars=rng.uniform(size=6))
    assert np.all(obs.gmix.get_full_pars() != new_gmix.get_full_pars())
    obs.gmix = new_gmix
    assert np.all(obs.gmix.get_full_pars() == new_gmix.get_full_pars())
    obs.gmix = None
    assert not obs.has_gmix()
    with pytest.raises(RuntimeError):
        obs.get_gmix()

    new_psf = Observation(
        image=rng.normal(size=obs.psf.image.shape), meta={'ispsf': True})
    assert np.all(obs.psf.image != new_psf.image)
    obs.psf = new_psf
    assert np.all(obs.psf.image == new_psf.image)
    assert np.all(obs.get_psf().image == new_psf.image)
    obs.psf = None
    assert not obs.has_psf()
    with pytest.raises(RuntimeError):
        obs.get_psf()
    with pytest.raises(RuntimeError):
        obs.get_psf_gmix()


def test_observation_s2n(image_data):
    obs = Observation(
        image=image_data['image'],
        weight=image_data['weight'])
    s2n = obs.get_s2n()

    s2n_true = (
        np.sum(image_data['image']) /
        np.sqrt(np.sum(1.0/image_data['weight'])))

    assert np.allclose(s2n, s2n_true)

    # if we are not storing pixels, when we can have an all zero weight map
    # in this case s2n is -9999
    obs = Observation(
        image=image_data['image'],
        weight=np.zeros_like(obs.weight),
        store_pixels=False,
    )
    assert np.allclose(obs.get_s2n(), -9999)


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


def test_observation_nopixels(image_data):
    obs = Observation(
        image=image_data['image'],
        weight=image_data['weight'],
        jacobian=image_data['jacobian'],
        store_pixels=False,
    )
    pixels = obs.pixels
    assert pixels is None


def test_observation_pixels_noignore_zero(image_data):

    weight = image_data['weight'].copy()
    weight[:, 5] = 0.0

    obs = Observation(
        image=image_data['image'],
        weight=weight,
        jacobian=image_data['jacobian'],
        ignore_zero_weight=False,
    )
    pixels = obs.pixels

    my_pixels = make_pixels(
        image_data['image'],
        weight,
        image_data['jacobian'],
        ignore_zero_weight=False,
    )

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


@pytest.mark.parametrize('copy_type', ['copy', 'copy.copy', 'copy.deepcopy'])
def test_observation_copy(image_data, copy_type):
    obs = Observation(
        image=image_data['image'],
        weight=image_data['weight'],
        bmask=image_data['bmask'],
        ormask=image_data['ormask'],
        noise=image_data['noise'],
        jacobian=image_data['jacobian'],
        gmix=image_data['gmix'],
        psf=image_data['psf'],
        meta=image_data['meta'],
        mfrac=image_data['mfrac'],
    )

    if copy_type == 'copy':
        new_obs = obs.copy()
    elif copy_type == 'copy.copy':
        new_obs = copy.copy(obs)
    else:
        new_obs = copy.deepcopy(obs)

    assert new_obs == obs

    rng = np.random.RandomState(seed=11)

    for attr, atype in (
        ('image', 'f8'),
        ('weight', 'f8'),
        ('mfrac', 'f8'),
        ('noise', 'f8'),
        ('bmask', 'i4'),
        ('ormask', 'i4'),
    ):
        old_arr = getattr(obs, attr)
        new_arr = rng.normal(size=image_data['image'].shape).astype(atype)
        setattr(new_obs, attr, new_arr)
        assert new_obs != obs
        setattr(new_obs, attr, old_arr)
        assert new_obs == obs

    old_jac = obs.jacobian
    new_jac = DiagonalJacobian(x=8, y=13, scale=1.2)
    new_obs.jacobian = new_jac
    assert new_obs != obs
    new_obs.jacobian = old_jac
    assert new_obs == obs

    old_meta = obs.meta
    new_meta = {'new': 5}
    new_obs.meta = new_meta
    assert new_obs != obs
    new_obs.meta = old_meta
    assert new_obs == obs

    new_meta = {'blue': 10}
    new_meta.update(new_obs.meta)
    new_obs.meta = new_meta
    obs.update_meta_data({'blue': 10})
    assert obs == new_obs

    old_gmix = obs.gmix
    new_gmix = GMix(pars=rng.uniform(size=6))
    new_obs.gmix = new_gmix
    assert new_obs != obs
    new_obs.gmix = old_gmix
    assert new_obs == obs

    old_psf = obs.psf
    new_psf = Observation(
        image=rng.normal(size=obs.psf.image.shape),
        meta={'ispsf': True},
    )
    new_obs.psf = new_psf
    assert new_obs != obs
    new_obs.psf = old_psf
    assert new_obs == obs


def _dotest_readonly_attrs(obs):
    attrs = ['image', 'weight', 'bmask', 'ormask', 'noise', 'mfrac']
    val = 9999

    for attr in attrs:
        ref = getattr(obs, attr)

        with pytest.raises(ValueError):
            ref[5, 5] = val

        assert ref[5, 5] != val

    with pytest.raises(ValueError):
        obs.jacobian.set_cen(row=35, col=55)


def _dotest_writeable_attrs(obs):

    attrs = ['image', 'weight', 'bmask', 'ormask', 'noise', 'mfrac']
    val = 9999

    with obs.writeable():

        for attr in attrs:
            ref = getattr(obs, attr)

            ref[5, 5] = val

        obs.jacobian.set_cen(row=35, col=55)

    for attr in attrs:
        ref = getattr(obs, attr)

        assert ref[5, 5] == val

    row, col = obs.jacobian.get_cen()
    assert row == 35


def test_observation_context(image_data):

    obs = Observation(
        image=image_data['image'].copy(),
        weight=image_data['weight'].copy(),
        bmask=image_data['bmask'].copy(),
        ormask=image_data['ormask'].copy(),
        noise=image_data['noise'].copy(),
        jacobian=image_data['jacobian'],
        mfrac=image_data['mfrac'].copy(),
    )

    _dotest_readonly_attrs(obs)

    # should be a no-op outside of a context
    obs.writeable()
    _dotest_readonly_attrs(obs)

    with obs.writeable():
        _dotest_writeable_attrs(obs)


@pytest.mark.parametrize('ignore_zero_weight', [False, True])
@pytest.mark.parametrize('store_pixels', [False, True])
def test_observation_copy_propagate(image_data, store_pixels, ignore_zero_weight):
    obs = Observation(
        image=image_data['image'],
        store_pixels=store_pixels,
        ignore_zero_weight=ignore_zero_weight,
    )
    obs1 = obs.copy()
    assert obs.store_pixels == obs1.store_pixels
    assert obs.ignore_zero_weight == obs1.ignore_zero_weight


def test_observation_set_store_pixels(image_data):
    obs = Observation(
        image=image_data['image'],
        store_pixels=False,
    )
    assert obs.pixels is None
    obs.store_pixels = True
    assert obs.pixels is not None
    obs.store_pixels = False
    assert obs.pixels is None

    obs = Observation(
        image=image_data['image'],
        store_pixels=True,
    )
    assert obs.pixels is not None
    obs.store_pixels = False
    assert obs.pixels is None
    obs.store_pixels = True
    assert obs.pixels is not None

    obs = Observation(
        image=image_data['image'],
        store_pixels=False,
    )
    with obs.writeable():
        assert obs.pixels is None
        obs.store_pixels = True
        # we force an update here since this change should be expected
        assert obs.pixels is not None

        obs.store_pixels = False
        assert obs.pixels is None

        obs.store_pixels = True

    # final state should be this
    assert obs.pixels is not None


def test_observation_set_ignore_zero_weight(image_data):
    wgt = image_data["weight"].copy()
    wgt[0:2, 0:2] = 0

    obs = Observation(
        image=image_data['image'],
        weight=wgt,
        store_pixels=True,
        ignore_zero_weight=True,
    )
    assert obs.pixels is not None
    assert len(obs.pixels) == wgt.size-4
    obs.ignore_zero_weight = False
    assert len(obs.pixels) == wgt.size
    obs.ignore_zero_weight = True
    assert len(obs.pixels) == wgt.size-4

    obs = Observation(
        image=image_data['image'],
        weight=wgt,
        store_pixels=True,
        ignore_zero_weight=False,
    )
    assert obs.pixels is not None
    assert len(obs.pixels) == wgt.size
    obs.ignore_zero_weight = True
    assert len(obs.pixels) == wgt.size-4
    obs.ignore_zero_weight = False
    assert len(obs.pixels) == wgt.size

    obs = Observation(
        image=image_data['image'],
        weight=wgt,
        store_pixels=True,
        ignore_zero_weight=True,
    )
    with obs.writeable():
        assert obs.pixels is not None
        assert len(obs.pixels) == wgt.size-4

        # we force a change here since this is expected
        obs.ignore_zero_weight = False
        assert len(obs.pixels) == wgt.size

        obs.ignore_zero_weight = True
        assert len(obs.pixels) == wgt.size-4

        obs.ignore_zero_weight = False

    # final state should be this
    assert len(obs.pixels) == wgt.size
