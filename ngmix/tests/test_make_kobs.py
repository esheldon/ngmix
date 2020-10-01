import numpy as np
import pytest

from ngmix.jacobian import DiagonalJacobian
from ngmix.observation import Observation, ObsList, make_iilist, make_kobs


@pytest.fixture()
def obslist_data():
    dims = (12, 12)
    rng = np.random.RandomState(seed=10)
    jac = DiagonalJacobian(x=5.5, y=5.5, scale=0.25)

    obslist = ObsList()
    obslist.append(
        Observation(
            image=rng.normal(size=dims),
            jacobian=jac.copy(),
            weight=np.exp(rng.normal(size=dims)),
            meta={"blue": 10},
        )
    )
    obslist.append(
        Observation(
            image=rng.normal(size=dims),
            jacobian=jac.copy(),
            weight=np.exp(rng.normal(size=dims)),
            meta={"blue": 8},
        )
    )

    psf = Observation(
        image=rng.normal(size=dims),
        weight=np.exp(rng.normal(size=dims)),
        jacobian=jac.copy(),
        meta={"blue": 5},
    )
    obslist[1].psf = psf

    return obslist


def test_make_iilist_smoke(obslist_data):
    iilist, dim, dk = make_iilist(obslist_data)

    # sanity check
    assert len(iilist) == 1
    assert len(iilist[0]) == 2
    assert dim % 2 == 1
    assert dk > 0

    # double check the max is correct
    # only works with galsim 2 but that is ok
    for _ll in iilist:
        for _dd in _ll:
            if _dd["psf_ii"] is not None:
                _dim = 1 + _dd["psf_ii"].getGoodImageSize(
                    _dd["psf_ii"].nyquist_scale
                )
                _dk = _dd["psf_ii"].stepk
            else:
                _dim = 1 + _dd["ii"].getGoodImageSize(
                    _dd["ii"].nyquist_scale
                )
                _dk = _dd["ii"].stepk
            assert _dim <= dim
            if dim == _dim:
                assert _dk == dk

    rng = np.random.RandomState(seed=10)
    dims = (12, 12)
    for obs, sdata in zip(obslist_data, iilist[0]):
        assert sdata["scale"] == 0.25
        assert sdata["wcs"].pixelArea() == 0.25**2
        assert sdata["wcs"].isLocal()
        assert sdata["wcs"].isUniform()

        assert sdata["meta"] == obs.meta

        im = rng.normal(size=dims)
        assert np.allclose(sdata["ii"].image.array, im)
        assert np.allclose(sdata["realspace_gsimage"].array, im)
        assert np.allclose(sdata["weight"], np.exp(rng.normal(size=dims)))

        if obs.has_psf():
            assert sdata["psf_meta"] == obs.psf.meta
            psf_im = rng.normal(size=dims)
            assert np.allclose(sdata["psf_ii"].image.array, psf_im / np.sum(psf_im))
            assert np.allclose(sdata["psf_weight"], np.exp(rng.normal(size=dims)))

        else:
            assert sdata["psf_ii"] is None
            assert sdata["psf_weight"] is None
            assert sdata["psf_meta"] is None


def test_make_kobs(obslist_data):
    iilist, dim, dk = make_iilist(obslist_data)
    kmbobs = make_kobs(obslist_data)
    assert len(kmbobs) == 1
    assert len(kmbobs[0]) == 2

    assert not kmbobs[0][0].has_psf()
    assert kmbobs[0][1].has_psf()

    for i, obs in enumerate(kmbobs[0]):
        wgt = obslist_data[i].weight.max()*0.5 / dim / dim
        assert np.allclose(obs.weight.array, wgt)
        assert obs.kimage.array.shape == (dim, dim)
        assert obs.kimage.scale == dk
        assert all(obs.meta[k] == obslist_data[i].meta[k] for k in obslist_data[i].meta)
        if i == 1:
            wgt = obslist_data[i].psf.weight.max()*0.5 / dim / dim
            assert np.allclose(obs.psf.weight.array, wgt)
            assert obs.psf.kimage.array.shape == (dim, dim)
            assert obs.psf.kimage.scale == dk
            assert all(
                obs.psf.meta[k] == obslist_data[i].psf.meta[k]
                for k in obslist_data[i].psf.meta
            )
