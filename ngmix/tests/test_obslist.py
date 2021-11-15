import copy
import numpy as np
import pytest

from ngmix.observation import Observation, ObsList


def test_obslist_smoke():
    rng = np.random.RandomState(seed=11)
    meta = {'duh': 5}
    obslist = ObsList(meta=meta)
    for _ in range(3):
        obslist.append(Observation(image=rng.normal(size=(13, 15))))

    assert obslist.meta == meta
    rng = np.random.RandomState(seed=11)
    for obs in obslist:
        assert np.all(obs.image == rng.normal(size=(13, 15)))


@pytest.mark.parametrize('copy_type', ['copy', 'copy.copy', 'copy.deepcopy'])
def test_obslist_copy(copy_type):
    rng = np.random.RandomState(seed=11)
    meta = {'duh': 5}
    obslist = ObsList(meta=meta)
    for _ in range(3):
        obslist.append(Observation(image=rng.normal(size=(13, 15))))

    if copy_type == 'copy':
        new_obslist = obslist.copy()
    elif copy_type == 'copy.copy':
        new_obslist = copy.copy(obslist)
    else:
        new_obslist = copy.deepcopy(obslist)

    assert new_obslist == obslist

    obslist[1].image = obslist[1].image*5
    assert new_obslist != obslist

    with pytest.raises(ValueError):
        obslist == 3


def test_obslist_set():
    rng = np.random.RandomState(seed=11)
    meta = {'duh': 5}
    obslist = ObsList(meta=meta)
    for _ in range(3):
        obslist.append(Observation(image=rng.normal(size=(13, 15))))

    assert obslist.meta == meta
    new_meta = {'blah': 6}
    obslist.meta = new_meta
    assert obslist.meta == new_meta
    obslist.meta = None
    assert len(obslist.meta) == 0
    with pytest.raises(TypeError):
        obslist.meta = [10]
    with pytest.raises(TypeError):
        obslist.set_meta([10])

    new_meta = {'bla': 6}
    new_meta.update(obslist.meta)
    obslist.update_meta_data({'bla': 6})
    assert obslist.meta == new_meta
    with pytest.raises(TypeError):
        obslist.update_meta_data([10])

    rng = np.random.RandomState(seed=12)
    new_obs = Observation(image=rng.normal(size=(13, 15)))
    rng = np.random.RandomState(seed=11)
    for obs in obslist:
        assert np.all(obs.image == rng.normal(size=(13, 15)))
    obslist[1] = new_obs
    assert np.all(obslist[1].image == new_obs.image)


def test_obslist_s2n():
    rng = np.random.RandomState(seed=11)
    obslist = ObsList()
    numer = 0
    denom = 0
    for _ in range(3):
        obs = Observation(image=rng.normal(size=(13, 15)))
        numer += np.sum(obs.image)
        denom += np.sum(1.0/obs.weight)
        obslist.append(obs)

    s2n = obslist.get_s2n()
    assert s2n == numer / np.sqrt(denom)


def test_obslist_s2n_zeroweight():
    rng = np.random.RandomState(seed=11)
    obslist = ObsList()
    for _ in range(3):
        obs = Observation(
            image=rng.normal(size=(13, 15)),
            weight=np.zeros((13, 15)),
            store_pixels=False,
        )
        obslist.append(obs)

    assert np.allclose(obslist.get_s2n(), -9999)


def test_obslist_append_err():
    obslist = ObsList()
    with pytest.raises(AssertionError):
        obslist.append(None)
