import copy
import numpy as np
import pytest

from ngmix.observation import (
    Observation, ObsList, MultiBandObsList, get_mb_obs)


def test_multibandobslist_smoke():
    rng = np.random.RandomState(seed=11)
    meta = {'duh': 5}
    mbobs = MultiBandObsList(meta=meta)

    for _ in range(5):
        obslist = ObsList()
        for _ in range(3):
            obslist.append(Observation(image=rng.normal(size=(13, 15))))
        mbobs.append(obslist)

    assert mbobs.meta == meta
    rng = np.random.RandomState(seed=11)
    for obslist in mbobs:
        for obs in obslist:
            assert np.all(obs.image == rng.normal(size=(13, 15)))


@pytest.mark.parametrize('copy_type', ['copy', 'copy.copy', 'copy.deepcopy'])
def test_multibandobslist_copy(copy_type):
    rng = np.random.RandomState(seed=11)
    meta = {'duh': 5}

    mbobs = MultiBandObsList(meta=meta)

    for iband in range(2):
        bmeta = {'iband': iband}
        obslist = ObsList(meta=bmeta)
        for iepoch in range(3):
            emeta = {'iepoch': iepoch}
            obs = Observation(
                image=rng.normal(size=(13, 15)),
                meta=emeta,
            )

            obslist.append(obs)
        mbobs.append(obslist)

    if copy_type == 'copy':
        new_mbobs = mbobs.copy()
    elif copy_type == 'copy.copy':
        new_mbobs = copy.copy(mbobs)
    else:
        new_mbobs = copy.deepcopy(mbobs)

    assert new_mbobs == mbobs

    new_mbobs[1][1].image = new_mbobs[1][1].image * 5
    assert new_mbobs != mbobs

    with pytest.raises(ValueError):
        mbobs == 3


def test_multibandobslist_set():
    rng = np.random.RandomState(seed=11)
    meta = {'duh': 5}
    mbobs = MultiBandObsList(meta=meta)

    for _ in range(5):
        obslist = ObsList()
        for _ in range(3):
            obslist.append(Observation(image=rng.normal(size=(13, 15))))
        mbobs.append(obslist)

    assert mbobs.meta == meta
    new_meta = {'blah': 6}
    mbobs.meta = new_meta
    assert mbobs.meta == new_meta
    mbobs.meta = None
    assert len(mbobs.meta) == 0
    with pytest.raises(TypeError):
        mbobs.meta = [10]
    with pytest.raises(TypeError):
        mbobs.set_meta([10])

    new_meta = {'bla': 6}
    new_meta.update(mbobs.meta)
    mbobs.update_meta_data({'bla': 6})
    assert mbobs.meta == new_meta
    with pytest.raises(TypeError):
        mbobs.update_meta_data([10])

    rng = np.random.RandomState(seed=12)
    new_obs = Observation(image=rng.normal(size=(13, 15)))
    rng = np.random.RandomState(seed=11)
    for obslist in mbobs:
        for obs in obslist:
            assert np.all(obs.image == rng.normal(size=(13, 15)))
    mbobs[1][2] = new_obs
    assert np.all(mbobs[1][2].image == new_obs.image)

    rng = np.random.RandomState(seed=13)
    obslist = ObsList()
    for _ in range(4):
        obslist.append(Observation(image=rng.normal(size=(13, 15))))
    mbobs[2] = obslist
    rng = np.random.RandomState(seed=13)
    for obs in mbobs[2]:
        assert np.all(obs.image == rng.normal(size=(13, 15)))


def test_multibandobslist_s2n():
    rng = np.random.RandomState(seed=11)
    mbobs = MultiBandObsList()

    numer = 0
    denom = 0
    for _ in range(5):
        obslist = ObsList()
        for _ in range(3):
            img = rng.normal(size=(13, 15))
            obslist.append(Observation(image=img))

            numer += np.sum(img)
            denom += np.sum(1.0 / obslist[-1].weight)
        mbobs.append(obslist)

    s2n = mbobs.get_s2n()
    assert s2n == numer / np.sqrt(denom)


def test_multibandobslist_s2n_zeroweight():
    rng = np.random.RandomState(seed=11)
    mbobs = MultiBandObsList()

    for _ in range(5):
        obslist = ObsList()
        for _ in range(3):
            img = rng.normal(size=(13, 15))
            obslist.append(
                Observation(
                    image=img,
                    weight=np.zeros((13, 15)),
                    store_pixels=False,
                )
            )

        mbobs.append(obslist)

    assert np.allclose(mbobs.get_s2n(), -9999)


def test_multibandobslist_append_err():
    mbobs = MultiBandObsList()
    with pytest.raises(AssertionError):
        mbobs.append(None)


def test_get_mbobs():
    rng = np.random.RandomState(seed=11)
    obs = Observation(image=rng.normal(size=(11, 13)))

    mbobs = get_mb_obs(obs)
    rng = np.random.RandomState(seed=11)
    assert np.all(mbobs[0][0].image == rng.normal(size=(11, 13)))
    assert len(mbobs) == 1
    assert len(mbobs[0]) == 1

    rng = np.random.RandomState(seed=12)
    obslist = ObsList()
    for _ in range(3):
        obslist.append(Observation(image=rng.normal(size=(11, 13))))
    mbobs = get_mb_obs(obslist)
    rng = np.random.RandomState(seed=12)
    for obs in mbobs[0]:
        assert np.all(obs.image == rng.normal(size=(11, 13)))
    assert len(mbobs) == 1
    assert len(mbobs[0]) == 3

    rng = np.random.RandomState(seed=13)
    mbobs = MultiBandObsList()
    for _ in range(5):
        obslist = ObsList()
        for _ in range(3):
            obslist.append(Observation(image=rng.normal(size=(13, 15))))
        mbobs.append(obslist)

    new_mbobs = get_mb_obs(mbobs)
    rng = np.random.RandomState(seed=13)
    for obslist in new_mbobs:
        for obs in obslist:
            assert np.all(obs.image == rng.normal(size=(13, 15)))

    with pytest.raises(ValueError):
        get_mb_obs(None)
