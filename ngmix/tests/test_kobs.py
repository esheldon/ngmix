import numpy as np
import pytest
import galsim

from ngmix.observation import KObservation, KObsList, KMultiBandObsList, get_kmb_obs


def test_get_kmb_obs():
    rng = np.random.RandomState(seed=11)
    obs = KObservation(kimage=galsim.ImageCD(rng.normal(size=(11, 13)), scale=0.3))
    mbobs = get_kmb_obs(obs)
    rng = np.random.RandomState(seed=11)
    assert np.all(mbobs[0][0].kimage.array == rng.normal(size=(11, 13)))
    assert len(mbobs) == 1
    assert len(mbobs[0]) == 1

    rng = np.random.RandomState(seed=12)
    obslist = KObsList()
    for _ in range(3):
        obslist.append(KObservation(
            kimage=galsim.ImageCD(rng.normal(size=(12, 17)), scale=0.3)))
    mbobs = get_kmb_obs(obslist)
    rng = np.random.RandomState(seed=12)
    for i in range(3):
        assert np.all(mbobs[0][i].kimage.array == rng.normal(size=(12, 17)))
    assert len(mbobs) == 1
    assert len(mbobs[0]) == 3

    rng = np.random.RandomState(seed=13)
    mbobs = KMultiBandObsList()
    for _ in range(5):
        obslist = KObsList()
        for _ in range(3):
            obslist.append(KObservation(
                kimage=galsim.ImageCD(rng.normal(size=(13, 15)), scale=0.3)))
        mbobs.append(obslist)

    new_mbobs = get_kmb_obs(mbobs)
    rng = np.random.RandomState(seed=13)
    for obslist in new_mbobs:
        for obs in obslist:
            assert np.all(obs.kimage.array == rng.normal(size=(13, 15)))

    with pytest.raises(ValueError):
        get_kmb_obs(None)
