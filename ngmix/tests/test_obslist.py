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

    new_meta = {'bla': 6}
    new_meta.update(obslist.meta)
    obslist.update_meta_data({'bla': 6})
    assert obslist.meta == new_meta

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


def test_obslist_append_err():
    obslist = ObsList()
    with pytest.raises(AssertionError):
        obslist.append(None)
#
#
# class MultiBandObsList(list):
#     """
#     Hold a list of lists of ObsList objects, each representing a filter
#     band
#
#     This class provides a bit of type safety and ease of type checking
#     """
#
#     def __init__(self, meta=None):
#         super(MultiBandObsList,self).__init__()
#
#         self.set_meta(meta)
#
#     def append(self, obs_list):
#         """
#         Add a new ObsList
#
#         over-riding this for type safety
#         """
#         assert isinstance(obs_list,ObsList),\
#             'obs_list should be of type ObsList'
#         super(MultiBandObsList,self).append(obs_list)
#
#     @property
#     def meta(self):
#         """
#         getter for meta
#
#         currently this simply returns a reference
#         """
#         return self._meta
#
#     @meta.setter
#     def meta(self, meta):
#         """
#         set the meta
#
#         this does consistency checks and can trigger an update
#         of the pixels array
#         """
#         self.set_meta(meta)
#
#     def set_meta(self, meta):
#         """
#         Add some metadata
#         """
#
#         if meta is None:
#             meta={}
#
#         if not isinstance(meta,dict):
#             raise TypeError("meta data must be in "
#                             "dictionary form, got %s" % type(meta))
#
#         self._meta = meta
#
#     def update_meta_data(self, meta):
#         """
#         Add some metadata
#         """
#
#         if not isinstance(meta,dict):
#             raise TypeError("meta data must be in dictionary form")
#         self._meta.update(meta)
#
#     def get_s2n(self):
#         """
#         get the the simple s/n estimator
#
#         sum(I)/sqrt( sum( 1/w ) ) = Isum/sqrt(Vsum)
#
#         returns
#         -------
#         s2n: float
#             The supid s/n estimator
#         """
#
#         Isum, Vsum, Npix = self.get_s2n_sums()
#         if Vsum > 0.0:
#             s2n = Isum/numpy.sqrt(Vsum)
#         else:
#             s2n=-9999.0
#         return s2n
#
#     def get_s2n_sums(self):
#         """
#         get the sums for the simple s/n estimator
#
#         sum(I)/sqrt( sum( 1/w ) ) = Isum/sqrt(Vsum)
#
#         returns
#         -------
#         Isum, Vsum, Npix
#         """
#
#         Isum = 0.0
#         Vsum = 0.0
#         Npix = 0
#
#         for obslist in self:
#             tIsum,tVsum,tNpix = obslist.get_s2n_sums()
#             Isum += tIsum
#             Vsum += tVsum
#             Npix += tNpix
#
#         return Isum, Vsum, Npix
#
#     def __setitem__(self, index, obs_list):
#         """
#         over-riding this for type safety
#         """
#         assert isinstance(obs_list,ObsList),\
#             'obs_list should be of type ObsList'
#         super(MultiBandObsList,self).__setitem__(index, obs_list)
#
# def get_mb_obs(obs_in):
#     """
#     convert the input to a MultiBandObsList
#
#     Input should be an Observation, ObsList, or MultiBandObsList
#     """
#
#     if isinstance(obs_in,Observation):
#         obs_list=ObsList()
#         obs_list.append(obs_in)
#
#         obs=MultiBandObsList()
#         obs.append(obs_list)
#     elif isinstance(obs_in,ObsList):
#         obs=MultiBandObsList()
#         obs.append(obs_in)
#     elif isinstance(obs_in,MultiBandObsList):
#         obs=obs_in
#     else:
#         raise ValueError(
#             'obs should be Observation, ObsList, or MultiBandObsList'
#         )
#
#     return obs
