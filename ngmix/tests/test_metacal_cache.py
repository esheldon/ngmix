import time
import numpy as np

from flaky import flaky

import ngmix
import ngmix.metacal.metacal
from ._galsim_sims import _get_obs
from ..metacal.metacal import _cached_galsim_stuff


@flaky(max_runs=10)
def test_metacal_cache():
    _cached_galsim_stuff.cache_clear()

    # first warm up numba
    rng = np.random.RandomState(seed=100)
    obs = _get_obs(rng, noise=0.005, set_noise_image=True, psf_fwhm=0.8, n=300)
    t0 = time.time()
    ngmix.metacal.get_all_metacal(obs, rng=rng, types=["noshear"])
    t0 = time.time() - t0
    print("first time: %r seconds" % t0, flush=True)
    print(_cached_galsim_stuff.cache_info(), flush=True)

    # now cache it
    rng = np.random.RandomState(seed=10)
    obs = _get_obs(rng, noise=0.005, set_noise_image=True, n=300)
    t1 = time.time()
    ngmix.metacal.get_all_metacal(obs, rng=rng, types=["noshear"])
    t1 = time.time() - t1
    print("second time: %r seconds" % t1, flush=True)
    print(_cached_galsim_stuff.cache_info(), flush=True)

    # now use cache
    rng = np.random.RandomState(seed=10)
    obs = _get_obs(rng, noise=0.005, set_noise_image=True, n=300)
    t2 = time.time()
    ngmix.metacal.get_all_metacal(obs, rng=rng, types=["noshear"])
    t2 = time.time() - t2
    print("third time: %r seconds (< %r?)" % (t2, t1*0.7), flush=True)
    print(_cached_galsim_stuff.cache_info(), flush=True)

    # numba should be slower always but we do not care how much
    assert t1 < t0

    # we expect roughly 30% gains
    assert t2 < t1*0.7
