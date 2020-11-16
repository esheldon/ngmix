import numpy as np

from ..priors import make_rng, srandu


def test_make_rng_make():
    rng_out = make_rng()
    assert isinstance(rng_out, np.random.RandomState)


def test_make_rng_ident():
    rng = np.random.RandomState(seed=10)
    rng_out = make_rng(rng)
    assert rng_out is rng


def test_srandu():
    assert np.all((srandu(num=10000) > -1.0) & (srandu(num=10000) < 1.0))
    rng = np.random.RandomState(seed=10)
    nums = srandu(num=10000, rng=rng)
    rng = np.random.RandomState(seed=10)
    nums_again = srandu(num=10000, rng=rng)
    assert np.array_equal(nums, nums_again)
    rng = np.random.RandomState(seed=10)
    assert np.allclose(np.mean(srandu(num=1000000, rng=rng)), 0.0, atol=1e-1, rtol=0)
    assert np.allclose(
        np.std(srandu(num=1000000, rng=rng)), np.sqrt(4/12), atol=1e-1, rtol=0
    )
