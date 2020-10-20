import numpy as np

from ..priors import make_rng


def test_make_rng_make():
    rng_out = make_rng()
    assert isinstance(rng_out, np.random.RandomState)


def test_make_rng_ident():
    rng = np.random.RandomState(seed=10)
    rng_out = make_rng(rng)
    assert rng_out is rng
