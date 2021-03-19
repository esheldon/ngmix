import numpy as np

import pytest

from ..priors import PriorBase, GPriorBase


def test_priors_priorbase():
    rng = np.random.RandomState(seed=10)

    pr = PriorBase(rng=rng, bounds=None)
    assert not pr.has_bounds()
    pr = PriorBase(rng=rng, bounds=(3, 4))
    assert pr.has_bounds()
    pr = PriorBase(rng=rng, bounds=[3, 4])
    assert pr.has_bounds()

    pr = PriorBase(bounds=None, rng=rng)
    assert pr.rng is rng


def test_priors_gpriorbase_raises():
    rng = np.random.RandomState(seed=312)
    pr = GPriorBase(pars=None, rng=rng)
    for method, num in [
        ("fill_prob_array1d", 2),
        ("fill_lnprob_array2d", 3),
        ("fill_prob_array2d", 3),
        ("get_lnprob_scalar2d", 2),
        ("get_prob_scalar2d", 2),
        ("get_prob_scalar1d", 1),
    ]:
        mth = getattr(pr, method)
        with pytest.raises(RuntimeError):
            args = tuple([None] * num)
            mth(*args)
