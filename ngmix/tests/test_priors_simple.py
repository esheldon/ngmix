import numpy as np

import pytest

from ..priors import FlatPrior, LOWVAL, TwoSidedErf
from ..gexceptions import GMixRangeError


def test_priors_flatprior():
    pr = FlatPrior(-0.5, 0.5, rng=np.random.RandomState(seed=10))
    _s = pr.sample()

    pr = FlatPrior(-0.5, 0.5, rng=np.random.RandomState(seed=10))
    assert pr.minval == -0.5
    assert pr.maxval == 0.5
    s = pr.sample()
    assert isinstance(s, float)
    assert s == _s

    with pytest.raises(GMixRangeError):
        pr.get_prob_scalar(-1)
    with pytest.raises(GMixRangeError):
        pr.get_prob_scalar(1)
    assert pr.get_prob_scalar(0) == 1
    assert pr.get_prob_scalar(-0.5) == 1
    assert pr.get_prob_scalar(0.5) == 1

    with pytest.raises(GMixRangeError):
        pr.get_lnprob_scalar(-1)
    with pytest.raises(GMixRangeError):
        pr.get_lnprob_scalar(1)
    assert pr.get_lnprob_scalar(0) == 0
    assert pr.get_lnprob_scalar(-0.5) == 0
    assert pr.get_lnprob_scalar(0.5) == 0

    with pytest.raises(GMixRangeError):
        pr.get_fdiff(-1)
    with pytest.raises(GMixRangeError):
        pr.get_fdiff(1)
    assert pr.get_fdiff(0) == 0
    assert pr.get_fdiff(-0.5) == 0
    assert pr.get_fdiff(0.5) == 0

    s = pr.sample(nrand=1)
    assert isinstance(s, np.ndarray)
    assert s.shape == (1,)

    s = pr.sample(nrand=10)
    assert isinstance(s, np.ndarray)
    assert s.shape == (10,)

    s = pr.sample(nrand=1000000)
    assert np.allclose(np.mean(s), 0.0, rtol=0, atol=1e-3)


def test_priors_twosidederf():
    pr = TwoSidedErf(-0.5, 0.05, 0.5, 0.05, rng=np.random.RandomState(seed=10))
    _s = pr.sample()

    pr = TwoSidedErf(-0.5, 0.05, 0.5, 0.05, rng=np.random.RandomState(seed=10))
    assert pr.minval == -0.5
    assert pr.maxval == 0.5
    s = pr.sample()
    assert isinstance(s, float)
    assert s == _s

    assert pr.get_prob_scalar(-1) == 0
    assert pr.get_prob_scalar(1) == 0
    assert pr.get_prob_scalar(0) == 1
    assert pr.get_prob_scalar(-0.5) == 0.5
    assert pr.get_prob_scalar(0.5) == 0.5

    assert pr.get_lnprob_scalar(-1) == LOWVAL
    assert pr.get_lnprob_scalar(1) == LOWVAL
    assert pr.get_lnprob_scalar(0) == 0
    assert pr.get_lnprob_scalar(-0.5) == np.log(0.5)
    assert pr.get_lnprob_scalar(0.5) == np.log(0.5)

    assert pr.get_prob_array(np.array([-1]))[0] == 0
    assert pr.get_prob_array(np.array([1]))[0] == 0
    assert pr.get_prob_array(np.array([0]))[0] == 1
    assert pr.get_prob_array(np.array([-0.5]))[0] == 0.5
    assert pr.get_prob_array(np.array([0.5]))[0] == 0.5

    assert pr.get_lnprob_array(np.array([-1]))[0] == LOWVAL
    assert pr.get_lnprob_array(np.array([1]))[0] == LOWVAL
    assert pr.get_lnprob_array(np.array([0]))[0] == 0
    assert pr.get_lnprob_array(np.array([-0.5]))[0] == np.log(0.5)
    assert pr.get_lnprob_array(np.array([0.5]))[0] == np.log(0.5)

    assert pr.get_lnprob_scalar(-1) == LOWVAL
    assert pr.get_lnprob_scalar(1) == LOWVAL
    assert pr.get_lnprob_scalar(0) == 0
    assert pr.get_lnprob_scalar(-0.5) == np.log(0.5)
    assert pr.get_lnprob_scalar(0.5) == np.log(0.5)

    assert pr.get_fdiff(-1) == np.sqrt(-2.0 * LOWVAL)
    assert pr.get_fdiff(1) == np.sqrt(-2.0 * LOWVAL)
    assert pr.get_fdiff(0) == 0
    assert pr.get_fdiff(-0.5) == np.sqrt(-2.0 * np.log(0.5))
    assert pr.get_fdiff(0.5) == np.sqrt(-2.0 * np.log(0.5))

    assert pr.get_fdiff(np.array([-1]))[0] == np.sqrt(-2.0 * LOWVAL)
    assert pr.get_fdiff(np.array([1]))[0] == np.sqrt(-2.0 * LOWVAL)
    assert pr.get_fdiff(np.array([0]))[0] == 0
    assert pr.get_fdiff(np.array([-0.5]))[0] == np.sqrt(-2.0 * np.log(0.5))
    assert pr.get_fdiff(np.array([0.5]))[0] == np.sqrt(-2.0 * np.log(0.5))

    s = pr.sample(nrand=1)
    assert isinstance(s, np.ndarray)
    assert s.shape == (1,)

    s = pr.sample(nrand=10)
    assert isinstance(s, np.ndarray)
    assert s.shape == (10,)

    s = pr.sample(nrand=1000000)
    assert np.allclose(np.mean(s), 0.0, rtol=0, atol=1e-3)
