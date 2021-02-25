import numpy as np

import pytest

from ..priors import (
    FlatPrior,
    LOWVAL,
    TwoSidedErf,
    Normal,
    LMBounds,
    Bounded1D,
    LimitPDF,
    LogNormal,
    Sinh,
    TruncatedGaussian,
)
from ..gexceptions import GMixRangeError


@pytest.mark.parametrize('klass', [Bounded1D, LimitPDF])
def test_priors_bounded1d(klass):
    pr = klass(
        Normal(-0.5, 0.5, rng=np.random.RandomState(seed=10)),
        (-0.75, 1.0),
    )
    _s = pr.sample(1000)

    pr = klass(
        Normal(-0.5, 0.5, rng=np.random.RandomState(seed=10)),
        (-0.75, 1.0),
    )
    assert pr.bounds == (-0.75, 1.0)
    assert pr.has_bounds()
    s = pr.sample(1000)
    assert isinstance(s, np.ndarray)
    assert np.array_equal(_s, s)

    s = pr.sample()
    assert isinstance(s, float)

    with pytest.raises(ValueError):
        klass(
            Normal(-0.5, 0.5, rng=np.random.RandomState(seed=10)),
            (-0.75, 1.0, 1.0),
        )

    with pytest.raises(ValueError):
        klass(
            Normal(-0.5, 0.5, rng=np.random.RandomState(seed=10)),
            (1.0, -1.0),
        )

    with pytest.raises(ValueError):
        klass(
            Normal(-0.5, 0.5, rng=np.random.RandomState(seed=10)),
            1.0,
        )


def test_priors_lmbounds():
    pr = LMBounds(-0.5, 0.5, rng=np.random.RandomState(seed=10))
    _s = pr.sample()

    pr = LMBounds(-0.5, 0.5, rng=np.random.RandomState(seed=10))
    assert pr.mean == 0.0
    assert pr.sigma == 0.28
    assert pr.bounds == (-0.5, 0.5)
    assert pr.has_bounds()
    s = pr.sample()
    assert isinstance(s, float)
    assert s == _s

    assert pr.get_fdiff(10.0) == 0.0
    assert pr.get_fdiff(0.1) == 0.0

    s = pr.sample(nrand=10)
    assert isinstance(s, np.ndarray)


def test_priors_normal():
    pr = Normal(-0.5, 0.5, rng=np.random.RandomState(seed=10))
    _s = pr.sample()

    pr = Normal(-0.5, 0.5, rng=np.random.RandomState(seed=10))
    assert pr.mean == -0.5
    assert pr.sigma == 0.5
    s = pr.sample()
    assert isinstance(s, float)
    assert s == _s

    assert pr.get_prob_scalar(-0.5) == 1
    assert pr.get_lnprob_scalar(-0.5) == 0

    assert pr.get_prob_array(np.array([-0.5]))[0] == 1
    assert pr.get_lnprob_array(np.array([-0.5]))[0] == 0

    assert pr.get_fdiff(-0.5) == 0

    s = pr.sample(nrand=1)
    assert isinstance(s, np.ndarray)
    assert s.shape == (1,)

    s = pr.sample(nrand=10)
    assert isinstance(s, np.ndarray)
    assert s.shape == (10,)

    s = pr.sample(nrand=1000000)
    assert np.allclose(np.mean(s), -0.5, rtol=0, atol=1e-3)
    assert np.allclose(np.std(s), 0.5, rtol=0, atol=1e-3)


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


@pytest.mark.parametrize('shift', [None, 0, 0.1])
def test_priors_lognormal(shift):
    with pytest.raises(ValueError):
        LogNormal(-10, 1, shift=shift, rng=np.random.RandomState(seed=10))

    mean = 1.0
    sigma = 0.5

    pr = LogNormal(mean, sigma, shift=shift, rng=np.random.RandomState(seed=10))
    _s = pr.sample()

    pr = LogNormal(mean, sigma, shift=shift, rng=np.random.RandomState(seed=10))
    assert pr.mean == mean
    assert pr.sigma == sigma
    s = pr.sample()
    assert isinstance(s, float)
    assert s == _s

    s = pr.sample(nrand=1)
    assert isinstance(s, np.ndarray)
    assert s.shape == (1,)

    s = pr.sample(nrand=10)
    assert isinstance(s, np.ndarray)
    assert s.shape == (10,)

    s = pr.sample_brute()
    assert isinstance(s, float)

    s = pr.sample_brute(nrand=1)
    assert isinstance(s, np.ndarray)
    assert s.shape == (1,)

    s = pr.sample_brute(nrand=10)
    assert isinstance(s, np.ndarray)
    assert s.shape == (10,)

    s = pr.sample(nrand=1000000)

    if shift is None:
        shiftval = 0.0
    else:
        shiftval = shift

    assert np.allclose(s.mean(), mean + shiftval, rtol=0, atol=1e-3)
    assert np.allclose(s.std(), sigma, rtol=0, atol=1e-3)

    with pytest.raises(GMixRangeError):
        pr.get_prob_scalar(-10)

    mode_val = pr.get_prob_scalar(pr.mode + shiftval)
    low_val = pr.get_prob_scalar(pr.mode + shiftval - 1.0e-3)
    high_val = pr.get_prob_scalar(pr.mode + shiftval + 1.0e-3)

    assert mode_val > low_val
    assert mode_val > high_val

    mode_val_arr = pr.get_lnprob_array(np.array([pr.mode + shiftval]))
    assert isinstance(mode_val_arr, np.ndarray)
    assert np.allclose(mode_val_arr, 0)

    with pytest.raises(GMixRangeError):
        pr.get_lnprob_array(np.array([pr.mode + shiftval, -10]))

    mode_val_arr = pr.get_prob_array(np.array([pr.mode + shiftval]))
    assert isinstance(mode_val_arr, np.ndarray)
    assert np.allclose(mode_val_arr, 1)

    assert pr.get_fdiff(pr.mode + shiftval) == 0


def test_priors_lognormal_fit():
    mean = 1.0
    sigma = 0.5

    pr = LogNormal(mean, sigma, rng=np.random.RandomState(seed=10))
    samps = pr.sample(2000000)
    h, be = np.histogram(samps, bins=np.linspace(0, 10, 500))
    h = h / np.sum(h)
    bc = (be[1:] + be[:-1])/2.0
    res = pr.fit(bc, h)
    assert np.allclose(res['pars'][:2], [mean, sigma], rtol=0, atol=1e-3)


def test_priors_sinh():
    mean = 1.0
    scale = 0.5

    pr = Sinh(mean, scale, rng=np.random.RandomState(seed=10))
    _s = pr.sample()

    pr = Sinh(mean, scale, rng=np.random.RandomState(seed=10))
    assert pr.mean == mean
    assert pr.scale == scale
    s = pr.sample()
    assert isinstance(s, float)
    assert s == _s

    assert pr.get_fdiff(mean) == 0.0

    s = pr.sample(nrand=1)
    assert isinstance(s, np.ndarray)
    assert s.shape == (1,)

    s = pr.sample(nrand=10)
    assert isinstance(s, np.ndarray)
    assert s.shape == (10,)

    s = pr.sample(nrand=1000000)
    assert np.allclose(s.mean(), mean, rtol=0, atol=1e-3)
    assert s.max() < mean + scale
    assert s.min() > mean - scale


def test_priors_truncated_gaussian():
    import scipy.stats
    mean = 1.0
    sigma = 0.5
    minval = 0.3
    maxval = 1.1

    pr = TruncatedGaussian(
        mean, sigma, minval, maxval, rng=np.random.RandomState(seed=10),
    )
    _s = pr.sample()

    pr = TruncatedGaussian(
        mean, sigma, minval, maxval, rng=np.random.RandomState(seed=10),
    )
    assert pr.mean == mean
    assert pr.sigma == sigma
    assert pr.minval == minval
    assert pr.maxval == maxval

    s = pr.sample()
    assert isinstance(s, float)
    assert s == _s

    s = pr.sample(nrand=1)
    assert isinstance(s, np.ndarray)
    assert s.shape == (1,)

    s = pr.sample(nrand=10)
    assert isinstance(s, np.ndarray)
    assert s.shape == (10,)

    s = pr.sample(nrand=1000000)

    a = (minval - mean)/sigma
    b = (maxval - mean)/sigma
    spdf = scipy.stats.truncnorm(a=a, b=b, loc=mean, scale=sigma)
    assert np.allclose(s.mean(), spdf.mean(), rtol=0, atol=1e-3)
    assert np.allclose(s.std(), spdf.std(), rtol=0, atol=1e-3)

    # make sure these don't raise
    for val in [mean, minval, maxval]:
        pr.get_lnprob_scalar(val)

    # make sure these do raise
    with pytest.raises(GMixRangeError):
        pr.get_lnprob_scalar(minval - 0.1)
    with pytest.raises(GMixRangeError):
        pr.get_lnprob_scalar(maxval + 0.1)

    arr = pr.get_lnprob_array(np.array([minval, mean, maxval]))
    assert arr.shape == (3,)
    assert arr[0] < arr[1]
    assert arr[2] < arr[1]

    assert pr.get_fdiff(0.4*mean) == -0.6*mean/sigma
    with pytest.raises(GMixRangeError):
        pr.get_fdiff(minval - mean)
