import numpy as np

# import pytest

from ..priors import (
    Student2D,
    CenPrior,
)
# from ..gexceptions import GMixRangeError


def test_priors_student2d():
    mean1 = 1.0
    mean2 = 1.3
    sigma1 = 0.5
    sigma2 = 0.7

    pr = Student2D(
        mean1, mean2, sigma1, sigma2, rng=np.random.RandomState(seed=10),
    )
    _s1, _s2 = pr.sample()

    pr = Student2D(
        mean1, mean2, sigma1, sigma2, rng=np.random.RandomState(seed=10),
    )
    assert pr.mean1 == mean1
    assert pr.mean2 == mean2
    assert pr.sigma1 == sigma1
    assert pr.sigma2 == sigma2

    s1, s2 = pr.sample()
    assert isinstance(s1, float)
    assert s1 == _s1

    s1, s2 = pr.sample(nrand=1)
    assert isinstance(s1, np.ndarray) and isinstance(s2, np.ndarray)
    assert s1.shape == (1,) and s2.shape == (1,)

    s1, s2 = pr.sample(nrand=10)
    assert isinstance(s1, np.ndarray) and isinstance(s2, np.ndarray)
    assert s1.shape == (10,) and s2.shape == (10,)


def test_priors_cenprior():
    cen1 = 1.0
    cen2 = 1.3
    sigma1 = 0.5
    sigma2 = 0.7

    pr = CenPrior(
        cen1, cen2, sigma1, sigma2, rng=np.random.RandomState(seed=10),
    )
    _s1, _s2 = pr.sample()

    pr = CenPrior(
        cen1, cen2, sigma1, sigma2, rng=np.random.RandomState(seed=10),
    )
    assert pr.cen1 == cen1
    assert pr.cen2 == cen2
    assert pr.sigma1 == sigma1
    assert pr.sigma2 == sigma2

    s1, s2 = pr.sample()
    assert isinstance(s1, float)
    assert s1 == _s1

    s1, s2 = pr.sample(nrand=1)
    assert isinstance(s1, np.ndarray) and isinstance(s2, np.ndarray)
    assert s1.shape == (1,) and s2.shape == (1,)

    s1, s2 = pr.sample(nrand=10)
    assert isinstance(s1, np.ndarray) and isinstance(s2, np.ndarray)
    assert s1.shape == (10,) and s2.shape == (10,)

    s1, s2 = pr.sample(nrand=1000000)
    assert np.allclose(s1.mean(), cen1, rtol=0, atol=2e-3)
    assert np.allclose(s2.mean(), cen2, rtol=0, atol=2e-3)
    assert np.allclose(s1.std(), sigma1, rtol=0, atol=2e-3)
    assert np.allclose(s2.std(), sigma2, rtol=0, atol=2e-3)
