import numpy as np

# import pytest

from ..priors import (
    Student2D,
    CenPrior,
    TruncatedSimpleGauss2D,
    ZDisk2D,
    ZAnnulus,
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


def test_priors_truncgauss2d():
    cen1 = 1.0
    cen2 = 1.3
    sigma1 = 0.5
    sigma2 = 0.7
    maxval = 0.5

    pr = TruncatedSimpleGauss2D(
        cen1, cen2, sigma1, sigma2, maxval, rng=np.random.RandomState(seed=10),
    )
    _s1, _s2 = pr.sample()
    pr = TruncatedSimpleGauss2D(
        cen1, cen2, sigma1, sigma2, maxval, rng=np.random.RandomState(seed=10),
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

    r2 = (s1 - cen1)**2 + (s2 - cen2)**2
    assert np.all(r2 < maxval**2)


def test_priors_zdisk2d():
    radius = 0.5

    pr = ZDisk2D(radius, rng=np.random.RandomState(seed=10))
    _s = pr.sample1d()
    _s1, _s2 = pr.sample2d()

    pr = ZDisk2D(radius, rng=np.random.RandomState(seed=10))

    assert pr.radius == radius

    s = pr.sample1d()
    s1, s2 = pr.sample2d()
    assert isinstance(s, float)
    assert s == _s
    assert s1 == _s1
    assert s2 == _s2

    s = pr.sample1d(nrand=1)
    s1, s2 = pr.sample2d(nrand=1)
    assert (
        isinstance(s, np.ndarray) and isinstance(s1, np.ndarray) and
        isinstance(s2, np.ndarray)
    )
    assert s1.shape == (1,) and s2.shape == (1,) and s.shape == (1,)

    s = pr.sample1d(nrand=10)
    s1, s2 = pr.sample2d(nrand=10)
    assert s1.shape == (10,) and s2.shape == (10,) and s.shape == (10,)

    s = pr.sample1d(nrand=1000000)
    s1, s2 = pr.sample2d(nrand=1000000)
    assert np.allclose(s1.mean(), 0, rtol=0, atol=2e-3)
    assert np.allclose(s2.mean(), 0, rtol=0, atol=2e-3)

    expected_meanr = 2.0 / 3.0 * radius
    r = np.sqrt(s1**2 + s2**2)
    assert np.all(s < radius)
    assert np.all(r < radius)
    assert np.allclose(s.mean(), expected_meanr, rtol=0, atol=2e-3)
    assert np.allclose(r.mean(), expected_meanr, rtol=0, atol=2e-3)


def test_priors_zannulus():
    rmin = 0.5
    rmax = 1.0

    pr = ZAnnulus(rmin, rmax, rng=np.random.RandomState(seed=10))
    _s = pr.sample1d()
    _s1, _s2 = pr.sample2d()

    pr = ZAnnulus(rmin, rmax, rng=np.random.RandomState(seed=10))

    assert pr.radius == rmax
    assert pr.rmin == rmin

    s = pr.sample1d()
    s1, s2 = pr.sample2d()
    assert isinstance(s, float)
    assert s == _s
    assert s1 == _s1
    assert s2 == _s2

    s = pr.sample1d(nrand=1)
    s1, s2 = pr.sample2d(nrand=1)
    assert (
        isinstance(s, np.ndarray) and isinstance(s1, np.ndarray) and
        isinstance(s2, np.ndarray)
    )
    assert s1.shape == (1,) and s2.shape == (1,) and s.shape == (1,)

    s = pr.sample1d(nrand=10)
    s1, s2 = pr.sample2d(nrand=10)
    assert s1.shape == (10,) and s2.shape == (10,) and s.shape == (10,)

    s = pr.sample1d(nrand=1000000)
    s1, s2 = pr.sample2d(nrand=1000000)
    assert np.allclose(s1.mean(), 0, rtol=0, atol=2e-3)
    assert np.allclose(s2.mean(), 0, rtol=0, atol=2e-3)

    expected_meanr = 2.0 / 3.0 * (
        (rmax**3 - rmin**3)/(rmax**2 - rmin**2)
    )
    r = np.sqrt(s1**2 + s2**2)
    assert np.all((s < rmax) & (s > rmin))
    assert np.all((r < rmax) & (r > rmin))
    assert np.allclose(s.mean(), expected_meanr, rtol=0, atol=2e-3)
    assert np.allclose(r.mean(), expected_meanr, rtol=0, atol=2e-3)
