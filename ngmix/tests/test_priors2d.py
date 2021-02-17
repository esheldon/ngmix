import numpy as np

import pytest

from ..priors import (
    Student2D,
    CenPrior,
    TruncatedSimpleGauss2D,
    LOWVAL,
)
from ..gexceptions import GMixRangeError


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

    arr = pr.get_lnprob_array(np.array([1, 2, 3]), np.array([3, 2, 1]))
    assert arr.shape == (3,)

    arr = pr.get_lnprob_array(np.array([1, 2, 3]), np.array([3, 2, 1]))
    assert arr.shape == (3,)

    arr = pr.get_lnprob_scalar(1, 2)
    assert isinstance(arr, float)


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

    lm1, lm2 = pr.get_fdiff(cen1, cen2)
    assert np.allclose(lm1, 0)
    assert np.allclose(lm2, 0)
    lm1, lm2 = pr.get_fdiff(0, 0)
    assert np.allclose(lm1, -cen1/sigma1)
    assert np.allclose(lm2, -cen2/sigma2)

    lps = pr.get_lnprob_scalar(cen1, cen2)
    assert np.allclose(lps, 0)
    lps = pr.get_lnprob_scalar(0, 0)
    assert np.allclose(lps, -0.5*cen1**2/sigma1**2 - 0.5*cen2**2/sigma2**2)

    lps = pr.get_lnprob_scalar_sep(cen1, cen2)
    assert np.allclose(lps, 0)
    lps = pr.get_lnprob_scalar_sep(0, 0)
    assert np.allclose(lps, [-0.5*cen1**2/sigma1**2, -0.5*cen2**2/sigma2**2])

    assert np.allclose(
        np.exp(pr.get_lnprob_scalar(cen1, 0)),
        pr.get_prob_scalar(cen1, 0),
    )


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

    lps = pr.get_lnprob_nothrow(cen1, cen2)
    assert np.allclose(lps, 0)
    lps = pr.get_lnprob_nothrow(0, 0)
    assert np.allclose(lps, LOWVAL)

    lps = pr.get_lnprob_scalar(cen1, cen2)
    assert np.allclose(lps, 0)
    lps = pr.get_lnprob_scalar(0.9*cen1, 0.9*cen2)
    assert np.allclose(lps, -0.5*(0.1*cen1)**2/sigma1**2 - 0.5*(0.1*cen2)**2/sigma2**2)

    with pytest.raises(GMixRangeError):
        pr.get_lnprob_scalar(0, 0)

    lps = pr.get_lnprob_array(
        np.array([0, 0.9*cen1, cen1]), np.array([0.9*cen2, 0, cen2])
    )
    assert lps.shape == (3,)
    assert not np.isfinite(lps[0])
    assert not np.isfinite(lps[1])

    assert np.allclose(
        np.exp(pr.get_lnprob_scalar(0.9*cen1, 0.8*cen2)),
        pr.get_prob_scalar(0.9*cen1, 0.8*cen2),
    )
