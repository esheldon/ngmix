import numpy as np

import pytest

from ..priors import GPriorGauss, GPriorBA


def test_priors_gpriorgauss():
    pr = GPriorGauss(0.1, rng=np.random.RandomState(seed=10))

    g1, g2 = pr.sample2d()
    assert isinstance(g1, float)
    assert isinstance(g2, float)

    g1, g2 = pr.sample2d(nrand=1)
    assert isinstance(g1, np.ndarray)
    assert g1.shape == (1,)
    assert isinstance(g2, np.ndarray)
    assert g2.shape == (1,)

    g1, g2 = pr.sample2d(nrand=100)
    assert isinstance(g1, np.ndarray)
    assert g1.shape == (100,)
    assert isinstance(g2, np.ndarray)
    assert g2.shape == (100,)

    g1, g2 = pr.sample2d(nrand=100000)
    assert np.allclose(np.mean(g1), 0.0, atol=1e-3)
    assert np.allclose(np.std(g1), 0.1, atol=1e-3)
    assert np.allclose(np.mean(g1), 0.0, atol=1e-3)
    assert np.allclose(np.std(g2), 0.1, atol=1e-3)
    assert np.all(np.sqrt(g1**2 + g2**2) <= 1.0)
    assert np.all(np.abs(g1) <= 1.0)
    assert np.all(np.abs(g2) <= 1.0)

    with pytest.raises(NotImplementedError) as e:
        pr.sample1d()
        assert "no 1d for gauss" in e.value


def test_priors_gpriorba():
    pr = GPriorBA(A=1.0, sigma=0.5)
    assert pr.sigma == 0.5
    assert pr.A == 1.0
