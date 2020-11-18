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
    pr = GPriorBA(A=1.0, sigma=0.5, rng=np.random.RandomState(seed=4535))
    assert pr.sigma == 0.5
    assert pr.A == 1.0

    # make sure histogram of 1d samples matches prob dist we expect
    g_samps = pr.sample1d(200000)
    h, be = np.histogram(g_samps, bins=np.linspace(0, 1, 100))
    h = h / np.sum(h)
    bc = (be[1:] + be[:-1])/2.0
    g_probs = pr.get_prob_array1d(bc)
    g_probs = g_probs / np.sum(g_probs)
    assert np.allclose(h, g_probs, atol=1e-3, rtol=0), np.max(np.abs(h-g_probs))
    for i in range(len(bc)):
        g_probs[i] = pr.get_prob_scalar1d(bc[i])
    g_probs = g_probs / np.sum(g_probs)
    assert np.allclose(h, g_probs, atol=1e-3, rtol=0), np.max(np.abs(h-g_probs))

    # do the same test but in 2d
