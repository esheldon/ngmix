import numpy as np

import pytest

from ..priors import (
    GPriorGauss, GPriorBA, ZDisk2D
)
from ..gexceptions import GMixRangeError


def test_priors_gpriorgauss():
    pr = GPriorGauss(0.1, rng=np.random.RandomState(seed=10))
    _g1, _g2 = pr.sample2d()

    pr = GPriorGauss(0.1, rng=np.random.RandomState(seed=10))
    g1, g2 = pr.sample2d()
    assert isinstance(g1, float)
    assert isinstance(g2, float)
    assert g1 == _g1
    assert g2 == _g2

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


@pytest.mark.parametrize('sigma', [0.05, 0.1, 0.2, 0.3, 0.6])
def test_priors_gpriorba(sigma):
    """
    try a few to make sure the peak finder works
    """
    pr = GPriorBA(sigma=0.5, rng=np.random.RandomState(seed=4535))
    _g1, _g2 = pr.sample2d()

    pr = GPriorBA(sigma=0.5, rng=np.random.RandomState(seed=4535))
    assert pr.sigma == 0.5
    assert pr.A == 1.0

    g1, g2 = pr.sample2d()
    assert isinstance(g1, float)
    assert isinstance(g2, float)
    assert g1 == _g1
    assert g2 == _g2

    # make sure histogram of samples matches prob dist we expect
    for i in range(3):
        if i == 0:
            g1, g2 = None, None
            g_samps = pr.sample1d(200000)
        elif i == 1:
            g1, g2 = pr.sample2d_brute(200000)
            g_samps = np.sqrt(g1**2 + g2**2)
        else:
            g1, g2 = pr.sample2d(200000)
            g_samps = np.sqrt(g1**2 + g2**2)

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

        # test 2d via using 1d samples and factor of g
        g_probs = pr.get_prob_array2d(bc, np.zeros_like(bc)) * bc
        g_probs = g_probs / np.sum(g_probs)
        assert np.allclose(h, g_probs, atol=1e-3, rtol=0), np.max(np.abs(h-g_probs))

        for i in range(len(bc)):
            g_probs[i] = pr.get_prob_scalar2d(bc[i], 0.0) * bc[i]
        g_probs = g_probs / np.sum(g_probs)
        assert np.allclose(h, g_probs, atol=1e-3, rtol=0), np.max(np.abs(h-g_probs))

        g_probs = np.exp(pr.get_lnprob_array2d(bc, np.zeros_like(bc))) * bc
        g_probs = g_probs / np.sum(g_probs)
        assert np.allclose(h, g_probs, atol=1e-3, rtol=0), np.max(np.abs(h-g_probs))

        for i in range(len(bc)):
            g_probs[i] = np.exp(pr.get_lnprob_scalar2d(bc[i], 0.0)) * bc[i]
        g_probs = g_probs / np.sum(g_probs)
        assert np.allclose(h, g_probs, atol=1e-3, rtol=0), np.max(np.abs(h-g_probs))

        # test 1d in each comp
        for gc in [g1, g2]:
            if gc is not None:
                h, be = np.histogram(gc, bins=np.linspace(0, 1, 100))
                h = h / np.sum(h)
                bc = (be[1:] + be[:-1])/2.0

                g_probs = pr.get_prob_array2d(bc, np.zeros_like(bc))
                g_probs = g_probs / np.sum(g_probs)
                assert np.allclose(h, g_probs, atol=3e-3, rtol=0), (
                    np.max(np.abs(h-g_probs))
                )

                for i in range(len(bc)):
                    g_probs[i] = pr.get_prob_scalar2d(bc[i], 0.0)
                g_probs = g_probs / np.sum(g_probs)
                assert np.allclose(h, g_probs, atol=3e-3, rtol=0), (
                    np.max(np.abs(h-g_probs))
                )

                g_probs = np.exp(pr.get_lnprob_array2d(bc, np.zeros_like(bc)))
                g_probs = g_probs / np.sum(g_probs)
                assert np.allclose(h, g_probs, atol=3e-3, rtol=0), (
                    np.max(np.abs(h-g_probs))
                )

                for i in range(len(bc)):
                    g_probs[i] = np.exp(pr.get_lnprob_scalar2d(bc[i], 0.0))
                g_probs = g_probs / np.sum(g_probs)
                assert np.allclose(h, g_probs, atol=3e-3, rtol=0), (
                    np.max(np.abs(h-g_probs))
                )

    # make sure recomputing the max value kind of works
    mvloc, mv = pr.maxval1d_loc, pr.maxval1d
    pr.set_maxval1d()
    assert np.allclose(mvloc, pr.maxval1d_loc, atol=1e-2, rtol=0)
    assert np.allclose(mv, pr.maxval1d, atol=1e-2, rtol=0)

    # finally try and fit samples of g1, g2 to get back the prior we put in
    g_samps = pr.sample1d(200000)
    h, be = np.histogram(g_samps, bins=np.linspace(0, 1, 100))
    h = h / np.sum(h)
    bc = (be[1:] + be[:-1])/2.0
    pr.dofit(bc, h)
    assert np.allclose(pr.fit_pars[1], 0.5, rtol=0, atol=3e-3)

    # not sure what to test here - the real test is if the priors change
    # things in a LM fit for an object which is poorly constrained
    # for now we will make sure the outputs are the right shape etc
    g1 = np.ones(10) * 0.1
    g2 = np.ones(10) * 0.1
    fdiff = pr.get_fdiff(g1, g2)
    assert fdiff.shape == (10,)
    fdiff = pr.get_fdiff(0.1, 0.1)
    assert isinstance(fdiff, float)

    # som additional exceptions this class raises
    with pytest.raises(GMixRangeError):
        pr.get_lnprob_scalar2d(0.5, 1)


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

    p = pr.get_lnprob_scalar1d(0.4)
    assert np.allclose(p, 0)
    with pytest.raises(GMixRangeError):
        pr.get_lnprob_scalar1d(1.4)

    p = pr.get_prob_scalar1d(0.4)
    assert np.allclose(p, 1)
    p = pr.get_prob_scalar1d(1.4)
    assert np.allclose(p, 0)

    p = pr.get_lnprob_scalar2d(0.4, 0)
    assert np.allclose(p, 0)
    with pytest.raises(GMixRangeError):
        pr.get_lnprob_scalar2d(0.4, 0.4)

    p = pr.get_prob_scalar2d(0.4, 0)
    assert np.allclose(p, 1)
    p = pr.get_prob_scalar2d(0.4, 0.4)
    assert np.allclose(p, 0)

    p = pr.get_prob_array2d(0.4, 0)
    assert np.allclose(p, 1)
    p = pr.get_prob_array2d(0.4, 0.4)
    assert np.allclose(p, 0)

    p = pr.get_prob_array2d(np.array([0.4, 1.4]), np.array([0, 1]))
    assert np.allclose(p, [1, 0])
