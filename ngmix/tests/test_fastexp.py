import time
import numpy as np
from numba import njit
import pytest

from ngmix.fastexp_nb import (
    fexp, _make_smooth_exp_coeffs, _EXP5_SMOOTH_COEFFS,
)

# test values between -15 and 0
vals = [-7.8864744, -4.2333561, -11.02660361, -9.07802778,
        -12.01531878, -8.4256256, -8.70588303]


@pytest.mark.parametrize('x', vals)
def test_fastexp_smoke(x):
    assert np.allclose(np.exp(x), fexp(x), rtol=4.0e-5)


def test_fastexp_accuracy():
    x = np.linspace(-15.0, 0.0, 100_000)
    relerr = np.array([fexp(xx) for xx in x]) / np.exp(x) - 1
    assert np.abs(relerr).max() < 2.5e-6

    # the error oscillates in sign, so models rendered with fexp are
    # not scaled by the mean error
    assert np.abs(relerr.mean()) < 1.0e-7


def test_fastexp_matches_coeffs():
    """
    the literals compiled into exp5_smooth match _EXP5_SMOOTH_COEFFS,
    which in turn match the generator
    """
    gen = _make_smooth_exp_coeffs()
    assert np.abs(gen / _EXP5_SMOOTH_COEFFS - 1).max() < 1.0e-10

    rng = np.random.RandomState(31415)
    for x in rng.uniform(low=-15, high=0, size=100):
        ival = int(x - 0.5)
        f = x - ival
        expected = np.exp(ival) * np.polyval(
            _EXP5_SMOOTH_COEFFS[::-1], f,
        )
        assert abs(fexp(x) / expected - 1) < 1.0e-14


def test_fastexp_c2_conditions():
    """
    the polynomial and its first two derivatives satisfy
    P(1/2) = e P(-1/2) etc., which makes the piecewise function C2 at
    the half integer break points
    """
    coeffs = _EXP5_SMOOTH_COEFFS.copy()
    for _ in range(3):
        left = np.polyval(coeffs[::-1], 0.5)
        right = np.exp(1.0) * np.polyval(coeffs[::-1], -0.5)
        assert abs(left / right - 1) < 1.0e-13
        coeffs = coeffs[1:] * np.arange(1, coeffs.size)


def test_fastexp_smooth_at_boundaries():
    """
    no steps in the value or first derivative at the half integer
    break points of the lookup table
    """
    eps = 1.0e-6
    for bnd in np.arange(-14.5, 0.0, 1.0):
        scale = np.exp(bnd)

        vleft = fexp(bnd - eps)
        vmid = fexp(bnd)
        vright = fexp(bnd + eps)

        # the change across the break point is the smooth change
        # ~ 2 * eps * exp(bnd), not a step
        assert abs(vright - vleft) < 5.0 * eps * scale

        # the one sided derivatives agree
        dleft = (vmid - vleft) / eps
        dright = (vright - vmid) / eps
        assert abs(dright - dleft) < 1.0e-4 * scale


@njit
def _do_fexp(x):
    csum = 0.0
    for i in range(x.size):
        csum += fexp(x[i])

    return csum


def test_fastexp_timing():
    x = np.linspace(-12, -7, 50)

    for _ in range(2):
        _do_fexp(x)

    t0 = time.time()
    for _ in range(1000):
        slow_sum = np.exp(x).sum()
    t0 = time.time() - t0

    t0f = time.time()
    for _ in range(1000):
        fast_sum = _do_fexp(x)
    t0f = time.time() - t0f

    assert t0f < t0, {'numpy': t0, 'fastexp': t0f}
    assert np.allclose(slow_sum, fast_sum, rtol=4.0e-5)
