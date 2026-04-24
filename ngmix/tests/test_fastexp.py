import time
import numpy as np
from numba import njit
import pytest

from ngmix.fastexp_nb import fexp

# test values between -15 and 0
vals = [-7.8864744, -4.2333561, -11.02660361, -9.07802778,
        -12.01531878, -8.4256256, -8.70588303]


@pytest.mark.parametrize('x', vals)
def test_fastexp_smoke(x):
    assert np.allclose(np.exp(x), fexp(x), rtol=4.0e-5)


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
