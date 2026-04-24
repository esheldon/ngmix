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


def _do_exp(x):
    csum = 0.0
    for i in range(x.size):
        csum += np.exp(x[i])

    return csum


def test_fastexp_timing():
    x = np.array(vals)
    for _ in range(2):
        _do_fexp(x)

    t0 = time.time()
    for _ in range(1000):
        _do_exp(x)
    t0 = time.time() - t0

    t0f = time.time()
    for _ in range(1000):
        _do_fexp(x)
    t0f = time.time() - t0f

    # it should be faster
    assert t0f < t0, {'numpy': t0, 'fastexp': t0f}
