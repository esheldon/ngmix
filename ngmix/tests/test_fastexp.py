import time
import numpy as np
import pytest

from ngmix.fastexp import fexp

# test values between -15 and 0
vals = [-7.8864744, -4.2333561, -11.02660361, -9.07802778,
        -12.01531878, -8.4256256, -8.70588303]


@pytest.mark.parametrize('x', vals)
def test_fastexp_smoke(x):
    assert np.allclose(np.exp(x), fexp(x), rtol=4.0e-5)


@pytest.mark.parametrize('x', vals)
def test_fastexp_timing(x):
    # call a few tims for numba overhead
    for _ in range(2):
        fexp(x)

    t0 = time.time()
    for _ in range(1000):
        np.exp(x)
    t0 = time.time() - t0

    t0f = time.time()
    for _ in range(1000):
        fexp(x)
    t0f = time.time() - t0f

    # it should be faster
    assert t0f < t0, {'numpy': t0, 'fastexp': t0f}
