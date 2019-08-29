import time
import numpy as np
import pytest

from ngmix.fastexp import expd


@pytest.mark.parametrize('x', [-200, -100, -10, -2, -0.5, -1e-5, 0])
def test_fastexp_smoke(x):
    assert np.allclose(np.exp(x), expd(x))


@pytest.mark.parametrize('x', [-200, -100, -10, -2, -0.5, -1e-5, 0])
def test_fastexp_timing(x):
    # call a few tims for numba overhead
    for _ in range(2):
        expd(x)

    t0 = time.time()
    for _ in range(1000):
        np.exp(x)
    t0 = time.time() - t0

    t0f = time.time()
    for _ in range(1000):
        expd(x)
    t0f = time.time() - t0f

    # it should be faster
    assert t0f < t0, {'numpy': t0, 'fastexp': t0f}
