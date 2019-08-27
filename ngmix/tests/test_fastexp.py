import time
import numpy as np
import pytest

from ngmix.fastexp_nb import exp3


@pytest.mark.parametrize('x', [-200, -100, -10, -2, -0.5, 0])
def test_fastexp_smoke(x):
    assert np.allclose(np.exp(x), exp3(x), atol=1e-4, rtol=1e-2)


@pytest.mark.parametrize('x', [-200, -100, -10, -2, -0.5, 0])
def test_fastexp_timing(x):
    t0 = time.time()
    for _ in range(1000):
        np.exp(x)
    t0 = time.time() - t0

    t0f = time.time()
    for _ in range(1000):
        exp3(x)
    t0f = time.time() - t0f

    # it should be at least 3x faster
    assert t0f < t0 * 0.33333, {'numpy': t0, 'fastexp': t0f}
