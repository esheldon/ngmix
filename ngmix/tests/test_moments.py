import numpy as np

import pytest

from ..moments import (
    sigma_to_fwhm,
    fwhm_to_sigma,
    sigma_to_r50,
    r50_to_sigma,
    fwhm_to_T,
    T_to_fwhm,
    r50_to_T,
    T_to_r50,
    moms_to_e1e2,
    get_Tround,
    get_T,
    get_sheared_M1M2T,
    get_sheared_g1g2T,
    get_sheared_moments,
    mom2e,
    mom2g,
    e2mom,
    g2mom,
)


@pytest.mark.parametrize("to_func,from_func", [
    (sigma_to_fwhm, fwhm_to_sigma),
    (sigma_to_r50, r50_to_sigma),
    (fwhm_to_T, T_to_fwhm),
    (r50_to_T, T_to_r50),
])
def test_moments_size_roundtrips(to_func, from_func):
    val = 1.2

    tval = to_func(val)
    fval = from_func(tval)
    assert np.allclose(val, fval)

    to_func, from_func = from_func, to_func
    tval = to_func(val)
    fval = from_func(tval)
    assert np.allclose(val, fval)
