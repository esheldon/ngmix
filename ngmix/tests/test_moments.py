import numpy as np

import pytest

from ..gexceptions import GMixRangeError

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


def test_moments_mom_roundtrips():
    e1e2T = (-0.1, 0.2, 1.4)
    ivals = (0.3, -0.05, 0.1)

    assert np.allclose(e1e2T, mom2e(*e2mom(*e1e2T)))
    assert np.allclose(ivals, e2mom(*mom2e(*ivals)))

    assert np.allclose(e1e2T, mom2g(*g2mom(*e1e2T)))
    assert np.allclose(ivals, g2mom(*mom2g(*ivals)))


def test_moments_tround_roundtrip():
    g1, g2 = 0.0, 0.1
    Tinit = 2.1
    assert np.allclose(get_Tround(get_T(Tinit, g1, g2), g1, g2), Tinit)
    assert np.allclose(get_T(get_Tround(Tinit, g1, g2), g1, g2), Tinit)


def test_moments_sheared_consistent():
    irr, irc, icc = 0.3, -0.05, 0.5
    s1, s2 = -0.1, 0.1
    m1 = icc - irr
    m2 = 2 * irc
    T = irr + icc

    sirr, sirc, sicc = get_sheared_moments(irr, irc, icc, s1, s2)

    sm1, sm2, sT = get_sheared_M1M2T(m1, m2, T, s1, s2)

    assert np.allclose(sm1, sicc - sirr)
    assert np.allclose(sm2, 2 * sirc)
    assert np.allclose(sT, sirr + sicc)


def test_moments_moms_to_e1e2_raises():
    with pytest.raises(GMixRangeError):
        moms_to_e1e2(0.1, 0.2, -0.1)

    with pytest.raises(GMixRangeError):
        moms_to_e1e2(np.array([0.1]), np.array([0.2]), np.array([-0.1]))
