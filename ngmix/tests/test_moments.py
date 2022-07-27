import numpy as np

import pytest
import galsim
from ..gaussmom import GaussMom
from ..jacobian import Jacobian
from ..observation import Observation

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
    regularize_mom_shapes,
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


@pytest.mark.parametrize("fwhm_reg", [0, 0.8])
@pytest.mark.parametrize("has_nan", [True, False])
def test_regularize_mom_shapes(fwhm_reg, has_nan):
    rng = np.random.RandomState(seed=100)

    fwhm = 0.9
    image_size = 107
    cen = (image_size - 1)/2
    gs_wcs = galsim.ShearWCS(
        0.125, galsim.Shear(g1=0, g2=0)).jacobian()

    obj = galsim.Gaussian(
        fwhm=fwhm
    ).shear(
        g1=-0.1, g2=0.3
    ).withFlux(
        400)
    im = obj.drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs,
        method='no_pixel').array
    noise = np.sqrt(np.sum(im**2)) / 1e2
    wgt = np.ones_like(im) / noise**2

    fitter = GaussMom(fwhm=1.2)

    # get true flux
    jac = Jacobian(
        y=cen, x=cen,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)
    obs = Observation(
        image=im,
        jacobian=jac,
        weight=wgt,
    )
    res = fitter.go(obs=obs)

    if has_nan:
        res["sums"][0] = np.nan
        res["sums"][1] = np.nan

    res_reg = regularize_mom_shapes(res, fwhm_reg)

    if has_nan:
        assert np.isnan(res_reg["sums"][0])
        assert np.isnan(res_reg["sums"][1])
    assert np.all(np.isfinite(res_reg["sums"][2:]))

    T_reg = fwhm_to_T(fwhm_reg)

    if not has_nan:
        assert np.allclose(res["sums"][[0, 1]], res_reg["sums"][[0, 1]])
    assert np.allclose(res["sums"][4] + T_reg * res["sums"][5], res_reg["sums"][4])
    if fwhm_reg > 0:
        assert not np.allclose(res["sums"][4], res_reg["sums"][4])
    assert np.allclose(res["sums"][[2, 3, 5]], res_reg["sums"][[2, 3, 5]])
    for col in ["flux", "flux_err", "T", "T_err", "T_flags", "s2n"]:
        assert np.allclose(res[col], res_reg[col])
    for col in ["e1", "e2", "e", "e_err", "e_cov"]:
        if fwhm_reg > 0:
            assert not np.allclose(res[col], res_reg[col])
        else:
            assert np.allclose(res[col], res_reg[col])
