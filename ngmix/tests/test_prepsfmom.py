import galsim
import numpy as np
import pytest

from ngmix.prepsfmom import KSigmaMom, PrePSFGaussMom, _make_mom_res
from ngmix import Jacobian
from ngmix import Observation


def _report_info(s, arr, mn, err):
    if mn is not None and err is not None:
        print(
            "%s:" % s,
            np.mean(arr), mn, np.mean(arr)/mn - 1,
            np.std(arr), err, np.std(arr)/err - 1,
            flush=True,
        )
    else:
        print(
            "%s:" % s,
            np.mean(arr), None, None,
            np.std(arr), None, None,
            flush=True,
        )


@pytest.mark.parametrize("cls", [KSigmaMom, PrePSFGaussMom])
def test_prepsfmom_raises_nopsf(cls):
    fitter = cls(1.2)
    obs = Observation(image=np.zeros((10, 10)))
    with pytest.raises(RuntimeError) as e:
        fitter.go(obs)

    assert "PSF must be set" in str(e.value)

    fitter = cls(1.2)
    obs = Observation(image=np.zeros((10, 10)))
    fitter.go(obs, no_psf=True)


@pytest.mark.parametrize("cls", [KSigmaMom, PrePSFGaussMom])
def test_prepsfmom_raises_badjacob(cls):
    fitter = cls(1.2)

    gs_wcs = galsim.ShearWCS(
        0.2, galsim.Shear(g1=-0.1, g2=0.06)).jacobian()
    jac = Jacobian(
        y=0, x=0,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

    psf_jac = Jacobian(
        y=0, x=0,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy*2)

    obs = Observation(
        image=np.zeros((10, 10)),
        jacobian=jac,
        psf=Observation(image=np.zeros((10, 10)), jacobian=psf_jac),
    )

    with pytest.raises(RuntimeError) as e:
        fitter.go(obs)
    assert "same WCS Jacobia" in str(e.value)


def _stack_list_of_dicts(res):
    def _get_dtype(v):
        if isinstance(v, float):
            return ('f8',)
        elif isinstance(v, int):
            return ('i4',)
        elif isinstance(v, str):
            return ('U256',)
        elif hasattr(v, "dtype") and hasattr(v, "shape"):
            if "float" in str(v.dtype):
                dstr = "f8"
            else:
                dstr = "i8"

            if len(v.shape) == 1:
                return (dstr, v.shape[0])
            else:
                return (dstr, v.shape)
        else:
            raise RuntimeError("cannot interpret dtype of '%s'" % v)

    dtype = []
    for k, v in res[0].items():
        dtype.append((k,) + _get_dtype(v))
    d = np.zeros(len(res), dtype=dtype)
    for i in range(len(res)):
        for k, v in res[i].items():
            d[k][i] = v

    return d


@pytest.mark.parametrize("cls", [KSigmaMom, PrePSFGaussMom])
@pytest.mark.parametrize('snr', [1e1, 1e3])
@pytest.mark.parametrize('pixel_scale', [0.125, 0.25])
@pytest.mark.parametrize('fwhm,psf_fwhm', [(0.6, 0.9), (1.5, 0.9)])
@pytest.mark.parametrize('mom_fwhm', [2.0, 1.5, 1.2])
@pytest.mark.parametrize('image_size', [57, 58])
@pytest.mark.parametrize('psf_image_size', [33, 34])
@pytest.mark.parametrize('pad_factor', [2, 1, 1.5])
def test_prepsfmom_gauss(
    pad_factor, image_size, psf_image_size, fwhm, psf_fwhm, pixel_scale, snr, mom_fwhm,
    cls,
):
    """fast test at a range of parameters to check that things come out ok"""
    rng = np.random.RandomState(seed=100)

    cen = (image_size - 1)/2
    psf_cen = (psf_image_size - 1)/2
    gs_wcs = galsim.ShearWCS(
        pixel_scale, galsim.Shear(g1=-0.1, g2=0.06)).jacobian()
    scale = np.sqrt(gs_wcs.pixelArea())
    shift = rng.uniform(low=-scale/2, high=scale/2, size=2)
    psf_shift = rng.uniform(low=-scale/2, high=scale/2, size=2)
    xy = gs_wcs.toImage(galsim.PositionD(shift))
    psf_xy = gs_wcs.toImage(galsim.PositionD(psf_shift))

    jac = Jacobian(
        y=cen + xy.y, x=cen + xy.x,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

    psf_jac = Jacobian(
        y=psf_cen + psf_xy.y, x=psf_cen + psf_xy.x,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

    gal = galsim.Gaussian(
        fwhm=fwhm
    ).shear(
        g1=-0.1, g2=0.2
    ).withFlux(
        400
    ).shift(
        dx=shift[0], dy=shift[1]
    )
    psf = galsim.Gaussian(
        fwhm=psf_fwhm
    ).shear(
        g1=0.3, g2=-0.15
    )
    im = galsim.Convolve([gal, psf]).drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs
    ).array
    noise = np.sqrt(np.sum(im**2)) / snr
    wgt = np.ones_like(im) / noise**2

    psf_im = psf.shift(
        dx=psf_shift[0], dy=psf_shift[1]
    ).drawImage(
        nx=psf_image_size,
        ny=psf_image_size,
        wcs=gs_wcs
    ).array

    fitter = cls(
        fwhm=mom_fwhm,
        pad_factor=pad_factor,
    )

    # get true flux
    im_true = gal.drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs,
        method='no_pixel').array
    obs = Observation(
        image=im_true,
        jacobian=jac,
    )
    res = cls(fwhm=mom_fwhm, pad_factor=pad_factor).go(obs=obs, no_psf=True)
    flux_true = res["flux"]
    T_true = res["T"]
    g1_true = res["e"][0]
    g2_true = res["e"][1]

    res = []
    for _ in range(100):
        _im = im + rng.normal(size=im.shape, scale=noise)
        obs = Observation(
            image=_im,
            weight=wgt,
            jacobian=jac,
            psf=Observation(image=psf_im, jacobian=psf_jac),
        )

        _res = fitter.go(obs=obs)
        if _res['flags'] == 0:
            res.append(_res)

    res = _stack_list_of_dicts(res)

    if np.mean(res["flux"])/np.mean(res["flux_err"]) > 7:
        print("\n")
        _report_info("snr", np.mean(res["flux"])/np.mean(res["flux_err"]), None, None)
        _report_info("flux", res["flux"], flux_true, np.mean(res["flux_err"]))
        _report_info("T", res["T"], T_true, np.mean(res["T_err"]))
        _report_info("g1", res["e"][:, 0], g1_true, np.mean(res["e_err"][0]))
        _report_info("g2", res["e"][:, 1], g2_true, np.mean(res["e_err"][1]))
        mom_cov = np.cov(res["mom"].T)
        print("mom cov ratio:\n", np.mean(res["mom_cov"], axis=0)/mom_cov, flush=True)
        assert np.allclose(
            np.abs(np.mean(res["flux"]) - flux_true)/np.mean(res["flux_err"]),
            0,
            atol=4,
            rtol=0,
        )
        assert np.allclose(
            np.mean(res["flux"]), flux_true, atol=0, rtol=0.1)
        assert np.allclose(
            np.std(res["flux"]), np.mean(res["flux_err"]), atol=0, rtol=0.2)


@pytest.mark.parametrize("cls,mom_fwhm,snr", [
    (KSigmaMom, 2.0, 1e2),
    pytest.param(
        PrePSFGaussMom, 8, 5e2,
        marks=pytest.mark.xfail(
            reason="Gaussian pre-PSF moment errors do not yet work!")
    ),
])
@pytest.mark.parametrize('pixel_scale', [0.25])
@pytest.mark.parametrize('fwhm,psf_fwhm', [
    (2.0, 1.0),
])
@pytest.mark.parametrize('image_size', [
    101,
])
@pytest.mark.parametrize('pad_factor', [
    1.5,
])
def test_prepsfmom_mn_cov(
    pad_factor, image_size, fwhm, psf_fwhm, pixel_scale, snr, mom_fwhm, cls,
):
    """Slower test to make sure means and errors are right
    w/ tons of monte carlo samples.
    """
    rng = np.random.RandomState(seed=100)

    cen = (image_size - 1)/2
    gs_wcs = galsim.ShearWCS(
        pixel_scale, galsim.Shear(g1=-0.1, g2=0.06)).jacobian()
    scale = np.sqrt(gs_wcs.pixelArea())
    shift = rng.uniform(low=-scale/2, high=scale/2, size=2)
    psf_shift = rng.uniform(low=-scale/2, high=scale/2, size=2)
    xy = gs_wcs.toImage(galsim.PositionD(shift))
    psf_xy = gs_wcs.toImage(galsim.PositionD(psf_shift))

    jac = Jacobian(
        y=cen + xy.y, x=cen + xy.x,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

    psf_jac = Jacobian(
        y=26 + psf_xy.y, x=26 + psf_xy.x,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

    gal = galsim.Gaussian(
        fwhm=fwhm
    ).shear(
        g1=-0.1, g2=0.2
    ).withFlux(
        400
    ).shift(
        dx=shift[0], dy=shift[1]
    )
    psf = galsim.Gaussian(
        fwhm=psf_fwhm
    ).shear(
        g1=0.3, g2=-0.15
    )
    im = galsim.Convolve([gal, psf]).drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs
    ).array
    noise = np.sqrt(np.sum(im**2)) / snr
    wgt = np.ones_like(im) / noise**2

    psf_im = psf.shift(
        dx=psf_shift[0], dy=psf_shift[1]
    ).drawImage(
        nx=53,
        ny=53,
        wcs=gs_wcs
    ).array

    fitter = cls(
        fwhm=mom_fwhm,
        pad_factor=pad_factor,
    )

    # get true flux
    im_true = gal.drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs,
        method='no_pixel').array
    obs = Observation(
        image=im_true,
        jacobian=jac,
    )
    res = cls(fwhm=mom_fwhm, pad_factor=pad_factor).go(obs=obs, no_psf=True)
    flux_true = res["flux"]
    T_true = res["T"]
    g1_true = res["e"][0]
    g2_true = res["e"][1]

    res = []
    for _ in range(10_000):
        _im = im + rng.normal(size=im.shape, scale=noise)
        obs = Observation(
            image=_im,
            weight=wgt,
            jacobian=jac,
            psf=Observation(image=psf_im, jacobian=psf_jac),
        )

        _res = fitter.go(obs=obs)
        if _res['flags'] == 0:
            res.append(_res)

    res = _stack_list_of_dicts(res)

    print("\n")
    _report_info("snr", np.mean(res["flux"])/np.mean(res["flux_err"]), None, None)
    _report_info("flux", res["flux"], flux_true, np.mean(res["flux_err"]))
    _report_info("T", res["T"], T_true, np.mean(res["T_err"]))
    _report_info("g1", res["e"][:, 0], g1_true, np.mean(res["e_err"][0]))
    _report_info("g2", res["e"][:, 1], g2_true, np.mean(res["e_err"][1]))
    mom_cov = np.cov(res["mom"].T)
    print("mom cov ratio:\n", np.mean(res["mom_cov"], axis=0)/mom_cov, flush=True)
    print("mom cov meas:\n", mom_cov, flush=True)
    print("mom cov pred:\n", np.mean(res["mom_cov"], axis=0), flush=True)

    assert np.allclose(np.mean(res["flux"]), flux_true, atol=0, rtol=1e-2)
    assert np.allclose(np.mean(res["T"]), T_true, atol=0, rtol=1e-2)
    assert np.allclose(np.mean(res["e"][:, 0]), g1_true, atol=0, rtol=1e-2)
    assert np.allclose(np.mean(res["e"][:, 1]), g2_true, atol=0, rtol=1e-2)

    assert np.allclose(np.std(res["flux"]), np.mean(res["flux_err"]), atol=0, rtol=2e-2)
    assert np.allclose(np.std(res["T"]), np.mean(res["T_err"]), atol=0, rtol=2e-2)
    assert np.allclose(
        np.std(res["e"][:, 0]), np.mean(res["e_err"][:, 0]), atol=0, rtol=2e-2)
    assert np.allclose(
        np.std(res["e"][:, 1]), np.mean(res["e_err"][:, 1]), atol=0, rtol=2e-2)

    assert np.allclose(
        res["mom_cov"], np.mean(res["mom_cov"], axis=0), atol=0, rtol=4e-1)


@pytest.mark.parametrize("cls,mom_fwhm,snr", [
    (KSigmaMom, 2.0, 1e2),
    pytest.param(
        PrePSFGaussMom, 8, 5e2,
        marks=pytest.mark.xfail(
            reason="Gaussian pre-PSF moment errors do not yet work!")
    ),
])
@pytest.mark.parametrize('pixel_scale', [0.25])
@pytest.mark.parametrize('fwhm', [
    2,
])
@pytest.mark.parametrize('image_size', [
    101,
])
@pytest.mark.parametrize('pad_factor', [
    1.5,
])
def test_prepsfmom_mn_cov_nopsf(
    pad_factor, image_size, fwhm, pixel_scale, snr, mom_fwhm, cls,
):
    """Slower test to make sure means and errors are right
    w/ tons of monte carlo samples.
    """
    rng = np.random.RandomState(seed=100)

    cen = (image_size - 1)/2
    gs_wcs = galsim.ShearWCS(
        pixel_scale, galsim.Shear(g1=-0.1, g2=0.06)).jacobian()
    scale = np.sqrt(gs_wcs.pixelArea())
    shift = rng.uniform(low=-scale/2, high=scale/2, size=2)
    xy = gs_wcs.toImage(galsim.PositionD(shift))

    jac = Jacobian(
        y=cen + xy.y, x=cen + xy.x,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

    gal = galsim.Gaussian(
        fwhm=fwhm
    ).shear(
        g1=-0.1, g2=0.2
    ).withFlux(
        400
    ).shift(
        dx=shift[0], dy=shift[1]
    )
    im = gal.drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs
    ).array
    noise = np.sqrt(np.sum(im**2)) / snr
    wgt = np.ones_like(im) / noise**2

    fitter = cls(
        fwhm=mom_fwhm,
        pad_factor=pad_factor,
    )

    # get true flux
    im_true = gal.drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs,
    ).array
    obs = Observation(
        image=im_true,
        jacobian=jac,
    )
    res = cls(fwhm=mom_fwhm, pad_factor=pad_factor).go(obs=obs, no_psf=True)
    flux_true = res["flux"]
    T_true = res["T"]
    g1_true = res["e"][0]
    g2_true = res["e"][1]

    res = []
    for _ in range(10_000):
        _im = im + rng.normal(size=im.shape, scale=noise)
        obs = Observation(
            image=_im,
            weight=wgt,
            jacobian=jac,
        )

        _res = fitter.go(obs=obs, no_psf=True)
        if _res['flags'] == 0:
            res.append(_res)

    res = _stack_list_of_dicts(res)

    print("\n")
    _report_info("snr", np.mean(res["flux"])/np.mean(res["flux_err"]), None, None)
    _report_info("flux", res["flux"], flux_true, np.mean(res["flux_err"]))
    _report_info("T", res["T"], T_true, np.mean(res["T_err"]))
    _report_info("g1", res["e"][:, 0], g1_true, np.mean(res["e_err"][0]))
    _report_info("g2", res["e"][:, 1], g2_true, np.mean(res["e_err"][1]))
    mom_cov = np.cov(res["mom"].T)
    print("mom cov ratio:\n", np.mean(res["mom_cov"], axis=0)/mom_cov, flush=True)
    print("mom cov meas:\n", mom_cov, flush=True)
    print("mom cov pred:\n", np.mean(res["mom_cov"], axis=0), flush=True)

    assert np.allclose(np.mean(res["flux"]), flux_true, atol=0, rtol=1e-2)
    assert np.allclose(np.mean(res["T"]), T_true, atol=0, rtol=1e-2)
    assert np.allclose(np.mean(res["e"][:, 0]), g1_true, atol=0, rtol=1e-2)
    assert np.allclose(np.mean(res["e"][:, 1]), g2_true, atol=0, rtol=1e-2)

    assert np.allclose(np.std(res["flux"]), np.mean(res["flux_err"]), atol=0, rtol=2e-2)
    assert np.allclose(np.std(res["T"]), np.mean(res["T_err"]), atol=0, rtol=2e-2)
    assert np.allclose(
        np.std(res["e"][:, 0]), np.mean(res["e_err"][:, 0]), atol=0, rtol=2e-2)
    assert np.allclose(
        np.std(res["e"][:, 1]), np.mean(res["e_err"][:, 1]), atol=0, rtol=2e-2)

    assert np.allclose(
        res["mom_cov"], np.mean(res["mom_cov"], axis=0), atol=0, rtol=4e-1)


def test_prepsfmom_make_mom_res_flags():
    mom = np.ones(4)
    mom_cov = np.diag(np.ones(4))

    # weird cov
    for i in range(4):
        _mom_cov = mom_cov.copy()
        _mom_cov[i, i] = -1
        res = _make_mom_res(mom, _mom_cov)
        assert (res["flags"] & 0x40) != 0
        assert "zero or neg moment var" in res["flagstr"]
        if i == 0:
            assert (res["flux_flags"] & 0x40) != 0
            assert "zero or neg flux var" in res["flux_flagstr"]
        else:
            assert res["flux_flags"] == 0
            assert res["flux_flagstr"] == ""

        if i < 2:
            assert (res["T_flags"] & 0x40) != 0
            assert "zero or neg flux/T var" in res["T_flagstr"]
        else:
            assert res["T_flags"] == 0
            assert res["T_flagstr"] == ""

    # neg flux
    _mom = mom.copy()
    _mom[0] = -1
    res = _make_mom_res(_mom, mom_cov)
    assert (res["flags"] & 0x4) != 0
    assert "flux <= 0" in res["flagstr"]
    assert res["flux_flags"] == 0
    assert res["flux_flagstr"] == ""
    assert (res["T_flags"] & 0x4) != 0
    assert "flux <= 0" in res["T_flagstr"]

    # neg T
    _mom = mom.copy()
    _mom[1] = -1
    res = _make_mom_res(_mom, mom_cov)
    assert (res["flags"] & 0x8) != 0
    assert "T <= 0" in res["flagstr"]
    assert res["flux_flags"] == 0
    assert res["flux_flagstr"] == ""
    assert res["T_flags"] == 0
    assert res["T_flagstr"] == ""

    # bad shape errs
    for i in [2, 3]:
        _mom_cov = mom_cov.copy()
        _mom_cov[1, i] = np.nan
        _mom_cov[i, 1] = np.nan
        res = _make_mom_res(mom, _mom_cov)
        assert (res["flags"] & 0x100) != 0
        assert "non-finite shape errors" in res["flagstr"]
        assert res["flux_flags"] == 0
        assert res["flux_flagstr"] == ""
        assert res["T_flags"] == 0
        assert res["T_flagstr"] == ""


@pytest.mark.parametrize("cls", [KSigmaMom, PrePSFGaussMom])
@pytest.mark.parametrize('pixel_scale', [0.125, 0.25])
@pytest.mark.parametrize('fwhm,psf_fwhm', [(0.6, 0.9), (1.5, 0.9)])
@pytest.mark.parametrize('image_size', [57, 58])
@pytest.mark.parametrize('psf_image_size', [33, 34])
@pytest.mark.parametrize('pad_factor', [2, 1, 1.5, 4])
def test_prepsfmom_gauss_true_flux(
    pad_factor, psf_image_size, image_size, fwhm, psf_fwhm, pixel_scale, cls
):
    rng = np.random.RandomState(seed=100)

    snr = 1e8
    mom_fwhm = 1000.0

    cen = (image_size - 1)/2
    psf_cen = (psf_image_size - 1)/2
    gs_wcs = galsim.ShearWCS(
        pixel_scale, galsim.Shear(g1=-0.1, g2=0.06)).jacobian()
    scale = np.sqrt(gs_wcs.pixelArea())
    shift = rng.uniform(low=-scale/2, high=scale/2, size=2)
    psf_shift = rng.uniform(low=-scale/2, high=scale/2, size=2)
    xy = gs_wcs.toImage(galsim.PositionD(shift))
    psf_xy = gs_wcs.toImage(galsim.PositionD(psf_shift))

    jac = Jacobian(
        y=cen + xy.y, x=cen + xy.x,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

    psf_jac = Jacobian(
        y=psf_cen + psf_xy.y, x=psf_cen + psf_xy.x,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

    gal = galsim.Gaussian(
        fwhm=fwhm
    ).shear(
        g1=-0.1, g2=0.2
    ).withFlux(
        400
    ).shift(
        dx=shift[0], dy=shift[1]
    )
    psf = galsim.Gaussian(
        fwhm=psf_fwhm
    ).shear(
        g1=0.3, g2=-0.15
    )
    im = galsim.Convolve([gal, psf]).drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs
    ).array
    noise = np.sqrt(np.sum(im**2)) / snr
    wgt = np.ones_like(im) / noise**2

    psf_im = psf.shift(
        dx=psf_shift[0], dy=psf_shift[1]
    ).drawImage(
        nx=psf_image_size,
        ny=psf_image_size,
        wcs=gs_wcs
    ).array

    fitter = cls(
        fwhm=mom_fwhm,
        pad_factor=pad_factor,
    )

    # get true flux
    im_true = gal.drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs,
        method='no_pixel').array
    obs = Observation(
        image=im_true,
        jacobian=jac,
    )
    res = fitter.go(obs=obs, no_psf=True)
    flux_true = res["flux"]
    assert np.allclose(flux_true, 400, atol=0, rtol=1e-4)

    obs = Observation(
        image=im,
        weight=wgt,
        jacobian=jac,
        psf=Observation(image=psf_im, jacobian=psf_jac),
    )
    res = fitter.go(obs=obs, no_psf=True)
    flux_true = res["flux"]
    assert np.allclose(flux_true, 400, atol=0, rtol=1e-4)
