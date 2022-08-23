import galsim
import numpy as np
import pytest
import time
from flaky import flaky
from numpy.testing import assert_allclose

from ngmix.prepsfmom import (
    KSigmaMom, PGaussMom,
    _build_square_apodization_mask,
    PrePSFMom,
    _gauss_kernels,
    _zero_pad_and_compute_fft_cached_impl,
)
from ngmix import Jacobian
from ngmix import Observation
from ngmix.moments import make_mom_result
import ngmix.flags


def _report_info(s, arr, mn, err):
    if mn is not None and err is not None:
        print(
            "%s:" % s,
            np.mean(arr), mn, np.mean(arr)/mn - 1,
            np.std(arr), err, np.std(arr)/err - 1,
            np.abs(np.mean(arr))/np.std(arr),
            flush=True,
        )
    else:
        print(
            "%s:" % s,
            np.mean(arr), None, None,
            np.std(arr), None, None,
            None,
            flush=True,
        )


def test_prepsfmom_kind():
    fitter = PrePSFMom(2.0, 'gauss')
    assert fitter.kind == 'pgauss'
    fitter = PrePSFMom(2.0, 'pgauss')
    assert fitter.kind == 'pgauss'
    fitter = PrePSFMom(2.0, 'ksigma')
    assert fitter.kind == 'ksigma'
    fitter = PGaussMom(2.0)
    assert fitter.kind == 'pgauss'
    fitter = KSigmaMom(2.0)
    assert fitter.kind == 'ksigma'


@pytest.mark.parametrize("cls", [KSigmaMom, PGaussMom])
def test_prepsfmom_raises_nopsf(cls):
    fitter = cls(20)
    obs = Observation(image=np.zeros((1000, 1000)))
    with pytest.raises(RuntimeError) as e:
        fitter.go(obs)

    assert "PSF must be set" in str(e.value)

    fitter = cls(20)
    obs = Observation(image=np.zeros((1000, 1000)))
    fitter.go(obs, no_psf=True)


@pytest.mark.parametrize("cls", [KSigmaMom, PGaussMom])
def test_prepsfmom_raises_nonsquare(cls):
    fitter = cls(20)
    obs = Observation(image=np.zeros((100, 90)))
    with pytest.raises(ValueError) as e:
        fitter.go(obs)

    assert "square" in str(e.value)


@pytest.mark.parametrize("cls", [KSigmaMom, PGaussMom])
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


@flaky(max_runs=10)
def test_prepsfmom_speed_and_cache():
    image_size = 48
    psf_image_size = 53
    pixel_scale = 0.263
    fwhm = 0.9
    psf_fwhm = 0.9
    snr = 20
    mom_fwhm = 2

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

    # now we test the speed + caching
    _gauss_kernels.cache_clear()
    _zero_pad_and_compute_fft_cached_impl.cache_clear()

    # the first fit will do numba stuff, so we exclude it
    # we also perturb the various inputs to fool our caches
    fitter = PGaussMom(
        fwhm=mom_fwhm + 1e-3,
    )

    obs = Observation(
        image=im + 1e-6,
        weight=wgt,
        jacobian=jac,
        psf=Observation(image=psf_im + 1e-8, jacobian=psf_jac),
    )

    dt = time.time()
    fitter.go(obs=obs)
    dt1 = time.time() - dt
    print("\n%0.4f ms for first fit" % (dt1*1000))

    # we miss once here for kernels, twice for images
    assert _gauss_kernels.cache_info().misses == 1
    assert _zero_pad_and_compute_fft_cached_impl.cache_info().misses == 2

    # the second fit will have numba cached, but not the other kernel and FFT caches
    fitter = PGaussMom(
        fwhm=mom_fwhm,
    )

    obs = Observation(
        image=im,
        weight=wgt,
        jacobian=jac,
        psf=Observation(image=psf_im, jacobian=psf_jac),
    )

    dt = time.time()
    fitter.go(obs=obs)
    dt2 = time.time() - dt
    print("%0.4f ms for second fit" % (dt2*1000))

    # we miss twice for kernels, total of 3 times since psf changed
    assert _gauss_kernels.cache_info().misses == 2
    assert _zero_pad_and_compute_fft_cached_impl.cache_info().misses == 4

    # now we test with full caching
    nfit = 2000
    dt = time.time()
    for _ in range(nfit):
        with obs.writeable():
            obs.image += 1e-6
        fitter.go(obs=obs)
    dt3 = time.time() - dt

    print("%0.4f ms per fit" % (dt3/nfit*1000))

    # we should never miss again for the calls above
    assert _gauss_kernels.cache_info().misses == 2
    assert _zero_pad_and_compute_fft_cached_impl.cache_info().misses == 4 + nfit

    # if numba stuff is cached this does not work so commented out
    # assert dt2 < dt1
    assert dt3/nfit < dt2*0.6


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


@pytest.mark.parametrize("cls", [KSigmaMom, PGaussMom])
@pytest.mark.parametrize('snr', [1e1, 1e3])
@pytest.mark.parametrize('pixel_scale', [0.125, 0.25])
@pytest.mark.parametrize('fwhm,psf_fwhm', [(0.6, 0.9), (1.5, 0.9)])
@pytest.mark.parametrize('mom_fwhm', [2.0, 1.5, 1.2])
@pytest.mark.parametrize('image_size', [57, 58])
@pytest.mark.parametrize('psf_image_size', [33, 34])
@pytest.mark.parametrize('pad_factor', [3.5, 2])
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
        mom_cov = np.cov(res["sums"].T)
        print("sums cov ratio:\n", np.mean(res["sums_cov"], axis=0)/mom_cov, flush=True)
        assert_allclose(
            np.abs(np.mean(res["flux"]) - flux_true)/np.mean(res["flux_err"]),
            0,
            atol=4,
            rtol=0,
        )
        assert_allclose(
            np.mean(res["flux"]), flux_true, atol=0, rtol=0.1)
        assert_allclose(
            np.std(res["flux"]), np.mean(res["flux_err"]), atol=0, rtol=0.2)


@pytest.mark.parametrize("cls,mom_fwhm,snr", [
    (KSigmaMom, 2.0, 1e2),
    (PGaussMom, 2.0, 1e2),
])
@pytest.mark.parametrize('pixel_scale', [0.25])
@pytest.mark.parametrize('fwhm,psf_fwhm', [(2.0, 1.0)])
@pytest.mark.parametrize('image_size', [53])
@pytest.mark.parametrize('pad_factor', [1.5])
@pytest.mark.parametrize('fwhm_smooth', [0, 1])
def test_prepsfmom_mn_cov_psf(
    pad_factor, image_size, fwhm, psf_fwhm, pixel_scale, snr, mom_fwhm, cls,
    fwhm_smooth,
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
        fwhm_smooth=fwhm_smooth,
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
    mom_cov = np.cov(res["sums"].T)
    print("sums cov ratio:\n", np.mean(res["sums_cov"], axis=0)/mom_cov, flush=True)
    print("sums cov meas:\n", mom_cov, flush=True)
    print("sums cov pred:\n", np.mean(res["sums_cov"], axis=0), flush=True)

    assert_allclose(np.mean(res["flux"]), flux_true, atol=0, rtol=1e-2)
    assert_allclose(np.mean(res["T"]), T_true, atol=0, rtol=1e-2)
    assert_allclose(np.mean(res["e"][:, 0]), g1_true, atol=0, rtol=2e-2)
    assert_allclose(np.mean(res["e"][:, 1]), g2_true, atol=0, rtol=2e-2)

    assert_allclose(np.std(res["flux"]), np.mean(res["flux_err"]), atol=0, rtol=2e-2)
    assert_allclose(np.std(res["T"]), np.mean(res["T_err"]), atol=0, rtol=2e-2)
    assert_allclose(
        np.std(res["e"][:, 0]), np.mean(res["e_err"][:, 0]), atol=0, rtol=2e-2)
    assert_allclose(
        np.std(res["e"][:, 1]), np.mean(res["e_err"][:, 1]), atol=0, rtol=2e-2)

    assert_allclose(
        mom_cov[2:, 2:],
        np.mean(res["sums_cov"][:, 2:, 2:], axis=0),
        atol=2.5e-1,
        rtol=0,
    )

    assert_allclose(
        np.diagonal(mom_cov[2:, 2:]),
        np.diagonal(np.mean(res["sums_cov"][:, 2:, 2:], axis=0)),
        atol=0,
        rtol=2e-2,
    )


@pytest.mark.parametrize("cls,mom_fwhm,snr", [(PGaussMom, 2.0, 1e2)])
@pytest.mark.parametrize('pixel_scale', [0.25])
@pytest.mark.parametrize('fwhm,psf_fwhm', [(2.0, 1.0)])
@pytest.mark.parametrize('image_size', [53])
@pytest.mark.parametrize('pad_factor', [1.5])
def test_prepsfmom_fwhm_smooth_snr(
    pad_factor, image_size, fwhm, psf_fwhm, pixel_scale, snr, mom_fwhm, cls,
):
    def _run_sim_fwhm_smooth(fwhm_smooth):
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
            fwhm_smooth=fwhm_smooth,
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

        res = []
        for _ in range(1_000):
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

        return np.abs(np.mean(res["e"], axis=0))/np.std(res["e"], axis=0)

    e_snr = _run_sim_fwhm_smooth(0)
    e_snr_smooth = _run_sim_fwhm_smooth(1)

    assert np.all(e_snr_smooth > e_snr)


@pytest.mark.parametrize("cls,mom_fwhm,snr", [
    (PGaussMom, 2.0, 1e2),
    (KSigmaMom, 2.0, 1e2),
])
@pytest.mark.parametrize('pixel_scale', [0.25])
@pytest.mark.parametrize('fwhm', [2])
@pytest.mark.parametrize('image_size', [53])
@pytest.mark.parametrize('pad_factor', [1.5])
@pytest.mark.parametrize('fwhm_smooth', [0, 1])
def test_prepsfmom_mn_cov_nopsf(
    pad_factor, image_size, fwhm, pixel_scale, snr, mom_fwhm, cls, fwhm_smooth,
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
        fwhm_smooth=fwhm_smooth,
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
    res = cls(
        fwhm=mom_fwhm, pad_factor=pad_factor,
        fwhm_smooth=fwhm_smooth,
    ).go(obs=obs, no_psf=True)
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
    mom_cov = np.cov(res["sums"].T)
    print("sums cov ratio:\n", np.mean(res["sums_cov"], axis=0)/mom_cov, flush=True)
    print("sums cov meas:\n", mom_cov, flush=True)
    print("sums cov pred:\n", np.mean(res["sums_cov"], axis=0), flush=True)

    assert_allclose(np.mean(res["flux"]), flux_true, atol=0, rtol=1e-2)
    assert_allclose(np.mean(res["T"]), T_true, atol=0, rtol=1e-2)
    assert_allclose(np.mean(res["e"][:, 0]), g1_true, atol=0, rtol=1e-2)
    assert_allclose(np.mean(res["e"][:, 1]), g2_true, atol=0, rtol=1e-2)

    assert_allclose(np.std(res["flux"]), np.mean(res["flux_err"]), atol=0, rtol=2e-2)
    assert_allclose(np.std(res["T"]), np.mean(res["T_err"]), atol=0, rtol=2e-2)
    assert_allclose(
        np.std(res["e"][:, 0]), np.mean(res["e_err"][:, 0]), atol=0, rtol=2e-2)
    assert_allclose(
        np.std(res["e"][:, 1]), np.mean(res["e_err"][:, 1]), atol=0, rtol=2e-2)

    assert_allclose(
        mom_cov[2:, 2:],
        np.mean(res["sums_cov"][:, 2:, 2:], axis=0),
        atol=2.5e-1,
        rtol=0,
    )

    assert_allclose(
        np.diagonal(mom_cov[2:, 2:]),
        np.diagonal(np.mean(res["sums_cov"][:, 2:, 2:], axis=0)),
        atol=0,
        rtol=2e-2,
    )


def test_moments_make_mom_result_flags():
    mom = np.ones(6)
    mom_cov = np.diag(np.ones(6))

    # weird cov
    for i in range(2, 6):
        _mom_cov = mom_cov.copy()
        _mom_cov[i, i] = -1
        res = make_mom_result(mom, _mom_cov, sums_norm=1)
        assert (res["flags"] & ngmix.flags.NONPOS_VAR) != 0
        assert ngmix.flags.NAME_MAP[ngmix.flags.NONPOS_VAR] in res["flagstr"]
        if i == 5:
            assert (res["flux_flags"] & ngmix.flags.NONPOS_VAR) != 0
            assert ngmix.flags.NAME_MAP[ngmix.flags.NONPOS_VAR] in res["flux_flagstr"]
        else:
            assert res["flux_flags"] == 0
            assert res["flux_flagstr"] == ""

        if i >= 4:
            assert (res["T_flags"] & ngmix.flags.NONPOS_VAR) != 0
            assert ngmix.flags.NAME_MAP[ngmix.flags.NONPOS_VAR] in res["T_flagstr"]
        else:
            assert res["T_flags"] == 0
            assert res["T_flagstr"] == ""

    # neg flux
    _mom = mom.copy()
    _mom[5] = -1
    res = make_mom_result(_mom, mom_cov, sums_norm=1)
    assert (res["flags"] & ngmix.flags.NONPOS_FLUX) != 0
    assert ngmix.flags.NAME_MAP[ngmix.flags.NONPOS_FLUX] in res["flagstr"]
    assert res["flux_flags"] == 0
    assert res["flux_flagstr"] == ""
    assert (res["T_flags"] & ngmix.flags.NONPOS_FLUX) != 0
    assert ngmix.flags.NAME_MAP[ngmix.flags.NONPOS_FLUX] in res["T_flagstr"]

    # neg T
    _mom = mom.copy()
    _mom[4] = -1
    res = make_mom_result(_mom, mom_cov, sums_norm=1)
    assert (res["flags"] & ngmix.flags.NONPOS_SIZE) != 0
    assert ngmix.flags.NAME_MAP[ngmix.flags.NONPOS_SIZE] in res["flagstr"]
    assert res["flux_flags"] == 0
    assert res["flux_flagstr"] == ""
    assert res["T_flags"] == 0
    assert res["T_flagstr"] == ""

    # bad shape errs
    for i in [2, 3]:
        _mom_cov = mom_cov.copy()
        _mom_cov[4, i] = np.nan
        _mom_cov[i, 4] = np.nan
        res = make_mom_result(mom, _mom_cov, sums_norm=1)
        assert (res["flags"] & ngmix.flags.NONPOS_SHAPE_VAR) != 0
        assert ngmix.flags.NAME_MAP[ngmix.flags.NONPOS_SHAPE_VAR] in res["flagstr"]
        assert res["flux_flags"] == 0
        assert res["flux_flagstr"] == ""
        assert res["T_flags"] == 0
        assert res["T_flagstr"] == ""


@pytest.mark.parametrize("cls", [PGaussMom, KSigmaMom])
@pytest.mark.parametrize('pixel_scale', [0.125, 0.25])
@pytest.mark.parametrize('fwhm,psf_fwhm', [(0.6, 0.9)])
@pytest.mark.parametrize('image_size', [250])
@pytest.mark.parametrize('psf_image_size', [33, 34])
@pytest.mark.parametrize('pad_factor', [4, 3.5])
def test_prepsfmom_gauss_true_flux(
    pad_factor, psf_image_size, image_size, fwhm, psf_fwhm, pixel_scale, cls
):
    rng = np.random.RandomState(seed=100)

    snr = 1e8
    mom_fwhm = 15.0

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
    assert_allclose(flux_true, 400, atol=0, rtol=5e-3)

    obs = Observation(
        image=im,
        weight=wgt,
        jacobian=jac,
        psf=Observation(image=psf_im, jacobian=psf_jac),
    )
    res = fitter.go(obs=obs)
    flux_true = res["flux"]
    assert_allclose(flux_true, 400, atol=0, rtol=5e-3)


@pytest.mark.parametrize('pixel_scale', [0.25, 0.125])
@pytest.mark.parametrize('image_size', [107])
@pytest.mark.parametrize('pad_factor', [3.5, 4])
@pytest.mark.parametrize('mom_fwhm', [2, 2.5])
@pytest.mark.parametrize('cls', [PGaussMom, KSigmaMom])
@pytest.mark.parametrize('fwhm_smooth', [0, 1.5])
def test_prepsfmom_mom_norm(
    pad_factor, image_size, pixel_scale, mom_fwhm, cls, fwhm_smooth,
):
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

    obs = Observation(
        image=np.ones((image_size, image_size)),
        jacobian=jac,
    )
    res = cls(
        fwhm=mom_fwhm, pad_factor=pad_factor, fwhm_smooth=fwhm_smooth,
    ).go(
        obs=obs, no_psf=True,
    )
    assert_allclose(res["sums_norm"], res["flux"], atol=0, rtol=2e-4)


@pytest.mark.parametrize('pixel_scale', [0.25, 0.125])
@pytest.mark.parametrize('fwhm', [2, 0.5])
@pytest.mark.parametrize('image_size', [107])
@pytest.mark.parametrize('pad_factor', [3.5, 4])
@pytest.mark.parametrize('mom_fwhm', [2, 2.5])
def test_prepsfmom_comp_to_gaussmom_simple(
    pad_factor, image_size, fwhm, pixel_scale, mom_fwhm
):
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
    res = PGaussMom(
        fwhm=mom_fwhm, pad_factor=pad_factor,
    ).go(
        obs=obs, no_psf=True, return_kernels=True,
    )

    from ngmix.gaussmom import GaussMom
    res_gmom = GaussMom(fwhm=mom_fwhm).go(obs=obs)

    for k in sorted(res):
        if k in res_gmom:
            print("%s:" % k, res[k], res_gmom[k])

    for k in ["flux", "flux_err", "T", "T_err", "e", "e_cov"]:
        assert_allclose(res[k], res_gmom[k], atol=0, rtol=1e-2)


@pytest.mark.parametrize('pixel_scale', [0.25, 0.125])
@pytest.mark.parametrize('fwhm', [2, 0.5])
@pytest.mark.parametrize('image_size', [107])
@pytest.mark.parametrize('pad_factor', [3.5, 4])
@pytest.mark.parametrize('mom_fwhm', [2, 2.5])
@pytest.mark.parametrize('fwhm_smooth', [0, 1.5])
def test_prepsfmom_comp_to_gaussmom_fwhm_smooth(
    pad_factor, image_size, fwhm, pixel_scale, mom_fwhm, fwhm_smooth
):
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
    res = PGaussMom(
        fwhm=mom_fwhm, pad_factor=pad_factor, fwhm_smooth=fwhm_smooth,
    ).go(
        obs=obs, no_psf=True,
    )

    from ngmix.gaussmom import GaussMom
    if fwhm_smooth > 0:
        im_true_smooth = galsim.Convolve(
            [gal, galsim.Gaussian(fwhm=fwhm_smooth)]
        ).drawImage(
            nx=image_size,
            ny=image_size,
            wcs=gs_wcs,
        ).array
    else:
        im_true_smooth = im_true
    obs_smooth = Observation(
        image=im_true_smooth,
        jacobian=jac,
    )
    res_gmom = GaussMom(fwhm=mom_fwhm).go(obs=obs_smooth)

    for k in sorted(res):
        if k in res_gmom:
            print("%s:" % k, res[k], res_gmom[k])

    assert_allclose(res["flux"], res_gmom["flux"], atol=0, rtol=5e-4)
    assert_allclose(res["T"], res_gmom["T"], atol=0, rtol=1e-3)
    assert_allclose(res["e"], res_gmom["e"], atol=0, rtol=1e-3)
    # the errors do not match - this is because the underlying noise model is
    # different - the pure gaussian moments weight map is an error on the convolved
    # profile whereas the pre-PSF case uses error propagation through the
    # smoothing kernel treating the weight map as applying to the unconvolved profile
    # thus we do not test the errors


def _sim_apodize(flux_factor, ap_rad):
    """
    we are simulating an object at the center with a bright object right on the
    edge of the stamp.

    We then apply apodization to the image and measure the same Gaussian moment
    with either the Fourier-space code or the real-space one.

    We compare the case with zero apodization to non-zero in the test below
    and assert that with apodization the results from Fourier-space match the
    real-space results better.
    """
    rng = np.random.RandomState(seed=100)
    image_size = 53
    pixel_scale = 0.25
    fwhm = 0.9
    mom_fwhm = 2.0
    pad_factor = 4

    cen = (image_size - 1)/2
    gs_wcs = galsim.ShearWCS(
        pixel_scale, galsim.Shear(g1=-0, g2=0.0)).jacobian()
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

    # get true flux
    im_true = gal.drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs,
    ).array

    im = im_true.copy()
    im += galsim.Exponential(
        half_light_radius=fwhm
    ).shear(
        g1=-0.5, g2=0.2
    ).shift(
        cen*pixel_scale,
        0,
    ).withFlux(
        400*flux_factor
    ).drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs,
        method="real_space",
    ).array

    obs = Observation(
        image=im,
        jacobian=jac,
    )
    res = PGaussMom(fwhm=mom_fwhm, pad_factor=pad_factor, ap_rad=ap_rad).go(
        obs=obs, no_psf=True, return_kernels=True,
    )

    ap_mask = np.ones_like(im)
    if ap_rad > 0:
        _build_square_apodization_mask(ap_rad, ap_mask)
    obs_ap = Observation(
        image=im * ap_mask,
        jacobian=jac,
    )

    from ngmix.gaussmom import GaussMom
    res_gmom = GaussMom(fwhm=mom_fwhm).go(obs=obs_ap)

    return res, res_gmom


@pytest.mark.parametrize("flux_factor", [1e2, 1e3, 1e5])
def test_prepsfmom_apodize(flux_factor):
    res, res_geom = _sim_apodize(flux_factor, 1.5)
    ap_diffs = np.array([
        np.abs(res[k] - res_geom[k])
        for k in ["e1", "e2", "T", "flux"]
    ])
    print("apodized:", ap_diffs)

    res, res_geom = _sim_apodize(flux_factor, 0)
    zero_diffs = np.array([
        np.abs(res[k] - res_geom[k])
        for k in ["e1", "e2", "T", "flux"]
    ])
    print("non-apodized:", zero_diffs)

    assert np.all(zero_diffs > ap_diffs)
