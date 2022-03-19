import galsim
import numpy as np
import pytest
import time

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


def _make_prepsfmom_sim(
    *, image_size, psf_image_size, pixel_scale, rng, fwhm, psf_fwhm,
    snr, extra_psf_fwhm
):
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
    if psf_fwhm is not None:
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
    else:
        im = gal.drawImage(
            nx=image_size,
            ny=image_size,
            wcs=gs_wcs
        ).array

    noise = np.sqrt(np.sum(im**2)) / snr
    wgt = np.ones_like(im) / noise**2

    if psf_fwhm is not None:
        psf_im = psf.shift(
            dx=psf_shift[0], dy=psf_shift[1]
        ).drawImage(
            nx=psf_image_size,
            ny=psf_image_size,
            wcs=gs_wcs
        ).array
    else:
        psf_im = None

    if extra_psf_fwhm is not None:
        extra_psf = galsim.Gaussian(
            fwhm=extra_psf_fwhm
        ).shear(
            g1=-0.2, g2=0.3
        )
        extra_psf_shift = rng.uniform(low=-scale/2, high=scale/2, size=2)
        extra_psf_xy = gs_wcs.toImage(galsim.PositionD(extra_psf_shift))
        extra_psf_im = extra_psf.shift(
            dx=extra_psf_shift[0], dy=extra_psf_shift[1]
        ).drawImage(
            nx=image_size,
            ny=image_size,
            wcs=gs_wcs
        ).array
        extra_psf_jac = Jacobian(
            y=cen + extra_psf_xy.y, x=cen + extra_psf_xy.x,
            dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
            dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy,
        )
    else:
        extra_psf_im = None
        extra_psf_jac = None

    if psf_fwhm is not None:
        # true image has pixel removed if we deconvolve the PSF
        im_true = gal.drawImage(
            nx=image_size,
            ny=image_size,
            wcs=gs_wcs,
            method='no_pixel').array
    else:
        im_true = gal.drawImage(
            nx=image_size,
            ny=image_size,
            wcs=gs_wcs
        ).array

    return dict(
        im=im,
        im_true=im_true,
        jac=jac,
        psf_jac=psf_jac,
        psf_im=psf_im,
        noise=noise,
        wgt=wgt,
        extra_psf_im=extra_psf_im,
        extra_psf_jac=extra_psf_jac,
    )


def _run_prepsfmom_sims(sdata, fitter, rng, nitr):
    # get true flux
    obs = Observation(
        image=sdata["im_true"],
        jacobian=sdata["jac"],
    )
    res = fitter.go(obs=obs, no_psf=True)
    flux_true = res["flux"]
    T_true = res["T"]
    g1_true = res["e"][0]
    g2_true = res["e"][1]

    no_psf = sdata["psf_im"] is None
    if sdata["extra_psf_im"] is not None:
        assert not no_psf

    res = []
    for _ in range(nitr):
        _im = sdata["im"] + rng.normal(size=sdata["im"].shape, scale=sdata["noise"])
        if sdata["extra_psf_im"] is None:
            obs = Observation(
                image=_im,
                weight=sdata["wgt"],
                jacobian=sdata["jac"],
                psf=(
                    Observation(image=sdata["psf_im"], jacobian=sdata["psf_jac"])
                    if not no_psf
                    else None
                ),
            )

            _res = fitter.go(obs=obs, no_psf=no_psf)
        else:
            obs = Observation(
                image=_im,
                weight=sdata["wgt"],
                jacobian=sdata["jac"],
                psf=Observation(
                    image=sdata["extra_psf_im"],
                    jacobian=sdata["extra_psf_jac"],
                ),
            )

            _res = fitter.go(
                obs=obs,
                extra_deconv_psfs=[Observation(
                    image=sdata["psf_im"],
                    jacobian=sdata["psf_jac"]
                )],
                extra_conv_psfs=[obs.psf],
            )

        if _res['flags'] == 0:
            res.append(_res)

    res = _stack_list_of_dicts(res)

    return dict(
        flux_true=flux_true,
        T_true=T_true,
        g1_true=g1_true,
        g2_true=g2_true,
        res=res,
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
def test_prepsfmom_raises_nopsf_with_extra(cls):
    fitter = cls(20)
    obs = Observation(image=np.zeros((1000, 1000)))
    with pytest.raises(RuntimeError) as e:
        fitter.go(obs, extra_deconv_psfs=[10], no_psf=True)
    assert "You can only use extra conv." in str(e.value)

    with pytest.raises(RuntimeError) as e:
        fitter.go(obs, extra_conv_psfs=[10], no_psf=True)
    assert "You can only use extra conv." in str(e.value)

    with pytest.raises(RuntimeError) as e:
        fitter.go(obs, extra_conv_psfs=[10], extra_deconv_psfs=[11], no_psf=True)
    assert "You can only use extra conv." in str(e.value)


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


@pytest.mark.parametrize("cls", [KSigmaMom, PGaussMom])
def test_prepsfmom_raises_badjacob_extra_conv(cls):
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
        psf=Observation(image=np.zeros((10, 10)), jacobian=jac),
    )

    with pytest.raises(RuntimeError) as e:
        fitter.go(
            obs,
            extra_conv_psfs=[Observation(image=np.zeros((10, 10)), jacobian=psf_jac)],
        )
    assert "same WCS Jacobia" in str(e.value)

    with pytest.raises(RuntimeError) as e:
        fitter.go(
            obs,
            extra_deconv_psfs=[Observation(image=np.zeros((10, 10)), jacobian=psf_jac)],
        )
    assert "same WCS Jacobia" in str(e.value)

    with pytest.raises(RuntimeError) as e:
        fitter.go(
            obs,
            extra_conv_psfs=[Observation(image=np.zeros((10, 10)), jacobian=psf_jac)],
            extra_deconv_psfs=[Observation(image=np.zeros((10, 10)), jacobian=psf_jac)],
        )
    assert "same WCS Jacobia" in str(e.value)


def test_prepsfmom_speed_and_cache():
    mom_fwhm = 2

    rng = np.random.RandomState(seed=100)

    sdata = _make_prepsfmom_sim(
        image_size=48,
        psf_image_size=53,
        pixel_scale=0.263,
        fwhm=0.9,
        psf_fwhm=0.9,
        snr=20,
        rng=rng,
        extra_psf_fwhm=None,
    )

    # now we test the speed + caching
    _gauss_kernels.cache_clear()
    _zero_pad_and_compute_fft_cached_impl.cache_clear()

    # the first fit will do numba stuff, so we exclude it
    # we also perturb the various inputs to fool our caches
    fitter = PGaussMom(
        fwhm=mom_fwhm + 1e-3,
    )

    im = sdata["im"] + rng.normal(size=sdata["im"].shape, scale=sdata["noise"])
    obs = Observation(
        image=im,
        weight=sdata["wgt"],
        jacobian=sdata["jac"],
        psf=Observation(image=sdata["psf_im"] + 1e-8, jacobian=sdata["psf_jac"]),
    )

    dt = time.time()
    fitter.go(obs=obs)
    dt = time.time() - dt
    print("\n%0.4f ms for first fit" % (dt*1000))

    # we miss once here
    assert _gauss_kernels.cache_info().misses == 1
    assert _zero_pad_and_compute_fft_cached_impl.cache_info().misses == 1

    # the second fit will have numba cached, but not the other kernel and FFT caches
    fitter = PGaussMom(
        fwhm=mom_fwhm,
    )

    obs = Observation(
        image=im,
        weight=sdata["wgt"],
        jacobian=sdata["jac"],
        psf=Observation(image=sdata["psf_im"], jacobian=sdata["psf_jac"]),
    )

    dt = time.time()
    fitter.go(obs=obs)
    dt = time.time() - dt
    print("%0.4f ms for second fit" % (dt*1000))

    # we miss twice since we changed the moments width and psf slightly
    assert _gauss_kernels.cache_info().misses == 2
    assert _zero_pad_and_compute_fft_cached_impl.cache_info().misses == 2

    # now we test with full caching
    nfit = 1000
    dt = time.time()
    for _ in range(nfit):
        fitter.go(obs=obs)
    dt = time.time() - dt

    print("%0.4f ms per fit" % (dt/nfit*1000))

    # we should never miss again for the calls above
    assert _gauss_kernels.cache_info().misses == 2
    assert _zero_pad_and_compute_fft_cached_impl.cache_info().misses == 2


@pytest.mark.parametrize("cls", [KSigmaMom, PGaussMom])
@pytest.mark.parametrize('snr', [1e1, 1e3])
@pytest.mark.parametrize('pixel_scale', [0.125, 0.25])
@pytest.mark.parametrize('fwhm,psf_fwhm', [(0.6, 0.9), (1.5, 0.9)])
@pytest.mark.parametrize('mom_fwhm', [2.0, 1.5, 1.2])
@pytest.mark.parametrize('image_size', [57, 58])
@pytest.mark.parametrize('psf_image_size', [33, 34])
@pytest.mark.parametrize('pad_factor', [3.5, 2])
@pytest.mark.parametrize("extra_psf_fwhm", [None, 0.8, 1.2])
def test_prepsfmom_gauss(
    pad_factor, image_size, psf_image_size, fwhm, psf_fwhm, pixel_scale, snr, mom_fwhm,
    cls, extra_psf_fwhm,
):
    """fast test at a range of parameters to check that things come out ok"""
    rng = np.random.RandomState(seed=100)

    sdata = _make_prepsfmom_sim(
        image_size=image_size,
        psf_image_size=psf_image_size,
        pixel_scale=pixel_scale,
        fwhm=fwhm,
        psf_fwhm=psf_fwhm,
        snr=snr,
        rng=rng,
        extra_psf_fwhm=extra_psf_fwhm,
    )

    fitter = cls(
        fwhm=mom_fwhm,
        pad_factor=pad_factor,
    )

    sres = _run_prepsfmom_sims(sdata, fitter, rng, 100)
    flux_true = sres["flux_true"]
    T_true = sres["T_true"]
    g1_true = sres["g1_true"]
    g2_true = sres["g2_true"]
    res = sres["res"]

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
    (PGaussMom, 2.0, 1e2),
])
@pytest.mark.parametrize('pixel_scale', [0.25])
@pytest.mark.parametrize('fwhm,psf_fwhm', [
    (2.0, 1.0),
])
@pytest.mark.parametrize('image_size', [
    53,
])
@pytest.mark.parametrize('pad_factor', [
    1.5,
])
@pytest.mark.parametrize("extra_psf_fwhm", [None, 0.8, 1.2])
def test_prepsfmom_mn_cov(
    pad_factor, image_size, fwhm, psf_fwhm, pixel_scale, snr, mom_fwhm, cls,
    extra_psf_fwhm,
):
    """Slower test to make sure means and errors are right
    w/ tons of monte carlo samples.
    """
    rng = np.random.RandomState(seed=100)

    sdata = _make_prepsfmom_sim(
        image_size=image_size,
        psf_image_size=53,
        pixel_scale=pixel_scale,
        fwhm=fwhm,
        psf_fwhm=psf_fwhm,
        snr=snr,
        rng=rng,
        extra_psf_fwhm=extra_psf_fwhm,
    )

    fitter = cls(
        fwhm=mom_fwhm,
        pad_factor=pad_factor,
    )

    sres = _run_prepsfmom_sims(sdata, fitter, rng, 10_000)
    flux_true = sres["flux_true"]
    T_true = sres["T_true"]
    g1_true = sres["g1_true"]
    g2_true = sres["g2_true"]
    res = sres["res"]

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
        mom_cov[2:, 2:],
        np.mean(res["mom_cov"][:, 2:, 2:], axis=0),
        atol=2.5e-1,
        rtol=0,
    )

    assert np.allclose(
        np.diagonal(mom_cov[2:, 2:]),
        np.diagonal(np.mean(res["mom_cov"][:, 2:, 2:], axis=0)),
        atol=0,
        rtol=2e-2,
    )


@pytest.mark.parametrize("cls,mom_fwhm,snr", [
    (KSigmaMom, 2.0, 1e2),
    (PGaussMom, 2.0, 1e2),
])
@pytest.mark.parametrize('pixel_scale', [0.25])
@pytest.mark.parametrize('fwhm', [
    2,
])
@pytest.mark.parametrize('image_size', [
    53,
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

    sdata = _make_prepsfmom_sim(
        image_size=image_size,
        psf_image_size=53,
        pixel_scale=pixel_scale,
        fwhm=fwhm,
        psf_fwhm=None,
        snr=snr,
        rng=rng,
        extra_psf_fwhm=None,
    )

    fitter = cls(
        fwhm=mom_fwhm,
        pad_factor=pad_factor,
    )

    sres = _run_prepsfmom_sims(sdata, fitter, rng, 10_000)
    flux_true = sres["flux_true"]
    T_true = sres["T_true"]
    g1_true = sres["g1_true"]
    g2_true = sres["g2_true"]
    res = sres["res"]

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
        mom_cov[2:, 2:],
        np.mean(res["mom_cov"][:, 2:, 2:], axis=0),
        atol=2.5e-1,
        rtol=0,
    )

    assert np.allclose(
        np.diagonal(mom_cov[2:, 2:]),
        np.diagonal(np.mean(res["mom_cov"][:, 2:, 2:], axis=0)),
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
        res = make_mom_result(mom, _mom_cov)
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
    res = make_mom_result(_mom, mom_cov)
    assert (res["flags"] & ngmix.flags.NONPOS_FLUX) != 0
    assert ngmix.flags.NAME_MAP[ngmix.flags.NONPOS_FLUX] in res["flagstr"]
    assert res["flux_flags"] == 0
    assert res["flux_flagstr"] == ""
    assert (res["T_flags"] & ngmix.flags.NONPOS_FLUX) != 0
    assert ngmix.flags.NAME_MAP[ngmix.flags.NONPOS_FLUX] in res["T_flagstr"]

    # neg T
    _mom = mom.copy()
    _mom[4] = -1
    res = make_mom_result(_mom, mom_cov)
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
        res = make_mom_result(mom, _mom_cov)
        assert (res["flags"] & ngmix.flags.NONPOS_SHAPE_VAR) != 0
        assert ngmix.flags.NAME_MAP[ngmix.flags.NONPOS_SHAPE_VAR] in res["flagstr"]
        assert res["flux_flags"] == 0
        assert res["flux_flagstr"] == ""
        assert res["T_flags"] == 0
        assert res["T_flagstr"] == ""


@pytest.mark.parametrize("extra_psf_fwhm", [None, 0.8, 1.1])
@pytest.mark.parametrize("cls", [PGaussMom, KSigmaMom])
@pytest.mark.parametrize('pixel_scale', [0.125, 0.25])
@pytest.mark.parametrize('fwhm,psf_fwhm', [(0.6, 0.9)])
@pytest.mark.parametrize('image_size', [250])
@pytest.mark.parametrize('psf_image_size', [33, 34])
@pytest.mark.parametrize('pad_factor', [4, 3.5])
def test_prepsfmom_gauss_true_flux_T(
    pad_factor, psf_image_size, image_size, fwhm, psf_fwhm, pixel_scale,
    cls, extra_psf_fwhm
):
    """This test ensures the kernels are normalize properly so
    we have the correct total flux, etc."""
    rng = np.random.RandomState(seed=100)
    snr = 1e8
    mom_fwhm = 15.0

    sdata = _make_prepsfmom_sim(
        image_size=image_size,
        psf_image_size=psf_image_size,
        pixel_scale=pixel_scale,
        fwhm=fwhm,
        psf_fwhm=psf_fwhm,
        snr=snr,
        rng=rng,
        extra_psf_fwhm=extra_psf_fwhm,
    )

    fitter = cls(
        fwhm=mom_fwhm,
        pad_factor=pad_factor,
    )

    # get true flux
    obs = Observation(
        image=sdata["im_true"],
        jacobian=sdata["jac"],
    )
    res = fitter.go(obs=obs, no_psf=True)
    flux_true = res["flux"]
    T_true = res["T"]
    g1_true = res["e1"]
    g2_true = res["e2"]
    assert np.allclose(flux_true, 400, atol=0, rtol=5e-3)

    if extra_psf_fwhm is None:
        obs = Observation(
            image=sdata["im"],
            weight=sdata["wgt"],
            jacobian=sdata["jac"],
            psf=Observation(image=sdata["psf_im"], jacobian=sdata["psf_jac"]),
        )
        res = fitter.go(obs=obs)
        flux_true = res["flux"]
        assert np.allclose(flux_true, 400, atol=0, rtol=5e-3)
        assert np.allclose(res["T"], T_true, atol=0, rtol=5e-4)
        assert np.allclose(res["e1"], g1_true, atol=5e-3, rtol=5e-3)
        assert np.allclose(res["e2"], g2_true, atol=5e-3, rtol=5e-3)
    else:
        # this should fail since it is the wrong PSF
        obs = Observation(
            image=sdata["im"],
            weight=sdata["wgt"],
            jacobian=sdata["jac"],
            psf=Observation(
                image=sdata["extra_psf_im"], jacobian=sdata["extra_psf_jac"]
            ),
        )
        res = fitter.go(obs=obs)
        flux_true = res["flux"]
        # we use a big relative difference since the T should be way off
        assert not np.allclose(res["T"], T_true, atol=0, rtol=1e-1)
        assert not np.allclose(res["e1"], g1_true, atol=0, rtol=1e-1)
        assert not np.allclose(res["e2"], g2_true, atol=0, rtol=1e-1)

        # this should work since we have the PSF correction in there
        obs = Observation(
            image=sdata["im"],
            weight=sdata["wgt"],
            jacobian=sdata["jac"],
            psf=Observation(
                image=sdata["extra_psf_im"],
                jacobian=sdata["extra_psf_jac"]
            ),
        )
        res = fitter.go(
            obs=obs,
            extra_deconv_psfs=[Observation(
                image=sdata["psf_im"], jacobian=sdata["psf_jac"],
            )],
            extra_conv_psfs=[obs.psf],
        )
        flux_true = res["flux"]
        assert np.allclose(flux_true, 400, atol=0, rtol=5e-3)
        assert np.allclose(res["T"], T_true, atol=0, rtol=5e-4)
        assert np.allclose(res["e1"], g1_true, atol=5e-3, rtol=5e-3)
        assert np.allclose(res["e2"], g2_true, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize('pixel_scale', [0.25, 0.125])
@pytest.mark.parametrize('fwhm', [
    2, 0.5,
])
@pytest.mark.parametrize('image_size', [
    107,
])
@pytest.mark.parametrize('pad_factor', [
    3.5, 4,
])
@pytest.mark.parametrize('mom_fwhm', [
    2, 2.5,
])
def test_prepsfmom_comp_to_gaussmom(
    pad_factor, image_size, fwhm, pixel_scale, mom_fwhm,
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
    res = PGaussMom(fwhm=mom_fwhm, pad_factor=pad_factor).go(
        obs=obs, no_psf=True, return_kernels=True,
    )

    from ngmix.gaussmom import GaussMom
    res_gmom = GaussMom(fwhm=mom_fwhm).go(obs=obs)

    for k in sorted(res):
        if k in res_gmom:
            print("%s:" % k, res[k], res_gmom[k])

    for k in ["flux", "flux_err", "T", "T_err", "e", "e_cov"]:
        assert np.allclose(res[k], res_gmom[k], atol=0, rtol=1e-2)


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
