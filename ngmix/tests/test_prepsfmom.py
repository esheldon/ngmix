import galsim
import numpy as np
import pytest

from ngmix.prepsfmom import PrePSFGaussMom
from ngmix import Jacobian
from ngmix import Observation
from ngmix.moments import fwhm_to_T
from ngmix.shape import e1e2_to_g1g2


@pytest.mark.parametrize('weight_fac', [1, 1e5])
@pytest.mark.parametrize('wcs_g1', [-0.5, 0])
@pytest.mark.parametrize('wcs_g2', [0, 0.5])
@pytest.mark.parametrize('g1_true', [0, 0.2])
@pytest.mark.parametrize('g2_true', [-0.1, 0])
def test_prepsfmom_smoke(g1_true, g2_true, wcs_g1, wcs_g2, weight_fac):
    rng = np.random.RandomState(seed=100)

    fwhm = 0.9
    image_size = 107
    cen = (image_size - 1)/2
    gs_wcs = galsim.ShearWCS(
        0.125, galsim.Shear(g1=wcs_g1, g2=wcs_g2)).jacobian()

    obj = galsim.Gaussian(
        fwhm=fwhm
    ).shear(
        g1=g1_true, g2=g2_true
    ).withFlux(
        400)
    im = obj.drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs,
        method='no_pixel').array
    noise = np.sqrt(np.sum(im**2)) / 1e18
    wgt = np.ones_like(im) / noise**2
    scale = np.sqrt(gs_wcs.pixelArea())

    g1arr = []
    g2arr = []
    Tarr = []
    farr = []

    fitter = PrePSFGaussMom(fwhm=fwhm * weight_fac, pad_factor=1.2)

    # get true flux
    jac = Jacobian(
        y=cen, x=cen,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)
    obs = Observation(
        image=im,
        jacobian=jac,
    )
    flux_true = fitter.go(obs=obs)["flux"]

    for _ in range(50):
        shift = rng.uniform(low=-scale/2, high=scale/2, size=2)
        xy = gs_wcs.toImage(galsim.PositionD(shift))

        im = obj.shift(
            dx=shift[0], dy=shift[1]
        ).drawImage(
            nx=image_size,
            ny=image_size,
            wcs=gs_wcs,
            method='no_pixel',
            dtype=np.float64).array

        jac = Jacobian(
            y=cen + xy.y, x=cen + xy.x,
            dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
            dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

        _im = im + (rng.normal(size=im.shape) * noise)
        obs = Observation(
            image=_im,
            weight=wgt,
            jacobian=jac)

        # use a huge weight so that we get the raw moments back out
        res = fitter.go(obs=obs)
        if res['flags'] == 0:
            if weight_fac > 1:
                # for unweighted we need to convert e to g
                _g1, _g2 = e1e2_to_g1g2(res['e'][0], res['e'][1])
            else:
                # we are weighting by the round gaussian before shearing.
                # Turns out this gives e that equals the shear g
                _g1, _g2 = res['e'][0], res['e'][1]
            g1arr.append(_g1)
            g2arr.append(_g2)
            Tarr.append(res['pars'][4])
            farr.append(res['pars'][5])

    g1 = np.mean(g1arr)
    g2 = np.mean(g2arr)

    gtol = 5e-7
    assert np.abs(g1 - g1_true) < gtol, (g1, np.std(g1arr)/np.sqrt(len(g1arr)))
    assert np.abs(g2 - g2_true) < gtol, (g2, np.std(g2arr)/np.sqrt(len(g2arr)))

    # T test should only pass when the weight function is constant so
    # weight_fac needs to be rally big
    if g1_true == 0 and g2_true == 0 and weight_fac > 1:
        T = np.mean(Tarr)
        assert np.abs(T - fwhm_to_T(fwhm)) < 1e-6

    if weight_fac > 1:
        assert np.allclose(flux_true, np.sum(im))
    mean_flux = np.mean(farr)
    assert np.abs(mean_flux - flux_true) < 5e-4, (np.mean(farr), np.std(farr))


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


@pytest.mark.parametrize('pixel_scale', [0.125, 0.5])
@pytest.mark.parametrize('image_size', [107, 112])
@pytest.mark.parametrize('pad_factor', [1, 3.1, 1.3234])
def test_prepsfmom_error(pad_factor, image_size, pixel_scale):
    rng = np.random.RandomState(seed=100)

    fwhm = 0.9
    cen = (image_size - 1)/2
    gs_wcs = galsim.ShearWCS(
        pixel_scale, galsim.Shear(g1=-0.1, g2=0.23)).jacobian()
    scale = np.sqrt(gs_wcs.pixelArea())
    shift = rng.uniform(low=-scale/2, high=scale/2, size=2) * 0
    xy = gs_wcs.toImage(galsim.PositionD(shift))

    obj = galsim.Gaussian(
        fwhm=fwhm
    ).shear(
        g1=-0.1, g2=0.1
    ).withFlux(
        400)
    im = obj.shift(
        dx=shift[0], dy=shift[1]
    ).drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs,
        method='no_pixel').array
    noise = np.sqrt(np.sum(im**2)) / 1e3
    wgt = np.ones_like(im) / noise**2

    g1arr = []
    g2arr = []
    Tarr = []
    farr = []

    fitter = PrePSFGaussMom(fwhm=1.2, pad_factor=pad_factor)

    # get true flux
    jac = Jacobian(
        y=cen + xy.y, x=cen + xy.x,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)
    obs = Observation(
        image=im,
        jacobian=jac,
    )
    res = fitter.go(obs=obs)
    flux_true = res["flux"]
    T_true = res["T"]
    g1_true = res["e"][0]
    g2_true = res["e"][1]

    for _ in range(1000):
        _im = im + (rng.normal(size=im.shape) * noise)
        obs = Observation(
            image=_im,
            weight=wgt,
            jacobian=jac,
        )

        # use a huge weight so that we get the raw moments back out
        res = fitter.go(obs=obs)
        if res['flags'] == 0:
            _g1, _g2 = res['e'][0], res['e'][1]
            g1arr.append(_g1)
            g2arr.append(_g2)
            Tarr.append(res['T'])
            farr.append(res['flux'])

    print("\n")
    _report_info("flux", farr, flux_true, res["flux_err"])
    _report_info("T", Tarr, T_true, res["T_err"])
    _report_info("g1", g1arr, g1_true, res["e_err"][0])
    _report_info("g2", g2arr, g2_true, res["e_err"][1])

    assert np.allclose(np.mean(farr), flux_true, atol=0, rtol=1e-3)
    assert np.allclose(np.mean(Tarr), T_true, atol=0, rtol=1e-3)
    assert np.allclose(np.mean(g1arr), g1_true, atol=0, rtol=1e-2)
    assert np.allclose(np.mean(g2arr), g2_true, atol=0, rtol=1e-2)

    etol = 0.1
    assert np.allclose(np.std(farr), res["flux_err"], atol=0, rtol=etol)
    assert np.allclose(np.std(Tarr), res["T_err"], atol=0, rtol=etol*3)
    assert np.allclose(np.std(g1arr), res["e_err"][0], atol=0, rtol=etol*3)
    assert np.allclose(np.std(g2arr), res["e_err"][1], atol=0, rtol=etol*3)


@pytest.mark.parametrize('snr', [1e1, 1e3])
@pytest.mark.parametrize('pixel_scale', [0.125, 0.25])
@pytest.mark.parametrize('fwhm,psf_fwhm', [(0.6, 0.9), (1.5, 0.9)])
@pytest.mark.parametrize('mom_fwhm', [1.2, 1.5, 2.0])
@pytest.mark.parametrize('image_size', [57, 58])
@pytest.mark.parametrize('pad_factor', [2, 1, 1.5])
@pytest.mark.parametrize('psf_trunc_fac', [1e-3, 1e-5])
def test_prepsfmom_psf(
    pad_factor, image_size, fwhm, psf_fwhm, pixel_scale, snr, psf_trunc_fac,
    mom_fwhm,
):
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
        y=cen + psf_xy.y, x=cen + psf_xy.x,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

    gal = galsim.Gaussian(
        fwhm=fwhm
    ).shear(
        g1=-0.1, g2=0.1
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
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs
    ).array

    g1arr = []
    g2arr = []
    Tarr = []
    farr = []
    momarr = []
    snrarr = []
    fitter = PrePSFGaussMom(
        fwhm=mom_fwhm,
        pad_factor=pad_factor,
        psf_trunc_fac=psf_trunc_fac,
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
    res = fitter.go(obs=obs)
    flux_true = res["flux"]
    T_true = res["T"]
    g1_true = res["e"][0]
    g2_true = res["e"][1]

    for _ in range(100):
        _im = im + (rng.normal(size=im.shape) * noise)
        obs = Observation(
            image=_im,
            weight=wgt,
            jacobian=jac,
            psf=Observation(image=psf_im, jacobian=psf_jac),
        )

        res = fitter.go(obs=obs)
        if res['flags'] == 0:
            _g1, _g2 = res['e'][0], res['e'][1]
            g1arr.append(_g1)
            g2arr.append(_g2)
            Tarr.append(res['T'])
            farr.append(res['flux'])
            snrarr.append(res["flux"] / res["flux_err"])
            momarr.append(res["mom"])

    print("\n")
    _report_info("snr", snrarr, None, None)
    _report_info("flux", farr, flux_true, res["flux_err"])
    _report_info("T", Tarr, T_true, res["T_err"])
    _report_info("g1", g1arr, g1_true, res["e_err"][0])
    _report_info("g2", g2arr, g2_true, res["e_err"][1])
    mom_cov = np.cov(np.array(momarr).T)
    print("mom cov ratio:\n", res["mom_cov"]/mom_cov, flush=True)
    assert np.allclose(np.mean(farr), flux_true, atol=0, rtol=0.1)
    assert np.allclose(np.std(farr), res["flux_err"], atol=0, rtol=0.2)

    # at low SNR we get a lot of problems with division by noisy things
    if snr > 100:
        assert np.allclose(np.mean(Tarr), T_true, atol=0, rtol=0.1)
        assert np.allclose(np.mean(g1arr), g1_true, atol=0, rtol=0.2)
        assert np.allclose(np.mean(g2arr), g2_true, atol=0, rtol=0.2)
