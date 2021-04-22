import galsim
import numpy as np
import pytest

from ngmix.prepsfmom import PrePSFMom
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

    fitter = PrePSFMom(fwhm=fwhm * weight_fac, pad_factor=1.2)

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


@pytest.mark.parametrize('image_size', [107, 112])
@pytest.mark.parametrize('pad_factor', [3.1, 1.3234, 1])
def test_prepsfmom_error(pad_factor, image_size):
    rng = np.random.RandomState(seed=100)

    fwhm = 0.9
    cen = (image_size - 1)/2
    gs_wcs = galsim.ShearWCS(
        0.125, galsim.Shear(g1=-0.1, g2=0.23)).jacobian()

    obj = galsim.Gaussian(
        fwhm=fwhm
    ).shear(
        g1=-0.1, g2=0.1
    ).withFlux(
        400)
    im = obj.drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs,
        method='no_pixel').array
    noise = np.sqrt(np.sum(im**2)) / 1e3
    wgt = np.ones_like(im) / noise**2
    scale = np.sqrt(gs_wcs.pixelArea())

    g1arr = []
    g2arr = []
    Tarr = []
    farr = []
    momarr = []

    fitter = PrePSFMom(fwhm=fwhm, pad_factor=pad_factor)

    # get true flux
    jac = Jacobian(
        y=cen, x=cen,
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

    for _ in range(100):
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
            jacobian=jac,
        )

        # use a huge weight so that we get the raw moments back out
        res = fitter.go(obs=obs, return_kernels=True)
        if res['flags'] == 0:
            _g1, _g2 = res['e'][0], res['e'][1]
            g1arr.append(_g1)
            g2arr.append(_g2)
            Tarr.append(res['T'])
            farr.append(res['flux'])
            momarr.append(res["mom"])

    etol = 0.1
    assert np.allclose(np.std(farr), res["flux_err"], atol=0, rtol=etol)
    assert np.allclose(np.std(Tarr), res["T_err"], atol=0, rtol=etol*3)
    assert np.allclose(np.std(g1arr), res["e_err"][0], atol=0, rtol=etol*3)
    assert np.allclose(np.std(g2arr), res["e_err"][1], atol=0, rtol=etol*3)

    assert np.allclose(np.mean(farr), flux_true, atol=0, rtol=1e-3)
    assert np.allclose(np.mean(Tarr), T_true, atol=0, rtol=1e-3)
    assert np.allclose(np.mean(g1arr), g1_true, atol=0, rtol=1e-2)
    assert np.allclose(np.mean(g2arr), g2_true, atol=0, rtol=1e-2)


@pytest.mark.parametrize('image_size', [107, 112])
@pytest.mark.parametrize('pad_factor', [3.34123, 1.3234, 1])
def test_prepsfmom_psf(pad_factor, image_size):
    rng = np.random.RandomState(seed=100)

    fwhm = 0.9
    psf_fwhm = 0.6
    cen = (image_size - 1)/2
    gs_wcs = galsim.ShearWCS(
        0.125, galsim.Shear(g1=-0, g2=0)).jacobian()

    gal = galsim.Gaussian(
        fwhm=fwhm
    ).shear(
        g1=-0.1, g2=0.1
    ).withFlux(
        400)
    psf = galsim.Gaussian(
        fwhm=psf_fwhm
    ).shear(
        g1=0.3, g2=-0.15
    )
    im = galsim.Convolve([gal, psf]).drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs).array
    noise = np.sqrt(np.sum(im**2)) / 1e3
    wgt = np.ones_like(im) / noise**2
    scale = np.sqrt(gs_wcs.pixelArea())

    psf_im = psf.drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs).array

    g1arr = []
    g2arr = []
    Tarr = []
    farr = []
    momarr = []

    fitter = PrePSFMom(fwhm=fwhm, pad_factor=pad_factor)

    # get true flux
    im = gal.drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs,
        method='no_pixel').array
    jac = Jacobian(
        y=cen, x=cen,
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

    for _ in range(100):
        shift = rng.uniform(low=-scale/2, high=scale/2, size=2)
        xy = gs_wcs.toImage(galsim.PositionD(shift))

        im = galsim.Convolve([gal, psf]).shift(
            dx=shift[0], dy=shift[1]
        ).drawImage(
            nx=image_size,
            ny=image_size,
            wcs=gs_wcs,
            dtype=np.float64).array

        psf_im = psf.shift(
            dx=shift[0]*0, dy=shift[1]*0
        ).drawImage(
            nx=image_size,
            ny=image_size,
            wcs=gs_wcs).array

        _jac = Jacobian(
            y=cen + xy.y, x=cen + xy.x,
            dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
            dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

        _im = im + (rng.normal(size=im.shape) * noise)
        obs = Observation(
            image=_im,
            weight=wgt,
            jacobian=_jac,
            psf=Observation(image=psf_im, jacobian=jac),
        )

        # use a huge weight so that we get the raw moments back out
        res = fitter.go(obs=obs, return_kernels=True)
        if res['flags'] == 0:
            _g1, _g2 = res['e'][0], res['e'][1]
            g1arr.append(_g1)
            g2arr.append(_g2)
            Tarr.append(res['T'])
            farr.append(res['flux'])
            momarr.append(res["mom"])

    assert np.allclose(np.mean(farr), flux_true, atol=0, rtol=1e-3)
    assert np.allclose(np.mean(Tarr), T_true, atol=0, rtol=1e-3)
    assert np.allclose(np.mean(g1arr), g1_true, atol=0, rtol=1e-2)
    assert np.allclose(np.mean(g2arr), g2_true, atol=0, rtol=1e-2)

    # deconvolving the PSF correlates the noise which makes the errors wrong
    etol = 0.1
    assert np.allclose(np.std(farr), res["flux_err"], atol=0, rtol=etol)
    assert np.allclose(np.std(Tarr), res["T_err"], atol=0, rtol=etol*3)
    assert np.allclose(np.std(g1arr), res["e_err"][0], atol=0, rtol=etol*3)
    assert np.allclose(np.std(g2arr), res["e_err"][1], atol=0, rtol=etol*3)
