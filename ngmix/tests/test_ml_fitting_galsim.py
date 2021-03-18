import galsim
import numpy as np
import pytest

import ngmix
from ._galsim_sims import _get_obs
from ._priors import get_prior_galsimfit


@pytest.mark.parametrize('wcs_g1', [-0.03, 0.0, 0.03])
@pytest.mark.parametrize('wcs_g2', [-0.03, 0.0, 0.03])
@pytest.mark.parametrize('model', ['exp', 'dev', 'gauss'])
@pytest.mark.parametrize('use_prior', [True, False])
def test_ml_fitting_galsim(wcs_g1, wcs_g2, model, use_prior):

    rng = np.random.RandomState(seed=2312)
    scale = 0.263
    prior = get_prior_galsimfit(model=model, rng=rng, scale=scale)

    psf_fwhm = 0.9
    psf_image_size = 33
    image_size = 51

    noise = 0.001

    hlr = 0.1
    flux = 400
    gs_wcs = galsim.ShearWCS(
        scale,
        galsim.Shear(g1=wcs_g1, g2=wcs_g2),
    ).jacobian()

    psf_im = galsim.Gaussian(fwhm=psf_fwhm).drawImage(
        nx=psf_image_size,
        ny=psf_image_size,
        wcs=gs_wcs,
    ).array

    psf_cen = (psf_image_size - 1.0)/2.0
    psf_jac = ngmix.Jacobian(
        y=psf_cen, x=psf_cen,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy,
    )

    psf_obs = ngmix.Observation(
        image=psf_im,
        jacobian=psf_jac,
    )

    guess = prior.sample()
    g1arr = []
    g2arr = []
    farr = []
    xarr = []
    yarr = []

    if use_prior:
        send_prior = prior
    else:
        send_prior = None

    fitter = ngmix.galsimfit.GalsimLM(model=model, prior=send_prior)

    if model == 'exp' and use_prior:
        ntrial = 50
    else:
        ntrial = 1

    for _ in range(ntrial):
        shift = rng.uniform(low=-scale/2, high=scale/2, size=2)
        xy = gs_wcs.toImage(galsim.PositionD(shift))

        g1_true, g2_true = prior.g_prior.sample2d()
        gal = galsim.Exponential(
            half_light_radius=hlr,
        ).shear(
            g1=g1_true, g2=g2_true
        ).withFlux(
            flux,
        )

        obj = galsim.Convolve([gal, galsim.Gaussian(fwhm=psf_fwhm)])

        im = obj.shift(
            dx=shift[0], dy=shift[1]
        ).drawImage(
            nx=image_size,
            ny=image_size,
            wcs=gs_wcs,
            dtype=np.float64,
        ).array
        wgt = np.ones_like(im) / noise**2

        cen = (np.array(im.shape)-1)/2
        jac = ngmix.Jacobian(
            y=cen[0] + xy.y, x=cen[1] + xy.x,
            dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
            dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy,
        )

        _im = im + (rng.normal(size=im.shape) * noise)
        obs = ngmix.Observation(
            image=_im,
            weight=wgt,
            jacobian=jac,
            psf=psf_obs,
        )

        guess[0] = rng.uniform(low=-0.1, high=0.1)
        guess[1] = rng.uniform(low=-0.1, high=0.1)
        guess[2] = rng.uniform(low=-0.1, high=0.1)
        guess[3] = rng.uniform(low=-0.1, high=0.1)
        guess[4] = hlr * rng.uniform(low=0.9, high=1.1)
        guess[5] = flux * rng.uniform(low=0.9, high=1.1)

        res = fitter.go(obs=obs, guess=guess + rng.normal(size=guess.size) * 0.01)
        if res['flags'] == 0:
            _g1, _g2, _ = res['g'][0], res['g'][1], res['pars'][4]
            g1arr.append(_g1 - g1_true)
            g2arr.append(_g2 - g2_true)
            farr.append(res['pars'][5])
            xarr.append(res['pars'][1])
            yarr.append(res['pars'][0])

    if model == 'exp' and use_prior:
        g1diff = np.mean(g1arr)
        g2diff = np.mean(g2arr)

        print('g1vals')
        print(g1arr)
        print(g1diff)
        print('g2vals')
        print(g2arr)
        print(g2diff)

        gtol = 1.0e-5

        assert np.abs(g1diff) < gtol
        assert np.abs(g2diff) < gtol

        xerr = np.std(xarr) / np.sqrt(len(xarr))
        assert np.abs(np.mean(xarr)) < xerr * 5
        yerr = np.std(yarr) / np.sqrt(len(yarr))
        assert np.abs(np.mean(yarr)) < yerr * 5


def test_ml_fitting_galsim_spergel_smoke():

    rng = np.random.RandomState(seed=2312)
    scale = 0.263
    prior = get_prior_galsimfit(model='spergel', rng=rng, scale=scale)

    psf_fwhm = 0.9
    psf_image_size = 33
    image_size = 51

    noise = 0.001

    hlr = 0.1
    flux = 400

    gs_wcs = galsim.ShearWCS(
        scale,
        galsim.Shear(g1=0.01, g2=-0.01),
    ).jacobian()

    psf_im = galsim.Gaussian(fwhm=psf_fwhm).drawImage(
        nx=psf_image_size,
        ny=psf_image_size,
        wcs=gs_wcs,
    ).array

    psf_cen = (psf_image_size - 1.0)/2.0
    psf_jac = ngmix.Jacobian(
        y=psf_cen, x=psf_cen,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy,
    )

    psf_obs = ngmix.Observation(
        image=psf_im,
        jacobian=psf_jac,
    )

    guess = prior.sample()

    fitter = ngmix.galsimfit.GalsimLMSpergel(prior=prior)

    shift = rng.uniform(low=-scale/2, high=scale/2, size=2)
    xy = gs_wcs.toImage(galsim.PositionD(shift))

    g1_true, g2_true = prior.g_prior.sample2d()
    gal = galsim.Exponential(
        half_light_radius=hlr,
    ).shear(
        g1=g1_true, g2=g2_true
    ).withFlux(
        flux,
    )

    obj = galsim.Convolve([gal, galsim.Gaussian(fwhm=psf_fwhm)])

    im = obj.shift(
        dx=shift[0], dy=shift[1]
    ).drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs,
        dtype=np.float64,
    ).array
    wgt = np.ones_like(im) / noise**2

    cen = (np.array(im.shape)-1)/2
    jac = ngmix.Jacobian(
        y=cen[0] + xy.y, x=cen[1] + xy.x,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy,
    )

    _im = im + (rng.normal(size=im.shape) * noise)
    obs = ngmix.Observation(
        image=_im,
        weight=wgt,
        jacobian=jac,
        psf=psf_obs,
    )

    guess[0] = rng.uniform(low=-0.1, high=0.1)
    guess[1] = rng.uniform(low=-0.1, high=0.1)
    guess[2] = rng.uniform(low=-0.1, high=0.1)
    guess[3] = rng.uniform(low=-0.1, high=0.1)
    guess[4] = hlr * rng.uniform(low=0.9, high=1.1)
    guess[5] = rng.uniform(low=0, high=2)
    guess[6] = flux * rng.uniform(low=0.9, high=1.1)

    res = fitter.go(obs=obs, guess=guess + rng.normal(size=guess.size) * 0.01)
    assert res['flags'] == 0


def test_ml_fitting_galsim_moffat_smoke():

    rng = np.random.RandomState(seed=2312)

    scale = 0.263
    fwhm = 0.9
    beta = 2.5
    image_size = 33
    flux = 1.0

    noise = 1.0e-5

    moff = galsim.Moffat(fwhm=fwhm, beta=beta, flux=flux)
    im = moff.drawImage(
        nx=image_size,
        ny=image_size,
        scale=scale,
    ).array
    im += rng.normal(scale=noise, size=im.shape)
    weight = im*0 + 1.0/noise**2

    cen = (image_size - 1.0)/2.0
    jac = ngmix.DiagonalJacobian(
        y=cen, x=cen,
        scale=scale,
    )

    obs = ngmix.Observation(
        image=im,
        weight=weight,
        jacobian=jac,
    )

    fitter = ngmix.galsimfit.GalsimLMMoffat()

    guess = np.zeros(7)

    guess[0] = rng.uniform(low=-0.1, high=0.1)
    guess[1] = rng.uniform(low=-0.1, high=0.1)
    guess[2] = rng.uniform(low=-0.1, high=0.1)
    guess[3] = rng.uniform(low=-0.1, high=0.1)
    guess[4] = moff.half_light_radius * rng.uniform(low=0.9, high=1.1)
    guess[5] = rng.uniform(low=1.5, high=3)
    guess[6] = flux * rng.uniform(low=0.9, high=1.1)

    res = fitter.go(obs=obs, guess=guess)
    assert res['flags'] == 0


def test_ml_fitting_galsim_errors():
    rng = np.random.RandomState(seed=8)

    fitter = ngmix.galsimfit.GalsimLM(model='exp')
    with pytest.raises(ValueError):
        fitter.go(obs=None, guess=None)

    obs = _get_obs(rng)
    with pytest.raises(ngmix.GMixRangeError):
        fitter.go(obs=obs, guess=[0, 0, 0, 0, 0, 0])

    guess = [0, 0, 0, 0, 1, 1]
    gmod = ngmix.galsimfit.GalsimLMFitModel(obs=obs, model='exp', guess=guess)
    with pytest.raises(ngmix.GMixRangeError):
        gmod.make_model([0, 0, 1.e9, 0, 1, 1])

    with pytest.raises(ngmix.GMixRangeError):
        gmod.make_round_model([0, 0, 0, 0, -1, 1])

    with pytest.raises(NotImplementedError):
        ngmix.galsimfit.GalsimLMFitModel(obs=obs, model='blah', guess=guess)

    with pytest.raises(ValueError):
        fitter.go(obs=obs, guess=np.zeros(1000))
