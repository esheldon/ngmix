import galsim
import numpy as np
import pytest

import ngmix
from ngmix.fitting import LM
from ngmix import Jacobian
from ngmix import Observation
from ngmix.moments import fwhm_to_T


@pytest.mark.parametrize('wcs_g1', [-0.5, 0, 0.2])
@pytest.mark.parametrize('wcs_g2', [-0.2, 0, 0.5])
@pytest.mark.parametrize('g1_true', [-0.1, 0, 0.2])
@pytest.mark.parametrize('g2_true', [-0.2, 0, 0.1])
def test_ml_max_fitting_gauss_smoke(g1_true, g2_true, wcs_g1, wcs_g2):
    rng = np.random.RandomState(seed=42)

    # allow some small relative bias due to approximate exp function
    tol = 1.0e-5

    flux = 400  # in galsim units
    image_size = 33
    cen = (image_size - 1)/2
    gs_wcs = galsim.ShearWCS(
        0.25, galsim.Shear(g1=wcs_g1, g2=wcs_g2)).jacobian()
    scale = np.sqrt(gs_wcs.pixelArea())

    g_prior = ngmix.priors.GPriorBA(sigma=0.2, rng=rng)
    cen_prior = ngmix.priors.CenPrior(
        cen1=0, cen2=0, sigma1=scale, sigma2=scale, rng=rng,
    )
    T_prior = ngmix.priors.FlatPrior(minval=0.1, maxval=2, rng=rng)
    F_prior = ngmix.priors.FlatPrior(minval=1e-4, maxval=1e9, rng=rng)
    prior = ngmix.joint_prior.PriorSimpleSep(
        cen_prior=cen_prior,
        g_prior=g_prior,
        T_prior=T_prior,
        F_prior=F_prior,
    )

    obj = galsim.Gaussian(
        fwhm=0.9
    ).shear(
        g1=g1_true, g2=g2_true
    ).withFlux(
        flux,
    )
    im = obj.drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs,
        method='no_pixel').array

    noise = np.sqrt(np.sum(im**2)) / 1.e6
    wgt = np.ones_like(im) / noise**2

    g1arr = []
    g2arr = []
    Tarr = []
    farr = []
    xarr = []
    yarr = []
    for _ in range(10):
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
            y=cen + xy.y,
            x=cen + xy.x,
            dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
            dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy,
        )

        _im = im + rng.normal(size=im.shape, scale=noise)
        obs = Observation(
            image=_im,
            weight=wgt,
            jacobian=jac,
        )

        prior = None
        fitter = LM(model='gauss', prior=prior)

        guess = np.zeros(6)
        guess[0:2] = rng.uniform(low=-0.1*scale, high=0.1*scale, size=2)
        guess[2] = g1_true + rng.uniform(low=-0.01, high=0.01)
        guess[3] = g2_true + rng.uniform(low=-0.01, high=0.01)
        guess[4] = fwhm_to_T(0.9) * rng.uniform(low=0.99, high=1.01)
        guess[5] = flux * scale**2 * rng.uniform(low=0.99, high=1.01)

        # guess = guess + rng.normal(size=6) * 0.01
        fitter.go(obs=obs, guess=guess)
        res = fitter.get_result()

        if res['flags'] == 0:
            _g1, _g2, _T = res['g'][0], res['g'][1], res['pars'][4]
            g1arr.append(_g1)
            g2arr.append(_g2)
            Tarr.append(_T)
            farr.append(res['pars'][5])
            xarr.append(res['pars'][1])
            yarr.append(res['pars'][0])

    assert len(g1arr) > 0
    g1 = np.mean(g1arr)
    g2 = np.mean(g2arr)
    assert np.abs(g1 - g1_true) < tol
    assert np.abs(g2 - g2_true) < tol

    if g1_true == 0 and g2_true == 0:
        T = np.mean(Tarr)
        Ttrue = fwhm_to_T(0.9)
        assert T/Ttrue-1 < tol

    fmn = np.mean(farr)/scale/scale

    assert np.abs(fmn/flux-1) < tol

    xerr = np.std(xarr) / np.sqrt(len(xarr))
    assert np.abs(np.mean(xarr)) < xerr * 5
    yerr = np.std(yarr) / np.sqrt(len(yarr))
    assert np.abs(np.mean(yarr)) < yerr * 5
