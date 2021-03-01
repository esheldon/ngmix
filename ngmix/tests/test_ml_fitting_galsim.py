import galsim
import numpy as np
import pytest

import ngmix
from ngmix.galsimfit import GalsimLM
from ngmix import Jacobian
from ngmix import Observation


def get_prior(*, rng, scale):
    g_prior = ngmix.priors.GPriorBA(sigma=0.1, rng=rng)
    cen_prior = ngmix.priors.CenPrior(
        cen1=0, cen2=0, sigma1=scale, sigma2=scale, rng=rng,
    )
    r50_prior = ngmix.priors.FlatPrior(minval=0.01, maxval=10, rng=rng)
    F_prior = ngmix.priors.FlatPrior(minval=1e-4, maxval=1e9, rng=rng)

    prior = ngmix.joint_prior.PriorGalsimSimpleSep(
        cen_prior=cen_prior,
        g_prior=g_prior,
        r50_prior=r50_prior,
        F_prior=F_prior,
    )

    return prior


@pytest.mark.parametrize('wcs_g1', [-0.03, 0.0, 0.03])
@pytest.mark.parametrize('wcs_g2', [-0.03, 0.0, 0.03])
@pytest.mark.parametrize('model', ['exp'])
def test_ml_fitting_galsim(wcs_g1, wcs_g2, model):

    rng = np.random.RandomState(seed=2312)
    scale = 0.263
    prior = get_prior(rng=rng, scale=scale)

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
    psf_jac = Jacobian(
        y=psf_cen, x=psf_cen,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy,
    )

    psf_obs = Observation(
        image=psf_im,
        jacobian=psf_jac,
    )

    guess = prior.sample()
    g1arr = []
    g2arr = []
    farr = []
    xarr = []
    yarr = []

    fitter = GalsimLM(model='exp', prior=prior)

    for _ in range(50):
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
        jac = Jacobian(
            y=cen[0] + xy.y, x=cen[1] + xy.x,
            dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
            dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy,
        )

        _im = im + (rng.normal(size=im.shape) * noise)
        obs = Observation(
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

        fitter.go(obs=obs, guess=guess + rng.normal(size=guess.size) * 0.01)
        res = fitter.get_result()
        if res['flags'] == 0:
            _g1, _g2, _ = res['g'][0], res['g'][1], res['pars'][4]
            g1arr.append(_g1 - g1_true)
            g2arr.append(_g2 - g2_true)
            farr.append(res['pars'][5])
            xarr.append(res['pars'][1])
            yarr.append(res['pars'][0])

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
