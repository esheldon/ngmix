import galsim
import numpy as np
import pytest

import ngmix
from ngmix.fitting import LMSimple
from ngmix import Jacobian
from ngmix import Observation
from ngmix.moments import fwhm_to_T

GTOL = 1e-4


@pytest.mark.parametrize('wcs_g1', [-0.5, 0, 0.2])
@pytest.mark.parametrize('wcs_g2', [-0.2, 0, 0.5])
@pytest.mark.parametrize('g1_true', [-0.1, 0, 0.2])
@pytest.mark.parametrize('g2_true', [-0.2, 0, 0.1])
def test_ml_fitting_gauss_smoke(g1_true, g2_true, wcs_g1, wcs_g2):
    rng = np.random.RandomState(seed=10)

    image_size = 33
    cen = (image_size - 1)/2
    gs_wcs = galsim.ShearWCS(
        0.25, galsim.Shear(g1=wcs_g1, g2=wcs_g2)).jacobian()
    scale = np.sqrt(gs_wcs.pixelArea())

    g_prior = ngmix.priors.GPriorBA(0.5)
    cen_prior = ngmix.priors.CenPrior(0, 0, scale, scale)
    T_prior = ngmix.priors.FlatPrior(0.1, 2)
    F_prior = ngmix.priors.FlatPrior(1e-4, 1e9)
    prior = ngmix.joint_prior.PriorSimpleSep(
        cen_prior,
        g_prior,
        T_prior,
        F_prior)

    obj = galsim.Gaussian(
        fwhm=0.9
    ).shear(
        g1=g1_true, g2=g2_true
    ).withFlux(
        400)
    im = obj.drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs,
        method='no_pixel').array
    noise = np.sqrt(np.sum(im**2)) / 1e16
    wgt = np.ones_like(im) / noise**2

    guess = np.ones(6) * 0.1
    guess[0] = 0
    guess[1] = 0
    guess[2] = g1_true
    guess[3] = g2_true
    guess[4] = fwhm_to_T(0.9)
    guess[5] = 400

    g1arr = []
    g2arr = []
    Tarr = []
    for _ in range(50):
        shift = rng.uniform(low=-scale/2, high=scale/2, size=2)
        xy = gs_wcs.toImage(galsim.PositionD(shift))

        im = obj.shift(
            dx=shift[0], dy=shift[1]
        ).drawImage(
            nx=image_size,
            ny=image_size,
            wcs=gs_wcs,
            method='no_pixel').array

        jac = Jacobian(
            y=cen + xy.y, x=cen + xy.x,
            dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
            dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

        _im = im + (rng.normal(size=im.shape) * noise)
        obs = Observation(
            image=_im,
            weight=wgt,
            jacobian=jac)
        fitter = LMSimple(obs, 'gauss', prior=prior)
        fitter.go(guess + rng.normal(size=6) * 0.01)
        res = fitter.get_result()
        if res['flags'] == 0:
            _g1, _g2, _T = res['g'][0], res['g'][1], res['pars'][4]
            g1arr.append(_g1)
            g2arr.append(_g2)
            Tarr.append(_T)

    g1 = np.mean(g1arr)
    g2 = np.mean(g2arr)
    assert np.abs(g1 - g1_true) < GTOL
    assert np.abs(g2 - g2_true) < GTOL

    if g1 == 0 and g2 == 0:
        T = np.mean(Tarr)
        T_err = np.std(Tarr) / np.sqrt(len(Tarr))
        assert np.abs(T - fwhm_to_T(0.9)) < T_err * 5
