import galsim
import numpy as np
import pytest

import ngmix
from ngmix.fitting import LMSimple
from ngmix import Jacobian
from ngmix import Observation
from ngmix.moments import fwhm_to_T

GTOL = 3e-4


@pytest.mark.parametrize('s2n', [1e2, 1e16])
@pytest.mark.parametrize('wcs_g1', [-0.5, 0, 0.2])
@pytest.mark.parametrize('wcs_g2', [-0.2, 0, 0.5])
@pytest.mark.parametrize('g1_true', [-0.1, 0, 0.2])
@pytest.mark.parametrize('g2_true', [-0.2, 0, 0.1])
def test_ml_fitting_exp_obj_gauss_psf_smoke(
        g1_true, g2_true, wcs_g1, wcs_g2, s2n):
    jc = galsim.ShearWCS(
        0.25, galsim.Shear(g1=wcs_g1, g2=wcs_g2)).jacobian()
    jac = Jacobian(
        y=16, x=16,
        dudx=jc.dudx, dudy=jc.dudy, dvdx=jc.dvdx, dvdy=jc.dvdy)
    rng = np.random.RandomState(seed=10)

    g_prior = ngmix.priors.GPriorBA(0.5)
    cen_prior = ngmix.priors.CenPrior(0, 0, jac.scale, jac.scale)
    T_prior = ngmix.priors.FlatPrior(0.1, 2)
    F_prior = ngmix.priors.FlatPrior(1e-4, 1e9)
    prior = ngmix.joint_prior.PriorSimpleSep(
        cen_prior,
        g_prior,
        T_prior,
        F_prior)

    gs_wcs = jac.get_galsim_wcs()
    gal = galsim.Exponential(
        half_light_radius=0.5
    ).shear(
        g1=g1_true, g2=g2_true
    ).withFlux(
        400)
    obj = galsim.Convolve([gal, galsim.Gaussian(fwhm=0.5)])

    psf_im = galsim.Gaussian(fwhm=0.5).drawImage(
        nx=33, ny=33, wcs=gs_wcs, method='no_pixel').array
    psf_gmix = ngmix.gmix.make_gmix_model(
        [0, 0, 0, 0, fwhm_to_T(0.5), 1], "gauss")
    psf_obs = Observation(
        image=psf_im,
        gmix=psf_gmix
    )

    im = obj.drawImage(
        nx=33,
        ny=33,
        wcs=gs_wcs,
        method='no_pixel').array

    noise = np.sqrt(np.sum(im**2))/s2n

    wgt = np.ones_like(im) / noise**2

    guess = np.ones(6) * 0.1
    guess[0] = 0
    guess[1] = 0
    guess[2] = g1_true
    guess[3] = g2_true
    guess[4] = 2
    guess[5] = 400

    g1arr = []
    g2arr = []
    Tarr = []
    for _ in range(100):
        _im = im + (rng.normal(size=im.shape) * noise)
        obs = Observation(
            image=_im,
            weight=wgt,
            jacobian=jac,
            psf=psf_obs)
        fitter = LMSimple(obs, 'exp', prior=prior)
        fitter.go(guess + rng.normal(size=6) * 0.01)
        res = fitter.get_result()
        if res['flags'] == 0:
            _g1, _g2, _T = res['g'][0], res['g'][1], res['pars'][4]
            g1arr.append(_g1)
            g2arr.append(_g2)
            Tarr.append(_T)

    g1 = np.mean(g1arr)
    g1_err = np.std(g1arr) / np.sqrt(len(g1arr))
    g2 = np.mean(g2arr)
    g2_err = np.std(g2arr) / np.sqrt(len(g2arr))
    assert np.abs(g1 - g1_true) < max(GTOL, g1_err * 5)
    assert np.abs(g2 - g2_true) < max(GTOL, g2_err * 5)

    if g1 == 0 and g2 == 0:
        T = np.mean(Tarr)
        T_err = np.std(Tarr) / np.sqrt(len(Tarr))
        assert np.abs(T - fwhm_to_T(0.9)) < T_err * 5
