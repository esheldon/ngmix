"""
just test moment errors
"""
import pytest
import numpy as np
import ngmix


def _get_obs(rng):
    import galsim

    noise = 0.001
    psf_noise = 1.0e-6

    scale = 0.263

    psf_fwhm = 0.9
    gal_fwhm = 0.7

    psf = galsim.Gaussian(fwhm=psf_fwhm)
    obj0 = galsim.Gaussian(fwhm=gal_fwhm)

    obj = galsim.Convolve(psf, obj0)

    psf_im = psf.drawImage(scale=scale).array
    im = obj.drawImage(scale=scale).array

    psf_im += rng.normal(scale=psf_noise, size=psf_im.shape)
    im += rng.normal(scale=noise, size=im.shape)

    cen = (np.array(im.shape)-1.0)/2.0
    psf_cen = (np.array(psf_im.shape)-1.0)/2.0

    j = ngmix.DiagonalJacobian(row=cen[0], col=cen[1], scale=scale)
    pj = ngmix.DiagonalJacobian(row=psf_cen[0], col=psf_cen[1], scale=scale)

    wt = im*0 + 1.0/noise**2
    psf_wt = psf_im*0 + 1.0/psf_noise**2

    psf_obs = ngmix.Observation(
        psf_im,
        weight=psf_wt,
        jacobian=pj,
    )

    obs = ngmix.Observation(
        im,
        weight=wt,
        jacobian=j,
        psf=psf_obs,
    )

    return obs


@pytest.mark.parametrize('models', [
    ('gauss', 'gauss'),
    ('gauss', 'exp'),
    ('em1', 'exp'),
    ('em3', 'exp'),
])
def test_bootstrap_max_smoke(models):

    psf_model, obj_model = models

    rng = np.random.RandomState(3421)
    obs = _get_obs(rng)

    boot = ngmix.bootstrap.Bootstrapper(obs)

    psf_Tguess = 0.9*0.263**2
    boot.fit_psfs(
        psf_model,
        psf_Tguess,
    )

    pars = {
        'method': 'lm',
        'lm_pars': {},
    }
    boot.fit_max(
        obj_model,
        pars,
    )

    res = boot.get_fitter().get_result()
    assert res['flags'] == 0


@pytest.mark.parametrize('fixnoise', [True, False])
def test_bootstrap_mcal_smoke(fixnoise):

    psf_model = 'gauss'
    obj_model = 'gauss'

    rng = np.random.RandomState(481)
    obs = _get_obs(rng)

    mcal_boot = ngmix.bootstrap.MaxMetacalBootstrapper(obs)

    psf_Tguess = 0.9*0.263**2
    pars = {
        'method': 'lm',
        'lm_pars': {},
    }

    types = ['noshear', '1p', '1m', '2p', '2m']
    mcal_boot.fit_metacal(
        psf_model,
        obj_model,
        pars,
        psf_Tguess,
        metacal_pars={
            'psf': 'fitgauss',
            'types': types,
            'fixnoise': fixnoise,
        },
    )

    res = mcal_boot.get_metacal_result()
    for type in types:
        assert type in res

        assert res[type]['flags'] == 0
