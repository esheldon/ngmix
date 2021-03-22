import pytest
import numpy as np
import ngmix
from ngmix.runners import Runner, PSFRunner
from ngmix.guessers import (
    GMixPSFGuesser, TFluxGuesser, TPSFFluxGuesser, CoellipPSFGuesser,
)
from ngmix.fitting import CoellipFitter
from ngmix.em import GMixEM
from ngmix.fitting import Fitter
from ._sims import get_model_obs

FRAC_TOL = 5.0e-4


@pytest.mark.parametrize('psf_model_type', ['em', 'coellip'])
@pytest.mark.parametrize('model', ['exp', 'dev'])
def test_runner_lm_simple_smoke(model, psf_model_type):
    """
    Smoke test a Runner running the LM fitter
    """

    rng = np.random.RandomState(17710)

    data = get_model_obs(
        rng=rng,
        model=model,
        noise=1.0e-4,
    )

    psf_ngauss = 3
    if psf_model_type == 'em':
        psf_guesser = GMixPSFGuesser(
            rng=rng,
            ngauss=psf_ngauss,
        )

        psf_fitter = GMixEM(tol=1.0e-5)
    else:
        psf_guesser = CoellipPSFGuesser(
            rng=rng,
            ngauss=psf_ngauss,
        )

        psf_fitter = CoellipFitter(ngauss=psf_ngauss)

    psf_runner = PSFRunner(
        fitter=psf_fitter,
        guesser=psf_guesser,
        ntry=2,
    )
    psf_runner.go(obs=data['obs'])

    guesser = TFluxGuesser(
        rng=rng,
        T=0.25,
        flux=100.0,
    )
    fitter = Fitter(model=model)

    runner = Runner(
        fitter=fitter,
        guesser=guesser,
        ntry=2,
    )
    res = runner.go(obs=data['obs'])
    assert res['flags'] == 0


@pytest.mark.parametrize('guesser_type', ['TF', 'TPSFFlux'])
@pytest.mark.parametrize('psf_model_type', ['coellip', 'em'])
@pytest.mark.parametrize('model', ['exp', 'dev'])
@pytest.mark.parametrize('noise', [1.0e-8, 0.01])
def test_runner_lm_simple(model, psf_model_type, noise, guesser_type):
    """
    Test a Runner running the LM fitter
    """

    res_old = None
    for i in range(2):
        rng = np.random.RandomState(283)

        data = get_model_obs(
            rng=rng,
            model=model,
            noise=noise,
        )
        obs = data['obs']

        psf_ngauss = 3
        if psf_model_type == 'em':
            psf_guesser = GMixPSFGuesser(
                rng=rng,
                ngauss=psf_ngauss,
            )

            psf_fitter = GMixEM(tol=1.0e-5)
        else:
            psf_guesser = CoellipPSFGuesser(
                rng=rng,
                ngauss=psf_ngauss,
            )

            psf_fitter = CoellipFitter(ngauss=psf_ngauss)

        psf_runner = PSFRunner(
            fitter=psf_fitter,
            guesser=psf_guesser,
            ntry=2,
        )
        psf_runner.go(obs=obs)
        assert obs.psf.has_gmix()

        # always gets coverage
        assert guesser_type in ['TF', 'TPSFFlux']
        if guesser_type == 'TF':
            guesser = TFluxGuesser(
                rng=rng,
                T=0.25,
                flux=100.0,
            )
        elif guesser_type == 'TPSFFlux':
            guesser = TPSFFluxGuesser(
                rng=rng,
                T=0.25,
            )

        fitter = Fitter(model=model)

        runner = Runner(
            fitter=fitter,
            guesser=guesser,
            ntry=2,
        )
        res = runner.go(obs=obs)

        assert res['flags'] == 0

        pixel_scale = obs.jacobian.scale
        if noise <= 1.0e-8:
            assert abs(res['pars'][0]-data['pars'][0]) < pixel_scale/10
            assert abs(res['pars'][1]-data['pars'][1]) < pixel_scale/10

            assert abs(res['pars'][2]-data['pars'][2]) < 0.01
            assert abs(res['pars'][3]-data['pars'][3]) < 0.01

            assert abs(res['pars'][4]/data['pars'][4] - 1) < FRAC_TOL
            assert abs(res['pars'][5]/data['pars'][5] - 1) < FRAC_TOL

        # check reconstructed image allowing for noise
        imfit = res.make_image()
        imtol = 0.001 / pixel_scale**2 + noise*5
        assert np.all(np.abs(imfit - obs.image) < imtol)

        if i == 1:
            assert np.all(res['pars'] == res_old['pars'])
        else:
            res_old = res


def test_gaussmom_runner():
    """
    Test a Runner using GaussMom
    """

    rng = np.random.RandomState(8821)

    data = get_model_obs(
        rng=rng,
        model='gauss',
        noise=0.1,
    )

    obs = data['obs']

    fitter = ngmix.gaussmom.GaussMom(fwhm=1.2)

    runner = Runner(fitter=fitter)
    res = runner.go(obs=obs)
    assert res['flags'] == 0
    assert res['pars'].size == 6
