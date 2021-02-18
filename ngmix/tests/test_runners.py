import pytest
import numpy as np
from ngmix.runners import Runner, PSFRunner
from ngmix.guessers import PSFGMixGuesser, TFluxGuesser, CoellipPSFGuesser
from ngmix.fitting import LMCoellip
from ngmix.em import GMixEM
from ngmix.fitting import LMSimple
from ._sims import get_model_obs

FRAC_TOL = 5.0e-4


@pytest.mark.parametrize('psf_model_type', ['em', 'coellip'])
@pytest.mark.parametrize('model', ['exp', 'dev'])
def test_runner_lm_simple_smoke(model, psf_model_type):
    """
    Smoke test a Runner running the LMSimple fitter
    """

    rng = np.random.RandomState(17710)

    data = get_model_obs(
        rng=rng,
        model=model,
        noise=1.0e-4,
    )

    psf_ngauss = 3
    if psf_model_type == 'em':
        psf_guesser = PSFGMixGuesser(
            rng=rng,
            ngauss=psf_ngauss,
        )

        psf_fitter = GMixEM(tol=1.0e-5)
    else:
        psf_guesser = CoellipPSFGuesser(
            rng=rng,
            ngauss=psf_ngauss,
        )

        psf_fitter = LMCoellip(ngauss=psf_ngauss)

    psf_runner = PSFRunner(
        fitter=psf_fitter,
        guesser=psf_guesser,
        ntry=2,
    )
    psf_runner.go(obs=data['obs'], set_result=True)

    guesser = TFluxGuesser(
        rng=rng,
        T=0.25,
        fluxes=100.0,
    )
    fitter = LMSimple(model=model)

    runner = Runner(
        fitter=fitter,
        guesser=guesser,
        ntry=2,
    )
    runner.go(obs=data['obs'])

    fitter = runner.fitter
    res = fitter.get_result()
    assert res['flags'] == 0


@pytest.mark.parametrize('psf_model_type', ['em', 'coellip'])
@pytest.mark.parametrize('model', ['exp', 'dev'])
@pytest.mark.parametrize('noise', [1.0e-8, 0.01])
def test_runner_lm_simple(model, psf_model_type, noise):
    """
    Smoke test a Runner running the LMSimple fitter
    """

    rng = np.random.RandomState(283)

    data = get_model_obs(
        rng=rng,
        model=model,
        noise=noise,
    )
    obs = data['obs']

    psf_ngauss = 3
    if psf_model_type == 'em':
        psf_guesser = PSFGMixGuesser(
            rng=rng,
            ngauss=psf_ngauss,
        )

        psf_fitter = GMixEM(tol=1.0e-5)
    else:
        psf_guesser = CoellipPSFGuesser(
            rng=rng,
            ngauss=psf_ngauss,
        )

        psf_fitter = LMCoellip(ngauss=psf_ngauss)

    psf_runner = PSFRunner(
        fitter=psf_fitter,
        guesser=psf_guesser,
        ntry=2,
    )
    psf_runner.go(obs=obs, set_result=True)

    guesser = TFluxGuesser(
        rng=rng,
        T=0.25,
        fluxes=100.0,
    )
    fitter = LMSimple(model=model)

    runner = Runner(
        fitter=fitter,
        guesser=guesser,
        ntry=2,
    )
    runner.go(obs=obs)

    fitter = runner.fitter
    res = fitter.get_result()
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
    imfit = fitter.make_image()
    imtol = 0.001 / pixel_scale**2 + noise*5
    assert np.all(np.abs(imfit - obs.image) < imtol)