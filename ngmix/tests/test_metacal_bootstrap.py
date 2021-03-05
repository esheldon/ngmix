"""
just test moment errors
"""
import pytest
import numpy as np
from ngmix.runners import Runner, PSFRunner
from ngmix.guessers import SimplePSFGuesser, TFluxAndPriorGuesser
from ngmix.fitting import LM
from ngmix.gaussmom import GaussMom
from ngmix.metacal_bootstrap import metacal_bootstrap, MetacalBootstrapper
from ._sims import get_model_obs
from ._priors import get_prior

FRAC_TOL = 5.0e-4


@pytest.mark.parametrize('noise', [1.0e-8, 0.01])
@pytest.mark.parametrize('use_bootstrapper', [False, True])
@pytest.mark.parametrize('nband', [None, 2])
@pytest.mark.parametrize('nepoch', [None, 2])
def test_metacal_bootstrap_max_smoke(noise, use_bootstrapper, nband, nepoch):
    """
    test a metacal bootstrapper with maxlike fitting
    """

    rng = np.random.RandomState(2830)

    model = 'gauss'
    fit_model = 'gauss'

    data = get_model_obs(
        rng=rng,
        model=model,
        noise=noise,
        nepoch=nepoch,
        nband=nband,
    )
    obs = data['obs']

    prior = get_prior(
        fit_model=fit_model,
        rng=rng,
        scale=0.2,
        T_range=[-1.0, 1.e3],
        F_range=[0.01, 1000.0],
        nband=nband,
    )

    flux_guess = data['gmix'].get_flux()
    Tguess = data['gmix'].get_T()
    guesser = TFluxAndPriorGuesser(
        rng=rng, T=Tguess, flux=flux_guess, prior=prior,
    )
    psf_guesser = SimplePSFGuesser(rng=rng)

    fitter = LM(model=fit_model, prior=prior)
    psf_fitter = LM(model='gauss')

    psf_runner = PSFRunner(
        fitter=psf_fitter,
        guesser=psf_guesser,
        ntry=2,
    )
    runner = Runner(
        fitter=fitter,
        guesser=guesser,
        ntry=2,
    )

    mcal_kws = {'psf': 'gauss'}
    if use_bootstrapper:
        boot = MetacalBootstrapper(
            runner=runner, psf_runner=psf_runner,
            rng=rng,
            **mcal_kws,
        )
        boot.go(obs)
        resdict = boot.get_result()
    else:
        resdict = metacal_bootstrap(
            obs=obs, runner=runner, psf_runner=psf_runner,
            rng=rng,
            **mcal_kws,
        )

    for key in ['noshear', '1p', '1m', '2p', '2m']:
        assert key in resdict
        assert resdict[key]['flags'] == 0


@pytest.mark.parametrize('noise', [1.0e-8, 0.01])
@pytest.mark.parametrize('use_bootstrapper', [False, True])
def test_metacal_bootstrap_gaussmom_smoke(noise, use_bootstrapper):
    """
    test a metacal bootstrapper with gaussian moments
    """

    rng = np.random.RandomState(2830)

    data = get_model_obs(
        rng=rng,
        model='gauss',
        noise=noise,
    )
    obs = data['obs']

    fwhm = 1.2
    fitter = GaussMom(fwhm=fwhm)
    psf_fitter = GaussMom(fwhm=fwhm)

    psf_runner = PSFRunner(fitter=psf_fitter)
    runner = Runner(fitter=fitter)

    mcal_kws = {'psf': 'gauss'}
    if use_bootstrapper:
        boot = MetacalBootstrapper(
            runner=runner, psf_runner=psf_runner,
            rng=rng,
            **mcal_kws,
        )
        boot.go(obs)
        resdict = boot.get_result()
    else:
        resdict = metacal_bootstrap(
            obs=obs, runner=runner, psf_runner=psf_runner,
            rng=rng,
            **mcal_kws,
        )

    for key in ['noshear', '1p', '1m', '2p', '2m']:
        assert key in resdict
        assert resdict[key]['flags'] == 0
