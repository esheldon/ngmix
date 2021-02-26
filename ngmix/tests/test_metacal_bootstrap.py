"""
just test moment errors
"""
import pytest
import numpy as np
from ngmix import priors, joint_prior
from ngmix.runners import Runner, PSFRunner
from ngmix.guessers import SimplePSFGuesser, TFluxAndPriorGuesser
from ngmix.fitting import LM
from ngmix.gaussmom import GaussMom
from ngmix.metacal_bootstrap import metacal_bootstrap, MetacalBootstrapper
from ._sims import get_model_obs

FRAC_TOL = 5.0e-4


def get_prior(*, rng, cen, cen_width, T_range, F_range, nband):
    """
    For testing

    Make PriorSimpleSep uniform in all priors except the
    center, which is gaussian
    """
    cen_prior = priors.CenPrior(
        cen[0], cen[1], cen_width, cen_width, rng=rng,
    )
    g_prior = priors.GPriorBA(0.3, rng=rng)
    T_prior = priors.FlatPrior(T_range[0], T_range[1], rng=rng)
    F_prior = priors.FlatPrior(F_range[0], F_range[1], rng=rng)

    if nband is not None:
        F_prior = [F_prior]*nband

    pr = joint_prior.PriorSimpleSep(
        cen_prior=cen_prior,
        g_prior=g_prior,
        T_prior=T_prior,
        F_prior=F_prior,
    )
    return pr


@pytest.mark.parametrize('noise', [1.0e-8, 0.01])
@pytest.mark.parametrize('use_bootstrapper', [False, True])
@pytest.mark.parametrize('nband', [None, 2])
@pytest.mark.parametrize('nepoch', [None, 2])
def test_metacal_bootstrap_max_smoke(noise, use_bootstrapper, nband, nepoch):
    """
    Smoke test a Runner running the LM fitter
    """

    rng = np.random.RandomState(2830)

    data = get_model_obs(
        rng=rng,
        model='gauss',
        noise=noise,
        nepoch=nepoch,
        nband=nband,
    )
    obs = data['obs']

    prior = get_prior(
        rng=rng,
        cen=[0.0, 0.0],
        cen_width=1.0,
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

    fitter = LM(model="gauss", prior=prior)
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
            runner=runner, psf_runner=psf_runner, **mcal_kws,
        )
        boot.go(obs)
        resdict = boot.get_result()
    else:
        resdict = metacal_bootstrap(
            obs=obs, runner=runner, psf_runner=psf_runner, **mcal_kws,
        )

    for key in ['noshear', '1p', '1m', '2p', '2m']:
        assert key in resdict
        assert resdict[key]['flags'] == 0


@pytest.mark.parametrize('noise', [1.0e-8, 0.01])
@pytest.mark.parametrize('use_bootstrapper', [False, True])
def test_metacal_bootstrap_gaussmom_smoke(noise, use_bootstrapper):
    """
    Smoke test a Runner running the LM fitter
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
            runner=runner, psf_runner=psf_runner, **mcal_kws,
        )
        boot.go(obs)
        resdict = boot.get_result()
    else:
        resdict = metacal_bootstrap(
            obs=obs, runner=runner, psf_runner=psf_runner, **mcal_kws,
        )

    for key in ['noshear', '1p', '1m', '2p', '2m']:
        assert key in resdict
        assert resdict[key]['flags'] == 0
