"""
just test moment errors
"""
import pytest
import numpy as np
import ngmix
from ngmix.runners import Runner, PSFRunner
from ngmix.guessers import SimplePSFGuesser, TFluxAndPriorGuesser
from ngmix.fitting import Fitter
from ngmix.gaussmom import GaussMom
from ngmix.metacal import metacal_bootstrap, MetacalBootstrapper
from ._sims import get_model_obs
from ._priors import get_prior
from ._galsim_sims import _get_obs

FRAC_TOL = 5.0e-4


@pytest.mark.parametrize('noise', [1.0e-8, 0.01])
@pytest.mark.parametrize('use_bootstrapper', [False, True])
@pytest.mark.parametrize('nband', [None, 2])
@pytest.mark.parametrize('nepoch', [None, 2])
def test_metacal_bootstrap_max_smoke(
    noise, use_bootstrapper, nband, nepoch, metacal_caching
):
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

    fitter = Fitter(model=fit_model, prior=prior)
    psf_fitter = Fitter(model='gauss')

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

    if use_bootstrapper:
        boot = MetacalBootstrapper(
            runner=runner, psf_runner=psf_runner,
            rng=rng,
        )
        resdict, obsdict = boot.go(obs)

        _ = boot.fitter  # for coverage
    else:
        resdict, obsdict = metacal_bootstrap(
            obs=obs, runner=runner, psf_runner=psf_runner,
            rng=rng,
        )

    for key in ['noshear', '1p', '1m', '2p', '2m']:
        assert key in resdict
        assert key in obsdict
        assert resdict[key]['flags'] == 0

        if isinstance(obsdict[key], ngmix.Observation):
            assert obsdict[key].has_psf()
            assert 'result' in obsdict[key].psf.meta


@pytest.mark.parametrize('noise', [1.0e-8, 0.01])
@pytest.mark.parametrize('use_bootstrapper', [False, True])
def test_metacal_bootstrap_gaussmom_smoke(
    noise, use_bootstrapper, metacal_caching,
):
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

    if use_bootstrapper:
        boot = MetacalBootstrapper(
            runner=runner, psf_runner=psf_runner,
            rng=rng,
        )
        resdict, obsdict = boot.go(obs)
    else:
        resdict, obsdict = metacal_bootstrap(
            obs=obs, runner=runner, psf_runner=psf_runner,
            rng=rng,
        )

    for key in ['noshear', '1p', '1m', '2p', '2m']:
        assert key in resdict
        assert key in obsdict
        assert resdict[key]['flags'] == 0

        if isinstance(obsdict[key], ngmix.Observation):
            assert obsdict[key].has_psf()
            assert 'result' in obsdict[key].psf.meta


def test_metacal_bootstrap_gaussmom_response(metacal_caching):
    """
    test a metacal bootstrapper with gaussian moments
    """

    rng = np.random.RandomState(2830)
    ntrial = 50

    fwhm = 1.2
    fitter = GaussMom(fwhm=fwhm)
    psf_fitter = GaussMom(fwhm=fwhm)

    psf_runner = PSFRunner(fitter=psf_fitter)
    runner = Runner(fitter=fitter)

    boot = MetacalBootstrapper(
        runner=runner, psf_runner=psf_runner,
        rng=rng,
        types=['1p', '1m'],
    )

    Rvals = np.zeros(ntrial)
    for i in range(ntrial):
        obs = _get_obs(
            rng=rng,
            set_noise_image=False,
        )

        resdict, obsdict = boot.go(obs)

        res1p = resdict['1p']
        res1m = resdict['1m']

        Rvals[i] = (res1p['e'][0] - res1m['e'][0])/0.02

    Rmean = Rvals.mean()
    # this response value comes from 2.0.0, and had an error less than of
    # 1.0e-5 on it.  Allow for differences in rng/arch etc. by taking 1.0e-4
    assert abs(Rmean - 0.28535) < 1.0e-4
