import pytest
import numpy as np

import ngmix
from ngmix import guessers
from ._sims import get_model_obs


@pytest.mark.parametrize(
    'guesser_type',
    ['TFlux', 'TPSFFlux', 'Pars', 'R50Flux', 'R50NuFlux'],
)
@pytest.mark.parametrize('nband', [None, 1, 2])
def test_noprior_guessers_smoke(guesser_type, nband):
    rng = np.random.RandomState(587)
    data = get_model_obs(
        model='gauss',  # it has T=0 for a star
        rng=rng, star=True,
        set_psf_gmix=True, nepoch=10,
        nband=nband,
    )

    nband_use = nband if nband is not None else 1

    if guesser_type == 'TFlux':
        T_center = 0.001
        flux_center = [1.0]*nband_use
        guesser = guessers.TFluxGuesser(rng=rng, T=T_center, flux=flux_center)
        npars = 5 + nband_use
    elif guesser_type == 'TPSFFlux':
        T_center = 0.001
        guesser = guessers.TPSFFluxGuesser(rng=rng, T=T_center)
        npars = 5 + nband_use
    elif guesser_type == 'Pars':
        pars = [0.0, 0.0, 0.0, 0.0, 0.1] + [1.0]*nband_use
        guesser = guessers.ParsGuesser(rng=rng, pars=pars)
        npars = 5 + nband_use
    elif guesser_type == 'R50Flux':
        r50_center = 0.2
        flux_center = [1.0]*nband_use
        guesser = guessers.R50FluxGuesser(rng=rng, r50=r50_center, flux=flux_center)
        npars = 5 + nband_use
    elif guesser_type == 'R50NuFlux':
        r50_center = 0.2
        flux_center = [1.0]*nband_use
        nu_center = 1
        guesser = guessers.R50NuFluxGuesser(
            rng=rng, r50=r50_center, nu=nu_center, flux=flux_center,
        )
        npars = 6 + nband_use
    else:
        raise ValueError('bad guesser %s' % guesser_type)

    if nband is None:
        obs_use = data['obslist']
    else:
        obs_use = data['mbobs']

    guess = guesser(obs=obs_use)
    assert guess.size == npars


@pytest.mark.parametrize('nband', [None, 1, 2])
def test_T_flux_prior_guesser_smoke(nband):
    nband_use = nband if nband is not None else 1

    rng = np.random.RandomState(19487)
    data = get_model_obs(
        model='gauss',  # it has T=0 for a star
        rng=rng, star=True,
        set_psf_gmix=True, nepoch=10,
        nband=nband,
    )

    prior = ngmix.joint_prior.PriorSimpleSep(
        cen_prior=ngmix.priors.CenPrior(0.0, 0.0, 0.1, 0.1, rng=rng),
        g_prior=ngmix.priors.GPriorBA(0.3, rng=rng),
        T_prior=ngmix.priors.FlatPrior(-1.0, 1.e5, rng=rng),
        F_prior=[ngmix.priors.FlatPrior(-1.0, 1.e5, rng=rng)]*nband_use,
    )
    T_center = 0.001
    flux_center = [1.0]*nband_use

    guesser = guessers.TFluxAndPriorGuesser(
        rng=rng, T=T_center, flux=flux_center, prior=prior,
    )

    if nband is None:
        obs_use = data['obslist']
    else:
        obs_use = data['mbobs']

    guess = guesser(obs=obs_use)
    nband_use = nband if nband is not None else 1
    assert guess.size == 5 + nband_use
