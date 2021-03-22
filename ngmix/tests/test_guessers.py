import pytest
import numpy as np

import ngmix
from ngmix import guessers
from ngmix.gmix import get_coellip_npars
from ._sims import get_model_obs, get_psf_obs
from ._priors import get_prior


@pytest.mark.parametrize(
    'guesser_type',
    ['TFlux', 'TPSFFlux', 'Pars', 'Pars+Width', 'R50Flux', 'R50NuFlux'],
)
@pytest.mark.parametrize('nband', [None, 1, 2])
@pytest.mark.parametrize('with_prior', [False, True])
def test_noprior_guessers_smoke(guesser_type, nband, with_prior):

    guess_old = None
    for i in range(2):
        rng = np.random.RandomState(587)

        if with_prior:
            prior = get_prior(fit_model='exp', rng=rng, scale=0.263)
        else:
            prior = None

        data = get_model_obs(
            model='gauss',  # it has T=0 for a star
            rng=rng, star=True,
            set_psf_gmix=True, nepoch=10,
            nband=nband,
        )

        nband_use = nband if nband is not None else 1

        # this always gets coverage
        assert guesser_type in [
            'TFlux', 'TPSFFlux', 'Pars', 'Pars+Width', 'R50Flux', 'R50NuFlux'
        ]
        if guesser_type == 'TFlux':
            T_center = 0.001
            flux_center = [1.0]*nband_use
            guesser = guessers.TFluxGuesser(
                rng=rng, T=T_center, flux=flux_center, prior=prior,
            )
            npars = 5 + nband_use
        elif guesser_type == 'TPSFFlux':
            T_center = 0.001
            guesser = guessers.TPSFFluxGuesser(rng=rng, T=T_center, prior=prior)
            npars = 5 + nband_use
        elif guesser_type == 'Pars':
            pars = [0.0, 0.0, 0.0, 0.0, 0.1] + [1.0]*nband_use
            guesser = guessers.ParsGuesser(rng=rng, pars=pars, prior=prior)
            npars = 5 + nband_use

        elif guesser_type == 'Pars+Width':
            pars = [0.0, 0.0, 0.0, 0.0, 0.1] + [1.0]*nband_use
            widths = [0.1]*len(pars)
            guesser = guessers.ParsGuesser(
                rng=rng, pars=pars, prior=prior, widths=widths,
            )
            npars = 5 + nband_use

        elif guesser_type == 'R50Flux':
            r50_center = 0.2
            flux_center = [1.0]*nband_use
            guesser = guessers.R50FluxGuesser(
                rng=rng, r50=r50_center, flux=flux_center, prior=prior,
            )
            npars = 5 + nband_use
        elif guesser_type == 'R50NuFlux':
            r50_center = 0.2
            flux_center = [1.0]*nband_use
            nu_center = 1
            guesser = guessers.R50NuFluxGuesser(
                rng=rng, r50=r50_center, nu=nu_center, flux=flux_center,
                prior=prior,
            )
            npars = 6 + nband_use

        guess = guesser(obs=data['obs'])
        assert guess.size == npars

        guess = guesser(obs=data['obs'], nrand=3)
        assert guess.shape[0] == 3
        assert guess.shape[1] == npars

        if i == 1:
            assert np.all(guess == guess_old)
        else:
            guess_old = guess


@pytest.mark.parametrize(
    'guesser_type',
    ['TFluxAndPrior', 'TPSFFluxAndPrior', 'BD', 'BDF', 'Prior'],
)
@pytest.mark.parametrize('nband', [None, 1, 2])
def test_prior_guessers_smoke(guesser_type, nband):
    nband_use = nband if nband is not None else 1

    guess_old = None
    for i in range(2):

        rng = np.random.RandomState(19487)
        data = get_model_obs(
            model='gauss',  # it has T=0 for a star
            rng=rng, star=True,
            set_psf_gmix=True, nepoch=10,
            nband=nband,
        )

        # this always gets coverage
        assert guesser_type in [
            'TFluxAndPrior', 'TPSFFluxAndPrior', 'BD', 'BDF', 'Prior',
        ]

        T_center = 0.001
        flux_center = [1.0]*nband_use
        if 'Prior' in guesser_type:
            prior = ngmix.joint_prior.PriorSimpleSep(
                cen_prior=ngmix.priors.CenPrior(0.0, 0.0, 0.1, 0.1, rng=rng),
                g_prior=ngmix.priors.GPriorBA(0.3, rng=rng),
                T_prior=ngmix.priors.FlatPrior(-1.0, 1.e5, rng=rng),
                F_prior=[ngmix.priors.FlatPrior(-1.0, 1.e5, rng=rng)]*nband_use,
            )

            if 'PSF' in guesser_type:
                guesser = guessers.TPSFFluxAndPriorGuesser(
                    rng=rng, T=T_center, prior=prior,
                )
            elif 'Flux' in guesser_type:
                guesser = guessers.TFluxAndPriorGuesser(
                    rng=rng, T=T_center, flux=flux_center, prior=prior,
                )
            else:
                guesser = guessers.PriorGuesser(prior=prior)
            npars = 5 + nband_use
        elif guesser_type == 'BD':
            prior = ngmix.joint_prior.PriorBDSep(
                cen_prior=ngmix.priors.CenPrior(0.0, 0.0, 0.1, 0.1, rng=rng),
                g_prior=ngmix.priors.GPriorBA(0.3, rng=rng),
                logTratio_prior=ngmix.priors.Normal(1.0, 0.5, rng=rng),
                fracdev_prior=ngmix.priors.Normal(0.5, 0.1, rng=rng),
                T_prior=ngmix.priors.FlatPrior(-1.0, 1.e5, rng=rng),
                F_prior=[ngmix.priors.FlatPrior(-1.0, 1.e5, rng=rng)]*nband_use,
            )
            guesser = guessers.BDGuesser(
                T=T_center, flux=flux_center, prior=prior,
            )
            npars = 7 + nband_use
        elif guesser_type == 'BDF':
            prior = ngmix.joint_prior.PriorBDFSep(
                cen_prior=ngmix.priors.CenPrior(0.0, 0.0, 0.1, 0.1, rng=rng),
                g_prior=ngmix.priors.GPriorBA(0.3, rng=rng),
                fracdev_prior=ngmix.priors.Normal(0.5, 0.1, rng=rng),
                T_prior=ngmix.priors.FlatPrior(-1.0, 1.e5, rng=rng),
                F_prior=[ngmix.priors.FlatPrior(-1.0, 1.e5, rng=rng)]*nband_use,
            )
            guesser = guessers.BDFGuesser(
                T=T_center, flux=flux_center, prior=prior,
            )
            npars = 6 + nband_use

        guess = guesser(obs=data['obs'])
        assert guess.size == npars

        if i == 1:
            assert np.all(guess == guess_old)
        else:
            guess_old = guess


@pytest.mark.parametrize(
    'guesser_type',
    ['GMix', 'Simple', 'Coellip'],
)
@pytest.mark.parametrize(
    'ngauss', [1, 2, 3, 4, 5],
)
@pytest.mark.parametrize(
    'guess_from_moms', [False, True],
)
def test_psf_guessers_smoke(guesser_type, ngauss, guess_from_moms):

    guess_old = None
    for i in range(2):
        rng = np.random.RandomState(19487)
        data = get_psf_obs(rng=rng)

        # this always gets coverage
        assert guesser_type in ['GMix', 'Simple', 'Coellip']

        if guesser_type == 'GMix':
            guesser = guessers.GMixPSFGuesser(
                rng=rng, ngauss=ngauss, guess_from_moms=guess_from_moms,
            )
            npars = ngauss
        elif guesser_type == 'Simple':
            # note actually redundant for each ngauss
            guesser = guessers.SimplePSFGuesser(
                rng=rng, guess_from_moms=guess_from_moms,
            )
            npars = 6
        elif guesser_type == 'Coellip':
            # note actually redundant for each ngauss
            guesser = guessers.CoellipPSFGuesser(
                rng=rng, ngauss=ngauss, guess_from_moms=guess_from_moms,
            )
            npars = get_coellip_npars(ngauss)

        guess = guesser(obs=data['obs'])
        assert len(guess) == npars

        if i == 1:
            if guesser_type == 'GMix':
                guess_old = guess_old.get_full_pars()
                guess = guess.get_full_pars()
            assert np.all(guess == guess_old)
        else:
            guess_old = guess


def test_guessers_shape_guess():
    rng = np.random.RandomState(88)
    g = ngmix.guessers.get_shape_guess(
        rng=rng, g1=1.5, g2=0.1, nrand=10, width=[0.1]*2,
    )
    assert np.all(np.sqrt(g[:, 0]**2 + g[:, 1]**2) < 1)


def test_guessers_errors():
    rng = np.random.RandomState(8)
    with pytest.raises(ngmix.GMixRangeError):
        ngmix.guessers.R50FluxGuesser(rng=rng, r50=-1, flux=100)

    with pytest.raises(ValueError):
        ngmix.guessers.GMixPSFGuesser(rng=rng, ngauss=1000)

    with pytest.raises(ValueError):
        ngmix.guessers.CoellipPSFGuesser(rng=rng, ngauss=1000)
