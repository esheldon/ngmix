"""
just test moment errors
"""
import pytest
import numpy as np
import ngmix
from ngmix.runners import Runner, PSFRunner
from ngmix.guessers import GMixPSFGuesser, TFluxGuesser, CoellipPSFGuesser
from ngmix.fitting import LMCoellip
from ngmix.em import GMixEM
from ngmix.fitting import LM
from ngmix.bootstrap import bootstrap, Bootstrapper
from ._sims import get_model_obs
from ._priors import get_prior

FRAC_TOL = 5.0e-4


@pytest.mark.parametrize('psf_model_type', ['em', 'coellip'])
@pytest.mark.parametrize('model', ['gauss', 'exp', 'dev'])
@pytest.mark.parametrize('noise', [1.0e-7, 0.01])
@pytest.mark.parametrize('guess_from_moms', [True, False])
@pytest.mark.parametrize('use_prior', [False, True])
@pytest.mark.parametrize('use_bootstrapper', [False, True])
def test_bootstrap(model, psf_model_type, guess_from_moms, noise,
                   use_prior, use_bootstrapper):
    """
    Smoke test a Runner running the LM fitter
    """

    rng = np.random.RandomState(2830)

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
            guess_from_moms=guess_from_moms,
        )

        psf_fitter = GMixEM(tol=1.0e-5)
    else:
        psf_guesser = CoellipPSFGuesser(
            rng=rng,
            ngauss=psf_ngauss,
            guess_from_moms=guess_from_moms,
        )

        psf_fitter = LMCoellip(ngauss=psf_ngauss)

    psf_runner = PSFRunner(
        fitter=psf_fitter,
        guesser=psf_guesser,
        ntry=2,
    )

    guesser = TFluxGuesser(
        rng=rng,
        T=0.25,
        flux=100.0,
    )
    prior = get_prior(
        fit_model=model,
        rng=rng,
        scale=obs.jacobian.scale,
        T_range=[-1.0, 1.e3],
        F_range=[0.01, 1000.0],
    )

    fitter = LM(model=model, prior=prior)

    runner = Runner(
        fitter=fitter,
        guesser=guesser,
        ntry=2,
    )

    if use_bootstrapper:
        boot = Bootstrapper(runner=runner, psf_runner=psf_runner)
        res = boot.go(obs)
        _ = boot.fitter  # for coverage
    else:
        res = bootstrap(obs=obs, runner=runner, psf_runner=psf_runner)

    assert res['flags'] == 0

    pixel_scale = obs.jacobian.scale
    if noise <= 1.0e-7:
        assert abs(res['pars'][0]-data['pars'][0]) < pixel_scale/10
        assert abs(res['pars'][1]-data['pars'][1]) < pixel_scale/10

        print('true:', data['pars'][2:2+2])
        print('fit: ', res['pars'][2:2+2])
        assert abs(res['pars'][2]-data['pars'][2]) < 0.01
        assert abs(res['pars'][3]-data['pars'][3]) < 0.01

        # dev is hard to get right
        if model != 'dev':
            assert abs(res['pars'][4]/data['pars'][4] - 1) < FRAC_TOL

        assert abs(res['pars'][5]/data['pars'][5] - 1) < FRAC_TOL

    # check reconstructed image allowing for noise
    imfit = res.make_image()
    maxdiff = np.abs(imfit - obs.image).max()
    immax = obs.image.max()
    max_reldiff = maxdiff/immax - 1
    reltol = 0.001 + noise * 5 / immax

    assert max_reldiff < reltol


@pytest.mark.parametrize('nband', [None, 2])
@pytest.mark.parametrize('nepoch', [None, 1, 10])
def test_remove_failed_psf(nepoch, nband):
    """
    test removing obs with bad fits
    """

    rng = np.random.RandomState(2830)

    data = get_model_obs(
        rng=rng,
        model='gauss',
        noise=0.01,
        nepoch=nepoch,
        nband=nband,
    )

    if nepoch is None:
        test_nepoch = 1
    else:
        test_nepoch = nepoch

    if isinstance(data['obs'], ngmix.MultiBandObsList):
        mbobs = data['obs']
        for obslist in mbobs:
            assert len(obslist) == test_nepoch

            index = rng.randint(len(obslist))
            for i, obs in enumerate(obslist):
                if i == index:
                    flags = 1
                else:
                    flags = 0
                obs.psf.meta['result'] = {'flags': flags}

        if nepoch is None or nepoch == 1:
            with pytest.raises(ngmix.gexceptions.BootPSFFailure):
                _ = ngmix.bootstrap.remove_failed_psf_obs(mbobs)
        else:
            new_mbobs = ngmix.bootstrap.remove_failed_psf_obs(mbobs)
            for obslist in new_mbobs:
                assert len(obslist) == test_nepoch-1
    elif isinstance(data['obs'], ngmix.ObsList):
        obslist = data['obs']
        assert len(obslist) == test_nepoch

        index = rng.randint(len(obslist))
        for i, obs in enumerate(obslist):
            if i == index:
                flags = 1
            else:
                flags = 0
            obs.psf.meta['result'] = {'flags': flags}

        if test_nepoch == 1:
            with pytest.raises(ngmix.gexceptions.BootPSFFailure):
                _ = ngmix.bootstrap.remove_failed_psf_obs(obslist)
        else:
            new_obslist = ngmix.bootstrap.remove_failed_psf_obs(obslist)
            assert len(new_obslist) == test_nepoch-1
    else:
        obs = data['obs']

        obs.psf.meta['result'] = {'flags': 1}

        with pytest.raises(ngmix.gexceptions.BootPSFFailure):
            _ = ngmix.bootstrap.remove_failed_psf_obs(obs)

    with pytest.raises(ValueError):
        _ = ngmix.bootstrap.remove_failed_psf_obs(None)
