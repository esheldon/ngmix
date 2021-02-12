"""
just test moment errors
"""
import pytest
import numpy as np
from ngmix import priors, joint_prior
from ngmix.runners import Runner, PSFRunner
from ngmix.guessers import EMPSFGuesser, TFluxGuesser, CoellipPSFGuesser
from ngmix.fitting import LMCoellip
from ngmix.em import GMixEM
from ngmix.fitting import LMSimple
from ngmix.bootstrap import bootstrap
from ._sims import get_model_obs

FRAC_TOL = 5.0e-4


def get_prior(*, rng, cen, cen_width, T_range, F_range):
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

    pr = joint_prior.PriorSimpleSep(cen_prior, g_prior, T_prior, F_prior)
    return pr


@pytest.mark.parametrize('psf_model_type', ['em', 'coellip'])
@pytest.mark.parametrize('model', ['gauss', 'exp', 'dev'])
@pytest.mark.parametrize('noise', [1.0e-8, 0.01])
@pytest.mark.parametrize('guess_from_moms', [True, False])
@pytest.mark.parametrize('use_prior', [False, True])
def test_bootstrap(model, psf_model_type, guess_from_moms, noise, use_prior):
    """
    Smoke test a Runner running the LMSimple fitter
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
        psf_guesser = EMPSFGuesser(
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
        fluxes=100.0,
    )
    prior = get_prior(
        rng=rng,
        cen=[0.0, 0.0],
        cen_width=obs.jacobian.scale,
        T_range=[-1.0, 1.e3],
        F_range=[0.01, 1000.0],
    )
    fitter = LMSimple(model=model, prior=prior)

    runner = Runner(
        fitter=fitter,
        guesser=guesser,
        ntry=2,
    )

    bootstrap(obs=obs, runner=runner, psf_runner=psf_runner)

    fitter = runner.fitter
    res = fitter.get_result()
    assert res['flags'] == 0

    pixel_scale = obs.jacobian.scale
    if noise <= 1.0e-8:
        assert abs(res['pars'][0]-data['pars'][0]) < pixel_scale/10
        assert abs(res['pars'][1]-data['pars'][1]) < pixel_scale/10

        assert abs(res['pars'][2]-data['pars'][2]) < 0.01
        assert abs(res['pars'][3]-data['pars'][3]) < 0.01

        # dev is hard to get right
        if model != 'dev':
            assert abs(res['pars'][4]/data['pars'][4] - 1) < FRAC_TOL

        assert abs(res['pars'][5]/data['pars'][5] - 1) < FRAC_TOL

    # check reconstructed image allowing for noise
    imfit = fitter.make_image()
    maxdiff = np.abs(imfit - obs.image).max()
    immax = obs.image.max()
    # imtol = 0.001 / pixel_scale**2 + noise*5
    max_reldiff = maxdiff/immax - 1
    reltol = 0.001 + noise * 5 / immax
    if max_reldiff > reltol:
        from espy import images
        images.compare_images(obs.image, imfit)

    assert max_reldiff < reltol
