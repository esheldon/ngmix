import pytest
import numpy as np
from ngmix.runners import PSFRunner
from ngmix.guessers import GuesserEMPSF, GuesserCoellipPSF
from ngmix.em import GMixEM
from ngmix.fitting import LMCoellip
from ._sims import get_ngauss_obs, get_psf_obs


@pytest.mark.parametrize('guess_from_moms', [False, True])
@pytest.mark.parametrize('ngauss', [1, 2, 3, 4, 5])
def test_em_psf_runner_smoke(ngauss, guess_from_moms):
    """
    Smoke test a PSFRunner running the EM fitter
    """

    rng = np.random.RandomState(8821)

    data = get_psf_obs()

    obs = data['obs']

    guesser = GuesserEMPSF(
        rng=rng,
        ngauss=ngauss,
        guess_from_moms=guess_from_moms,
    )
    # better tolerance needed for this psf fit
    fitter = GMixEM(tol=1.0e-5)

    runner = PSFRunner(
        fitter=fitter,
        guesser=guesser,
        ntry=2,
    )
    runner.go(obs=obs)

    fitter = runner.fitter
    res = fitter.get_result()
    assert res['flags'] == 0


@pytest.mark.parametrize('with_psf_obs', [False, True])
@pytest.mark.parametrize('guess_from_moms', [False, True])
def test_em_psf_runner(with_psf_obs, guess_from_moms):
    """
    Test a PSFRunner running the EM fitter
    """

    rng = np.random.RandomState(8821)

    if with_psf_obs:
        data = get_ngauss_obs(
            rng=rng,
            ngauss=1,
            noise=0.0,
            with_psf=True,
        )
    else:
        data = get_psf_obs()

    obs = data['obs']

    guesser = GuesserEMPSF(
        rng=rng,
        ngauss=3,
        guess_from_moms=guess_from_moms,
    )
    # better tolerance needed for this psf fit
    fitter = GMixEM(tol=1.0e-5)

    runner = PSFRunner(
        fitter=fitter,
        guesser=guesser,
        ntry=2,
    )
    runner.go(obs=obs)

    fitter = runner.fitter
    res = fitter.get_result()
    assert res['flags'] == 0

    # check reconstructed image allowing for noise
    imfit = fitter.make_image()

    if with_psf_obs:
        comp_image = obs.psf.image
    else:
        comp_image = obs.image

    imtol = 0.001 / obs.jacobian.scale**2
    assert np.abs(imfit - comp_image).max() < imtol


@pytest.mark.parametrize('guess_from_moms', [False, True])
@pytest.mark.parametrize('ngauss', [1, 2, 3, 4, 5])
def test_coellip_psf_runner_smoke(ngauss, guess_from_moms):
    """
    Smoke test a PSFRunner running the coelliptical fitter
    """

    rng = np.random.RandomState(9321)

    # make a complex psf so the fitting with high ngauss doesn't become too
    # degenerate
    data = get_psf_obs()
    data2 = get_psf_obs(T=1.0)
    data3 = get_psf_obs(T=2.0)

    combined_im = (
        data['obs'].image + data2['obs'].image + data3['obs'].image
    )
    obs = data['obs']
    obs.image = combined_im

    guesser = GuesserCoellipPSF(
        rng=rng,
        ngauss=ngauss,
        guess_from_moms=guess_from_moms,
    )
    fitter = LMCoellip(ngauss=ngauss)

    runner = PSFRunner(
        fitter=fitter,
        guesser=guesser,
        ntry=2,
    )
    runner.go(obs=obs)

    fitter = runner.fitter
    res = fitter.get_result()
    assert res['flags'] == 0


@pytest.mark.parametrize('with_psf_obs', [False, True])
@pytest.mark.parametrize('guess_from_moms', [False, True])
def test_coellip_psf_runner(with_psf_obs, guess_from_moms):
    """
    Test a PSFRunner running the coelliptical fitter
    """

    rng = np.random.RandomState(21)

    if with_psf_obs:
        data = get_ngauss_obs(
            rng=rng,
            ngauss=1,
            noise=0.0,
            with_psf=True,
        )
    else:
        data = get_psf_obs()

    obs = data['obs']

    ngauss = 3
    guesser = GuesserCoellipPSF(
        rng=rng,
        ngauss=ngauss,
        guess_from_moms=guess_from_moms,
    )
    # better tolerance needed for this psf fit
    fitter = LMCoellip(ngauss=ngauss)

    runner = PSFRunner(
        fitter=fitter,
        guesser=guesser,
        ntry=4,
    )
    runner.go(obs=obs)

    fitter = runner.fitter
    res = fitter.get_result()
    assert res['flags'] == 0

    # check reconstructed image allowing for noise
    imfit = fitter.make_image()

    if with_psf_obs:
        comp_image = obs.psf.image
    else:
        comp_image = obs.image

    imtol = 0.001 / obs.jacobian.scale**2
    assert np.abs(imfit - comp_image).max() < imtol
