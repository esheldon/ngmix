import pytest
import numpy as np
import ngmix
from ngmix.runners import PSFRunner
from ngmix.guessers import GMixPSFGuesser, SimplePSFGuesser, CoellipPSFGuesser
from ngmix.em import EMFitter
from ngmix.admom import AdmomFitter
from ngmix.fitting import CoellipFitter, Fitter
from ._sims import get_ngauss_obs, get_psf_obs, get_model_obs


@pytest.mark.parametrize('guess_from_moms', [False, True])
@pytest.mark.parametrize('ngauss', [1, 2, 3, 4, 5])
def test_em_psf_runner_smoke(ngauss, guess_from_moms):
    """
    Smoke test a PSFRunner running the EM fitter
    """

    rng = np.random.RandomState(8821)

    data = get_psf_obs(rng=rng)

    obs = data['obs']

    guesser = GMixPSFGuesser(
        rng=rng,
        ngauss=ngauss,
        guess_from_moms=guess_from_moms,
    )
    # better tolerance needed for this psf fit
    fitter = EMFitter(tol=1.0e-5)

    runner = PSFRunner(
        fitter=fitter,
        guesser=guesser,
        ntry=2,
    )
    res = runner.go(obs=obs)

    assert res['flags'] == 0


@pytest.mark.parametrize('with_psf_obs', [False, True])
@pytest.mark.parametrize('guess_from_moms', [False, True])
def test_em_psf_runner(with_psf_obs, guess_from_moms):
    """
    Test a PSFRunner running the EM fitter

    with_psf_obs means it is an ordinary obs with a psf obs also.
    The code knows to fit the psf obs not the main obs
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
        data = get_psf_obs(rng=rng)

    obs = data['obs']

    guesser = GMixPSFGuesser(
        rng=rng,
        ngauss=3,
        guess_from_moms=guess_from_moms,
    )
    # better tolerance needed for this psf fit
    fitter = EMFitter(tol=1.0e-5)

    runner = PSFRunner(
        fitter=fitter,
        guesser=guesser,
        ntry=2,
    )
    res = runner.go(obs=obs)

    assert res['flags'] == 0

    # check reconstructed image allowing for noise
    imfit = res.make_image()

    if with_psf_obs:
        comp_image = obs.psf.image
    else:
        comp_image = obs.image

    imtol = 0.001 / obs.jacobian.scale**2
    assert np.abs(imfit - comp_image).max() < imtol


@pytest.mark.parametrize('guess_from_moms', [False, True])
@pytest.mark.parametrize('model', ["gauss", "turb"])
def test_simple_psf_runner_smoke(model, guess_from_moms):
    """
    Smoke test a PSFRunner running the simple fitter
    """

    rng = np.random.RandomState(3893)

    # make a complex psf so the fitting with high ngauss doesn't become too
    # degenerate
    data = get_psf_obs(model=model, rng=rng)

    guesser = SimplePSFGuesser(
        rng=rng,
        guess_from_moms=guess_from_moms,
    )
    fitter = Fitter(model=model)

    runner = PSFRunner(
        fitter=fitter,
        guesser=guesser,
        ntry=2,
    )
    res = runner.go(obs=data['obs'])

    assert res['flags'] == 0


@pytest.mark.parametrize('guess_from_moms', [False, True])
@pytest.mark.parametrize('model', ["gauss", "turb"])
def test_simple_psf_runner(model, guess_from_moms):
    """
    Smoke test a PSFRunner running the simple fitter
    """

    rng = np.random.RandomState(3893)

    # make a complex psf so the fitting with high ngauss doesn't become too
    # degenerate
    data = get_psf_obs(model=model, rng=rng)

    guesser = SimplePSFGuesser(
        rng=rng,
        guess_from_moms=guess_from_moms,
    )
    fitter = Fitter(model=model)

    runner = PSFRunner(
        fitter=fitter,
        guesser=guesser,
        ntry=2,
    )
    res = runner.go(obs=data['obs'])

    assert res['flags'] == 0

    # check reconstructed image allowing for noise
    imfit = res.make_image()

    obs = data['obs']

    imtol = 0.001 / obs.jacobian.scale**2
    assert np.abs(imfit - obs.image).max() < imtol


@pytest.mark.parametrize('guess_from_moms', [False, True])
@pytest.mark.parametrize('ngauss', [1, 2, 3, 4, 5])
def test_coellip_psf_runner_smoke(ngauss, guess_from_moms):
    """
    Smoke test a PSFRunner running the coelliptical fitter
    """

    rng = np.random.RandomState(9321)

    # make a complex psf so the fitting with high ngauss doesn't become too
    # degenerate
    data = get_psf_obs(rng=rng)
    data2 = get_psf_obs(T=1.0, rng=rng)
    data3 = get_psf_obs(T=2.0, rng=rng)

    combined_im = (
        data['obs'].image + data2['obs'].image + data3['obs'].image
    )
    obs = data['obs']
    obs.image = combined_im

    guesser = CoellipPSFGuesser(
        rng=rng,
        ngauss=ngauss,
        guess_from_moms=guess_from_moms,
    )
    fitter = CoellipFitter(ngauss=ngauss)

    runner = PSFRunner(
        fitter=fitter,
        guesser=guesser,
        ntry=2,
    )
    res = runner.go(obs=obs)

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
        data = get_psf_obs(rng=rng)

    obs = data['obs']

    ngauss = 3
    guesser = CoellipPSFGuesser(
        rng=rng,
        ngauss=ngauss,
        guess_from_moms=guess_from_moms,
    )
    # better tolerance needed for this psf fit
    fitter = CoellipFitter(ngauss=ngauss)

    runner = PSFRunner(
        fitter=fitter,
        guesser=guesser,
        ntry=4,
    )
    res = runner.go(obs=obs)

    assert res['flags'] == 0

    # check reconstructed image allowing for noise
    imfit = res.make_image()

    if with_psf_obs:
        comp_image = obs.psf.image
    else:
        comp_image = obs.image

    imtol = 0.001 / obs.jacobian.scale**2
    assert np.abs(imfit - comp_image).max() < imtol


@pytest.mark.parametrize('guess_from_moms', [False, True])
@pytest.mark.parametrize('ngauss', [1, 2, 3, 4, 5])
def test_admom_psf_runner_smoke(ngauss, guess_from_moms):
    """
    Smoke test a PSFRunner running the Admom fitter
    """

    rng = np.random.RandomState(5661)

    data = get_psf_obs(rng=rng, model='gauss')

    obs = data['obs']

    guesser = GMixPSFGuesser(
        rng=rng,
        ngauss=1,
        guess_from_moms=guess_from_moms,
    )
    fitter = AdmomFitter()

    runner = PSFRunner(
        fitter=fitter,
        guesser=guesser,
        ntry=2,
    )
    res = runner.go(obs=obs)

    assert res['flags'] == 0


@pytest.mark.parametrize('with_psf_obs', [False, True])
@pytest.mark.parametrize('guess_from_moms', [False, True])
def test_admom_psf_runner(with_psf_obs, guess_from_moms):
    """
    Test a PSFRunner running the EM fitter

    with_psf_obs means it is an ordinary obs with a psf obs also.
    The code knows to fit the psf obs not the main obs
    """

    rng = np.random.RandomState(8821)

    if with_psf_obs:
        data = get_ngauss_obs(
            rng=rng,
            ngauss=1,
            noise=0.0,
            with_psf=True,
            psf_model='gauss',
        )
    else:
        data = get_psf_obs(rng=rng, model='gauss')

    obs = data['obs']

    guesser = GMixPSFGuesser(
        rng=rng,
        ngauss=1,
        guess_from_moms=guess_from_moms,
    )
    fitter = AdmomFitter()

    runner = PSFRunner(
        fitter=fitter,
        guesser=guesser,
        ntry=2,
    )
    res = runner.go(obs=obs)

    assert res['flags'] == 0

    # check reconstructed image allowing for noise
    imfit = res.make_image()

    if with_psf_obs:
        comp_image = obs.psf.image
    else:
        comp_image = obs.image

    imtol = 0.001 / obs.jacobian.scale**2
    assert np.abs(imfit - comp_image).max() < imtol


@pytest.mark.parametrize('nband', [None, 3])
@pytest.mark.parametrize('nepoch', [None, 1, 3])
@pytest.mark.parametrize('set_result', [True, False])
def test_gaussmom_psf_runner(nband, nepoch, set_result):
    """
    Test a PSFRunner using GaussMom
    """

    rng = np.random.RandomState(8821)

    data = get_model_obs(
        rng=rng,
        model='gauss',
        noise=0.1,
        nband=nband,
        nepoch=nepoch,
    )

    obs = data['obs']

    fitter = ngmix.gaussmom.GaussMom(fwhm=1.2)

    runner = PSFRunner(fitter=fitter, set_result=set_result)
    res = runner.go(obs=obs)

    if nband is not None:
        assert isinstance(res, list)
        assert isinstance(res[0], list)

        for tobslist in obs:
            for tobs in tobslist:
                if set_result:
                    assert 'result' in tobs.psf.meta
                else:
                    assert 'result' not in tobs.psf.meta
    elif nepoch is not None:
        assert isinstance(res, list)
        for tobs in obs:
            if set_result:
                assert 'result' in tobs.psf.meta
            else:
                assert 'result' not in tobs.psf.meta
    else:
        assert hasattr(res, 'keys')
        if set_result:
            assert 'result' in obs.psf.meta
        else:
            assert 'result' not in obs.psf.meta
