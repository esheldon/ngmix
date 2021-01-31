import pytest
import numpy as np
from ngmix.runners import PSFRunner
from ngmix.guessers import GuesserEMPSF
from ngmix.em import GMixEM
from ._sims import get_obs, get_psf_obs


@pytest.mark.parametrize('with_psf_obs', [False, True])
def test_em_psf_runner(with_psf_obs):
    """
    see if we can recover the input with and without noise to high precision
    even with a bad guess

    Use ngmix to make the image to make sure there are
    no pixelization effects
    """

    rng = np.random.RandomState(8821)

    if with_psf_obs:
        data = get_obs(
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
