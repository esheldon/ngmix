"""
just test moment errors
"""
import pytest
import numpy as np
import ngmix
from ngmix.simobs import simulate_obs
from ._sims import get_model_obs

FRAC_TOL = 5.0e-4


@pytest.mark.parametrize('with_gmix', [True, False])
@pytest.mark.parametrize('add_noise', [True, False])
@pytest.mark.parametrize('add_all', [True, False])
@pytest.mark.parametrize('noise_factor', [None, 1.2])
@pytest.mark.parametrize('use_raw_weight', [True, False])
@pytest.mark.parametrize('convolve_psf', [True, False])
@pytest.mark.parametrize('nband', [None, 2])
@pytest.mark.parametrize('nepoch', [None, 2])
def test_simobs_smoke(
    with_gmix, add_noise, add_all,
    noise_factor, use_raw_weight, convolve_psf,
    nband, nepoch,
):
    """
    Smoke test a Runner running the LM fitter
    """

    rng = np.random.RandomState(2830)

    data = get_model_obs(
        rng=rng,
        model='gauss',
        noise=0.1,
        nepoch=nepoch,
        nband=nband,
        set_psf_gmix=convolve_psf,
    )
    obs = data['obs']
    if add_all and nband is None and nepoch is None:
        weight = obs.weight.copy()
        weight[0, 0] = 0.0
        obs.weight = weight

    if with_gmix:
        if nband is not None:
            gmix = [data['gmix'].copy() for i in range(nband)]
        else:
            gmix = data['gmix'].copy()
    else:
        gmix = None

    if nband is None and nepoch is None and use_raw_weight:
        obs.weight_raw = obs.weight.copy()
        if not convolve_psf:
            obs.set_psf(None)

    _ = simulate_obs(
        gmix=gmix,
        obs=obs,
        add_noise=add_noise,
        rng=rng,
        add_all=add_all,
        noise_factor=noise_factor,
        use_raw_weight=use_raw_weight,
        convolve_psf=convolve_psf,
    )


def test_simobs_errors():
    rng = np.random.RandomState(0)

    with pytest.raises(ValueError):
        ngmix.simobs.simulate_obs(obs=None, gmix=None)

    with pytest.raises(ValueError):
        ngmix.simobs.simulate_obs(obs=None, gmix=3)

    nband = 3
    data = get_model_obs(
        rng=rng,
        model='gauss',
        noise=0.1,
        nepoch=2,
        nband=nband,
        set_psf_gmix=True,
    )
    obs = data['obs']

    with pytest.raises(ValueError):
        ngmix.simobs.simulate_obs(obs=obs, gmix=None)

    with pytest.raises(ValueError):
        ngmix.simobs.simulate_obs(obs=obs, gmix=3)

    with pytest.raises(ValueError):
        ngmix.simobs.simulate_obs(obs=obs, gmix=[None])

    gm = ngmix.GMixModel([0, 0, 0, 0, 4, 1], "gauss")
    with pytest.raises(ValueError):
        ngmix.simobs.simulate_obs(obs=obs, gmix=[gm]*(nband+1))

    obs[0][0].psf.set_gmix(None)
    with pytest.raises(RuntimeError):
        ngmix.simobs.simulate_obs(obs=obs, gmix=[gm]*nband)

    obs[0][0].set_psf(None)
    with pytest.raises(RuntimeError):
        ngmix.simobs.simulate_obs(obs=obs, gmix=[gm]*nband)
