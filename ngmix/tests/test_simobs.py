"""
just test moment errors
"""
import pytest
import numpy as np
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
def test_metacal_bootstrap_max_smoke(
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
