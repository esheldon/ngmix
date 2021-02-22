import pytest
import numpy as np

from ngmix.guessers import TFluxGuesser, TPSFFluxGuesser
from ._sims import get_model_obs


@pytest.mark.parametrize('nband', [None, 1, 2])
def test_template_T_flux_guesser_smoke(nband):
    """
    see if we can recover the psf flux within errors
    """

    rng = np.random.RandomState(587)
    data = get_model_obs(
        model='gauss',  # it has T=0 for a star
        rng=rng, star=True,
        set_psf_gmix=True, nepoch=10,
        nband=nband,
    )

    nband_use = nband if nband is not None else 1
    T_center = 0.001
    flux_center = [1.0]*nband_use

    guesser = TFluxGuesser(rng=rng, T=T_center, flux=flux_center)

    if nband is None:
        obs_use = data['obslist']
    else:
        obs_use = data['mbobs']

    guess = guesser(obs=obs_use)
    nband_use = nband if nband is not None else 1
    assert guess.size == 5 + nband_use


@pytest.mark.parametrize('nband', [None, 1, 2])
def test_template_T_psf_flux_guesser_smoke(nband):
    """
    see if we can recover the psf flux within errors
    """

    rng = np.random.RandomState(458)
    data = get_model_obs(
        model='gauss',  # it has T=0 for a star
        rng=rng, star=True,
        set_psf_gmix=True, nepoch=10,
        nband=nband,
    )

    T_center = 0.001
    guesser = TPSFFluxGuesser(rng=rng, T=T_center)

    if nband is None:
        obs_use = data['obslist']
    else:
        obs_use = data['mbobs']

    guess = guesser(obs=obs_use)
    nband_use = nband if nband is not None else 1
    assert guess.size == 5 + nband_use
