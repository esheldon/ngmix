import pytest
import numpy as np

from ngmix.fitting import TemplateFluxFitter
from ._sims import get_model_obs


@pytest.mark.parametrize('noise', [1.0, 5.0, 100.0])
def test_template_psf_flux(noise):
    """
    see if we can recover the psf flux within errors
    """

    rng = np.random.RandomState(42587)
    data = get_model_obs(
        model='gauss',  # it has T=0 for a star
        rng=rng, noise=noise, star=True,
        set_psf_gmix=True, nepoch=10,
    )

    obslist = data['obslist']

    fitter = TemplateFluxFitter(do_psf=True)
    fitter.go(obs=obslist)

    res = fitter.get_result()
    assert res['flags'] == 0

    flux = res['flux']
    flux_err = res['flux_err']
    assert abs(flux - data['pars'][5]) < 5*flux_err