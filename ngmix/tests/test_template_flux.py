import pytest
import numpy as np

from ngmix.fitting import TemplateFluxFitter
from ._sims import get_model_obs

NSIG = 4


@pytest.mark.parametrize('noise', [1.0, 5.0, 100.0])
@pytest.mark.parametrize('nband', [None, 1, 2])
def test_template_psf_flux(noise, nband):
    """
    see if we can recover the psf flux within errors
    """

    rng = np.random.RandomState(42587)
    data = get_model_obs(
        model='gauss',  # it has T=0 for a star
        rng=rng, noise=noise, star=True,
        set_psf_gmix=True, nepoch=10, nband=nband,
    )

    fitter = TemplateFluxFitter(do_psf=True)

    nband_use = nband if nband is not None else 1

    if nband is None:
        fitter.go(obs=data['obslist'])
        res = fitter.get_result()
        assert res['flags'] == 0
        assert (res['flux'] - data['pars'][5]) < NSIG*res['flux_err']
    else:
        for iband, obslist in enumerate(data['mbobs']):
            fitter.go(obs=obslist)
            res = fitter.get_result()
            assert res['flags'] == 0
            assert (res['flux'] - data['pars'][5+iband]) < NSIG*res['flux_err']
