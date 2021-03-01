import pytest
import numpy as np

from ngmix.fitting import TemplateFluxFitter
from ngmix.galsimfit import GalsimTemplateFluxFitter
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

    if nband is None:
        fitter.go(obs=data['obs'])
        res = fitter.get_result()
        assert res['flags'] == 0
        assert (res['flux'] - data['pars'][5]) < NSIG*res['flux_err']
    else:
        for iband, obslist in enumerate(data['obs']):
            fitter.go(obs=obslist)
            res = fitter.get_result()
            assert res['flags'] == 0
            assert (res['flux'] - data['pars'][5+iband]) < NSIG*res['flux_err']


@pytest.mark.parametrize('noise', [1.0, 5.0, 100.0])
@pytest.mark.parametrize('nband', [None, 1, 2])
def test_template_psf_flux_galsim(noise, nband):
    """
    see if we can recover the psf flux within errors
    """

    rng = np.random.RandomState(42587)
    data = get_model_obs(
        model='gauss',  # it has T=0 for a star
        rng=rng, noise=noise, star=True,
        set_psf_gmix=True, nepoch=10, nband=nband,
    )

    fitter = GalsimTemplateFluxFitter()


    if nband is None:
        fitter.go(obs=data['obs'])
        scale = data['obs'][0].jacobian.scale
        res = fitter.get_result()
        assert res['flags'] == 0
        assert (res['flux']*scale**2 - data['pars'][5]) < NSIG*res['flux_err']
    else:
        for iband, obslist in enumerate(data['obs']):
            fitter.go(obs=obslist)
            scale = data['obs'][0][0].jacobian.scale
            res = fitter.get_result()
            assert res['flags'] == 0
            assert (res['flux']**scale - data['pars'][5+iband]) < NSIG*res['flux_err']
