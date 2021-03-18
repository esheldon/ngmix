import pytest
import numpy as np

from ngmix.fitting import TemplateFluxFitter
from ngmix.galsimfit import GalsimTemplateFluxFitter
from ._sims import get_model_obs

NSIG = 4


@pytest.mark.parametrize('noise', [1.0, 5.0, 100.0])
@pytest.mark.parametrize('nband', [None, 1, 2])
@pytest.mark.parametrize('do_psf', [True, False])
def test_template_flux(noise, nband, do_psf):
    """
    see if we can recover the psf flux within errors
    """

    rng = np.random.RandomState(8312)
    if do_psf:
        set_psf_templates = True
        set_templates = False
    else:
        set_templates = True
        set_psf_templates = False

    data = get_model_obs(
        model='gauss',  # it has T=0 for a star
        rng=rng, noise=noise, star=True,
        set_templates=set_templates,
        set_psf_templates=set_psf_templates,
        nepoch=10,
        nband=nband,
    )

    fitter = TemplateFluxFitter(do_psf=do_psf)

    if nband is None:
        res = fitter.go(obs=data['obs'])
        assert res['flags'] == 0
        assert (res['flux'] - data['pars'][5]) < NSIG*res['flux_err']
    else:
        for iband, obslist in enumerate(data['obs']):
            res = fitter.go(obs=obslist)
            assert res['flags'] == 0
            assert (res['flux'] - data['pars'][5+iband]) < NSIG*res['flux_err']


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
        res = fitter.go(obs=data['obs'])
        assert res['flags'] == 0
        assert (res['flux'] - data['pars'][5]) < NSIG*res['flux_err']
    else:
        for iband, obslist in enumerate(data['obs']):
            res = fitter.go(obs=obslist)
            assert res['flags'] == 0
            assert (res['flux'] - data['pars'][5+iband]) < NSIG*res['flux_err']


@pytest.mark.parametrize('noise', [1.0, 5.0, 100.0])
@pytest.mark.parametrize('nband', [None, 1, 2])
@pytest.mark.parametrize('nepoch', [None, 10])
def test_template_psf_flux_galsim(noise, nband, nepoch):
    """
    see if we can recover the psf flux within errors
    """

    rng = np.random.RandomState(42587)
    data = get_model_obs(
        model='gauss',  # it has T=0 for a star
        rng=rng, noise=noise, star=True,
        set_psf_gmix=True, nepoch=nepoch, nband=nband,
    )

    fitter = GalsimTemplateFluxFitter()

    if nband is None:
        res = fitter.go(obs=data['obs'])
        assert res['flags'] == 0
        assert (res['flux'] - data['pars'][5]) < NSIG*res['flux_err']
    else:
        for iband, obslist in enumerate(data['obs']):
            res = fitter.go(obs=obslist)
            assert res['flags'] == 0
            assert (res['flux'] - data['pars'][5+iband]) < NSIG*res['flux_err']


def test_template_flux_errors():
    """
    see if we can recover the psf flux within errors
    """

    fitter = TemplateFluxFitter()

    with pytest.raises(ValueError):
        fitter.go(obs=None)

    rng = np.random.RandomState(42587)
    data = get_model_obs(model='gauss', rng=rng)

    # no gmix or template set
    with pytest.raises(ValueError):
        fitter.go(obs=data['obs'])
