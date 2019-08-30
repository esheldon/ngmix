import numpy as np
import galsim

import pytest

from ngmix.gmix import make_gmix_model, GMix
from ngmix.shape import g1g2_to_e1e2

# Things to test:
#     def __init__(self, ngauss=None, pars=None):
#     def set_cen(self, row, col):
#     def set_flux(self, psum):
#     def get_gaussap_flux(self, fwhm=None, sigma=None, T=None):
#     def set_norms(self):
#     def set_norms_if_needed(self):
#     def fill(self, pars):
#     def copy(self):
#     def get_sheared(self, s1, s2=None):
#     def convolve(self, psf):
#     def make_image(self, dims, jacobian=None, fast_exp=False):
#     def make_round(self, preserve_size=False):
#     def fill_fdiff(self, obs, fdiff, start=0):
#     def get_weighted_moments(self, obs, maxrad):
#     def get_weighted_sums(self, obs, maxrad, res=None):
#     def get_model_s2n(self, obs):
#     def get_loglike(self, obs, more=False):
#     def reset(self):
#     def make_galsim_object(self, Tmin=1e-6, gsparams=None):


@pytest.mark.parametrize('model', ['gauss', 'exp', 'dev'])
@pytest.mark.parametrize('row', [-0.5, 0, 0.4])
@pytest.mark.parametrize('col', [-0.4, 0, 0.5])
@pytest.mark.parametrize('flux', [56])
@pytest.mark.parametrize('g1', [-0.2, 0, 0.3])
@pytest.mark.parametrize('g2', [-0.3, 0, 0.2])
@pytest.mark.parametrize('T', [0.3])
def test_gmix_get_pars(model, row, col, flux, g1, g2, T):
    e1, e2 = g1g2_to_e1e2(g1, g2)
    sigma = np.sqrt(T/2)
    pars = np.array([row, col, g1, g2, T, flux])
    gm = make_gmix_model(pars, model)
    if model == 'gauss':
        assert len(gm) == 1
    elif model == 'exp':
        assert len(gm) == 6
    elif model == 'dev':
        assert len(gm) == 10

    assert np.allclose(gm.get_cen(), np.array([row, col]))
    assert np.allclose(gm.get_flux(), flux)
    assert np.allclose(gm.get_T(), T)
    assert np.allclose(gm.get_sigma(), sigma)
    assert np.allclose(gm.get_e1e2T(), [e1, e2, T])
    assert np.allclose(gm.get_g1g2T(), [g1, g2, T])
    assert np.allclose(gm.get_e1e2sigma(), [e1, e2, sigma])
    assert np.allclose(gm.get_g1g2sigma(), [g1, g2, sigma])
#     def get_data(self):
#     def get_full_pars(self):


def test_gmix_set_pars(model, row, col, flux, g1, g2, T):
    assert False


def test_gmix_reset(model, row, col, flux, g1, g2, T):
    assert False


def test_gmix_make_image(model, gs_obj, g1, g2, T, wcs_g1, wcs_g2):
    assert False


def test_gmix_convolve(model, row, col, flux, g1, g2, T):
    assert False


def test_gmix_loglike_fdiff(model, row, col, flux, g1, g2, T):
    assert False


def test_gmix_make_round(model, row, col, flux, g1, g2, T):
    assert False


def test_gmix_get_sheared(model, row, col, flux, g1, g2, T):
    assert False


def test_gmix_make_galsim(model, gs_obj, g1, g2, T, wcs_g1, wcs_g2):
    assert False
