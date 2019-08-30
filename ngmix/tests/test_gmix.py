import numpy as np
import galsim

import pytest

from ngmix.gmix import make_gmix_model
from ngmix.shape import g1g2_to_e1e2
from ngmix.shape import Shape
import ngmix.moments
from ngmix import DiagonalJacobian, Observation


@pytest.mark.parametrize('model', ['gauss', 'exp', 'dev', 'turb'])
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
    elif model == 'turb':
        assert len(gm) == 3
    else:
        raise AssertionError("len(gm) was not checked!")

    assert np.allclose(gm.get_cen(), np.array([row, col]))
    assert np.allclose(gm.get_flux(), flux)
    assert np.allclose(gm.get_T(), T)
    assert np.allclose(gm.get_sigma(), sigma)
    assert np.allclose(gm.get_e1e2T(), [e1, e2, T])
    assert np.allclose(gm.get_g1g2T(), [g1, g2, T])
    assert np.allclose(gm.get_e1e2sigma(), [e1, e2, sigma])
    assert np.allclose(gm.get_g1g2sigma(), [g1, g2, sigma])

    # we are going to simply check a few things here since the tests
    # above use all of this data too
    data = gm.get_data()
    full_pars = gm.get_full_pars()
    if model == 'gauss':
        assert len(data) == 1
        assert len(full_pars) == 6
    elif model == 'exp':
        assert len(data) == 6
        assert len(full_pars) == 6*6
    elif model == 'dev':
        assert len(data) == 10
        assert len(full_pars) == 10*6
    elif model == 'turb':
        assert len(data) == 3
        assert len(full_pars) == 3*6
    else:
        raise AssertionError("len(gm) was not checked!")

    assert np.allclose(np.sum(full_pars[::6]), flux)
    assert np.allclose(
        np.sum(full_pars[::6] * full_pars[1::6]) / np.sum(full_pars[::6]),
        row)
    assert np.allclose(
        np.sum(full_pars[::6] * full_pars[2::6]) / np.sum(full_pars[::6]),
        col)
    assert np.allclose(np.sum(data['p']), flux)
    assert np.allclose(
        np.sum(data['p'] * data['row']) / np.sum(data['p']), row)
    assert np.allclose(
        np.sum(data['p'] * data['col']) / np.sum(data['p']), col)


@pytest.mark.parametrize('model', ['gauss', 'exp', 'dev', 'turb'])
def test_gmix_set_cen(model):
    row, col, g1, g2, T, flux = -0.5, -0.4, -0.2, 0.3, 0.8, 56
    pars = np.array([row, col, g1, g2, T, flux])
    gm = make_gmix_model(pars, model)

    assert np.allclose(gm.get_cen(), [row, col])
    gm.set_cen(0.1, -0.1)
    assert np.allclose(gm.get_cen(), [0.1, -0.1])
    assert np.allclose(gm.get_g1g2T(), [g1, g2, T])
    assert np.allclose(gm.get_flux(), flux)


@pytest.mark.parametrize('model', ['gauss', 'exp', 'dev', 'turb'])
def test_gmix_set_flux(model):
    row, col, g1, g2, T, flux = -0.5, -0.4, -0.2, 0.3, 0.8, 56
    pars = np.array([row, col, g1, g2, T, flux])
    gm = make_gmix_model(pars, model)

    assert np.allclose(gm.get_flux(), flux)
    gm.set_flux(99.5)
    assert np.allclose(gm.get_g1g2T(), [g1, g2, T])
    assert np.allclose(gm.get_cen(), [row, col])
    assert np.allclose(gm.get_flux(), 99.5)


@pytest.mark.parametrize('model', ['gauss', 'exp', 'dev', 'turb'])
def test_gmix_fill_pars(model):
    row, col, g1, g2, T, flux = -0.5, -0.4, -0.2, 0.3, 0.8, 56
    pars = np.array([row, col, g1, g2, T, flux])
    gm = make_gmix_model(pars, model)

    assert np.allclose(gm.get_flux(), flux)
    assert np.allclose(gm.get_g1g2T(), [g1, g2, T])
    assert np.allclose(gm.get_cen(), [row, col])

    new_pars = [-0.1, 0.1, 0.5, -0.6, 0.7, 99.5]
    gm.fill(new_pars)

    assert np.allclose(gm.get_flux(), 99.5)
    assert np.allclose(gm.get_g1g2T(), [0.5, -0.6, 0.7])
    assert np.allclose(gm.get_cen(), [-0.1, 0.1])


@pytest.mark.parametrize('model', ['gauss', 'exp', 'dev', 'turb'])
def test_gmix_reset(model):
    row, col, g1, g2, T, flux = -0.5, -0.4, -0.2, 0.3, 0.8, 56
    pars = np.array([row, col, g1, g2, T, flux])
    gm = make_gmix_model(pars, model)

    assert np.allclose(gm.get_flux(), flux)
    assert np.allclose(gm.get_g1g2T(), [g1, g2, T])
    assert np.allclose(gm.get_cen(), [row, col])
    gm.reset()

    assert np.allclose(gm.get_full_pars(), 0)


@pytest.mark.parametrize('model', ['gauss', 'exp', 'dev', 'turb'])
def test_gmix_copy(model):
    row, col, g1, g2, T, flux = -0.5, -0.4, -0.2, 0.3, 0.8, 56
    pars = np.array([row, col, g1, g2, T, flux])
    gm = make_gmix_model(pars, model)

    # we make one, copy it, adjust the copy and then test again
    assert np.allclose(gm.get_flux(), flux)
    assert np.allclose(gm.get_g1g2T(), [g1, g2, T])
    assert np.allclose(gm.get_cen(), [row, col])

    new_gm = gm.copy()
    new_gm.reset()

    assert np.allclose(new_gm.get_full_pars(), 0)
    assert np.allclose(gm.get_flux(), flux)
    assert np.allclose(gm.get_g1g2T(), [g1, g2, T])
    assert np.allclose(gm.get_cen(), [row, col])


@pytest.mark.parametrize('model', ['gauss', 'exp', 'dev', 'turb'])
def test_gmix_get_sheared(model):
    row, col, g1, g2, T, flux = -0.5, -0.4, 0, 0, 0.8, 56
    pars = np.array([row, col, g1, g2, T, flux])
    gm = make_gmix_model(pars, model)

    assert np.allclose(gm.get_flux(), flux)
    assert np.allclose(gm.get_g1g2T(), [g1, g2, T])
    assert np.allclose(gm.get_cen(), [row, col])

    new_gm = gm.get_sheared(-0.8, 0.01)

    assert np.allclose(gm.get_flux(), flux)
    assert np.allclose(gm.get_g1g2T(), [g1, g2, T])
    assert np.allclose(gm.get_cen(), [row, col])
    assert np.allclose(new_gm.get_flux(), flux)
    assert np.allclose(new_gm.get_cen(), [row, col])
    assert np.allclose(new_gm.get_g1g2T()[0:2], [-0.8, 0.01])

    shear = Shape(g1=-0.8, g2=0.01)
    new_gm = gm.get_sheared(shear)

    assert np.allclose(gm.get_flux(), flux)
    assert np.allclose(gm.get_g1g2T(), [g1, g2, T])
    assert np.allclose(gm.get_cen(), [row, col])
    assert np.allclose(new_gm.get_flux(), flux)
    assert np.allclose(new_gm.get_cen(), [row, col])
    assert np.allclose(new_gm.get_g1g2T()[0:2], [-0.8, 0.01])


@pytest.mark.parametrize('model', ['gauss', 'exp', 'dev', 'turb'])
@pytest.mark.parametrize('preserve_size', [False, True])
def test_gmix_make_round(model, preserve_size):
    row, col, g1, g2, T, flux = -0.5, -0.4, -0.2, 0.3, 0.8, 56
    pars = np.array([row, col, g1, g2, T, flux])
    gm = make_gmix_model(pars, model)

    irr, irc, icc = ngmix.moments.e2mom(*gm.get_e1e2T())
    max_eig = np.linalg.eigvals(np.array([[irr, irc], [irc, icc]])).max()

    # we make one, copy it, adjust the copy and then test again
    assert np.allclose(gm.get_flux(), flux)
    assert np.allclose(gm.get_g1g2T(), [g1, g2, T])
    assert np.allclose(gm.get_cen(), [row, col])

    rgm = gm.make_round(preserve_size=preserve_size)
    assert np.allclose(gm.get_flux(), flux)
    assert np.allclose(gm.get_g1g2T(), [g1, g2, T])
    assert np.allclose(gm.get_cen(), [row, col])

    assert np.allclose(rgm.get_cen(), [row, col])
    assert np.allclose(rgm.get_flux(), flux)
    assert np.allclose(rgm.get_g1g2T()[0:2], [0, 0])
    if preserve_size:
        half_T = rgm.get_T()/2 + 1e-10  # small constant for float stuff
        assert half_T >= max_eig


@pytest.mark.parametrize('model', ['gauss', 'exp', 'dev', 'turb'])
def test_gmix_make_galsim_make_image(model):
    row, col, g1, g2, T, flux = -0.5, -0.4, -0.2, 0.3, 0.8, 56
    pars = np.array([row, col, g1, g2, T, flux])
    gm = make_gmix_model(pars, model)
    gs_obj = gm.make_galsim_object()

    assert np.allclose(gs_obj.flux, 56)
    assert np.allclose(gs_obj.centroid.x, -0.4)
    assert np.allclose(gs_obj.centroid.y, -0.5)

    if isinstance(gs_obj, galsim.Sum):
        obj_list = gs_obj.obj_list
        data = gm.get_data()
        assert len(obj_list) == len(data)
        for i in range(len(obj_list)):
            assert np.allclose(data['p'][i], obj_list[i].flux)
            assert np.allclose(obj_list[i].centroid.x, -0.4)
            assert np.allclose(obj_list[i].centroid.y, -0.5)

    # test making an image
    # only works for scale=1 because ngmix uses pixel and galsim is in
    # world coords
    dims = (213, 113)
    jac = DiagonalJacobian(row=106, col=56, scale=1)
    gs_jac = jac.get_galsim_wcs()
    im = gm.make_image(dims, jacobian=jac)
    gs_im = gs_obj.drawImage(
        nx=dims[1], ny=dims[0], wcs=gs_jac, method='no_pixel').array
    assert np.allclose(im, gs_im)


@pytest.mark.parametrize('model', ['gauss', 'exp', 'dev', 'turb'])
def test_gmix_convolve(model):
    row, col, g1, g2, T, flux = -0.5, -0.4, -0.2, 0.3, 2.8, 1
    pars = np.array([row, col, g1, g2, T, flux])
    gm = make_gmix_model(pars, model)

    kernel = make_gmix_model([0, 0, 0.1, -0.1, 2.5, 1], 'turb')

    gs_obj = galsim.Convolve(
        [gm.make_galsim_object(), kernel.make_galsim_object()])
    cgm = gm.convolve(kernel)

    assert len(cgm) == 3*len(gm)

    # test making an image
    # only works for scale=1 because ngmix uses pixel and galsim is in
    # world coords
    dims = (13, 11)
    jac = DiagonalJacobian(row=6, col=5, scale=1)
    gs_jac = jac.get_galsim_wcs()
    im = cgm.make_image(dims, jacobian=jac)
    gs_im = gs_obj.drawImage(
        nx=dims[1], ny=dims[0], wcs=gs_jac, method='no_pixel').array

    # not perfect but close
    assert np.allclose(
        im, gs_im, atol=3e-6 if model == 'dev' else 1e-6, rtol=0)


def test_gmix_convolve_gauss():
    row, col, g1, g2, T, flux = -0.5, -0.4, -0.2, 0.3, 2.8, 5
    pars = np.array([row, col, g1, g2, T, flux])
    gm = make_gmix_model(pars, 'gauss')

    krow, kcol, kg1, kg2, kT, kflux = 0, 1.0, 0.5, 0.4, 1.2, 6
    kpars = np.array([krow, kcol, kg1, kg2, kT, kflux])
    kernel = make_gmix_model(kpars, 'gauss')
    cgm = gm.convolve(kernel)

    mom = ngmix.moments.g2mom(g1, g2, T)
    kmom = ngmix.moments.g2mom(kg1, kg2, kT)
    cmom = (
        mom[0] + kmom[0],
        mom[1] + kmom[1],
        mom[2] + kmom[2])
    cg1, cg2, cT = ngmix.moments.mom2g(*cmom)

    # convolve ignores the PSF center and flux
    assert np.allclose(cgm.get_cen()[0], row)
    assert np.allclose(cgm.get_cen()[1], col)
    assert np.allclose(cgm.get_flux(), flux)
    assert np.allclose(cgm.get_g1g2T(), [cg1, cg2, cT])


@pytest.mark.parametrize('start', [0, 13])
def test_gmix_loglike_fdiff(start):
    row, col, g1, g2, T, flux = -0.5, -0.4, 0, 0, 2.8, 5
    sigma = np.sqrt(T/2)
    pars = np.array([row, col, g1, g2, T, flux])
    gm = make_gmix_model(pars, 'gauss')

    rng = np.random.RandomState(seed=10)
    dims = (13, 15)
    obs = Observation(
        image=rng.normal(size=dims),
        weight=np.exp(rng.normal(size=dims)),
        jacobian=DiagonalJacobian(row=6.5, col=7.1, scale=0.25))
    fdiff = np.zeros(dims[0] * dims[1] + start, dtype=np.float64)
    gm.fill_fdiff(obs, fdiff, start=start)

    loglike = 0
    loc = start
    for r in range(dims[0]):
        for c in range(dims[1]):
            v, u = obs.jacobian(r, c)
            chi2 = ((u - col)**2 + (v - row)**2)/sigma**2
            model = (
                5 * np.exp(-0.5 * chi2) / 2.0 / np.pi / sigma**2)
            _fdiff = (model - obs.image[r, c]) * np.sqrt(obs.weight[r, c])
            assert np.allclose(_fdiff, fdiff[loc])

            loglike += _fdiff**2

            loc += 1

    assert np.allclose(loglike * -0.5, gm.get_loglike(obs))
