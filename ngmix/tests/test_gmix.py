import copy
import numpy as np
import galsim

import pytest

import ngmix
from ngmix.gmix import make_gmix_model
from ngmix.shape import g1g2_to_e1e2
from ngmix.shape import Shape
import ngmix.moments
from ngmix.moments import MOMENTS_NAME_MAP
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

    # this will always get coverage
    assert model in ['gauss', 'exp', 'dev', 'turb']

    if model == 'gauss':
        assert len(gm) == 1
    elif model == 'exp':
        assert len(gm) == 6
    elif model == 'dev':
        assert len(gm) == 10
    elif model == 'turb':
        assert len(gm) == 3

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

    gmfull = make_gmix_model(full_pars, 'full')
    assert np.all(gmfull.get_full_pars() == full_pars)


def test_gmix_s2n_smoke():
    pars = [0, 0, 0, 0, 5, 1]
    gm = ngmix.GMixModel(pars, "gauss")
    im = gm.make_image([10, 10])

    obs = ngmix.Observation(im)

    gm.get_model_s2n(obs)


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


@pytest.mark.parametrize('copy_type', ['copy', 'copy.copy', 'copy.deepcopy'])
@pytest.mark.parametrize('model', ['gauss', 'exp', 'dev', 'turb'])
def test_gmix_copy_and_equals(model, copy_type):
    row, col, g1, g2, T, flux = -0.5, -0.4, -0.2, 0.3, 0.8, 56
    pars = np.array([row, col, g1, g2, T, flux])
    gm = make_gmix_model(pars, model)

    # we make one, copy it, adjust the copy and then test again
    assert np.allclose(gm.get_flux(), flux)
    assert np.allclose(gm.get_g1g2T(), [g1, g2, T])
    assert np.allclose(gm.get_cen(), [row, col])

    if copy_type == 'copy':
        new_gm = gm.copy()
    elif copy_type == 'copy.copy':
        new_gm = copy.copy(gm)
    else:
        new_gm = copy.deepcopy(gm)

    assert new_gm == gm

    new_gm.set_cen(row=3, col=5)
    assert new_gm != gm

    assert np.allclose(gm.get_flux(), flux)
    assert np.allclose(gm.get_g1g2T(), [g1, g2, T])
    assert np.allclose(gm.get_cen(), [row, col])

    new_gm.reset()
    assert np.allclose(new_gm.get_full_pars(), 0)

    with pytest.raises(ValueError):
        gm == 3


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
    """
    test that the approx fdiff calculation agrees well enough with that
    calculated with the exp() value

    restrict comparisons to the valid range, which is +/- 5 sigma
    """

    rtol = 4.0e-5

    row, col, g1, g2, T, flux = -0.5, -0.4, 0, 0, 2.8, 5
    sigma = np.sqrt(T/2)
    pars = np.array([row, col, g1, g2, T, flux])
    gm = make_gmix_model(pars, 'gauss')

    jacob = DiagonalJacobian(row=6.5, col=7.1, scale=0.25)
    area = jacob.area

    rng = np.random.RandomState(seed=10)
    dims = (13, 15)
    obs = Observation(
        image=rng.normal(size=dims),
        weight=np.exp(rng.normal(size=dims)),
        jacobian=jacob,
    )
    fdiff = np.zeros(dims[0] * dims[1] + start, dtype=np.float64)
    gm.fill_fdiff(obs, fdiff, start=start)

    cen = (np.array(dims)-1)/2

    row_range = int(cen[0] - 2*sigma), int(cen[0] + 2*sigma)+1
    col_range = int(cen[1] - 2*sigma), int(cen[1] + 2*sigma)+1

    loc = start

    pnorm = 5 / 2.0 / np.pi / sigma**2
    for r in range(row_range[0], row_range[1]):
        for c in range(col_range[0], col_range[1]):
            loc = start + r*dims[1] + c

            v, u = obs.jacobian(r, c)
            chi2 = ((u - col)**2 + (v - row)**2)/sigma**2
            model = pnorm * np.exp(-0.5 * chi2) * area
            _fdiff = (model - obs.image[r, c]) * np.sqrt(obs.weight[r, c])
            print(_fdiff, fdiff[loc])
            assert np.allclose(_fdiff, fdiff[loc], rtol=rtol)


def test_gmix_errors():
    with pytest.raises(ValueError):
        ngmix.GMix()

    with pytest.raises(ValueError):
        ngmix.GMixModel([1]*6, "blah")

    with pytest.raises(ValueError):
        ngmix.GMix(pars=[1]*10)

    with pytest.raises(ValueError):
        ngmix.GMix(ngauss=-1)

    with pytest.raises(ValueError):
        ngmix.GMixModel([1]*10, "gauss")

    pars = [0, 0, 0, 0, 10, 1]
    gm = ngmix.GMixModel(pars, "gauss")
    with pytest.raises(ValueError):
        gm.get_sheared(0.1)

    with pytest.raises(TypeError):
        gm.convolve(3)

    with pytest.raises(ValueError):
        gm.make_image([1])

    obs = ngmix.Observation(np.zeros((10, 10)))
    with pytest.raises(ValueError):
        gm.fill_fdiff(obs, np.zeros((10, 10)), start=100000)

    with pytest.raises(TypeError):
        gm.make_galsim_object(gsparams=3)

    with pytest.raises(ValueError):
        gm = ngmix.GMixCoellip([1]*3)

    with pytest.raises(ValueError):
        ngmix.gmix.get_model_name(-1)

    with pytest.raises(ValueError):
        ngmix.gmix.get_model_num('asdfsa')

    with pytest.raises(ValueError):
        ngmix.gmix.get_model_ngauss('asdfsa')

    with pytest.raises(ValueError):
        ngmix.gmix.get_model_npars('asdfsa')

    with pytest.raises(ValueError):
        tgm = gm.copy()
        tgm._model_name = 'sdfasd'
        tgm._set_fill_func()


def test_extra():
    """
    some extra stuff to get coverage
    """
    pars = [0, 0, 0, 0, 10, 1]
    gm = ngmix.GMixModel(pars, "gauss")

    fwhm = 0.9
    T = ngmix.moments.fwhm_to_T(fwhm)
    sigma = ngmix.moments.fwhm_to_sigma(fwhm)

    apflux = gm.get_gaussap_flux(fwhm=fwhm)
    np.allclose(apflux, gm.get_gaussap_flux(T=T))
    np.allclose(apflux, gm.get_gaussap_flux(sigma=sigma))

    g = ngmix.gmix.GMixCM(0.5, 1.0, pars)
    g.copy()

    npars = ngmix.gmix.get_coellip_npars(5)
    assert npars == 4 + 2*5
    assert ngmix.gmix.get_coellip_ngauss(npars) == (npars - 4) // 2


def test_gmix_lists():
    gml = ngmix.gmix.GMixList()
    gml.append(ngmix.GMix(ngauss=3))
    gml.append(ngmix.GMix(ngauss=3))
    gml[1] = ngmix.GMix(ngauss=3)

    mgml = ngmix.gmix.MultiBandGMixList()
    mgml.append(gml)
    mgml.append(gml)

    mgml[1] = gml


def test_gmix_concat():
    gm1 = ngmix.GMixModel([0, 0, 0.1, -0.2, 4.0, 1.0], 'dev')
    gm2 = ngmix.GMixModel([-1, 3, 0.2, 0.1, 8.0, 2.0], 'exp')

    gmc = ngmix.gmix.gmix_concat([gm1, gm2])
    assert len(gmc) == len(gm1) + len(gm2)

    gm1data = gm1.get_data()
    gm2data = gm2.get_data()
    gmcdata = gmc.get_data()

    for i in range(gm1data.size):
        for name in ('p', 'row', 'col', 'irr', 'irc', 'icc'):
            assert gm1data[name][i] == gmcdata[name][i]

    for i in range(gm2data.size):
        ic = len(gm1) + i
        for name in ('p', 'row', 'col', 'irr', 'irc', 'icc'):
            assert gm2data[name][i] == gmcdata[name][ic]


@pytest.mark.parametrize('do_higher', [False, True])
def test_higher_order_smoke(do_higher):
    fwhm = 0.9
    T = ngmix.moments.fwhm_to_T(fwhm)
    scale = 0.263

    obj = galsim.Gaussian(fwhm=fwhm)
    im = obj.drawImage(
        scale=scale,
    ).array

    cen = (np.array(im.shape) - 1) / 2
    jacobian = ngmix.DiagonalJacobian(row=cen[0], col=cen[1], scale=scale)
    obs = Observation(image=im, jacobian=jacobian)

    wt = ngmix.GMixModel(
        pars=[0, 0, 0, 0, T, 1.0],
        model='gauss',
    )

    res = wt.get_weighted_moments(obs, with_higher_order=do_higher)
    if do_higher:
        assert res['sums'].shape == (17, )
        assert res['sums_cov'].shape == (17, 17)

        assert MOMENTS_NAME_MAP['M00'] == MOMENTS_NAME_MAP['MF']
        assert MOMENTS_NAME_MAP['M10'] == MOMENTS_NAME_MAP['Mu']
        assert MOMENTS_NAME_MAP['M01'] == MOMENTS_NAME_MAP['Mv']
        assert MOMENTS_NAME_MAP['M11'] == MOMENTS_NAME_MAP['MT']
        assert MOMENTS_NAME_MAP['M20'] == MOMENTS_NAME_MAP['M1']
        assert MOMENTS_NAME_MAP['M02'] == MOMENTS_NAME_MAP['M2']

        for name, ind in MOMENTS_NAME_MAP.items():
            # make sure the index is valid
            res['sums'][ind]
    else:
        assert res['sums'].shape == (6, )
        assert res['sums_cov'].shape == (6, 6)


def test_higher_order():
    rng = np.random.RandomState(seed=35)
    fwhm = 0.9
    T = ngmix.moments.fwhm_to_T(fwhm)
    sigma = ngmix.moments.fwhm_to_sigma(fwhm)
    scale = 0.125
    image_size = 107

    ntrial = 100

    rho4s = np.zeros(ntrial)
    rho6s = np.zeros(ntrial)
    rho8s = np.zeros(ntrial)

    rho_M21 = np.zeros(ntrial)
    rho_M12 = np.zeros(ntrial)
    rho_M30 = np.zeros(ntrial)
    rho_M03 = np.zeros(ntrial)

    rho_M31 = np.zeros(ntrial)
    rho_M13 = np.zeros(ntrial)

    rho_M40 = np.zeros(ntrial)
    rho_M14 = np.zeros(ntrial)

    for i in range(ntrial):
        obj = galsim.Gaussian(fwhm=fwhm)

        row_offset, col_offset = rng.uniform(low=-0.5, high=0.5, size=2)

        im = obj.drawImage(
            nx=image_size,
            ny=image_size,
            offset=(col_offset, row_offset),
            scale=scale,
            method='no_pixel',
        ).array

        imcen = (np.array(im.shape) - 1) / 2

        cen = imcen + (row_offset, col_offset)

        jacobian = ngmix.DiagonalJacobian(row=cen[0], col=cen[1], scale=scale)

        obs = Observation(image=im, jacobian=jacobian)

        wt = ngmix.GMixModel(
            pars=[0, 0, 0, 0, T, 1.0],
            model='gauss',
        )

        res = wt.get_weighted_moments(obs, with_higher_order=True)

        f_ind = MOMENTS_NAME_MAP["MF"]

        M22_ind = MOMENTS_NAME_MAP["M22"]
        rho4s[i] = res['sums'][M22_ind] / res['sums'][f_ind] / sigma**4

        M33_ind = MOMENTS_NAME_MAP["M33"]
        rho6s[i] = res['sums'][M33_ind] / res['sums'][f_ind] / sigma**6

        M44_ind = MOMENTS_NAME_MAP["M44"]
        rho8s[i] = res['sums'][M44_ind] / res['sums'][f_ind] / sigma**8

        M21_ind = MOMENTS_NAME_MAP["M21"]
        rho_M21[i] = res['sums'][M21_ind] / res['sums'][f_ind] / sigma**3

        M12_ind = MOMENTS_NAME_MAP["M12"]
        rho_M12[i] = res['sums'][M12_ind] / res['sums'][f_ind] / sigma**3

        M30_ind = MOMENTS_NAME_MAP["M30"]
        rho_M30[i] = res['sums'][M30_ind] / res['sums'][f_ind] / sigma**3

        M03_ind = MOMENTS_NAME_MAP["M03"]
        rho_M03[i] = res['sums'][M03_ind] / res['sums'][f_ind] / sigma**3

        M31_ind = MOMENTS_NAME_MAP["M31"]
        rho_M31[i] = res['sums'][M31_ind] / res['sums'][f_ind] / sigma**4

        M13_ind = MOMENTS_NAME_MAP["M13"]
        rho_M13[i] = res['sums'][M13_ind] / res['sums'][f_ind] / sigma**4

        M40_ind = MOMENTS_NAME_MAP["M40"]
        rho_M40[i] = res['sums'][M40_ind] / res['sums'][f_ind] / sigma**4

        M14_ind = MOMENTS_NAME_MAP["M14"]
        rho_M14[i] = res['sums'][M14_ind] / res['sums'][f_ind] / sigma**4

    rho4_mean = rho4s.mean()
    rho4_std = rho4s.std()
    print(f'rho4: {rho4_mean:.3g} std: {rho4_std:.3g}')
    assert np.abs(rho4_mean - 2) < 1e-5

    rho6_mean = rho6s.mean()
    rho6_std = rho6s.std()
    print(f'rho6: {rho6_mean:.3g} std: {rho6_std:.3g}')
    assert np.abs(rho6_mean - 6) < 1e-5

    rho8_mean = rho8s.mean()
    rho8_std = rho8s.std()
    print(f'rho8: {rho8_mean:.3g} std: {rho8_std:.3g}')
    assert np.abs(rho8_mean - 24) < 1e-5

    rho_M21_mean = rho_M21.mean()
    assert np.abs(rho_M21_mean) < 1e-5

    rho_M12_mean = rho_M12.mean()
    assert np.abs(rho_M12_mean) < 1e-5

    rho_M30_mean = rho_M30.mean()
    assert np.abs(rho_M30_mean) < 1e-5

    rho_M03_mean = rho_M03.mean()
    assert np.abs(rho_M03_mean) < 1e-5

    rho_M31_mean = rho_M31.mean()
    assert np.abs(rho_M31_mean) < 1e-5

    rho_M13_mean = rho_M13.mean()
    assert np.abs(rho_M13_mean) < 1e-5

    rho_M40_mean = rho_M40.mean()
    assert np.abs(rho_M40_mean) < 1e-5

    rho_M14_mean = rho_M14.mean()
    assert np.abs(rho_M14_mean) < 1e-5
