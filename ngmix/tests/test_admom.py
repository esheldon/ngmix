import copy
import galsim
import numpy as np
import pytest

import ngmix
from ngmix.moments import fwhm_to_T
import ngmix.flags


@pytest.mark.parametrize('wcs_g1', [-0.5, 0, 0.2])
@pytest.mark.parametrize('wcs_g2', [-0.2, 0, 0.5])
@pytest.mark.parametrize('g1_true', [-0.1, 0, 0.2])
@pytest.mark.parametrize('g2_true', [-0.2, 0, 0.1])
def test_admom(g1_true, g2_true, wcs_g1, wcs_g2):
    rng = np.random.RandomState(seed=100)

    fwhm = 0.9
    image_size = 107
    cen = (image_size - 1)/2
    gs_wcs = galsim.ShearWCS(
        0.125, galsim.Shear(g1=wcs_g1, g2=wcs_g2)).jacobian()

    obj = galsim.Gaussian(
        fwhm=fwhm
    ).shear(
        g1=g1_true, g2=g2_true
    ).withFlux(
        400)
    im = obj.drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs,
        method='no_pixel').array
    noise = np.sqrt(np.sum(im**2)) / 1e18
    wgt = np.ones_like(im) / noise**2
    scale = np.sqrt(gs_wcs.pixelArea())

    g1arr = []
    g2arr = []
    Tarr = []
    for _ in range(50):
        shift = rng.uniform(low=-scale/2, high=scale/2, size=2)
        xy = gs_wcs.toImage(galsim.PositionD(shift))

        im = obj.shift(
            dx=shift[0], dy=shift[1]
        ).drawImage(
            nx=image_size,
            ny=image_size,
            wcs=gs_wcs,
            method='no_pixel',
            dtype=np.float64,
        ).array

        jac = ngmix.Jacobian(
            y=cen + xy.y, x=cen + xy.x,
            dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
            dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

        _im = im + (rng.normal(size=im.shape) * noise)
        obs = ngmix.Observation(
            image=_im,
            weight=wgt,
            jacobian=jac)

        Tguess = fwhm_to_T(fwhm) + rng.normal()*0.01
        res = ngmix.admom.run_admom(obs=obs, guess=Tguess)

        if res['flags'] == 0:
            gm = res.get_gmix()
            _g1, _g2, _T = gm.get_g1g2T()

            g1arr.append(_g1)
            g2arr.append(_g2)
            Tarr.append(_T)

            fim = res.make_image()
            assert fim.shape == im.shape

        res['flags'] = 5
        with pytest.raises(RuntimeError):
            res.make_image()
        with pytest.raises(RuntimeError):
            res.get_gmix()

    g1 = np.mean(g1arr)
    g2 = np.mean(g2arr)
    gtol = 1.5e-6
    assert np.abs(g1 - g1_true) < gtol, (g1, np.std(g1arr)/np.sqrt(len(g1arr)))
    assert np.abs(g2 - g2_true) < gtol, (g2, np.std(g2arr)/np.sqrt(len(g2arr)))

    if g1_true == 0 and g2_true == 0:
        T = np.mean(Tarr)
        assert np.abs(T - fwhm_to_T(fwhm)) < 1e-6

    with pytest.raises(ValueError):
        ngmix.admom.run_admom(None, None)

    # cover some branches
    tres = copy.deepcopy(res)

    tres['flags'] = 0
    tres['sums_cov'][:, :] = np.nan
    tres = ngmix.admom.admom.get_result(tres)
    assert np.isnan(tres['e1err'])

    tres = copy.deepcopy(res)
    tres['flags'] = 0
    tres['pars'][4] = -1
    tres = ngmix.admom.admom.get_result(tres)
    assert tres['flags'] == ngmix.flags.NONPOS_SIZE


@pytest.mark.parametrize('snr', [20, 10, 5])
@pytest.mark.parametrize('g1_true', [0, 0.2])
@pytest.mark.parametrize('g2_true', [-0.2, 0])
def test_admom_find_cen(g1_true, g2_true, snr):
    """
    test that the center is right within expected errors, and that the fit is
    robust.  We expect a failure rate less than 0.3 percent
    """
    rng = np.random.RandomState(seed=55)
    ntrial = 1000
    ntry = 2

    bd = galsim.BaseDeviate(seed=rng.randint(0, 2**31))
    gsnoise = galsim.GaussianNoise(bd)

    psf_fwhm = 0.8
    gal_hlr = 0.5
    image_size = 48
    cen = (image_size - 1)/2
    scale = 0.2
    shift_scale = scale/2

    psf = galsim.Gaussian(fwhm=psf_fwhm)

    weight_fwhm = 1.2
    drow_arr = []
    dcol_arr = []
    for _ in range(ntrial):
        drow, dcol = rng.uniform(
            low=-shift_scale,
            high=shift_scale,
            size=2,
        )

        obj = galsim.Exponential(
            half_light_radius=gal_hlr,
        ).shift(
            dcol, drow,
        ).shear(
            g1=g1_true, g2=g2_true
        )

        obj = galsim.Convolve(obj, psf)

        gsim = obj.drawImage(
            nx=image_size,
            ny=image_size,
            scale=scale,
        )
        noise_var = gsim.addNoiseSNR(noise=gsnoise, snr=snr)

        im = gsim.array

        wgt = np.ones_like(im) / noise_var

        jac = ngmix.DiagonalJacobian(row=cen, col=cen, scale=scale)

        obs = ngmix.Observation(
            image=im,
            weight=wgt,
            jacobian=jac,
        )

        for itrial in range(ntry):

            res = ngmix.admom.find_cen_admom(
                obs=obs, fwhm=weight_fwhm, rng=rng, ntry=4,
            )
            if res['flags'] == 0:
                break

        if res['flags'] == 0:
            gm = res.get_gmix()
            drow_meas, dcol_meas = gm.get_cen()

            drow_arr.append(drow_meas - drow)
            dcol_arr.append(dcol_meas - dcol)

    frac_fail = len(drow_arr)/ntrial - 1

    assert abs(frac_fail) < 0.003

    drow_arr = np.hstack(drow_arr)
    dcol_arr = np.hstack(dcol_arr)

    drow_mean = drow_arr.mean()
    dcol_mean = dcol_arr.mean()

    drow_err = drow_arr.std()/np.sqrt(drow_arr.size)
    dcol_err = dcol_arr.std()/np.sqrt(dcol_arr.size)

    assert np.abs(drow_mean)/drow_err < 5, (drow_mean/drow_err)
    assert np.abs(dcol_mean)/dcol_err < 5, (dcol_mean/dcol_err)


def check(name, vals, expected, errs=None):
    mn = vals.mean()
    std = vals.std()
    err = std / np.sqrt(vals.size)
    print(f'{name} expected: {expected} mean: {mn} +/- {err}')
    assert np.abs(mn - expected) / err < 4


def check_error(name, vals, errs, tol):
    std = vals.std()
    merr = np.mean(errs)

    print(f'{name} std: {std} predicted {merr}')
    assert np.abs(merr/std - 1) < 0.4


@pytest.mark.parametrize('dozeros', [False, True])
def test_admom_fill(dozeros):
    """
    test admom flux and filling in zero weight pixels
    """
    rng = np.random.RandomState(seed=550)
    ntrial = 1000

    flux = 100
    noise = 0.5
    # noise = 0.01
    fwhm = 1.1
    # T = ngmix.moments.fwhm_to_T(fwhm)
    image_size = 48
    cen = (image_size - 1)/2
    scale = 0.2
    shift_scale = scale/2

    fluxes = []
    flux_errors = []
    e1vals = []
    e2vals = []
    e1err_vals = []
    e2err_vals = []
    Tvals = []
    Terr_vals = []
    true_e1vals = []
    true_e2vals = []

    for _ in range(ntrial):
        drow, dcol = rng.uniform(
            low=-shift_scale,
            high=shift_scale,
            size=2,
        )

        while True:
            g1, g2 = rng.normal(scale=0.05, size=2)
            if np.sqrt(g1**2 + g2**2) < 1:
                shear = galsim.Shear(g1=g1, g2=g2)
                break

        obj = galsim.Gaussian(
            flux=flux,
            fwhm=fwhm,
        ).shear(
            g1=g1, g2=g2,
        ).shift(
            dcol, drow,
        )

        gsim = obj.drawImage(
            nx=image_size,
            ny=image_size,
            scale=scale,
            method='no_pixel',
        )
        im = gsim.array
        im += rng.normal(scale=noise, size=im.shape)

        wgt = im * 0 + 1 / noise**2
        if dozeros:
            randcol = rng.randint(
                low=image_size//2 - 2,
                high=image_size//2 + 2,
            )
            wgt[:, randcol] = 0
            im[:, randcol] = 0

        jac = ngmix.DiagonalJacobian(row=cen, col=cen, scale=scale)

        obs = ngmix.Observation(
            image=im,
            weight=wgt,
            jacobian=jac,
            ignore_zero_weight=False,
        )

        # guess = ngmix.GMixModel([drow, dcol, g1, g2, T, flux], "gauss")
        # print('guess:', guess.get_e1e2T())
        res = ngmix.admom.run_admom(
            obs=obs,
            guess=1,
            # guess=guess,
            rng=rng,
        )
        if res['flux_flags'] == 0:
            fluxes.append(res['flux'])
            flux_errors.append(res['flux_err'])
            true_e1vals.append(shear.e1)
            true_e2vals.append(shear.e2)
            e1vals.append(res['e1'])
            e2vals.append(res['e2'])
            e1err_vals.append(res['e1err'])
            e2err_vals.append(res['e2err'])
            Tvals.append(res['T'])
            Terr_vals.append(res['T_err'])

    fluxes = np.array(fluxes)
    flux_errors = np.array(flux_errors)
    true_e1vals = np.array(true_e1vals)
    true_e2vals = np.array(true_e2vals)
    e1vals = np.array(e1vals)
    e2vals = np.array(e2vals)
    Tvals = np.array(Tvals)

    check('flux', fluxes, flux)
    check_error('flux err:', fluxes, flux_errors, tol=0.4)
    check('e1', e1vals - true_e1vals, 0)
    check('e2', e2vals - true_e2vals, 0)
    check_error('e1 err', e1vals - true_e1vals, e1err_vals, tol=0.1)
    check_error('e2 err', e2vals - true_e2vals, e2err_vals, tol=0.1)
    # T will be biased by noise
    # check('T', Tvals, T)
    check_error('T err:', Tvals, Terr_vals, tol=0.1)

    # raise ValueError('blah')
