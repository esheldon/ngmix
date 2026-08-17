import galsim
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

import ngmix
from ngmix.moments import fwhm_to_T
from ngmix import Jacobian, Observation, ObsList, MultiBandObsList
from ngmix.prepsfadmom import run_prepsf_admom, PAdmomFitter
import ngmix.flags


GSPARAMS = galsim.GSParams(
    folding_threshold=1.0e-8,
    maxk_threshold=1.0e-8,
    kvalue_accuracy=1.0e-8,
    xvalue_accuracy=1.0e-8,
)


def _gauss_cov(e1, e2, T):
    """
    covariance matrix [[Ivv, Ivu], [Ivu, Iuu]] with e1 = <uu-vv>/T
    """
    return 0.5 * np.array([
        [T * (1 - e1), T * e2],
        [T * e2, T * (1 + e1)],
    ])


def _cov_to_gauss(Sigma, flux):
    """
    galsim gaussian with the given covariance matrix
    """
    T = Sigma[0, 0] + Sigma[1, 1]
    e1 = (Sigma[1, 1] - Sigma[0, 0]) / T
    e2 = 2 * Sigma[0, 1] / T
    return galsim.Gaussian(
        sigma=np.linalg.det(Sigma) ** 0.25, gsparams=GSPARAMS,
    ).shear(e1=e1, e2=e2) * flux


def _make_obs(
    e1, e2, T, flux, psf_fwhm, gs_wcs, dim=48,
    offset_pix=(0.0, 0.0), noise=1.0e-9, rng=None,
):
    """
    gaussian galaxy convolved with a round gaussian psf, drawn
    analytically (gaussian (x) gaussian is gaussian) so there is no
    galsim fft rendering error
    """
    Tpsf = fwhm_to_T(psf_fwhm)
    Sigma = _gauss_cov(e1, e2, T) + np.diag([Tpsf / 2, Tpsf / 2])
    obj = _cov_to_gauss(Sigma, flux)
    psf = galsim.Gaussian(fwhm=psf_fwhm, gsparams=GSPARAMS)

    cen = (dim - 1) / 2

    im = obj.drawImage(
        nx=dim, ny=dim, wcs=gs_wcs,
        offset=(offset_pix[1], offset_pix[0]),
    ).array
    if rng is not None:
        im = im + rng.normal(scale=noise, size=im.shape)
    psf_im = psf.drawImage(nx=dim, ny=dim, wcs=gs_wcs).array

    jac = Jacobian(
        y=cen, x=cen,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy,
    )
    psf_obs = Observation(psf_im, jacobian=jac)
    return Observation(
        im,
        jacobian=jac,
        weight=np.ones_like(im) / noise**2,
        psf=psf_obs,
    )


@pytest.mark.parametrize('wcs_g1,wcs_g2', [(0, 0), (-0.2, 0.1)])
@pytest.mark.parametrize('e1_true,e2_true', [(0.2, -0.1), (0, 0)])
def test_prepsfadmom_gauss(e1_true, e2_true, wcs_g1, wcs_g2):
    """
    noiseless gaussian galaxy: everything should be recovered nearly
    exactly, including with an offset center and a sheared wcs
    """
    T_true = 0.6
    flux_true = 3.5

    gs_wcs = galsim.ShearWCS(
        0.25, galsim.Shear(g1=wcs_g1, g2=wcs_g2)).jacobian()

    obs = _make_obs(
        e1_true, e2_true, T_true, flux_true, 0.9, gs_wcs,
        offset_pix=(0.6, -0.4),
    )

    res = run_prepsf_admom(obs, guess=0.5, rng=np.random.RandomState(3))

    assert res['flags'] == 0
    assert np.abs(res['e1'] - e1_true) < 1.0e-4
    assert np.abs(res['e2'] - e2_true) < 1.0e-4
    assert np.abs(res['T'] / T_true - 1) < 1.0e-3
    assert np.abs(res['flux'] / flux_true - 1) < 1.0e-3

    # the offset in sky coordinates
    dv = gs_wcs.dvdx * (-0.4) + gs_wcs.dvdy * 0.6
    du = gs_wcs.dudx * (-0.4) + gs_wcs.dudy * 0.6
    assert np.abs(res['cen'][0] - dv) < 1.0e-3
    assert np.abs(res['cen'][1] - du) < 1.0e-3

    # gmix construction round trips
    gm = res.get_gmix()
    assert np.abs(gm.get_T() / T_true - 1) < 1.0e-3


def test_prepsfadmom_nosmooth_matches_smooth():
    """
    for noiseless gaussians the smoothed and unsmoothed fits agree
    """
    gs_wcs = galsim.PixelScale(0.25).jacobian()
    obs = _make_obs(0.1, -0.05, 0.6, 3.5, 0.9, gs_wcs)

    res_sm = run_prepsf_admom(obs, guess=0.5, rng=np.random.RandomState(3))
    res_raw = run_prepsf_admom(
        obs, guess=0.5, fwhm_smooth=0, rng=np.random.RandomState(3),
    )

    assert res_sm['flags'] == 0
    assert res_raw['flags'] == 0
    assert res_sm['fwhm_smooth'] > 0.9
    assert res_raw['fwhm_smooth'] == 0

    assert np.abs(res_sm['e1'] - res_raw['e1']) < 1.0e-4
    assert np.abs(res_sm['e2'] - res_raw['e2']) < 1.0e-4
    assert np.abs(res_sm['T'] / res_raw['T'] - 1) < 1.0e-3
    assert np.abs(res_sm['flux'] / res_raw['flux'] - 1) < 1.0e-3


def test_prepsfadmom_exp_pixel_handling():
    """
    for a small exp galaxy the unsmoothed kspace fit should match
    real-space adaptive moments run on a finely sampled pre-psf
    rendering (the continuous limit); real-space admom at the native
    scale is biased by point sampling at this resolution
    """
    scale = 0.263
    dim = 48
    rng = np.random.RandomState(42)

    gal = galsim.Exponential(
        half_light_radius=0.5, gsparams=GSPARAMS,
    ).shear(g1=0.08, g2=0.03) * 3.5
    psf = galsim.Gaussian(fwhm=0.9, gsparams=GSPARAMS)

    cen = (dim - 1) / 2
    im = galsim.Convolve(gal, psf, gsparams=GSPARAMS).drawImage(
        nx=dim, ny=dim, scale=scale).array
    psf_im = psf.drawImage(nx=dim, ny=dim, scale=scale).array
    jac = ngmix.DiagonalJacobian(scale=scale, row=cen, col=cen)
    obs = Observation(
        im, jacobian=jac, weight=np.ones_like(im) * 1.0e18,
        psf=Observation(psf_im, jacobian=jac),
    )

    res = run_prepsf_admom(obs, guess=0.5, fwhm_smooth=0, rng=rng)
    assert res['flags'] == 0

    # continuous limit reference
    fine = 4
    fdim = dim * fine
    fcen = (fdim - 1) / 2
    im_fine = gal.drawImage(
        nx=fdim, ny=fdim, scale=scale / fine, method='no_pixel').array
    obs_fine = Observation(
        im_fine,
        jacobian=ngmix.DiagonalJacobian(scale=scale / fine, row=fcen, col=fcen),
    )
    amres = ngmix.admom.run_admom(obs_fine, guess=0.5, rng=rng)
    assert amres['flags'] == 0

    assert np.abs(res['e1'] - amres['e1']) < 2.0e-3
    assert np.abs(res['e2'] - amres['e2']) < 2.0e-3
    assert np.abs(res['T'] / amres['T'] - 1) < 5.0e-3


def test_prepsfadmom_multiband():
    """
    joint multiband fit: common structure, per band fluxes.  colors are
    exact for any profile common across bands, including non-gaussian
    profiles for which the absolute normalization has a (common) bias
    """
    gs_wcs = galsim.PixelScale(0.25).jacobian()
    psf_fwhms = [1.1, 0.9, 0.8]
    fluxes = [2.0, 3.5, 4.5]
    e1, e2, T = 0.1, -0.05, 0.5

    # gaussian case: everything exact
    mbobs = MultiBandObsList()
    for pf, fl in zip(psf_fwhms, fluxes):
        obslist = ObsList()
        obslist.append(_make_obs(e1, e2, T, fl, pf, gs_wcs))
        mbobs.append(obslist)

    res = run_prepsf_admom(mbobs, guess=0.5, rng=np.random.RandomState(3))
    assert res['flags'] == 0
    assert res['flux'].shape == (3,)
    assert np.abs(res['e1'] - e1) < 1.0e-4
    assert np.abs(res['e2'] - e2) < 1.0e-4
    assert np.abs(res['T'] / T - 1) < 1.0e-3
    assert np.all(np.abs(res['flux'] / fluxes - 1) < 1.0e-3)

    # the smoothing must cover the worst psf
    assert res['fwhm_smooth'] > 1.1

    # non-gaussian profile: the flux normalization bias is identical in
    # all bands, so colors are exact
    gal = galsim.Exponential(
        half_light_radius=0.5, gsparams=GSPARAMS,
    ).shear(g1=0.08, g2=0.03)

    dim = 48
    cen = (dim - 1) / 2
    jac = Jacobian(
        y=cen, x=cen,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy,
    )
    mbobs = MultiBandObsList()
    for pf, fl in zip(psf_fwhms, fluxes):
        psf = galsim.Gaussian(fwhm=pf, gsparams=GSPARAMS)
        im = galsim.Convolve(gal * fl, psf, gsparams=GSPARAMS).drawImage(
            nx=dim, ny=dim, wcs=gs_wcs).array
        psf_im = psf.drawImage(nx=dim, ny=dim, wcs=gs_wcs).array
        obslist = ObsList()
        obslist.append(Observation(
            im, jacobian=jac, weight=np.ones_like(im) * 1.0e18,
            psf=Observation(psf_im, jacobian=jac),
        ))
        mbobs.append(obslist)

    res = run_prepsf_admom(mbobs, guess=0.5, rng=np.random.RandomState(3))
    assert res['flags'] == 0

    fnorm = res['flux'] / np.array(fluxes)
    assert np.all(np.abs(fnorm / fnorm[0] - 1) < 1.0e-4)


def test_prepsfadmom_multiepoch():
    """
    multiple epochs in one band accumulate into a single flux
    consistent with a single epoch fit
    """
    gs_wcs = galsim.PixelScale(0.25).jacobian()
    obs = _make_obs(0.1, -0.05, 0.6, 3.5, 0.9, gs_wcs)

    obslist = ObsList()
    obslist.append(obs)
    obslist.append(obs)

    res1 = run_prepsf_admom(obs, guess=0.5, rng=np.random.RandomState(3))
    res2 = run_prepsf_admom(obslist, guess=0.5, rng=np.random.RandomState(3))

    # the guesses differ slightly through the rng, so agreement is to
    # within the convergence tolerance rather than exact
    assert res2['flags'] == 0
    assert np.abs(res2['flux'] / res1['flux'] - 1) < 1.0e-4
    assert np.abs(res2['T'] / res1['T'] - 1) < 1.0e-4

    # scalar flux for non multiband input
    assert np.isscalar(res2['flux']) or res2['flux'].ndim == 0


def test_prepsfadmom_noise():
    """
    with noise at s/n ~ 15-20 the smoothed fitter should have no
    failures, small bias, and error estimates that match the observed
    scatter
    """
    ntrial = 200
    rng = np.random.RandomState(31415)
    gs_wcs = galsim.PixelScale(0.25).jacobian()

    e1_true, e2_true, T_true, flux_true = 0.2, -0.1, 0.6, 3.5

    # set the noise from a noiseless image for a target matched s/n
    obs0 = _make_obs(e1_true, e2_true, T_true, flux_true, 0.9, gs_wcs)
    noise = np.sqrt(np.sum(obs0.image ** 2)) / 15.0

    fitter = PAdmomFitter(rng=rng)

    e1s, e2s, fluxes = [], [], []
    e1errs, fluxerrs = [], []
    nfail = 0
    for i in range(ntrial):
        obs = _make_obs(
            e1_true, e2_true, T_true, flux_true, 0.9, gs_wcs,
            noise=noise, rng=rng,
        )
        res = fitter.go(obs, guess=0.6)
        if res['flags'] != 0:
            nfail += 1
            continue
        e1s.append(res['e1'])
        e2s.append(res['e2'])
        fluxes.append(res['flux'])
        e1errs.append(res['e1err'])
        fluxerrs.append(res['flux_err'])

    assert nfail < ntrial * 0.02

    e1s, e2s, fluxes = np.array(e1s), np.array(e2s), np.array(fluxes)

    # noise bias is expected to be small for the smoothed fitter
    assert np.abs(np.median(e1s) - e1_true) < 0.05
    assert np.abs(np.median(e2s) - e2_true) < 0.05
    assert np.abs(np.median(fluxes) / flux_true - 1) < 0.1

    # the errors include the first order response of the adaptive
    # weight to the noise, so both the flux and shape errors should
    # match the observed scatter
    assert np.abs(np.mean(fluxerrs) / np.std(fluxes) - 1) < 0.15
    assert np.abs(np.mean(e1errs) / np.std(e1s) - 1) < 0.3


def test_prepsfadmom_small_object():
    """
    an object much smaller than the psf is recovered with the smoothed
    fitter, a regime where real-space adaptive moments cannot measure
    the pre-psf size at all
    """
    ntrial = 100
    rng = np.random.RandomState(2718)
    gs_wcs = galsim.PixelScale(0.25).jacobian()

    T_true = 0.05
    obs0 = _make_obs(0.1, -0.05, T_true, 3.5, 0.9, gs_wcs)
    noise = np.sqrt(np.sum(obs0.image ** 2)) / 100.0

    fitter = PAdmomFitter(rng=rng)

    Ts = []
    for i in range(ntrial):
        obs = _make_obs(
            0.1, -0.05, T_true, 3.5, 0.9, gs_wcs, noise=noise, rng=rng,
        )
        res = fitter.go(obs, guess=0.1)
        # T and flux are usable even when the subtracted T scatters
        # non-positive
        assert res['T_flags'] == 0
        assert res['flux_flags'] == 0
        Ts.append(res['T'])

    Ts = np.array(Ts)
    assert np.abs(np.mean(Ts) - T_true) < 5 * np.std(Ts) / np.sqrt(Ts.size)


def test_prepsfadmom_guess_types():
    """
    guesses can be a T value, a gmix, or left unset
    """
    gs_wcs = galsim.PixelScale(0.25).jacobian()
    obs = _make_obs(0.1, -0.05, 0.6, 3.5, 0.9, gs_wcs)

    rng = np.random.RandomState(3)
    res_t = run_prepsf_admom(obs, guess=0.5, rng=rng)

    gm_guess = ngmix.GMixModel([0.0, 0.0, 0.05, -0.02, 0.5, 1.0], 'gauss')
    res_gm = run_prepsf_admom(obs, guess=gm_guess)

    res_none = run_prepsf_admom(obs, rng=np.random.RandomState(3))

    for res in [res_t, res_gm, res_none]:
        assert res['flags'] == 0
        assert np.abs(res['T'] / 0.6 - 1) < 1.0e-3


def test_prepsfadmom_errors():
    """
    error conditions raise
    """
    gs_wcs = galsim.PixelScale(0.25).jacobian()
    obs = _make_obs(0.1, -0.05, 0.6, 3.5, 0.9, gs_wcs)

    # no psf raises unless no_psf is set
    obs_nopsf = Observation(
        obs.image, jacobian=obs.jacobian, weight=obs.weight,
    )
    with pytest.raises(RuntimeError):
        run_prepsf_admom(obs_nopsf, guess=0.5, rng=np.random.RandomState(3))

    # no_psf requires explicit fwhm_smooth
    with pytest.raises(ValueError):
        run_prepsf_admom(
            obs_nopsf, guess=0.5, no_psf=True, rng=np.random.RandomState(3),
        )

    # with explicit smoothing, no_psf works and measures the pre-pixel
    # object
    res = run_prepsf_admom(
        obs_nopsf, guess=0.5, no_psf=True, fwhm_smooth=1.0,
        rng=np.random.RandomState(3),
    )
    assert res['flags'] == 0
    Ttot = 0.6 + fwhm_to_T(0.9)
    assert np.abs(res['T'] / Ttot - 1) < 1.0e-2

    # use_noise_image requires obs.noise to be set
    with pytest.raises(ValueError):
        run_prepsf_admom(
            obs, guess=0.5, use_noise_image=True,
            rng=np.random.RandomState(3),
        )


def test_prepsfadmom_nonsquare():
    """
    non square images are zero padded to square internally: a
    rectangular stamp cut from a square stamp gives the same result,
    including the errors, since the noise scaling uses the actual
    pixel count
    """
    gs_wcs = galsim.ShearWCS(
        0.25, galsim.Shear(g1=-0.1, g2=0.06),
    ).jacobian()
    obs = _make_obs(0.1, -0.05, 0.6, 3.5, 0.9, gs_wcs)

    # ap_rad=0 so the only difference between the stamps is the
    # trimmed pixels themselves, which hold negligible flux
    res = run_prepsf_admom(
        obs, guess=0.6, ap_rad=0, rng=np.random.RandomState(5),
    )
    assert res['flags'] == 0

    ntrim = 8
    for axis in (0, 1):
        row0, col0 = obs.jacobian.row0, obs.jacobian.col0
        if axis == 0:
            rim = obs.image[ntrim:-ntrim, :]
            rwgt = obs.weight[ntrim:-ntrim, :]
            row0 = row0 - ntrim
        else:
            rim = obs.image[:, ntrim:-ntrim]
            rwgt = obs.weight[:, ntrim:-ntrim]
            col0 = col0 - ntrim

        rjac = Jacobian(
            y=row0, x=col0,
            dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
            dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy,
        )
        robs = Observation(
            rim, jacobian=rjac, weight=rwgt, psf=obs.psf,
        )
        rres = run_prepsf_admom(
            robs, guess=0.6, ap_rad=0, rng=np.random.RandomState(5),
        )
        assert rres['flags'] == 0

        for key in ['flux', 'T', 'e1', 'e2']:
            assert np.allclose(rres[key], res[key], rtol=0, atol=1.0e-6)
        for key in ['flux_err', 'T_err']:
            assert np.allclose(rres[key], res[key], rtol=1.0e-6, atol=0)


def test_prepsfadmom_noise_image_white():
    """
    for white noise, the per-mode noise power measured from an attached
    noise image agrees with the weight map errors in the mean over
    realizations, and the measured moments are unchanged.  The single
    realization scatter of the errors is several percent, set by the
    number of independent noise modes under the kernel
    """
    nreal = 20
    rng = np.random.RandomState(991)
    gs_wcs = galsim.PixelScale(0.25).jacobian()

    obs0 = _make_obs(0.1, -0.05, 0.6, 3.5, 0.9, gs_wcs)
    noise = np.sqrt(np.sum(obs0.image ** 2)) / 40.0

    obs = _make_obs(
        0.1, -0.05, 0.6, 3.5, 0.9, gs_wcs, noise=noise, rng=rng,
    )
    res = run_prepsf_admom(obs, guess=0.6, rng=np.random.RandomState(5))
    assert res['flags'] == 0

    errs = {key: [] for key in ['flux_err', 'T_err', 'e1err', 'e2err']}
    for i in range(nreal):
        nobs = Observation(
            obs.image, jacobian=obs.jacobian, weight=obs.weight,
            psf=obs.psf,
            noise=rng.normal(scale=noise, size=obs.image.shape),
        )
        nres = run_prepsf_admom(
            nobs, guess=0.6, use_noise_image=True,
            rng=np.random.RandomState(5),
        )
        assert nres['flags'] == 0

        # the noise image affects only the errors
        for key in ['flux', 'T', 'e1', 'e2']:
            assert np.allclose(nres[key], res[key], rtol=0, atol=0)

        for key in errs:
            assert np.allclose(nres[key], res[key], rtol=0.3, atol=0)
            errs[key].append(nres[key])

    for key in errs:
        assert np.allclose(np.mean(errs[key]), res[key], rtol=0.05, atol=0)


def test_prepsfadmom_noise_image_correlated():
    """
    for correlated noise, the errors from the per-mode noise power
    match the observed scatter with the same fidelity as the weight map
    errors do for white noise, while the white noise assumption
    underestimates the scatter badly
    """
    ntrial = 200
    filt_sigma = 1.25
    rng = np.random.RandomState(8231)
    gs_wcs = galsim.PixelScale(0.25).jacobian()

    e1_true, e2_true, T_true, flux_true = 0.2, -0.1, 0.6, 3.5
    obs0 = _make_obs(e1_true, e2_true, T_true, flux_true, 0.9, gs_wcs)
    dims = obs0.image.shape

    # white noise for s/n ~ 20.  The correlated noise boosts the power
    # at the scales of the kernel by roughly the effective area of the
    # filter, so its pixel std is scaled down to keep a similar
    # effective s/n
    sigma_white = np.sqrt(np.sum(obs0.image ** 2)) / 20.0
    sigma_corr = sigma_white / 4.0

    # the pixel std of a filtered unit white field, for normalization
    fac = gaussian_filter(
        rng.normal(size=(2000, 2000)), filt_sigma, mode='wrap',
    ).std()

    def _corr_noise():
        return gaussian_filter(
            rng.normal(size=dims), filt_sigma, mode='wrap',
        ) * (sigma_corr / fac)

    fitter_white = PAdmomFitter(rng=rng)
    fitter_pm = PAdmomFitter(use_noise_image=True, rng=rng)

    fluxes_a, e1s_a, fluxerrs_a, e1errs_a = [], [], [], []
    fluxes_b, e1s_b, fluxerrs_b, e1errs_b = [], [], [], []
    nfail = 0
    for i in range(ntrial):
        obs = Observation(
            obs0.image + rng.normal(scale=sigma_white, size=dims),
            jacobian=obs0.jacobian,
            weight=np.ones(dims) / sigma_white ** 2,
            psf=obs0.psf,
        )
        res = fitter_white.go(obs, guess=0.6)

        nobs = Observation(
            obs0.image + _corr_noise(),
            jacobian=obs0.jacobian,
            weight=np.ones(dims) / sigma_corr ** 2,
            psf=obs0.psf,
            noise=_corr_noise(),
        )
        nres = fitter_pm.go(nobs, guess=0.6)

        if res['flags'] != 0 or nres['flags'] != 0:
            nfail += 1
            continue
        fluxes_a.append(res['flux'])
        e1s_a.append(res['e1'])
        fluxerrs_a.append(res['flux_err'])
        e1errs_a.append(res['e1err'])
        fluxes_b.append(nres['flux'])
        e1s_b.append(nres['e1'])
        fluxerrs_b.append(nres['flux_err'])
        e1errs_b.append(nres['e1err'])

    assert nfail < ntrial * 0.02

    ratio_a_flux = np.std(fluxes_a) / np.mean(fluxerrs_a)
    ratio_b_flux = np.std(fluxes_b) / np.mean(fluxerrs_b)
    ratio_a_e1 = np.std(e1s_a) / np.mean(e1errs_a)
    ratio_b_e1 = np.std(e1s_b) / np.mean(e1errs_b)

    # with the delta method weight response the errors are absolutely
    # calibrated in both cases
    assert np.abs(ratio_a_flux - 1) < 0.15
    assert np.abs(ratio_b_flux - 1) < 0.2
    assert np.abs(ratio_a_e1 - 1) < 0.2
    assert np.abs(ratio_b_e1 - 1) < 0.2

    # the white noise assumption would report errors scaled from case a
    # by the pixel std, badly underestimating the observed scatter
    werr_b = np.mean(fluxerrs_a) * sigma_corr / sigma_white
    assert np.std(fluxes_b) / werr_b > 2.0

    # the per-mode errors track the scatter as well as the white errors
    # do for white noise
    assert np.abs(ratio_b_flux / ratio_a_flux - 1) < 0.2
    assert np.abs(ratio_b_e1 / ratio_a_e1 - 1) < 0.2


def test_prepsfadmom_noise_multiband():
    """
    per band flux errors with the joint weight: the weight response
    term couples each band flux to the joint moment conditions, so
    the errors need the cross covariance of the band flux sums with
    the joint sums.  Check the reported errors match the observed
    scatter in each band
    """
    ntrial = 200
    rng = np.random.RandomState(271828)
    gs_wcs = galsim.PixelScale(0.25).jacobian()

    e1_true, e2_true, T_true = 0.2, -0.1, 0.6
    band_fluxes = [3.5, 5.5]
    psf_fwhms = [0.9, 0.8]

    # per band noise for flux s/n ~ 15 and 25
    noises = []
    for flux, pf, s2n in zip(band_fluxes, psf_fwhms, [15.0, 25.0]):
        obs0 = _make_obs(e1_true, e2_true, T_true, flux, pf, gs_wcs)
        noises.append(np.sqrt(np.sum(obs0.image ** 2)) / s2n)

    fitter = PAdmomFitter(rng=rng)

    fluxes = [[], []]
    fluxerrs = [[], []]
    nfail = 0
    for i in range(ntrial):
        mbobs = MultiBandObsList()
        for flux, pf, noise in zip(band_fluxes, psf_fwhms, noises):
            obslist = ObsList()
            obslist.append(_make_obs(
                e1_true, e2_true, T_true, flux, pf, gs_wcs,
                noise=noise, rng=rng,
            ))
            mbobs.append(obslist)

        res = fitter.go(mbobs, guess=0.6)
        if res['flags'] != 0:
            nfail += 1
            continue
        for band in range(2):
            fluxes[band].append(res['flux'][band])
            fluxerrs[band].append(res['flux_err'][band])

    assert nfail < ntrial * 0.02

    for band in range(2):
        ratio = np.mean(fluxerrs[band]) / np.std(fluxes[band])
        assert np.abs(ratio - 1) < 0.15


def test_prepsfadmom_covariance_aware_s2n():
    """
    the total flux s/n is the joint (Wald) value wherever the
    cross-band flux covariance is available: on the model paths
    s2n satisfies sqrt(F^T C^-1 F) with the reported flux_cov and
    sits below the independent-band quadrature sum (positive
    shared-response correlation).  The plain gauss delta path
    assembles no cross-band covariance and keeps the quadrature
    value; the star path's covariance is exactly diagonal so the
    joint value equals the quadrature one
    """
    rng = np.random.RandomState(1234)
    gs_wcs = galsim.PixelScale(0.25).jacobian()

    e1_true, e2_true, T_true = 0.2, -0.1, 0.6
    band_fluxes = [3.5, 5.5]
    psf_fwhms = [0.9, 0.8]

    noises = []
    for flux, pf, s2n in zip(band_fluxes, psf_fwhms, [15.0, 25.0]):
        obs0 = _make_obs(e1_true, e2_true, T_true, flux, pf, gs_wcs)
        noises.append(np.sqrt(np.sum(obs0.image ** 2)) / s2n)

    mbobs = MultiBandObsList()
    for flux, pf, noise in zip(band_fluxes, psf_fwhms, noises):
        obslist = ObsList()
        obslist.append(_make_obs(
            e1_true, e2_true, T_true, flux, pf, gs_wcs,
            noise=noise, rng=rng,
        ))
        mbobs.append(obslist)

    def quad_of(res):
        return np.sqrt(np.sum((res['flux'] / res['flux_err']) ** 2))

    for kw in ({'model': 'exp'}, {'model': 'exp', 'full_errors': True}):
        fitter = PAdmomFitter(rng=np.random.RandomState(5), **kw)
        res = fitter.go(mbobs, guess=0.6)
        assert res['flags'] == 0
        C = res['flux_cov']
        assert C[0, 1] > 0
        F = res['flux']
        expected = np.sqrt(F @ np.linalg.solve(C, F))
        assert np.allclose(res['s2n'], expected)
        assert res['s2n'] < quad_of(res)

    fitter = PAdmomFitter(model='gauss', rng=np.random.RandomState(5))
    res = fitter.go(mbobs, guess=0.6)
    assert res['flags'] == 0
    assert res.get('flux_cov') is None
    assert np.allclose(res['s2n'], quad_of(res))

    fitter = PAdmomFitter(model='star', rng=np.random.RandomState(5))
    res = fitter.go(mbobs, guess=0.6)
    assert res['flags'] == 0
    assert np.allclose(res['s2n'], quad_of(res))


def _make_exp_mix_obs(
    e1, e2, T, flux, psf_fwhm, gs_wcs, dim=48,
    offset_pix=(0.0, 0.0), noise=1.0e-9, rng=None, model='exp',
):
    """
    render the ngmix gaussian expansion of the named profile
    exactly, each component convolved with the gaussian psf
    analytically
    """
    from ngmix.prepsfadmom.models import get_profile_comps, cov_from_e

    Tpsf = fwhm_to_T(psf_fwhm)
    parts = []
    for frac, cT in get_profile_comps(model):
        Sigma = cov_from_e(e1, e2, cT * T) + np.diag([Tpsf / 2] * 2)
        parts.append(_cov_to_gauss(Sigma, flux * frac))
    obj = galsim.Add(parts)
    psf = galsim.Gaussian(fwhm=psf_fwhm, gsparams=GSPARAMS)

    cen = (dim - 1) / 2
    im = obj.drawImage(
        nx=dim, ny=dim, wcs=gs_wcs,
        offset=(offset_pix[1], offset_pix[0]),
    ).array
    if rng is not None:
        im = im + rng.normal(scale=noise, size=im.shape)
    psf_im = psf.drawImage(nx=dim, ny=dim, wcs=gs_wcs).array

    jac = Jacobian(
        y=cen, x=cen,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy,
    )
    psf_obs = Observation(psf_im, jacobian=jac)
    return Observation(
        im, jacobian=jac, weight=np.ones_like(im) / noise ** 2,
        psf=psf_obs,
    )


def test_prepsfadmom_model_exp():
    """
    noiseless exact 6-gaussian exponential: the moment matched family
    parameters and the total flux are recovered nearly exactly,
    including with an offset center and a sheared wcs
    """
    e1_true, e2_true, T_true, flux_true = 0.1, -0.05, 0.5, 4.0

    gs_wcs = galsim.ShearWCS(
        0.25, galsim.Shear(g1=-0.1, g2=0.06),
    ).jacobian()

    obs = _make_exp_mix_obs(
        e1_true, e2_true, T_true, flux_true, 0.9, gs_wcs,
        offset_pix=(0.4, -0.3),
    )
    res = run_prepsf_admom(
        obs, guess=0.4, model='exp', rng=np.random.RandomState(3),
    )
    assert res['flags'] == 0
    assert res['model'] == 'exp'
    assert np.abs(res['e1'] - e1_true) < 1.0e-3
    assert np.abs(res['e2'] - e2_true) < 1.0e-3
    assert np.abs(res['T'] / T_true - 1) < 2.0e-3
    assert np.abs(res['flux'] / flux_true - 1) < 1.0e-3

    dv = gs_wcs.dvdx * (-0.3) + gs_wcs.dvdy * 0.4
    du = gs_wcs.dudx * (-0.3) + gs_wcs.dudy * 0.4
    assert np.abs(res['cen'][0] - dv) < 1.0e-3
    assert np.abs(res['cen'][1] - du) < 1.0e-3

    gm = res.get_gmix()
    assert np.abs(gm.get_T() / T_true - 1) < 2.0e-3


def test_prepsfadmom_model_exp_galsim():
    """
    a true galsim exponential in three bands: with the exp model the
    fluxes are total fluxes, recovered at the fidelity of the
    6-gaussian expansion (~0.2 percent), and colors are exact
    """
    gs_wcs = galsim.PixelScale(0.25).jacobian()
    psf_fwhms = [1.1, 0.9, 0.8]
    fluxes = [2.0, 3.5, 4.5]

    gal = galsim.Exponential(
        half_light_radius=0.5, gsparams=GSPARAMS,
    ).shear(g1=0.08, g2=0.03)

    dim = 48
    cen = (dim - 1) / 2
    jac = Jacobian(
        y=cen, x=cen,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy,
    )
    mbobs = MultiBandObsList()
    for pf, fl in zip(psf_fwhms, fluxes):
        psf = galsim.Gaussian(fwhm=pf, gsparams=GSPARAMS)
        im = galsim.Convolve(gal * fl, psf, gsparams=GSPARAMS).drawImage(
            nx=dim, ny=dim, wcs=gs_wcs).array
        psf_im = psf.drawImage(nx=dim, ny=dim, wcs=gs_wcs).array
        obslist = ObsList()
        obslist.append(Observation(
            im, jacobian=jac, weight=np.ones_like(im) * 1.0e18,
            psf=Observation(psf_im, jacobian=jac),
        ))
        mbobs.append(obslist)

    res = run_prepsf_admom(
        mbobs, guess=0.5, model='exp', rng=np.random.RandomState(3),
    )
    assert res['flags'] == 0

    fnorm = res['flux'] / np.array(fluxes)
    # total fluxes, limited by the 6-gaussian expansion fidelity
    assert np.all(np.abs(fnorm - 1) < 0.01)
    # colors are exact
    assert np.all(np.abs(fnorm / fnorm[0] - 1) < 1.0e-3)


def test_prepsfadmom_model_star():
    """
    star model: a pre-psf delta function.  The flux is recovered
    exactly for noiseless data, T is zero by construction, and with
    noise the fixed weight flux errors are exact since the weight is
    frozen
    """
    rng = np.random.RandomState(88)
    gs_wcs = galsim.PixelScale(0.25).jacobian()
    dim = 48
    cen = (dim - 1) / 2
    flux_true = 7.0

    psf = galsim.Gaussian(fwhm=0.9)
    psf_im = psf.drawImage(nx=dim, ny=dim, wcs=gs_wcs).array
    im0 = psf.withFlux(flux_true).drawImage(
        nx=dim, ny=dim, wcs=gs_wcs, offset=(-0.3, 0.4),
    ).array

    jac = Jacobian(
        y=cen, x=cen, dudx=0.25, dudy=0, dvdx=0, dvdy=0.25,
    )
    psf_obs = Observation(psf_im, jacobian=jac)

    obs = Observation(
        im0, jacobian=jac, weight=np.ones_like(im0) * 1.0e18,
        psf=psf_obs,
    )
    res = run_prepsf_admom(
        obs, model='star', fwhm_smooth=1.2,
        rng=np.random.RandomState(3),
    )
    assert res['flags'] == 0
    assert res['model'] == 'star'
    assert np.abs(res['flux'] / flux_true - 1) < 1.0e-3
    assert res['T'] == 0.0
    assert np.isnan(res['e1'])
    assert np.isnan(res['T_err'])

    # the center is recovered, in sky units
    assert np.abs(res['cen'][0] - 0.4 * 0.25) < 1.0e-3
    assert np.abs(res['cen'][1] + 0.3 * 0.25) < 1.0e-3

    # the star gmix is a T=0 delta function, allowed so it can be
    # convolved with a psf later
    gm = res.get_gmix()
    assert gm.get_T() == 0.0

    # frozen weight: the fixed weight errors are exact
    ntrial = 200
    noise = np.sqrt(np.sum(im0 ** 2)) / 20.0
    wgt = np.ones_like(im0) / noise ** 2
    fitter = PAdmomFitter(model='star', fwhm_smooth=1.2, rng=rng)
    fxs, fes = [], []
    for i in range(ntrial):
        nobs = Observation(
            im0 + rng.normal(scale=noise, size=im0.shape),
            jacobian=jac, weight=wgt, psf=psf_obs,
        )
        nres = fitter.go(nobs)
        assert nres['flags'] == 0
        fxs.append(nres['flux'])
        fes.append(nres['flux_err'])

    assert np.abs(np.mean(fes) / np.std(fxs) - 1) < 0.15
    assert np.abs(np.median(fxs) / flux_true - 1) < 0.02


def test_prepsfadmom_model_exp_noise():
    """
    exp model with noise.

    Test the fit is robust, the total flux is recovered, and the sandwich
    errors for flux, T and e match the observed scatter up to second order
    effects at this s/n
    """
    ntrial = 200
    rng = np.random.RandomState(42)
    gs_wcs = galsim.PixelScale(0.25).jacobian()
    dim = 48
    cen = (dim - 1) / 2

    jac = Jacobian(
        y=cen, x=cen, dudx=0.25, dudy=0, dvdx=0, dvdy=0.25,
    )
    psf = galsim.Gaussian(fwhm=0.9)
    psf_im = psf.drawImage(nx=dim, ny=dim, wcs=gs_wcs).array
    psf_obs = Observation(psf_im, jacobian=jac)

    gal = galsim.Exponential(
        half_light_radius=0.5,
    ).shear(e1=0.1, e2=-0.05) * 4.0
    im0 = galsim.Convolve(gal, psf).drawImage(
        nx=dim, ny=dim, wcs=gs_wcs,
    ).array
    noise = np.sqrt(np.sum(im0 ** 2)) / 20.0
    wgt = np.ones_like(im0) / noise ** 2

    fitter = PAdmomFitter(model='exp', rng=rng)
    fxs, fes, Ts, Tes, e1s, e1es = [], [], [], [], [], []
    nfail = 0
    for i in range(ntrial):
        obs = Observation(
            im0 + rng.normal(scale=noise, size=im0.shape),
            jacobian=jac, weight=wgt, psf=psf_obs,
        )
        res = fitter.go(obs, guess=0.4)
        if res['flags'] != 0:
            nfail += 1
            continue
        fxs.append(res['flux'])
        fes.append(res['flux_err'])
        Ts.append(res['T'])
        Tes.append(res['T_err'])
        e1s.append(res['e1'])
        e1es.append(res['e1err'])

    assert nfail < ntrial * 0.02
    assert np.abs(np.median(fxs) / 4.0 - 1) < 0.03
    assert 0.8 < np.mean(fes) / np.std(fxs) < 1.05
    assert 0.8 < np.mean(Tes) / np.std(Ts) < 1.05
    assert 0.8 < np.mean(e1es) / np.std(e1s) < 1.05


def test_prepsfadmom_runner():
    """
    PAdmomFitter works in the standard runner framework: a PSFRunner
    fills the psf gmix, which the fitter uses for the automatic
    smoothing choice, and the object fit runs through Runner with a
    guesser providing gmix guesses
    """
    rng = np.random.RandomState(11)
    gs_wcs = galsim.PixelScale(0.25).jacobian()

    obs0 = _make_obs(0.1, -0.05, 0.6, 3.5, 0.9, gs_wcs)
    noise = np.sqrt(np.sum(obs0.image ** 2)) / 20.0
    obs = _make_obs(
        0.1, -0.05, 0.6, 3.5, 0.9, gs_wcs, noise=noise, rng=rng,
    )

    psf_runner = ngmix.runners.PSFRunner(
        fitter=ngmix.admom.AdmomFitter(rng=rng),
        guesser=ngmix.guessers.GMixPSFGuesser(rng=rng, ngauss=1),
        ntry=2,
    )
    psf_runner.go(obs=obs)
    assert obs.psf.has_gmix()

    runner = ngmix.runners.Runner(
        fitter=PAdmomFitter(rng=rng),
        guesser=ngmix.guessers.GMixPSFGuesser(
            rng=rng, ngauss=1, guess_from_moms=True,
        ),
        ntry=2,
    )
    res = runner.go(obs=obs)
    assert res['flags'] == 0

    # the smoothing came from the fitted psf gmix, with no internal
    # psf fitting
    Tpsf = obs.psf.gmix.get_T()
    expected = 1.05 * ngmix.moments.T_to_fwhm(Tpsf)
    assert np.abs(res['fwhm_smooth'] / expected - 1) < 1.0e-8

    # sensible recovery at this s/n
    assert np.abs(res['flux'] / 3.5 - 1) < 0.2
    assert np.abs(res['T'] / 0.6 - 1) < 0.4


@pytest.mark.parametrize('model', ['gauss', 'exp'])
def test_prepsfadmom_e_cov(model):
    """
    the reported e1-e2 covariance is consistent with the empirical
    covariance of the measured shapes.  The noise correlation is
    small for these configurations, so this is a consistency check
    at the statistical precision of the trials, plus structural
    checks per fit
    """
    ntrial = 400
    rng = np.random.RandomState(77)
    gs_wcs = galsim.PixelScale(0.25).jacobian()

    if model == 'gauss':
        obs0 = _make_obs(0.35, 0.25, 0.6, 4.0, 0.9, gs_wcs)
    else:
        obs0 = _make_exp_mix_obs(0.35, 0.25, 0.5, 4.0, 0.9, gs_wcs)
    im0 = obs0.image
    noise = np.sqrt(np.sum(im0 ** 2)) / 25.0
    wgt = np.ones_like(im0) / noise ** 2

    fitter = PAdmomFitter(model=model, rng=rng)
    e1s, e2s, covs = [], [], []
    nfail = 0
    for i in range(ntrial):
        obs = Observation(
            im0 + rng.normal(scale=noise, size=im0.shape),
            jacobian=obs0.jacobian, weight=wgt, psf=obs0.psf,
        )
        res = fitter.go(obs, guess=0.5)
        if res['flags'] != 0:
            nfail += 1
            continue

        # symmetric, and consistent with the errors (the linearized
        # covariance matrix is positive semi definite)
        ecov = res['e_cov']
        assert ecov[0, 1] == ecov[1, 0]
        assert np.abs(ecov[0, 1]) <= (
            res['e1err'] * res['e2err'] * (1 + 1.0e-10)
        )

        e1s.append(res['e1'])
        e2s.append(res['e2'])
        covs.append(ecov[0, 1])

    assert nfail < ntrial * 0.02

    e1s = np.array(e1s)
    e2s = np.array(e2s)
    emp = np.cov(e1s, e2s)[0, 1]
    rep = np.mean(covs)
    se = e1s.std() * e2s.std() / np.sqrt(e1s.size)
    assert np.abs(emp - rep) < 4 * se


def test_prepsfadmommodel_sandwich_gauss_anchor():
    """
    the model sandwich evaluated for a single gaussian family reduces
    exactly to the analytic gauss delta method, for arbitrary inputs,
    and the family response is dSfam = 4 dM
    """
    from ngmix.prepsfadmom.errors import (
        flux_var_delta, model_sandwich,
    )

    rng = np.random.RandomState(9)
    Tsmooth = 0.8
    Sigma = np.array([[0.9, 0.08], [0.08, 1.1]])
    # the gauss fixed point: smoothed family covariance = weight
    Sfam = Sigma - np.diag([Tsmooth / 2] * 2)

    sums = np.zeros(6)
    sums[5] = 5.0
    G = rng.normal(size=(6, 6))
    cov = G @ G.T
    fsums = np.array([2.0, 3.0])
    fvars = np.array([0.3, 0.4])
    fmcovs = rng.normal(size=(2, 3)) * 0.1

    raw_delta = flux_var_delta(Sigma, sums, cov, fsums, fvars, fmcovs)
    raw_sw, fam_cov, fcov_raw = model_sandwich(
        'gauss', Sfam, Sigma, Tsmooth, sums, cov, fsums, fvars, fmcovs,
    )
    assert np.allclose(np.diag(fcov_raw), raw_sw, rtol=0, atol=0)
    assert np.allclose(raw_sw, raw_delta, rtol=1.0e-5, atol=0)

    # B = 1/4 for the single gaussian family, so the family
    # covariance is 16 times the measured ratio covariance
    mvec = np.array([
        Sigma[1, 1] - Sigma[0, 0], 2 * Sigma[0, 1],
        Sigma[0, 0] + Sigma[1, 1],
    ]) / 2
    css = cov[2:5, 2:5]
    csf = cov[2:5, 5]
    cff = cov[5, 5]
    cmm = (
        css - np.outer(mvec, csf) - np.outer(csf, mvec)
        + np.outer(mvec, mvec) * cff
    ) / sums[5] ** 2
    assert np.allclose(fam_cov, 16 * cmm, rtol=1.0e-4, atol=0)


@pytest.mark.parametrize('model', ['gauss', 'exp'])
def test_prepsfadmom_model_star_data(model):
    """
    Test that the star data fit with the adaptive models scatters T through
    zero for both the gauss fit (via the smoothing subtraction) and the exp
    fit (via the unconstrained family covariance).  The shape is undefined
    when the size or the family covariance is non positive, but T and the
    fluxes remain usable
    """
    ntrial = 200
    rng = np.random.RandomState(55)
    gs_wcs = galsim.PixelScale(0.25).jacobian()
    dim = 48
    cen = (dim - 1) / 2

    jac = Jacobian(
        y=cen, x=cen, dudx=0.25, dudy=0, dvdx=0, dvdy=0.25,
    )
    psf = galsim.Gaussian(fwhm=0.9)
    psf_im = psf.drawImage(nx=dim, ny=dim, wcs=gs_wcs).array
    psf_obs = Observation(psf_im, jacobian=jac)

    im0 = psf_im * 7.0
    noise = np.sqrt(np.sum(im0 ** 2)) / 25.0
    wgt = np.ones_like(im0) / noise ** 2

    fitter = PAdmomFitter(model=model, rng=rng)
    Ts, Tes = [], []
    for i in range(ntrial):
        obs = Observation(
            im0 + rng.normal(scale=noise, size=im0.shape),
            jacobian=jac, weight=wgt, psf=psf_obs,
        )
        res = fitter.go(obs, guess=0.1)
        # size and fluxes are usable even when the shape is not
        assert res['T_flags'] == 0
        assert res['flux_flags'] == 0
        Ts.append(res['T'])
        Tes.append(res['T_err'])

    Ts = np.array(Ts)
    # substantial scatter on both sides of zero, impossible with a
    # clipped size; the mean carries only the usual second order
    # admom noise bias, small compared to the per object scatter
    frac_neg = (Ts < 0).mean()
    assert 0.2 < frac_neg < 0.8
    assert np.abs(np.mean(Ts)) < 0.5 * np.std(Ts)

    # the size errors are calibrated through zero
    assert np.abs(np.mean(Tes) / np.std(Ts) - 1) < 0.15


def test_prepsfadmom_model_exp_highe():
    """
    a highly elliptical exp
    """
    e1_true, e2_true, T_true, flux_true = 0.92, 0.05, 0.5, 4.0
    gs_wcs = galsim.PixelScale(0.25).jacobian()

    obs = _make_exp_mix_obs(
        e1_true, e2_true, T_true, flux_true, 0.9, gs_wcs, dim=64,
    )
    res = run_prepsf_admom(
        obs, guess=0.4, model='exp', rng=np.random.RandomState(3),
    )
    assert res['flags'] == 0
    assert np.abs(res['e1'] - e1_true) < 1.0e-3
    assert np.abs(res['e2'] - e2_true) < 1.0e-3
    assert np.abs(res['T'] / T_true - 1) < 2.0e-3
    assert np.abs(res['flux'] / flux_true - 1) < 1.0e-3


def test_prepsfadmom_model_errors():
    """
    model error conditions raise
    """
    with pytest.raises(ValueError):
        PAdmomFitter(model='not-a-model')

    gs_wcs = galsim.PixelScale(0.25).jacobian()
    obs = _make_obs(0.1, -0.05, 0.6, 3.5, 0.9, gs_wcs)

    # the star model requires positive smoothing
    with pytest.raises(ValueError):
        run_prepsf_admom(
            obs, model='star', fwhm_smooth=0,
            rng=np.random.RandomState(3),
        )


def test_prepsfadmom_model_ksums_kernel_parity():
    """
    the numba kernel backed model_ksums must match the python
    gauss_model_ksums reference summed over components, for all model
    types and both exp parametrizations, with offsets and a
    non-trivial k-space area factor
    """
    from ngmix.prepsfadmom.models import (
        model_ksums, gauss_model_ksums, get_profile_comps, cov_from_e,
    )

    Tsmooth = 0.35
    smooth_cov = np.diag([Tsmooth / 2, Tsmooth / 2])
    Sw = np.array([[0.31, 0.04], [0.04, 0.27]])
    dv, du = 0.83, -1.21
    detAtinv = 3.7
    F = np.array([2.5, 1.5])

    Sfam = cov_from_e(0.2, -0.1, 0.8)
    models = [
        {'type': 'gauss', 'cov_sm': cov_from_e(0.1, -0.05, 0.6) + smooth_cov,
         'F': F},
        {'type': 'star', 'cov_sm': smooth_cov, 'F': F},
        {'type': 'exp', 'e1': 0.2, 'e2': -0.1, 'T': 0.8, 'F': F},
        {'type': 'exp', 'cov': Sfam, 'F': F},
        {'type': 'dev', 'e1': 0.2, 'e2': -0.1, 'T': 0.8, 'F': F},
        {'type': 'dev', 'cov': Sfam, 'F': F},
    ]

    for model in models:
        for band in range(F.size):
            if model['type'] in ('exp', 'dev'):
                expected = np.zeros(6)
                for frac, cT in get_profile_comps(model['type']):
                    So = cT * Sfam + smooth_cov
                    expected += gauss_model_ksums(
                        model['F'][band] * frac, So, dv, du, Sw, detAtinv,
                    )
            else:
                expected = gauss_model_ksums(
                    model['F'][band], model['cov_sm'], dv, du, Sw, detAtinv,
                )

            sums = model_ksums(
                model, band, dv, du, Sw, detAtinv, Tsmooth,
            )
            assert np.allclose(sums, expected, rtol=1.0e-12, atol=1.0e-14)


def test_prepsfadmom_e_flags():
    """
    e_flags marks unusable ellipticities while the per-quantity T and
    flux flags stay clean
    """
    gs_wcs = galsim.PixelScale(0.25).jacobian()
    obs = _make_obs(0.1, -0.05, 0.6, 3.5, 0.9, gs_wcs)
    rng = np.random.RandomState(9)

    # a clean fit has usable shapes
    res = run_prepsf_admom(obs, model='gauss', rng=rng)
    assert res['flags'] == 0
    assert res['e_flags'] == 0
    assert np.all(np.isfinite(res['e']))

    # a star has no shape by construction: the overall flags stay
    # clean but e_flags is set
    res = run_prepsf_admom(obs, model='star', rng=rng)
    assert res['flags'] == 0
    assert res['e_flags'] != 0
    assert res['T_flags'] == 0
    assert res['flux_flags'] == 0
    assert np.all(np.isnan(res['e']))


def test_prepsfadmom_deweight_nonpos():
    """
    measured moments exceeding the weight in both eigendirections
    give N = M^-1 - Sigma^-1 negative definite with positive
    determinant; deweight must flag instead of returning a negative
    definite weight, which would drive chi2 < 0 in the k-sum kernels
    and index the fastexp lookup table out of range
    """
    from ngmix.prepsfadmom import deweight

    Sigma = np.diag([0.5, 0.5])
    M = np.diag([1.0, 1.0])
    newSigma, flags = deweight(M, Sigma)
    assert flags != 0


def test_prepsfadmom_ksums_nonpos_weight_guard():
    """
    the k-sum kernel must return finite (zero) sums for a non
    positive definite weight rather than reading past the fastexp
    table
    """
    from ngmix.prepsfadmom.prepsfadmom_nb import admom_ksums

    dim = 8
    iy = np.array([0, 1, 2])
    ix = np.array([1, 2, 3])
    kim = np.ones(3, dtype=np.complex128)
    kv = np.array([0.1, 0.2, 0.3])
    ku = np.array([0.2, 0.1, 0.1])
    sums = np.zeros(6)

    admom_ksums(
        kim, iy, ix, dim, 0.0, 0.0, kv, ku,
        -1.0, 0.0, -1.0, 1.0, sums,
    )
    assert np.all(np.isfinite(sums))
    assert np.all(sums == 0)


def test_prepsfadmom_model_dev():
    """
    noiseless exact 10-gaussian de Vaucouleurs: the moment matched
    family parameters and the total flux are recovered, including
    with an offset center and a sheared wcs
    """
    e1_true, e2_true, T_true, flux_true = 0.1, -0.05, 0.5, 4.0

    gs_wcs = galsim.ShearWCS(
        0.25, galsim.Shear(g1=-0.1, g2=0.06),
    ).jacobian()

    obs = _make_exp_mix_obs(
        e1_true, e2_true, T_true, flux_true, 0.9, gs_wcs,
        offset_pix=(0.4, -0.3), model='dev', dim=96,
    )
    res = run_prepsf_admom(
        obs, guess=0.4, model='dev', rng=np.random.RandomState(3),
    )
    assert res['flags'] == 0
    assert res['model'] == 'dev'
    assert np.abs(res['e1'] - e1_true) < 1.0e-2
    assert np.abs(res['e2'] - e2_true) < 1.0e-2
    assert np.abs(res['T'] / T_true - 1) < 2.0e-2
    assert np.abs(res['flux'] / flux_true - 1) < 1.0e-2


def test_prepsfadmom_robustness_vs_admom():
    """
    engine-level robustness against regular real-space adaptive
    moments on identical noisy observations.  Hard failures (flux
    and T unusable) occur at the same rate as regular admom
    failures, while the post-hoc undefined-shape case (deweighted
    galaxy covariance not positive definite, e_flags set with T
    and flux still usable) is the separate price of pre-psf shapes
    at the resolution limit.  Reference rates from a 2000-trial
    grid: hard-failure rates agree within 0.02 for s2n 5-50 and
    T/Tpsf 0.1-2; shape availability at the cells below is
    0.38, 0.88 and 0.44
    """
    rng = np.random.RandomState(771)
    gs_wcs = galsim.ShearWCS(0.2, galsim.Shear(g1=0, g2=0)).jacobian()
    psf_fwhm = 0.8
    Tpsf = fwhm_to_T(psf_fwhm)
    ntrial = 150

    for s2n, trat, shape_min in [
        (10, 0.25, 0.20),
        (10, 1.00, 0.75),
        (7, 0.50, 0.30),
    ]:
        T = trat * Tpsf

        # noise for the target matched-filter s2n, from a round
        # noiseless template at this size
        obs0 = _make_obs(0, 0, T, 1.0, psf_fwhm, gs_wcs)
        noise = np.sqrt(np.sum(obs0.image ** 2)) / s2n

        npa_flux = npa_T = npa_shape = nam = 0
        for k in range(ntrial):
            e1, e2 = rng.uniform(-0.3, 0.3, size=2)
            off = rng.uniform(-0.5, 0.5, size=2)
            obs = _make_obs(
                e1, e2, T, 1.0, psf_fwhm, gs_wcs,
                offset_pix=off, noise=noise, rng=rng,
            )

            pres = run_prepsf_admom(
                obs, guess=T, model='gauss',
                rng=np.random.RandomState(rng.randint(2 ** 31)),
            )
            npa_flux += (pres['flux_flags'] == 0)
            npa_T += (pres['T_flags'] == 0)
            npa_shape += (pres['e_flags'] == 0)

            ares = ngmix.admom.run_admom(
                obs, guess=T + Tpsf,
                rng=np.random.RandomState(rng.randint(2 ** 31)),
            )
            nam += (ares['flags'] == 0)

        # shape usable implies T usable
        assert npa_shape <= npa_T

        # hard-failure robustness matches regular admom
        assert abs(npa_flux - nam) <= 0.06 * ntrial

        # both engines nearly always give usable flux/T here
        assert npa_flux >= 0.90 * ntrial
        assert nam >= 0.90 * ntrial

        # pre-psf shape availability floor for this cell
        assert npa_shape >= shape_min * ntrial


def test_prepsfadmom_model_flux_cov():
    """
    the cross-band flux covariance from the shared family response
    (result['flux_cov']) matches the empirical covariance over
    noise realizations; the independence combination over-predicts
    the color variance by the shared term
    """
    ntrial = 400
    rng = np.random.RandomState(6021)
    gs_wcs = galsim.PixelScale(0.25).jacobian()

    e1t, e2t, Tt = 0.1, -0.05, 0.6
    fluxes_true = [3.5, 5.0]
    obs0 = _make_obs(e1t, e2t, Tt, fluxes_true[0], 0.9, gs_wcs)
    noise = np.sqrt(np.sum(obs0.image ** 2)) / 12.0

    fitter = PAdmomFitter(rng=rng, model='exp')

    fpairs, rcovs = [], []
    for i in range(ntrial):
        mbobs = MultiBandObsList()
        for ft in fluxes_true:
            obs = _make_obs(
                e1t, e2t, Tt, ft, 0.9, gs_wcs,
                noise=noise, rng=rng,
            )
            ol = ObsList()
            ol.append(obs)
            mbobs.append(ol)
        res = fitter.go(mbobs, guess=0.6)
        if res['flags'] != 0:
            continue
        fpairs.append(res['flux'])
        rcovs.append(res['flux_cov'])
    fpairs = np.array(fpairs)
    rcovs = np.array(rcovs)
    assert len(fpairs) > 0.9 * ntrial

    # reported diagonal is flux_err ** 2 by construction; check the
    # off diagonal against the empirical cross-band covariance
    emp = np.cov(fpairs.T)
    rep = rcovs.mean(axis=0)
    se01 = np.sqrt(
        (emp[0, 0] * emp[1, 1] + emp[0, 1] ** 2) / len(fpairs)
    )
    assert np.abs(rep[0, 1] - emp[0, 1]) < 4 * se01
    # the shared family response correlates the bands positively
    assert rep[0, 1] > 3 * se01

    # color variance: the covariance-aware prediction matches the
    # empirical log-ratio variance; independence over-predicts
    lr = np.log(fpairs[:, 0] / fpairs[:, 1])
    f0, f1 = fpairs[:, 0].mean(), fpairs[:, 1].mean()
    pred_cov = (
        rep[0, 0] / f0 ** 2 + rep[1, 1] / f1 ** 2
        - 2 * rep[0, 1] / (f0 * f1)
    )
    pred_ind = rep[0, 0] / f0 ** 2 + rep[1, 1] / f1 ** 2
    ratio = pred_cov / lr.var()
    assert 0.75 < ratio < 1.3
    assert pred_ind / lr.var() > ratio * 1.1
