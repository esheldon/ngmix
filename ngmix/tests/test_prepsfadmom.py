import galsim
import numpy as np
import pytest

import ngmix
from ngmix.moments import fwhm_to_T
from ngmix import Jacobian, Observation, ObsList, MultiBandObsList
from ngmix.prepsfadmom import run_prepsf_admom, PrePSFAdmomFitter
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

    fitter = PrePSFAdmomFitter(rng=rng)

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

    # the errors assume a fixed weight, as in real-space adaptive
    # moments.  The shape errors match the scatter well; the flux
    # errors underestimate the scatter because the flux couples to the
    # fitted size, at the same level as for real-space admom (~30%
    # at this s/n)
    assert 0.55 < np.mean(fluxerrs) / np.std(fluxes) < 1.1
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

    fitter = PrePSFAdmomFitter(rng=rng)

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

    # non square images raise
    im = np.zeros((48, 50))
    jac = ngmix.DiagonalJacobian(scale=0.25, row=23.5, col=24.5)
    obs_rect = Observation(im, jacobian=jac)
    with pytest.raises(ValueError):
        run_prepsf_admom(
            obs_rect, guess=0.5, no_psf=True, fwhm_smooth=1.0,
            rng=np.random.RandomState(3),
        )
