"""
Tests for the noise-power sandwich covariance available through
Fitter(..., use_noise_image=True).

The closure tests fit an exp model to many noise realizations and
compare the empirical scatter of the recovered flux to the
reported errors via pulls:

  white noise      the sandwich must agree with the standard LM
                   errors, and both must pull at 1
  correlated noise (white noise convolved with a small kernel,
                   weight map set to the true pixel variance so
                   chi^2/dof is 1 and the standard errors do not
                   adapt) the standard errors understate the flux
                   variance; the sandwich must pull at 1
"""
import numpy as np
import pytest

import ngmix


PIXEL_SCALE = 0.2
PSF_FWHM = 0.8
DIM = 49
PSF_DIM = 33
HLR = 0.5
FLUX = 1000.0
TGUESS = 0.36


def _smooth(im, kernel):
    from scipy.signal import fftconvolve
    return fftconvolve(im, kernel, mode='same')


def _off_clip_ring(fit_model, pars, band, obs, margin=0.2):
    """
    mask of pixels away from the chi^2 = FASTEXP_MAX_CHI2
    truncation boundary of every composed gaussian.  The
    central-difference reference renders the clipped model, so it
    differentiates the migrating truncation boundary; that is a
    property of the clipped objective, not of the derivative
    approximation under test.  The fd steps move the rings by
    |dchi2| < 0.03, well inside the margin
    """
    from ngmix.fastexp_nb import FASTEXP_MAX_CHI2

    band_pars = fit_model.get_band_pars(pars=pars, band=band)
    gm = ngmix.gmix.make_gmix_model(band_pars, fit_model.model)
    if obs.has_psf_gmix():
        gm = gm.convolve(obs.psf.gmix)
    gpars = gm.get_full_pars().reshape(-1, 6)

    dims = obs.image.shape
    rows, cols = np.mgrid[0:dims[0], 0:dims[1]]
    vv, uu = obs.jacobian.get_vu(
        row=rows.ravel().astype('f8'),
        col=cols.ravel().astype('f8'),
    )
    off = np.ones(vv.size, dtype=bool)
    for _, vc, uc, irr, irc, icc in gpars:
        det = irr * icc - irc * irc
        dv = vv - vc
        du = uu - uc
        chi2 = (
            icc * dv ** 2 - 2 * irc * dv * du + irr * du ** 2
        ) / det
        off &= np.abs(chi2 - FASTEXP_MAX_CHI2) > margin
    return off.reshape(dims)


def _make_obs(rng, sigma, kernel=None, nband=1, with_noise=True):
    import galsim

    psf = galsim.Gaussian(fwhm=PSF_FWHM)
    psf_im = psf.drawImage(
        nx=PSF_DIM, ny=PSF_DIM, scale=PIXEL_SCALE,
    ).array
    psf_cen = (PSF_DIM - 1) / 2
    psf_jac = ngmix.DiagonalJacobian(
        scale=PIXEL_SCALE, row=psf_cen, col=psf_cen,
    )

    gal = galsim.Convolve(
        galsim.Exponential(half_light_radius=HLR, flux=FLUX),
        psf,
    )
    im0 = gal.drawImage(
        nx=DIM, ny=DIM, scale=PIXEL_SCALE,
    ).array
    cen = (DIM - 1) / 2
    jac = ngmix.DiagonalJacobian(
        scale=PIXEL_SCALE, row=cen, col=cen,
    )

    # fit the psf for the gmix used in convolution
    psf_obs = ngmix.Observation(
        psf_im.copy(),
        weight=np.ones_like(psf_im) * 1.0e12,
        jacobian=psf_jac,
    )
    am = ngmix.admom.AdmomFitter(rng=rng)
    pres = am.go(psf_obs, guess=0.1)
    psf_obs.set_gmix(pres.get_gmix())

    mbobs = ngmix.MultiBandObsList()
    for _ in range(nband):
        nz = rng.normal(scale=sigma, size=im0.shape)
        nz_extra = rng.normal(scale=sigma, size=im0.shape)
        if kernel is not None:
            nz = _smooth(nz, kernel)
            nz_extra = _smooth(nz_extra, kernel)
            pixvar = sigma ** 2 * np.sum(kernel ** 2)
        else:
            pixvar = sigma ** 2

        kw = {}
        if with_noise:
            kw['noise'] = nz_extra
        obs = ngmix.Observation(
            im0 + nz,
            weight=np.full(im0.shape, 1.0 / pixvar),
            jacobian=jac,
            psf=psf_obs.copy(),
            **kw
        )
        obslist = ngmix.ObsList()
        obslist.append(obs)
        mbobs.append(obslist)
    return mbobs


def _get_prior(rng):
    return ngmix.joint_prior.PriorSimpleSep(
        cen_prior=ngmix.priors.CenPrior(
            0.0, 0.0, PIXEL_SCALE, PIXEL_SCALE, rng=rng,
        ),
        g_prior=ngmix.priors.GPriorBA(0.3, rng=rng),
        T_prior=ngmix.priors.TwoSidedErf(
            -1.0, 0.1, 1.0e3, 1.0, rng=rng,
        ),
        F_prior=ngmix.priors.TwoSidedErf(
            -1.0e2, 1.0, 1.0e9, 1.0e8, rng=rng,
        ),
    )


def _fit_many(rng, sigma, kernel, use_noise_image, ntrial, nband=1):
    prior = _get_prior(rng)
    fitter = ngmix.fitting.Fitter(
        model='exp', prior=prior,
        use_noise_image=use_noise_image,
    )
    pulls = []
    for _ in range(ntrial):
        mbobs = _make_obs(rng, sigma, kernel=kernel, nband=nband)
        guess = np.concatenate([
            rng.uniform(-0.01, 0.01, 2),
            rng.uniform(-0.02, 0.02, 2),
            [TGUESS * rng.uniform(0.9, 1.1)],
            FLUX * rng.uniform(0.9, 1.1, nband),
        ])
        res = fitter.go(obs=mbobs, guess=guess)
        if res['flags'] != 0:
            continue
        flux = np.atleast_1d(res['flux'])
        flux_err = np.atleast_1d(res['flux_err'])
        pulls.append((flux - FLUX) / flux_err)
    return np.array(pulls)


def test_noise_cov_white_closure():
    """on white noise the sandwich must agree with the standard
    LM errors and pull at 1"""
    rng = np.random.RandomState(11)
    sigma = 8.0

    pulls_std = _fit_many(
        rng, sigma, kernel=None, use_noise_image=False, ntrial=100,
    )
    pulls_sand = _fit_many(
        rng, sigma, kernel=None, use_noise_image=True, ntrial=100,
    )
    assert abs(pulls_std.std() - 1) < 0.2
    assert abs(pulls_sand.std() - 1) < 0.2


def test_noise_cov_correlated_closure():
    """on correlated noise with an honest pixel-variance weight
    map the standard errors understate the flux variance; the
    sandwich must restore pulls of 1"""
    rng = np.random.RandomState(31)
    sigma = 8.0
    kernel = np.ones((3, 3)) / 9

    pulls_std = _fit_many(
        rng, sigma, kernel=kernel, use_noise_image=False,
        ntrial=100,
    )
    pulls_sand = _fit_many(
        rng, sigma, kernel=kernel, use_noise_image=True,
        ntrial=100,
    )
    # positively correlated noise inflates the flux variance well
    # beyond the white expectation
    assert pulls_std.std() > 1.5
    assert abs(pulls_sand.std() - 1) < 0.2


def test_noise_cov_multiband():
    """multi-band fits get per-band sandwich flux errors"""
    rng = np.random.RandomState(51)
    sigma = 8.0
    kernel = np.ones((3, 3)) / 9

    pulls = _fit_many(
        rng, sigma, kernel=kernel, use_noise_image=True,
        ntrial=50, nband=3,
    )
    assert pulls.shape[1] == 3
    for band in range(3):
        assert abs(pulls[:, band].std() - 1) < 0.25


def test_noise_cov_boundary_flags():
    """a solution at the parameter-space boundary makes the
    derivative model unevaluable (the gmix construction guards
    |g| near 1, for the analytic path and the stepped one
    alike); the sandwich must set the covariance flags, not
    raise"""
    from ngmix.fitting.noise_cov import apply_noise_cov

    rng = np.random.RandomState(91)
    mbobs = _make_obs(rng, 8.0)
    prior = _get_prior(rng)
    fitter = ngmix.fitting.Fitter(model='exp', prior=prior)
    guess = np.array([0.0, 0.0, 0.0, 0.0, TGUESS, FLUX])
    res = fitter.go(obs=mbobs, guess=guess)
    assert res['flags'] == 0

    res['pars'][2] = 1.0 - 5.0e-5
    apply_noise_cov(fit_model=res, result=res)
    assert res['flags'] != 0
    assert 'bad noise covariance' in res['errmsg']


@pytest.mark.parametrize('model', ['gauss', 'exp', 'dev'])
def test_noise_cov_analytic_vs_fd(model):
    """the analytic derivative images agree with the central
    differences away from the boundary, both image by image and
    through the assembled sandwich covariance"""
    from ngmix.fitting import noise_cov
    from ngmix.fitting.noise_cov import (
        calc_noise_cov, _dmodel_images,
    )

    rng = np.random.RandomState(55)
    mbobs = _make_obs(rng, 4.0, nband=2)
    prior = _get_prior(rng)
    fitter = ngmix.fitting.Fitter(model=model, prior=prior)
    guess = np.array([0.0, 0.0, 0.0, 0.0, TGUESS, FLUX, FLUX])
    fit_model = fitter.go(obs=mbobs, guess=guess)
    assert fit_model['flags'] == 0
    pars = fit_model['pars']
    npars = pars.size
    nshape = npars - 2

    for band in range(2):
        kpars = list(range(nshape)) + [nshape + band]
        for obs in fit_model.obs[band]:
            ana = _dmodel_images(
                fit_model=fit_model, pars=pars, band=band,
                obs=obs, kpars=kpars,
            )
            ref = _dmodel_images(
                fit_model=fit_model, pars=pars, band=band,
                obs=obs, kpars=kpars, force_fd=True,
            )
            # close but not identical: bitwise equality
            # would mean the analytic branch silently fell
            # back to the finite differences
            assert not np.array_equal(ana[0], ref[0])
            # compare away from the chi^2 truncation rings of
            # the composed gaussians, which the differences
            # differentiate (see _off_clip_ring).  With the
            # smooth fexp the only remaining differences are
            # its derivative approximation (~3e-5 relative)
            # and the fd truncation error
            off_ring = _off_clip_ring(
                fit_model=fit_model, pars=pars, band=band,
                obs=obs,
            )
            for a, r in zip(ana, ref):
                scale = np.abs(r).max()
                assert np.allclose(
                    a[off_ring], r[off_ring],
                    atol=1.0e-4 * scale,
                )

    cov = calc_noise_cov(
        fit_model=fit_model, pars=pars,
        pars_cov0=fit_model['pars_cov0'],
    )
    saved = noise_cov._ANALYTIC_MODELS
    noise_cov._ANALYTIC_MODELS = ()
    try:
        ref = calc_noise_cov(
            fit_model=fit_model, pars=pars,
            pars_cov0=fit_model['pars_cov0'],
        )
    finally:
        noise_cov._ANALYTIC_MODELS = saved
    assert np.allclose(
        np.sqrt(np.diag(cov)), np.sqrt(np.diag(ref)),
        rtol=2.0e-4,
    )
    # off-diagonal deviations measured against the error scale:
    # elementwise relative tolerances blow up on the near-zero
    # correlation elements
    escale = np.sqrt(np.outer(np.diag(ref), np.diag(ref)))
    assert np.all(np.abs(cov - ref) < 4.0e-4 * escale)


def test_noise_cov_requires_noise():
    """use_noise_image without an attached noise image raises"""
    rng = np.random.RandomState(71)
    mbobs = _make_obs(rng, 8.0, with_noise=False)
    fitter = ngmix.fitting.Fitter(
        model='exp', use_noise_image=True,
    )
    guess = np.array([0.0, 0.0, 0.0, 0.0, TGUESS, FLUX])
    with pytest.raises(ValueError):
        fitter.go(obs=mbobs, guess=guess)
