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
