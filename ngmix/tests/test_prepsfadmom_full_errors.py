"""
tests for the full-errors building blocks: the stored
image->kim transfer against an impulse through the identical
prep, the closed-form kernels against admom_ksums, the analytic
theta derivatives against finite differences of the exact-exp
kernels, the rfft influence-kernel path against the brute fft2
reference, and the sums covariance against empirical draws
"""
import galsim
import numpy as np

import ngmix
from ngmix.prepsfadmom.prep import prep_epoch
from ngmix.prepsfadmom.full_errors import (
    moment_kernels, dsums_dtheta, influence_kernels, sums_cov,
)

SCALE = 0.2
PSF_FWHM = 0.8
DIM = 48
PSF_DIM = 33
FWHM_SMOOTH = 0.9


def _make_obs(rng, noise=3.0):
    psf = galsim.Gaussian(fwhm=PSF_FWHM)
    psf_im = psf.drawImage(
        nx=PSF_DIM, ny=PSF_DIM, scale=SCALE,
    ).array
    psf_cen = (PSF_DIM - 1) / 2
    psf_obs = ngmix.Observation(
        psf_im.copy(),
        weight=np.ones_like(psf_im) * 1.0e12,
        jacobian=ngmix.DiagonalJacobian(
            scale=SCALE, row=psf_cen, col=psf_cen,
        ),
    )
    gal = galsim.Convolve(
        galsim.Exponential(half_light_radius=0.5, flux=1000.0),
        psf,
    )
    im = gal.drawImage(nx=DIM, ny=DIM, scale=SCALE).array
    im = im + rng.normal(scale=noise, size=im.shape)
    cen = (DIM - 1) / 2
    return ngmix.Observation(
        im,
        weight=np.full(im.shape, 1.0 / noise ** 2),
        jacobian=ngmix.DiagonalJacobian(
            scale=SCALE, row=cen, col=cen,
        ),
        psf=psf_obs,
    )


def _prep(obs):
    return prep_epoch(
        obs, band=0, fwhm_smooth=FWHM_SMOOTH, ap_rad=0,
        store_transfer=True,
    )


def test_full_errors_ktransfer_impulse():
    """the stored transfer equals the kim of a unit impulse at
    image pixel (0, 0) through the identical prep"""
    rng = np.random.RandomState(5)
    obs = _make_obs(rng)
    ep = _prep(obs)
    assert ep['ktransfer'] is not None

    im = np.zeros(obs.image.shape)
    im[0, 0] = 1.0
    iobs = ngmix.Observation(
        im,
        weight=obs.weight.copy(),
        jacobian=obs.jacobian.copy(),
        psf=obs.psf.copy(),
    )
    iep = _prep(iobs)
    assert np.allclose(
        ep['ktransfer'], iep['kim'], rtol=1e-10, atol=1e-13,
    )


def test_full_errors_kernels_vs_admom_ksums():
    """the closed-form kernels reproduce admom_ksums to the
    accuracy of its table exponential"""
    from ngmix.prepsfadmom.prepsfadmom import get_phase_angles
    from ngmix.prepsfadmom.prepsfadmom_nb import admom_ksums

    rng = np.random.RandomState(7)
    obs = _make_obs(rng)
    ep = _prep(obs)
    Sw = np.array([[0.5, 0.05], [0.05, 0.6]])
    v0, u0 = 0.13, -0.21

    G = moment_kernels(ep, Sw, v0, u0)
    kim = (
        rng.normal(size=ep['kim'].shape)
        + 1j * rng.normal(size=ep['kim'].shape)
    )
    sums = np.zeros(6)
    alpha, beta = get_phase_angles(ep, v0, u0)
    admom_ksums(
        kim, ep['iy'], ep['ix'], ep['dim'], alpha, beta,
        ep['kv'], ep['ku'], Sw[0, 0], Sw[0, 1], Sw[1, 1],
        ep['df2'], sums,
    )
    assert np.allclose(
        (G @ kim).real, sums, rtol=5e-4, atol=1e-12,
    )


def test_full_errors_dsums_dtheta():
    """the analytic theta derivatives match finite differences
    of the exact-exp kernel contraction"""
    rng = np.random.RandomState(9)
    obs = _make_obs(rng)
    ep = _prep(obs)
    Sw0 = np.array([[0.5, 0.05], [0.05, 0.6]])
    v0, u0 = 0.13, -0.21

    D = dsums_dtheta(ep, Sw0, v0, u0)
    Dfd = np.zeros((6, 5))
    hs = [1e-5, 1e-5, 1e-5, 1e-6, 1e-6]
    for t in range(5):
        for sign in (1, -1):
            Sw = Sw0.copy()
            v, u = v0, u0
            if t < 3:
                r, c = [(0, 0), (0, 1), (1, 1)][t]
                Sw[r, c] += sign * hs[t]
                Sw[c, r] = Sw[r, c]
            elif t == 3:
                v = v0 + sign * hs[t]
            else:
                u = u0 + sign * hs[t]
            sums = (
                moment_kernels(ep, Sw, v, u) @ ep['kim']
            ).real
            Dfd[:, t] += sign * sums / (2 * hs[t])
    rel = np.abs(D - Dfd) / (np.abs(Dfd) + 1e-8)
    assert rel.max() < 1e-5


def test_full_errors_influence_rfft_vs_reference():
    """the rfft influence-kernel path equals the brute full-grid
    fft2 construction"""
    rng = np.random.RandomState(11)
    obs = _make_obs(rng)
    ep = _prep(obs)
    Sw = np.array([[0.5, 0.05], [0.05, 0.6]])
    G = moment_kernels(ep, Sw, 0.1, -0.1)

    h_ref = influence_kernels(
        ep, G, obs.image.shape, reference=True,
    )
    h = influence_kernels(ep, G, obs.image.shape)
    assert np.allclose(h, h_ref, rtol=1e-9, atol=1e-12)


def test_full_errors_sums_cov_empirical():
    """the influence-kernel covariance matches the empirical
    covariance of the sums over noise draws through the prep"""
    rng = np.random.RandomState(13)
    obs = _make_obs(rng)
    ep = _prep(obs)
    Sw = np.array([[0.5, 0.05], [0.05, 0.6]])
    G = moment_kernels(ep, Sw, 0.0, 0.0)

    h = influence_kernels(ep, G, obs.image.shape)
    cov = sums_cov(h, obs.weight)

    draws = []
    for _ in range(4000):
        nobs = ngmix.Observation(
            rng.normal(scale=3.0, size=obs.image.shape),
            weight=obs.weight.copy(),
            jacobian=obs.jacobian.copy(),
            psf=obs.psf.copy(),
        )
        nep = _prep(nobs)
        draws.append(nep['kim'])
    S = (np.array(draws) @ G.T).real
    covE = np.cov(S.T)
    dd = np.sqrt(np.diag(cov) / np.diag(covE))
    assert np.all(np.abs(dd - 1) < 0.08)


def _make_gal_obs(rng, profile, noise=3.0):
    psf = galsim.Gaussian(fwhm=PSF_FWHM)
    psf_im = psf.drawImage(
        nx=PSF_DIM, ny=PSF_DIM, scale=SCALE,
    ).array
    psf_cen = (PSF_DIM - 1) / 2
    psf_obs = ngmix.Observation(
        psf_im.copy(),
        weight=np.ones_like(psf_im) * 1.0e12,
        jacobian=ngmix.DiagonalJacobian(
            scale=SCALE, row=psf_cen, col=psf_cen,
        ),
    )
    if profile == 'exp':
        gal = galsim.Exponential(
            half_light_radius=0.5, flux=1000.0,
        )
    else:
        gal = galsim.DeVaucouleurs(
            half_light_radius=0.5, flux=1000.0, trunc=4.0,
        )
    dim = 64
    im = galsim.Convolve(gal, psf).drawImage(
        nx=dim, ny=dim, scale=SCALE,
    ).array
    im = im + rng.normal(scale=noise, size=im.shape)
    cen = (dim - 1) / 2
    return ngmix.Observation(
        im,
        weight=np.full(im.shape, 1.0 / noise ** 2),
        jacobian=ngmix.DiagonalJacobian(
            scale=SCALE, row=cen, col=cen,
        ),
        psf=psf_obs,
    )


def test_padmom_full_errors_validation():
    """full_errors requires exp/dev and ap_rad=0"""
    import pytest
    from ngmix.prepsfadmom import PAdmomFitter

    with pytest.raises(ValueError):
        PAdmomFitter(model='star', full_errors=True, ap_rad=0)
    with pytest.raises(ValueError):
        PAdmomFitter(model='exp', full_errors=True, ap_rad=1.5)
    PAdmomFitter(model='exp', full_errors=True, ap_rad=0)
    PAdmomFitter(model='gauss', full_errors=True, ap_rad=0)


def test_padmom_full_errors_matched():
    """on matched exp truth the full errors agree with the
    model_sandwich errors and fill a consistent flux_cov"""
    from ngmix.prepsfadmom import PAdmomFitter

    rng = np.random.RandomState(17)
    obs = _make_gal_obs(rng, 'exp')
    f0 = PAdmomFitter(
        model='exp', ap_rad=0, rng=np.random.RandomState(3),
    )
    f1 = PAdmomFitter(
        model='exp', ap_rad=0, full_errors=True,
        rng=np.random.RandomState(3),
    )
    r0 = f0.go(obs, guess=0.5)
    r1 = f1.go(obs, guess=0.5)
    assert r1['flags'] == 0
    assert np.abs(r1['T_err'] / r0['T_err'] - 1) < 0.2
    assert np.abs(r1['flux_err'] / r0['flux_err'] - 1) < 0.2
    assert np.abs(r1['e1err'] / r0['e1err'] - 1) < 0.25
    fcov = r1['flux_cov']
    assert fcov.shape == (1, 1) and fcov[0, 0] > 0
    assert np.allclose(
        np.sqrt(fcov[0, 0]), r1['flux_err'], rtol=1e-6,
    )


def test_padmom_full_errors_mismatch_mc():
    """dev truth fit with exp: the sandwich under-predicts the
    T and flux errors; the full errors stay calibrated"""
    from ngmix.prepsfadmom import PAdmomFitter

    ntrial = 200
    rng = np.random.RandomState(19)
    T0, Tf0, F0, Ff0 = [], [], [], []
    for k in range(ntrial):
        obs = _make_gal_obs(rng, 'dev')
        f1 = PAdmomFitter(
            model='exp', ap_rad=0, full_errors=True,
            rng=np.random.RandomState(k),
        )
        res = f1.go(obs, guess=0.5)
        if res['flags'] != 0 or res['T_flags'] != 0:
            continue
        T0.append(res['T'])
        Tf0.append(res['T_err'])
        F0.append(res['flux'])
        Ff0.append(res['flux_err'])
    T0, Tf0 = np.array(T0), np.array(Tf0)
    F0, Ff0 = np.array(F0), np.array(Ff0)
    assert len(T0) > 0.9 * ntrial

    rT = T0.std() / np.sqrt(np.mean(Tf0 ** 2))
    rF = F0.std() / np.sqrt(np.mean(Ff0 ** 2))
    # the sandwich reads ~1.17 (T) and ~1.11 (flux) here
    assert 0.8 < rT < 1.15
    assert 0.85 < rF < 1.15


def test_padmom_full_errors_gauss():
    """the gauss estimator with full errors: agreement with the
    delta-method errors on matched gauss-ish truth, and a
    calibrated MC on dev truth"""
    from ngmix.prepsfadmom import PAdmomFitter

    rng = np.random.RandomState(23)
    obs = _make_gal_obs(rng, 'exp')
    f0 = PAdmomFitter(
        model='gauss', ap_rad=0, rng=np.random.RandomState(3),
    )
    f1 = PAdmomFitter(
        model='gauss', ap_rad=0, full_errors=True,
        rng=np.random.RandomState(3),
    )
    r0 = f0.go(obs, guess=0.5)
    r1 = f1.go(obs, guess=0.5)
    assert r1['flags'] == 0
    assert np.isfinite(r1['T_err']) and np.isfinite(
        r1['flux_err'],
    )
    assert r1['flux_cov'].shape == (1, 1)
    assert np.allclose(
        np.sqrt(r1['flux_cov'][0, 0]), r1['flux_err'],
        rtol=1e-6,
    )
    # same order as the delta method on this smooth scene
    assert 0.7 < r1['T_err'] / r0['T_err'] < 1.4
    assert 0.7 < r1['flux_err'] / r0['flux_err'] < 1.4

    ntrial = 200
    rng = np.random.RandomState(29)
    T, Te, F, Fe = [], [], [], []
    for k in range(ntrial):
        obs = _make_gal_obs(rng, 'dev')
        res = PAdmomFitter(
            model='gauss', ap_rad=0, full_errors=True,
            rng=np.random.RandomState(k),
        ).go(obs, guess=0.5)
        if res['flags'] != 0 or res['T_flags'] != 0:
            continue
        T.append(res['T'])
        Te.append(res['T_err'])
        F.append(res['flux'])
        Fe.append(res['flux_err'])
    T, Te = np.array(T), np.array(Te)
    F, Fe = np.array(F), np.array(Fe)
    assert len(T) > 0.9 * ntrial
    rT = T.std() / np.sqrt(np.mean(Te ** 2))
    rF = F.std() / np.sqrt(np.mean(Fe ** 2))
    assert 0.8 < rT < 1.2
    assert 0.85 < rF < 1.15
