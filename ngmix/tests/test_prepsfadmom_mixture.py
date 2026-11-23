"""
tests for the mixture ('exp', 'dev') models in the pre-psf adaptive
moments fitter, the model argument and the fixcen option

The scenes are rendered from the same ngmix gaussian expansions the
models fit, so the model is exact and the fits must recover the
truth at high s2n
"""
import numpy as np
import pytest

import ngmix
from ngmix.moments import fwhm_to_T
from ngmix.prepsfadmom import run_prepsf_admom, PAdmomFitter

SCALE = 0.2
DIM = 48
PSF_FWHM = 0.8
FWHM_SMOOTH = 1.05 * PSF_FWHM


def g_to_e(g1, g2):
    gsq = g1 * g1 + g2 * g2
    fac = 2.0 / (1.0 + gsq)
    return fac * g1, fac * g2


def make_mixture_obs(
    model, T, g1, g2, flux, noise_sigma, rng=None,
    voff=0.0, uoff=0.0, dim=DIM,
):
    """
    the ngmix gaussian expansion of the named profile ('exp' or
    'dev') convolved with a round gaussian psf; the model of the
    same name represents this data exactly
    """
    cen = (dim - 1) / 2
    jacobian = ngmix.DiagonalJacobian(scale=SCALE, row=cen, col=cen)

    psf_T = fwhm_to_T(PSF_FWHM)
    psf_gm = ngmix.GMixModel([0, 0, 0, 0, psf_T, 1.0], 'gauss')
    psf_im = psf_gm.make_image((dim, dim), jacobian=jacobian)
    psf_obs = ngmix.Observation(
        psf_im, weight=np.ones((dim, dim)) * 1.0e12,
        jacobian=jacobian,
    )

    gm0 = ngmix.GMixModel([voff, uoff, g1, g2, T, flux], model)
    im = gm0.convolve(psf_gm).make_image((dim, dim), jacobian=jacobian)

    if rng is not None:
        im = im + rng.normal(scale=noise_sigma, size=im.shape)

    return ngmix.Observation(
        im, weight=np.ones((dim, dim)) / noise_sigma ** 2,
        jacobian=jacobian, psf=psf_obs,
    )


def test_model_api():
    """
    the model is a name; unknown names and non-names raise clearly
    """
    for model in ('gauss', 'exp', 'dev', 'star'):
        assert PAdmomFitter(model=model).model == model

    with pytest.raises(ValueError, match='bad model'):
        PAdmomFitter(model='bulgedisk')
    with pytest.raises(ValueError, match='bad model'):
        PAdmomFitter(model='bdf')
    with pytest.raises(ValueError, match='bad model'):
        PAdmomFitter(model={'type': 'exp'})


def test_fixcen():
    """
    with fixcen the center stays at the guess for every model, and
    a centered fit still recovers the truth
    """
    obs = make_mixture_obs('exp', 0.5, 0.08, -0.04, 100.0, 1.0e-6)

    results = {}
    for model in ('gauss', 'exp', 'dev'):
        guess = ngmix.GMixModel([0, 0, 0, 0, 0.3, 1.0], 'gauss')
        res = run_prepsf_admom(
            obs, model=model, fwhm_smooth=FWHM_SMOOTH, ap_rad=0.0,
            fixcen=True, guess=guess,
            rng=np.random.RandomState(5),
        )
        assert res['flags'] == 0
        assert res['cen'][0] == 0.0
        assert res['cen'][1] == 0.0
        results[model] = res

    res = results['exp']
    e1t, e2t = g_to_e(0.08, -0.04)
    assert abs(res['T'] / 0.5 - 1) < 1.0e-3
    assert abs(res['e1'] - e1t) < 1.0e-3
    assert abs(res['flux'] / 100.0 - 1) < 1.0e-3


def test_gauss_entries():
    """
    the gauss (converged-weight) shape entries are filled for
    every model and are identical across models on the same data,
    since the weight iteration is model independent; for the
    gauss model they match the primary shape outputs
    """
    obs = make_mixture_obs('exp', 0.5, 0.08, -0.04, 100.0, 0.1,
                           rng=np.random.RandomState(3))

    results = {}
    for model in ('gauss', 'exp', 'dev'):
        results[model] = run_prepsf_admom(
            obs, model=model, fwhm_smooth=FWHM_SMOOTH, ap_rad=0.0,
            rng=np.random.RandomState(5),
        )
        res = results[model]
        assert res['flags'] == 0
        assert res['gauss_e_flags'] == 0
        assert np.isfinite(res['gauss_e1'])
        assert res['gauss_e1err'] > 0
        assert res['gauss_T'] > 0
        assert res['gauss_T_err'] > 0

    rg = results['gauss']
    assert abs(rg['gauss_e1'] - rg['e1']) < 1.0e-9
    assert abs(rg['gauss_e2'] - rg['e2']) < 1.0e-9
    assert abs(rg['gauss_T'] - rg['T']) < 1.0e-9

    for model in ('exp', 'dev'):
        r = results[model]
        assert abs(r['gauss_e1'] - rg['gauss_e1']) < 1.0e-4
        assert abs(r['gauss_e2'] - rg['gauss_e2']) < 1.0e-4
        assert abs(r['gauss_T'] / rg['gauss_T'] - 1) < 1.0e-3


def test_model_sandwich_guard():
    """
    an indefinite family covariance (saddle: det of the smoothed
    components negative) produces nan predictions; the sandwich
    detects it and returns (None, None, None) instead of warning
    and propagating nan
    """
    import warnings
    from ngmix.prepsfadmom.errors import model_sandwich

    fam_cov = np.array([[5.0, 0.0], [0.0, -5.0]])
    wt_cov = np.diag([0.3, 0.3])
    sums = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    cov = np.eye(6)
    fsums = np.ones(1)
    fvars = np.ones(1)
    fmcovs = np.zeros((1, 3))

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        out = model_sandwich(
            'exp', fam_cov, wt_cov, 0.2,
            sums, cov, fsums, fvars, fmcovs,
        )
    assert out == (None, None, None)


@pytest.mark.parametrize('model', ['exp', 'dev'])
@pytest.mark.parametrize('offset', [(0.0, 0.0), (0.31, -0.22)])
def test_mixture_exactness(model, offset):
    """
    the mixture models recover a scene rendered from their own
    expansion, centered or not
    """
    voff, uoff = offset
    obs = make_mixture_obs(
        model, 0.5, 0.08, -0.04, 100.0, 1.0e-6, voff=voff, uoff=uoff,
    )
    res = run_prepsf_admom(
        obs, model=model, fwhm_smooth=FWHM_SMOOTH, ap_rad=0.0,
        rng=np.random.RandomState(5),
    )
    e1t, e2t = g_to_e(0.08, -0.04)
    assert res['flags'] == 0
    assert abs(res['cen'][0] - voff) < 1.0e-3
    assert abs(res['cen'][1] - uoff) < 1.0e-3
    assert abs(res['T'] / 0.5 - 1) < 1.0e-3
    assert abs(res['e1'] - e1t) < 1.0e-3
    assert abs(res['e2'] - e2t) < 1.0e-3
    assert abs(res['flux'] / 100.0 - 1) < 1.0e-3
