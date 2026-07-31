"""
Tests for the analytic jacobian of the LM fdiff vector, used as
Dfun in leastsqbound for the simple models.

The referee is the central difference of calc_fdiff itself: the
analytic jacobian must be the derivative of the actual objective,
fast exponential and apodized chi^2 truncation included.
"""
import numpy as np
import pytest

import ngmix
from ngmix.fitting.results import FitModel, get_step

from .test_fitting_noise_cov import (
    _make_obs, _get_prior, TGUESS, FLUX, PIXEL_SCALE,
)


def _fd_jacobian(fit_model, pars):
    """central differences of calc_fdiff, the reference for the
    analytic jacobian"""
    jac = np.zeros((fit_model.fdiff_size, pars.size))
    for ipar in range(pars.size):
        step = get_step(pars=pars, ipar=ipar, nband=fit_model.nband)
        p = pars.copy()
        p[ipar] += step
        fp = fit_model.calc_fdiff(p)
        p[ipar] -= 2 * step
        fm = fit_model.calc_fdiff(p)
        jac[:, ipar] = (fp - fm) / (2 * step)
    return jac


def _get_nprior(fit_model, pars):
    """the number of rows actually filled by the prior, where
    the pixel rows begin; this can be fewer than the
    n_prior_pars slots"""
    return fit_model._fill_priors(
        pars=pars, fdiff=np.zeros(fit_model.fdiff_size),
    )


def _compare_jacobians(fit_model, pars):
    ana = fit_model.calc_jacobian(pars)
    ref = _fd_jacobian(fit_model, pars)

    # the analytic path forward-differences the smooth priors at
    # a sqrt(machine epsilon) scale step; measured agreement
    # with the central reference is a few 1e-7
    npp = _get_nprior(fit_model, pars)
    assert np.allclose(ana[:npp], ref[:npp], rtol=1.0e-5, atol=5.0e-6)

    # the smooth fexp and the apodized truncation make the model
    # twice differentiable in the parameters everywhere, so every
    # pixel row is compared.  The remaining differences are the
    # fexp derivative approximation (~3e-5 relative) and the fd
    # truncation error; the measured max is 3e-5 of the column
    # scale.  These bounds are tight enough to catch a missing
    # window-slope term in the apodization band, which shows at
    # the few 1e-4 level
    for ipar in range(pars.size):
        a = ana[npp:, ipar]
        r = ref[npp:, ipar]
        scale = np.abs(r).max()
        d = np.abs(a - r)
        assert np.all(d <= 1.0e-4 * scale)
        assert np.sqrt(np.sum(d ** 2) / np.sum(r ** 2)) < 5.0e-5


@pytest.mark.parametrize('model', ['gauss', 'exp', 'dev'])
def test_lm_jacobian_vs_fd(model):
    """the analytic jacobian agrees with central differences of
    calc_fdiff at and away from the solution, multi band"""
    rng = np.random.RandomState(15)
    mbobs = _make_obs(rng, 4.0, nband=2)
    prior = _get_prior(rng)
    fitter = ngmix.fitting.Fitter(model=model, prior=prior)
    guess = np.array([0.0, 0.0, 0.0, 0.0, TGUESS, FLUX, FLUX])
    fit_model = fitter.go(obs=mbobs, guess=guess)
    assert fit_model['flags'] == 0

    pars = fit_model['pars']
    pars_off = (
        pars * np.array([1, 1, 1, 1, 1.2, 0.9, 1.1])
        + np.array([0.02, -0.03, 0.05, -0.04, 0, 0, 0])
    )
    _compare_jacobians(fit_model, pars)
    _compare_jacobians(fit_model, pars_off)


def test_lm_jacobian_nopsf():
    """fitting with no psf set, as when fitting the psf itself"""
    rng = np.random.RandomState(19)
    obs0 = _make_obs(rng, 4.0)[0][0]
    obs = ngmix.Observation(
        obs0.image, weight=obs0.weight, jacobian=obs0.jacobian,
    )
    prior = _get_prior(rng)
    fitter = ngmix.fitting.Fitter(model='gauss', prior=prior)
    guess = np.array([0.0, 0.0, 0.0, 0.0, TGUESS, FLUX])
    fit_model = fitter.go(obs=obs, guess=guess)
    assert fit_model['flags'] == 0

    _compare_jacobians(fit_model, fit_model['pars'])


def test_lm_jacobian_range_error():
    """out of range parameters give a zero jacobian, the analog
    of the constant LOWVAL fdiff"""
    rng = np.random.RandomState(21)
    mbobs = _make_obs(rng, 8.0)
    prior = _get_prior(rng)
    fitter = ngmix.fitting.Fitter(model='gauss', prior=prior)
    guess = np.array([0.0, 0.0, 0.0, 0.0, TGUESS, FLUX])
    fit_model = fitter.go(obs=mbobs, guess=guess)
    assert fit_model['flags'] == 0

    for bad in [
        [0.0, 0.0, 1.5, 0.0, TGUESS, FLUX],   # |g| > 1
        [0.0, 0.0, 0.0, 0.0, 0.0, FLUX],      # T = 0
        [0.0, 0.0, 0.0, 0.0, TGUESS, 0.0],    # flux = 0
    ]:
        jac = fit_model.calc_jacobian(np.array(bad))
        assert np.all(jac == 0)


def test_lm_jacobian_bad_model():
    """only the simple models have the analytic jacobian"""
    rng = np.random.RandomState(23)
    mbobs = _make_obs(rng, 8.0)
    guess = np.array([0.0, 0.0, 0.0, 0.0, TGUESS, 0.5, FLUX])
    fit_model = FitModel(obs=mbobs, model='bdf', guess=guess)
    with pytest.raises(ValueError):
        fit_model.calc_jacobian(guess)


@pytest.mark.parametrize('model', ['gauss', 'exp', 'dev'])
def test_lm_jacobian_fit(model):
    """fits with the analytic jacobian find the same solution as
    the finite difference fits, in far fewer function
    evaluations"""
    rng = np.random.RandomState(31)
    prior = _get_prior(rng)
    # tight tolerances so both fits settle close to the true
    # minimum and can be compared well within the errors
    fit_pars = {'maxfev': 4000, 'ftol': 1.0e-8, 'xtol': 1.0e-8}
    fitter_ana = ngmix.fitting.Fitter(
        model=model, prior=prior, fit_pars=fit_pars,
    )
    fitter_fd = ngmix.fitting.Fitter(
        model=model, prior=prior, fit_pars=fit_pars,
        analytic_jacobian=False,
    )

    nfev_ana = []
    nfev_fd = []
    for _ in range(10):
        mbobs = _make_obs(rng, 4.0)
        guess = np.concatenate([
            rng.uniform(-0.01, 0.01, 2),
            rng.uniform(-0.02, 0.02, 2),
            [TGUESS * rng.uniform(0.9, 1.1)],
            [FLUX * rng.uniform(0.9, 1.1)],
        ])
        res_a = fitter_ana.go(obs=mbobs, guess=guess)
        res_f = fitter_fd.go(obs=mbobs, guess=guess)
        assert res_a['flags'] == 0
        assert res_f['flags'] == 0

        dpars = np.abs(res_a['pars'] - res_f['pars'])
        assert np.all(dpars < 0.02 * res_f['pars_err'])
        assert np.allclose(
            res_a['pars_err'], res_f['pars_err'], rtol=0.05,
        )

        nfev_ana.append(res_a['nfev'])
        nfev_fd.append(res_f['nfev'])

    assert np.median(nfev_ana) < 0.6 * np.median(nfev_fd)


def test_lm_jacobian_bounds():
    """the analytic jacobian composes with the bounds transform
    in leastsqbound"""
    rng = np.random.RandomState(41)

    prior = ngmix.joint_prior.PriorSimpleSep(
        cen_prior=ngmix.priors.CenPrior(
            0.0, 0.0, PIXEL_SCALE, PIXEL_SCALE, rng=rng,
        ),
        g_prior=ngmix.priors.GPriorBA(0.3, rng=rng),
        T_prior=ngmix.priors.Normal(
            mean=0.4, sigma=1.0, bounds=[0.01, 100.0], rng=rng,
        ),
        F_prior=ngmix.priors.Normal(
            mean=FLUX, sigma=FLUX, bounds=[1.0, 1.0e9], rng=rng,
        ),
    )
    assert prior.bounds is not None
    mbobs = _make_obs(rng, 4.0)
    guess = np.array([0.0, 0.0, 0.0, 0.0, TGUESS, FLUX])

    fit_pars = {'maxfev': 4000, 'ftol': 1.0e-8, 'xtol': 1.0e-8}
    fitter_ana = ngmix.fitting.Fitter(
        model='gauss', prior=prior, fit_pars=fit_pars,
    )
    fitter_fd = ngmix.fitting.Fitter(
        model='gauss', prior=prior, fit_pars=fit_pars,
        analytic_jacobian=False,
    )
    res_a = fitter_ana.go(obs=mbobs, guess=guess)
    res_f = fitter_fd.go(obs=mbobs, guess=guess)
    assert res_a['flags'] == 0
    assert res_f['flags'] == 0
    # compare the parameters directly: the covariance through the
    # bounds transform is not a reliable yardstick
    assert np.allclose(
        res_a['pars'], res_f['pars'], rtol=2.0e-4, atol=1.0e-5,
    )
