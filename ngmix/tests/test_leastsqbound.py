import pytest
import numpy as np
import ngmix
from ngmix.bootstrap import bootstrap
from ._sims import get_model_obs
from ._priors import get_prior

FRAC_TOL = 5.0e-4


class FitModelValueErrorNaN(ngmix.fitting.LMFitModel):
    def calc_fdiff(self, pars):
        raise ValueError('NaNs')


class LMValueErrorNaN(ngmix.fitting.LM):
    def _make_fit_model(self, obs, guess):
        return FitModelValueErrorNaN(
            obs=obs, model=self.model, guess=guess, prior=self.prior,
        )


class FitModelValueError(ngmix.fitting.LMFitModel):
    def calc_fdiff(self, pars):
        raise ValueError('blah')


class LMValueError(ngmix.fitting.LM):
    def _make_fit_model(self, obs, guess):
        return FitModelValueError(
            obs=obs, model=self.model, guess=guess, prior=self.prior,
        )


class FitModelZeroDivision(ngmix.fitting.LMFitModel):
    def calc_fdiff(self, pars):
        raise ZeroDivisionError('blah')


class LMZeroDivision(ngmix.fitting.LM):
    def _make_fit_model(self, obs, guess):
        return FitModelZeroDivision(
            obs=obs, model=self.model, guess=guess, prior=self.prior,
        )


@pytest.mark.parametrize('use_prior', [True, False])
def test_leastsqbound_smoke(use_prior):
    rng = np.random.RandomState(2830)
    ntrial = 10
    scale = 0.263

    fit_model = 'exp'

    noise = 0.1
    psf_ngauss = 3

    psf_guesser = ngmix.guessers.CoellipPSFGuesser(
        rng=rng,
        ngauss=psf_ngauss,
    )

    psf_fitter = ngmix.fitting.LMCoellip(ngauss=psf_ngauss)

    psf_runner = ngmix.runners.PSFRunner(
        fitter=psf_fitter,
        guesser=psf_guesser,
        ntry=2,
    )

    if use_prior:
        prior = get_prior(
            fit_model=fit_model,
            rng=rng,
            scale=scale,
            T_range=[-1.0, 1.e3],
            F_range=[0.01, 1000.0],
        )
    else:
        prior = None

    fitter = ngmix.fitting.LM(model=fit_model, prior=prior)
    guesser = ngmix.guessers.TFluxGuesser(
        rng=rng,
        T=0.25,
        flux=100.0,
        prior=prior,
    )

    runner = ngmix.runners.Runner(
        fitter=fitter,
        guesser=guesser,
        ntry=2,
    )

    allflags = np.zeros(ntrial)
    for i in range(ntrial):
        data = get_model_obs(
            rng=rng,
            model='exp',
            noise=noise,
        )
        obs = data['obs']

        try:
            res = bootstrap(obs=obs, runner=runner, psf_runner=psf_runner)
            allflags[i] = res['flags']
        except ngmix.BootPSFFailure:
            allflags[i] = 1

    assert np.any(allflags == 0)


@pytest.mark.parametrize('fracdev_bounds', [None, (0, 1)])
def test_leastsqbound_bounds(fracdev_bounds):
    rng = np.random.RandomState(2830)

    ntrial = 10
    fit_model = 'bd'
    scale = 0.263

    noise = 0.1
    psf_ngauss = 3

    psf_guesser = ngmix.guessers.CoellipPSFGuesser(
        rng=rng,
        ngauss=psf_ngauss,
    )

    psf_fitter = ngmix.fitting.LMCoellip(ngauss=psf_ngauss)

    psf_runner = ngmix.runners.PSFRunner(
        fitter=psf_fitter,
        guesser=psf_guesser,
        ntry=2,
    )

    prior = get_prior(
        fit_model=fit_model,
        rng=rng,
        scale=scale,
        T_range=[-1.0, 1.e3],
        F_range=[0.01, 1000.0],
        fracdev_bounds=fracdev_bounds,
    )

    guesser = ngmix.guessers.BDFGuesser(
        T=0.25,
        flux=100.0,
        prior=prior,
    )

    fitter = ngmix.fitting.LM(model=fit_model, prior=prior)
    runner = ngmix.runners.Runner(
        fitter=fitter,
        guesser=guesser,
        ntry=2,
    )

    allflags = np.zeros(ntrial)
    for i in range(ntrial):
        data = get_model_obs(
            rng=rng,
            model='exp',
            noise=noise,
        )
        obs = data['obs']

        try:
            res = bootstrap(obs=obs, runner=runner, psf_runner=psf_runner)
            allflags[i] = res['flags']
        except ngmix.BootPSFFailure:
            allflags[i] = 1

    assert np.any(allflags != 0)


def test_leastsqbound_errors():
    rng = np.random.RandomState(2830)

    fit_model = 'exp'

    data = get_model_obs(rng=rng, model='exp', noise=0.1)
    obs = data['obs']

    guess = np.array([0, 0, 0, 0, 10, 1])

    # only certain ValueError are caught
    with pytest.raises(ValueError):
        fitter = LMValueError(model=fit_model)
        fitter.go(obs=obs, guess=guess)

    fitter = LMValueErrorNaN(model=fit_model)
    res = fitter.go(obs=obs, guess=guess)
    assert res['flags'] == ngmix.flags.LM_FUNC_NOTFINITE

    fitter = LMZeroDivision(model=fit_model)
    res = fitter.go(obs=obs, guess=guess)
    assert res['flags'] == ngmix.flags.DIV_ZERO


@pytest.mark.parametrize('psf_noise', [1.0e-6, 1.0e9])
@pytest.mark.parametrize('fit_model', ['exp', 'bd'])
def test_leastsqbound_bad_data(fit_model, psf_noise):
    rng = np.random.RandomState(2830)

    ntrial = 10

    psf_model = 'gauss'
    # fit_model = 'bd'
    # fit_model = 'exp'
    scale = 0.263

    # psf_ngauss = 3

    psf_guesser = ngmix.guessers.SimplePSFGuesser(rng=rng)

    psf_fitter = ngmix.fitting.LM(model=psf_model)

    psf_runner = ngmix.runners.PSFRunner(
        fitter=psf_fitter,
        guesser=psf_guesser,
    )

    if fit_model == 'bd':
        prior = get_prior(
            fit_model=fit_model,
            rng=rng,
            scale=scale,
            T_range=[0.1, 1.e3],
            F_range=[0.1, 1000.0],
            fracdev_bounds=[0, 1],
        )
    else:
        prior = None

    if fit_model == 'bd':
        guesser = ngmix.guessers.BDFGuesser(
            T=0.25,
            flux=100.0,
            prior=prior,
        )
    else:
        guesser = ngmix.guessers.TFluxGuesser(
            rng=rng,
            T=0.25,
            flux=100.0,
            prior=prior,
        )

    fitter = ngmix.fitting.LM(model=fit_model, prior=prior)
    runner = ngmix.runners.Runner(
        fitter=fitter,
        guesser=guesser,
    )

    allflags = np.zeros(ntrial)
    for i in range(ntrial):
        data = get_model_obs(
            rng=rng,
            model='exp',
            psf_noise=psf_noise,
            psf_model=psf_model,
        )
        obs = data['obs']

        if psf_noise > 0.0:
            obs.image = rng.normal(scale=1.e9, size=obs.image.shape)

        try:
            res = bootstrap(obs=obs, runner=runner, psf_runner=psf_runner)
            allflags[i] = res['flags']
        except ngmix.BootPSFFailure:
            allflags[i] = 1

    assert np.any(allflags != 0)
