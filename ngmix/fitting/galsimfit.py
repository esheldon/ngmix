"""
fitting using galsim to create the models
"""
__all__ = [
    'GalsimFitter', 'GalsimSpergelFitter',
    'GalsimMoffatFitter', 'GalsimPSFFluxFitter',
]
from .galsim_results import (
    GalsimFitModel, GalsimSpergelFitModel,
    GalsimMoffatFitModel, GalsimPSFFitModel,
)

from ..defaults import DEFAULT_LM_PARS
from .leastsqbound import run_leastsq

from .. import observation


class GalsimFitter(object):
    """
    Fit using galsim 6 parameter models

    Parameters
    ----------
    model: string
        e.g. 'exp', 'spergel'
    prior: ngmix prior, optional
        For example ngmix.priors.PriorSimpleSep can
        be used as a separable prior on center, g, size, flux.
    fit_pars: dict, optional
        parameters for the lm fitter, e.g. maxfev, ftol, xtol
    """

    def __init__(self, model, prior=None, fit_pars=None):
        self.prior = prior
        self.model = model

        if fit_pars is not None:
            self.fit_pars = fit_pars.copy()
        else:
            self.fit_pars = DEFAULT_LM_PARS.copy()

    def go(self, obs, guess):
        """
        Run leastsq and get the result
        """

        fit_model = self._make_fit_model(obs=obs, guess=guess)

        result = run_leastsq(
            fit_model.calc_fdiff,
            guess=guess,
            n_prior_pars=fit_model.n_prior_pars,
            bounds=fit_model.bounds,
            k_space=True,
            **self.fit_pars
        )

        fit_model.set_fit_result(result)

        return fit_model

    def _make_fit_model(self, obs, guess):
        return GalsimFitModel(
            obs=obs, model=self.model, guess=guess, prior=self.prior,
        )


class GalsimSpergelFitter(GalsimFitter):
    """
    Fit the spergel profile to the input observations

    Parameters
    ----------
    prior: ngmix prior, optional
        For example ngmix.priors.PriorSimpleSep can
        be used as a separable prior on center, g, size, flux.
    fit_pars: dict, optional
        parameters for the lm fitter, e.g. maxfev, ftol, xtol
    """

    def __init__(self, prior=None, fit_pars=None):
        super().__init__(model="spergel", prior=prior, fit_pars=fit_pars)

    def _make_fit_model(self, obs, guess):
        return GalsimSpergelFitModel(
            obs=obs, guess=guess, prior=self.prior,
        )


class GalsimMoffatFitter(GalsimFitter):
    """
    Fit a moffat model using galsim

    Parameters
    ----------
    model: string
        e.g. 'exp', 'spergel'
    prior: ngmix prior, optional
        For example ngmix.priors.PriorSimpleSep can
        be used as a separable prior on center, g, size, flux.
    fit_pars: dict, optional
        parameters for the lm fitter, e.g. maxfev, ftol, xtol
    """

    def __init__(self, prior=None, fit_pars=None):
        super().__init__(model="moffat", prior=prior, fit_pars=fit_pars)

    def _make_fit_model(self, obs, guess):
        return GalsimMoffatFitModel(
            obs=obs, guess=guess, prior=self.prior,
        )


class GalsimPSFFluxFitter(object):
    """
    Calculate psf flux or template fluxe using galsim

    Parameters
    ----------
    model: galsim model, optional
        A Galsim model, e.g. Exponential.  If not sent,
        a psf flux is measured
    draw_method: str
        Galsim drawing method, default 'auto'
    interp: string
        type of interpolation when using the PSF image
        rather than psf models.  Default lanzcos15
    """
    def __init__(
        self,
        model=None,
        draw_method='auto',
        interp=observation.DEFAULT_XINTERP,
    ):
        self.model = model
        self.draw_method = draw_method
        self.interp = interp

    def go(self, obs):

        fit_model = GalsimPSFFitModel(
            obs=obs, model=self.model,
            draw_method=self.draw_method, interp=self.interp,
        )
        fit_model.go()

        return fit_model
