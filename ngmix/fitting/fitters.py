"""
- todo
    - remove old unused fitters
"""
__all__ = ['Fitter', 'CoellipFitter', 'PSFFluxFitter']
import logging

from .leastsqbound import run_leastsq
from .. import gmix
from ..defaults import DEFAULT_LM_PARS
from .results import FitModel, CoellipFitModel, PSFFluxFitModel

LOGGER = logging.getLogger(__name__)


class Fitter(object):
    """
    A class for doing a fit using levenberg marquardt

    Parameters
    ----------
    model: str
        The model to fit
    prior: ngmix prior
        A prior for fitting
    fit_pars: dict
        Parameters to send to the leastsq fitting routine
    """

    def __init__(self, model, prior=None, fit_pars=None):
        self.prior = prior
        self.model = gmix.get_model_num(model)
        self.model_name = gmix.get_model_name(self.model)

        if fit_pars is not None:
            self.fit_pars = fit_pars.copy()
        else:
            self.fit_pars = DEFAULT_LM_PARS.copy()

    def go(self, obs, guess):
        """
        Run leastsq and set the result

        Parameters
        ----------
        obs: Observation, ObsList, or MultiBandObsList
            Observation(s) to fit
        guess: array
            Array of initial parameters for the fit

        Returns
        --------
        a dict-like which contains the result as well as functions used for the
        fitting.

        """

        fit_model = self._make_fit_model(obs=obs, guess=guess)

        result = run_leastsq(
            fit_model.calc_fdiff,
            guess=guess,
            n_prior_pars=fit_model.n_prior_pars,
            bounds=fit_model.bounds,
            **self.fit_pars
        )

        fit_model.set_fit_result(result)
        return fit_model

    def _make_fit_model(self, obs, guess):
        return FitModel(
            obs=obs, model=self.model, guess=guess, prior=self.prior,
        )


class CoellipFitter(Fitter):
    """
    class to perform a fit using a model of coelliptical gaussians

    Parameters
    ----------
    ngauss: int
        The number of coelliptical gaussians to fit
    prior: ngmix prior
        A prior for fitting
    fit_pars: dict
        Parameters to send to the leastsq fitting routine
    """

    def __init__(self, ngauss, prior=None, fit_pars=None):
        self._ngauss = ngauss
        super().__init__(model="coellip", prior=prior, fit_pars=fit_pars)

    def _make_fit_model(self, obs, guess):
        return CoellipFitModel(
            obs=obs, ngauss=self._ngauss, guess=guess, prior=self.prior,
        )


class PSFFluxFitter(object):
    """
    Calculate a psf flux or template flux.  We fix the center, so this is
    linear.  This uses a simple cross-correlation between model and data.

    The center of the jacobian(s) must point to a common place on the sky, and
    if the center is input (to reset the gmix centers),) it is relative to that
    position

    Parameters
    -----------
    do_psf: bool, optional
        If True, use the gaussian mixtures in the psf observation as templates.
        In this mode the code calculates a "psf flux".  If set for False,
        templates are taken from the primary observations. Default True.
    normalize_psf: True or False
        if True, then normalize PSF gmix to flux of unity, otherwise use input
        normalization.  Default True
    """

    def __init__(self, do_psf=True, normalize_psf=True):
        self.do_psf = do_psf
        self.normalize_psf = normalize_psf

    def go(self, obs):
        """
        perform the template flux fit and return the result

        Returns
        --------
        a dict-like which contains the result as well as functions used for the
        fitting. The class is TemplateFluxFitModel
        """
        fit_model = PSFFluxFitModel(
            obs=obs, do_psf=self.do_psf, normalize_psf=self.normalize_psf,
        )
        fit_model.go()
        return fit_model
