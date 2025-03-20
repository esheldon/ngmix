__all__ = ['FitModel', 'CoellipFitModel', 'PSFFluxFitModel']
import copy
import numpy as np
from .. import gmix
from ..gexceptions import GMixRangeError
from ..defaults import PDEF, CDEF, LOWVAL, BIGVAL, copy_if_needed
from ..observation import Observation, ObsList, get_mb_obs
from ..gmix.gmix_nb import gmix_convolve_fill, fill_fdiff
from ..gmix import GMixList, MultiBandGMixList
from ..flags import ZERO_DOF, DIV_ZERO, BAD_VAR


class FitModel(dict):
    """
    A class to represent a fitting model, the result of the fit, as well as
    generate images and mixtures for the best fit model

    Parameters
    ----------
    obs: observation(s)
        Observation, ObsList, or MultiBandObsList
    model: str
        The model to fit
    prior: ngmix prior
        A prior for fitting
    """

    def __init__(self, obs, model, guess, prior=None):
        self.prior = prior
        self.model = gmix.get_model_num(model)
        self.model_name = gmix.get_model_name(self.model)
        self['model'] = self.model_name

        self._set_obs(obs)
        self._set_totpix()
        self._set_npars()

        self._set_n_prior_pars()
        self._set_fdiff_size()
        self._make_pixel_list()

        self._set_bounds()
        self._setup_fit(guess)

    def set_fit_result(self, result):
        """
        Get some fit statistics for the input pars.
        """

        self.update(result)

        if self["flags"] == 0:
            cres = self.calc_lnprob(self['pars'], more=True)
            self.update(cres)

            if self["s2n_denom"] > 0:
                s2n = self["s2n_numer"] / np.sqrt(self["s2n_denom"])
            else:
                s2n = 0.0

            chi2 = self["lnprob"] / (-0.5)
            dof = self["npix"] - self.npars
            chi2per = chi2 / dof

            self["chi2per"] = chi2per
            self["dof"] = dof
            self["s2n_w"] = s2n
            self["s2n"] = s2n

            self._set_g()
            self._set_T()
            self._set_flux()

    def get_gmix(self, band=0):
        """
        Get a gaussian mixture at the fit parameter set, which
        definition depends on the sub-class

        Parameters
        ----------
        band: int, optional
            Band index, default 0
        """
        pars = self.get_band_pars(pars=self["pars"], band=band)
        return gmix.make_gmix_model(pars, self.model)

    def get_convolved_gmix(self, band=0, obsnum=0):
        """
        get a gaussian mixture at the fit parameters, convolved by the psf if
        fitting a pre-convolved model

        Parameters
        ----------
        band: int, optional
            Band index, default 0
        obsnum: int, optional
            Number of observation for the given band,
            default 0
        """

        gm = self.get_gmix(band)

        obs = self.obs[band][obsnum]
        if obs.has_psf_gmix():
            gm = gm.convolve(obs.psf.gmix)

        return gm

    def make_image(self, band=0, obsnum=0):
        """
        Get an image of the best fit mixture

        Returns
        -------
        image: array
            Image of the model, including the PSF if a psf was sent
        """
        gm = self.get_convolved_gmix(band=band, obsnum=obsnum)
        obs = self.obs[band][obsnum]
        return gm.make_image(
            obs.image.shape,
            jacobian=obs.jacobian,
        )

    def get_band_pars(self, pars, band):
        """
        get pars for the specified band

        Parameters
        ----------
        pars: array-like
            Array-like of parameters
        band: int
            The band as an integer

        Returns
        -------
        parameters just associated with the requested band
        """
        return get_band_pars(model=self.model_name, pars=pars, band=band)

    def calc_lnprob(self, pars, more=False):
        """
        This is all we use for mcmc approaches, but also used generally for the
        "set_fit_stats" method.  For the max likelihood fitter we also have a
        _get_ydiff method

        Parameters
        ----------
        pars: array-like
            Array-like of parameters
        more: bool
            If true, a dict with more information is returned

        Returns
        -------
        the log(probability) unless more=True, in which case a dict
        with keys
            lnprob: log(probability)
            s2n_numer: numerator for S/N
            s2n_denom: denominator for S/N
            npix: number of pixels used
        """

        try:

            # these are the log pars (if working in log space)
            ln_priors = self._get_priors(pars)

            lnprob = 0.0
            s2n_numer = 0.0
            s2n_denom = 0.0
            npix = 0

            self._fill_gmix_all(pars)
            for band in range(self.nband):

                obs_list = self.obs[band]
                gmix_list = self._gmix_all[band]

                for obs, gm in zip(obs_list, gmix_list):

                    res = gm.get_loglike(obs, more=more)

                    if more:
                        lnprob += res["loglike"]
                        s2n_numer += res["s2n_numer"]
                        s2n_denom += res["s2n_denom"]
                        npix += res["npix"]
                    else:
                        lnprob += res

            # total over all bands
            lnprob += ln_priors

        except GMixRangeError:
            lnprob = LOWVAL
            s2n_numer = 0.0
            s2n_denom = BIGVAL
            npix = 0

        if more:
            return {
                "lnprob": lnprob,
                "s2n_numer": s2n_numer,
                "s2n_denom": s2n_denom,
                "npix": npix,
            }
        else:
            return lnprob

    @property
    def bounds(self):
        """
        get bounds used for fitting with leastsqsbound
        """
        return copy.deepcopy(self._bounds)

    def _set_obs(self, obs_in):
        """
        Set the obs attribute based on the input.

        Parmameters
        -----------
        obs: Observation, ObsList, or MultiBandObsList
            The input observations
        """

        self.obs = get_mb_obs(obs_in)

        self.nband = len(self.obs)
        nimage = 0
        for obslist in self.obs:
            for obs in obslist:
                nimage += 1
        self.nimage = nimage

    def _set_totpix(self):
        """
        Set the total number of pixels
        """

        totpix = 0
        for obs_list in self.obs:
            for obs in obs_list:
                totpix += obs.pixels.size

        self.totpix = totpix

    def _set_npars(self):
        """
        Set the number of parameters.  nband should be set in set_lists, called
        before this
        """
        self.npars = gmix.get_model_npars(self.model) + self.nband - 1

    def _make_model(self, band_pars):
        gm0 = gmix.make_gmix_model(band_pars, self.model)
        return gm0

    def _init_gmix_all(self, pars):
        """
        input pars are in linear space

        initialize the list of lists of gaussian mixtures
        """

        if self.obs[0][0].has_psf_gmix():
            self.dopsf = True
        else:
            self.dopsf = False

        gmix_all0 = MultiBandGMixList()
        gmix_all = MultiBandGMixList()

        for band, obs_list in enumerate(self.obs):
            gmix_list0 = GMixList()
            gmix_list = GMixList()

            # pars for this band, in linear space
            band_pars = self.get_band_pars(pars=pars, band=band)

            for obs in obs_list:
                gm0 = self._make_model(band_pars)
                if self.dopsf:
                    psf_gmix = obs.psf.gmix
                    gm = gm0.convolve(psf_gmix)
                else:
                    gm = gm0.copy()

                gmix_list0.append(gm0)
                gmix_list.append(gm)

            gmix_all0.append(gmix_list0)
            gmix_all.append(gmix_list)

        self._gmix_all0 = gmix_all0
        self._gmix_all = gmix_all

    def _convolve_gmix(self, gm, gm0, psf_gmix):
        """
        norms get set
        """
        gmix_convolve_fill(gm._data, gm0._data, psf_gmix._data)

    def _fill_gmix_all(self, pars):
        """
        input pars are in linear space

        Fill the list of lists of gmix objects for the given parameters
        """

        if not self.dopsf:
            self._fill_gmix_all_nopsf(pars)
            return

        for band, obs_list in enumerate(self.obs):
            gmix_list0 = self._gmix_all0[band]
            gmix_list = self._gmix_all[band]

            band_pars = self.get_band_pars(pars=pars, band=band)

            for i, obs in enumerate(obs_list):

                psf_gmix = obs.psf.gmix

                gm0 = gmix_list0[i]
                gm = gmix_list[i]

                gm0._fill(band_pars)
                self._convolve_gmix(gm, gm0, psf_gmix)

    def _fill_gmix_all_nopsf(self, pars):
        """
        Fill the list of lists of gmix objects for the given parameters
        """

        for band, obs_list in enumerate(self.obs):
            gmix_list = self._gmix_all[band]

            band_pars = self.get_band_pars(pars=pars, band=band)

            for i, obs in enumerate(obs_list):

                gm = gmix_list[i]

                gm._fill(band_pars)

    def _get_priors(self, pars):
        """
        get the sum of ln(prob) from the priors or 0.0 if
        no priors were sent
        """
        if self.prior is None:
            return 0.0
        else:
            return self.prior.get_lnprob_scalar(pars)

    def _setup_fit(self, guess):
        """
        setup the mixtures based on the initial guess
        """

        guess = np.array(guess, dtype="f8", copy=copy_if_needed)

        npars = guess.size
        mess = "guess has npars=%d, expected %d" % (npars, self.npars)
        assert npars == self.npars, mess

        try:
            # this can raise GMixRangeError
            self._init_gmix_all(guess)
            self._make_gmix_list()
        except ZeroDivisionError:
            raise GMixRangeError("got zero division")

    def _set_n_prior_pars(self):
        # center1 + center2 + shape + T + fluxes
        if self.prior is None:
            self.n_prior_pars = 0
        else:
            self.n_prior_pars = get_lm_n_prior_pars(
                model=self.model_name, nband=self.nband,
            )

    def _set_fdiff_size(self):
        self.fdiff_size = self.totpix + self.n_prior_pars

    def _set_bounds(self):
        """
        get bounds on parameters
        """
        self._bounds = None
        if self.prior is not None:
            if hasattr(self.prior, "bounds"):
                self._bounds = self.prior.bounds

    def _set_g(self):
        self["g"] = self["pars"][2:2+2].copy()
        self["g_cov"] = self["pars_cov"][2:2+2, 2:2+2].copy()
        self["g_err"] = self["pars_err"][2:2+2].copy()

    def _set_T(self):
        self["T"] = self["pars"][4]
        self["T_err"] = np.sqrt(self["pars_cov"][4, 4])

    def _set_flux(self):
        _set_flux(res=self, nband=self.nband)

    def _make_pixel_list(self):
        """
        lists of references.
        """
        pixels_list = []

        for band in range(self.nband):
            obs_list = self.obs[band]
            for obs in obs_list:
                pixels_list.append(obs._pixels)

        self._pixels_list = pixels_list

    def _make_gmix_list(self):
        """
        lists of references.
        """
        gmix_data_list = []

        for band in range(self.nband):

            gmix_list = self._gmix_all[band]

            for gm in gmix_list:
                gmdata = gm.get_data()
                gmix_data_list.append(gmdata)

        self._gmix_data_list = gmix_data_list

    def calc_fdiff(self, pars):
        """
        vector with (model-data)/error.

        The npars elements contain -ln(prior)
        """

        # we cannot keep sending existing array into leastsq, don't know why
        fdiff = np.zeros(self.fdiff_size)

        try:

            # all norms are set after fill
            self._fill_gmix_all(pars)

            start = self._fill_priors(pars=pars, fdiff=fdiff)

            for pixels, gm in zip(self._pixels_list, self._gmix_data_list):
                fill_fdiff(
                    gm, pixels, fdiff, start,
                )

                start += pixels.size

        except GMixRangeError:
            fdiff[:] = LOWVAL

        return fdiff

    def _fill_priors(self, pars, fdiff):
        """
        Fill priors at the beginning of the array.

        ret the position after last par

        We require all the lnprobs are < 0, equivalent to
        the peak probability always being 1.0

        I have verified all our priors have this property.
        """

        if self.prior is None:
            nprior = 0
        else:
            nprior = self.prior.fill_fdiff(pars, fdiff)

        return nprior


class CoellipFitModel(FitModel):
    """
    A class to represent a fitting a coelliptical gaussians model, the result
    of the fit, as well as generate images and mixtures for the best fit model


    Parameters
    ----------
    obs: observation(s)
        Observation, ObsList, or MultiBandObsList
    model: str
        The model to fit
    prior: ngmix prior
        A prior for fitting
    """

    def __init__(self, obs, ngauss, guess, prior=None):
        self._ngauss = ngauss
        super().__init__(obs=obs, model='coellip', guess=guess, prior=prior)

    def _set_flux(self):
        """
        this should be doable
        """
        pass

    def _set_n_prior_pars(self):
        assert self.nband == 1, "Coellip can only fit one band"

        if self.prior is None:
            self.n_prior_pars = 0
        else:
            ngauss = self._ngauss
            self.n_prior_pars = 1 + 1 + 1 + ngauss + ngauss

    def _set_npars(self):
        """
        single band, npars determined from ngauss
        """
        self.npars = 4 + 2 * self._ngauss

    def get_band_pars(self, pars, band):
        """
        Get linear pars for the specified band
        """

        return pars.copy()


class PSFFluxFitModel(dict):
    """
    A class to represent fitting a psf flux fitter.  The class can be
    used as a generic template flux fitter as well

    Parameters
    ----------
    obs: observation(s)
        Observation, ObsList, or MultiBandObsList
    do_psf: bool, optional
        If True, use the gaussian mixtures in the psf observation as templates.
        In this mode the code calculates a "psf flux".  Default True.
    normalize_psf: True or False
        if True, then normalize PSF gmix to flux of unity, otherwise use input
        normalization.  Default True
    """

    def __init__(self, obs, do_psf=True, normalize_psf=True):
        self.do_psf = do_psf

        self.normalize_psf = normalize_psf

        self['model'] = 'template'
        self.npars = 1
        self._set_obs(obs)

    def go(self):
        """
        calculate the flux using zero-lag cross-correlation
        """

        flags = 0

        xcorr_sum = 0.0
        msq_sum = 0.0

        chi2 = 0.0

        nobs = len(self.obs)

        flux = PDEF
        flux_err = CDEF

        for ipass in [1, 2]:
            for iobs in range(nobs):
                obs = self.obs[iobs]

                im = obs.image
                wt = obs.weight

                if ipass == 1:
                    model = self._get_model(iobs)
                    xcorr_sum += (model * im * wt).sum()
                    msq_sum += (model * model * wt).sum()
                else:
                    model = self._get_model(iobs, flux=flux)

                    chi2 += self._get_chi2(model, im, wt)

            if ipass == 1:
                if msq_sum == 0:
                    break
                flux = xcorr_sum / msq_sum

        # chi^2 per dof and error checking
        dof = self.get_dof()
        chi2per = 9999.0
        if dof > 0:
            chi2per = chi2 / dof
        else:
            flags |= ZERO_DOF

        # final flux calculation with error checking
        if msq_sum == 0 or self.totpix == 1:
            flags |= DIV_ZERO
        else:

            arg = chi2 / msq_sum / (self.totpix - 1)
            if arg >= 0.0:
                flux_err = np.sqrt(arg)
            else:
                flags |= BAD_VAR

        result = {
            "flags": flags,
            "chi2per": chi2per,
            "dof": dof,
            "flux": flux,
            "flux_err": flux_err,
        }
        self.update(result)

    def _get_chi2(self, model, im, wt):
        """
        get the chi^2 for this image/model

        we can simulate when needed
        """
        chi2 = ((model - im) ** 2 * wt).sum()
        return chi2

    def _get_model(self, iobs, flux=None):
        """
        get the model image
        """
        if self.use_template:
            if flux is not None:
                model = self.template_list[iobs].copy()
                norm = self.norm_list[iobs]
                model *= (norm * flux) / model.sum()
            else:
                model = self.template_list[iobs]

        else:

            if flux is None:
                gm = self.gmix_list[iobs]
            else:
                gm = self.gmix_list[iobs].copy()
                norm = self.norm_list[iobs]
                gm.set_flux(flux * norm)

            obs = self.obs[iobs]
            dims = obs.image.shape
            jac = obs.jacobian
            model = gm.make_image(dims, jacobian=jac)

        return model

    def get_dof(self):
        """
        Effective def based on effective number of pixels
        """
        npix = self.get_effective_npix()
        dof = npix - self.npars
        if dof <= 0:
            dof = 1.0e-6
        return dof

    def _set_obs(self, obs_in):
        """
        Input should be an Observation, ObsList
        """

        if isinstance(obs_in, Observation):
            obs_list = ObsList()
            obs_list.append(obs_in)
        elif isinstance(obs_in, ObsList):
            obs_list = obs_in
        else:
            raise ValueError("obs should be Observation or ObsList")

        tobs = obs_list[0]
        if self.do_psf:
            tobs = tobs.psf

        if not tobs.has_gmix():
            if not hasattr(tobs, "template"):
                raise ValueError("neither gmix or template image are set")

        self.obs = obs_list
        if tobs.has_gmix():
            self._set_gmix_and_norms()
        else:
            self._set_templates_and_norms()

        self._set_totpix()

    def _set_gmix_and_norms(self):
        self.use_template = False
        gmix_list = []
        norm_list = []
        for obs in self.obs:
            # these return copies, ok to modify
            if self.do_psf:
                gmix = obs.get_psf_gmix()
                if self.normalize_psf:
                    gmix.set_flux(1.0)
            else:
                gmix = obs.get_gmix()
                gmix.set_flux(1.0)

            gmix_list.append(gmix)
            norm_list.append(gmix.get_flux())

        self.gmix_list = gmix_list
        self.norm_list = norm_list

    def _set_templates_and_norms(self):
        self.use_template = True

        template_list = []
        norm_list = []
        for obs in self.obs:
            if self.do_psf:
                template = obs.psf.template.copy()
                norm = template.sum()
                if self.normalize_psf:
                    template *= 1.0 / norm
                    norm = 1.0
            else:
                template = obs.template.copy()
                template *= 1.0 / template.sum()
                norm = 1.0

            template_list.append(template)
            norm_list.append(norm)

        self.template_list = template_list
        self.norm_list = norm_list

    def _set_totpix(self):
        """
        Make sure the data are consistent.
        """

        totpix = 0
        for obs in self.obs:
            totpix += obs.pixels.size

        self.totpix = totpix

    def get_effective_npix(self):
        """
        We don't use all pixels, only those with weight > 0
        """
        if not hasattr(self, "eff_npix"):

            npix = 0
            for obs in self.obs:
                wt = obs.weight

                w = np.where(wt > 0)
                npix += w[0].size

            self.eff_npix = npix

        return self.eff_npix


def get_band_pars(model, pars, band):
    """
    extract parameters for given band

    Parameters
    ----------
    model: string
        Model name
    pars: all parameters
        Parameters for all bands
    band: int
        Band number

    Returns
    -------
    the subset of parameters for this band
    """

    num = gmix.get_model_npars(model)
    band_pars = np.zeros(num)

    assert model != 'coellip'

    if model == 'bd':
        band_pars[0:7] = pars[0:7]
        band_pars[7] = pars[7 + band]
    elif model == 'bdf':
        band_pars = np.zeros(num)
        band_pars[0:6] = pars[0:6]
        band_pars[6] = pars[6 + band]
    else:
        band_pars[0:5] = pars[0:5]
        band_pars[5] = pars[5 + band]

    return band_pars


def get_lm_n_prior_pars(model, nband):
    """
    get the number of slots for priors in LM

    Parameters
    ----------
    model: string
        The model being fit
    nband: int
        Number of bands
    prior: joint prior, optional
        If None, the result is always zero
    """

    if model == 'bd':
        # center1 + center2 + shape + T + log10(Td/Te) + fracdev + fluxes
        npp = 1 + 1 + 1 + 1 + 1 + 1 + nband
    elif model == 'bdf':
        # center1 + center2 + shape + T + fracdev + fluxes
        npp = 1 + 1 + 1 + 1 + 1 + nband
    elif model in ['exp', 'dev', 'gauss', 'turb']:
        # simple models
        npp = 1 + 1 + 1 + 1 + 1 + nband
    else:
        raise ValueError('bad model: %s' % model)

    return npp


def _set_flux(res, nband):
    """
    set the flux in the result dict for standard models.
    Does not work for coellip

    Parameters
    ----------
    res: dict
        The result dict.  Must contain 'pars'
    nband: int
        Number of bands
    """
    from numpy import sqrt, diag

    model = res['model']
    assert model != 'coellip'

    if model == 'bd':
        start = 7
    elif model == 'bdf':
        start = 6
    else:
        start = 5

    if nband == 1:
        res["flux"] = res["pars"][start]
        res["flux_err"] = np.sqrt(res["pars_cov"][start, start])
    else:
        res["flux"] = res["pars"][start:]
        res["flux_cov"] = res["pars_cov"][start:, start:]
        res["flux_err"] = sqrt(diag(res["flux_cov"]))
