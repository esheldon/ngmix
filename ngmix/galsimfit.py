"""
fitting using galsim to create the models
"""

from __future__ import print_function, absolute_import, division

try:
    xrange
except NameError:
    xrange=range

import numpy

from .fitting import (
    LMSimple,
    TemplateFluxFitter,
    run_leastsq,
    _default_lm_pars,
)

from . import observation
from .observation import Observation, ObsList, MultiBandObsList

from .priors import LOWVAL
from .gexceptions import GMixRangeError

class GalsimRunner(object):
    """
    wrapper to generate guesses and run the fitter a few times

    Can be used to run GalsimSimple and SpergelFitter fitters

    parameters
    ----------
    obs: observation
        An instance of Observation, ObsList, or MultiBandObsList
    model: string
        e.g. 'exp', 'spergel'
    guesser: ngmix guesser
        E.g. R50FluxGuesser for 6 parameter models, R50NuFluxGuesser for
        a spergel model
    lm_pars: dict
        parameters for the lm fitter, e.g. maxfev, ftol, xtol
    prior: ngmix prior
        For example when fitting simple models,
        ngmix.priors.PriorSimpleSep can be used as a separable prior on
        center, g, size, flux.

        For spergel, PriorSpergelSep can be used.
    """
    def __init__(self,
                 obs,
                 model,
                 guesser,
                 lm_pars=None,
                 prior=None):

        self.obs=obs
        self.model=model
        self.guesser=guesser
        self.prior=prior

        self.lm_pars={}
        if lm_pars is not None:
            self.lm_pars.update(lm_pars)

    def get_fitter(self):
        return self.fitter

    def go(self, ntry=1):

        fitter=self._create_fitter()
        for i in xrange(ntry):

            guess=self.get_guess()
            fitter.go(guess)

            res=fitter.get_result()
            if res['flags']==0:
                break

        res['ntry'] = i+1
        self.fitter=fitter

    def get_guess(self):

        if self.model=='spergel':
            while True:
                guess=self.guesser()
                nu=guess[5]
                if nu > -0.84 and nu < 3.99:
                    break
        else:
            guess = self.guesser()

        return guess

    def _create_fitter(self):
        if self.model=='spergel':
            return SpergelFitter(
                self.obs,
                lm_pars=self.lm_pars,
                prior=self.prior,
            )
        else:
            return GalsimSimple(
                self.obs,
                self.model,
                lm_pars=self.lm_pars,
                prior=self.prior,
            )

class GalsimSimple(LMSimple):
    """
    Fit using galsim 6 parameter models

    parameters
    ----------
    obs: observation
        An instance of Observation, ObsList, or MultiBandObsList
    model: string
        e.g. 'exp', 'spergel'
    lm_pars: dict, optional
        parameters for the lm fitter, e.g. maxfev, ftol, xtol
    prior: ngmix prior, optional
        For example ngmix.priors.PriorSimpleSep can
        be used as a separable prior on center, g, size, flux.
    """
    def __init__(self, obs, model, **keys):
        self.keys=keys
        self.model_name=model
        self._set_model_class()

        self._set_kobs(obs)
        self._init_model_images()

        self._set_fitting_pars(**keys)
        self._set_prior(**keys)

        self._set_band_pars()
        self._set_totpix()

        self._set_fdiff_size()

    def go(self, guess):
        """
        Run leastsq and set the result
        """

        guess=self._get_guess(guess)

        result = run_leastsq(self._calc_fdiff,
                             guess,
                             self.n_prior_pars,
                             k_space=True,
                             **self.lm_pars)

        result['model'] = self.model_name
        if result['flags']==0:
            result['g'] = result['pars'][2:2+2].copy()
            result['g_cov'] = result['pars_cov'][2:2+2, 2:2+2].copy()
            stat_dict=self.get_fit_stats(result['pars'])
            result.update(stat_dict)

        self._result=result

    def _calc_fdiff(self, pars):
        """

        vector with (model-data)/error.

        The npars elements contain -ln(prior)
        """

        # we cannot keep sending existing array into leastsq, don't know why
        fdiff=numpy.zeros(self.fdiff_size)

        try:


            self._fill_models(pars)

            start=self._fill_priors(pars, fdiff)

            for band in xrange(self.nband):

                kobs_list=self.mb_kobs[band]
                for kobs in kobs_list:

                    meta =kobs.meta
                    kmodel = meta['kmodel']
                    ierr = meta['ierr']

                    scratch = meta['scratch']

                    # model-data
                    scratch.array[:,:] = kmodel.array[:,:]
                    scratch -= kobs.kimage

                    # (model-data)/err
                    scratch.array.real[:,:] *= ierr.array[:,:]
                    scratch.array.imag[:,:] *= ierr.array[:,:]

                    # now copy into the full fdiff array
                    imsize = scratch.array.size

                    fdiff[start:start+imsize] = scratch.array.real.ravel()

                    start += imsize

                    fdiff[start:start+imsize] = scratch.array.imag.ravel()

                    start += imsize

        except GMixRangeError as err:
            fdiff[:] = LOWVAL

        return fdiff

    def _fill_models(self, pars):
        """
        input pars are in linear space

        Fill the list of lists of gmix objects for the given parameters
        """
        try:
            for band,kobs_list in enumerate(self.mb_kobs):
                # pars for this band, in linear space
                band_pars=self.get_band_pars(pars, band)

                for i,kobs in enumerate(kobs_list):

                    gal = self.make_model(band_pars)

                    meta=kobs.meta

                    kmodel=meta['kmodel']

                    gal._drawKImage(kmodel)

                    if kobs.has_psf():
                        kmodel *= kobs.psf.kimage
        except RuntimeError as err:
            raise GMixRangeError(str(err))

    def make_model(self, pars):
        """
        make the galsim model
        """

        model = self.make_round_model(pars)

        shift = pars[0:0+2]
        g1    = pars[2]
        g2    = pars[3]

        # argh another generic error
        try:
            model = model.shear(g1=g1, g2=g2)
        except ValueError as err:
            raise GMixRangeError(str(err))

        model = model.shift(shift)
        return model

    def make_round_model(self, pars):
        """
        make the round galsim model, unshifted
        """

        r50   = pars[4]
        flux  = pars[5]

        if r50 < 0.0001:
            raise GMixRangeError("low r50: %g" % r50)

        # this throws a generic runtime error so there is no way to tell what
        # went wrong

        try:
            model = self._model_class(
                half_light_radius=r50,
                flux=flux,
            )
        except RuntimeError as err:
            raise GMixRangeError(str(err))

        return model

    def _set_model_class(self):
        import galsim
        if self.model_name=='exp':
            self._model_class=galsim.Exponential
        elif self.model_name=='dev':
            self._model_class=galsim.DeVaucouleurs
        elif self.model_name=='gauss':
            self._model_class=galsim.Gaussian
        else:
            raise NotImplementedError("can't fit '%s'" % self.model_name)

    def get_band_pars(self, pars_in, band):
        """
        Get pars for the specified band

        input pars are [c1, c2, e1, e2, r50, flux1, flux2, ....]
        """

        pars=self._band_pars

        pars[0:5] = pars_in[0:5]
        pars[5] = pars_in[5+band]
        return pars

    def _set_fitting_pars(self, **keys):
        """
        set the fit pars, in this case for the LM algorithm
        """
        lm_pars=keys.get('lm_pars',None)
        if lm_pars is None:
            lm_pars=_default_lm_pars
        self.lm_pars=lm_pars

    def _set_totpix(self):
        """
        Make sure the data are consistent.
        """

        totpix=0
        for kobs_list in self.mb_kobs:
            for kobs in kobs_list:
                totpix += kobs.kimage.array.size

        self.totpix=totpix


    def _convert2kobs(self, obs):
        kobs = observation.make_kobs(obs, **self.keys)

        return kobs

    def _set_kobs(self, obs_in, **keys):
        """
        Input should be an Observation, ObsList, or MultiBandObsList
        """

        if isinstance(obs_in, (Observation, ObsList, MultiBandObsList)):
            kobs=self._convert2kobs(obs_in)
        else:
            kobs=observation.get_kmb_obs(obs_in)

        self.mb_kobs = kobs
        self.nband=len(kobs)

    def _set_prior(self, **keys):
        self.prior = keys.get('prior',None)
        if self.prior is None:
            self.n_prior_pars=0
        else:
            #                 c1  c2  e1e2  r50  fluxes
            self.n_prior_pars=1 + 1 + 1   + 1  + self.nband

    def _set_fdiff_size(self):
        # we have 2*totpix, since we use both real and imaginary
        # parts
        self.fdiff_size = self.n_prior_pars + 2*self.totpix


    def _create_models_in_kobs(self, kobs):
        ex=kobs.kimage

        meta=kobs.meta
        meta['kmodel'] = ex.copy()
        meta['scratch'] = ex.copy()

    def _init_model_images(self):
        """
        add model image entries to the metadata for
        each observation

        these will get filled in
        """

        for kobs_list in self.mb_kobs:
            for kobs in kobs_list:
                meta=kobs.meta

                weight = kobs.weight
                ierr = weight.copy()
                ierr.setZero()

                w=numpy.where(weight.array > 0)
                if w[0].size > 0:
                    ierr.array[w] = numpy.sqrt(weight.array[w])

                meta['ierr'] = ierr
                self._create_models_in_kobs(kobs)

    def _check_guess(self, guess):
        """
        check the guess by making a model and checking for an
        exception
        """

        guess=numpy.array(guess,dtype='f8',copy=False)
        if guess.size != self.npars:
            raise ValueError("expected %d entries in the "
                             "guess, but got %d" % (self.npars,guess.size))

        for band in xrange(self.nband):
            band_pars = self.get_band_pars(guess, band)
            # just doing this to see if an exception is raised. This
            # will bother flake8
            gal = self.make_model(band_pars)

        return guess

    def _get_guess(self, guess):
        """
        make sure the guess has the right size and meets the model
        restrictions
        """

        guess=self._check_guess(guess)
        return guess


    def _set_npars(self):
        """
        nband should be set in set_lists, called before this
        """
        self.npars=5 + self.nband

    def _set_band_pars(self):
        """
        this is the array we fill with pars for a specific band
        """
        self._set_npars()

        npars_band = self.npars - self.nband + 1
        self._band_pars=numpy.zeros(npars_band)

    def get_fit_stats(self, pars):
        """
        Get some fit statistics for the input pars.
        """

        res={}
        res['s2n_r'] = self.calc_s2n_r(pars)
        return res

    def calc_s2n_r(self, pars):
        """
        we already have the round r50, so just create the
        models and don't shear them
        """

        s2n_sum=0.0
        for band,kobs_list in enumerate(self.mb_kobs):
            # pars for this band, in linear space
            band_pars=self.get_band_pars(pars, band)

            for i,kobs in enumerate(kobs_list):
                meta=kobs.meta
                weight=kobs.weight

                round_pars=band_pars.copy()
                round_pars[2:2+2] = 0.0
                gal = self.make_model(round_pars)

                kmodel=meta['kmodel']

                gal.drawKImage(image=kmodel)

                if kobs.has_psf():
                    kmodel *= kobs.psf.kimage
                kmodel.real.array[:,:] *= kmodel.real.array[:,:]
                kmodel.imag.array[:,:] *= kmodel.imag.array[:,:]

                kmodel.real.array[:,:] *= weight.array[:,:]
                kmodel.imag.array[:,:] *= weight.array[:,:]

                s2n_sum += kmodel.real.array.sum()
                s2n_sum += kmodel.imag.array.sum()

        if s2n_sum > 0.0:
            s2n = numpy.sqrt(s2n_sum)
        else:
            s2n = 0.0

        return s2n

class SpergelFitter(GalsimSimple):
    """
    Fit the spergel profile to the input observations
    """
    def __init__(self, obs, **keys):
        super(SpergelFitter,self).__init__(obs, 'spergel', **keys)

    def _set_model_class(self):
        import galsim

        self._model_class=galsim.Spergel

    def make_round_model(self, pars):
        """
        make the galsim Spergel model
        """
        import galsim

        r50   = pars[4]
        nu    = pars[5]
        flux  = pars[6]

        # generic RuntimeError thrown
        try:
            gal = galsim.Spergel(
                nu,
                half_light_radius=r50,
                flux=flux,
            )
        except RuntimeError as err:
            raise GMixRangeError(str(err))

        return gal

    def get_band_pars(self, pars_in, band):
        """
        Get linear pars for the specified band

        input pars are [c1, c2, e1, e2, r50, nu, flux1, flux2, ....]
        """

        pars=self._band_pars

        pars[0:6] = pars_in[0:6]
        pars[6] = pars_in[6+band]
        return pars

    def _set_prior(self, **keys):
        self.prior = keys.get('prior',None)
        if self.prior is None:
            self.n_prior_pars=0
        else:
            #                 c1  c2  e1e2  r50  nu   fluxes
            self.n_prior_pars=1 + 1 + 1   + 1  + 1  + self.nband


    def _set_npars(self):
        """
        nband should be set in set_lists, called before this
        """
        self.npars=6 + self.nband

class MoffatFitter(GalsimSimple):
    """
    Fit the moffat profile with free beta to the input observations
    """
    def __init__(self, obs, **keys):
        super(MoffatFitter,self).__init__(obs, 'moffat', **keys)

    def _set_model_class(self):
        import galsim

        self._model_class=galsim.Moffat

    def make_round_model(self, pars):
        """
        make the galsim Moffat model
        """
        import galsim

        r50   = pars[4]
        beta  = pars[5]
        flux  = pars[6]

        # generic RuntimeError thrown
        try:
            gal = galsim.Moffat(
                beta,
                half_light_radius=r50,
                flux=flux,
            )
        except RuntimeError as err:
            raise GMixRangeError(str(err))

        return gal

    def get_band_pars(self, pars_in, band):
        """
        Get linear pars for the specified band

        input pars are [c1, c2, e1, e2, r50, beta, flux1, flux2, ....]
        """

        pars=self._band_pars

        pars[0:6] = pars_in[0:6]
        pars[6] = pars_in[6+band]
        return pars

    def _set_prior(self, **keys):
        self.prior = keys.get('prior',None)
        if self.prior is None:
            self.n_prior_pars=0
        else:
            #                 c1  c2  e1e2  r50  beta   fluxes
            self.n_prior_pars=1 + 1 + 1   + 1  + 1    + self.nband


    def _set_npars(self):
        """
        nband should be set in set_lists, called before this
        """
        self.npars=6 + self.nband


class GalsimTemplateFluxFitter(TemplateFluxFitter):
    def __init__(self, obs, model=None, psf_models=None, **keys):
        """
        parameters
        -----------
        obs: Observation or ObsList
            See ngmix.observation.Observation.
        model: galsim model, optional
            A Galsim model, e.g. Exponential.  If not sent,
            a psf flux is measured
        psf_models: galsim model or list thereof, optional
            If not sent, the psf images from the observations
            are used.

        interp: string
            type of interpolation when using the PSF image
            rather than psf models.  Default lanzcos15
        simulate_err: bool, optional
            If set, noise is added according to the weight
            map. Useful when trying to calculate the noise
            on a model rather than from real data.
        rng: numpy random number generator, optional
            For use when simulate_err=True

        TODO:
            - try more complex wcs
        """

        self.model=model
        self.psf_models=psf_models

        if self.model is not None:
            self.model = self.model.withFlux(1.0)

        self.keys=keys
        self.normalize_psf = keys.get('normalize_psf',True)
        assert self.normalize_psf is True,\
            "currently must have normalize_psf=True"

        self.interp=keys.get('interp',observation.DEFAULT_XINTERP)

        self.simulate_err=keys.get('simulate_err',False)
        if self.simulate_err:
            rng=keys.get("rng",None)
            if rng is None:
                rng = numpy.random.RandomState()
            self.rng=rng

        self._set_obs(obs)
        self._set_psf_models()

        self.model_name='template'
        self.npars=1

        self._set_totpix()

        self._set_template_images()

    def _get_model(self, iobs, flux=None):
        """
        get the model image
        """

        if flux is not None:
            model = self.template_list[iobs].copy()
            model *= flux/model.sum()
        else:
            model = self.template_list[iobs]

        return model

    def _set_psf_models(self):
        if self.psf_models is not None:
            if len(self.psf_models) != len(self.obs):
                raise ValueError("psf models must be same "
                                 "size as observations ")

            self.psf_models = [p.withFlux(1.0) for p in self.psf_models]
        else:
            self._set_psf_models_from_images()

    def _set_psf_models_from_images(self):
        """
        we use the psfs stored in the observations as
        the psf model
        """
        import galsim

        models=[]
        for obs in self.obs:

            psf_jac = obs.psf.jacobian
            psf_im = obs.psf.image.copy()

            psf_im *= 1.0/psf_im.sum()

            nrow, ncol = psf_im.shape
            canonical_center = (numpy.array((ncol,nrow))-1.0)/2.0
            jrow, jcol = psf_jac.get_cen()
            offset = (jcol, jrow) - canonical_center

            psf_gsimage = galsim.Image(
                psf_im,
                wcs=obs.psf.jacobian.get_galsim_wcs(),
            )
            psf_ii = galsim.InterpolatedImage(
                psf_gsimage,
                offset=offset,
                x_interpolant=self.interp,
            )

            models.append(psf_ii)

        self.psf_models = models

    def _set_template_images(self):
        """
        set the images used for the templates
        """
        import galsim

        image_list=[]

        for i,obs in enumerate(self.obs):

            psf_model=self.psf_models[i]

            if self.model is not None:
                obj = galsim.Convolve(self.model, psf_model)
            else:
                obj = psf_model


            nrow,ncol=obs.image.shape

            gim = self._do_draw(
                obj,
                ncol,
                nrow,
                obs.jacobian,
            )

            image_list.append( gim.array )

        self.template_list=image_list

    def _do_draw(self, obj, ncol, nrow, jac):
        wcs = jac.get_galsim_wcs()

        # note reverse for galsim
        canonical_center = (numpy.array( (ncol,nrow) )-1.0)/2.0
        jrow, jcol = jac.get_cen()
        offset = (jcol, jrow) - canonical_center
        try:
            gim = obj.drawImage(
                nx=ncol,
                ny=nrow,
                wcs=wcs,
                offset=offset,
                method='no_pixel', # pixel is assumed to be in psf
            )
        except RuntimeError as err:
            # argh another generic exception
            raise GMixRangeError(str(err))

        return gim

    def _set_obs(self, obs_in):
        """
        Input should be an Observation, ObsList
        """

        if isinstance(obs_in,Observation):
            obs_list=ObsList()
            obs_list.append(obs_in)

            if self.psf_models is not None:
                if not isinstance(self.psf_models,(list,tuple)):
                    self.psf_models = [self.psf_models]

        elif isinstance(obs_in,ObsList):
            obs_list=obs_in

        else:
            raise ValueError("obs should be Observation or ObsList")

        self.obs=obs_list
