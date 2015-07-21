"""
- todo
    - remove old unused fitters
"""
from __future__ import print_function

try:
    xrange = xrange
    # We have Python 2
except:
    xrange = range
    # We have Python 3

from sys import stdout
import numpy
from numpy import array, zeros, ones, diag
from numpy import exp, sqrt, where, log, log10, isfinite
from numpy import linalg
from numpy.linalg import LinAlgError
import time
from pprint import pprint

from . import gmix
from .gmix import GMix, GMixList, MultiBandGMixList

from . import _gmix

from .jacobian import Jacobian, UnitJacobian

from . import priors
from .priors import LOWVAL, BIGVAL

from .gexceptions import GMixRangeError, GMixFatalError

from .observation import Observation,ObsList,MultiBandObsList,get_mb_obs

from . import stats


MAX_TAU=0.1
MIN_ARATE=0.2
MCMC_NTRY=1

BAD_VAR=2**0
LOW_ARATE=2**1
#LARGE_TAU=2**2

# error codes in LM start at 2**0 and go to 2**3
# this is because we set 2**(ier-5)
LM_SINGULAR_MATRIX = 2**4
LM_NEG_COV_EIG = 2**5
LM_NEG_COV_DIAG = 2**6
EIG_NOTFINITE = 2**7
LM_FUNC_NOTFINITE = 2**8

LM_DIV_ZERO = 2**9

BAD_STATS=2**9

PDEF=-9.999e9
CDEF=9.999e9

class FitterBase(object):
    """
    Base for other fitters

    The basic input is the Observation (or ObsList or MultiBandObsList)

    Designed to fit many images at once.  For this reason, a jacobian
    transformation is used to put all on the same system; this is part of each
    Observation object. For the same reason, the center of the model is
    relative to "zero", which points to the common center used by all
    transformation objects; the row0,col0 in pixels for each should correspond
    to that center in the common coordinates (e.g. sky coords)

    Fluxes and sizes will also be in the transformed system.
    
    """
    def __init__(self, obs, model, **keys):
        self.keys=keys

        self.margsky = keys.get('margsky', False)
        self.use_logpars=keys.get('use_logpars',False)

        # psf fitters might not have this set to 1
        self.nsub=keys.get('nsub',1)

        self.set_obs(obs)

        self.prior = keys.get('prior',None)

        # in this case, image, weight, jacobian, psf are going to
        # be lists of lists.

        self.model=gmix.get_model_num(model)
        self.model_name=gmix.get_model_name(self.model)
        self._set_npars()

        self._set_totpix()

        self._gmix_all=None

        #robust fitting
        self.nu = keys.get('nu', 0.0)

        if 'aperture' in keys:
            self.set_aperture(keys['aperture'])

    def get_result(self):
        """
        Result will not be non-None until sampler is run
        """

        if not hasattr(self,'_result'):
            raise ValueError("No result, you must run_mcmc and calc_result first")
        return self._result

    def get_gmix(self, band=0):
        """
        Get a gaussian mixture at the "best" parameter set, which
        definition depends on the sub-class
        """
        res=self.get_result()
        pars=self.get_band_pars(res['pars'], band)
        return gmix.make_gmix_model(pars, self.model)

    def set_aperture(self, aper):
        """
        set the circular aperture for likelihood evaluations. only used by
        calc_lnprob currently
        """
        self.obs.set_aperture(aper)

    def set_obs(self, obs_in):
        """
        Input should be an Observation, ObsList, or MultiBandObsList
        """

        self.obs = get_mb_obs(obs_in)


        self.nband=len(self.obs)

        if self.margsky:
            for band_obs in self.obs:
                for tobs in band_obs:
                    tobs.model_image=tobs.image*0
                    tobs.image_mean=_gmix.get_image_mean(tobs.image, tobs.weight)


    def _set_totpix(self):
        """
        Make sure the data are consistent.
        """

        totpix=0
        for obs_list in self.obs:
            for obs in obs_list:
                shape=obs.image.shape
                totpix += shape[0]*shape[1]

        self.totpix=totpix

    def _set_npars(self):
        """
        nband should be set in set_lists, called before this
        """
        self.npars=gmix.get_model_npars(self.model) + self.nband-1


    def get_effective_npix_old(self):
        """
        Because of the weight map, each pixel gets a different weight in the
        chi^2.  This changes the effective degrees of freedom.  The extreme
        case is when the weight is zero; these pixels are essentially not used.

        We replace the number of pixels with

            eff_npix = sum(weights)maxweight
        """
        raise RuntimeError("this is bogus")
        if not hasattr(self, 'eff_npix'):
            wtmax = 0.0
            wtsum = 0.0

            for obs_list in self.obs:
                for obs in obs_list:
                    wt=obs.weight

                    this_wtmax = wt.max()
                    if this_wtmax > wtmax:
                        wtmax = this_wtmax

                    wtsum += wt.sum()

            self.eff_npix=wtsum/wtmax

        if self.eff_npix <= 0:
            self.eff_npix=1.e-6

        return self.eff_npix


    def calc_lnprob(self, pars, more=False, get_priors=False):
        """
        This is all we use for mcmc approaches, but also used generally for the
        "get_fit_stats" method.  For the max likelihood fitter we also have a
        _get_ydiff method
        """

        nsub=self.nsub



        try:

            # these are the log pars (if working in log space)
            ln_priors = self._get_priors(pars)

            lnprob = 0.0
            s2n_numer=0.0
            s2n_denom=0.0
            npix = 0


            self._fill_gmix_all(pars)
            for band in xrange(self.nband):

                obs_list=self.obs[band]
                gmix_list=self._gmix_all[band]

                for obs,gm in zip(obs_list, gmix_list):
                    
                    if self.nu > 2.0:
                        res = gm.get_loglike_robust(obs, self.nu, nsub=nsub, more=True)
                    elif self.margsky:
                        res = gm.get_loglike_margsky(obs, obs.model_image, 
                                                     nsub=nsub, more=True)
                    else:
                        res = gm.get_loglike(obs, nsub=nsub, more=True)

                    lnprob    += res['loglike']
                    s2n_numer += res['s2n_numer']
                    s2n_denom += res['s2n_denom']
                    npix      += res['npix']

            # total over all bands
            lnprob += ln_priors

        except GMixRangeError:
            lnprob = LOWVAL
            s2n_numer=0.0
            s2n_denom=BIGVAL
            npix = 0

        if more:
            return {'lnprob':lnprob,
                    's2n_numer':s2n_numer,
                    's2n_denom':s2n_denom,
                    'npix':npix}
        else:
            if get_priors:
                return lnprob, ln_priors
            else:
                return lnprob

    def get_fit_stats(self, pars):
        """
        Get some fit statistics for the input pars.

        pars must be in the log scaling!
        """
        npars=self.npars

        res=self.calc_lnprob(pars, more=True)

        if res['s2n_denom'] > 0:
            s2n=res['s2n_numer']/sqrt(res['s2n_denom'])
        else:
            s2n=0.0

        chi2    = res['lnprob']/(-0.5)
        dof     = res['npix'] - self.npars
        chi2per = chi2/dof

        res['chi2per'] = chi2per
        res['dof']     = dof
        res['s2n_w']   = s2n

        return res

    def _make_model(self, band_pars):
        gm0=gmix.make_gmix_model(band_pars, self.model)
        return gm0

    def _init_gmix_all(self, pars):
        """
        input pars are in linear space

        initialize the list of lists of gaussian mixtures
        """

        if self.obs[0][0].has_psf():
            self.dopsf=True
        else:
            self.dopsf=False
            
        gmix_all0 = MultiBandGMixList()
        gmix_all  = MultiBandGMixList()

        for band,obs_list in enumerate(self.obs):
            gmix_list0=GMixList()
            gmix_list=GMixList()

            # pars for this band, in linear space
            band_pars=self.get_band_pars(pars, band)

            for obs in obs_list:
                gm0 = self._make_model(band_pars)
                if self.dopsf:
                    psf_gmix=obs.psf.gmix
                    gm=gm0.convolve(psf_gmix)
                else:
                    gm=gm0.copy()

                gmix_list0.append(gm0)
                gmix_list.append(gm)

            gmix_all0.append(gmix_list0)
            gmix_all.append(gmix_list)

        self._gmix_all0 = gmix_all0
        self._gmix_all  = gmix_all

    def _fill_gmix(self, gm, band_pars):
        _gmix.gmix_fill(gm._data, band_pars, gm._model)

    def _convolve_gmix(self, gm, gm0, psf_gmix):
        _gmix.convolve_fill(gm._data, gm0._data, psf_gmix._data)

    def _fill_gmix_all(self, pars):
        """
        input pars are in linear space

        Fill the list of lists of gmix objects for the given parameters
        """

        if not self.dopsf:
            self._fill_gmix_all_nopsf(pars)
            return

        for band,obs_list in enumerate(self.obs):
            gmix_list0=self._gmix_all0[band]
            gmix_list=self._gmix_all[band]

            # pars for this band, in linear space
            band_pars=self.get_band_pars(pars, band)

            for i,obs in enumerate(obs_list):

                psf_gmix=obs.psf.gmix

                gm0=gmix_list0[i]
                gm=gmix_list[i]

                #gm0.fill(band_pars)
                self._fill_gmix(gm0, band_pars)
                #_gmix.gmix_fill(gm0._data, band_pars, gm0._model)
                self._convolve_gmix(gm, gm0, psf_gmix)
                #_gmix.convolve_fill(gm._data, gm0._data, psf_gmix._data)

    def _fill_gmix_all_nopsf(self, pars):
        """
        Fill the list of lists of gmix objects for the given parameters
        """

        for band,obs_list in enumerate(self.obs):
            gmix_list0=self._gmix_all0[band]
            gmix_list=self._gmix_all[band]

            # pars for this band, in linear space
            band_pars=self.get_band_pars(pars, band)

            for i,obs in enumerate(obs_list):

                gm0=gmix_list0[i]
                gm=gmix_list[i]

                try:
                    _gmix.gmix_fill(gm0._data, band_pars, gm0._model)
                    _gmix.gmix_fill(gm._data, band_pars, gm._model)
                except ZeroDivisionError:
                    raise GMixRangeError("zero division")


    def _get_priors(self, pars):
        """
        get the sum of ln(prob) from the priors or 0.0 if
        no priors were sent
        """
        if self.prior is None:
            return 0.0
        else:
            return self.prior.get_lnprob_scalar(pars)

    def plot_residuals(self, title=None, show=False,
                       width=1920, height=1200,**keys):
        import images
        import biggles

        biggles.configure('screen','width', width)
        biggles.configure('screen','height', height)

        res=self.get_result()
        try:
            self._fill_gmix_all(res['pars'])
        except GMixRangeError as gerror:
            print(str(gerror))
            return None

        plist=[]
        for band in xrange(self.nband):

            band_list=[]

            obs_list=self.obs[band]
            gmix_list=self._gmix_all[band]
            
            nim=len(gmix_list)

            ttitle='band: %s' % band
            if title is not None:
                ttitle='%s %s' % (title, ttitle)

            for i in xrange(nim):

                this_title = '%s cutout: %d' % (ttitle, i+1)

                obs=obs_list[i]
                gm=gmix_list[i]

                im=obs.image
                wt=obs.weight
                j=obs.jacobian

                model=gm.make_image(im.shape,jacobian=j, nsub=self.nsub)

                showim = im*wt
                showmod = model*wt

                sub_tab=images.compare_images(showim, showmod,show=False,
                                              label1='galaxy',
                                              label2='model',
                                              **keys)
                sub_tab.title=this_title

                band_list.append(sub_tab)

                if show:
                    sub_tab.show()

            plist.append(band_list)
        return plist

    def calc_cov(self, h, m, diag_on_fail=True):
        """
        Run get_cov() to calculate the covariance matrix at the best-fit point.
        If all goes well, add 'pars_cov', 'pars_err', and 'g_cov' to the result
        array

        Note in get_cov, if the Hessian is singular, a diagonal cov matrix is
        attempted to be inverted. If that finally fails LinAlgError is raised.
        In that case we catch it and set a flag EIG_NOTFINITE and the cov is
        not added to the result dict

        Also if there are negative diagonal elements of the cov matrix, the 
        EIG_NOTFINITE flag is set and the cov is not added to the result dict
        """

        res=self.get_result()

        bad=True

        try:
            cov = self.get_cov(res['pars'], h=h, m=m, diag_on_fail=diag_on_fail)

            cdiag = diag(cov)

            w,=where(cdiag <= 0)
            if w.size == 0:

                err = sqrt(cdiag)
                w,=where(isfinite(err))
                if w.size != err.size:
                    print_pars(err, front="diagonals not finite:")
                else:
                    # everything looks OK
                    bad=False
            else:
                print_pars(cdiag,front='    diagonals negative:')

        except LinAlgError:
            print("caught LinAlgError")

        if bad:
            res['flags'] |= EIG_NOTFINITE
        else:
            res['pars_cov'] = cov
            res['pars_err']= err

            if len(err) >= 6:
                res['g_cov'] = cov[2:2+2, 2:2+2]

    def get_cov(self, pars, h, m, diag_on_fail=True):
        """
        calculate the covariance matrix at the specified point

        This method understands the natural bounds on ellipticity.
        If the ellipticity is larger than 1-m*h then it is scaled
        back, perserving the angle.

        If the Hessian is singular, an attempt is made to invert
        a diagonal version. If that fails, LinAlgError is raised.

        parameters
        ----------
        pars: array
            Array of parameters at which to evaluate the cov matrix
        h: step size, optional
            Step size for finite differences, default 1.0e-3
        m: scalar
            The max allowed ellipticity is 1-m*h.
            Note the derivatives require evaluations at +/- h,
            so m should be greater than 1.

        Raises
        ------
        LinAlgError:
            If the hessian is singular a diagonal version is tried
            and if that fails finally a LinAlgError is raised.
        """
        import covmatrix

        # get a copy as an array
        pars=numpy.array(pars)

        g1=pars[2]
        g2=pars[3]

        g=sqrt(g1**2 + g2**2)

        maxg=1.0-m*h

        if g > maxg:
            fac = maxg/g
            g1 *= fac
            g2 *= fac
            pars[2] = g1
            pars[3] = g2

        # we could call covmatrix.get_cov directly but we want to fall back
        # to a diagonal hessian if it is singular

        hess=covmatrix.calc_hess(self.calc_lnprob, pars, h)

        try:
            cov = -linalg.inv(hess)
        except LinAlgError:
            # pull out a diagonal version of the hessian
            # this might still fail

            if diag_on_fail:
                hdiag=diag(diag(hess))
                cov = -linalg.inv(hess)
            else:
                raise
        return cov



class TemplateFluxFitter(FitterBase):
    """
    We fix the center, so this is linear.  Just cross-correlations
    between model and data.

    The center of the jacobian(s) must point to a common place on the sky, and
    if the center is input (to reset the gmix centers),) it is relative to that
    position

    parameters
    -----------
    obs: Observation or ObsList
        See ngmix.observation.Observation.  The observation should
        have a gmix set.
    cen: 2-element sequence, optional

        The center in sky coordinates, relative to the jacobian center(s).  If
        not sent, the gmix (or psf gmix) object(s) in the observation(s) should
        be set to the wanted center.

    """
    def __init__(self, obs, **keys):

        self.keys=keys
        self.do_psf=keys.get('do_psf',False)
        self.cen=keys.get('cen',None)

        if self.cen is None:
            self.cen_was_sent=False
        else:
            self.cen_was_sent=True

        self.set_obs(obs)

        self.model_name='template'
        self.npars=1

        self._set_totpix()

    def go(self):
        """
        calculate the flux using zero-lag cross-correlation
        """
        xcorr_sum=0.0
        msq_sum=0.0

        chi2=0.0

        cen=self.cen
        nobs=len(self.obs)

        for ipass in [1,2]:
            for iobs in xrange(nobs):
                obs=self.obs[iobs]
                gm = self.gmix_list[iobs]

                im=obs.image
                wt=obs.weight
                j=obs.jacobian

                if ipass==1:
                    gm.set_psum(1.0)
                    model=gm.make_image(im.shape, jacobian=j)
                    xcorr_sum += (model*im*wt).sum()
                    msq_sum += (model*model*wt).sum()
                else:
                    gm.set_psum(flux)
                    model=gm.make_image(im.shape, jacobian=j)
                    chi2 +=( (model-im)**2 *wt ).sum()
            if ipass==1:
                flux = xcorr_sum/msq_sum

        dof=self.get_dof()
        chi2per=9999.0
        if dof > 0:
            chi2per=chi2/dof

        flags=0
        arg=chi2/msq_sum/(self.totpix-1) 
        if arg >= 0.0:
            flux_err = sqrt(arg)
        else:
            flags=BAD_VAR
            flux_err=9999.0

        self._result={'model':self.model_name,
                      'flags':flags,
                      'chi2per':chi2per,
                      'dof':dof,
                      'flux':flux,
                      'flux_err':flux_err}

    def get_dof(self):
        """
        Effective def based on effective number of pixels
        """
        npix=self.get_effective_npix()
        dof = npix-self.npars
        if dof <= 0:
            dof = 1.e-6
        return dof



    def set_obs(self, obs_in):
        """
        Input should be an Observation, ObsList
        """

        if isinstance(obs_in,Observation):
            obs_list=ObsList()
            obs_list.append(obs_in)
        elif isinstance(obs_in,ObsList):
            obs_list=obs_in
        else:
            raise ValueError("obs should be Observation or ObsList")

        cen=self.cen
        gmix_list=[]
        for obs in obs_list:
            # these return copies, ok to modify
            if self.do_psf:
                gmix=obs.get_psf_gmix()
            else:
                gmix=obs.get_gmix()

            if self.cen_was_sent:
                gmix.set_cen(cen[0], cen[1])

            gmix_list.append(gmix)

        self.obs = obs_list
        self.gmix_list = gmix_list

    def _set_totpix(self):
        """
        Make sure the data are consistent.
        """

        totpix=0
        for obs in self.obs:
            shape=obs.image.shape
            totpix += shape[0]*shape[1]

        self.totpix=totpix

    def get_effective_npix(self):
        """
        We don't use all pixels, only those with weight > 0
        """
        if not hasattr(self, 'eff_npix'):

            npix=0
            for obs in self.obs:
                wt=obs.weight

                w=where(wt > 0)
                npix += w[0].size

            self.eff_npix=npix

        return self.eff_npix

    def get_npix(self):
        """
        just get the total number of pixels in all images
        """
        if not hasattr(self, '_npix'):
            npix=0
            for obs in self.obs:
                npix += obs.image.size

            self._npix=npix

        return self._npix

class FracdevFitterMax(FitterBase):
    def __init__(self, obs, exp_pars, dev_pars, **keys):
        """
        obs must have psf (with gmix) set
        """
        self.prior=keys.get('prior',None)
        method=keys.get('method','nm')
        if self.prior is not None:
            assert method=='nm','if prior sent method must be nm'
            if self.prior.ndim > 1:
                if len(exp_pars) > 6:
                    raise RuntimeError("currently only joint "
                                       "fracdev prior single band")

        self.method=method

        self.margsky=False
        self.use_logpars=keys.get('use_logpars',False)
        self.set_obs(obs)
        self._set_images(exp_pars, dev_pars)

        self.model_name='fracdev'

        self._set_totpix()

        if 'aperture' in keys:
            self.set_aperture(keys['aperture'])

        pars=keys.get('pars',None)
        if pars is None:
            pars={'maxfev':4000}

        self.pars=pars
        self.npars=1

        self.fdiff_size=self.totpix

    def go(self, fracdev_guess):
        """
        run max like fit
        """
        fracdev_guess=array(fracdev_guess,dtype='f8',ndmin=1,copy=False)
        if fracdev_guess.size != 1:
            raise ValueError("fracdev is a scalar or len 1 array")

        if self.method=='lm':
            return self._go_lm(fracdev_guess)
        else:
            return self._go_nm(fracdev_guess)

    def _go_lm(self, fracdev_guess):
        """
        Run leastsq and set the result
        """

        n_prior_pars=0
        result = run_leastsq(self._calc_fdiff,
                             fracdev_guess,
                             n_prior_pars,
                             **self.pars)

        result['model'] = self.model_name
        if result['flags']==0:
            result['fracdev'] = result['pars'][0].copy()
            result['fracdev_err'] = sqrt(result['pars_cov'][0,0])

        self._result=result
 
    def _go_nm(self, fracdev_guess):
        """
        Run leastsq and set the result
        """
        from .simplex import minimize_neldermead_rel as minimize_neldermead

        result = minimize_neldermead(self._calc_neg_lnprob,
                                     fracdev_guess,
                                     **self.pars)

        self._result = result

        result['model'] = self.model_name
        if result['success']:
            result['flags'] = 0
        else:
            result['flags'] = 1

        if 'x' in result:
            pars=result['x']
            result['pars'] = pars

            h=1.0e-3
            m=5.0
            self.calc_cov(h, m)

            result['fracdev'] = result['pars'][0].copy()

            if 'pars_err' in result:
                result['fracdev_err'] = result['pars_err'][0]

    def get_cov(self, pars, h, m, diag_on_fail=True):
        """
        calculate the covariance matrix at the specified point

        This method understands the natural bounds on ellipticity.
        If the ellipticity is larger than 1-m*h then it is scaled
        back, perserving the angle.

        If the Hessian is singular, an attempt is made to invert
        a diagonal version. If that fails, LinAlgError is raised.

        parameters
        ----------
        pars: array
            Array of parameters at which to evaluate the cov matrix
        h: step size, optional
            Step size for finite differences, default 1.0e-3
        m: scalar
            The max allowed ellipticity is 1-m*h.
            Note the derivatives require evaluations at +/- h,
            so m should be greater than 1.

        Raises
        ------
        LinAlgError:
            If the hessian is singular a diagonal version is tried
            and if that fails finally a LinAlgError is raised.
        """
        import covmatrix

        hess=covmatrix.calc_hess(self.calc_lnprob, pars, h)

        try:
            cov = -linalg.inv(hess)
        except LinAlgError:
            # pull out a diagonal version of the hessian
            # this might still fail
            if diag_on_fail:
                hdiag=diag(diag(hess))
                cov = -linalg.inv(hess)
            else:
                raise
        return cov



    def _calc_fdiff(self, pars, **keys):
        """

        vector with (model-data)/error.

        The npars elements contain -ln(prior)
        """

        fracdev = pars[0]

        # we cannot keep sending existing array into leastsq, don't know why
        fdiff=zeros(self.fdiff_size)

        if fracdev <= -1 or fracdev >= 2.0:
            fdiff[:] = LOWVAL
            return fdiff

        start=0

        for band in xrange(self.nband):

            obs_list=self.obs[band]

            images   = self._images[band]
            sweights = self._sweights[band]
            eimages  = self._eimages[band]
            dimages  = self._dimages[band]

            for epoch,obs in enumerate(obs_list):

                image   = images[epoch]
                sweight = sweights[epoch]

                npix = image.size

                tfdiff    = eimages[epoch].copy()
                dev_image = dimages[epoch].copy()

                tfdiff *= (1.0-fracdev)

                dev_image *= fracdev

                tfdiff += dev_image

                tfdiff -= image
                tfdiff *= sweight

                fdiff[start:start+npix] = tfdiff

                start += npix


        return fdiff

    def _calc_neg_lnprob(self, pars):
        return -self.calc_lnprob(pars)

    def calc_lnprob(self, pars):
        """
        return log(prob)
        """

        fracdev = pars[0]
        ifracdev = 1.0-pars[0]

        if fracdev <= -1 or fracdev >= 2.0:
            return LOWVAL


        lnprob = 0.0

        for band in xrange(self.nband):

            obs_list=self.obs[band]

            images   = self._images[band]
            weights  = self._weights[band]
            eimages  = self._eimages[band]
            dimages  = self._dimages[band]

            for epoch,obs in enumerate(obs_list):

                image   = images[epoch]
                weight = weights[epoch]

                npix = image.size

                exp_image = eimages[epoch].copy()
                dev_image = dimages[epoch].copy()

                exp_image *= ifracdev
                dev_image *= fracdev

                tfdiff = exp_image

                tfdiff += dev_image

                tfdiff -= image
                tfdiff *= tfdiff
                tfdiff *= weight

                lnprob += tfdiff.sum()


        lnprob *= (-0.5)

        if self.prior is not None:
            if self.prior.ndim==1:
                #print("using prior")
                lnprob += self.prior.get_lnprob_scalar(pars)
            elif self.prior.ndim==2:
                #print("using joint F prior")
                # it is a joint prior on F and fracdev
                Fe = ifracdev* self.lin_exp_F[0]
                Fd = fracdev * self.lin_dev_F[0]
                F = Fe + Fd

                bad=False
                if self.use_logpars:
                    if F <= 0:
                        bad=True
                    else:
                        allpars=numpy.array( [log(F), fracdev] )
                else:
                    allpars=numpy.array( [F, fracdev] )

                if bad:
                    lnprob = -numpy.inf
                else:
                    lnp = self.prior.get_lnprob_scalar(allpars)
                    lnprob += lnp

            else:
                #print("using joint prior")
                # it is a joint prior on TF and fracdev
                Fe = ifracdev* self.lin_exp_F[0]
                Fd = fracdev * self.lin_dev_F[0]
                F = Fe + Fd
                T = (Fe*self.lin_exp_T + Fd*self.lin_dev_T)/F

                bad=False
                if self.use_logpars:
                    if T <= 0 or F <= 0:
                        bad=True
                    else:
                        allpars=numpy.array( [log(T), log(F), fracdev] )
                else:
                    allpars=numpy.array( [T, F, fracdev] )

                if bad:
                    lnprob = -numpy.inf
                else:
                    lnp = self.prior.get_lnprob_scalar(allpars)
                    lnprob += lnp

        return lnprob



    def _set_images(self, exp_pars, dev_pars):
        exp_pars=array(exp_pars,dtype='f8',ndmin=1,copy=True)
        dev_pars=array(dev_pars,dtype='f8',ndmin=1,copy=True)

        if self.use_logpars:
            exp_pars[4:] = exp(exp_pars[4:])
            dev_pars[4:] = exp(dev_pars[4:])
        self.lin_exp_T = exp_pars[4]
        self.lin_dev_T = dev_pars[4]
        self.lin_exp_F = exp_pars[5:].copy()
        self.lin_dev_F = dev_pars[5:].copy()

        self.TdByTe = dev_pars[4]/exp_pars[4]

        nb=self.nband

        enb=exp_pars.size-5
        dnb=dev_pars.size-5
        mess="expected %d bands, got %d"
        if enb != nb:
            raise ValueError(mess % (nb,enb))
        if dnb != nb:
            raise ValueError(mess % (nb,dnb))


        tepars = zeros(6)
        tdpars = zeros(6)

        tepars[0:5] = exp_pars[0:5]
        tdpars[0:5] = dev_pars[0:5]


        all_images=[]
        all_weights=[]
        all_sweights=[]
        all_eimages=[]
        all_dimages=[]

        for band in xrange(nb):
            # these will get reset later based on fracdev
            tepars[5] = exp_pars[5+band]
            tdpars[5] = dev_pars[5+band]

            band_obs = self.obs[band]
            nepoch = len(band_obs)

            images=[]
            weights=[]
            sweights=[]
            eimages=[]
            dimages=[]

            for iepoch in xrange(nepoch):
                epoch_obs = band_obs[iepoch]

                image=epoch_obs.image.copy()

                psf_gmix=epoch_obs.psf.gmix

                cweight = epoch_obs.weight.clip(min=0.0)
                sweight = sqrt(cweight)

                tegm = gmix.GMixModel(tepars,'exp')
                tdgm = gmix.GMixModel(tdpars,'dev')

                egm = tegm.convolve( psf_gmix )
                dgm = tdgm.convolve( psf_gmix )

                eimage = egm.make_image(image.shape,
                                        jacobian=epoch_obs.jacobian)
                dimage = dgm.make_image(image.shape,
                                        jacobian=epoch_obs.jacobian)


                images.append(image.ravel())
                weights.append(cweight.ravel())
                sweights.append(sweight.ravel())
                eimages.append(eimage.ravel())
                dimages.append(dimage.ravel())

            all_images.append(images)
            all_weights.append(weights)
            all_sweights.append(sweights)
            all_eimages.append(eimages)
            all_dimages.append(dimages)

        self._images   = all_images
        self._weights = all_weights
        self._sweights = all_sweights
        self._eimages  = all_eimages
        self._dimages  = all_dimages


class FracdevFitter(FitterBase):
    def __init__(self, obs, exp_pars, dev_pars, **keys):
        """
        obs must have psf (with gmix) set
        """

        self.npars=1
        self.use_logpars=keys.get('use_logpars',False)

        self.margsky=False
        self.set_obs(obs)

        self.model_name='fracdev'

        self._set_totpix()

        self._set_arrays(exp_pars, dev_pars)

        self._do_lstsq()

    def _do_lstsq(self):
        """
        (data - expmod) = (devmod-expmod) * fracdev

        or

        Y = X * fracdev
        """
        
        X = self.X[:,numpy.newaxis]
        Y = self.Y

        self._result={'model': 'fracdev', 'flags':0,'nfev':1}


        result=self._result

        try:
            pars, resid, _, _ = numpy.linalg.lstsq(X, Y)

            result['pars'] = pars
            result['fracdev'] = pars[0]

            self.calc_cov()

            result['fracdev_err'] = result['pars_err'][0]

        except LinAlgError:
            result['flags'] |= EIG_NOTFINITE


    def calc_cov(self):
        """
        calculate the error
        """
        import covmatrix

        res=self.get_result()

        h=1.0e-3
        hess=covmatrix.calc_hess(self.calc_lnprob, res['pars'], h)
        try:
            cov = -linalg.inv(hess)
        except LinAlgError:
            # pull out a diagonal version of the hessian
            # this might still fail
            hdiag=diag(diag(hess))
            cov = -linalg.inv(hess)

        pars_err=sqrt(cov[0,0])
        res['pars_cov'] = cov
        res['pars_err'] = sqrt(diag(cov))

    def calc_lnprob(self, pars):
        fracdev = pars[0]

        lnp_array = (self.Y - fracdev*self.X)
        lnp_array *= lnp_array

        lnp = lnp_array.sum()
        lnp *= (-0.5)

        return lnp

    def _set_arrays(self, exp_pars, dev_pars):
        """
        (data - expmod) = (devmod-expmod) * fracdev

        or

        Y = X * fracdev
        """
        X = zeros(self.totpix)
        Y = zeros(self.totpix)

        exp_pars=array(exp_pars,dtype='f8',ndmin=1,copy=True)
        dev_pars=array(dev_pars,dtype='f8',ndmin=1,copy=True)

        if self.use_logpars:
            exp_pars[4:] = exp(exp_pars[4:])
            dev_pars[4:] = exp(dev_pars[4:])

        nb=self.nband

        enb=exp_pars.size-5
        dnb=dev_pars.size-5
        mess="expected %d bands, got %d"
        if enb != nb:
            raise ValueError(mess % (nb,enb))
        if dnb != nb:
            raise ValueError(mess % (nb,dnb))


        tepars = zeros(6)
        tdpars = zeros(6)

        tepars[0:5] = exp_pars[0:5]
        tdpars[0:5] = dev_pars[0:5]

        start=0
        for band in xrange(nb):
            # these will get reset later based on fracdev
            tepars[5] = exp_pars[5+band]
            tdpars[5] = dev_pars[5+band]

            band_obs = self.obs[band]
            nepoch = len(band_obs)

            for iepoch in xrange(nepoch):
                epoch_obs = band_obs[iepoch]

                image=epoch_obs.image.copy()

                psf_gmix=epoch_obs.psf.gmix

                cweight = epoch_obs.weight.clip(min=0.0)
                sweight = sqrt(cweight)

                tegm = gmix.GMixModel(tepars,'exp')
                tdgm = gmix.GMixModel(tdpars,'dev')

                egm = tegm.convolve( psf_gmix )
                dgm = tdgm.convolve( psf_gmix )

                eimage = egm.make_image(image.shape,
                                        jacobian=epoch_obs.jacobian)
                dimage = dgm.make_image(image.shape,
                                        jacobian=epoch_obs.jacobian)


                end = start + image.size
                X[start:end] = ( (dimage-eimage)*sweight ).ravel()
                Y[start:end] = ( (image-eimage)*sweight  ).ravel()

                start += image.size


        self.X = X
        self.Y = Y



class MaxSimple(FitterBase):
    """
    A class for direct maximization of the likelihood.
    Useful for seeding model parameters.
    """
    def __init__(self, obs, model, method='Nelder-Mead', **keys):
        super(MaxSimple,self).__init__(obs, model, **keys)
        self._obs = obs
        self._model = model
        self.method = method
        self._band_pars = numpy.zeros(6)
        
    def _setup_data(self, guess):
        """
        initialize the gaussian mixtures
        """

        if hasattr(self,'_result'):
            del self._result

        self.flags=0

        npars=guess.size
        mess="guess has npars=%d, expected %d" % (npars,self.npars)
        assert (npars==self.npars),mess

        try:
            # this can raise GMixRangeError
            self._init_gmix_all(guess)
        except ZeroDivisionError:
            raise GMixRangeError("got zero division")

    def get_band_pars(self, pars_in, band):
        """
        Get linear pars for the specified band
        """

        pars=self._band_pars

        if self.use_logpars:
            _gmix.convert_simple_double_logpars_band(pars_in, pars, band)
        else:
            pars[0:5] = pars_in[0:5]
            pars[5] = pars_in[5+band]

        return pars

    def neglnprob(self, pars):
        return -1.0*self.calc_lnprob(pars)

    def run_max(self, guess, **keys):
        """
        Run maximizer and set the result.

        extra keywords for nm are 
        --------------------------
        xtol: float, optional
            Tolerance in the vertices, relative to the vertex with
            the lowest function value.  Default 1.0e-4
        ftol: float, optional
            Tolerance in the function value, relative to the
            lowest function value for all vertices.  Default 1.0e-4
        maxiter: int, optional
            Default is npars*200
        maxfev:
            Default is npars*200
        """
        if self.method in ['nm','Nelder-Mead']:
            self.run_max_nm(guess, **keys)
        else:
            import scipy.optimize

            options={}
            options.update(keys)

            guess=numpy.array(guess,dtype='f8',copy=False)
            self._setup_data(guess)
            
            result = scipy.optimize.minimize(self.neglnprob,
                                             guess,
                                             method=self.method,
                                             options=options)
            self._result = result

            result['model'] = self.model_name
            if result['success']:
                result['flags'] = 0
            else:
                result['flags'] = result['status']

            if 'x' in result:
                pars=result['x']
                result['pars'] = pars
                result['g'] = pars[2:2+2]
            
                # based on last entry
                fit_stats = self.get_fit_stats(pars)
                result.update(fit_stats)
    go=run_max

    def run_max_nm(self, guess, **keys):
        """
        Run maximizer and set the result.

        extra keywords are 
        ------------------
        xtol: float, optional
            Tolerance in the vertices, relative to the vertex with
            the lowest function value.  Default 1.0e-4
        ftol: float, optional
            Tolerance in the function value, relative to the
            lowest function value for all vertices.  Default 1.0e-4
        maxiter: int, optional
            Default is npars*200
        maxfev:
            Default is npars*200
        """
        #from .simplex import minimize_neldermead
        from .simplex import minimize_neldermead_rel as minimize_neldermead

        options={}
        options.update(keys)

        guess=numpy.array(guess,dtype='f8',copy=False)
        self._setup_data(guess)
        
        result = minimize_neldermead(self.neglnprob,
                                     guess,
                                     **keys)
        self._result = result

        result['model'] = self.model_name
        if result['success']:
            result['flags'] = 0
        else:
            result['flags'] = 1

        if 'x' in result:
            pars=result['x']
            result['pars'] = pars
            result['g'] = pars[2:2+2]
        
            # based on last entry
            fit_stats = self.get_fit_stats(pars)
            result.update(fit_stats)

            h=1.0e-3
            m=5.0
            self.calc_cov(h, m)

class MaxCoellip(MaxSimple):
    """
    A class for direct maximization of the likelihood.
    Useful for seeding model parameters.
    """
    def __init__(self, obs, ngauss, method='Nelder-Mead', **keys):

        self._ngauss=ngauss

        super(MaxCoellip,self).__init__(obs, 'coellip', method=method, **keys)

        if self.nband != 1:
            raise ValueError("MaxCoellip only supports one band")

        # over-write the band pars created by MaxSimple
        self._band_pars=zeros(self.npars)

    def _set_npars(self):
        """
        single band, npars determined from ngauss
        """
        self.npars=4 + 2*self._ngauss

    def get_band_pars(self, pars_in, band):
        """
        Get linear pars for the specified band
        """

        if self.use_logpars:
            _gmix.convert_simple_double_logpars(pars_in, pars)
        else:
            pars=self._band_pars
            pars[:] = pars_in[:]
        return pars

      


_default_lm_pars={'maxfev':4000,
                  'ftol': 1.0e-5,
                  'xtol': 1.0e-5}

class LMSimple(FitterBase):
    """
    A class for doing a fit using levenberg marquardt

    """
    def __init__(self, obs, model, **keys):
        super(LMSimple,self).__init__(obs, model, **keys)

        # this is a dict
        # can contain maxfev (maxiter), ftol (tol in sum of squares)
        # xtol (tol in solution), etc

        lm_pars=keys.get('lm_pars',None)
        if lm_pars is None:
            lm_pars=_default_lm_pars
        self.lm_pars=lm_pars


        # center1 + center2 + shape + T + fluxes
        self.n_prior_pars=1 + 1 + 1 + 1 + self.nband

        self.fdiff_size=self.totpix + self.n_prior_pars

        self._band_pars=zeros(6)


    def run_lm(self, guess):
        """
        Run leastsq and set the result
        """

        guess=array(guess,dtype='f8',copy=False)
        self._setup_data(guess)

        result = run_leastsq(self._calc_fdiff,
                             guess,
                             self.n_prior_pars,
                             **self.lm_pars)

        result['model'] = self.model_name
        if result['flags']==0:
            result['g'] = result['pars'][2:2+2].copy()
            result['g_cov'] = result['pars_cov'][2:2+2, 2:2+2].copy()
            stat_dict=self.get_fit_stats(result['pars'])
            result.update(stat_dict)

        self._result=result
    run_max=run_lm
    go=run_lm
    
    def _setup_data(self, guess):
        """
        try very hard to initialize the mixtures
        """

        if hasattr(self,'_result'):
            del self._result

        self.flags=0

        npars=guess.size
        mess="guess has npars=%d, expected %d" % (npars,self.npars)
        assert (npars==self.npars),mess

        try:
            # this can raise GMixRangeError
            self._init_gmix_all(guess)
        except ZeroDivisionError:
            raise GMixRangeError("got zero division")

    def get_band_pars(self, pars_in, band):
        """
        Get linear pars for the specified band
        """

        pars=self._band_pars

        if self.use_logpars:
            _gmix.convert_simple_double_logpars_band(pars_in, pars, band)
        else:
            pars[0:5] = pars_in[0:5]
            pars[5] = pars_in[5+band]


        return pars


    def _calc_fdiff(self, pars, more=False):
        """

        vector with (model-data)/error.

        The npars elements contain -ln(prior)
        """

        # we cannot keep sending existing array into leastsq, don't know why
        fdiff=zeros(self.fdiff_size)

        s2n_numer=0.0
        s2n_denom=0.0
        npix = 0

        try:


            self._fill_gmix_all(pars)

            start=self._fill_priors(pars, fdiff)

            for band in xrange(self.nband):

                obs_list=self.obs[band]
                gmix_list=self._gmix_all[band]

                for obs,gm in zip(obs_list, gmix_list):

                    res = gm.fill_fdiff(obs, fdiff, start=start, nsub=self.nsub)

                    s2n_numer += res['s2n_numer']
                    s2n_denom += res['s2n_denom']
                    npix += res['npix']

                    start += obs.image.size

        except GMixRangeError as err:
            fdiff[:] = LOWVAL
            s2n_numer=0.0
            s2n_denom=BIGVAL

        if more:
            return {'fdiff':fdiff,
                    's2n_numer':s2n_numer,
                    's2n_denom':s2n_denom,
                    'npix':npix}
        else:
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
            nprior=0
        else:
            nprior=self.prior.fill_fdiff(pars, fdiff)

        return nprior


class LMGaussMom(LMSimple):
    """
    Fit gaussian in moment space
    """
    def __init__(self, obs, **keys):

        super(LMGaussMom,self).__init__(obs, 'gaussmom', **keys)

        #                 c1 c2  M1  M2   T   Ii
        self.n_prior_pars=1 + 1 + 1 + 1 + 1 + self.nband
        self.fdiff_size=self.totpix + self.n_prior_pars

    def go(self, guess):
        """
        Run leastsq and set the result
        """

        guess=array(guess,dtype='f8',copy=False)
        self._setup_data(guess)

        result = run_leastsq(self._calc_fdiff,
                             guess,
                             self.n_prior_pars,
                             **self.lm_pars)

        result['model'] = self.model_name
        if result['flags']==0:
            stat_dict=self.get_fit_stats(result['pars'])
            result.update(stat_dict)

        self._result=result

    def get_band_pars(self, pars_in, band):
        """
        Get linear pars for the specified band

        pars are [c1,c2,M1,M2,T,I1,I2...]

        Where M1 = Icc-Irr
              m2 = 2*Irc
        """

        pars=self._band_pars

        pars[0:5] = pars_in[0:5]
        pars[5] = pars_in[5+band]

        return pars

    def calc_cov(self, h, *args, **kw):
        """
        Run get_cov() to calculate the covariance matrix at the best-fit point.
        If all goes well, add 'pars_cov', 'pars_err', and 'g_cov' to the result
        array

        Note in get_cov, if the Hessian is singular, a diagonal cov matrix is
        attempted to be inverted. If that finally fails LinAlgError is raised.
        In that case we catch it and set a flag EIG_NOTFINITE and the cov is
        not added to the result dict

        Also if there are negative diagonal elements of the cov matrix, the 
        EIG_NOTFINITE flag is set and the cov is not added to the result dict
        """

        diag_on_fail=kw.get('diag_on_fail',True)

        res=self.get_result()

        bad=True

        try:
            cov = self.get_cov(res['pars'], h=h, diag_on_fail=diag_on_fail)

            cdiag = diag(cov)

            w,=where(cdiag <= 0)
            if w.size == 0:

                err = sqrt(cdiag)
                w,=where(isfinite(err))
                if w.size != err.size:
                    print_pars(err, front="diagonals not finite:")
                else:
                    # everything looks OK
                    bad=False
            else:
                print_pars(cdiag,front='    diagonals negative:')

        except LinAlgError:
            print("caught LinAlgError")

        if bad:
            res['flags'] |= EIG_NOTFINITE
        else:
            res['pars_cov'] = cov
            res['pars_err']= err

            if len(err) >= 6:
                res['g_cov'] = cov[2:2+2, 2:2+2]

    def get_cov(self, pars, h, diag_on_fail=True):
        """
        calculate the covariance matrix at the specified point

        parameters
        ----------
        pars: array
            Array of parameters at which to evaluate the cov matrix
        h: step size, optional
            Step size for finite differences, default 1.0e-3

        Raises
        ------
        LinAlgError:
            If the hessian is singular a diagonal version is tried
            and if that fails finally a LinAlgError is raised.
        """
        import covmatrix

        # get a copy as an array
        pars=numpy.array(pars)

        # we could call covmatrix.get_cov directly but we want to fall back
        # to a diagonal hessian if it is singular

        hess=covmatrix.calc_hess(self.calc_lnprob, pars, h)

        try:
            cov = -linalg.inv(hess)
        except LinAlgError:
            # pull out a diagonal version of the hessian
            # this might still fail

            if diag_on_fail:
                hdiag=diag(diag(hess))
                cov = -linalg.inv(hess)
            else:
                raise
        return cov

class LMSimpleRound(LMSimple):
    """
    This version fits [cen1,cen2,T,F]
    """
    def __init__(self, *args, **keys):
        super(LMSimpleRound,self).__init__(*args, **keys)

        self.npars = self.npars-2
        self.n_prior_pars=self.n_prior_pars - 1
        self.fdiff_size=self.totpix + self.n_prior_pars

        if self.use_logpars:
            # allbands
            self._pars_allbands=zeros(self.npars+2)


    def run_lm(self, guess):
        """
        Run leastsq and set the result
        """

        guess=array(guess,dtype='f8',copy=False)
        self._setup_data(guess)

        result = run_leastsq(self._calc_fdiff,
                             guess,
                             self.n_prior_pars,
                             **self.lm_pars)

        result['model'] = self.model_name
        if result['flags']==0:
            stat_dict=self.get_fit_stats(result['pars'])
            result.update(stat_dict)

        self._result=result

    def get_band_pars(self, pars_in, band):
        """
        Get linear pars for the specified band
        """


        pars=self._band_pars
        if self.use_logpars:
            _get_simple_band_pars_round_logpars(pars_in,
                                                self._pars_allbands,
                                                pars,
                                                band)
        else:
            _get_simple_band_pars_round_linpars(pars_in,
                                                pars,
                                                band)
        return pars


class LMSimpleFixT(LMSimple):
    """
    This version fits [cen1,cen2,g1,g2,F]
    """
    def __init__(self, *args, **keys):
        super(LMSimpleFixT,self).__init__(*args, **keys)

        self.npars = self.npars-1
        self.n_prior_pars=self.n_prior_pars - 1
        self.fdiff_size=self.totpix + self.n_prior_pars

        self.T = keys['T']

    def run_lm(self, guess):
        """
        Run leastsq and set the result
        """

        guess=array(guess,dtype='f8',copy=False)
        self._setup_data(guess)

        result = run_leastsq(self._calc_fdiff, guess, self.n_prior_pars, **self.lm_pars)

        result['model'] = self.model_name
        if result['flags']==0:
            stat_dict=self.get_fit_stats(result['pars'])
            result.update(stat_dict)

        self._result=result

    def get_band_pars(self, pars_in, band):
        """
        Get linear pars for the specified band
        """


        pars=self._band_pars
        pars[0:4] = pars_in[0:4]
        pars[4]=self.T

        if self.use_logpars:
            pars[5]=exp( pars_in[4+band] )
        else:
            pars[5]=pars_in[4]

        return pars

class LMSimpleGOnly(LMSimple):
    """
    This version fits [cen1,cen2,g1,g2,F]
    """
    def __init__(self, *args, **keys):
        super(LMSimpleGOnly,self).__init__(*args, **keys)

        self.pars_in0 = array(keys['pars'], dtype='f8')

        pars_in=self.pars_in0.copy()
        if self.use_logpars:
            pars_in[4:] = exp(self.pars_in0[4:])
        self.pars_in=pars_in

        self.npars = 2
        self.n_prior_pars=1
        self.fdiff_size=self.totpix + self.n_prior_pars

    def run_lm(self, guess):
        """
        Run leastsq and set the result
        """

        guess=array(guess,dtype='f8',copy=False)
        self._setup_data(guess)

        result = run_leastsq(self._calc_fdiff, guess, self.n_prior_pars, **self.lm_pars)

        result['model'] = self.model_name
        if result['flags']==0:
            stat_dict=self.get_fit_stats(result['pars'])
            result.update(stat_dict)

        self._result=result

    def get_band_pars(self, pars_in, band):
        """
        pars in are [g1,g2]
        """

        pars=self.pars_in
        pars[2]=pars_in[0]
        pars[3]=pars_in[1]
        return pars


def _get_simple_band_pars_round_logpars(pars_in, pars_allband, band_pars, band):
    """
    pars in are [cen1,cen2,log(T),log(F)]
    """
    # all band
    pars_allband[0:2] = pars_in[0:2]
    # 2:2+2 remain zero for roundness
    pars_allband[4:] = pars_in[2:]

    _gmix.convert_simple_double_logpars_band(pars_allband, band_pars, band)

def _get_simple_band_pars_round_linpars(pars_in, band_pars, band):
    """
    pars in are [cen1,cen2,T,F]
    """
    band_pars[0:2] = pars_in[0:2]
    # 2:2+2 remain zero for roundness
    band_pars[4] = pars_in[2]
    band_pars[5] = pars_in[3+band]

def _get_simple_band_pars_fixT_linpars(T, pars_in, band_pars, band):
    """
    pars in are [cen1,cen2,T,F]
    """
    band_pars[0:4] = pars_in[0:4]
    band_pars[4]=T
    band_pars[5] = pars_in[3+band]

def _get_simple_band_pars_fixT_logpars(T, pars_in, band_pars, band):
    """
    pars in are [cen1,cen2,T,F]
    """
    band_pars[0:4] = pars_in[0:4]
    band_pars[4]=T
    band_pars[5] = pars_in[3+band]

    _gmix.convert_simple_double_logpars_band(pars_allband, band_pars, band)


class LMComposite(LMSimple):
    """
    exp+dev model with pre-determined fracdev and ratio Tdev/Texp
    """
    def __init__(self, obs, fracdev, TdByTe, **keys):
        super(LMComposite,self).__init__(obs, 'cm', **keys)

        self.fracdev=fracdev
        self.TdByTe=TdByTe

    def get_gmix(self, band=0):
        """
        Get a gaussian mixture at the "best" parameter set, which
        definition depends on the sub-class
        """
        res=self.get_result()
        pars=self.get_band_pars(res['pars'], band)
        return gmix.GMixCM(self.fracdev,
                                  self.TdByTe,
                                  pars)


    def _make_model(self, band_pars):
        gm0=gmix.GMixCM(self.fracdev, self.TdByTe, band_pars)
        return gm0

    def _fill_gmix(self, gm, band_pars):
        _gmix.gmix_fill_cm(gm._data, band_pars)

    def _convolve_gmix(self, gm, gm0, psf_gmix):
        _gmix.convolve_fill(gm._data,
                            gm0._data['gmix'][0],
                            psf_gmix._data)

# most of this is the same as LMSimple
class LMCompositeRound(LMComposite):
    """
    This version fits [cen1,cen2,T,F]
    """
    def __init__(self, *args, **keys):
        super(LMCompositeRound,self).__init__(*args, **keys)

        self.npars = self.npars-2
        self.n_prior_pars=self.n_prior_pars - 1
        self.fdiff_size=self.totpix + self.n_prior_pars

        if self.use_logpars:
            self._pars_allbands=zeros(self.npars+2)


    def run_lm(self, guess):
        """
        Run leastsq and set the result
        """

        guess=array(guess,dtype='f8',copy=False)
        self._setup_data(guess)

        result = run_leastsq(self._calc_fdiff, guess, self.n_prior_pars, **self.lm_pars)

        result['model'] = self.model_name
        if result['flags']==0:
            stat_dict=self.get_fit_stats(result['pars'])
            result.update(stat_dict)

        self._result=result

    def get_band_pars(self, pars_in, band):
        """
        Get linear pars for the specified band
        """

        pars=self._band_pars
        if self.use_logpars:
            _get_simple_band_pars_round_logpars(pars_in,
                                                self._pars_allbands,
                                                pars,
                                                band)
        else:
            _get_simple_band_pars_round_linpars(pars_in,
                                                pars,
                                                band)

        return pars


class LMSersic(LMSimple):
    def __init__(self, image, weight, jacobian, guess, **keys):
        super(LMSimple,self).__init__(image, weight, jacobian, "sersic", **keys)
        # this is a dict
        # can contain maxfev (maxiter), ftol (tol in sum of squares)
        # xtol (tol in solution), etc
        self.lm_pars=keys['lm_pars']

        self.guess=array( guess, dtype='f8' )

        self.n_prior=keys['n_prior']

        n_prior_pars=7
        self.fdiff_size=self.totpix + n_prior_pars

    def get_band_pars(self, pars, band):
        raise RuntimeError("adapt to new style")
        if band > 0:
            raise ValueError("support more than one band")
        return pars.copy()


NOTFINITE_BIT=11
def run_leastsq(func, guess, n_prior_pars, **keys):
    """
    run leastsq from scipy.optimize.  Deal with certain
    types of errors

    TODO make this do all the checking and fill in cov etc.  return
    a dict

    parameters
    ----------
    func:
        the function to minimize
    guess:
        guess at pars
    n_prior_pars:
        number of slots in fdiff for priors

    some useful keywords
    maxfev:
        maximum number of function evaluations. e.g. 1000
    epsfcn:
        Step for jacobian estimation (derivatives). 1.0e-6
    ftol:
        Relative error desired in sum of squares, 1.0e06
    xtol:
        Relative error desired in solution. 1.0e-6
    """
    from scipy.optimize import leastsq

    npars=guess.size

    res={}
    try:
        lm_tup = leastsq(func, guess, full_output=1, **keys)

        pars, pcov0, infodict, errmsg, ier = lm_tup

        if ier == 0:
            # wrong args, this is a bug
            raise ValueError(errmsg)

        flags = 0
        if ier > 4:
            flags = 2**(ier-5)
            pars,pcov,perr=_get_def_stuff(npars)
            print('    ',errmsg)

        elif pcov0 is None:    
            # why on earth is this not in the flags?
            flags += LM_SINGULAR_MATRIX 
            errmsg = "singular covariance"
            print('    ',errmsg)
            print_pars(pars,front='    pars at singular:')
            junk,pcov,perr=_get_def_stuff(npars)
        else:
            # Scale the covariance matrix returned from leastsq; this will
            # recover the covariance of the parameters in the right units.
            fdiff=func(pars)

            # npars: to remove priors

            dof = fdiff.size - n_prior_pars - npars

            s_sq = (fdiff[n_prior_pars:]**2).sum()/dof
            pcov = pcov0 * s_sq 

            cflags = _test_cov(pcov)
            if cflags != 0:
                flags += cflags
                errmsg = "bad covariance matrix"
                print('    ',errmsg)
                junk1,junk2,perr=_get_def_stuff(npars)
            else:
                # only if we reach here did everything go well
                perr=sqrt( numpy.diag(pcov) )

        res['flags']=flags
        res['nfev'] = infodict['nfev']
        res['ier'] = ier
        res['errmsg'] = errmsg

        res['pars'] = pars
        res['pars_err']=perr
        res['pars_cov0'] = pcov0
        res['pars_cov']=pcov

    except ValueError as e:
        serr=str(e)
        if 'NaNs' in serr or 'infs' in serr:
            pars,pcov,perr=_get_def_stuff(npars)

            res['pars']=pars
            res['pars_cov0']=pcov
            res['pars_cov']=pcov
            res['nfev']=-1
            res['flags']=LM_FUNC_NOTFINITE
            res['errmsg']="not finite"
            print('    not finite')
        else:
            raise e

    except ZeroDivisionError:
        pars,pcov,perr=_get_def_stuff(npars)

        res['pars']=pars
        res['pars_cov0']=pcov
        res['pars_cov']=pcov
        res['nfev']=-1

        res['flags']=LM_DIV_ZERO
        res['errmsg']="zero division"
        print('    zero division')

    return res

def _get_def_stuff(npars):
    pars=zeros(npars) + PDEF
    cov=zeros( (npars,npars) ) + CDEF
    err=zeros(npars) + CDEF
    return pars,cov,err

def _test_cov(pcov):
    flags=0
    try:
        e,v = numpy.linalg.eig(pcov)
        weig,=numpy.where(e < 0)
        if weig.size > 0:
            flags += LM_NEG_COV_EIG 

        wneg,=numpy.where(numpy.diag(pcov) < 0)
        if wneg.size > 0:
            flags += LM_NEG_COV_DIAG 

    except numpy.linalg.linalg.LinAlgError:
        flags |= EIG_NOTFINITE 

    return flags

class MCMCBase(FitterBase):
    """
    A base class for MCMC runs using emcee.
    
    Extra user-facing methods are run_mcmc(), calc_result(), get_trials(), get_sampler(), make_plots()
    """
    def __init__(self, obs, model, **keys):
        super(MCMCBase,self).__init__(obs, model, **keys)

        # this should be a numpy.random.RandomState object, unlike emcee which
        # through the random_state parameter takes the tuple state
        self.random_state = keys.get('random_state',None)

        # emcee specific
        self.nwalkers=keys['nwalkers']
        self.mca_a=keys.get('mca_a',2.0)

    def get_trials(self):
        """
        Get the set of trials
        """

        if not hasattr(self,'_trials'):
            raise RuntimeError("you need to run the mcmc chain first")

        return self._trials

    def get_lnprobs(self):
        """
        Get the set of ln(prob) values
        """

        if not hasattr(self,'_lnprobs'):
            raise RuntimeError("you need to run the mcmc chain first")

        return self._lnprobs

    def get_best_pars(self):
        """
        get the parameters with the highest probability
        """
        if not hasattr(self,'_lnprobs'):
            raise RuntimeError("you need to run the mcmc chain first")

        return self._best_pars.copy()

    def get_best_lnprob(self):
        """
        get the highest probability
        """
        if not hasattr(self,'_lnprobs'):
            raise RuntimeError("you need to run the mcmc chain first")

        return self._best_lnprob


    def get_sampler(self):
        """
        get the emcee sampler
        """
        return self.sampler

    def get_arate(self):
        """
        get the acceptance rate
        """
        return self._arate

    def get_tau(self):
        """
        2*tau/nstep
        """
        return self._tau

    def run_mcmc(self, pos0, nstep, thin=1, **kw):
        """
        run steps, starting at the input position(s)

        input and output pos are in linear space

        keywords to run_mcmc/sample are passed along, such as thin
        """

        pos0=array(pos0, dtype='f8')

        if not hasattr(self,'sampler'):
            self._setup_sampler_and_data(pos0)

        sampler=self.sampler
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(pos0, nstep, thin=thin, **kw)

        trials  = sampler.flatchain
        lnprobs = sampler.lnprobability.reshape(self.nwalkers*nstep/thin)

        self._trials=trials
        self._lnprobs=lnprobs

        w=lnprobs.argmax()
        bp=lnprobs[w]
        if self._best_lnprob is None or bp > self._best_lnprob:
            self._best_lnprob=bp
            self._best_pars=trials[w,:]

        arates = sampler.acceptance_fraction
        self._arate = arates.mean()
        self._set_tau()

        self._last_pos=pos
        return pos
    go=run_mcmc

    def get_last_pos(self):
        return self._last_pos

    def get_weights(self):
        """
        default weights are none
        """
        return None

    def get_stats(self, sigma_clip=False, weights=None, **kw):
        """
        get mean and covariance.

        parameters
        ----------
        weights: array
            Extra weights to apply.
        """
        this_weights = self.get_weights()

        if this_weights is not None and weights is not None:
            weights = this_weights * weights
        elif this_weights is not None:
            weights=this_weights
        else:
            # input weights are used, None or no
            pass
        
        trials=self.get_trials()

        pars,pars_cov = stats.calc_mcmc_stats(trials, sigma_clip=sigma_clip, weights=weights, **kw)

        return pars, pars_cov

    def calc_result(self, sigma_clip=False, weights=None, **kw):
        """
        Calculate the mcmc stats and the "best fit" stats
        """

        pars,pars_cov = self.get_stats(sigma_clip=sigma_clip, weights=weights, **kw)
        pars_err=sqrt(diag(pars_cov))
        res={'model':self.model_name,
             'flags':self.flags,
             'pars':pars,
             'pars_cov':pars_cov,
             'pars_err':pars_err,
             'tau':self._tau,
             'arate':self._arate}

        # note get_fits_stats expects pars in log space
        fit_stats = self.get_fit_stats(pars)
        res.update(fit_stats)

        self._result=res
        

    def _setup_sampler_and_data(self, pos):
        """
        try very hard to initialize the mixtures

        we work in T,F as log(1+x) so watch for low values
        """

        self.flags=0
        self._tau=0.0

        npars=pos.shape[1]
        mess="pos has npars=%d, expected %d" % (npars,self.npars)
        assert (npars==self.npars),mess

        self.sampler = self._make_sampler()
        self._best_lnprob=None

        ok=False
        for i in xrange(self.nwalkers):
            try:
                self._init_gmix_all(pos[i,:])
                ok=True
                break
            except GMixRangeError as gerror:
                continue
            except ZeroDivisionError:
                continue

        if not ok:
            print('failed init gmix from input guess: %s' % str(gerror))
            raise gerror

    def _set_tau(self):
        """
        auto-correlation for emcee
        """
        import emcee

        trials=self.get_trials()

        # actually 2*tau
        tau2 = emcee.autocorr.integrated_time(trials,window=100)
        tau2 = tau2.max()
        self._tau=tau2

        """
        if hasattr(emcee.ensemble,'acor'):
            if emcee.ensemble.acor is not None:
                acor=self.sampler.acor
                tau = acor.max()
        elif hasattr(emcee.ensemble,'autocorr'):
            if emcee.ensemble.autocorr is not None:
                acor=self.sampler.acor
                tau = acor.max()
        self._tau=tau
        """

    def _make_sampler(self):
        """
        Instantiate the sampler
        """
        import emcee
        sampler = emcee.EnsembleSampler(self.nwalkers, 
                                        self.npars, 
                                        self.calc_lnprob,
                                        a=self.mca_a)

        if self.random_state is not None:

            # this is a property, runs set_state internally. sadly this will
            # fail silently which is the stupidest thing I have ever seen in my
            # entire life.  If I want to set the state it is important to me!
            
            #print('            replacing random state')
            #sampler.random_state=self.random_state.get_state()

            # OK, we will just hope that _random doesn't change names in the future.
            # but at least we get control back
            sampler._random = self.random_state

        return sampler


    def make_plots(self,
                   show=False,
                   prompt=False,
                   do_residual=False,
                   do_triangle=False,
                   width=1200,
                   height=1200,
                   separate=False,
                   title=None,
                   weights=None,
                   **keys):
        """
        Plot the mcmc chain and some residual plots
        """
        import mcmc
        import biggles

        biggles.configure('screen','width', width)
        biggles.configure('screen','height', height)

        names=self.get_par_names()

        if separate:
            # returns a tuple burn_plt, hist_plt
            plotfunc =mcmc.plot_results_separate
        else:
            plotfunc =mcmc.plot_results

        trials=self.get_trials()
        pdict={}
        pdict['trials']=plotfunc(trials,
                                 names=names,
                                 title=title,
                                 show=show,
                                 **keys)


        if weights is not None:
            pdict['wtrials']=plotfunc(trials,
                                      weights=weights,
                                      names=names,
                                      title='%s weighted' % title,
                                      show=show)

        if do_residual:
            pdict['resid']=self.plot_residuals(title=title,show=show,
                                               width=width,
                                               height=height,
                                               **keys)

        if do_triangle:
            try:
                # we will crash on a batch job if we don't do this.
                # also if pyplot has already been imported, it will
                # crash (god I hate matplotlib)
                import matplotlib as mpl
                mpl.use('Agg')
                import triangle
                figure = triangle.corner(trials, 
                                         labels=names,
                                         quantiles=[0.16, 0.5, 0.84],
                                         show_titles=True,
                                         title_args={"fontsize": 12},
                                         bins=25)
                pdict['triangle'] = figure
            except:
                print("could not do triangle")

        if show and prompt:
            key=raw_input('hit a key: ')
            if key=='q':
                stop

        return pdict


    def get_par_names(self):
        raise RuntimeError("over-ride me")


class MCMCSimple(MCMCBase):
    """
    Add additional features to the base class to support simple models
    """
    def __init__(self, obs, model,  **keys):
        super(MCMCSimple,self).__init__(obs, model, **keys)

        # where g1,g2 are located in a pars array
        self.g1i = 2
        self.g2i = 3

        self._band_pars=zeros(6)

    def calc_result(self, **kw):
        """
        Some extra stats for simple models
        """

        super(MCMCSimple,self).calc_result(**kw)

        g1i=self.g1i
        g2i=self.g2i

        self._result['g'] = self._result['pars'][g1i:g1i+2].copy()
        self._result['g_cov'] = self._result['pars_cov'][g1i:g1i+2, g1i:g1i+2].copy()

    def get_band_pars(self, pars_in, band):
        """
        Get linear pars for the specified band
        """

        pars=self._band_pars

        if self.use_logpars:
            _gmix.convert_simple_double_logpars_band(pars_in, pars, band)
        else:
            pars[0:5] = pars_in[0:5]
            pars[5] = pars_in[5+band]

        return pars


    def get_par_names(self, dolog=False):
        names=['cen1','cen2', 'g1','g2', 'T']
        if self.nband == 1:
            names += ['F']
        else:
            for band in xrange(self.nband):
                names += ['F_%s' % band]

        return names

class MCMCGaussMom(MCMCSimple):
    """
    Fit gaussian in moment space, no psf
    """
    def __init__(self, obs, **keys):

        super(MCMCGaussMom,self).__init__(obs, 'gaussmom', **keys)

    def calc_result(self, **kw):
        """
        Some extra stats for simple models
        """
        super(MCMCSimple,self).calc_result(**kw)

    def get_band_pars(self, pars_in, band):
        """
        Get linear pars for the specified band

        pars are [c1,c2,M1,M2,T,I1,I2...]

        Where M1 = Icc-Irr
              m2 = 2*Irc
        """

        pars=self._band_pars

        pars[0:4] = pars_in[0:4]
        pars[5] = pars_in[5+band]

        return pars

    def get_par_names(self, **kw):
        """
        parameter names for each dimension
        """

        names=['cen1','cen2', 'M1','M2','T']

        if self.nband == 1:
            names += ['F']
        else:
            for band in xrange(self.nband):
                names += ['F_%s' % band]

        return names

class MCMCGaussMomSum(MCMCSimple):
    """
    Fit gaussian in moment space, no psf
    """
    def __init__(self, obs, **keys):

        super(MCMCGaussMomSum,self).__init__(obs, 'gauss', **keys)

        self.model=gmix.GMIX_FULL
        self.model_name='full'

    def calc_result(self, **kw):
        """
        Some extra stats for simple models
        """
        super(MCMCSimple,self).calc_result(**kw)

    def get_band_pars(self, pars_in, band):
        """

        pars are [c1,c2,M1sum,M2sum,Tsum,Isum,...]
        """

        #c1    = pars_in[0]
        #c2    = pars_in[1]
        c1sum    = pars_in[0]
        c2sum    = pars_in[1]
        M1sum = pars_in[2]
        M2sum = pars_in[3]
        Tsum  = pars_in[4]
        Isum  = pars_in[5+band]

        c1=c1sum/Isum
        c2=c2sum/Isum
        M1 = M1sum/Isum
        M2 = M2sum/Isum
        T  = Tsum/Isum

        Irr = (T-M1)*0.5
        Irc = M2/2
        Icc = (T+M1)*0.5

        pars=self._band_pars

        pars[0] = Isum
        pars[1] = c1
        pars[2] = c2
        pars[3] = Irr
        pars[4] = Irc
        pars[5] = Icc

        return pars

    def get_par_names(self, dolog=False):
        """
        parameter names for each dimension
        """
        names=[]

        names=['cen1','cen2', 'M1sum','M2sum','Tsum']

        if self.nband == 1:
            names += ['Isum']
        else:
            for band in xrange(self.nband):
                names += ['Isum_%s' % band]

        return names



class MCMCSimpleEta(MCMCSimple):
    """
    search eta space
    """

    def get_band_pars(self, pars_in, band):
        """
        Get linear pars for the specified band
        """

        pars=self._band_pars

        status=_gmix.convert_simple_eta2g_band(pars_in, pars, band)
        if status != 1:
            raise GMixRangeError("shape out of bounds")
        #print("eta:",pars_in[2],pars_in[3])
        #print("g:  ",pars[2], pars[3])
        return pars


    def get_par_names(self, dolog=False):
        names=['cen1','cen2', 'eta1','eta2', 'T']
        if self.nband == 1:
            names += ['F']
        else:
            for band in xrange(self.nband):
                names += ['F_%s' % band]

        return names


class MH(object):
    """
    Run a Monte Carlo Markov Chain (MCMC) using metropolis hastings.
    
    parameters
    ----------
    lnprob_func: function or method
        A function to calculate the log proability given the input
        parameters.  Can be a method of a class.
            ln_prob = lnprob_func(pars)
            
    stepper: function or method 
        A function to take a step given the input parameters.
        Can be a method of a class.
            newpars = stepper(pars)

    seed: floating point, optional
        An optional seed for the random number generator.
    random_state: optional
        A random number generator with method .uniform()
        e.g. numpy.random.RandomState.  Takes precedence over
        seed

    examples
    ---------
    m=mcmc.MH(lnprob_func, stepper, seed=34231)
    m.run(pars_start, nstep)

    means, cov = m.get_stats()

    trials = m.get_trials()
    loglike = m.get_loglike()
    arate = m.get_acceptance_rate()

    """
    def __init__(self, lnprob_func, stepper,
                 seed=None, random_state=None):
        self._lnprob_func=lnprob_func
        self._stepper=stepper

        self.set_random_state(seed=seed, state=random_state)

    def get_trials(self):
        """
        Get the trials array
        """
        return self._trials

    def get_loglike(self):
        """
        Get the log like array
        """
        return self._loglike
    get_lnprob=get_loglike

    def get_acceptance_rate(self):
        """
        Get the acceptance rate
        """
        return self._arate
    get_arate=get_acceptance_rate

    def get_accepted(self):
        """
        Get the accepted array
        """
        return self._accepted


    def get_stats(self, sigma_clip=False, weights=None, **kw):
        """
        get mean and covariance.

        parameters
        ----------
        weights: array
            Extra weights to apply.
        """
        from .stats import calc_mcmc_stats
        stats = calc_mcmc_stats(self._trials, sigma_clip=sigma_clip, weights=weights, **kw)
        return stats

    def set_random_state(self, seed=None, state=None):
        """
        set the random state

        parameters
        ----------
        seed: integer, optional
            If state= is not set, the random state is set to
            numpy.random.RandomState(seed=seed)
        state: optional
            A random number generator with method .uniform()
            e.g. numpy.random.RandomState.  Takes precedence over
            seed
        """
        if state is not None:
            self._random_state=state
        else:
            self._random_state=numpy.random.RandomState(seed=seed)

    def run_mcmc(self, pars_start, nstep):
        """
        Run the MCMC chain.  Append new steps if trials already
        exist in the chain.

        parameters
        ----------
        pars_start: sequence
            Starting point for the chain in the n-d parameter space.
        nstep: integer
            Number of steps in the chain.
        """
        
        self._init_data(pars_start, nstep)

        for i in xrange(1,nstep):
            self._step()

        self._arate=self._accepted.sum()/float(self._accepted.size)
        return self._trials[-1,:]

    def _step(self):
        """
        Take the next step in the MCMC chain.  
        
        Calls the stepper lnprob_func methods sent during construction.  If the
        new loglike is not greater than the previous, or a uniformly generated
        random number is greater than the the ratio of new to old likelihoods,
        the new step is not used, and the new parameters are the same as the
        old.  Otherwise the new step is kept.

        This is an internal function that is called by the .run method.
        It is not intended for call by the user.
        """

        index=self._current

        oldpars=self._oldpars
        oldlike=self._oldlike

        # Take a step and evaluate the likelihood
        newpars = self._stepper(oldpars)
        newlike = self._lnprob_func(newpars)

        log_likeratio = newlike-oldlike

        randnum = self._random_state.uniform()
        log_randnum = numpy.log(randnum)

        # we allow use of -infinity as a sign we are out of bounds
        if (isfinite(newlike) 
                and ( (newlike > oldlike) | (log_randnum < log_likeratio)) ):

            self._accepted[index]  = 1
            self._loglike[index]   = newlike
            self._trials[index, :] = newpars

            self._oldpars = newpars
            self._oldlike = newlike

        else:
            self._accepted[index] = 0
            self._loglike[index]  = oldlike
            self._trials[index,:] = oldpars

        self._current += 1

    def _init_data(self, pars_start, nstep):
        """
        Set the trials and accept array.
        """

        pars_start=array(pars_start,dtype='f8',copy=False)
        npars = pars_start.size

        self._trials   = numpy.zeros( (nstep, npars) )
        self._loglike  = numpy.zeros(nstep)
        self._accepted = numpy.zeros(nstep, dtype='i1')
        self._current  = 1

        self._oldpars = pars_start.copy()
        self._oldlike = self._lnprob_func(pars_start)

        self._trials[0,:] = pars_start
        self._loglike[0]  = self._oldlike
        self._accepted[0] = 1

class MHTemp(MH):
    """
    Run a Monte Carlo Markov Chain (MCMC) using metropolis hastings
    with the specified temperature.
    
    parameters
    ----------
    lnprob_func: function or method
        A function to calculate the log proability given the input
        parameters.  Can be a method of a class.
            ln_prob = lnprob_func(pars)
    stepper: function or method 
        A function to take a step given the input parameters.
        Can be a method of a class.
            newpars = stepper(pars)
    T: float
        Temperature.

    seed: floating point, optional
        An optional seed for the random number generator.
    state: optional
        A random number generator with method .uniform()
        e.g. numpy.random.RandomState.  Takes precedence over
        seed

    examples
    ---------
    T=1.5
    m=mcmc.MHTemp(lnprob_func, stepper, T, seed=34231)
    m.run(pars_start, nstep)
    trials = m.get_trials()

    means,cov = m.get_stats()

    # the above uses the weights, so is equivalent to
    # the following

    weights = m.get_weights()

    wsum=weights.sum()
    mean0 = (weights*trials[:,0]).sum()/wsum

    fdiff0 = trials[:,0]-mean0
    var00 = (weights*fdiff0*fdiff0).sum()/wsum

    fdiff1 = trials[:,1]-mean1

    var01 = (weights*fdiff0*fdiff1).sum()/wsum

    etc. for the other parameters and covariances
    """

    def __init__(self, lnprob_func, stepper, T,
                 seed=None, random_state=None):

        super(MHTemp,self).__init__(lnprob_func, stepper,
                                    seed=seed,
                                    random_state=random_state)
        self.T=T
        self.Tinv=1.0/self.T

    def get_stats(self, weights=None):
        """
        get mean and covariance.

        parameters
        ----------
        weights: array
            Extra weights to apply.
        """
        this_weights = self.get_weights()

        if weights is not None:
            weights = this_weights * weights
        else:
            weights = this_weights
        
        return super(MHTemp,self).get_stats(weights=weights)

    def get_loglike_T(self):
        """
        Get the log like array ln(like)/T
        """
        return self._loglike_T

    def get_weights(self):
        """
        get weights that put the loglike back at temp=1
        """
        if not hasattr(self,'_weights'):
            self._max_loglike = self._loglike.max()
            logdiff = self._loglike-self._max_loglike
            self._weights = exp(logdiff*(1.0 - self.Tinv))
        return self._weights

    def _step(self):
        """
        Take the next step in the MCMC chain.  
        
        Calls the stepper lnprob_func methods sent during construction.  If the
        new loglike is not greater than the previous, or a uniformly generated
        random number is greater than the the ratio of new to old likelihoods,
        the new step is not used, and the new parameters are the same as the
        old.  Otherwise the new step is kept.

        This is an internal function that is called by the .run method.
        It is not intended for call by the user.
        """

        index=self._current

        oldpars=self._oldpars
        oldlike=self._oldlike
        oldlike_T=self._oldlike_T

        # Take a step and evaluate the likelihood
        newpars = self._stepper(oldpars)
        newlike = self._lnprob_func(newpars)
        newlike_T = newlike*self.Tinv

        log_likeratio = newlike_T-oldlike_T

        randnum = self._random_state.uniform()
        log_randnum = numpy.log(randnum)

        # we allow use of -infinity as a sign we are out of bounds
        if (isfinite(newlike_T) 
                and ( (newlike_T > oldlike_T) | (log_randnum < log_likeratio)) ):

            self._accepted[index]  = 1
            self._loglike[index]   = newlike
            self._loglike_T[index]   = newlike_T
            self._trials[index, :] = newpars

            self._oldpars = newpars
            self._oldlike = newlike
            self._oldlike_T = newlike_T

        else:
            self._accepted[index] = 0
            self._loglike[index]  = oldlike
            self._loglike_T[index]  = oldlike_T
            self._trials[index,:] = oldpars

        self._current += 1

    def _init_data(self, pars_start, nstep):
        """
        Set the trials and accept array.
        """
        super(MHTemp,self)._init_data(pars_start, nstep)

        T=self.T
        oldlike_T = self._oldlike*self.Tinv

        loglike_T = self._loglike.copy()
        loglike_T[0] = oldlike_T

        self._oldlike_T=oldlike_T
        self._loglike_T = loglike_T

   
class MHSimple(MCMCSimple):
    def __init__(self, obs, model, step_sizes, **keys):
        """
        not inheriting init from MCMCSsimple or MCMCbase

        step sizes in linear space
        """
        FitterBase.__init__(self, obs, model, **keys)

        # where g1,g2 are located in a pars array
        self.g1i = 2
        self.g2i = 3

        self._band_pars=zeros(6)

        self.set_step_sizes(step_sizes)

        seed=keys.get('seed',None)
        state=keys.get('random_state',None)
        self.set_random_state(seed=seed, state=state)

    def set_step_sizes(self, step_sizes):
        """
        set the step sizes to the input
        """
        step_sizes=numpy.asanyarray(step_sizes, dtype='f8')
        sdim = step_sizes.shape
        if len(sdim) == 1:
            ns=step_sizes.size
            mess="step_sizes has size=%d, expected %d" % (ns,self.npars)
            assert (ns == self.npars),mess

            mess="step sizes must all be > 0"
            assert numpy.all(step_sizes > 0),mess

        elif len(sdim) == 2:
            mess="step_sizes needs to be a square matrix, has dims %dx%d." % sdim
            assert (sdim[0] == sdim[1]),mess
            ns=sdim[0]
            mess="step_sizes has size=%d, expected %d" % (ns,self.npars)
            assert (ns == self.npars),mess
            assert numpy.all(numpy.linalg.eigvals(step_sizes) > 0),"step_sizes must be positive definite."
        else:
            assert len(sdim) <= 2, "step_sizes cannot have dimension greater than 2, has %d dims." % len(sdim)
        self._step_sizes=step_sizes
        self._ndim_step_sizes = len(sdim)
        
    def set_random_state(self, seed=None, state=None):
        """
        set the random state

        parameters
        ----------
        state: optional
            A random number generator with method .uniform()
            e.g. an instance of numpy.random.RandomState
        seed: integer, optional
            If state= is not set, the random state is set to
            numpy.random.RandomState(seed=seed)
        """
        if state is not None:
            self.random_state=state
        else:
            self.random_state=numpy.random.RandomState(seed=seed)

    def run_mcmc(self, pos0, nstep):
        """
        run steps, starting at the input position
        """

        pos0=array(pos0,dtype='f8',copy=False)

        if not hasattr(self,'sampler'):
            self._setup_sampler_and_data(pos0)

        sampler=self.sampler

        pos = sampler.run_mcmc(pos0, nstep)

        trials = sampler.get_trials()
        lnprobs = sampler.get_lnprob()

        self._trials=trials
        self._lnprobs=lnprobs

        w=lnprobs.argmax()
        bp=lnprobs[w]
        if self._best_lnprob is None or bp > self._best_lnprob:
            self._best_lnprob=bp
            self._best_pars=trials[w,:]

        self._arate = sampler.get_arate()
        self._set_tau()

        self._last_pos=pos
        return pos

    def take_step(self, pos):
        """
        Take gaussian steps
        """
        if self._ndim_step_sizes == 1:
            return pos+self._step_sizes*self.random_state.normal(size=self.npars)
        else:
            return numpy.random.multivariate_normal(pos, self._step_sizes)

    def _setup_sampler_and_data(self, pos):
        """
        pos in linear space

        Try to initialize the gaussian mixtures. If failure, most
        probablly a GMixRangeError will be raised
        """

        self.flags=0

        npars=pos.size
        mess="pos has npars=%d, expected %d" % (npars,self.npars)
        assert (npars==self.npars),mess

        # initialize all the gmix objects; may raise an error
        self._init_gmix_all(pos)

        self.sampler = MH(self.calc_lnprob, self.take_step,
                          random_state=self.random_state)
        self._best_lnprob=None


    def _set_tau(self):
        """
        auto-correlation scale lenght*2 divided by the number of steps
        """
        import emcee

        trials=self.get_trials()

        # actually 2*tau
        tau2 = emcee.autocorr.integrated_time(trials,window=100)
        tau2 = tau2.max()
        self._tau=tau2


class MHTempSimple(MHSimple):
    """
    Run with a temperature != 1.  Use the weights when
    getting stats
    """
    def __init__(self, obs, model, step_sizes, **keys):
        super(MHTempSimple,self).__init__(obs, model, step_sizes, **keys)
        self.temp=keys.get('temp',1.0)
        print("MHTempSimple doing temperature:",self.temp)
 
    def get_weights(self):
        """
        Get the temperature weights
        """
        return self.sampler.get_weights()

    def _setup_sampler_and_data(self, pos):
        """
        Try to initialize the gaussian mixtures. If failure, most
        probablly a GMixRangeError will be raised
        """

        self.flags=0
        self.pos=pos

        npars=pos.size
        mess="pos has npars=%d, expected %d" % (npars,self.npars)
        assert (npars==self.npars),mess

        # initialize all the gmix objects; may raise an error
        self._init_gmix_all(pos)

        self.sampler = MHTemp(self.calc_lnprob, self.take_step, self.temp,
                              random_state=self.random_state)
        self._best_lnprob=None


class MCMCSersic(MCMCSimple):
    def __init__(self, obs, **keys):

        raise RuntimeError("adapt to new system")
        self.g1i=2
        self.g2i=3

        MCMCBase.__init__(self, obs, "sersic", **keys)


    def _setup_sampler_and_data(self, pos):
        """
        try very hard to initialize the mixtures
        """

        self.flags=0
        self._tau=0.0
        self.pos=pos
        self.npars=pos.shape[1]

        self.sampler = self._make_sampler()
        self._best_lnprob=None

        ok=False
        for i in xrange(self.nwalkers):
            try:
                self._init_gmix_all(self.pos[i,:])
                ok=True
                break
            except GMixRangeError as gerror:
                continue
            except ZeroDivisionError:
                continue

        if ok:
            return

        print('failed init gmix lol from input guess:',str(gerror))
        print('getting a new guess')
        for j in xrange(10):
            self.pos=self._get_random_guess()
            ok=False
            for i in xrange(self.nwalkers):
                try:
                    self._init_gmix_all(self.pos[i,:])
                    ok=True
                    break
                except GMixRangeError as gerror:
                    continue
                except ZeroDivisionError:
                    continue
            if ok:
                break

        if not ok:
            raise gerror

    def run_mcmc(self, pos, nstep):
        """
        user can run steps
        """

        if not hasattr(self,'sampler'):
            self._setup_sampler_and_data(pos)

        sampler=self.sampler
        sampler.reset()
        self.pos, prob, state = sampler.run_mcmc(self.pos, nstep)

        lnprobs = sampler.lnprobability.reshape(self.nwalkers*nstep)
        w=lnprobs.argmax()
        bp=lnprobs[w]
        if self._best_lnprob is None or bp > self._best_lnprob:
            self._best_lnprob=bp
            self._best_pars=sampler.flatchain[w,:]

        arates = sampler.acceptance_fraction
        self._arate = arates.mean()

        self._trials=trials

        return self.pos

    def _get_priors(self, pars):
        """
        # go in simple
        add any priors that were sent on construction
        """

        lnp=0.0

        if self.cen_prior is not None:
            lnp += self.cen_prior.get_lnprob(pars[0], pars[1])

        if self.g_prior is not None:
            if self.g_prior_during:
                lnp += self.g_prior.get_lnprob_scalar2d(pars[2],pars[3])
            else:
                # may have bounds
                g = sqrt(pars[2]**2 + pars[3]**2)
                if g > self.g_prior.gmax:
                    raise GMixRangeError("g too big")
        else:
            g = sqrt(pars[2]**2 + pars[3]**2)
            if g >= 0.99999:
                raise GMixRangeError("g too big")

        if self.T_prior is not None:
            lnp += self.T_prior.get_lnprob_scalar(pars[4])

        if self.counts_prior is not None:
            for i,cp in enumerate(self.counts_prior):
                counts=pars[5+i]
                lnp += cp.get_lnprob_scalar(counts)

        lnp += self.n_prior.get_lnprob_scalar(pars[6])

        return lnp

    def get_par_names(self):
        names=['cen1','cen2', 'g1','g2','T','F','n']
        return names

    def _set_npars(self):
        """
        this is actually set elsewhere
        """
        pass

    def get_band_pars(self, pars, band):
        if band > 0:
            raise ValueError("support multi-band for sersic")
        return pars.copy()

class MCMCSersicJointHybrid(MCMCSersic):
    def __init__(self, image, weight, jacobian, **keys):
        raise RuntimeError("adapt to new system")

        self.g1i=2
        self.g2i=3

        self.joint_prior=keys.get('joint_prior',None)

        if (self.joint_prior is None):
            raise ValueError("send joint_prior for sersic joint")

        self.prior_during=keys.get('prior_during',False)

        MCMCBase.__init__(self, image, weight, jacobian, "sersic", **keys)


    def _get_priors(self, pars):
        """
        Apply simple priors
        """
        lnp=0.0
        
        if self.cen_prior is not None:
            lnp += self.cen_prior.get_lnprob(pars[0], pars[1])

        jp=self.joint_prior

        # this is just the structural parameters
        lnp += jp.get_lnprob_scalar(pars[4:])

        if self.prior_during:
            lnp += jp.g_prior.get_lnprob_scalar2d(pars[2],pars[3])

        return lnp

    def get_band_pars(self, pars, band):
        """
        Extract pars for the specified band and convert to linear
        """
        raise RuntimeError("adapt to new style")
        if band != 0:
            raise ValueError("deal with more than one band")
        linpars=pars.copy()

        linpars[4] = 10.0**linpars[4]
        linpars[5] = 10.0**linpars[5]
        linpars[6] = 10.0**linpars[6]

        return linpars


    def _get_priors(self, pars):
        """
        Apply simple priors
        """
        lnp=0.0
        
        if self.cen_prior is not None:
            lnp += self.cen_prior.get_lnprob(pars[0], pars[1])

        jp=self.joint_prior

        # this is just the structural parameters
        lnp += jp.get_lnprob_scalar(pars[4:])

        if self.prior_during:
            lnp += jp.g_prior.get_lnprob_scalar2d(pars[2],pars[3])

        return lnp

    def _get_PQR(self):
        """
        get the marginalized P,Q,R from Bernstein & Armstrong
        """

        g_prior=self.joint_prior.g_prior
        trials=self._trials
        g1=trials[:,2]
        g2=trials[:,3]

        #print("get pqr joint simple hybrid")
        sh=self.shear_expand
        if sh is None:
            Pi,Qi,Ri = g_prior.get_pqr_num(g1,g2)
        else:
            print("        expanding about shear:",sh)
            Pi,Qi,Ri = g_prior.get_pqr_num(g1,g2, s1=sh[0], s2=sh[1])
        
        if self.prior_during:
            # We measured the posterior surface.  But the integrals are over
            # the likelihood.  So divide by the prior.
            #
            # Also note the p we divide by is in principle different from the
            # Pi above, which are evaluated at the shear expansion value

            print("undoing prior for pqr")

            prior_vals=self._get_g_prior_vals()

            w,=numpy.where(prior_vals > 0.0)

            Pinv = 1.0/prior_vals[w]
            Pinv_sum=Pinv.sum()

            Pi = Pi[w]
            Qi = Qi[w,:]
            Ri = Ri[w,:,:]

            # this is not unity if expanding about some shear
            Pi *= Pinv
            Qi[:,0] *= Pinv 
            Qi[:,1] *= Pinv

            Ri[:,0,0] *= Pinv
            Ri[:,0,1] *= Pinv
            Ri[:,1,0] *= Pinv
            Ri[:,1,1] *= Pinv

            P = Pi.sum()/Pinv_sum
            Q = Qi.sum(axis=0)/Pinv_sum
            R = Ri.sum(axis=0)/Pinv_sum
        else:
            P = Pi.mean()
            Q = Qi.mean(axis=0)
            R = Ri.mean(axis=0)
 
        return P,Q,R


    def get_gmix(self):
        """
        Get a gaussian mixture at the "best" parameter set, which
        definition depends on the sub-class
        """
        raise RuntimeError("adapt to new style")
        logpars=self._result['pars']
        pars=logpars.copy()
        pars[4] = 10.0**logpars[4]
        pars[5] = 10.0**logpars[5]
        pars[6] = 10.0**logpars[6]

        gm=gmix.make_gmix_model(pars, self.model)
        return gm


    def _get_g_prior_vals(self):
        if not hasattr(self,'joint_prior_vals'):
            trials=self._trials
            g1,g2=trials[:,2],trials[:,3]
            self.joint_prior_vals = self.joint_prior.g_prior.get_prob_array2d(g1,g2)
        return self.joint_prior_vals

    def get_par_names(self):
        names=[r'$cen_1$',
               r'$cen_2$',
               r'$g_1$',
               r'$g_2$',
               r'$log_{10}(T)$',
               r'$log_{10}(F)$',
               r'$log_{10}(n)$']
        return names



class MCMCSersicDefault(MCMCSimple):
    def __init__(self, image, weight, jacobian, **keys):
        raise RuntimeError("adapt to new system")

        self.full_guess=keys.get('full_guess',None)
        self.g1i=2
        self.g2i=3

        self.n_prior=keys.get('n_prior',None)

        if (self.full_guess is None
                or self.n_prior is None):
            raise ValueError("send full guess n_prior for sersic")

        MCMCBase.__init__(self, image, weight, jacobian, "sersic", **keys)


    def _get_priors(self, pars):
        """
        # go in simple
        add any priors that were sent on construction
        """

        lnp=0.0

        if self.cen_prior is not None:
            lnp += self.cen_prior.get_lnprob(pars[0], pars[1])

        if self.g_prior is not None:
            if self.g_prior_during:
                lnp += self.g_prior.get_lnprob_scalar2d(pars[2],pars[3])
            else:
                # may have bounds
                g = sqrt(pars[2]**2 + pars[3]**2)
                if g > self.g_prior.gmax:
                    raise GMixRangeError("g too big")
        
        if self.T_prior is not None:
            lnp += self.T_prior.get_lnprob_scalar(pars[4])

        if self.counts_prior is not None:
            for i,cp in enumerate(self.counts_prior):
                counts=pars[5+i]
                lnp += cp.get_lnprob_scalar(counts)

        lnp += self.n_prior.get_lnprob_scalar(pars[6])

        return lnp

    def _get_guess(self):
        return self.full_guess

    def get_par_names(self):
        names=['cen1','cen2', 'g1','g2','T','F','n']
        return names

    def _set_npars(self):
        """
        nband should be set in set_lists, called before this
        """
        self.npars=self.full_guess.shape[1]

    def get_band_pars(self, pars, band):
        if band > 0:
            raise ValueError("support multi-band for sersic")
        return pars.copy()



class MCMCCoellip(MCMCSimple):
    """
    Add additional features to the base class to support simple models
    """
    def __init__(self, image, weight, jacobian, **keys):

        raise RuntimeError("adapt to new system")

        self.full_guess=keys.get('full_guess',None)
        self.ngauss=gmix.get_coellip_ngauss(self.full_guess.shape[1])
        self.g1i=2
        self.g2i=3

        if self.full_guess is None:
            raise ValueError("send full guess for coellip")

        MCMCBase.__init__(self, image, weight, jacobian, "coellip", **keys)

        self.priors_are_log=keys.get('priors_are_log',False)

        # should make this configurable
        self.first_T_prior=keys.get('first_T_prior',None)
        if self.first_T_prior is not None:
            print("will use first_T_prior")

        # halt tendency to wander off
        #self.sigma_max=keys.get('sigma_max',30.0)
        #self.T_max = 2*self.sigma_max**2

    def _get_guess(self):
        return self.full_guess

    def get_par_names(self):
        names=['cen1','cen2', 'g1','g2']

        for i in xrange(self.ngauss):
            names.append(r'$T_%s$' % i)
        for i in xrange(self.ngauss):
            names.append(r'$F_%s$' % i)

        return names


    def _set_npars(self):
        """
        nband should be set in set_lists, called before this
        """
        self.npars=self.full_guess.shape[1]

    def _get_priors(self, pars):
        """
        # go in simple
        add any priors that were sent on construction
        """

        lnp=0.0

        if self.cen_prior is not None:
            lnp += self.cen_prior.get_lnprob(pars[0], pars[1])

        if self.g_prior is not None:
            if self.g_prior_during:
                lnp += self.g_prior.get_lnprob_scalar2d(pars[2],pars[3])
            else:
                # may have bounds
                g = sqrt(pars[2]**2 + pars[3]**2)
                if g > self.g_prior.gmax:
                    raise GMixRangeError("g too big")
        
        # make sure the first one is constrained in size
        if self.first_T_prior is not None:
            lnp += self.first_T_prior.get_lnprob_scalar(pars[4])

        wbad,=where( pars[4:] <= 0.0 )
        if wbad.size != 0:
            raise GMixRangeError("gauss T or counts too small")


        if self.counts_prior is not None or self.T_prior is not None:
            ngauss=self.ngauss

            Tvals = pars[4:4+ngauss]

            #wbad,=where( (Tvals <= 0.0) | (Tvals > self.T_max) )
            wbad,=where( (Tvals <= 0.0) )
            if wbad.size != 0:
                raise GMixRangeError("out of bounds T values")

            counts_vals = pars[4+ngauss:]
            counts_total=counts_vals.sum()

            if self.counts_prior is not None:
                if len(self.counts_prior) > 1:
                    raise ValueError("make work with multiple bands")

                priors_are_log=self.priors_are_log
                cp=self.counts_prior[0]
                if priors_are_log:
                    if counts_total < 1.e-10:
                        raise GMixRangeError("counts too small")
                    logF = log10(counts_total)
                    lnp += cp.get_lnprob_scalar(logF)
                else:
                    lnp += cp.get_lnprob_scalar(counts_total)

            if self.T_prior is not None:
                T_total = (counts_vals*Tvals).sum()/counts_total

                if priors_are_log:
                    if T_total < 1.e-10:
                        raise GMixRangeError("T too small")
                    logT = log10(T_total)
                    lnp += self.T_prior.get_lnprob_scalar(logT)
                else:
                    lnp += self.T_prior.get_lnprob_scalar(T_total)

        return lnp


    def get_band_pars(self, pars, band):
        if band > 0:
            raise ValueError("support multi-band for coellip")
        return pars.copy()


class MCMCSimpleFixed(MCMCSimple):
    """
    Fix everything but shapes
    """
    def __init__(self, image, weight, jacobian, model, **keys):
        raise RuntimeError("adapt to new system")
        super(MCMCSimpleFixed,self).__init__(image, weight, jacobian, model, **keys)

        # value of elements 2,3 are not important as those are the ones to be
        # varied
        self.fixed_pars=keys['fixed_pars']

        self.npars=2
        self.g1i = 0
        self.g2i = 1

    def _get_priors(self, pars):
        """
        # go in simple
        add any priors that were sent on construction
        
        """
        lnp=0.0
        
        if self.g_prior is not None:
            # may have bounds
            g = sqrt(pars[2]**2 + pars[3]**2)
            if g > self.g_prior.gmax:
                raise GMixRangeError("g too big")
 
        return lnp

    def get_band_pars(self, pars, band):
        raise RuntimeError("adapt to new style")
        bpars= self.fixed_pars[ [0,1,2,3,4,5+band] ]
        bpars[2:2+2] = pars
        return bpars

    def get_par_names(self):
        return ['g1','g2']


class MCMCBDC(MCMCSimple):
    """
    Add additional features to the base class to support coelliptical bulge+disk
    """
    def __init__(self, image, weight, jacobian, **keys):
        raise RuntimeError("adapt to new system")
        super(MCMCBDC,self).__init__(image, weight, jacobian, "bdc", **keys)

        if self.full_guess is None:
            raise ValueError("For BDC you must currently send a full guess")
        self.T_b_prior = keys.get('T_b_prior',None)
        self.T_d_prior = keys.get('T_d_prior',None)
        self.counts_b_prior = keys.get('counts_b_prior',None)
        self.counts_d_prior = keys.get('counts_d_prior',None)

        # we cover this one case, but otherwise the user just have
        # to give this in the right shape
        if self.counts_b_prior is not None:
            self.counts_b_prior=[self.counts_b_prior]
        if self.counts_d_prior is not None:
            self.counts_d_prior=[self.counts_d_prior]

    def _get_priors(self, pars):
        """
        # go in simple
        add any priors that were sent on construction
        """
        lnp=0.0
        
        if self.cen_prior is not None:
            lnp += self.cen_prior.get_lnprob(pars[0], pars[1])

        if self.g_prior is not None:
            # may have bounds
            g = sqrt(pars[2]**2 + pars[3]**2)
            if g > self.g_prior.gmax:
                raise GMixRangeError("g too big")
 
        # bulge size
        if self.T_b_prior is not None:
            lnp += self.T_b_prior.get_lnprob_scalar(pars[4])
        # disk size
        if self.T_d_prior is not None:
            lnp += self.T_d_prior.get_lnprob_scalar(pars[5])

        raise ValueError("fix to put prior on total counts and bdfrac")
        # bulge flux in each band
        if self.counts_b_prior is not None:
            for i,cp in enumerate(self.counts_b_prior):
                counts=pars[6+i]
                lnp += cp.get_lnprob_scalar(counts)

        # disk flux in each band
        if self.counts_d_prior is not None:
            for i,cp in enumerate(self.counts_d_prior):
                counts=pars[6+self.nband+i]
                lnp += cp.get_lnprob_scalar(counts)

        return lnp

    def get_band_pars(self, pars, band):
        """
        pars are 
            [c1,c2,g1,g2,Tb,Td, Fb1,Fb2,Fb3, ..., Fd1,Fd2,Fd3 ...]
        """
        raise RuntimeError("adapt to new style")
        Fbstart=6
        Fdstart=6+self.nband
        return pars[ [0,1,2,3,4,5, Fbstart+band, Fdstart+band] ]


    def get_par_names(self):
        names=['cen1','cen2', 'g1','g2','Tb','Td']
        if self.nband == 1:
            names += ['Fb','Fd']
        else:
            for band in xrange(self.nband):
                names += ['Fb_%s' % band]
            for band in xrange(self.nband):
                names += ['Fd_%s' % band]

        return names


class MCMCBDF(MCMCSimple):
    """
    Add additional features to the base class to support simple models
    """
    def __init__(self, image, weight, jacobian, **keys):
        raise RuntimeError("adapt to new system")
        super(MCMCBDF,self).__init__(image, weight, jacobian, "bdf", **keys)

        if self.full_guess is None:
            raise ValueError("For BDF you must currently send a full guess")

        # we alrady have T_prior and counts_prior from base class

        # fraction of flux in bulge
        self.bfrac_prior = keys.get('bfrac_prior',None)

        # demand flux for both components is > 0
        self.positive_components = keys.get('positive_components',True)

    def _get_priors(self, pars):
        """
        # go in simple
        add any priors that were sent on construction
        """
        lnp=0.0
        
        if self.cen_prior is not None:
            lnp += self.cen_prior.get_lnprob(pars[0], pars[1])

        if self.g_prior is not None:
            # may have bounds
            g = sqrt(pars[2]**2 + pars[3]**2)
            if g > self.g_prior.gmax:
                raise GMixRangeError("g too big")
 
        # prior on total size
        if self.T_prior is not None:
            lnp += self.T_prior.get_lnprob_scalar(pars[4])

        if self.positive_components:
            # both bulge and disk components positive
            if pars[5] <= 0.0 or pars[6] <= 0.0:
                raise GMixRangeError("out of bounds")

        # prior on total counts
        if self.counts_prior is not None:
            for i,cp in enumerate(self.counts_prior):
                counts=pars[5:].sum()
                lnp += cp.get_lnprob_scalar(counts)

        # prior on fraction of total flux in the bulge
        if self.bfrac_prior is not None:

            counts = pars[5:].sum()
            counts_b = pars[5]

            if counts == 0:
                raise GMixRangeError("total counts exactly zero")

            bfrac = counts_b/counts
            lnp += self.bfrac_prior.get_lnprob_scalar(bfrac)

        return lnp

    def get_band_pars(self, pars, band):
        """
        pars are 
            [c1,c2,g1,g2,T, Fb1,Fb2,Fb3, ..., Fd1,Fd2,Fd3 ...]
        """
        raise RuntimeError("adapt to new style")
        Fbstart=5
        Fdstart=5+self.nband
        return pars[ [0,1,2,3,4, Fbstart+band, Fdstart+band] ].copy()


    def get_par_names(self):
        names=['cen1','cen2', 'g1','g2','T']
        if self.nband == 1:
            names += ['Fb','Fd']
        else:
            fbnames = []
            fdnames = []
            for band in xrange(self.nband):
                fbnames.append('Fb_%s' % band)
                fdnames.append('Fd_%s' % band)
            names += fbnames
            names += fdnames
        return names


class MCMCBDFJoint(MCMCBDF):
    """
    BDF with a joint prior on [g1,g2,T,Fb,Fd]
    """
    def __init__(self, image, weight, jacobian, **keys):
        raise RuntimeError("adapt to new system")
        super(MCMCBDF,self).__init__(image, weight, jacobian, "bdf", **keys)

        if self.full_guess is None:
            raise ValueError("For BDF you must currently send a full guess")

        # we alrady have T_prior and counts_prior from base class

        # fraction of flux in bulge
        if self.joint_prior is None:
            raise ValueError("send joint prior for MCMCBDFJoint")

        self.Tfracdiff_max = keys['Tfracdiff_max']

    def _get_priors(self, pars):
        """
        # go in simple
        add any priors that were sent on construction
        """
        lnp=0.0
        
        if self.cen_prior is not None:
            lnp += self.cen_prior.get_lnprob(pars[0], pars[1])

        jp = self.joint_prior 
        if jp is not None:
            T_bounds = jp.T_bounds
            Flux_bounds = jp.Flux_bounds
            T=pars[4]
            Fb=pars[5]
            Fd=pars[6]
            if (T < T_bounds[0] or T > T_bounds[1]
                    or Fb < Flux_bounds[0] or Fb > Flux_bounds[1]
                    or Fd < Flux_bounds[0] or Fd > Flux_bounds[1]):
                raise GMixRangeError("T or flux out of range")
        else:
            # even without a prior, we want to enforce positive
            if pars[4] < 0.0 or pars[5] < 0.0 or pars[6] < 0.0:
                raise GMixRangeError("negative T or flux")


        #lnp = self.joint_prior.get_lnprob(pars[2:])

        return lnp

    def _get_PQR(self):
        """
        get the marginalized P,Q,R from Bernstein & Armstrong
        """

        sh=self.shear_expand
        if sh is None:
            Pi,Qi,Ri = self.joint_prior.get_pqr_num(self._trials[:, 2:])
        else:
            Pi,Qi,Ri = self.joint_prior.get_pqr_num(self._trials[:, 2:],
                                                    s1=sh[0],s2=sh[1])
        P,Q,R = self._get_mean_pqr(Pi,Qi,Ri)

        return P,Q,R


    def _do_trials(self):
        """
        run the sampler
        """
        import emcee

        if emcee.ensemble.acor is not None:
            have_acor=True
        else:
            have_acor=False

        # over-ridden
        guess=self._get_guess()
        for i in xrange(10):
            try:
                self._init_gmix_all(guess[0,:])
                break
            except GMixRangeError as gerror:
                # make sure we draw random guess if we got failure
                print('failed init gmix lol:',str(gerror) )
                print('getting a new guess')
                guess=self._get_random_guess()
        if i==9:
            raise gerror

        sampler = self._make_sampler()
        self.sampler=sampler

        self._tau=9999.0

        Tfracdiff_max=self.Tfracdiff_max


        burnin=self.burnin
        self.last_pos = guess

        print('        burnin runs:',burnin)

        ntry=10
        for i in xrange(ntry):

            if i == 3:
                burnin = burnin*2
                print('        burnin:',burnin)

            sampler.reset()
            self.last_pos, prob, state = sampler.run_mcmc(self.last_pos, burnin)

            trials  = sampler.flatchain
            wts = self.joint_prior.get_prob_array(trials[:,2:], throw=False)

            wsum=wts.sum()

            Tvals=trials[:,4]
            Tmean = (Tvals*wts).sum()/wsum
            Terr2 = ( wts**2 * (Tvals-Tmean)**2 ).sum()
            Terr = sqrt( Terr2 )/wsum

            if i > 0:
                Tfracdiff =abs(Tmean/Tmean_last-1.0)
                Tfracdiff_err = Terr/Tmean_last
                
                tfmess='Tmean: %.3g +/- %.3g Tfracdiff: %.3f +/- %.3f'
                tfmess=tfmess % (Tmean,Terr,Tfracdiff,Tfracdiff_err)

                if (Tfracdiff-1.5*Tfracdiff_err) < Tfracdiff_max:
                    print('        last burn',tfmess)
                    break

                print('        ',tfmess)

            Tmean_last=Tmean
            i += 1

        print('        final run:',self.nstep)
        sampler.reset()
        self.last_pos, prob, state = sampler.run_mcmc(self.last_pos, self.nstep)

        self._trials  = sampler.flatchain
        self.joint_prior_vals = self.joint_prior.get_prob_array(self._trials[:,2:], throw=False)

        arates = sampler.acceptance_fraction
        self._arate = arates.mean()

        lnprobs = sampler.lnprobability.reshape(self.nwalkers*self.nstep)
        w=lnprobs.argmax()
        bp=lnprobs[w]

        self._best_lnprob=bp
        self._best_pars=sampler.flatchain[w,:]

        self.flags=0










class MCMCSimpleJointHybrid(MCMCSimple):
    """
    Simple with a joint prior on [T,F],separate on g1,g2
    """
    def __init__(self, image, weight, jacobian, model, **keys):
        raise RuntimeError("adapt to new system")
        super(MCMCSimpleJointHybrid,self).__init__(image, weight, jacobian, model, **keys)

        if self.full_guess is None:
            raise ValueError("For joint simple you must currently send a full guess")

        if self.joint_prior is None:
            raise ValueError("send joint prior for MCMCSimpleJointHybrid")

        self.prior_during=keys.get('prior_during',False)

    def get_band_pars(self, pars, band):
        """
        Extract pars for the specified band and convert to linear
        """
        from .shape import eta1eta2_to_g1g2
        raise RuntimeError("adapt to new style")
        linpars=pars[ [0,1,2,3,4,5+band] ].copy()

        linpars[4] = 10.0**linpars[4]
        linpars[5] = 10.0**linpars[5]

        return linpars


    def _get_priors(self, pars):
        """
        Apply simple priors
        """
        lnp=0.0
        
        if self.cen_prior is not None:
            lnp += self.cen_prior.get_lnprob(pars[0], pars[1])

        jp=self.joint_prior

        # this is just the structural parameters
        lnp += jp.get_lnprob_scalar(pars[4:])

        if self.prior_during:
            lnp += jp.g_prior.get_lnprob_scalar2d(pars[2],pars[3])

        return lnp

    def _get_PQR(self):
        """
        get the marginalized P,Q,R from Bernstein & Armstrong
        """

        g_prior=self.joint_prior.g_prior
        trials=self._trials
        g1=trials[:,2]
        g2=trials[:,3]

        #print("get pqr joint simple hybrid")
        sh=self.shear_expand
        if sh is None:
            Pi,Qi,Ri = g_prior.get_pqr_num(g1,g2)
        else:
            print("        expanding about shear:",sh)
            Pi,Qi,Ri = g_prior.get_pqr_num(g1,g2, s1=sh[0], s2=sh[1])
        
        if self.prior_during:
            # We measured the posterior surface.  But the integrals are over
            # the likelihood.  So divide by the prior.
            #
            # Also note the p we divide by is in principle different from the
            # Pi above, which are evaluated at the shear expansion value

            print("undoing prior for pqr")

            prior_vals=self._get_g_prior_vals()

            w,=numpy.where(prior_vals > 0.0)

            Pinv = 1.0/prior_vals[w]
            Pinv_sum=Pinv.sum()

            Pi = Pi[w]
            Qi = Qi[w,:]
            Ri = Ri[w,:,:]

            # this is not unity if expanding about some shear
            Pi *= Pinv
            Qi[:,0] *= Pinv 
            Qi[:,1] *= Pinv

            Ri[:,0,0] *= Pinv
            Ri[:,0,1] *= Pinv
            Ri[:,1,0] *= Pinv
            Ri[:,1,1] *= Pinv

            P = Pi.sum()/Pinv_sum
            Q = Qi.sum(axis=0)/Pinv_sum
            R = Ri.sum(axis=0)/Pinv_sum
        else:
            P = Pi.mean()
            Q = Qi.mean(axis=0)
            R = Ri.mean(axis=0)
 
        return P,Q,R


    def get_gmix(self):
        """
        Get a gaussian mixture at the "best" parameter set, which
        definition depends on the sub-class
        """
        raise RuntimeError("adapt to new style")
        logpars=self._result['pars']
        pars=logpars.copy()
        pars[4] = 10.0**logpars[4]
        pars[5] = 10.0**logpars[5]

        gm=gmix.make_gmix_model(pars, self.model)
        return gm


    def _get_g_prior_vals(self):
        if not hasattr(self,'joint_prior_vals'):
            trials=self._trials
            g1,g2=trials[:,2],trials[:,3]
            self.joint_prior_vals = self.joint_prior.g_prior.get_prob_array2d(g1,g2)
        return self.joint_prior_vals

    def get_par_names(self):
        names=[r'$cen_1$',
               r'$cen_2$',
               r'$g_1$',
               r'$g_2$',
               r'$log_{10}(T)$']
        if self.nband == 1:
            names += [r'$log_{10}(F)$']
        else:
            for band in xrange(self.nband):
                names += [r'$log_{10}(F_%s)$' % band]
        return names


class MCMCBDFJointHybrid(MCMCSimpleJointHybrid):
    """
    BDF with a joint prior on [T,Fb,Fd] separate on g1,g2
    """

    def __init__(self, image, weight, jacobian, **keys):
        raise RuntimeError("adapt to new system")
        super(MCMCBDFJointHybrid,self).__init__(image, weight, jacobian, "bdf", **keys)

    def get_band_pars(self, pars, band):
        """
        Extract pars for the specified band and convert to linear
        """
        raise RuntimeError("adapt to new style")
        Fbstart=5
        Fdstart=5+self.nband

        linpars = pars[ [0,1,2,3,4, Fbstart+band, Fdstart+band] ].copy()

        linpars[4] = 10.0**linpars[4]
        linpars[5] = 10.0**linpars[5]
        linpars[6] = 10.0**linpars[6]

        return linpars

    def get_gmix(self):
        """
        Get a gaussian mixture at the "best" parameter set, which
        definition depends on the sub-class
        """
        raise RuntimeError("adapt to new style")
        logpars=self._result['pars']
        pars=logpars.copy()
        pars[4] = 10.0**logpars[4]
        pars[5] = 10.0**logpars[5]
        pars[6] = 10.0**logpars[6]

        gm=gmix.make_gmix_model(pars, self.model)
        return gm

    def get_par_names(self):
        names=[r'$cen_1$',
               r'$cen_2$',
               r'$g_1$',
               r'$g_2$',
               r'$log_{10}(T)$']
        if self.nband == 1:
            names += [r'$log_{10}(F_b)$',r'$log_{10}(F_d)$']
        else:
            for ftype in ['b','d']:
                for band in xrange(self.nband):
                    names += [r'$log_{10}(F_%s^%s)$' % (ftype,band)]
        return names



class MCMCSimpleJointLinPars(MCMCSimple):
    """
    Simple with a joint prior on [g1,g2,T,Fb,Fd]
    """
    def __init__(self, image, weight, jacobian, model, **keys):
        raise RuntimeError("adapt to new system")
        super(MCMCSimpleJointLinPars,self).__init__(image, weight, jacobian, model, **keys)

        if self.full_guess is None:
            raise ValueError("For joint simple you must currently send a full guess")

        if self.joint_prior is None:
            raise ValueError("send joint prior for MCMCSimpleJointLinPars")

        self.prior_during=keys['prior_during']

    def _get_eabs_pars(self, pars):
        """
        don't include centroid, and only total ellipticity
        """
        if len(pars.shape) == 2:
            eabs_pars=zeros( (pars.shape[0], self.ndim-3) )
            eabs_pars[:,0] = sqrt(pars[:,2]**2 + pars[:,3]**2)
            eabs_pars[:,1:] = pars[:,4:]
        else:
            eabs_pars=zeros(self.ndim-3)

            eabs_pars[0] = sqrt(pars[2]**2 + pars[3]**2)
            eabs_pars[1:] = pars[4:]

        return eabs_pars

    def _get_priors(self, pars):
        """
        Apply simple priors
        """
        lnp=0.0
        
        if self.cen_prior is not None:
            lnp += self.cen_prior.get_lnprob(pars[0], pars[1])

        eabs_pars=self._get_eabs_pars(pars)

        jp=self.joint_prior
        if self.prior_during:
            lnp += jp.get_lnprob_scalar(eabs_pars)
        else:
            # this can raise a GMixRangeError exception
            jp.check_bounds_scalar(eabs_pars)

        return lnp

    def _get_PQR(self):
        """
        get the marginalized P,Q,R from Bernstein & Armstrong
        """

        print("get pqr joint simple")
        sh=self.shear_expand
        if sh is None:
            Pi,Qi,Ri = self.joint_prior.get_pqr_num(self._trials[:,2:])
        else:
            Pi,Qi,Ri = self.joint_prior.get_pqr_num(self._trials[:,2:],
                                                    s1=sh[0],
                                                    s2=sh[1])
        
        if self.prior_during:
            # We measured the posterior surface.  But the integrals are over
            # the likelihood.  So divide by the prior.
            #
            # Also note the p we divide by is in principle different from the
            # Pi above, which are evaluated at the shear expansion value

            prior_vals=self._get_joint_prior_vals()

            w,=numpy.where(prior_vals > 0.0)

            Pinv = 1.0/prior_vals[w]
            Pinv_sum=Pinv.sum()

            Pi = Pi[w]
            Qi = Qi[w,:]
            Ri = Ri[w,:,:]

            # this is not unity if expanding about some shear
            Pi *= Pinv
            Qi[:,0] *= Pinv 
            Qi[:,1] *= Pinv

            Ri[:,0,0] *= Pinv
            Ri[:,0,1] *= Pinv
            Ri[:,1,0] *= Pinv
            Ri[:,1,1] *= Pinv

            P = Pi.sum()/Pinv_sum
            Q = Qi.sum(axis=0)/Pinv_sum
            R = Ri.sum(axis=0)/Pinv_sum
        else:
            P = Pi.mean()
            Q = Qi.mean(axis=0)
            R = Ri.mean(axis=0)
 
        return P,Q,R

    
    def _get_joint_prior_vals(self):
        if not hasattr(self,'joint_prior_vals'):
            eabs_pars=self._get_eabs_pars(self._trials)
            self.joint_prior_vals = self.joint_prior.get_prob_array(eabs_pars)
        return self.joint_prior_vals


class MCMCSimpleJointLogPars(MCMCSimple):
    """
    Simple with a joint prior on [g1,g2,T,Fb,Fd]
    """
    def __init__(self, image, weight, jacobian, model, **keys):
        raise RuntimeError("adapt to new style")
        super(MCMCSimpleJointLogPars,self).__init__(image, weight, jacobian, model, **keys)

        if self.full_guess is None:
            raise ValueError("For joint simple you must currently send a full guess")

        # we alrady have T_prior and counts_prior from base class

        # fraction of flux in bulge
        if self.joint_prior is None:
            raise ValueError("send joint prior for MCMCSimpleJointLogPars")

        self.prior_during=keys['prior_during']

    def get_band_pars(self, pars, band):
        """
        Extract pars for the specified band and convert to linear
        """
        raise RuntimeError("deal with non logpars")
        from .shape import eta1eta2_to_g1g2
        linpars=pars[ [0,1,2,3,4,5+band] ].copy()

        g1,g2=eta1eta2_to_g1g2(pars[2],pars[3])
        linpars[2] = g1
        linpars[3] = g2
        linpars[4] = 10.0**pars[4]
        linpars[5] = 10.0**pars[5]

        return linpars

    def _get_priors(self, pars):
        """
        Apply simple priors
        """
        lnp=0.0
        
        if self.cen_prior is not None:
            lnp += self.cen_prior.get_lnprob(pars[0], pars[1])

        jp=self.joint_prior
        if self.prior_during:
            lnp += jp.get_lnprob_scalar(pars[2:])
        else:
            # this can raise a GMixRangeError exception
            jp.check_bounds_scalar(pars[2:])

        return lnp

    def _get_PQR(self):
        """
        get the marginalized P,Q,R from Bernstein & Armstrong
        """

        print("get pqr joint simple")
        sh=self.shear_expand
        if sh is None:
            Pi,Qi,Ri = self.joint_prior.get_pqr_num(self._trials[:, 2:])
        else:
            Pi,Qi,Ri = self.joint_prior.get_pqr_num(self._trials[:, 2:],
                                                    s1=sh[0],
                                                    s2=sh[1])
        
        if self.prior_during:
            # We measured the posterior surface.  But the integrals are over
            # the likelihood.  So divide by the prior.
            #
            # Also note the p we divide by is in principle different from the
            # Pi above, which are evaluated at the shear expansion value

            prior_vals=self._get_joint_prior_vals()

            w,=numpy.where(prior_vals > 0.0)

            Pinv = 1.0/prior_vals[w]
            Pinv_sum=Pinv.sum()

            Pi = Pi[w]
            Qi = Qi[w,:]
            Ri = Ri[w,:,:]

            # this is not unity if expanding about some shear
            Pi *= Pinv
            Qi[:,0] *= Pinv 
            Qi[:,1] *= Pinv

            Ri[:,0,0] *= Pinv
            Ri[:,0,1] *= Pinv
            Ri[:,1,0] *= Pinv
            Ri[:,1,1] *= Pinv

            P = Pi.sum()/Pinv_sum
            Q = Qi.sum(axis=0)/Pinv_sum
            R = Ri.sum(axis=0)/Pinv_sum
        else:
            P = Pi.mean()
            Q = Qi.mean(axis=0)
            R = Ri.mean(axis=0)
 
        return P,Q,R

    
    def _get_joint_prior_vals(self):
        if not hasattr(self,'joint_prior_vals'):
            self.joint_prior_vals = self.joint_prior.get_prob_array(self._trials[:,2:])
        return self.joint_prior_vals

    def get_par_names(self):
        names=[r'$cen_1$',
               r'$cen_2$',
               r'$\eta_1$',
               r'$\eta_2$',
               r'$log_{10}(T)$']
        if self.nband == 1:
            names += [r'$log_{10}(T)$']
        else:
            for band in xrange(self.nband):
                names += [r'$log_{10}(F_%s)$' % band]
        return names


_default_min_err=array([1.e-4,1.e-4,1.e-3,1.e-3,1.0e-4,1.0e-4])
_default_max_err=array([1.0,1.0,5.0,5.0,1.0,1.0])

class PSampler(object):
    def __init__(self, pars, perr, samples,
                 max_use=None,
                 nsigma=4.0,
                 min_err=_default_min_err,
                 max_err=_default_max_err,
                 verbose=True):
        self._pars=array(pars)
        self._perr_orig=array(perr)
        self._npars = self._pars.size

        self._samples=samples
        self._nsigma=nsigma

        if max_use is None:
            max_use=samples.shape[0]

        self._max_use=max_use

        self._set_err(min_err, max_err)
        self.verbose=verbose

    def get_result(self):
        """
        get the result dictionary
        """
        return self._result

    def get_loglikes(self):
        """
        get the log likelihoods for used samples
        """
        return self._logl_vals

    def get_likelihoods(self):
        """
        get the likelihoods for used samples
        """
        return self._lvals

    def get_used_samples(self):
        """
        get the subset of the samples used
        """
        return self._used_samples
    def get_used_indices(self):
        """
        get indices of used samples
        """
        return self._used_indices

    def calc_loglikes(self, lnprob_func):
        """
        calculate the loglike for a subset of the points
        """
        res={'flags':0}
        self._result=res

        w=self._select_samples()

        samples=self._samples
        res['nuse'] = w.size
        if w.size == 0:
            res['flags']=1
        else:
            used_samples = samples[w]

            logl_vals = zeros(w.size)
            for i in xrange(w.size):
                tpars = used_samples[i,:]
                logl_vals[i] = lnprob_func(tpars)

            logl_max = logl_vals.max()
            logl_vals -= logl_max

            lvals = exp(logl_vals)

            self._used_samples = used_samples
            self._used_indices = w

            self._logl_vals=logl_vals
            self._lvals=lvals

            self._calc_result()

    def _calc_result(self):
        """
        Calculate the mcmc stats and the "best fit" stats
        """
        from numpy import diag

        result=self._result
        if result['flags'] != 0:
            return
        
        pars,pars_cov = self.get_stats()
        pars_err=sqrt(diag(pars_cov))

        neff = self._lvals.sum()
        efficiency = neff/self._lvals.size

        res={'pars':pars,
             'pars_cov':pars_cov,
             'pars_err':pars_err,
             'g':pars[2:2+2],
             'g_cov':pars_cov[2:2+2, 2:2+2],
             'neff':neff,
             'efficiency':efficiency}

        result.update(res)
 
    def get_stats(self):
        """
        get expectation values and covariance for
        g from the trials
        """
        from ngmix import stats

        samples = self.get_used_samples()
        likes   = self.get_likelihoods()

        pars, pars_cov = stats.calc_mcmc_stats(samples, weights=likes)

        return pars, pars_cov
 
    def _select_samples(self):
        from esutil.numpy_util import between
        samples=self._samples
        np = samples.shape[0]

        pars=self._pars
        perr=self._perr
        nsigma=self._nsigma

        logic = ones(np, dtype=bool)
        for i in xrange(self._npars):
            minval = pars[i]-nsigma*perr[i]
            maxval = pars[i]+nsigma*perr[i]
            logic = logic & between(samples[:,i], minval, maxval)

        w,=where(logic)

        if w.size > self._max_use:
            w=w[0:self._max_use]

            tmp=numpy.random.random(w.size)
            s=tmp.argsort()

            w = w[s]

        return w

    def _set_err(self, min_err, max_err):
        if min_err is None:
            min_err = self._pars_orig*0
        else:
            min_err=array(min_err,copy=False)

        if max_err is None:
            max_err = self._pars_orig*0 + numpy.inf
        else:
            max_err=array(max_err,copy=False)

        assert min_err.size==self._pars.size,"min_err must be same size as pars"
        assert max_err.size==self._pars.size,"max_err must be same size as pars"

        operr=self._perr_orig
        perr=self._perr_orig*0

        for i in xrange(perr.size):
            perr[i] = operr[i].clip(min=min_err[i],
                                    max=max_err[i])
        self._perr = perr

class ISampler(object):
    """
    sampler using multivariate T distribution
    """
    def __init__(self, pars, cov, df,
                 min_err=_default_min_err,
                 max_err=_default_max_err,
                 ifactor=1.0,asinh_pars=[],verbose=True):
        """
        min_err=0.001 for s/n=1000 T=4*Tpsf
        max_err=0.5 for small T s/n ~5
        """
        self._pars_orig=array(pars)
        self._cov_orig=array(cov)
        self._ifac = ifactor
        self._asinh_pars = asinh_pars
        self._npars = self._pars_orig.size
        self._set_pars_and_cov()        

        self.verbose=verbose
        self._set_minmax_err(min_err, max_err)

        self._clip_cov()

        self._df=df
        self._set_pdf()

    def _set_pars_and_cov(self):
        from math import asinh
        
        self._pars = self._pars_orig.copy()
        self._cov = self._cov_orig.copy()
        jac = numpy.ones(self._npars)
        
        for ind in self._asinh_pars:
            self._pars[ind] = asinh(self._pars_orig[ind])
            jac[ind] = 1.0/numpy.sqrt(1.0 + self._pars_orig[ind]*self._pars_orig[ind])                
        
        for i in xrange(self._npars):            
            for j in xrange(self._npars):
                self._cov[i,j] = jac[i]*jac[j]*self._cov_orig[i,j]

        self._cov = self._cov*self._ifac*self._ifac

    def _lndetjac(self,pars_orig):
        npars = pars_orig.shape[0]
        logdetjac = zeros(npars)
        for ind in self._asinh_pars:
            logdetjac += numpy.log(1.0/numpy.sqrt(1.0 + pars_orig[:,ind]*pars_orig[:,ind]))
        return logdetjac

    def pars_to_pars_orig(self,pars):
        pars_orig = pars.copy()
        for ind in self._asinh_pars:
            pars_orig[:,ind] = numpy.sinh(pars[:,ind])
        return pars_orig

    def make_samples(self, n=None):
        """
        run sample() and set internal trials attribute
        """
        self._trials = self.sample(n)
        self._trials_orig = self.pars_to_pars_orig(self._trials)
        
    make_trials=make_samples

    def get_result(self):
        """
        get the result dict
        """
        return self._result

    def calc_result(self, weights=None):
        """
        Calculate the mcmc stats and the "best fit" stats
        """
        from numpy import diag

        pars,pars_cov,neff = self.get_stats(weights=weights)
        pars_err=sqrt(diag(pars_cov))

        trials=self.get_trials()
        nsample=trials.shape[0]
        efficiency = neff/nsample

        res={'flags':0,
             'pars':pars,
             'pars_cov':pars_cov,
             'pars_err':pars_err,
             'g':pars[2:2+2],
             'g_cov':pars_cov[2:2+2, 2:2+2],
             'nsample':nsample,
             'neff':neff,
             'efficiency':efficiency}

        self._result=res
 
    def get_stats(self, weights=None):
        """
        get expectation values and covariance for
        g from the trials
        """
        from ngmix import stats
        trials=self.get_trials()
        iweights = self.get_iweights()

        # should we modify this for extra input weights?
        # maybe not: we want to know how well we sample
        # what was input
        #neff = iweights.sum()

        if weights is not None:
            weights = weights * iweights
            neff = weights.sum()/weights.max()
        else:
            weights = iweights
            neff = weights.sum()

        pars, pars_cov = stats.calc_mcmc_stats(trials, weights=weights)

        return pars, pars_cov, neff
 
    def get_trials(self):
        """
        return a ref to the trials
        """
        return self._trials_orig
    get_samples=get_trials

    def get_prob(self, pars):
        """
        get probability for input points

        depends on scipy
        """
        if not hasattr(self, '_pdf'):
            self._set_pdf()

        return self._pdf.pdf(pars)

    def get_lnprob(self, pars):
        """
        get log probability for input points

        depends on scipy
        """
        if not hasattr(self, '_pdf'):
            self._set_pdf()

        return self._pdf.logpdf(pars) 

    def get_iweights(self):
        """
        set self._iweights for self._trials given the
        lnprob_func.  You need to run make_trials first
        """
        return self._iweights

    def set_iweights(self, lnprob_func):
        """
        get importance sample weights for the input
        samples and lnprob function
        """

        proposed_lnprob = self.get_lnprob(self._trials)
        lndetjac = self._lndetjac(self._trials_orig)

        samples = self._trials_orig
        nsample = samples.shape[0]
        lnprob = zeros(nsample)        
        for i in xrange(nsample):
            lnprob[i] = lnprob_func(samples[i,:])

        lnpdiff = lnprob - proposed_lnprob - lndetjac
        lnpdiff -= lnpdiff.max()
        self._iweights = exp(lnpdiff)

    def sample(self, nrand=None):
        """
        Get nrand random deviates from the distribution
        """

        if nrand is None:
            is_scalar=True
            nrand=1
        else:
            is_scalar=False

        vals = numpy.zeros( (nrand,self._npars) )

        ngood=0
        nleft=nrand
        while ngood < nrand:
            
            samples = self._sample_raw(nleft)

            gtot = samples[:,2]**2 + samples[:,3]**2

            w,=numpy.where(gtot < 1.0)
            if w.size > 0:
                vals[ngood:ngood+w.size, :] = samples[w,:]
                ngood += w.size
                nleft -= w.size
 
        if is_scalar:
            vals = vals[0,:]

        return vals

    def make_plots(self,
                   weights=None,
                   title=None,
                   separate=False,
                   width=1200,
                   height=1200,
                   show=False,
                   prompt=False,
                   **keys):
        import mcmc
        import biggles

        biggles.configure('screen','width', width)
        biggles.configure('screen','height', height)

        if separate:
            # returns a tuple burn_plt, hist_plt
            plotfunc =mcmc.plot_results_separate
        else:
            plotfunc =mcmc.plot_results

        trials=self.get_trials()
        pdict={}
        pdict['trials']=plotfunc(trials,
                                 title=title,
                                 show=show,
                                 **keys)


        if weights is not None:
            pdict['wtrials']=plotfunc(trials,
                                      weights=weights,
                                      title='%s weighted' % title,
                                      show=show,
                                      **keys)

        if show and prompt:
            key=raw_input('hit a key: ')
            if key=='q':
                stop

        return pdict

    def _sample_raw(self, n):
        """
        sample without truncation
        """
        return self._pdf.rvs(n)

    #def _sample_raw(self, n):
    #    """
    #    sample from the cov, no truncation
    #    """
    #    from numpy.random import multivariate_normal

    #    vals=multivariate_normal(self._pars, self._cov, n)
    #    return vals

    def _set_pdf(self):
        from statsmodels.sandbox.distributions.mv_normal import MVT
        self._pdf=MVT(self._pars, self._cov, self._df)


    #def _set_pdf(self):
    #    """
    #    don't do automatically, since depends on scipy
    #    """
    #    from scipy.stats import multivariate_normal 
    #    self._pdf = multivariate_normal(mean=self._pars, cov=self._cov)

    def _clip_cov(self):
        """
        clip the steps to a desired range.  Can work with
        either diagonals or cov
        """
        from numpy import sqrt, diag

        cov=self._cov

        # correlation matrix
        dsigma = sqrt(diag(cov))
        corr = cov.copy()
        for i in xrange(cov.shape[0]):
            for j in xrange(cov.shape[1]):
                corr[i,j] /= dsigma[i]
                corr[i,j] /= dsigma[j]
        
        w,=numpy.where(dsigma < self._min_err)
        if w.size > 0:
            dsigma[w] = self._min_err[w]
        w,=numpy.where(dsigma > self._max_err)
        if w.size > 0:
            dsigma[w] = self._max_err[w]

        # remake the covariance matrix
        for i in xrange(corr.shape[0]):
            for j in xrange(corr.shape[1]):
                corr[i,j] *= dsigma[i]
                corr[i,j] *= dsigma[j]

        cov = corr.copy()

        # make sure the matrix is well behavied            
        if numpy.all(numpy.isfinite(cov)):
            eigvals=numpy.linalg.eigvals(cov)
            if numpy.any(eigvals <= 0):
                raise LinAlgError("bad cov")

        if self.verbose:
            print_pars(sqrt(diag(cov)), front="    using err:")

        self._cov = cov

    def _set_minmax_err(self, min_err, max_err):
        if min_err is None:
            min_err = pars*0
        else:
            min_err=array(min_err,copy=False)

        if max_err is None:
            max_err = pars*0 + numpy.inf
        else:
            max_err=array(max_err,copy=False)

        assert min_err.size==self._pars.size,"min_err must be same size as pars"
        assert max_err.size==self._pars.size,"max_err must be same size as pars"

        self._min_err=min_err
        self._max_err=max_err


class ISamplerMom(ISampler):
    def calc_result(self, weights=None):
        """
        Calculate the mcmc stats and the "best fit" stats
        """
        super(ISamplerMom,self).calc_result(weights=None)
        del self._result['g']
        del self._result['g_cov']

    def sample(self, nrand=None):
        """
        Get nrand random deviates from the distribution
        """

        if nrand is None:
            is_scalar=True
            nrand=1
        else:
            is_scalar=False

        vals = self._pdf.rvs(nrand)
 
        if is_scalar:
            vals = vals[0,:]

        return vals

# alias
GCovSamplerT=ISampler

def get_edge_aperture(dims, cen):
    """
    get circular aperture such that the entire aperture
    is visible in all directions without hitting an edge

    parameters
    ----------
    dims: 2-element sequence
        dimensions of the array [dim1, dim2]
    cen: 2-element sequence
        [cen1, cen2]

    returns
    -------
    min(min(cen[0],dims[0]-cen[0]),min(cen[1],dims[1]-cen[1]))
    """
    aperture=min(min(cen[0],dims[0]-cen[0]),min(cen[1],dims[1]-cen[1]))
    return aperture


def print_pars(pars, stream=stdout, fmt='%8.3g',front=None):
    """
    print the parameters with a uniform width
    """
    if front is not None:
        stream.write(front)
        stream.write(' ')
    if pars is None:
        stream.write('%s\n' % None)
    else:
        fmt = ' '.join( [fmt+' ']*len(pars) )
        stream.write(fmt % tuple(pars))
        stream.write('\n')



