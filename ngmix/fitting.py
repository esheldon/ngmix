"""

- todo
    - make sure the psf flux fitting in my other codes are now sending the
    center in sky coordinates
    - make sure the codes are not re-setting the jacobian!
    - split out pqr calculations
    * split out lensfit calculations
    - support only a single prior sent
        - take care of g prior not during by sending weights= to
        the calc_result
        - seperate lensfit/pqr then need to deal with remove prior for
        g prior during
    - support only full guess
        - everywhere that these can be sent, including T= keywords etc.
        - lots of _get_priors need to be adapted

"""
# there are a few additional imports not in this header for example we only
# import emcee if needed
from __future__ import print_function

try:
    xrange = xrange
    # We have Python 2
except:
    xrange = range
    # We have Python 3

from sys import stdout
import numpy
from numpy import array, zeros, diag, exp, sqrt, where, log, log10, isfinite
import time
from pprint import pprint

from . import gmix
from .gmix import GMix, GMixList, MultiBandGMixList

from . import _gmix

from .jacobian import Jacobian, UnitJacobian

from . import priors
from .priors import srandu, LOWVAL, BIGVAL

from .gexceptions import GMixRangeError, GMixFatalError

from .observation import Observation,ObsList,MultiBandObsList

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
LM_EIG_NOTFINITE = 2**7
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

        self._set_obs(obs)

        self.prior = keys.get('prior',None)

        # in this case, image, weight, jacobian, psf are going to
        # be lists of lists.

        self.model=gmix.get_model_num(model)
        self.model_name=gmix.get_model_name(self.model)
        self._set_npars()

        # the function to be called to fill a gaussian mixture
        #self._set_fill_call()

        self._set_totpix()

        self._gmix_all=None

    def get_result(self):
        """
        Result will not be non-None until go() is run
        """
        if not hasattr(self, '_result'):
            raise ValueError("No result, you must run_mcmc and calc_result!")

        return self._result

    def get_lin_result(self):
        """
        Result will not be non-None until go() is run
        """
        if not hasattr(self, '_lin_result'):
            raise ValueError("No lin result, you must run_mcmc and calc_lin_result!")

        return self._lin_result


    def get_gmix(self):
        """
        Get a gaussian mixture at the "best" parameter set, which
        definition depends on the sub-class
        """
        if not hasattr(self, '_lin_result'):
            raise RuntimeError("linear result not present")
        linres=self.get_lin_result()
        pars=linres['pars']
        return gmix.make_gmix_model(pars, self.model)

    def _set_obs(self, obs_in):
        """
        Input should be an Observation, ObsList, or MultiBandObsList
        """


        if isinstance(obs_in,Observation):
            obs_list=ObsList()
            obs_list.append(obs_in)

            obs=MultiBandObsList()
            obs.append(obs_list)
        elif isinstance(obs_in,ObsList):
            obs=MultiBandObsList()
            obs.append(obs_in)
        elif isinstance(obs_in,MultiBandObsList):
            obs=obs_in
        else:
            raise ValueError("obs should be Observation, ObsList, or MultiBandObsList")

        self.nband=len(obs)
        self.obs=obs

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


    def _set_fill_call(self):
        """
        making the call directly to the jitted function saves
        huge time
        """
        if self.model==gmix.GMIX_FULL:
            self._fill_gmix_func=gmix._fill_full
        elif self.model==gmix.GMIX_GAUSS:
            self._fill_gmix_func=gmix._fill_gauss
        elif self.model==gmix.GMIX_EXP:
            self._fill_gmix_func=gmix._fill_exp
        elif self.model==gmix.GMIX_DEV:
            self._fill_gmix_func=gmix._fill_dev
        elif self.model==gmix.GMIX_TURB:
            self._fill_gmix_func=gmix._fill_turb
        elif self.model==gmix.GMIX_BDC:
            self._fill_gmix_func=gmix._fill_bdc
        elif self.model==gmix.GMIX_BDF:
            self._fill_gmix_func=gmix._fill_bdf
        elif self.model==gmix.GMIX_SERSIC:
            pass
        elif self.model==gmix.GMIX_COELLIP:
            self._fill_gmix_func=gmix._fill_coellip
        else:
            raise GMixFatalError("unsupported model: "
                                 "'%s'" % self.model_name)



    def get_dof(self):
        """
        Effective def based on effective number of pixels
        """
        eff_npix=self.get_effective_npix()
        dof = eff_npix-self.npars
        if dof <= 0:
            dof = 1.e-6
        return dof

    def get_effective_npix(self):
        """
        Because of the weight map, each pixel gets a different weight in the
        chi^2.  This changes the effective degrees of freedom.  The extreme
        case is when the weight is zero; these pixels are essentially not used.

        We replace the number of pixels with

            eff_npix = sum(weights)maxweight
        """
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


    def calc_lnprob(self, pars, get_s2nsums=False, get_priors=False):
        """
        This is all we use for mcmc approaches, but also used generally for the
        "get_fit_stats" method.  For the max likelihood fitter we also have a
        _get_ydiff method
        """

        s2n_numer=0.0
        s2n_denom=0.0
        try:

            # these are the log pars (if working in log space)
            ln_priors = self._get_priors(pars)
            ln_prob = 0.0

            self._fill_gmix_all(pars)
            for band in xrange(self.nband):

                obs_list=self.obs[band]
                gmix_list=self._gmix_all[band]

                for obs,gm in zip(obs_list, gmix_list):

                    res = gm.get_loglike(obs, get_s2nsums=True)

                    ln_prob += res[0]
                    s2n_numer += res[1]
                    s2n_denom += res[2]

            ln_prob += ln_priors

        except GMixRangeError:
            ln_prob = LOWVAL
            s2n_numer=0.0
            s2n_denom=BIGVAL


        if get_s2nsums:
            return ln_prob, s2n_numer, s2n_denom
        else:
            if get_priors:
                return ln_prob, ln_priors
            else:
                return ln_prob

    def get_fit_stats(self, pars):
        """
        Get some fit statistics for the input pars.
        """
        npars=self.npars

        lnprob,s2n_numer,s2n_denom=self.calc_lnprob(pars, get_s2nsums=True)

        if s2n_denom > 0:
            s2n=s2n_numer/sqrt(s2n_denom)
        else:
            s2n=0.0

        dof=self.get_dof()
        eff_npix=self.get_effective_npix()

        chi2=lnprob/(-0.5)
        chi2per = chi2/dof

        aic = -2*lnprob + 2*npars
        bic = -2*lnprob + npars*numpy.log(eff_npix)

        return {'s2n_w':s2n,
                'lnprob':lnprob,
                'chi2per':chi2per,
                'dof':dof,
                'aic':aic,
                'bic':bic}


    def _init_gmix_all(self, pars):
        """
        initialize the list of lists of gaussian mixtures
        """
        psf=self.obs[0][0].psf
        if psf is None:
            dopsf=False
            print("no psf")
        else:
            dopsf=True

        gmix_all0 = MultiBandGMixList()
        gmix_all  = MultiBandGMixList()

        for band,obs_list in enumerate(self.obs):
            gmix_list0=GMixList()
            gmix_list=GMixList()

            # pars for this band, in linear space
            band_pars=self._get_band_pars(pars, band)

            for obs in obs_list:
                if dopsf:
                    psf_gmix=obs.psf.gmix

                    gm0=gmix.make_gmix_model(band_pars, self.model)
                    gm=gm0.convolve(psf_gmix)
                else:
                    gm0=gmix.make_gmix_model(band_pars, self.model)
                    gm=gm0


                gmix_list0.append(gm0)
                gmix_list.append(gm)

            gmix_all0.append(gmix_list0)
            gmix_all.append(gmix_list)

        self._gmix_all0 = gmix_all0
        self._gmix_all  = gmix_all

    def _fill_gmix_all(self, pars):
        """
        Fill the list of lists of gmix objects for the given parameters
        """

        psf=self.obs[0][0].psf
        if psf is None:
            self._fill_gmix_all_nopsf(pars)
            return

        for band,obs_list in enumerate(self.obs):
            gmix_list0=self._gmix_all0[band]
            gmix_list=self._gmix_all[band]

            # pars for this band, in linear space
            band_pars=self._get_band_pars(pars, band)

            for i,obs in enumerate(obs_list):

                psf_gmix=obs.psf.gmix

                gm0=gmix_list0[i]
                gm=gmix_list[i]

                #gm0.fill(band_pars)
                _gmix.gmix_fill(gm0._data, band_pars, gm0._model)
                _gmix.convolve_fill(gm._data, gm0._data, psf_gmix._data)

    def _fill_gmix_all_nopsf(self, pars):
        """
        Fill the list of lists of gmix objects for the given parameters
        """

        for band,obs_list in enumerate(self.obs):
            gmix_list=self._gmix_all[band]

            # pars for this band, in linear space
            band_pars=self._get_band_pars(pars, band)

            for i,obs in enumerate(obs_list):

                gm=gmix_list0[i]

                try:
                    gm.fill(band_pars)
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
                       width=1920, height=1200):
        import images
        import biggles

        biggles.configure('screen','width', width)
        biggles.configure('screen','height', height)

        try:
            self._fill_gmix_all(self._result['pars'])
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

                model=gm.make_image(im.shape,jacobian=j)

                showim = im*wt
                showmod = model*wt

                sub_tab=images.compare_images(showim, showmod,show=False)
                sub_tab.title=this_title

                band_list.append(sub_tab)

                if show:
                    sub_tab.show()

            plist.append(band_list)
        return plist


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

        self._set_obs(obs)

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

    def _set_obs(self, obs_in):
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
        Because of the weight map, each pixel gets a different weight in the
        chi^2.  This changes the effective degrees of freedom.  The extreme
        case is when the weight is zero; these pixels are essentially not used.

        We replace the number of pixels with

            eff_npix = sum(weights)maxweight
        """
        if not hasattr(self, 'eff_npix'):
            wtmax = 0.0
            wtsum = 0.0

            for obs in self.obs:
                wt=obs.weight
                this_wtmax = wt.max()

                if this_wtmax > wtmax:
                    wtmax = this_wtmax

                wtsum += wt.sum()

            eff_npix=wtsum/wtmax

            if eff_npix <= 0:
                eff_npix=1.e-6

            self.eff_npix=eff_npix

        return self.eff_npix


class LMSimple(FitterBase):
    """
    A class for doing a fit using levenberg marquardt

    """
    def __init__(self, obs, model, **keys):
        super(LMSimple,self).__init__(obs, model, **keys)

        # this is a dict
        # can contain maxfev (maxiter), ftol (tol in sum of squares)
        # xtol (tol in solution), etc
        self.lm_pars=keys['lm_pars']

        # center1 + center2 + shape1 + shape2 + T + fluxes
        n_prior_pars=1 + 1 + 1 + 1 + 1 + self.nband

        self.fdiff_size=self.totpix + n_prior_pars

        self._band_pars=zeros(6)

    def get_lin_result(self):
        """
        return result in linear space
        """
        if not hasattr(self, '_lin_result'):
            # this will raise an exception of there is not result yet
            res=self.get_result()

            lin_res={}
            lin_res.update(res)
            if res['flags']==0:
                perr_log=res['pars_err']

                pars=res['pars'].copy()
                perr=perr_log.copy()

                pars[4:4+2]=10.0**pars[4:4+2]
                perr[4:4+2] = pars[4:4+2]*perr_log[4:4+2]*log(10)

                lin_res['pars'] = pars
                lin_res['pars_err'] = perr
            else:
                lin_res['pars']=res['pars']*0 + PDEF
                lin_res['pars_err']=res['pars']*0 + CDEF

            self._lin_result=lin_res

        return self._lin_result

    def run_lm(self, guess):
        """
        Run leastsq and set the result
        """

        #if not hasattr(self,'_gmix_all0'):
        #    self._setup_data(guess)
        self._setup_data(guess)

        dof=self.get_dof()
        result = run_leastsq(self._calc_fdiff, guess, dof, **self.lm_pars)

        if result['flags']==0:
            result['g'] = result['pars'][2:2+2].copy()
            result['g_cov'] = result['pars_cov'][2:2+2, 2:2+2].copy()
            stat_dict=self.get_fit_stats(result['pars'])
            result.update(stat_dict)

        self._result=result

    def _setup_data(self, guess):
        """
        try very hard to initialize the mixtures
        """

        if hasattr(self,'_result'):
            del self._result
        if hasattr(self,'_lin_result'):
            del self._lin_result

        self.flags=0

        npars=guess.size
        mess="guess has npars=%d, expected %d" % (npars,self.npars)
        assert (npars==self.npars),mess

        try:
            # this can raise GMixRangeError
            self._init_gmix_all(guess)
        except ZeroDivisionError:
            raise GMixRangeError("got zero division")

    def _get_band_pars(self, log_pars, band):
        """
        Get linear pars for the specified band
        """

        pars=self._band_pars
        _gmix.convert_simple_double_logpars(log_pars, pars, band)
        return pars

    def _get_band_pars_old(self, log_pars, band):
        """
        Get linear pars for the specified band
        """
        pars=log_pars[ [0,1,2,3,4,5+band] ].copy()

        pars[4] = 10.0**pars[4]
        pars[5] = 10.0**pars[5]

        return pars



    def _calc_fdiff(self, pars, get_s2nsums=False):
        """
        vector with (model-data)/error.

        The npars elements contain -ln(prior)
        """

        # we cannot keep sending existing array into leastsq, don't know why
        fdiff=zeros(self.fdiff_size)

        s2n_numer=0.0
        s2n_denom=0.0

        try:

            self._fill_gmix_all(pars)

            start=self._fill_priors(pars, fdiff)

            for band in xrange(self.nband):

                obs_list=self.obs[band]
                gmix_list=self._gmix_all[band]

                for obs,gm in zip(obs_list, gmix_list):

                    res = gm.fill_fdiff(obs, fdiff, start=start)

                    s2n_numer += res[0]
                    s2n_denom += res[1]

                    start += obs.image.size

        except GMixRangeError:
            fdiff[:] = LOWVAL
            s2n_numer=0.0
            s2n_denom=BIGVAL

        if get_s2nsums:
            return fdiff, s2n_numer, s2n_denom
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

        '''
        raise RuntimeError("adapt to new system, also "
                           "should be sqrt(chi2)=sqrt(-2*ln(p))")

        index=0
        fdiff[index] = -self.cen_prior.get_lnprob(pars[0], pars[1])
        index += 1

        if self.g_prior is not None:
            fdiff[index] = -self.g_prior.get_lnprob_scalar2d(pars[2], pars[3])
        index += 1

        fdiff[index] = -self.T_prior.get_lnprob_scalar(pars[4])
        index += 1
        for i,cp in enumerate(self.counts_prior):
            counts=pars[5+i]
            fdiff[index] = -cp.get_lnprob_scalar(counts)
            index += 1

        # this leaves us after the priors
        return index
        '''

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

    def _get_band_pars(self, pars, band):
        if band > 0:
            raise ValueError("support more than one band")
        return pars.copy()


NOTFINITE_BIT=11
def run_leastsq(func, guess, dof, **keys):
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
            s_sq = (fdiff[npars:]**2).sum()/dof
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
        flags += LM_EIG_NOTFINITE 

    return flags

class MCMCBase(FitterBase):
    """
    A base class for MCMC runs.  Inherits from overall fitter base class, which provides
    the get_result() method.

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

        self.trials=None


    def get_trials(self, linear=False):
        """
        Get the set of trials from the production run
        """
        if linear:
            return self.get_lin_trials()
        else:
            if self.trials is None:
                raise RuntimeError("you need to run the mcmc chain first")
            return self.trials

    def get_lin_trials(self):
        """
        Get trials with all parameters in linear space
        """
        if not hasattr(self, 'lin_trials'):
            lin_trials=self.get_trials().copy()
            lin_trials[:,4] = 10.0**lin_trials[:,4]
            lin_trials[:,5:] = 10.0**lin_trials[:,5:]
            self.lin_trials=lin_trials

        return self.lin_trials

    def get_sampler(self):
        """
        get the emcee sampler
        """
        return self.sampler

    def run_mcmc(self, pos, nstep):
        """
        user can run steps
        """

        if not hasattr(self,'sampler'):
            self._setup_sampler_and_data(pos)

        sampler=self.sampler
        sampler.reset()
        self.pos, prob, state = sampler.run_mcmc(pos, nstep)

        lnprobs = sampler.lnprobability.reshape(self.nwalkers*nstep)
        w=lnprobs.argmax()
        bp=lnprobs[w]
        if self.best_lnprob is None or bp > self.best_lnprob:
            self.best_lnprob=bp
            self.best_pars=sampler.flatchain[w,:]

        arates = sampler.acceptance_fraction
        self.arate = arates.mean()

        self.trials  = sampler.flatchain

        return self.pos

    def get_weights(self):
        """
        default weights are none
        """
        return None

    def get_stats(self, linear=False, weights=None):
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
        
        trials=self.get_trials(linear=linear)

        pars,pars_cov = stats.calc_mcmc_stats(trials, weights=weights)
        return pars, pars_cov


    def calc_result(self, weights=None, linear=False):
        """
        Calculate the mcmc stats and the "best fit" stats
        """

        pars,pars_cov = self.get_stats(weights=weights, linear=linear)

        pars_err=sqrt(diag(pars_cov))

        self._set_tau()

        res={'model':self.model_name,
             'flags':self.flags,
             'pars':pars,
             'pars_cov':pars_cov,
             'pars_err':pars_err,
             'tau':self.tau,
             'arate':self.arate}

        if not linear:
            fit_stats = self.get_fit_stats(pars)
            res.update(fit_stats)

        if linear:
            self._lin_result=res
        else:
            self._result=res

    def calc_lin_result(self, weights=None):
        """
        Calculate the mcmc stats and the "best fit" stats
        """
        self.calc_result(weights=weights, linear=True)

    def _setup_sampler_and_data(self, pos):
        """
        try very hard to initialize the mixtures
        """

        self.flags=0
        self.tau=0.0
        self.pos=pos

        npars=pos.shape[1]
        mess="pos has npars=%d, expected %d" % (npars,self.npars)
        assert (npars==self.npars),mess

        self.sampler = self._make_sampler()
        self.best_lnprob=None

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

        if not ok:
            print('failed init gmix from input guess: %s' % str(gerror))
            raise gerror

    def _set_tau(self):
        """
        auto-correlation
        """
        import emcee
        if emcee.ensemble.acor is not None:
            acor=self.sampler.acor
            nstep=self.trials.shape[0]
            tau = (acor/nstep).max()
        else:
            tau=9999.0

        self.tau=tau

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
            
            print('    replacing random state')
            #sampler.random_state=self.random_state.get_state()

            # OK, we will just hope that _random doesn't change names in the future.
            # but at least we get control back
            sampler._random = self.random_state

        return sampler


    def make_plots(self,
                   show=False,
                   prompt=True,
                   do_residual=False,
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

        pdict={}
        pdict['trials']=plotfunc(self.trials,
                                 names=names,
                                 title=title,
                                 show=show,
                                 **keys)


        if weights is not None:
            pdict['wtrials']=plotfunc(self.trials,
                                      weights=weights,
                                      names=names,
                                      title='%s weighted' % title,
                                      show=show)

        if do_residual:
            pdict['resid']=self.plot_residuals(title=title,show=show,
                                               width=width,
                                               height=height)

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

    def calc_result(self, weights=None, linear=False):
        """
        Some extra stats for simple models
        """

        super(MCMCSimple,self).calc_result(weights=weights, linear=linear)

        g1i=self.g1i
        g2i=self.g2i

        if linear:
            self._lin_result['g'] = self._lin_result['pars'][g1i:g1i+2].copy()
            self._lin_result['g_cov'] = self._lin_result['pars_cov'][g1i:g1i+2, g1i:g1i+2].copy()
        else:
            self._result['g'] = self._result['pars'][g1i:g1i+2].copy()
            self._result['g_cov'] = self._result['pars_cov'][g1i:g1i+2, g1i:g1i+2].copy()

    def _get_band_pars(self, log_pars, band):
        """
        Get linear pars for the specified band
        """

        pars=self._band_pars
        _gmix.convert_simple_double_logpars(log_pars, pars, band)
        return pars
 
    def _get_band_pars_old(self, log_pars, band):
        """
        Get linear pars for the specified band
        """
        pars=log_pars[ [0,1,2,3,4,5+band] ].copy()

        pars[4] = 10.0**pars[4]
        pars[5] = 10.0**pars[5]

        return pars

    def get_par_names(self):
        names=['cen1','cen2', 'g1','g2', r'$log_{10}(T)$']
        if self.nband == 1:
            names += [r'$log_{10}(F)$']
        else:
            for band in xrange(self.nband):
                names += [r'$log_{10}(F_%s)$' % band]
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
        self.reset(seed=seed)

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


    def get_stats(self, weights=None):
        """
        get mean and covariance.

        parameters
        ----------
        weights: array
            Extra weights to apply.
        """
        from .stats import calc_mcmc_stats
        stats = calc_mcmc_stats(self._trials, weights=weights)
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

    def reset(self, seed=None):
        """
        Clear all data
        """
        self._trials=None
        self._loglike=None
        self._accepted=None

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
            self._weights = numpy.exp(logdiff*(1.0 - self.Tinv))
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

    def reset(self, seed=None):
        """
        Clear all data
        """
        super(MHTemp,self).reset(seed=seed)
        self._loglike_T=None

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
        """
        FitterBase.__init__(self, obs, model, **keys)

        self.trials=None

        # where g1,g2 are located in a pars array
        self.g1i = 2
        self.g2i = 3

        self._band_pars=zeros(6)

        step_sizes=numpy.asanyarray(step_sizes, dtype='f8')
        ns=step_sizes.size
        mess="step_sizes has size=%d, expected %d" % (ns,self.npars)
        assert (ns == self.npars),mess

        self.step_sizes=step_sizes

        seed=keys.get('seed',None)
        state=keys.get('random_state',None)
        self.set_random_state(seed=seed, state=state)

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

    def run_mcmc(self, pos, nstep):
        """
        user can run steps
        """

        if not hasattr(self,'sampler'):
            self._setup_sampler_and_data(pos)

        sampler=self.sampler
        sampler.reset()
        self.pos = sampler.run_mcmc(pos, nstep)

        trials = sampler.get_trials()
        lnprobs = sampler.get_lnprob()

        w=lnprobs.argmax()
        bp=lnprobs[w]
        if self.best_lnprob is None or bp > self.best_lnprob:
            self.best_lnprob=bp
            self.best_pars=trials[w,:]

        self.arate = sampler.get_arate()

        self.trials  = trials

        return self.pos


    def take_step(self, pos):
        """
        Take gaussian steps
        """
        return pos+self.step_sizes*self.random_state.normal(size=self.npars)

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

        self.sampler = MH(self.calc_lnprob, self.take_step,
                          random_state=self.random_state)
        self.best_lnprob=None

    def _set_tau(self):
        # we don't calculate this currently
        self.tau=9999.0

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
        self.best_lnprob=None


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
        self.tau=0.0
        self.pos=pos
        self.npars=pos.shape[1]

        self.sampler = self._make_sampler()
        self.best_lnprob=None

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
        if self.best_lnprob is None or bp > self.best_lnprob:
            self.best_lnprob=bp
            self.best_pars=sampler.flatchain[w,:]

        arates = sampler.acceptance_fraction
        self.arate = arates.mean()

        self.trials  = sampler.flatchain

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

    def _get_band_pars(self, pars, band):
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

    def _get_band_pars(self, pars, band):
        """
        Extract pars for the specified band and convert to linear
        """
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
        trials=self.trials
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
        logpars=self._result['pars']
        pars=logpars.copy()
        pars[4] = 10.0**logpars[4]
        pars[5] = 10.0**logpars[5]
        pars[6] = 10.0**logpars[6]

        gm=gmix.make_gmix_model(pars, self.model)
        return gm


    def _get_g_prior_vals(self):
        if not hasattr(self,'joint_prior_vals'):
            trials=self.trials
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

    def _get_band_pars(self, pars, band):
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


    def _get_band_pars(self, pars, band):
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

    def _get_band_pars(self, pars, band):
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

    def _get_band_pars(self, pars, band):
        """
        pars are 
            [c1,c2,g1,g2,Tb,Td, Fb1,Fb2,Fb3, ..., Fd1,Fd2,Fd3 ...]
        """
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

    def _get_band_pars(self, pars, band):
        """
        pars are 
            [c1,c2,g1,g2,T, Fb1,Fb2,Fb3, ..., Fd1,Fd2,Fd3 ...]
        """
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
            Pi,Qi,Ri = self.joint_prior.get_pqr_num(self.trials[:, 2:])
        else:
            Pi,Qi,Ri = self.joint_prior.get_pqr_num(self.trials[:, 2:],
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

        self.tau=9999.0

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

        self.trials  = sampler.flatchain
        self.joint_prior_vals = self.joint_prior.get_prob_array(self.trials[:,2:], throw=False)

        arates = sampler.acceptance_fraction
        self.arate = arates.mean()

        lnprobs = sampler.lnprobability.reshape(self.nwalkers*self.nstep)
        w=lnprobs.argmax()
        bp=lnprobs[w]

        self.best_lnprob=bp
        self.best_pars=sampler.flatchain[w,:]

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

    def _get_band_pars(self, pars, band):
        """
        Extract pars for the specified band and convert to linear
        """
        from .shape import eta1eta2_to_g1g2
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
        trials=self.trials
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
        logpars=self._result['pars']
        pars=logpars.copy()
        pars[4] = 10.0**logpars[4]
        pars[5] = 10.0**logpars[5]

        gm=gmix.make_gmix_model(pars, self.model)
        return gm


    def _get_g_prior_vals(self):
        if not hasattr(self,'joint_prior_vals'):
            trials=self.trials
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

    def _get_band_pars(self, pars, band):
        """
        Extract pars for the specified band and convert to linear
        """
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
            Pi,Qi,Ri = self.joint_prior.get_pqr_num(self.trials[:,2:])
        else:
            Pi,Qi,Ri = self.joint_prior.get_pqr_num(self.trials[:,2:],
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
            eabs_pars=self._get_eabs_pars(self.trials)
            self.joint_prior_vals = self.joint_prior.get_prob_array(eabs_pars)
        return self.joint_prior_vals


class MCMCSimpleJointLogPars(MCMCSimple):
    """
    Simple with a joint prior on [g1,g2,T,Fb,Fd]
    """
    def __init__(self, image, weight, jacobian, model, **keys):
        raise RuntimeError("adapt to new system")
        super(MCMCSimpleJointLogPars,self).__init__(image, weight, jacobian, model, **keys)

        if self.full_guess is None:
            raise ValueError("For joint simple you must currently send a full guess")

        # we alrady have T_prior and counts_prior from base class

        # fraction of flux in bulge
        if self.joint_prior is None:
            raise ValueError("send joint prior for MCMCSimpleJointLogPars")

        self.prior_during=keys['prior_during']

    def _get_band_pars(self, pars, band):
        """
        Extract pars for the specified band and convert to linear
        """
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
            Pi,Qi,Ri = self.joint_prior.get_pqr_num(self.trials[:, 2:])
        else:
            Pi,Qi,Ri = self.joint_prior.get_pqr_num(self.trials[:, 2:],
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
            self.joint_prior_vals = self.joint_prior.get_prob_array(self.trials[:,2:])
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




def print_pars(pars, stream=stdout, fmt='%10.6g',front=None):
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


def _get_as_list(arg, argname, allow_none=False):
    if arg is None:
        if allow_none:
            return None
        else:
            raise ValueError("None not allowed for %s" % argname)

    if isinstance(arg,list):
        return arg
    else:
        return [arg]



def test_gauss_psf_graph(counts=100.0, noise=0.1, nimages=10, n=10, groups=True, jfac=0.27):
    import pylab
    import cProfile

    import pycallgraph
    from pycallgraph import PyCallGraph
    from pycallgraph.output import GraphvizOutput

    graphviz = GraphvizOutput()
    output='profile-nimages%02d.png' % nimages
    print('profile image:',output)
    graphviz.output_file = output
    config=pycallgraph.Config(groups=groups)

    dims=[25,25]
    cen=[dims[0]/2., dims[1]/2.]

    g1=0.1
    g2=0.05
    T=8.0

    pars = [cen[0],cen[1], g1, g2, T, counts]
    gm=gmix.GMixModel(pars, "gauss")

    im=gm.make_image(dims)

    im[:,:] += noise*numpy.random.randn(im.size).reshape(im.shape)
    wt=zeros(im.shape) + 1./noise**2
    j=Jacobian(cen[0], cen[1], jfac, 0.0, 0.0, jfac)

    imlist=[im]*nimages
    wtlist=[wt]*nimages
    jlist=[j]*nimages

    # one run to warm up the jit compiler
    mc=MCMCGaussPSF(im, wt, j)
    mc.go()

    with PyCallGraph(config=config, output=graphviz):
        for i in xrange(n):
            mc=MCMCGaussPSF(imlist, wtlist, jlist)
            mc.go()

            res=mc.get_result()

            print(res['g'])

def test_gauss_psf(counts=100.0, noise=0.001, n=10, nimages=10, jfac=0.27):
    """
    timing tests
    """
    import pylab
    import time


    dims=[25,25]
    cen=[dims[0]/2., dims[1]/2.]

    g1=0.1
    g2=0.05
    T=8.0

    pars = [cen[0],cen[1], g1, g2, T, counts]
    gm=gmix.GMixModel(pars, "gauss")

    im=gm.make_image(dims)

    im[:,:] += noise*numpy.random.randn(im.size).reshape(im.shape)
    wt=zeros(im.shape) + 1./noise**2
    j=Jacobian(cen[0], cen[1], jfac, 0.0, 0.0, jfac)

    imlist=[im]*nimages
    wtlist=[wt]*nimages
    jlist=[j]*nimages

    # one run to warm up the jit compiler
    mc=MCMCGaussPSF(im, wt, j)
    mc.go()

    t0=time.time()
    for i in xrange(n):
        mc=MCMCGaussPSF(imlist, wtlist, jlist)
        mc.go()

        res=mc.get_result()

        print_pars(res['pars'],front='pars:')

    sec=time.time()-t0
    secper=sec/n
    print(secper,'seconds per')

    return sec,secper

def test_gauss_psf_jacob(counts_sky=100.0, noise_sky=0.001, nimages=10, jfac=10.0):
    """
    testing jacobian stuff
    """
    import images
    import mcmc
    dims=[25,25]
    cen=[dims[0]/2., dims[1]/2.]

    j=Jacobian(cen[0],cen[1], jfac, 0.0, 0.0, jfac)

    g1=0.1
    g2=0.05
    # in pixel coords
    Tpix=8.0
    Tsky=8.0*jfac**2
    counts_pix=counts_sky/jfac**2
    noise_pix=noise_sky/jfac**2


    pars = [0.0, 0.0, g1, g2, Tsky, counts_sky]
    gm=gmix.GMixModel(pars, "gauss")

    im=gm.make_image(dims, jacobian=j)

    im[:,:] += noise_pix*numpy.random.randn(im.size).reshape(im.shape)
    wt=zeros(im.shape) + 1./noise_pix**2

    imlist=[im]*nimages
    wtlist=[wt]*nimages
    jlist=[j]*nimages


    mc=MCMCGaussPSF(imlist, wtlist, jlist, T=Tsky, counts=counts_sky, burnin=400)
    mc.go()

    res=mc.get_result()

    print_pars(res['pars'], front='pars:')
    print_pars(res['pars_err'], front='perr:')

    gmfit=mc.get_gmix()
    imfit=gmfit.make_image(im.shape, jacobian=j)

    imdiff=im-imfit
    images.compare_images(im, imfit, label1='data',label2='fit')
    
    mcmc.plot_results(mc.get_trials())

def test_model(model, counts_sky=100.0, noise_sky=0.001, nimages=1, jfac=0.27, g_prior=None, show=False):
    """
    Test fitting the specified model.

    Send g_prior to do some lensfit/pqr calculations
    """
    import mcmc
    from . import em


    nwalkers=80
    burnin=800
    nstep=800

    dims=[25,25]
    cen=[dims[0]/2., dims[1]/2.]

    jfac2=jfac**2
    j=Jacobian(cen[0],cen[1], jfac, 0.0, 0.0, jfac)

    #
    # simulation
    #

    # PSF pars
    counts_sky_psf=100.0
    counts_pix_psf=counts_sky_psf/jfac2
    noise_sky_psf=0.01
    noise_pix_psf=noise_sky_psf/jfac2
    g1_psf=0.05
    g2_psf=-0.01
    Tpix_psf=4.0
    Tsky_psf=Tpix_psf*jfac2

    # object pars
    g1_obj=0.1
    g2_obj=0.05
    Tpix_obj=16.0
    Tsky_obj=Tpix_obj*jfac2

    counts_sky_obj=counts_sky
    noise_sky_obj=noise_sky
    counts_pix_obj=counts_sky_obj/jfac2
    noise_pix_obj=noise_sky_obj/jfac2

    pars_psf = [0.0, 0.0, g1_psf, g2_psf, Tsky_psf, counts_sky_psf]
    gm_psf=gmix.GMixModel(pars_psf, "gauss")

    pars_obj = array([0.0, 0.0, g1_obj, g2_obj, Tsky_obj, counts_sky_obj])
    npars=pars_obj.size
    gm_obj0=gmix.GMixModel(pars_obj, model)

    gm=gm_obj0.convolve(gm_psf)

    im_psf=gm_psf.make_image(dims, jacobian=j)
    im_psf[:,:] += noise_pix_psf*numpy.random.randn(im_psf.size).reshape(im_psf.shape)
    wt_psf=zeros(im_psf.shape) + 1./noise_pix_psf**2

    im_obj=gm.make_image(dims, jacobian=j)
    im_obj[:,:] += noise_pix_obj*numpy.random.randn(im_obj.size).reshape(im_obj.shape)
    wt_obj=zeros(im_obj.shape) + 1./noise_pix_obj**2

    #
    # fitting
    #


    # psf using EM
    im_psf_sky,sky=em.prep_image(im_psf)
    psf_obs = Observation(im_psf_sky, jacobian=j)
    mc_psf=em.GMixEM(psf_obs)

    emo_guess=gm_psf.copy()
    emo_guess._data['p'] = 1.0
    emo_guess._data['row'] += 0.1*srandu()
    emo_guess._data['col'] += 0.1*srandu()
    emo_guess._data['irr'] += 0.5*srandu()
    emo_guess._data['irc'] += 0.1*srandu()
    emo_guess._data['icc'] += 0.5*srandu()

    mc_psf.go(emo_guess, sky)
    res_psf=mc_psf.get_result()
    print('psf numiter:',res_psf['numiter'],'fdiff:',res_psf['fdiff'])

    psf_fit=mc_psf.get_gmix()


    obs=Observation(im_obj, weight=wt_obj, jacobian=j, psf_gmix=psf_fit)
    mc_obj=MCMCSimple(obs, model, nwalkers=nwalkers)

    guess=zeros( (nwalkers, npars) )
    guess[:,0] = 0.1*srandu(nwalkers)
    guess[:,1] = 0.1*srandu(nwalkers)

    # intentionally bad guesses
    guess[:,2] = 0.1*srandu(nwalkers)
    guess[:,3] = 0.1*srandu(nwalkers)
    guess[:,4] = 0.5*Tsky_obj*(1.0 + 0.1*srandu(nwalkers))
    guess[:,5] = 2.0*counts_sky_obj*(1.0 + 0.1*srandu(nwalkers))

    pos=mc_obj.run_mcmc(guess, burnin)
    pos=mc_obj.run_mcmc(pos, nstep)
    mc_obj.calc_result()

    res_obj=mc_obj.get_result()

    print_pars(pars_obj,            front='true pars:')
    print_pars(res_obj['pars'],     front='pars_obj: ')
    print_pars(res_obj['pars_err'], front='perr_obj: ')
    print('Tpix: %.4g +/- %.4g' % (res_obj['pars'][4]/jfac2, res_obj['pars_err'][4]/jfac2))
    print("s2n:",res_obj['s2n_w'],"arate:",res_obj['arate'],"tau:",res_obj['tau'])

    gmfit0=mc_obj.get_gmix()
    gmfit=gmfit0.convolve(psf_fit)

    if g_prior is not None:
        from . import lensfit
        trials = mc_obj.get_trials()
        gsens = lensfit.calc_sensitivity(trials[:,2:2+2], g_prior)
        print("lensfit gsens:",gsens)

    if show:
        import images
        imfit_psf=mc_psf.make_image(counts=im_psf.sum())
        images.compare_images(im_psf, imfit_psf, label1='psf',label2='fit')

        mc_obj.make_plots(do_residual=True,show=True,prompt=False)
        #imfit_obj=gmfit.make_image(im_obj.shape, jacobian=j)
        #images.compare_images(im_obj, imfit_obj, label1=model,label2='fit')
        #mcmc.plot_results(mc_obj.get_trials())


def get_mh_prior(T, F):
    from . import priors, joint_prior
    cen_prior=priors.CenPrior(0.0, 0.0, 0.5, 0.5)
    g_prior = priors.make_gprior_cosmos_sersic(type='erf')
    #g_prior_flat = priors.UDisk2DCut(cutval=0.97)
    #g_prior_flat = priors.ZDisk2DErf(1.0, width=0.001)
    #g_prior_flat = priors.ZDisk2DErf(1.0, width=0.001)
    g_prior_flat = priors.ZDisk2D(1.0)

    Twidth=0.3*T
    logTmean, logTsigma=priors.lognorm_convert(T, Twidth, base=10.0)
    T_prior = priors.Normal(logTmean, logTsigma)

    Fwidth=0.3*T
    logFmean, logFsigma=priors.lognorm_convert(F, Fwidth, base=10.0)
    F_prior = priors.Normal(logFmean, logFsigma)

    prior = joint_prior.PriorSimpleSep(cen_prior, g_prior, T_prior, F_prior)

    prior_gflat = joint_prior.PriorSimpleSep(cen_prior, g_prior_flat,
                                             T_prior, F_prior)

    return prior, prior_gflat

def test_model_mh(model,
                  burnin=5000,
                  nstep=5000,
                  noise_obj=0.01,
                  show=False,
                  temp=None):
    """
    Test fitting the specified model.

    Send g_prior to do some lensfit/pqr calculations
    """
    import mcmc
    from . import em



    dims=[25,25]
    cen=[dims[0]/2., dims[1]/2.]

    jacob=UnitJacobian(cen[0],cen[1])

    #
    # simulation
    #

    # PSF pars
    counts_psf=100.0
    noise_psf=0.01
    g1_psf=0.05
    g2_psf=-0.01
    T_psf=4.0

    # object pars
    counts_obj=100.0
    T_obj=16.0

    pars_psf = [0.0, 0.0, g1_psf, g2_psf, T_psf, counts_psf]
    gm_psf=gmix.GMixModel(pars_psf, "gauss")

    prior,prior_gflat=get_mh_prior(T_obj, counts_obj)

    pars_obj = prior.sample()

    #g1_obj, g2_obj = prior.g_prior.sample2d(1)
    #g1_obj=g1_obj[0]
    #g2_obj=g2_obj[0]

    #pars_obj = array([0.0, 0.0, g1_obj, g2_obj, T_obj, counts_obj])

    npars=pars_obj.size
    gm_obj0=gmix.GMixModel(pars_obj, model)

    gm=gm_obj0.convolve(gm_psf)

    im_psf=gm_psf.make_image(dims, jacobian=jacob)
    im_psf[:,:] += noise_psf*numpy.random.randn(im_psf.size).reshape(im_psf.shape)
    wt_psf=zeros(im_psf.shape) + 1./noise_psf**2

    im_obj=gm.make_image(dims, jacobian=jacob)
    im_obj[:,:] += noise_obj*numpy.random.randn(im_obj.size).reshape(im_obj.shape)
    wt_obj=zeros(im_obj.shape) + 1./noise_obj**2

    #
    # fitting
    #


    # psf using EM
    im_psf_sky,sky=em.prep_image(im_psf)
    psf_obs = Observation(im_psf_sky, jacobian=jacob)
    mc_psf=em.GMixEM(psf_obs)

    emo_guess=gm_psf.copy()
    emo_guess._data['p'] = 1.0
    emo_guess._data['row'] += 0.01*srandu()
    emo_guess._data['col'] += 0.01*srandu()
    emo_guess._data['irr'] += 0.01*srandu()
    emo_guess._data['irc'] += 0.01*srandu()
    emo_guess._data['icc'] += 0.01*srandu()

    mc_psf.go(emo_guess, sky)
    res_psf=mc_psf.get_result()
    print('psf numiter:',res_psf['numiter'],'fdiff:',res_psf['fdiff'])

    psf_fit=mc_psf.get_gmix()
    print("psf gmix:")
    print(psf_fit)
    print()

    # first fit with LM
    psf_obs.set_gmix(psf_fit)
    obs=Observation(im_obj, jacobian=jacob, weight=wt_obj, psf=psf_obs)

    lm_pars={'maxfev': 300,
             'ftol':   1.0e-6,
             'xtol':   1.0e-6,
             'epsfcn': 1.0e-6}

    lm_fitter=LMSimple(obs, model, lm_pars=lm_pars, prior=prior)

    # must be log!
    #guess=pars_obj.copy()
    #guess[4]=log10(guess[4])
    #guess[5]=log10(guess[5])
    guess=prior.sample()

    lm_fitter.run_lm(guess)
    lm_res_log=lm_fitter.get_result()
    lm_res_lin=lm_fitter.get_lin_result()

    mh_guess=lm_res_log['pars'].copy()
    step_sizes = 0.5*lm_res_log['pars_err'].copy()

    print_pars(lm_res_log['pars'], front="log lm result:")
    print_pars(lm_res_log['pars_err'], front="log lm err:   ")
    print_pars(lm_res_lin['pars'], front="lm result:")
    print_pars(lm_res_lin['pars_err'], front="lm err:   ")
    print()

    if temp is not None:
        print("doing temperature:",temp)
        step_sizes *= sqrt(temp)
        mh_fitter=MHTempSimple(obs, model, step_sizes,
                               temp=temp, prior=prior)
    else:
        mh_fitter=MHSimple(obs, model, step_sizes, prior=prior)

    pos=mh_fitter.run_mcmc(mh_guess, burnin)
    pos=mh_fitter.run_mcmc(pos, nstep)
    mh_fitter.calc_result()
    mh_fitter.calc_lin_result()

    res_obj=mh_fitter.get_result()
    res_obj_lin=mh_fitter.get_lin_result()

    print_pars(res_obj['pars'],     front='log pars_obj: ')
    print_pars(res_obj['pars_err'], front='log perr_obj: ')
    print_pars(pars_obj,            front='true pars:')
    print_pars(res_obj_lin['pars'],     front='pars_obj: ')
    print_pars(res_obj_lin['pars_err'], front='perr_obj: ')

    print('T: %.4g +/- %.4g' % (res_obj_lin['pars'][4], res_obj_lin['pars_err'][4]))
    print("arate:",res_obj['arate'],"s2n:",res_obj['s2n_w'],"tau:",res_obj['tau'])

    gmfit0=mh_fitter.get_gmix()
    gmfit=gmfit0.convolve(psf_fit)

    if show:
        import images
        imfit_psf=mc_psf.make_image(counts=im_psf.sum())
        images.compare_images(im_psf, imfit_psf, label1='psf',label2='fit')

        mh_fitter.make_plots(do_residual=True,show=True,prompt=False)

def test_many_model_coellip(ntrial,
                            model,
                            ngauss,
                            **keys):
    import time

    tm0=time.time()
    g1fit=zeros(ntrial)
    g2fit=zeros(ntrial)

    for i in xrange(ntrial):
        print("-"*40)
        print("%d/%d" % (i+1,ntrial))

        true_pars, fit_pars= test_model_coellip(model,
                                                ngauss,
                                                **keys)
        g1fit[i] = fit_pars[2]
        g2fit[i] = fit_pars[3]

        print(g1fit[i],g2fit[i])

    frac1_arr=g1fit/true_pars[2]-1
    frac2_arr=g2fit/true_pars[3]-1

    frac1 = frac1_arr.mean()
    frac1_err = frac1_arr.std()/sqrt(ntrial)
    frac2 = frac2_arr.mean()
    frac2_err = frac2_arr.std()/sqrt(ntrial)

    print("-"*40)
    print("%g +/- %g" % (frac1, frac1_err))
    print("%g +/- %g" % (frac2, frac2_err))

    tm=time.time()-tm0
    print("time per:",tm/ntrial)

def test_model_coellip(model, ngauss,
                       counts=100.0, noise=0.00001,
                       nwalkers=320,
                       g1=0.1, g2=0.1,
                       burnin=800,
                       nstep=800,
                       doplots=False):
    """
    fit an n gauss coellip model to a different model

    parameters
    ----------
    model:
        the true model
    ngauss:
        number of gaussians to fit to the true model
    """
    import images
    import mcmc
    from . import em

    #
    # simulation
    #

    # PSF pars
    counts_psf=100.0
    noise_psf=0.001
    g1_psf=0.00
    g2_psf=0.00
    T_psf=4.0

    # object pars
    counts_obj=counts
    g1_obj=g1
    g2_obj=g2
    if model=='exp':
        T_obj=16.0
    elif model=='dev':
        T_obj=64.0

    sigma=sqrt(T_obj/2.0)
    dim=int(round(5*sigma*2))
    dims=[dim]*2
    cen=[dims[0]/2., dims[1]/2.]
    print("dim: %g" % dims[0])

    jacob=UnitJacobian(cen[0],cen[1])

    pars_psf = [0.0, 0.0, g1_psf, g2_psf, T_psf, counts_psf]
    gm_psf=gmix.make_gmix_model(pars_psf, "gauss")

    pars_obj = [0.0, 0.0, g1_obj, g2_obj, T_obj, counts]
    gm_obj0=gmix.make_gmix_model(pars_obj, model)

    gm=gm_obj0.convolve(gm_psf)

    im_psf=gm_psf.make_image(dims, jacobian=jacob)
    im_psf[:,:] += noise_psf*numpy.random.randn(im_psf.size).reshape(im_psf.shape)
    wt_psf=zeros(im_psf.shape) + 1./noise_psf**2

    im_obj=gm.make_image(dims, jacobian=jacob)
    im_obj[:,:] += noise*numpy.random.randn(im_obj.size).reshape(im_obj.shape)
    wt_obj=zeros(im_obj.shape) + 1./noise**2

    #
    # fitting
    #

    # psf using EM
    im_psf_sky,sky=em.prep_image(im_psf)
    mc_psf=em.GMixEM(im_psf_sky, jacobian=jacob)
    emo_guess=gm_psf.copy()
    emo_guess._data['p'] = 1.0
    emo_guess._data['row'] += 0.1*srandu()
    emo_guess._data['col'] += 0.1*srandu()
    emo_guess._data['irr'] += 0.5*srandu()
    emo_guess._data['irc'] += 0.1*srandu()
    emo_guess._data['icc'] += 0.5*srandu()
    mc_psf.go(emo_guess, sky, maxiter=5000)
    res_psf=mc_psf.get_result()
    print('psf numiter:',res_psf['numiter'],'fdiff:',res_psf['fdiff'])

    psf_fit=mc_psf.get_gmix()
    imfit_psf=mc_psf.make_image(counts=im_psf.sum())
    #images.compare_images(im_psf, imfit_psf, label1='psf',label2='fit')

    g1_guess=0.0
    g2_guess=0.0
    full_guess=test_guess_coellip(nwalkers, ngauss,
                                  g1_guess, g2_guess, T_obj, counts_obj)

    cen_prior=priors.CenPrior(0.0, 0.0, 0.1, 0.1)
    priors_are_log=False
    if priors_are_log:
        counts_prior=priors.FlatPrior(log10(0.5*counts_obj),
                                      log10(2.0*counts_obj) )
        T_prior=priors.FlatPrior(log10(0.5*T_obj),
                                 log10(2.0*T_obj) )
    else:
        counts_prior=priors.FlatPrior(0.5*counts_obj, 2.0*counts_obj)
        T_prior=priors.FlatPrior(0.5*T_obj, 2.0*T_obj )

    mc_obj=MCMCCoellip(im_obj, wt_obj, jacob,
                       psf=psf_fit,
                       nwalkers=nwalkers,
                       burnin=burnin,
                       nstep=nstep,
                       priors_are_log=priors_are_log,
                       counts_prior=counts_prior,
                       cen_prior=cen_prior,
                       T_prior=T_prior,
                       full_guess=full_guess)
    mc_obj.go()


    res=mc_obj.get_result()
    if doplots:
        mc_obj.make_plots(show=True, do_residual=True,
                          width=1100,height=750,
                          separate=True)

    res_obj=mc_obj.get_result()
    gm=mc_obj.get_gmix()

    pars=res_obj['pars']
    perr=res_obj['pars_err']

    trials=mc_obj.get_trials()

    Ttrials = trials[:,4:4+ngauss]
    Ftrials = trials[:,4+ngauss:]

    Ftot = Ftrials.sum(axis=1)
    Ttot = (Ttrials*Ftrials).sum(axis=1)/Ftot

    Fmeas = Ftot.mean()
    Ferr = Ftot.std()

    Tmeas = Ttot.mean()
    Terr = Ttot.std()

    print("true T:",T_obj,"F:",counts_obj)
    print("s2n_w:",res["s2n_w"])
    print("arate:",res['arate'])
    print('Tgmix: %g Fluxgmix: %g' % (gm.get_T(),gm.get_psum()) )
    print('Tmeas: %g +/- %g Fluxmeas: %g +/- %g' % (Tmeas,Terr,Fmeas,Ferr))
    print_pars(res_obj['pars'], front='pars_obj:')
    print_pars(res_obj['pars_err'], front='perr_obj:')

    return pars_obj, res_obj['pars']

def test_guess_coellip(nwalkers, ngauss,
                       g1_obj, g2_obj, T_obj, counts_obj):
    npars=gmix.get_coellip_npars(ngauss)
    full_guess=zeros( (nwalkers, npars) )
    full_guess[:,0] = 0.1*srandu(nwalkers)
    full_guess[:,1] = 0.1*srandu(nwalkers)
    full_guess[:,2] = g1_obj + 0.1*srandu(nwalkers)
    full_guess[:,3] = g2_obj + 0.1*srandu(nwalkers)

    if ngauss==3:
        for i in xrange(ngauss):
            if i==0:
                full_guess[:,4+i] = 0.1*T_obj*(1.0 + 0.01*srandu(nwalkers))
                full_guess[:,4+ngauss+i] = 0.1*counts_obj*(1.0 + 0.01*srandu(nwalkers))
            elif i==1:
                full_guess[:,4+i] = 1.0*T_obj*(1.0 + 0.01*srandu(nwalkers))
                full_guess[:,4+ngauss+i] = 0.5*counts_obj*(1.0 + 0.01*srandu(nwalkers))
            elif i==2:
                full_guess[:,4+i] = 2.0*T_obj*(1.0 + 0.01*srandu(nwalkers))
                full_guess[:,4+ngauss+i] = 0.4*counts_obj*(1.0 + 0.01*srandu(nwalkers))
    elif ngauss==4:
        # implement this
        # 0.710759     3.66662     22.9798     173.704
        # 19.6636     18.3341     31.3521     29.5486
        # nromalized
        pars0=array([0.01183116, 0.06115546,  0.3829298 ,  2.89446939,
                     0.19880675,  0.18535747, 0.31701891,  0.29881687])
        #pars0=array([1.0e-6, 0.06115546,  0.3829298 ,  2.89446939,
        #             0.19880675,  0.18535747, 0.31701891,  0.29881687])

        for i in xrange(ngauss):
            full_guess[:,4+i] = T_obj*pars0[i]*(1.0 + 0.01*srandu(nwalkers))
            full_guess[:,4+ngauss+i] = counts_obj*pars0[ngauss+i]*(1.0 + 0.01*srandu(nwalkers))

            """
            if i==0:
                full_guess[:,4+i] = 0.01*T_obj*(1.0 + 0.01*srandu(nwalkers))
                full_guess[:,4+ngauss+i] = 0.1*counts_obj*(1.0 + 0.01*srandu(nwalkers))
            elif i==1:
                full_guess[:,4+i] = 0.1*T_obj*(1.0 + 0.01*srandu(nwalkers))
                full_guess[:,4+ngauss+i] = 0.2*counts_obj*(1.0 + 0.01*srandu(nwalkers))
            elif i==2:
                full_guess[:,4+i] = 1.0*T_obj*(1.0 + 0.01*srandu(nwalkers))
                full_guess[:,4+ngauss+i] = 0.5*counts_obj*(1.0 + 0.01*srandu(nwalkers))
            elif i==3:
                full_guess[:,4+i] = 2.0*T_obj*(1.0 + 0.01*srandu(nwalkers))
                full_guess[:,4+ngauss+i] = 0.2*counts_obj*(1.0 + 0.01*srandu(nwalkers))
            """
    else:
        raise ValueError("try other ngauss")

    #full_guess[:, 4:] = log10( full_guess[:,4:] )
    return full_guess



def make_sersic_images(model, hlr, flux, n, noise, g1, g2):
    import galsim 

    psf_sigma=1.414
    pixel_scale=1.0

    gal = galsim.Sersic(n,
                        half_light_radius=hlr,
                        flux=flux)
    gal.applyShear(g1=g1, g2=g2)

    psf = galsim.Gaussian(sigma=psf_sigma, flux=1.0)
    pixel=galsim.Pixel(pixel_scale)

    gal_final = galsim.Convolve([gal, psf, pixel])
    psf_final = galsim.Convolve([psf, pixel])

    # deal with massive unannounced api changes
    try:
        image_obj = gal_final.draw(scale=pixel_scale)
        psf_obj   = psf_final.draw(scale=pixel_scale)
    except:
        image_obj = gal_final.draw(dx=pixel_scale)
        psf_obj   = psf_final.draw(dx=pixel_scale)

    image_obj.addNoise(galsim.GaussianNoise(sigma=noise))

    image = image_obj.array.astype('f8')

    psf_image = psf_obj.array.astype('f8')

    wt = image*0 + ( 1.0/noise**2 )

    print("image dims:",image.shape)
    print("image sum:",image.sum())
    return image, wt, psf_image

def profile_sersic(model, **keys):
    import cProfile
    import pstats

    cProfile.runctx('test_sersic(model, **keys)',
                    globals(),locals(),
                    'profile_stats')
    p = pstats.Stats('profile_stats')
    p.sort_stats('time').print_stats()


def test_sersic(model,
                n=None, # only needed if model is 'sersic'
                hlr=2.0,
                counts=100.0,
                noise=0.00001,
                nwalkers=80,
                g1=0.1, g2=0.1,
                burnin=400,
                nstep=800,
                ntry=1,
                doplots=False):
    """
    fit an n gauss coellip model to a different model

    parameters
    ----------
    model:
        the true model
    n: optional
        if true model is sersic, send n
    """
    import images
    from . import em
    from . import gmix

    if model != 'sersic':
        if model=='exp':
            n=1.0
        elif model=='dev':
            n=4.0
        else:
            raise ValueError("bad model: '%s'" % model)
    #
    # simulation
    #

    # PSF pars
    sigma_psf=sqrt(2)

    im, wt, im_psf=make_sersic_images(model, hlr, counts, n, noise, g1, g2)
    cen=(im.shape[0]-1)/2.
    psf_cen=(im_psf.shape[0]-1)/2.

    jacob=UnitJacobian(cen,cen)
    psf_jacob=UnitJacobian(psf_cen,psf_cen)

    #
    # fitting
    #

    # psf using EM
    im_psf_sky,sky=em.prep_image(im_psf)
    mc_psf=em.GMixEM(im_psf_sky, jacobian=psf_jacob)

    psf_pars_guess=[1.0,
                    0.01*srandu(),
                    0.01*srandu(),
                    sigma_psf**2,
                    0.01*srandu(),
                    sigma_psf**2]
    emo_guess=gmix.GMix(pars=psf_pars_guess)

    mc_psf.go(emo_guess, sky, maxiter=5000)
    res_psf=mc_psf.get_result()
    print('psf numiter:',res_psf['numiter'],'fdiff:',res_psf['fdiff'])

    psf_fit=mc_psf.get_gmix()
    print("psf gmix:")
    print(psf_fit)

    # terrible
    T_guess=2*hlr**2

    cen_prior=priors.CenPrior(0.0, 0.0, 0.1, 0.1)

    counts_prior=priors.FlatPrior(0.01*counts, 100*counts)
    T_prior=priors.FlatPrior(0.01*T_guess, 100*T_guess)

    nmin = gmix.MIN_SERSIC_N
    nmax = gmix.MAX_SERSIC_N

    n_prior=priors.FlatPrior(nmin, nmax)
    #n_prior=priors.TruncatedGaussian(n, 0.001, nmin, nmax)

    for i in xrange(ntry):
        print("try: %s/%s" % (i+1,ntry))
        if i==0:
            full_guess=test_guess_sersic(nwalkers, T_guess, counts)
        else:
            best_pars=mc_obj.best_pars
            print_pars(best_pars,front="best pars: ")
            full_guess=test_guess_sersic_from_pars(nwalkers,best_pars)

        mc_obj=MCMCSersic(im, wt, jacob,
                          psf=psf_fit,
                          nwalkers=nwalkers,
                          counts_prior=counts_prior,
                          cen_prior=cen_prior,
                          n_prior=n_prior,
                          T_prior=T_prior,
                          full_guess=full_guess)
        pars=mc_obj.run_mcmc(full_guess, burnin)
        pars=mc_obj.run_mcmc(pars, nstep)

    mc_obj.calc_result()

    res=mc_obj.get_result()
    gm=mc_obj.get_gmix()
    gmc=gm.convolve(psf_fit)

    if doplots:
        mc_obj.make_plots(show=True, prompt=False,
                          width=1100,height=750,
                          separate=True)
        model_im=gmc.make_image(im.shape, jacobian=jacob)
        images.compare_images(im, model_im) 

    res=mc_obj.get_result()

    print('arate:',res['arate'])
    print_pars(res['pars'],     front='pars:')
    print_pars(res['pars_err'], front='perr:')

    return res['pars']


def test_guess_sersic(nwalkers, T, counts):
    from numpy.random import random as randu
    from . import gmix

    full_guess=zeros( (nwalkers, 7) )
    full_guess[:,0] = 0.1*srandu(nwalkers)
    full_guess[:,1] = 0.1*srandu(nwalkers)
    full_guess[:,2] = 0.1*srandu(nwalkers)
    full_guess[:,3] = 0.1*srandu(nwalkers)

    full_guess[:,4] = T*(1.0 + 0.2*srandu(nwalkers))
    full_guess[:,5] = counts*(1.0 + 0.2*srandu(nwalkers))

    nmin = gmix.MIN_SERSIC_N
    nmax = gmix.MAX_SERSIC_N

    full_guess[:,6] = nmin + (nmax-nmin)*randu(nwalkers)

    return full_guess

def test_guess_sersic_from_pars(nwalkers, pars):
    from numpy.random import random as randu
    from . import gmix

    full_guess=zeros( (nwalkers, 7) )
    full_guess[:,0] = pars[0] + 0.01*srandu(nwalkers)
    full_guess[:,1] = pars[1] + 0.01*srandu(nwalkers)
    full_guess[:,2] = pars[2] + 0.01*srandu(nwalkers)
    full_guess[:,3] = pars[3] + 0.01*srandu(nwalkers)

    full_guess[:,4] = pars[4]*(1.0 + 0.01*srandu(nwalkers))
    full_guess[:,5] = pars[5]*(1.0 + 0.01*srandu(nwalkers))

    nmin = gmix.MIN_SERSIC_N
    nmax = gmix.MAX_SERSIC_N

    nleft=nwalkers
    ngood=0
    while nleft > 0:
        vals = pars[6]*(1.0 + 0.01*srandu(nleft))
        w,=numpy.where( (vals > nmin) & (vals < nmax) )
        nkeep=w.size
        if nkeep > 0:
            full_guess[ngood:ngood+nkeep,6] = vals[w]
            nleft -= w.size
            ngood += w.size

    return full_guess


def test_model_priors(model,
                      counts_sky=100.0,
                      noise_sky=0.01,
                      nimages=1,
                      jfac=0.27,
                      do_lensfit=False,
                      do_pqr=False):
    """
    testing jacobian stuff
    """
    import images
    import mcmc
    from . import em

    dims=[25,25]
    cen=[dims[0]/2., dims[1]/2.]

    jfac2=jfac**2
    j=Jacobian(cen[0],cen[1], jfac, 0.0, 0.0, jfac)

    #
    # simulation
    #

    # PSF pars
    counts_sky_psf=100.0
    counts_pix_psf=counts_sky_psf/jfac2
    g1_psf=0.05
    g2_psf=-0.01
    Tpix_psf=4.0
    Tsky_psf=Tpix_psf*jfac2

    # object pars
    g1_obj=0.1
    g2_obj=0.05
    Tpix_obj=16.0
    Tsky_obj=Tpix_obj*jfac2

    counts_sky_obj=counts_sky
    noise_sky_obj=noise_sky
    counts_pix_obj=counts_sky_obj/jfac2
    noise_pix_obj=noise_sky_obj/jfac2

    pars_psf = [0.0, 0.0, g1_psf, g2_psf, Tsky_psf, counts_sky_psf]
    gm_psf=gmix.GMixModel(pars_psf, "gauss")

    pars_obj = [0.0, 0.0, g1_obj, g2_obj, Tsky_obj, counts_sky_obj]
    gm_obj0=gmix.GMixModel(pars_obj, model)

    gm=gm_obj0.convolve(gm_psf)

    im_psf=gm_psf.make_image(dims, jacobian=j)
    im_obj=gm.make_image(dims, jacobian=j)

    im_obj[:,:] += noise_pix_obj*numpy.random.randn(im_obj.size).reshape(im_obj.shape)
    wt_obj=zeros(im_obj.shape) + 1./noise_pix_obj**2

    #
    # priors
    #

    cen_prior=priors.CenPrior(0.0, 0.0, 0.1, 0.1)
    T_prior=priors.LogNormal(Tsky_obj, 0.1*Tsky_obj)
    counts_prior=priors.LogNormal(counts_sky_obj, 0.1*counts_sky_obj)
    g_prior = priors.GPriorBA(0.3)

    #
    # fitting
    #

    # psf using EM
    im_psf_sky,sky=em.prep_image(im_psf)
    mc_psf=em.GMixEM(im_psf_sky, jacobian=j)
    emo_guess=gm_psf.copy()
    emo_guess._data['p'] = 1.0
    emo_guess._data['row'] += 0.1*srandu()
    emo_guess._data['col'] += 0.1*srandu()
    emo_guess._data['irr'] += 0.5*srandu()
    emo_guess._data['irc'] += 0.1*srandu()
    emo_guess._data['icc'] += 0.5*srandu()
    mc_psf.go(emo_guess, sky)
    res_psf=mc_psf.get_result()
    print('psf numiter:',res_psf['numiter'],'fdiff:',res_psf['fdiff'])

    psf_fit=mc_psf.get_gmix()
    imfit_psf=mc_psf.make_image(counts=im_psf.sum())
    images.compare_images(im_psf, imfit_psf, label1='psf',label2='fit')

    # obj
    jlist=[j]*nimages
    imlist_obj=[im_obj]*nimages
    wtlist_obj=[wt_obj]*nimages
    psf_fit_list=[psf_fit]*nimages

    mc_obj=MCMCSimple(imlist_obj, wtlist_obj, jlist, model,
                      psf=psf_fit_list,
                      T=Tsky_obj,
                      counts=counts_sky_obj,
                      cen_prior=cen_prior,
                      T_prior=T_prior,
                      counts_prior=counts_prior,
                      g_prior=g_prior,
                      do_lensfit=do_lensfit,
                      do_pqr=do_pqr)
    mc_obj.go()

    res_obj=mc_obj.get_result()

    pprint(res_obj)
    print_pars(res_obj['pars'], front='pars_obj:')
    print_pars(res_obj['pars_err'], front='perr_obj:')
    print('Tpix: %.4g +/- %.4g' % (res_obj['pars'][4]/jfac2, res_obj['pars_err'][4]/jfac2))
    if do_lensfit:
        print('gsens:',res_obj['g_sens'])
    if do_pqr:
        print('P:',res_obj['P'])
        print('Q:',res_obj['Q'])
        print('R:',res_obj['R'])

    gmfit0=mc_obj.get_gmix()
    gmfit=gmfit0.convolve(psf_fit)
    imfit_obj=gmfit.make_image(im_obj.shape, jacobian=j)

    images.compare_images(im_obj, imfit_obj, label1=model,label2='fit')
    mcmc.plot_results(mc_obj.get_trials())


def test_model_mb(model,
                  counts_sky=[100.0, 88., 77., 95.0], # determines nband
                  noise_sky=0.1,
                  nimages=10, # in each band
                  jfac=0.27,
                  do_lensfit=False,
                  do_pqr=False,

                  nwalkers=80,
                  burnin=400,
                  nstep=800,

                  rand_center=True,

                  show=False):
    """
    testing mb stuff
    """
    import images
    import mcmc
    from . import em
    import time

    from ngmix.joint_prior import PriorSimpleSep
 
    jfac2=jfac**2

    dims=[25,25]
    cen=array( [dims[0]/2., dims[1]/2.] )

    # object pars
    g1_obj=0.1
    g2_obj=0.05
    Tpix_obj=16.0
    Tsky_obj=Tpix_obj*jfac2

    true_pars=array([0.0,0.0,g1_obj,g2_obj,Tsky_obj]+counts_sky)

    counts_sky_psf=100.0
    counts_pix_psf=counts_sky_psf/jfac2

    nband=len(counts_sky)

    mb_obs_list=MultiBandObsList()

    tmpsf=0.0
    for band in xrange(nband):

        if rand_center:
            cen_i = cen + srandu(2)
        else:
            cen_i = cen.copy()

        # not always at same center
        jacob=Jacobian(cen_i[0],
                       cen_i[1],
                       jfac,
                       0.0,
                       0.0,
                       jfac)
        counts_sky_obj=counts_sky[band]
        counts_pix_obj=counts_sky_obj/jfac2
        noise_pix_obj=noise_sky/jfac2

        obs_list=ObsList()
        for i in xrange(nimages):
            # PSF pars
            psf_cen1=0.1*srandu()
            psf_cen2=0.1*srandu()
            g1_psf= 0.05 + 0.1*srandu()
            g2_psf=-0.01 + 0.1*srandu()
            Tpix_psf=4.0*(1.0 + 0.1*srandu())
            Tsky_psf=Tpix_psf*jfac2

            pars_psf = [psf_cen1,psf_cen2, g1_psf, g2_psf, Tsky_psf, counts_sky_psf]
            gm_psf=gmix.GMixModel(pars_psf, "gauss")

            # 0 means at jacobian row0,col0
            pars_obj = [0.0, 0.0, g1_obj, g2_obj, Tsky_obj, counts_sky_obj]
            gm_obj0=gmix.GMixModel(pars_obj, model)

            gm=gm_obj0.convolve(gm_psf)

            im_psf=gm_psf.make_image(dims, jacobian=jacob, nsub=16)
            im_obj=gm.make_image(dims, jacobian=jacob, nsub=16)

            im_obj[:,:] += noise_pix_obj*numpy.random.randn(im_obj.size).reshape(im_obj.shape)
            wt_obj=zeros(im_obj.shape) + 1./noise_pix_obj**2

            # psf using EM
            tmpsf0=time.time()

            obs_i = Observation(im_obj, weight=wt_obj, jacobian=jacob)

            im_psf_sky,sky=em.prep_image(im_psf)

            psf_obs_i = Observation(im_psf_sky, jacobian=jacob)

            mc_psf=em.GMixEM(psf_obs_i)

            emo_guess=gm_psf.copy()
            emo_guess._data['p'] = 1.0
            mc_psf.go(emo_guess, sky)
            res_psf=mc_psf.get_result()

            tmpsf+=time.time()-tmpsf0
            #print 'psf numiter:',res_psf['numiter'],'fdiff:',res_psf['fdiff']

            psf_fit=mc_psf.get_gmix()

            psf_obs_i.set_gmix(psf_fit)

            obs_i.set_psf(psf_obs_i)

            obs_list.append(obs_i)

        mb_obs_list.append(obs_list)


    tmrest=time.time()
    #
    # priors
    # not really accurate since we are not varying the input
    #

    cen_prior=priors.CenPrior(0.0, 0.0, 0.1, 0.1)

    log10_T = log10(Tsky_obj)

    T_prior=priors.FlatPrior(log10_T-2.0, log10_T+2.0)
    counts_prior=[]
    for band in xrange(nband):
        counts=counts_sky[band]
        log10_counts = log10(counts)
        cp = priors.FlatPrior(log10_counts-2.0, log10_counts+2.0)
        counts_prior.append(cp)

    g_prior = priors.GPriorBA(0.3)

    prior=PriorSimpleSep(cen_prior,
                         g_prior,
                         T_prior,
                         counts_prior)
    #
    # fitting
    #

    mc_obj=MCMCSimple(mb_obs_list,
                      model,
                      prior=prior,
                      nwalkers=nwalkers)

    print("making guess from priors")
    guess=prior.sample(nwalkers)

    print("burnin",burnin)
    pos=mc_obj.run_mcmc(guess, burnin)
    print("steps",nstep)
    pos=mc_obj.run_mcmc(pos, nstep)

    mc_obj.calc_result()
    mc_obj.calc_lin_result()

    res=mc_obj.get_lin_result()

    tmrest = time.time()-tmrest

    tmtot=tmrest + tmpsf
    print('\ntime total:',tmtot)
    print('time psf:  ',tmpsf)
    print('time rest: ',tmrest)
    print()

    print('arate:',res['arate'])
    print_pars(true_pars, front='true:    ')
    print_pars(res['pars'], front='pars_obj:')
    print_pars(res['pars_err'], front='perr_obj:')

    if do_lensfit:
        print('gsens:',res['g_sens'])
    if do_pqr:
        print('P:',res['P'])
        print('Q:',res['Q'])
        print('R:',res['R'])
           
    if show:
        mc_obj.make_plots(show=True, do_residual=True)


def _get_test_psf_flux_pars(ngauss, cen, jfac, counts_sky):

    jfac2=jfac**2
    if ngauss==1:
        e1=0.1*srandu()
        e2=0.1*srandu()
        Tpix=4.0*(1.0 + 0.2*srandu())

        Tsky=Tpix*jfac2
        pars=array([counts_sky,
                    cen[0],
                    cen[1],
                    (Tsky/2.)*(1-e1),
                    (Tsky/2.)*e2,
                    (Tsky/2.)*(1+e1)],dtype='f8')

    elif ngauss==2:
        e1_1=0.1*srandu()
        e2_1=0.1*srandu()
        e1_2=0.1*srandu()
        e2_2=0.1*srandu()

        counts_frac1 = 0.6*(1.0 + 0.1*srandu())
        counts_frac2 = 1.0 - counts_frac1
        T1pix=4.0*(1.0 + 0.2*srandu())
        T2pix=8.0*(1.0 + 0.2*srandu())

        T1sky=T1pix*jfac2
        T2sky=T2pix*jfac2
        pars=array([counts_frac1*counts_sky,
                    cen[0],
                    cen[1],
                    (T1sky/2.)*(1-e1_1),
                    (T1sky/2.)*e2_1,
                    (T1sky/2.)*(1+e1_1),

                    counts_frac2*counts_sky,
                    cen[0],
                    cen[1],
                    (T2sky/2.)*(1-e1_2),
                    (T2sky/2.)*e2_2,
                    (T2sky/2.)*(1+e1_2)])


    else:
        raise ValueError("bad ngauss: %s" % ngauss)

    gm=gmix.GMix(pars=pars)
    return gm

def test_template_flux(ngauss,
                       send_center_as_keyword=True, # let the template fitting code reset the centers
                       do_psf=True,
                       counts_sky=100.0,
                       noise_sky=0.01,
                       nimages=1,
                       jfac=0.27,
                       jcen_offset=None,
                       show=False):
    """

    For do_psf, the gmix are in the psf observations, otherwise in the
    observation

    If reset_centers, the cen= is sent, otherwise the gmix centers are
    set before calling
    """
    from .em import GMixMaxIterEM
    import images
    import mcmc
    from . import em

    # arcsec
    #cen_sky = array([0.8, -1.2])
    cen_sky = array([1.8, -2.1])

    dims=[40,40]
    jcen=array( [dims[0]/2., dims[1]/2.] )
    if jcen_offset is not None:
        jcen_offset = array(jcen_offset)
        jcen += jcen_offset

    jcenfac=2.0
    jfac2=jfac**2
    noise_pix=noise_sky/jfac2

    ntry=10

    tm_em=0.0

    obs_list=ObsList()
    for i in xrange(nimages):
        # gmix is in sky coords.  Note center is cen_sky not the jacobian center
        gm=_get_test_psf_flux_pars(ngauss, cen_sky, jfac, counts_sky)

        # put row0,col0 at a random place
        j=Jacobian(jcen[0]+jcenfac*srandu(),jcen[1]+jcenfac*srandu(), jfac, 0.0, 0.0, jfac)

        im0=gm.make_image(dims, jacobian=j)
        if show:
            import images
            images.view(im0,title='image %s' % (i+1))

        im = im0 + noise_pix*numpy.random.randn(im0.size).reshape(dims)

        im0_skyset,sky=em.prep_image(im0)

        tobs=Observation(im0_skyset, jacobian=j)
        mc=em.GMixEM(tobs)

        # gm is also guess
        gm_guess=gm.copy()
        gm_guess.set_psum(1.0)
        gm_guess.set_cen(0.0, 0.0)
        for k in xrange(ntry):
            try:
                mc.go(gm_guess, sky, tol=1.e-5)
                break
            except GMixMaxIterEM:
                if (k==ntry-1):
                    raise
                else:
                    res=mc.get_result()
                    print('try:',k,'fdiff:',res['fdiff'],'numiter:',res['numiter'])
                    print(mc.get_gmix())
                    gm_guess.set_cen(0.1*srandu(), 0.1*srandu())
                    gm_guess._data['irr'] = gm._data['irr']*(1.0 + 0.1*srandu(ngauss))
                    gm_guess._data['icc'] = gm._data['icc']*(1.0 + 0.1*srandu(ngauss))
        psf_fit=mc.get_gmix()

        wt=0.0*im.copy() + 1./noise_pix**2

        obs=Observation(im, weight=wt, jacobian=j)

        if do_psf:
            tobs.set_gmix(psf_fit)
            obs.set_psf(tobs)
        else:
            obs.set_gmix(psf_fit)

        obs_list.append(obs)

        res=mc.get_result()
        print(i+1,res['numiter'])


    if send_center_as_keyword:
        fitter=TemplateFluxFitter(obs_list, cen=cen_sky, do_psf=do_psf)
    else:
        fitter=TemplateFluxFitter(obs_list, do_psf=do_psf)

    fitter.go()

    res=fitter.get_result()

    print("flux(sky):",counts_sky)
    print("meas: %g +/- %g" % (res['flux'], res['flux_err']))

def _make_sheared_pars(pars, shear_g1, shear_g2):
    from .shape import Shape
    shpars=pars.copy()

    sh=Shape(shpars[2], shpars[3])
    sh.shear(shear_g1, shear_g2)

    shpars[2]=sh.g1
    shpars[3]=sh.g2

    return shpars

def _make_obs(pars, model, noise_image, jacob, weight, psf):
    gm0=gmix.GMixModel(pars, model, logpars=True)
    gm=gm0.convolve(psf.gmix)
    im = gm.make_image(noise_image.shape, jacobian=jacob)

    im += noise_image

    obs=Observation(im, jacobian=jacob, weight=weight, psf=psf)

    return obs

class RetryError(Exception):
    """
    EM algorithm hit max iter
    """
    def __init__(self, value):
         self.value = value
    def __str__(self):
        return repr(self.value)

def _add_noise_obs(obs, frac=0.1):
    wm=numpy.median(obs.weight)
    n = sqrt(1.0/wm)
    new_im = obs.image.copy()

    new_noise = frac*n
    new_im += new_noise*numpy.random.randn(new_im.size).reshape(new_im.shape)

    new_total_noise = sqrt(n**2 + new_noise**2)
    new_wt = 0.0*obs.weight + 1.0/new_total_noise**2

    new_obs=Observation(new_im,
                        weight=new_wt,
                        jacobian=obs.jacobian,
                        psf=obs.psf)
    return new_obs
                            


def _do_lm_fit(obs, prior, sample_prior, model):
    lm_pars={'maxfev': 300,
             'ftol':   1.0e-6,
             'xtol':   1.0e-6,
             'epsfcn': 1.0e-6}

    #lm_fitter=LMSimple(obs, model, lm_pars=lm_pars, prior=prior)
    lm_fitter=LMSimple(obs, model, lm_pars=lm_pars)

    nmax=1000
    i=0
    while True:
        #if i > 0:
            #print_pars(guess,front="draw from prior:")
        #    guess=prior.sample()
        guess=sample_prior.sample()

        try:

            guess[2]=0.1*srandu()
            guess[3]=0.1*srandu()
            lm_fitter.run_lm(guess)
        
            res=lm_fitter.get_result()

            if res['flags']==0:
                break
                '''
                g=sqrt( res['g'][0]**2 + res['g'][1]**2 )
                if g < 0.97:
                    print("good")
                    break
                else:
                    print("bad g: %g" % g)
                '''
        except GMixRangeError as err:
            print("caught range error: %s" % str(err))

        if i > nmax:
            raise RetryError("too many tries")
        i += 1

    return res

def test_lm_metacal(model,
                    shear=0.04,
                    T_obj=16.0,
                    noise_obj=0.01,
                    npair=100):
    from .shape import Shape
    from . import em
    import lensing

    shear=Shape(shear, 0.0)
    h=0.02
    #h=0.05

    dims=[25,25]
    npix=dims[0]*dims[1]
    cen=[dims[0]/2., dims[1]/2.]
    jacob=UnitJacobian(cen[0],cen[1])
    wt_obj = zeros(dims) + 1.0/noise_obj**2


    # PSF pars
    counts_psf=100.0
    noise_psf=0.001
    g1_psf=0.05
    g2_psf=-0.01
    T_psf=4.0

    counts_obj=100.0
    T_obj=16.0

    prior,prior_gflat=get_mh_prior(T_obj, counts_obj)

    g_vals=zeros( (npair*2,2) )
    g_err_vals=zeros(npair*2)
    gsens_vals=zeros(npair*2)
    s2n_vals=zeros(npair*2)

    nretry=0
    for ii in xrange(npair):
        while True:
            try:
                if (ii % 100) == 0:
                    print("%d/%d" % (ii,npair))

                pars_obj_0 = prior.sample()

                shape1=Shape(pars_obj_0[2], pars_obj_0[3])

                shape2=shape1.copy()
                shape2.rotate(numpy.pi/2.)

                pars_psf = [pars_obj_0[0], pars_obj_0[1], g1_psf, g2_psf,
                            T_psf, counts_psf]
                gm_psf=gmix.GMixModel(pars_psf, "gauss")
                im_psf=gm_psf.make_image(dims, jacobian=jacob)

                noise_im_psf=noise_psf*numpy.random.randn(npix)
                noise_im_psf = noise_im_psf.reshape(dims)
                im_psf[:,:] += noise_im_psf

                im_psf_sky,sky=em.prep_image(im_psf)
                psf_obs = Observation(im_psf_sky, jacobian=jacob)

                mc_psf=em.GMixEM(psf_obs)

                emo_guess=gm_psf.copy()
                emo_guess._data['p'] = 1.0
                emo_guess._data['row'] += 0.01*srandu()
                emo_guess._data['col'] += 0.01*srandu()
                emo_guess._data['irr'] += 0.01*srandu()
                emo_guess._data['irc'] += 0.01*srandu()
                emo_guess._data['icc'] += 0.01*srandu()

                mc_psf.go(emo_guess, sky)
                res_psf=mc_psf.get_result()
                psf_fit=mc_psf.get_gmix()
                #print('psf numiter:',res_psf['numiter'],'fdiff:',res_psf['fdiff'])

                psf_obs.set_gmix(psf_fit)

                for ipair in [1,2]:

                    noise_image = noise_obj*numpy.random.randn(npix)
                    noise_image = noise_image.reshape(dims)

                    sheared_pars = pars_obj_0.copy()
                    if ipair==1:
                        i=2*ii
                        sh = shape1.copy()
                    if ipair==2:
                        i=2*ii+1
                        sh = shape2.copy()

                    sh.shear(shear.g1, shear.g2)
                    sheared_pars[2]=sh.g1
                    sheared_pars[3]=sh.g2

                    obs = _make_obs(sheared_pars, model, noise_image,
                                    jacob, wt_obj, psf_obs)

                    max_lnprob=-9999.e59
                    res=None
                    for irep in xrange(1):

                        tres=_do_lm_fit(obs, prior_gflat, prior, model)
                        if tres['lnprob'] > max_lnprob:
                            max_lnprob=tres['lnprob']
                            res=tres

                    gg=res['g']
                    g=sqrt(gg[0]**2 + gg[1]**2)
                    if g > 0.97:
                        nretry += 1
                        raise RetryError("bad g")
                    

                    # now metacal
                    pars_meas = res['pars']
                    pars_lo=_make_sheared_pars(pars_meas, -h, 0.0)
                    pars_hi=_make_sheared_pars(pars_meas, +h, 0.0)

                    obs_lo = _make_obs(pars_lo, model, noise_image,
                                       jacob, wt_obj, psf_obs)
                    obs_hi = _make_obs(pars_hi, model, noise_image,
                                       jacob, wt_obj, psf_obs)

                    #guess=pars_meas
                    res_lo=_do_lm_fit(obs_lo, prior_gflat, prior, model)
                    res_hi=_do_lm_fit(obs_hi, prior_gflat, prior, model)

                    pars_lo=res_lo['pars']
                    pars_hi=res_hi['pars']

                    gsens_vals[i] = (pars_hi[2]-pars_lo[2])/(2.*h)
                    s2n_vals[i]=res['s2n_w']

                    g_vals[i,0] = res['pars'][2]
                    g_vals[i,1] = res['pars'][3]
                    g_err_vals[i] = res['pars_err'][2]

                break

            except RetryError:
                print("retrying")
                pass
            except GMixRangeError:
                print("retrying range error")
                pass



    g_mean = g_vals.mean(axis=0)
    g_err = g_vals.std(axis=0)/sqrt(2*npair)
    gsens_mean=gsens_vals.mean()

    shear_mean =g_mean/gsens_mean

    # soften
    err2 = g_err_vals**2 + 0.001
    shear_err2_inv = ( 1.0/err2 ).sum()
    shear_err = sqrt( 1.0/shear_err2_inv )/gsens_mean

    print('s2n:',s2n_vals.mean())
    print("%g +/- %g" % (g_mean[0], g_err[0]))
    print(gsens_mean)
    print("%g +/- %g" % (shear_mean[0], shear_err))

    chunksize=int(g_vals.shape[0]/100.)
    if chunksize < 1:
        chunksize=1
    print("chunksize:",chunksize)

    shear, shear_cov = lensing.shear.shear_jackknife(g_vals,
                                                     chunksize=chunksize)
    print("%g +/- %g" % (shear[0], sqrt(shear_cov[0,0])))
    print("%g +/- %g" % (shear[0]/gsens_mean, sqrt(shear_cov[0,0]/gsens_mean)))
    print("nretry:",nretry)

    return g_vals, shear, shear_cov


