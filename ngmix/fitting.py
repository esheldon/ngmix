# there are a few additional imports not in this header for example we only
# import emcee if needed

from sys import stdout, stderr
import numpy
import time

from . import gmix
from .gmix import _exp3_ivals,_exp3_lookup
from .jacobian import Jacobian, UnitJacobian

from . import priors
from .priors import srandu, LOWVAL, BIGVAL

from .gexceptions import GMixRangeError

MAX_TAU=0.1
MIN_ARATE=0.2
MCMC_NTRY=1

BAD_VAR=2**0
LOW_ARATE=2**1
LARGE_TAU=2**2

# error codes in LM start at 2**0 and go to 2**3
# this is because we set 2**(ier-5)
LM_SINGULAR_MATRIX = 2**4
LM_NEG_COV_EIG = 2**5
LM_NEG_COV_DIAG = 2**6
LM_EIG_NOTFINITE = 2**7
LM_FUNC_NOTFINITE = 2**8

PDEF=-9.999e9
CDEF=9.999e9

class FitterBase(object):
    """
    Base for other fitters

    Designed to fit many images at once.  For this reason, a jacobian
    transformation is required to put all on the same system. For the
    same reason, the center of the model is relative to "zero", which
    points to the common center used by all transformation objects; the
    row0,col0 in pixels for each should correspond to that center in the
    common coordinates (e.g. sky coords)

    Fluxes and sizes will also be in the transformed system.  You can use the
    method get_flux_scaling to get the average scaling from the input images.
    This only makes sense for comparing to fluxes measured in the image system,
    for example zeropoints.  If you are using very many different cameras then
    you should just work in sky coordinates, or scale the images by the zero
    point corrections.
    
    """
    def __init__(self, image, weight, jacobian, model, **keys):
        self.keys=keys

        self.g_prior = keys.get('g_prior',None)
        self.cen_prior = keys.get('cen_prior',None)
        self.T_prior = keys.get('T_prior',None)
        self.counts_prior = keys.get('counts_prior',None)

        self.joint_TF_prior = keys.get('joint_TF_prior',None)

        # in this case, image, weight, jacobian, psf are going to
        # be lists of lists.

        # call this first, others depend on it
        self._set_lists(image, weight, jacobian, **keys)

        self.model=gmix.get_model_num(model)
        self.model_name=gmix.get_model_name(self.model)
        self._set_npars()

        # the function to be called to fill a gaussian mixture
        self._set_fill_call()

        self.totpix=self.verify()

        self._gmix_lol=None

    def _set_lists(self, im_lol, wt_lol, j_lol, **keys):
        """
        Internally we store everything as lists of lists.  The outer
        list is the bands, the inner is the list of images in the band.
        """

        psf_lol=keys.get('psf',None)

        if isinstance(im_lol,numpy.ndarray):
            # lists-of-lists for generic, including multi-band
            im_lol=[[im_lol]]
            wt_lol=[[wt_lol]]
            j_lol=[[j_lol]]

            if psf_lol is not None:
                psf_lol=[[psf_lol]]

            if self.counts_prior is not None:
                self.counts_prior=[self.counts_prior]

        elif isinstance(im_lol,list) and isinstance(im_lol[0],numpy.ndarray):
            im_lol=[im_lol]
            wt_lol=[wt_lol]
            j_lol=[j_lol]

            if psf_lol is not None:
                psf_lol=[psf_lol]
        elif (isinstance(im_lol,list) 
                and isinstance(im_lol[0],list)
                and isinstance(im_lol[0][0],numpy.ndarray)):
            # OK, good
            pass
        else:
            raise ValueError("images should be input as array, "
                             "list of arrays, or lists-of lists "
                             "of arrays")

        # can be 1
        self.nband=len(im_lol)
        nimages = numpy.array( [len(l) for l in im_lol], dtype='i4')

        self.im_lol=im_lol
        self.wt_lol=wt_lol
        self.psf_lol=psf_lol

        self.jacob_lol = j_lol
        mean_det=numpy.zeros(self.nband)
        for band in xrange(self.nband):
            jlist=self.jacob_lol[band]
            for j in jlist:
                mean_det[band] += j._data['det']
            mean_det[band] /= len(jlist)

        self.mean_det=mean_det

    def verify(self):
        """
        Make sure the data are consistent.
        """
        nb=self.nband
        wt_nb = len(self.wt_lol)
        j_nb  = len(self.jacob_lol)
        if (wt_nb != nb or j_nb != nb):
            nbt=(nb,wt_nb,j_nb)
            raise ValueError("lists of lists not all same size: "
                             "im: %s wt: %s jacob: %s" % nbt)
        if self.psf_lol is not None:
            psf_nb=len(self.psf_lol)
            if psf_nb != nb:
                nbt=(nb,psf_nb)
                raise ValueError("lists of lists not all same size: "
                                 "im: %s psf: %s" % nbt)


        totpix=0
        for i in xrange(self.nband):
            nim=len(self.im_lol[i])

            wt_n=len(self.wt_lol[i])
            j_n=len(self.jacob_lol[i])
            if wt_n != nim or j_n != nim:
                nt=(i,nim,wt_n,j_n)
                raise ValueError("lists for band %s not same length: "
                                 "im: %s wt: %s jacob: " % nt)
            if self.psf_lol is not None:
                psf_n = len(self.psf_lol[i])
                if psf_n != nim:
                    nt=(i,nim,psf_n)
                    raise ValueError("lists for band %s not same length: "
                                     "im: %s psf: " % nt)


            for j in xrange(nim):
                imsh=self.im_lol[i][j].shape
                wtsh=self.wt_lol[i][j].shape
                if imsh != wtsh:
                    raise ValueError("im.shape != wt.shape "
                                     "(%s != %s)" % (imsh,wtsh))

                totpix += imsh[0]*imsh[1]

        if self.counts_prior is not None:
            if not isinstance(self.counts_prior,list):
                raise ValueError("counts_prior must be a list, "
                                 "got %s" % type(self.counts_prior))
            nc=len(self.counts_prior)
            if nc != self.nband:
                raise ValueError("counts_prior list %s doesn't match "
                                 "number of bands %s" % (nc,self.nband))
            
        return totpix

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
        else:
            raise GMixFatalError("unsupported model: "
                                 "'%s'" % self.model_name)


    def get_result(self):
        """
        Result will not be non-None until go() is run
        """
        if not hasattr(self, '_result'):
            raise ValueError("No result, you must run go()!")

        return self._result
    
    def get_flux_scaling(self):
        """
        Scaling to take flux back into image coords.  Useful if comparing to a
        zero point calculated in image coords
        """
        return 1.0/self.mean_jacob_det

    def get_gmix(self):
        """
        Get a gaussian mixture at the "best" parameter set, which
        definition depends on the sub-class
        """
        pars=self._result['pars']
        gm=gmix.GMixModel(pars, self.model)
        return gm

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
            for wt_list in self.wt_lol:
                for wt in wt_list:
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
        get_priors=True only works 
        This is all we use for mcmc approaches, but also used generally for the
        "get_fit_stats" method.  For the max likelihood fitter we also have a
        _get_ydiff method
        """

        s2n_numer=0.0
        s2n_denom=0.0
        try:

            ln_priors = self._get_priors(pars)
            ln_prob = ln_priors

            self._fill_gmix_lol(pars)
            for band in xrange(self.nband):

                gmix_list=self._gmix_lol[band]
                im_list=self.im_lol[band]
                wt_list=self.wt_lol[band]
                jacob_list=self.jacob_lol[band]

                nim=len(im_list)
                for i in xrange(nim):
                    gm=gmix_list[i]
                    im=im_list[i]
                    wt=wt_list[i]
                    j=jacob_list[i]

                    #res = gm.get_loglike(im, wt, jacobian=j,
                    #                     get_s2nsums=True)
                    res = gmix._loglike_jacob_fast3(gm._data,
                                                    im,
                                                    wt,
                                                    j._data,
                                                    _exp3_ivals[0],
                                                    _exp3_lookup)

                    ln_prob += res[0]
                    s2n_numer += res[1]
                    s2n_denom += res[2]

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
        Get some statistics for the best fit.
        """
        npars=self.npars

        lnprob,s2n_numer,s2n_denom=self.calc_lnprob(pars, get_s2nsums=True)

        if s2n_denom > 0:
            s2n=s2n_numer/numpy.sqrt(s2n_denom)
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





    def _init_gmix_lol(self, pars):
        """
        initialize the list of lists of gmix
        """
        gmix_lol0 = []
        gmix_lol  = []

        for band in xrange(self.nband):
            gmix_list0=[]
            gmix_list=[]

            band_pars=self._get_band_pars(pars, band)
            psf_list=self.psf_lol[band]

            for psf in psf_list:
                gm0=gmix.GMixModel(band_pars, self.model)
                gm=gm0.convolve(psf)

                gmix_list0.append(gm0)
                gmix_list.append(gm)

            gmix_lol0.append(gmix_list0)
            gmix_lol.append(gmix_list)

        self._gmix_lol0 = gmix_lol0
        self._gmix_lol  = gmix_lol

    def _fill_gmix_lol(self, pars):
        """
        Fill the list of lists of gmix objects, potentially convolved with the
        psf in the individual images
        """
        for band in xrange(self.nband):
            gmix_list0=self._gmix_lol0[band]
            gmix_list=self._gmix_lol[band]

            band_pars=self._get_band_pars(pars, band)
            psf_list=self.psf_lol[band]

            for i,psf in enumerate(psf_list):
                gm0=gmix_list0[i]
                gm=gmix_list[i]

                # Calling the python versions was a huge time sync.
                # but we need some more error checking here
                try:
                    self._fill_gmix_func(gm0._data, band_pars)
                    gmix._convolve_fill(gm._data, gm0._data, psf._data)
                except ZeroDivisionError:
                    raise GMixRangeError("zero division")


    def _get_counts_guess(self, **keys):
        cguess=keys.get('counts_guess',None)
        if cguess is None:
            cguess = self._get_median_counts()
        else:
            cguess=numpy.array(cguess,ndmin=1)
        return cguess

    def _get_median_counts(self):
        """
        median of the counts across all input images, for each band
        """
        cguess=numpy.zeros(self.nband)
        for band in xrange(self.nband):

            im_list=self.im_lol[band]
            jacob_list=self.jacob_lol[band]

            nim=len(im_list)
            clist=numpy.zeros(nim)

            for i in xrange(nim):
                im=im_list[i]
                j=jacob_list[i]
                clist[i] = im.sum()*j._data['det']
            cguess[band] = numpy.median(clist) 
        return cguess

class PSFFluxFitter(FitterBase):
    """
    We fix the center, so this is linear.  Just cross-correlations
    between model and data.
    """
    def __init__(self, image, weight, jacobian, psf, **keys):
        self.keys=keys

        # in this case, image, weight, jacobian, psf are going to
        # be lists of lists.

        # call this first, others depend on it
        self._set_lists(image, weight, jacobian, psf, **keys)

        self.model_name='psf'
        self.npars=1

        self.totpix=self.verify()

    def go(self):
        """
        calculate the flux using zero-lag cross-correlation
        """
        xcorr_sum=0.0
        msq_sum=0.0

        chi2=0.0

        for ipass in [1,2]:
            for i in xrange(self.nimages):
                im=self.im_list[i]
                wt=self.wt_list[i]
                j=self.jacob_list[i]
                psf=self.psf_list[i]
                
                if ipass==1:
                    psf.set_psum(1.0)
                    psf.set_cen(0.0, 0.0)
                    model=psf.make_image(im.shape, jacobian=j)
                    xcorr_sum += (model*im*wt).sum()
                    msq_sum += (model*model*wt).sum()
                else:
                    psf.set_psum(flux)
                    model=psf.make_image(im.shape, jacobian=j)
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
            flux_err = numpy.sqrt(arg)
        else:
            flags=BAD_VAR
            flux_err=9999.0

        self._result={'model':self.model_name,
                      'flags':flags,
                      'chi2per':chi2per,
                      'dof':dof,
                      'flux':flux,
                      'flux_err':flux_err}

    def _set_lists(self, im_list, wt_list, j_list, psf_list_in, **keys):
        """
        Internally we store everything as lists of lists.  The outer
        list is the bands, the inner is the list of images in the band.
        """

        if isinstance(im_list,numpy.ndarray):
            # lists-of-lists for generic, including multi-band
            im_list=[im_list]
            wt_list=[wt_list]
            j_list=[j_list]
            psf_list_in=[psf_list_in]

        elif (isinstance(im_list,list) 
                and isinstance(im_list[0],numpy.ndarray)):
            # OK, good
            pass
        else:
            raise ValueError("images should be input as array or "
                             "list of arrays")

        self.nimages = len(im_list)

        self.im_list=im_list
        self.wt_list=wt_list
        self.jacob_list = j_list
        self.psf_list_in=psf_list_in

        mean_det=0.0
        for j in j_list:
            mean_det += j._data['det']
        mean_det /= len(j_list)

        self.mean_det=mean_det

        psf_list=[]
        for psf_in in self.psf_list_in:
            psf = psf_in.copy()

            #psfnorm1.set_psum(1.0)
            #psfnorm1.set_cen(0.0, 0.0)
            psf_list.append(psf)

        self.psf_list=psf_list

    def verify(self):
        """
        Make sure the data are consistent.
        """
        n_im=self.nimages
        n_wt = len(self.wt_list)
        n_j  = len(self.jacob_list)
        n_psf  = len(self.psf_list)
        if n_wt != n_im or n_wt != n_j or n_psf != n_im:
            nl=(n_im,n_wt,n_j,n_psf)
            raise ValueError("lists not all same size: "
                             "im: %s wt: %s jacob: %s psf: %s" % nl)


        totpix=0

        for j in xrange(n_im):
            imsh=self.im_list[j].shape
            wtsh=self.wt_list[j].shape
            if imsh != wtsh:
                raise ValueError("im.shape != wt.shape "
                                 "(%s != %s)" % (imsh,wtsh))

            totpix += imsh[0]*imsh[1]

            
        return totpix

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
            for wt in self.wt_list:
                this_wtmax = wt.max()
                if this_wtmax > wtmax:
                    wtmax = this_wtmax

                wtsum += wt.sum()

            self.eff_npix=wtsum/wtmax

        if self.eff_npix <= 0:
            self.eff_npix=1.e-6

        return self.eff_npix


class LMSimple(FitterBase):
    """
    A class for doing a fit using levenberg marquardt

    """
    def __init__(self, image, weight, jacobian, model, guess, **keys):
        super(LMSimple,self).__init__(image, weight, jacobian, model, **keys)

        # this is a dict
        # can contain maxfev (maxiter), ftol (tol in sum of squares)
        # xtol (tol in solution), etc
        self.lm_pars=keys['lm_pars']

        self.guess=numpy.array( guess, dtype='f8' )

        # center + shape + T + fluxes
        n_prior_pars=1 + 1 + 1 + self.nband

        self.fdiff_size=self.totpix + n_prior_pars

    def go(self):
        """
        Run leastsq and set the result
        """

        dof=self.get_dof()
        result = run_leastsq(self._calc_fdiff, self.guess, dof, **self.lm_pars)

        if result['flags']==0:
            result['g'] = result['pars'][2:2+2].copy()
            result['g_cov'] = result['pars_cov'][2:2+2, 2:2+2].copy()
            stats=self.get_fit_stats(result['pars'])
            result.update(stats)

        self._result=result

    def _get_band_pars(self, pars, band):
        return pars[ [0,1,2,3,4,5+band] ]

    def _calc_fdiff(self, pars, get_s2nsums=False):
        """
        vector with (model-data)/error.

        The npars elements contain -ln(prior)
        """

        # we cannot keep sending existing array into leastsq, don't know why
        fdiff=numpy.zeros(self.fdiff_size)

        if not hasattr(self,'_gmix_lol0'):
            self._init_gmix_lol(pars)

        s2n_numer=0.0
        s2n_denom=0.0

        try:

            self._fill_gmix_lol(pars)

            start=self._fill_priors(pars, fdiff)

            for band in xrange(self.nband):

                gmix_list=self._gmix_lol[band]
                im_list=self.im_lol[band]
                wt_list=self.wt_lol[band]
                jacob_list=self.jacob_lol[band]

                nim=len(im_list)
                for i in xrange(nim):
                    gm=gmix_list[i]
                    im=im_list[i]
                    wt=wt_list[i]
                    j=jacob_list[i]

                    res = gmix._fdiff_jacob_fast3(gm._data,
                                                  im,
                                                  wt,
                                                  j._data,
                                                  fdiff,
                                                  start,
                                                  _exp3_ivals[0],
                                                  _exp3_lookup)
                    s2n_numer += res[0]
                    s2n_denom += res[1]

                    start += im.size

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

        index=0
        fdiff[index] = -self.cen_prior.get_lnprob(pars[0], pars[1])
        index += 1
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

    def _get_priors(self, pars):
        """
        For the stats calculation
        """
        lnp=0.0
        
        lnp += self.cen_prior.get_lnprob(pars[0], pars[1])
        lnp += self.g_prior.get_lnprob_scalar2d(pars[2], pars[3])
        lnp += self.T_prior.get_lnprob_scalar(pars[4])
        for i,cp in enumerate(self.counts_prior):
            counts=pars[5+i]
            lnp += cp.get_lnprob_scalar(counts)

        return lnp


class LMSimpleJointTF(LMSimple):
    """
    A class for doing a fit using levenberg marquardt

    Joint size and flux distribution
    """
    def __init__(self, image, weight, jacobian, model, guess, **keys):
        super(LMSimpleJointTF,self).__init__(image, weight, jacobian, model, **keys)

        assert self.joint_TF_prior is not None,"send joint_TF_prior"
        assert self.nband == 1, "add support for multi-band and joint prior"

        # this is over-riding what we did in LMSimple

        # center + shape + (T/fluxes combined)
        n_prior_pars=1 + 1 + 1

        self.fdiff_size=self.totpix + n_prior_pars


    def _fill_priors(self, pars, fdiff):
        """
        Fill priors at the beginning of the array.

        ret the position after last par

        We require all the lnprobs are < 0, equivalent to
        the peak probability always being 1.0

        I have verified all our priors have this property.
        """
        raise ValueError("think how to do prior in LM, both if we need "
                         "factor of two and if need to max at zero ")

        index=0
        fdiff[index] = -self.cen_prior.get_lnprob(pars[0], pars[1])
        index += 1
        fdiff[index] = -self.g_prior.get_lnprob_scalar2d(pars[2], pars[3])
        index += 1

        
        fdiff[index] = -self.joint_TF_prior.get_lnprob_one(pars[4:])
        index += 1

        # this leaves us after the priors
        return index

    def _get_priors(self, pars):
        """
        For the stats calculation
        """
        lnp=0.0
        
        lnp += self.cen_prior.get_lnprob(pars[0], pars[1])
        lnp += self.g_prior.get_lnprob_scalar2d(pars[2], pars[3])

        lnp += self.joint_TF_prior.get_lnprob_one(pars[4:])

        return lnp


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
    import pprint

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
            print >>stderr,'    ',errmsg

        elif pcov0 is None:    
            # why on earth is this not in the flags?
            flags += LM_SINGULAR_MATRIX 
            errmsg = "singular covariance"
            print >>stderr,'    ',errmsg
            print_pars(pars,front='    pars at singular:',stream=stderr)
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
                print >>stderr,'    ',errmsg
                junk1,junk2,perr=_get_def_stuff(npars)
            else:
                # only if we reach here did everything go well
                perr=numpy.sqrt( numpy.diag(pcov) )

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
            print >>stderr,'    not finite'
        else:
            raise e

    return res

def _get_def_stuff(npars):
    pars=numpy.zeros(npars) + PDEF
    cov=numpy.zeros( (npars,npars) ) + CDEF
    err=numpy.zeros(npars) + CDEF
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
    A base class for MCMC runs.  Inherits from overall fitter base class.
    """
    def __init__(self, image, weight, jacobian, model, **keys):
        super(MCMCBase,self).__init__(image, weight, jacobian, model, **keys)

        # this should be a numpy.random.RandomState object, unlike emcee which
        # through the random_state parameter takes the tuple state
        self.random_state = keys.get('random_state',None)

        self.doiter=keys.get('iter',True)

        self.nstep=keys.get('nstep',200)
        self.burnin=keys.get('burnin',400)

        # emcee specific
        self.nwalkers=keys.get('nwalkers',20)
        self.mca_a=keys.get('mca_a',2.0)

        self.ntry=keys.get('ntry',MCMC_NTRY)
        self.min_arate=keys.get('min_arate',MIN_ARATE)

        self.draw_g_prior=keys.get('draw_g_prior',True)

        self.do_pqr=keys.get('do_pqr',False)
        self.do_lensfit=keys.get('do_lensfit',False)

        # expand around this shear value
        self.shear_expand = keys.get('shear_expand',None)

        if (self.do_lensfit or self.do_pqr) and self.g_prior is None:
            raise ValueError("send g_prior for lensfit or pqr")

        self.trials=None


    def get_trials(self):
        """
        Get the set of trials from the production run
        """
        return self.trials

    def get_last_pos(self):
        """
        Get last step from chain
        """
        return self.last_pos

    def get_sampler(self):
        """
        get the emcee sampler
        """
        return self.sampler

    def go(self):
        """
        Run the mcmc sampler and calculate some statistics
        """

        # not nstep can change
        self._do_trials()

        self.trials  = self.sampler.flatchain

        self.lnprobs = self.sampler.lnprobability.reshape(self.nwalkers*self.nstep)
        self.lnprobs -= self.lnprobs.max()

        # get the expectation values, sensitivity and errors
        self._calc_result()


    def _do_trials(self):
        """
        Actually run the sampler
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
                self._init_gmix_lol(guess[0,:])
                break
            except GMixRangeError as gerror:
                # make sure we draw random guess if we got failure
                print >>stderr,'failed init gmix lol:',str(gerror)
                print >>stderr,'getting a new guess'
                guess=self._get_random_guess()
        if i==9:
            raise gerror

        sampler = self._make_sampler()
        self.sampler=sampler

        total_burnin=0
        self.tau=9999.0
        pos=guess
        burnin = self.burnin

        self.best_pars=None
        self.best_lnprob=None

        for i in xrange(self.ntry):
            pos=self._run_some_trials(pos, burnin)

            tau_ok=True
            arate_ok=True
            if have_acor:
                try:
                    self.tau=self._get_tau(sampler, burnin)
                    if self.tau > MAX_TAU and self.doiter:
                        print >>stderr,"        tau",self.tau,">",MAX_TAU
                        tau_ok=False
                except:
                    # something went wrong with acor, run some more
                    print >>stderr,"        exception in acor, running more"
                    tau_ok=False

            if self.arate < self.min_arate and self.doiter:
                print >>stderr,"        arate ",self.arate,"<",self.min_arate
                arate_ok=False

            if tau_ok and arate_ok:
                break

        # if we get here we are hopefully burned in, now do a few more steps
        self.last_pos=self._run_some_trials(pos, self.nstep)

        self.flags=0
        if have_acor and self.tau > MAX_TAU:
            self.flags |= LARGE_TAU
        
        if self.arate < self.min_arate:
            self.flags |= LOW_ARATE

        self.sampler=sampler


    def _run_some_trials(self, pos_in, nstep):
        sampler=self.sampler
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(pos_in, nstep)

        arates = sampler.acceptance_fraction
        self.arate = arates.mean()

        lnprobs = sampler.lnprobability.reshape(self.nwalkers*nstep)
        w=lnprobs.argmax()
        bp=lnprobs[w]
        if self.best_lnprob is None or bp > self.best_lnprob:
            self.best_lnprob=bp
            self.best_pars=sampler.flatchain[w,:]
            #print_pars(self.best_pars, front='best pars:')

        return pos

    def _get_tau(self,sampler, nstep):
        acor=sampler.acor
        tau = (acor/nstep).max()
        return tau

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
            
            print >>stderr,'    replacing random state'
            #sampler.random_state=self.random_state.get_state()

            # OK, we will just hope that _random doesn't change names in the future.
            # but at least we get control back
            sampler._random = self.random_state

        return sampler

    def _get_priors(self, pars):
        """
        Basically a placeholder for no priors
        """
        return 0.0


    def _calc_result(self):
        """
        Will probably over-ride this
        """

        pars,pars_cov=self._get_trial_stats()
 
        self._result={'model':self.model_name,
                      'flags':self.flags,
                      'pars':pars,
                      'pars_cov':pars_cov,
                      'pars_err':numpy.sqrt(numpy.diag(pars_cov)),
                      'tau':self.tau,
                      'arate':self.arate}

        stats = self.get_fit_stats(pars)
        self._result.update(stats)

    def _get_trial_stats(self):
        """
        Get the means and covariance for the trials
        """

        # weights could be the prior values if we didn't apply
        # the prior while sampling
        # or it could be the weights in importance sampling
        # or None
        weights=self.get_weights()
        pars,pars_cov = extract_mcmc_stats(self.trials, weights=weights)

        return pars,pars_cov

    def get_weights(self):
        """
        Get the weights
        """
        if hasattr(self,'nwalkers'):
            # this was an mcmc run
            if self.g_prior is None:
                weights=None
                print >>stderr,'    weights are None'
            else:
                self._set_g_prior_vals()
                print >>stderr,'    weights are g prior'
                weights=self.g_prior_vals
        else:
            weights=None
            print >>stderr,'    weights are none'

        return weights

    def _set_g_prior_vals(self):
        """
        Set g prior vals for later use
        """

        if not hasattr(self,'g_prior_vals'):
            g1=self.trials[:,self.g1i]
            g2=self.trials[:,self.g2i]
            self.g_prior_vals = self.g_prior.get_prob_array2d(g1,g2)

    def _get_guess(self):
        raise RuntimeError("over-ride me")

    def get_par_names(self):
        names=['cen1','cen2', 'g1','g2', 'T']
        if self.nband == 1:
            names += ['Flux']
        else:
            for band in xrange(self.nband):
                names += ['Flux_%s' % i]
        return names


    def make_plots(self,
                   show=False,
                   do_residual=False,
                   title=None):
        """
        Plot the mcmc chain and some residual plots
        """
        import mcmc
        import biggles

        biggles.configure('screen','width', 1200)
        biggles.configure('screen','height', 1200)

        names=self.get_par_names()

        weights=self.get_weights() 
        plt=mcmc.plot_results(self.trials,
                              names=names,
                              title=title,
                              show=show)


        if weights is not None:
            wplt=mcmc.plot_results(self.trials,
                                   weights=weights,
                                   names=names,
                                   title='%s weighted' % title,
                                   show=show)

        if do_residual:
            resplots=self._plot_residuals(title=title,show=show)
            plots=(plt, resplots)
            if weights is not None:
                plots = (plt, wplt, resplots)
        else:
            if weights is not None:
                plots=(plt, wplt)
            else:
                plots=plt

        if show:
            key=raw_input('hit a key: ')
            if key=='q':
                stop

        return plots

    def _plot_residuals(self, title=None, show=False):
        import images
        import biggles

        biggles.configure('screen','width', 1920)
        biggles.configure('screen','height', 1200)

        try:
            self._fill_gmix_lol(self._result['pars'])
        except GMixRangeError as gerror:
            print >>stderr,str(gerror)
            return None

        tablist=[]
        for band in xrange(self.nband):
            gmix_list=self._gmix_lol[band]
            im_list=self.im_lol[band]
            wt_list=self.wt_lol[band]
            jacob_list=self.jacob_lol[band]
            
            nim=len(im_list)

            nrows,ncols=images.get_grid(nim)
            #tab=biggles.Table(nim,1)
            tab=biggles.Table(nrows,ncols)
            ttitle='band: %s' % band
            if title is not None:
                ttitle='%s %s' % (title, ttitle)
            tab.title=ttitle
            #imtot_list=[]
            for i in xrange(nim):
                row=i/ncols
                col=i % ncols

                im=im_list[i]
                wt=wt_list[i]
                j=jacob_list[i]
                gm=gmix_list[i]

                model=gm.make_image(im.shape,jacobian=j)

                # don't care about masked pixels
                residual=(model-im)*wt

                subtab=biggles.Table(1,3)
                imshow=im*wt
                subtab[0,0] = images.view(imshow, show=False)
                subtab[0,1] = images.view(model, show=False)
                subtab[0,2] = images.view(residual, show=False)

                #tab[i,0] = subtab
                tab[row,col] = subtab

                # might want them to have different stretches
                #imcols=im.shape[1]
                #imtot=numpy.zeros( (im.shape[0], 3*imcols ) )
                #imtot[:, 0:imcols]=im
                #imtot[:, imcols:2*imcols]=model
                #imtot[:, 2*imcols:]=residual

                #imtot_list.append(imtot)
            #images.view_mosaic(imtot_list, title='band: %s' % band)
            if show:
                tab.show()

            tablist.append(tab)
        return tablist



class MCMCSimple(MCMCBase):
    """
    Add additional features to the base class to support simple models
    """
    def __init__(self, image, weight, jacobian, model, **keys):
        super(MCMCSimple,self).__init__(image, weight, jacobian, model, **keys)

        self.full_guess=keys.get('full_guess',None)
        self.T_guess=keys.get('T_guess',None)
        if self.T_guess is None:
            self.T_guess=1.44

        self.counts_guess=self._get_counts_guess(**keys)

        ncg=self.counts_guess.size
        if ncg != self.nband:
                raise ValueError("counts_guess size %s doesn't match "
                                 "number of bands %s" % (ncg,self.nband))

        self.g1i = 2
        self.g2i = 3

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
            g = numpy.sqrt(pars[2]**2 + pars[3]**2)
            if g > self.g_prior.gmax:
                raise GMixRangeError("g too big")
    
        if self.T_prior is not None:
            lnp += self.T_prior.get_lnprob_scalar(pars[4])

        if self.counts_prior is not None:
            for i,cp in enumerate(self.counts_prior):
                counts=pars[5+i]
                lnp += cp.get_lnprob_scalar(counts)

        return lnp


    def _get_band_pars(self, pars, band):
        return pars[ [0,1,2,3,4,5+band] ]

    def _get_guess(self):
        """
        # go in simple
        The counts guess is stupid unless you have a well-trimmed
        PSF image
        """

        if self.full_guess is not None:
            return self._get_guess_from_full()
        else:
            return self._get_random_guess()
    
    def _get_guess_from_full(self):
        """
        Return last ``nwalkers'' entries
        """
        ntrial=self.full_guess.shape[0]
        #rint=numpy.random.random_integers(0,ntrial-1,size=self.nwalkers)
        return self.full_guess[ntrial-self.nwalkers:, :]

    def _get_random_guess(self):
        guess=numpy.zeros( (self.nwalkers,self.npars) )

        # center
        guess[:,0]=0.1*srandu(self.nwalkers)
        guess[:,1]=0.1*srandu(self.nwalkers)

        if self.g_prior is not None and self.draw_g_prior:
            guess[:,2],guess[:,3]=self.g_prior.sample2d(self.nwalkers)
        else:
            guess[:,2]=0.1*srandu(self.nwalkers)
            guess[:,3]=0.1*srandu(self.nwalkers)

        guess[:,4] = self.T_guess*(1 + 0.1*srandu(self.nwalkers))

        for band in xrange(self.nband):
            guess[:,5+band] = self.counts_guess[band]*(1 + 0.1*srandu(self.nwalkers))

        self._guess=guess
        return guess


    def _calc_result(self):
        """
        Some extra stats for simple models
        """

        if self.g_prior is not None:
            self._set_g_prior_vals()
            self._remove_zero_prior()

        super(MCMCSimple,self)._calc_result()

        g1i=self.g1i
        g2i=self.g2i

        self._result['g'] = self._result['pars'][g1i:g1i+2].copy()
        self._result['g_cov'] = self._result['pars_cov'][g1i:g1i+2, g1i:g1i+2].copy()

        self._result['nuse'] = self.trials.shape[0]

        if self.do_lensfit:
            g_sens=self._get_lensfit_gsens(self._result['pars'])
            self._result['g_sens']=g_sens

        if self.do_pqr:
            P,Q,R = self._get_PQR()
            self._result['P']=P
            self._result['Q']=Q
            self._result['R']=R

    def _get_lensfit_gsens(self, pars, gprior=None):
        """
        Miller et al. 2007 style sensitivity

        zero prior values should be removed before calling
        """

        g1i=self.g1i
        g2i=self.g2i

        g1vals=self.trials[:,g1i]
        g2vals=self.trials[:,g2i]

        dpri_by_g1 = self.g_prior.dbyg1_array(g1vals,g2vals)
        dpri_by_g2 = self.g_prior.dbyg2_array(g1vals,g2vals)

        g=pars[g1i:g1i+2]
        g1diff = g[0]-g1vals
        g2diff = g[1]-g2vals

        gsens = numpy.zeros(2)

        R1 = g1diff*dpri_by_g1
        R2 = g2diff*dpri_by_g2

        gsens[0]= 1.- R1.mean()
        gsens[1]= 1.- R2.mean()

        return gsens

    def _get_PQR(self):
        """
        get the marginalized P,Q,R from Bernstein & Armstrong

        If the prior is already in our mcmc chain, so we need to divide by the
        prior everywhere.

        zero prior values should be removed prior to calling
        """

        
        g1=self.trials[:,self.g1i]
        g2=self.trials[:,self.g2i]

        sh=self.shear_expand
        if sh is None:
            # expand around zero
            if hasattr(self.g_prior,'get_pqr'):
                Pi,Qi,Ri = self.g_prior.get_pqr(g1,g2)
            else:
                Pi,Qi,Ri = self.g_prior.get_pqr_num(g1,g2)
        else:
            # expand around a requested value.  BA analytic formulas
            # don't support this yet...
            #Pi,Qi,Ri = self.g_prior.get_pqr_num(g1,g2,s1=sh[0], s2=sh[1])
            Pi,Qi,Ri = self.g_prior.get_pqr_expand(g1,g2, sh[0], sh[1])

        P,Q,R = self._get_mean_pqr(Pi,Qi,Ri)

        return P,Q,R

    def _get_mean_pqr(self, Pi, Qi, Ri):
        """
        Get the mean P,Q,R marginalized over priors.  Optionally weighted for
        importance sampling
        """

        P = Pi.mean()
        Q = Qi.mean(axis=0)
        R = Ri.mean(axis=0)

        return P,Q,R

    def _remove_zero_prior(self):
        """
        """
        g_prior = self.g_prior_vals

        w,=numpy.where(g_prior > 0)
        if w.size == 0:
            raise ValueError("no prior values > 0!")

        ndiff=g_prior.size-w.size
        if ndiff > 0:
            print >>stderr,'        removed zero priors:',ndiff
            self.g_prior_vals = self.g_prior_vals[w]
            self.trials = self.trials[w,:]

        return ndiff


class MCMCSimpleFixed(MCMCSimple):
    """
    Fix everything but shapes
    """
    def __init__(self, image, weight, jacobian, model, **keys):
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
            g = numpy.sqrt(pars[2]**2 + pars[3]**2)
            if g > self.g_prior.gmax:
                raise GMixRangeError("g too big")
 
        return lnp

    def _get_band_pars(self, pars, band):
        bpars= self.fixed_pars[ [0,1,2,3,4,5+band] ]
        bpars[2:2+2] = pars
        return bpars

    def get_par_names(self):
        return ['g1','g2']


class MCMCSimpleJointTF(MCMCSimple):
    """
    Add additional features to the base class to support simple models
    """
    def __init__(self, image, weight, jacobian, model, **keys):
        super(MCMCSimpleJointTF,self).__init__(image, weight, jacobian, model, **keys)

        assert self.joint_TF_prior is not None,"send joint_TF_prior"
        assert self.nband == 1, "add support for multi-band and joint prior"

    def _get_priors(self, pars):
        """
        # go in simple
        add any priors that were sent on construction
        
        """
        raise ValueError("fix to check gmax and remove during check")
        lnp=0.0
        
        if self.cen_prior is not None:
            lnp += self.cen_prior.get_lnprob(pars[0], pars[1])

        if self.g_prior is not None and self.g_prior_during:
            lnp += self.g_prior.get_lnprob_scalar2d(pars[2], pars[3])
        
        jlnp = self.joint_TF_prior.get_lnprob_one(pars[4:])
        lnp += jlnp

        return lnp



class MHSimple(MCMCSimple):
    """
    Metropolis Hastings
    """
    def __init__(self, image, weight, jacobian, model, **keys):
        super(MHSimple,self).__init__(image, weight, jacobian, model, **keys)

        guess=keys.get('guess',None)
        if guess is None:
            if self.full_guess is not None:
                guess=self.full_guess
            else:
                raise ValueError("send guess= or full_guess=")

        self.guess=numpy.array(guess, copy=False)
        self.max_arate=keys.get('max_arate',0.6)

        self.step_sizes = numpy.array( keys['step_sizes'], dtype='f8')
        if self.step_sizes.size != self.npars:
            raise ValueError("got %d step sizes, expected %d" % (step_sizes.size, self.npars) )

    def go(self):
        """
        run trials and calculate stats
        """

        self._initialize_gmix()
        self._do_trials()
        self._calc_result()

    def _do_trials(self):
        """
        Do some trials with the MH sampler
        """
        sampler=MHSampler(self.calc_lnprob, self.step)
        self.sampler=sampler

        pars0 = self.guess
        for i in xrange(self.ntry):
            sampler.run(self.burnin, pars0)
            arate=sampler.get_acceptance_rate()

            if self.min_arate < arate < self.max_arate:
                break

            # safe aim is 0.4
            fac = arate/0.4
            self.step_sizes = self.step_sizes*fac

            print >>stderr,'    BAD ARATE:',arate
            pars0=( sampler.get_trials() )[-1,:]

        pars=sampler.get_trials()
        sampler.run(self.nstep, pars[0,:])

        self.trials = sampler.get_trials()
        self.arate  = sampler.get_acceptance_rate()
        self.tau    = 0.0

        self.flags  = 0
        if self.arate < self.min_arate:
            print >>stderr,'LOW ARATE:',self.arate
            self.flags |= LOW_ARATE

    def step(self, pars):
        """
        Take a step
        """
        from numpy.random import randn
        newpars = pars + self.step_sizes*randn(self.npars)
        return newpars

    def _initialize_gmix(self):
        """
        Need some valid pars to initialize this
        """
        for i in xrange(10):
            try:
                self._init_gmix_lol(self.guess)
                break
            except GMixRangeError as gerror:
                # make sure we draw random guess if we got failure
                print >>stderr,'failed init gmix lol:',str(gerror)
                print >>stderr,'getting a new guess'
                self.guess=self._get_random_guess()

    def _get_random_guess(self):
        guess=0*self.guess

        # center
        guess[0]=0.1*srandu(self.nwalkers)
        guess[1]=0.1*srandu(self.nwalkers)

        if self.g_prior is not None and self.draw_g_prior:
            guess[2],guess[3]=self.g_prior.sample2d(1)
        else:
            guess[2]=0.1*srandu()
            guess[3]=0.1*srandu()

        guess[4] = self.T_guess*(1 + 0.1*srandu())

        for band in xrange(self.nband):
            guess[5+band] = self.counts_guess[band]*(1 + 0.1*srandu())

        return guess


class MCMCBDC(MCMCSimple):
    """
    Add additional features to the base class to support coelliptical bulge+disk
    """
    def __init__(self, image, weight, jacobian, **keys):
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
        
        note g prior is *not* applied during the likelihood exploration
        if do_lensfit=True or do_pqr=True
        """
        lnp=0.0
        
        if self.cen_prior is not None:
            lnp += self.cen_prior.get_lnprob(pars[0], pars[1])

        if self.g_prior is not None:
            # may have bounds
            g = numpy.sqrt(pars[2]**2 + pars[3]**2)
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
                names += ['Fb_%s' % i]
            for band in xrange(self.nband):
                names += ['Fd_%s' % i]

        return names


class MCMCBDF(MCMCSimple):
    """
    Add additional features to the base class to support simple models
    """
    def __init__(self, image, weight, jacobian, **keys):
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
        
        note g prior is *not* applied during the likelihood exploration
        if do_lensfit=True or do_pqr=True
        """
        lnp=0.0
        
        if self.cen_prior is not None:
            lnp += self.cen_prior.get_lnprob(pars[0], pars[1])

        if self.g_prior is not None:
            # may have bounds
            g = numpy.sqrt(pars[2]**2 + pars[3]**2)
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
        return pars[ [0,1,2,3,4, Fbstart+band, Fdstart+band] ]


    def get_par_names(self):
        names=['cen1','cen2', 'g1','g2','T']
        if self.nband == 1:
            names += ['Fb','Fd']
        else:
            fbnames = []
            fdnames = []
            for band in xrange(self.nband):
                fbnames.append('Fb_%s' % i)
                fdnames.append('Fd_%s' % i)
            names += fbnames
            names += fdnames
        return names






class MCMCGaussPSF(MCMCSimple):
    def __init__(self, image, weight, jacobian, **keys):
        model=gmix.GMIX_GAUSS
        if 'psf' in keys:
            raise RuntimeError("don't send psf= when fitting a psf")
        super(MCMCGaussPSF,self).__init__(image, weight, jacobian, model, **keys)


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


def extract_mcmc_stats(data, weights=None):
    if weights is not None:
        return _extract_weighted_stats(data, weights)
    else:
        return _extract_stats(data)

def _extract_stats(data):
    ntrials=data.shape[0]
    npar = data.shape[1]

    means = numpy.zeros(npar,dtype='f8')
    cov = numpy.zeros( (npar,npar), dtype='f8')

    for i in xrange(npar):
        means[i] = data[:, i].mean()

    num=ntrials

    for i in xrange(npar):
        idiff = data[:,i]-means[i]
        for j in xrange(i,npar):
            if i == j:
                jdiff = idiff
            else:
                jdiff = data[:,j]-means[j]

            cov[i,j] = (idiff*jdiff).sum()/(num-1)

            if i != j:
                cov[j,i] = cov[i,j]

    return means, cov

def _extract_weighted_stats(data, weights):
    if weights.size != data.shape[0]:
        raise ValueError("weights not same size as data")

    npar = data.shape[1]

    wsum = weights.sum()

    if wsum <= 0.0:
        for i in xrange(data.shape[0]):
            print_pars(data[i,:])
        raise ValueError("wsum <= 0: %s" % wsum)

    means = numpy.zeros(npar,dtype='f8')
    cov = numpy.zeros( (npar,npar), dtype='f8')

    for i in xrange(npar):
        dsum = (data[:, i]*weights).sum()
        means[i] = dsum/wsum

    for i in xrange(npar):
        idiff = data[:,i]-means[i]
        for j in xrange(i,npar):
            if i == j:
                jdiff = idiff
            else:
                jdiff = data[:,j]-means[j]

            wvar = ( weights*idiff*jdiff ).sum()/wsum
            cov[i,j] = wvar

            if i != j:
                cov[j,i] = cov[i,j]

    return means, cov


def test_gauss_psf_graph(counts=100.0, noise=0.1, nimages=10, n=10, groups=True, jfac=0.27):
    import pylab
    import cProfile

    import pycallgraph
    from pycallgraph import PyCallGraph
    from pycallgraph.output import GraphvizOutput

    graphviz = GraphvizOutput()
    output='profile-nimages%02d.png' % nimages
    print 'profile image:',output
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
    wt=numpy.zeros(im.shape) + 1./noise**2
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

            print res['g']

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
    wt=numpy.zeros(im.shape) + 1./noise**2
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
    print secper,'seconds per'

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
    wt=numpy.zeros(im.shape) + 1./noise_pix**2

    imlist=[im]*nimages
    wtlist=[wt]*nimages
    jlist=[j]*nimages


    mc=MCMCGaussPSF(imlist, wtlist, jlist, T=Tsky, counts=counts_sky, burnin=400)
    mc.go()

    res=mc.get_result()

    print_pars(res['pars'], front='pars:', stream=stderr)
    print_pars(res['pars_err'], front='perr:', stream=stderr)
    s=mc.get_flux_scaling()
    #print 'flux in image coords: %.4g +/- %.4g' % (res['pars'][-1]*s,res['pars_err'][-1]*s)

    gmfit=mc.get_gmix()
    imfit=gmfit.make_image(im.shape, jacobian=j)

    imdiff=im-imfit
    images.compare_images(im, imfit, label1='data',label2='fit')
    
    mcmc.plot_results(mc.get_trials())

def test_model(model, counts_sky=100.0, noise_sky=0.001, nimages=1, jfac=0.27):
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

    pars_obj = [0.0, 0.0, g1_obj, g2_obj, Tsky_obj, counts_sky_obj]
    gm_obj0=gmix.GMixModel(pars_obj, model)

    gm=gm_obj0.convolve(gm_psf)

    im_psf=gm_psf.make_image(dims, jacobian=j)
    im_psf[:,:] += noise_pix_psf*numpy.random.randn(im_psf.size).reshape(im_psf.shape)
    wt_psf=numpy.zeros(im_psf.shape) + 1./noise_pix_psf**2

    im_obj=gm.make_image(dims, jacobian=j)
    im_obj[:,:] += noise_pix_obj*numpy.random.randn(im_obj.size).reshape(im_obj.shape)
    wt_obj=numpy.zeros(im_obj.shape) + 1./noise_pix_obj**2

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
    print 'psf numiter:',res_psf['numiter'],'fdiff:',res_psf['fdiff']

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
                      T=Tsky_obj, counts=counts_sky_obj)
    mc_obj.go()

    res_obj=mc_obj.get_result()

    print_pars(res_obj['pars'], front='pars_obj:', stream=stderr)
    print_pars(res_obj['pars_err'], front='perr_obj:', stream=stderr)
    print 'Tpix: %.4g +/- %.4g' % (res_obj['pars'][4]/jfac2, res_obj['pars_err'][4]/jfac2)

    gmfit0=mc_obj.get_gmix()
    gmfit=gmfit0.convolve(psf_fit)
    imfit_obj=gmfit.make_image(im_obj.shape, jacobian=j)

    images.compare_images(im_obj, imfit_obj, label1=model,label2='fit')
    mcmc.plot_results(mc_obj.get_trials())


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
    import pprint
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
    wt_obj=numpy.zeros(im_obj.shape) + 1./noise_pix_obj**2

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
    print 'psf numiter:',res_psf['numiter'],'fdiff:',res_psf['fdiff']

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

    pprint.pprint(res_obj)
    print_pars(res_obj['pars'], front='pars_obj:', stream=stderr)
    print_pars(res_obj['pars_err'], front='perr_obj:', stream=stderr)
    print 'Tpix: %.4g +/- %.4g' % (res_obj['pars'][4]/jfac2, res_obj['pars_err'][4]/jfac2)
    if do_lensfit:
        print 'gsens:',res_obj['g_sens']
    if do_pqr:
        print 'P:',res_obj['P']
        print 'Q:',res_obj['Q']
        print 'R:',res_obj['R']

    gmfit0=mc_obj.get_gmix()
    gmfit=gmfit0.convolve(psf_fit)
    imfit_obj=gmfit.make_image(im_obj.shape, jacobian=j)

    images.compare_images(im_obj, imfit_obj, label1=model,label2='fit')
    mcmc.plot_results(mc_obj.get_trials())


def test_model_mb(model,
                  counts_sky=[100.0,88., 77., 95.0], # true counts each band
                  noise_sky=[0.1,0.1,0.1,0.1],
                  nimages=10, # in each band
                  jfac=0.27,
                  do_lensfit=False,
                  do_pqr=False,
                  profile=False, groups=False,
                  burnin=400,
                  draw_g_prior=False,
                  png=None,
                  show=False):
    """
    testing mb stuff
    """
    import pprint
    import images
    import mcmc
    from . import em
    import time

    import pycallgraph
    from pycallgraph import PyCallGraph
    from pycallgraph.output import GraphvizOutput

    if groups:
        gstr='grouped'
    else:
        gstr='nogroup'
 
    jfac2=jfac**2

    dims=[25,25]
    cen=[dims[0]/2., dims[1]/2.]

    # object pars
    g1_obj=0.1
    g2_obj=0.05
    Tpix_obj=16.0
    Tsky_obj=Tpix_obj*jfac2

    true_pars=numpy.array([0.0,0.0,g1_obj,g2_obj,Tsky_obj]+counts_sky)

    #jlist=[Jacobian(cen[0],cen[1], jfac, 0.0, 0.0, jfac),
    #       Jacobian(cen[0]+0.5,cen[1]-0.1, jfac, 0.001, -0.001, jfac),
    #       Jacobian(cen[0]-0.2,cen[1]+0.1, jfac, -0.001, 0.001, jfac)]

    #
    # simulation
    #

    counts_sky_psf=100.0
    counts_pix_psf=counts_sky_psf/jfac2

    nband=len(counts_sky)
    im_lol = []
    wt_lol = []
    j_lol = []
    psf_lol = []

    tmpsf=0.0
    for band in xrange(nband):

        im_list=[]
        wt_list=[]
        j_list=[]
        psf_list=[]

        # not always at same center
        j=Jacobian(cen[0]+srandu(),
                   cen[1]+srandu(),
                   jfac,
                   0.0,
                   0.0,
                   jfac)
        counts_sky_obj=counts_sky[band]
        noise_sky_obj=noise_sky[band]
        counts_pix_obj=counts_sky_obj/jfac2
        noise_pix_obj=noise_sky_obj/jfac2

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

            im_psf=gm_psf.make_image(dims, jacobian=j, nsub=16)
            im_obj=gm.make_image(dims, jacobian=j, nsub=16)

            im_obj[:,:] += noise_pix_obj*numpy.random.randn(im_obj.size).reshape(im_obj.shape)
            wt_obj=numpy.zeros(im_obj.shape) + 1./noise_pix_obj**2

            # psf using EM
            tmpsf0=time.time()

            im_psf_sky,sky=em.prep_image(im_psf)
            mc_psf=em.GMixEM(im_psf_sky, jacobian=j)
            emo_guess=gm_psf.copy()
            emo_guess._data['p'] = 1.0
            mc_psf.go(emo_guess, sky)
            res_psf=mc_psf.get_result()

            tmpsf+=time.time()-tmpsf0
            #print 'psf numiter:',res_psf['numiter'],'fdiff:',res_psf['fdiff']

            psf_fit=mc_psf.get_gmix()

            im_list.append(im_obj)
            wt_list.append(wt_obj)
            j_list.append(j)
            psf_list.append(psf_fit)
        im_lol.append( im_list )
        wt_lol.append( wt_list )
        j_lol.append( j_list )
        psf_lol.append( psf_list )

    tmrest=time.time()
    #
    # priors
    # not really accurate since we are not varying the input
    #

    cen_prior=priors.CenPrior(0.0, 0.0, 0.1, 0.1)
    T_prior=priors.LogNormal(Tsky_obj, 0.1*Tsky_obj)
    counts_prior=[priors.LogNormal(counts_sky_obj, 0.1*counts_sky_obj)]*nband
    g_prior = priors.GPriorBA(0.3)

    #
    # fitting
    #

    if profile:
        name='profile-mb-%s-%dbands-%iimages-%s.png' % (model,nband,nimages,gstr)
        graphviz = GraphvizOutput()
        print 'profile image:',name
        graphviz.output_file = name
        config=pycallgraph.Config(groups=groups)

        with PyCallGraph(config=config, output=graphviz):
            mc_obj=MCMCSimple(im_lol, wt_lol, j_lol, model,
                              psf=psf_lol,
                              T=Tsky_obj*(1. + 0.1*srandu()),
                              counts=counts_sky*(1. + 0.1*srandu(nband)),
                              cen_prior=cen_prior,
                              T_prior=T_prior,
                              counts_prior=counts_prior,
                              g_prior=g_prior,
                              do_lensfit=do_lensfit,
                              do_pqr=do_pqr, mca_a=3.,
                              draw_g_prior=draw_g_prior,
                              burnin=burnin)
            mc_obj.go()
    else:
        mc_obj=MCMCSimple(im_lol, wt_lol, j_lol, model,
                          psf=psf_lol,
                          T=Tsky_obj*(1. + 0.1*srandu()),
                          counts=counts_sky*(1. + 0.1*srandu(nband)),
                          cen_prior=cen_prior,
                          T_prior=T_prior,
                          counts_prior=counts_prior,
                          g_prior=g_prior,
                          do_lensfit=do_lensfit,
                          do_pqr=do_pqr, mca_a=3.,
                          draw_g_prior=draw_g_prior,
                          burnin=burnin)
        mc_obj.go()

    res_obj=mc_obj.get_result()
    tmrest = time.time()-tmrest

    #pprint.pprint(res_obj)
    print 'arate:',res_obj['arate']
    print_pars(true_pars, front='true:    ', stream=stderr)
    print_pars(res_obj['pars'], front='pars_obj:', stream=stderr)
    print_pars(res_obj['pars_err'], front='perr_obj:', stream=stderr)
    #print 'Tpix: %.4g +/- %.4g' % (res_obj['pars'][4]/jfac2, res_obj['pars_err'][4]/jfac2)
    if do_lensfit:
        print 'gsens:',res_obj['g_sens']
    if do_pqr:
        print 'P:',res_obj['P']
        print 'Q:',res_obj['Q']
        print 'R:',res_obj['R']

    #gmfit0=mc_obj.get_gmix()
    #gmfit=gmfit0.convolve(psf_fit)
    #imfit_obj=gmfit.make_image(im_obj.shape, jacobian=j)

    #images.compare_images(im_obj, imfit_obj, label1=model,label2='fit')

           
    names=['cen1','cen2','g1','g2','T'] + ['counts%s' % (i+1) for i in xrange(nband)]
    plt=mcmc.plot_results(mc_obj.get_trials(),names=names,show=show)
    if show:
        plt.show()
    elif png is not None:
        plt.write_img(800,800,png)

    tmtot=tmrest + tmpsf
    print 'time total:',tmtot
    print 'time psf:  ',tmpsf
    print 'time rest: ',tmrest

    return tmtot,res_obj



def test_model_priors_anze(model,
                      counts_sky=100.0,
                      noise_sky=0.01,
                      nimages=1,
                      jfac=0.27,
                      do_lensfit=False,
                      do_pqr=False):
    """
    testing jacobian stuff
    """
    import pprint
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
    wt_obj=numpy.zeros(im_obj.shape) + 1./noise_pix_obj**2

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
    print 'psf numiter:',res_psf['numiter'],'fdiff:',res_psf['fdiff']

    psf_fit=mc_psf.get_gmix()
    imfit_psf=mc_psf.make_image(counts=im_psf.sum())
    #images.compare_images(im_psf, imfit_psf, label1='psf',label2='fit')

    # obj
    jlist=[j]*nimages
    imlist_obj=[im_obj]*nimages
    wtlist_obj=[wt_obj]*nimages
    psf_fit_list=[psf_fit]*nimages

    mc_obj=MCMCSimpleAnze(imlist_obj, wtlist_obj, jlist, model,
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

    """
    res_obj=mc_obj.get_result()

    pprint.pprint(res_obj)
    print_pars(res_obj['pars'], front='pars_obj:', stream=stderr)
    print_pars(res_obj['pars_err'], front='perr_obj:', stream=stderr)
    print 'Tpix: %.4g +/- %.4g' % (res_obj['pars'][4]/jfac2, res_obj['pars_err'][4]/jfac2)
    if do_lensfit:
        print 'gsens:',res_obj['g_sens']
    if do_pqr:
        print 'P:',res_obj['P']
        print 'Q:',res_obj['Q']
        print 'R:',res_obj['R']

    gmfit0=mc_obj.get_gmix()
    gmfit=gmfit0.convolve(psf_fit)
    imfit_obj=gmfit.make_image(im_obj.shape, jacobian=j)

    images.compare_images(im_obj, imfit_obj, label1=model,label2='fit')
    mcmc.plot_results(mc_obj.get_trials())
    """



def _get_test_psf_flux_pars(ngauss, jfac, counts_sky):

    jfac2=jfac**2
    if ngauss==1:
        e1=0.1*srandu()
        e2=0.1*srandu()
        Tpix=4.0*(1.0 + 0.2*srandu())

        Tsky=Tpix*jfac2
        pars=numpy.array([counts_sky,
                          0.0,
                          0.0,
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
        pars=numpy.array([counts_frac1*counts_sky,
                          0.0,
                          0.0,
                          (T1sky/2.)*(1-e1_1),
                          (T1sky/2.)*e2_1,
                          (T1sky/2.)*(1+e1_1),

                          counts_frac2*counts_sky,
                          0.0,
                          0.0,
                          (T2sky/2.)*(1-e1_2),
                          (T2sky/2.)*e2_2,
                          (T2sky/2.)*(1+e1_2)])


    else:
        raise ValueError("bad ngauss: %s" % ngauss)

    gm=gmix.GMix(pars=pars)
    return gm

def test_psf_flux(ngauss,
                  counts_sky=100.0,
                  noise_sky=0.01,
                  nimages=1,
                  jfac=0.27):
    """
    testing jacobian stuff

    flux fit time is negligible, EM fitting dominates
    """
    from .em import GMixMaxIterEM
    import pprint
    import images
    import mcmc
    from . import em

    dims=[25,25]
    cen=[dims[0]/2., dims[1]/2.]
    cenfac=0.0
    jfac2=jfac**2
    noise_pix=noise_sky/jfac2

    im_list=[]
    wt_list=[]
    j_list=[]
    psf_list=[]

    ntry=10

    tm_em=0.0
    for i in xrange(nimages):
        # gmix is in sky coords
        gm=_get_test_psf_flux_pars(ngauss, jfac, counts_sky)
        j=Jacobian(cen[0]+cenfac*srandu(),cen[1]+cenfac*srandu(), jfac, 0.0, 0.0, jfac)

        im0=gm.make_image(dims, jacobian=j)
        im = im0 + noise_pix*numpy.random.randn(im0.size).reshape(dims)

        im0_skyset,sky=em.prep_image(im0)
        mc=em.GMixEM(im0_skyset, jacobian=j)

        print 'true:'
        print gm
        # gm is also guess
        gm_guess=gm.copy()
        gm_guess.set_psum(1.0)
        gm_guess.set_cen(0.0, 0.0)
        tm0_em=time.time()
        for k in xrange(ntry):
            try:
                mc.go(gm_guess, sky, tol=1.e-5)
                break
            except GMixMaxIterEM:
                if (k==ntry-1):
                    raise
                else:
                    res=mc.get_result()
                    print 'try:',k,'fdiff:',res['fdiff'],'numiter:',res['numiter']
                    print mc.get_gmix()
                    gm_guess.set_cen(0.1*srandu(), 0.1*srandu())
                    gm_guess._data['irr'] = gm._data['irr']*(1.0 + 0.1*srandu(ngauss))
                    gm_guess._data['icc'] = gm._data['icc']*(1.0 + 0.1*srandu(ngauss))
        psf_fit=mc.get_gmix()
        tm_em += time.time()-tm0_em

        wt=0.0*im.copy() + 1./noise_pix**2

        im_list.append(im)
        wt_list.append(wt)
        j_list.append(j)
        psf_list.append(psf_fit)
        #psf_list.append(gm)

        #print 'fit: ',psf_fit
        res=mc.get_result()
        print i+1,res['numiter']


    tm_fit=time.time()
    fitter=PSFFluxFitter(im_list, wt_list, j_list, psf_list)
    fitter.go()
    tm_fit=time.time()-tm_fit

    res=fitter.get_result()

    #print res

    return res['flux'], res['flux_err'], tm_fit, tm_em

def profile_test_psf_flux(ngauss,
                          counts_sky=100.0,
                          noise_sky=0.01,
                          nimages=1,
                          jfac=0.27,
                          groups=False):

    import pycallgraph
    from pycallgraph import PyCallGraph
    from pycallgraph.output import GraphvizOutput

    graphviz = GraphvizOutput()
    output='profile-psfflux-ngauss%02d-%02d.png' % (ngauss,nimages)
    print 'profile image:',output
    graphviz.output_file = output
    config=pycallgraph.Config(groups=groups)


    with PyCallGraph(config=config, output=graphviz):
        for i in xrange(10):
            test_psf_flux(ngauss,
                          counts_sky=counts_sky,
                          noise_sky=noise_sky,
                          nimages=nimages,
                          jfac=jfac)

class MHSampler(object):
    """
    Run a Monte Carlo Markov Chain (MCMC) using Metropolis-Hastings
    
    The user inputs an object that has the methods "step" and "get_loglike"
    that can be used to generate the chain.

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

    examples
    ---------
    m=mcmc.MCMC(lnprob_func, stepper, seed=34231)
    m.run(nstep, par_guess)
    trials = m.get_trials()
    loglike = m.get_loglike()
    arate = m.get_acceptance_rate()

    """
    def __init__(self, lnprob_func, stepper, seed=None):
        self._lnprob_func=lnprob_func
        self._stepper=stepper
        self.reset(seed=seed)

    def reset(self, seed=None):
        """
        Clear all data
        """
        self._trials=None
        self._loglike=None
        self._accepted=None
        numpy.random.seed(seed)

    def run(self, nstep, pars_start):
        """
        Run the MCMC chain.  Append new steps if trials already
        exist in the chain.

        parameters
        ----------
        nstep: Number of steps in the chain.
        pars_start:  Starting point for the chain in the n-dimensional
                parameters space.
        """
        
        self._init_data(nstep, pars_start)

        for i in xrange(1,nstep):
            self._step()
        
        self._arate = self._accepted.sum()/float(self._accepted.size)

    def get_trials(self):
        """
        Get the trials array
        """
        return self._trials

    def get_loglike(self):
        """
        Get the trials array
        """
        return self._loglike

    def get_acceptance_rate(self):
        """
        Get the acceptance rate
        """
        return self._arate

    def get_accepted(self):
        """
        Get the accepted array
        """
        return self._accepted

    def _step(self):
        """
        Take the next step in the MCMC chain.  
        
        Calls the stepper lnprob_func methods sent during
        construction.  If the new loglike is not greater than the previous, or
        a uniformly generated random number is greater than the the ratio of
        new to old likelihoods, the new step is not used, and the new
        parameters are the same as the old.  Otherwise the new step is kept.

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

        randnum = numpy.random.random()
        log_randnum = numpy.log(randnum)

        # we allow use of -infinity as a sign we are out of bounds
        if (numpy.isfinite(newlike) 
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

    def _init_data(self, nstep, pars_start):
        """
        Set the trials and accept array.
        """

        pars_start=numpy.array(pars_start,dtype='f8')
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

