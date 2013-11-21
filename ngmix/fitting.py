from sys import stdout, stderr
import numpy
from . import gmix
from .gmix import _exp3_ivals,_exp3_lookup
from .jacobian import Jacobian, UnitJacobian

from . import priors
from .priors import srandu

from .gexceptions import GMixRangeError

import time

LOWVAL=-9999.0e47

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

        # in this case, image, weight, jacobian, psf are going to
        # be lists of lists.

        # call this first, others depend on it
        self._set_lists(image, weight, jacobian, **keys)

        self.model=gmix.get_model_num(model)
        self.model_name=gmix.get_model_name(self.model)
        self._set_npars()

        # the function to be called to fill a gaussian mixture
        self._set_fill_call()

        self.g_prior = keys.get('g_prior',None)
        self.cen_prior = keys.get('cen_prior',None)
        self.T_prior = keys.get('T_prior',None)
        self.counts_prior = keys.get('counts_prior',None)

        self.totpix=self.verify()

        self.make_plots=keys.get('make_plots',False)

        self._gmix_lol=None
        self._result=None

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
            raise ValueError("bdc not yet implemented")
        else:
            raise GMixFatalError("unsupported model: "
                                 "'%s'" % self.model_name)


    def get_result(self):
        """
        Result will not be non-None until go() is run
        """
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


    def calc_lnprob(self, pars, get_s2nsums=False):
        """
        This is all we use for mcmc approaches, but also used generally for the
        "get_fit_stats" method.  For the max likelihood fitter we also have a
        _get_ydiff method
        """
        try:

            lnprob = self._get_priors(pars)
            s2n_numer=0.0
            s2n_denom=0.0

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

                    lnprob += res[0]
                    s2n_numer += res[1]
                    s2n_denom += res[2]

        except GMixRangeError:
            lnprob = LOWVAL

        if get_s2nsums:
            return lnprob, s2n_numer, s2n_denom
        else:
            return lnprob

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




    def _get_band_pars(self, pars, band):
        return pars[ [0,1,2,3,4,5+band] ]


    def _init_gmix_lol(self, pars):
        """
        initialize the list of lists of gmix
        """
        self._gmix_lol0 = []
        self._gmix_lol  = []

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

            self._gmix_lol0.append(gmix_list0)
            self._gmix_lol.append(gmix_list)

    def _fill_gmix_lol(self, pars):
        """
        Fill the list of lists of gmix objects, potentially convolved with the
        psf in the individual images
        """
        if self._gmix_lol is None:
            self._init_gmix_lol(pars)
        else:
            for band in xrange(self.nband):
                gmix_list0=self._gmix_lol0[band]
                gmix_list=self._gmix_lol[band]

                band_pars=self._get_band_pars(pars, band)
                psf_list=self.psf_lol[band]

                for i,psf in enumerate(psf_list):
                    gm0=gmix_list0[i]
                    gm=gmix_list[i]

                    # Calling the python versions was a huge time sync!
                    #gm0.fill(band_pars)
                    self._fill_gmix_func(gm0._data, band_pars)
                    # Calling the python version was a huge time sync!
                    #gmix.convolve_fill(gm, gm0, psf)
                    gmix._convolve_fill(gm._data, gm0._data, psf._data)

    def _get_counts_guess(self, **keys):
        cguess=keys.get('counts',None)
        if cguess is None:
            cguess = self._get_median_counts()
        else:
            cguess=numpy.array(cguess)
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

        self.make_plots=keys.get('make_plots',False)

        self._result=None

    def go(self):
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

        flux_err = numpy.sqrt( chi2/msq_sum/(self.totpix-1) )
        self._result={'flags':0,
                      'chi2':chi2,
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

class MCMCBase(FitterBase):
    """
    A base class for MCMC runs.  Inherits from overall fitter base class.
    """
    def __init__(self, image, weight, jacobian, model, **keys):
        super(MCMCBase,self).__init__(image, weight, jacobian, model, **keys)

        self.doiter=keys.get('iter',True)
        self.nwalkers=keys.get('nwalkers',20)
        self.nstep=keys.get('nstep',200)
        self.burnin=keys.get('burnin',400)
        self.mca_a=keys.get('mca_a',2.0)

        self.draw_g_prior=keys.get('draw_g_prior',True)

        self.do_pqr=keys.get('do_pqr',False)
        self.do_lensfit=keys.get('do_lensfit',False)

        if (self.do_lensfit or self.do_pqr) and self.g_prior is None:
            raise ValueError("send g_prior for lensfit or pqr")

        # we don't apply the prior during the likelihood exploration to avoid
        # possibly dividing by zero during the lensfit and pqr calculations
        if self.do_pqr or self.do_lensfit:
            self.g_prior_during=False
        else:
            self.g_prior_during=True

        self.trials=None


    def get_trials(self):
        """
        Get the set of trials from the production run
        """
        return self.trials

    def go(self):
        """
        Run the mcmc sampler and calculate some statistics
        """
        self.sampler=self._do_trials()

        self.trials  = self.sampler.flatchain

        self.lnprobs = self.sampler.lnprobability.reshape(self.nwalkers*self.nstep)
        self.lnprobs -= self.lnprobs.max()

        # get the expectation values, sensitivity and errors
        self._calc_result()

        if self.make_plots:
            self._doplots()


    def _do_trials(self):
        """
        Actually run the sampler
        """
        # over-ridden
        guess=self._get_guess()

        sampler = self._get_sampler()

        pos, prob, state = sampler.run_mcmc(guess, self.burnin)
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(pos, self.nstep)

        if self.doiter:
            while True:
                try:
                    acor=sampler.acor
                    tau = (acor/self.nstep).max()
                    if tau > 0.1:
                        print "tau",tau,"greater than 0.1"
                    else:
                        break
                except:
                    # something went wrong with acor, run some more
                    pass
                pos, prob, state = sampler.run_mcmc(pos, self.nstep)


        return sampler

    def _get_sampler(self):
        """
        Instantiate the sampler
        """
        import emcee
        sampler = emcee.EnsembleSampler(self.nwalkers, 
                                        self.npars, 
                                        self.calc_lnprob,
                                        a=self.mca_a)
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

        pars,pcov=self._get_trial_stats()
 
        arates = self.sampler.acceptance_fraction
        arate = arates.mean()

        self._result={'flags':0,
                      'model':self.model,
                      'pars':pars,
                      'pcov':pcov,
                      'perr':numpy.sqrt(numpy.diag(pcov)),
                      'arate':arate}

        stats = self.get_fit_stats(pars)
        self._result.update(stats)

    def _get_trial_stats(self):
        """
        Get the means and covariance for the trials
        """
        if self.g_prior is not None and not self.g_prior_during:
            raise RuntimeError("prior during: don't know how to get g1,g2 "
                               "values in general. You need to over-ride")
        else:
            pars,pcov = extract_mcmc_stats(self.trials)
        
        return pars,pcov


    def _get_guess(self):
        raise RuntimeError("over-ride me")

class MCMCSimple(MCMCBase):
    def __init__(self, image, weight, jacobian, model, **keys):
        super(MCMCSimple,self).__init__(image, weight, jacobian, model, **keys)

        self.T_guess=keys.get('T',4.0)
        self.counts_guess=self._get_counts_guess(**keys)

        ncg=self.counts_guess.size
        if ncg != self.nband:
                raise ValueError("counts_guess size %s doesn't match "
                                 "number of bands %s" % (ncg,self.nband))

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

        if self.g_prior is not None and self.g_prior_during:
            lnp += self.g_prior.get_lnprob_scalar2d(pars[2], pars[3])
        
        if self.T_prior is not None:
            lnp += self.T_prior.get_lnprob_scalar(pars[4])

        if self.counts_prior is not None:
            for i,cp in enumerate(self.counts_prior):
                counts=pars[5+i]
                lnp += cp.get_lnprob_scalar(counts)

        return lnp


    def _get_guess(self):
        """
        # go in simple
        The counts guess is stupid unless you have a well-trimmed
        PSF image
        """

        guess=numpy.zeros( (self.nwalkers,self.npars) )

        # center
        guess[:,0]=0.1*srandu(self.nwalkers)
        guess[:,1]=0.1*srandu(self.nwalkers)

        if self.draw_g_prior:
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
        super(MCMCSimple,self)._calc_result()

        self._result['g'] = self._result['pars'][2:2+2].copy()
        self._result['gcov'] = self._result['pcov'][2:2+2, 2:2+2].copy()

        if self.do_lensfit:
            gsens=self._get_lensfit_gsens(self._result['pars'])
            self._result['gsens']=gsens

        if self.do_pqr:
            P,Q,R = self._get_PQR()
            self._result['P']=P
            self._result['Q']=Q
            self._result['R']=R


    def _get_trial_stats(self):
        """
        Get the stats from the trials
        """
        if self.g_prior is not None and not self.g_prior_during:
            g1vals = self.trials[:,2]
            g2vals = self.trials[:,3]
            gprior  = self.g_prior.get_prob_array2d(g1vals,g2vals)
            pars,pcov = extract_mcmc_stats(self.trials,weights=gprior)
        else:
            pars,pcov = extract_mcmc_stats(self.trials)
        
        return pars,pcov

    def _get_lensfit_gsens(self, pars, gprior=None):

        if self.g_prior is not None:
            g1vals=self.trials[:,2]
            g2vals=self.trials[:,3]

            gprior = self.g_prior.get_prob_array2d(g1vals,g2vals)

            dpri_by_g1 = self.g_prior.dbyg1_array(g1vals,g2vals)
            dpri_by_g2 = self.g_prior.dbyg2_array(g1vals,g2vals)

            psum = gprior.sum()

            g=pars[2:2+2]
            g1diff = g[0]-g1vals
            g2diff = g[1]-g2vals

            gsens = numpy.zeros(2)
            gsens[0]= 1.-(g1diff*dpri_by_g1).sum()/psum
            gsens[1]= 1.-(g2diff*dpri_by_g2).sum()/psum
        else:
            gsens=numpy.array([1.,1.])

        return gsens

    def _get_PQR(self):
        """
        get the marginalized P,Q,R from Bernstein & Armstrong

        Note if the prior is already in our mcmc chain, so we need to divide by
        the prior everywhere.  Because P*J=P at shear==0 this means P is always
        1

        """

        g1=self.trials[:,2]
        g2=self.trials[:,3]

        P,Q,R = self.g_prior.get_pqr(g1,g2)

        P = P.mean()
        Q = Q.mean(axis=0)
        R = R.mean(axis=0)

        return P,Q,R

class MCMCSimpleAnze(MCMCSimple):
    def __init__(self, image, weight, jacobian, model, **keys):
        super(MCMCSimpleAnze,self).__init__(image, weight, jacobian, model, **keys)

    def go(self):
        import game
        import esutil as eu

        guess=self._get_guess()
        guess1=guess[0,:]
        sampler=game.Game(self.lnprob_many, guess1,
                          sigreg=[0.01, 0.01, 0.1, 0.1, 0.1, 0.1])

        sampler.N1=50
        sampler.N1f=0
        sampler.blow=1.3
        sampler.mineffsamp=500
        sampler.maxiter=1000
        sampler.wemin=1.e-4

        sampler.run()

        print sampler.sample_list[-1]

        npars=guess1.size
        m=numpy.zeros(npars)
        m2=numpy.zeros(npars)
        sw=0.0

        nsamp=len(sampler.sample_list)
        trials=numpy.zeros( (nsamp, npars) )
        weights=numpy.zeros(nsamp)
        for i,sa in enumerate(sampler.sample_list):
            trials[i,:] = sa.pars
            weights[i] = sa.we

        #mcmc.plot_results(mc_obj.get_trials(), weights=weights)

        eu.plotting.bhist( trials[:, 0], weights=weights, binsize=0.0005,title='row')
        eu.plotting.bhist( trials[:, 1], weights=weights, binsize=0.0005,title='col')
        eu.plotting.bhist( trials[:, 2], weights=weights, binsize=0.001,title='g1')
        eu.plotting.bhist( trials[:, 3], weights=weights, binsize=0.001,title='g2')
        eu.plotting.bhist( trials[:, 4], weights=weights, binsize=0.01,title='T')
        eu.plotting.bhist( trials[:, 5], weights=weights, binsize=1.0,title='counts')
        #print 'mean:',(trials*we).sum(axis=1)/we.sum()

        """
        for sa in sampler.sample_list:
            m+=sa.pars*sa.we
            m2+=sa.pars**2*sa.we
            sw+=sa.we
        m/=sw
        m2/=sw
        m2-=m*m
        print m
        print numpy.sqrt(m2)
        """
 
        """
        self.trials  = self.sampler.flatchain

        self.lnprobs = self.sampler.lnprobability.reshape(self.nwalkers*self.nstep)
        self.lnprobs -= self.lnprobs.max()

        # get the expectation values, sensitivity and errors
        self._calc_result()

        if self.make_plots:
            self._doplots()
        """

    def lnprob_many(self, list_of_pars):
        return map(self.calc_lnprob, list_of_pars)

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
    wsum2 = wsum**2

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
    print_pars(res['perr'], front='perr:', stream=stderr)
    s=mc.get_flux_scaling()
    #print 'flux in image coords: %.4g +/- %.4g' % (res['pars'][-1]*s,res['perr'][-1]*s)

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
    print_pars(res_obj['perr'], front='perr_obj:', stream=stderr)
    print 'Tpix: %.4g +/- %.4g' % (res_obj['pars'][4]/jfac2, res_obj['perr'][4]/jfac2)

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
    print_pars(res_obj['perr'], front='perr_obj:', stream=stderr)
    print 'Tpix: %.4g +/- %.4g' % (res_obj['pars'][4]/jfac2, res_obj['perr'][4]/jfac2)
    if do_lensfit:
        print 'gsens:',res_obj['gsens']
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
    print_pars(res_obj['perr'], front='perr_obj:', stream=stderr)
    #print 'Tpix: %.4g +/- %.4g' % (res_obj['pars'][4]/jfac2, res_obj['perr'][4]/jfac2)
    if do_lensfit:
        print 'gsens:',res_obj['gsens']
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
    print_pars(res_obj['perr'], front='perr_obj:', stream=stderr)
    print 'Tpix: %.4g +/- %.4g' % (res_obj['pars'][4]/jfac2, res_obj['perr'][4]/jfac2)
    if do_lensfit:
        print 'gsens:',res_obj['gsens']
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
