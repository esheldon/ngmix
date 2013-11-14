from sys import stdout, stderr
import numpy
from . import gmix
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
    you should just work in sky coordinates.
    
    """
    def __init__(self, image, weight, jacobian, model, **keys):
        self.keys=keys

        self.model=gmix.get_model_num(model)
        self.model_name=gmix.get_model_name(self.model)
        self.npars=gmix.get_model_npars(self.model)

        self.im_list=_get_as_list(image,"image")
        self.wt_list=_get_as_list(weight,"weight")
        self._set_jacob_list(jacobian) # set jacob_list and mean_jacob_det

        self.psf_list=_get_as_list(keys.get('psf',None),
                                   "psf",
                                   allow_none=True)
        self.nimages=len(self.im_list)

        self.g_prior = keys.get('g_prior',None)
        self.cen_prior = keys.get('cen_prior',None)
        self.T_prior = keys.get('T_prior',None)
        self.counts_prior = keys.get('counts_prior',None)

        self.totpix=self.verify()

        self.make_plots=keys.get('make_plots',False)

        self._gmix_list=None
        self._result=None

    def verify(self):
        """
        Make sure lists are equal length, image and weight
        maps same size, etc.
        """
        nim=len(self.im_list)
        nwt=len(self.wt_list)
        if nim != nwt:
            raise ValueError("len(im_list) != len(wt_list) "
                             "(%d != %d)" % (nim,nwt))
        if self.psf_list is not None:
            npsf=len(self.psf_list)
            if npsf != nim:
                raise ValueError("len(im_list) != len(psf_list) "
                                 "(%d != %d)" % (nim,npsf))

        njacob=len(self.jacob_list)
        if njacob != nim:
            raise ValueError("len(im_list) != len(jacob_list) "
                             "(%d != %d)" % (nim,njacob))

        totpix=0
        for i in xrange(nim):
            imsh=self.im_list[i].shape
            wtsh=self.wt_list[i].shape
            if imsh != wtsh:
                raise ValueError("im.shape != wt.shape "
                                 "(%s != %s)" % (imsh,wtsh))
            totpix += self.im_list[i].size


        return totpix

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

    def _init_gmix_list(self, pars):
        if self.psf_list is not None:

            self._gmix_list0 = []
            self._gmix_list  = []
            for psf in self.psf_list:
                gm0=gmix.GMixModel(pars, self.model)
                gm=gm0.convolve(psf)

                self._gmix_list0.append(gm0)
                self._gmix_list.append(gm)

        else:
            self._gmix_list = []
            for i in xrange(self.nimages):
                gm=gmix.GMixModel(pars, self.model)
                self._gmix_list.append(gm)


    def _fill_gmix_list(self, pars):
        """
        Get a list of gmix objects, potentially convolved with
        the psf in the individual images
        """
        if self._gmix_list is None:
            self._init_gmix_list(pars)
        else:
            if self.psf_list is not None:

                for i,psf in enumerate(self.psf_list):
                    gm0=self._gmix_list0[i]
                    gm=self._gmix_list[i]

                    gm0.fill(pars)
                    gmix.convolve_fill(gm, gm0, psf)

            else:
                for gm in self._gmix_list:
                    gm.fill(pars)


    def _get_counts_guess(self, **keys):
        cguess=keys.get('counts',None)
        if cguess is None:
            cguess = self._get_median_counts()
        return cguess

    def _get_median_counts(self):
        """
        median of the counts across all input images
        """
        clist=numpy.zeros(self.nimages)
        for i in xrange(self.nimages):
            im=self.im_list[i]
            j=self.jacob_list[i]
            clist[i] = im.sum()*j._data['det']
        
        return numpy.median(clist)

    def _set_jacob_list(self, jacobian):
        self.jacob_list = _get_as_list(jacobian,"jacobian")
        mean_det=0.0
        for j in self.jacob_list:
            mean_det += j._data['det']
        self.mean_jacob_det=mean_det/len(self.jacob_list)



class MCMCBase(FitterBase):
    def __init__(self, image, weight, jacobian, model, **keys):
        super(MCMCBase,self).__init__(image, weight, jacobian, model, **keys)

        self.doiter=keys.get('iter',True)
        self.nwalkers=keys.get('nwalkers',20)
        self.nstep=keys.get('nstep',200)
        self.burnin=keys.get('burnin',400)
        self.mca_a=keys.get('mca_a',2.0)

        self.draw_g_prior=keys.get('draw_g_prior',True)

        self.T_guess=keys.get('T',4.0)
        self.counts_guess=self._get_counts_guess(**keys)

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
        self.sampler=self._do_trials()

        self.trials  = self.sampler.flatchain

        self.lnprobs = self.sampler.lnprobability.reshape(self.nwalkers*self.nstep)
        self.lnprobs -= self.lnprobs.max()

        # get the expectation values, sensitivity and errors
        self._calc_result()

        if self.make_plots:
            self._doplots()


    def _do_trials(self):

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
        import emcee
        sampler = emcee.EnsembleSampler(self.nwalkers, 
                                        self.npars, 
                                        self._calc_lnprob,
                                        a=self.mca_a)
        return sampler

    def _get_priors(self, pars):
        """
        Basically a placeholder for no priors
        """
        return 0.0

    def _calc_lnprob(self, pars):

        try:

            lnprob = self._get_priors(pars)

            self._fill_gmix_list(pars)

            for i in xrange(self.nimages):
                gm=self._gmix_list[i]
                im=self.im_list[i]
                wt=self.wt_list[i]
                j=self.jacob_list[i]

                lnprob += gm.get_loglike(im, wt, jacobian=j)


        except GMixRangeError:
            lnprob = LOWVAL

        return lnprob


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

        '''
        gmix_list=self._get_gmix_list(pars)
        if gmix_list is None:
            stats={}
        else:
            stats=self._get_fit_stats(gmix_list)
        '''

    def _get_trial_stats(self):
        """
        hmm.... really only know g1,g2 are in 2,3 for simple
        and some other models...
        """
        if self.g_prior is not None and not self.g_prior_during:
            raise RuntimeError("don't know how to get g1,g2 "
                               "values in general. You need to over-ride")
        else:
            pars,pcov = extract_mcmc_stats(self.trials)
        
        return pars,pcov


    def _get_guess(self):
        raise RuntimeError("over-ride me")

class MCMCSimple(MCMCBase):
    def __init__(self, image, weight, jacobian, model, **keys):
        super(MCMCSimple,self).__init__(image, weight, jacobian, model, **keys)


    def _get_priors(self, pars):
        """
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
            lnp += self.counts_prior.get_lnprob_scalar(pars[5])

        return lnp

    def _get_guess(self):
        """
        The counts guess is stupid unless you have a well-trimmed
        PSF image
        """

        guess=numpy.zeros( (self.nwalkers,self.npars) )

        # center
        guess[:,0]=0.1*srandu(self.nwalkers)
        guess[:,1]=0.1*srandu(self.nwalkers)

        guess[:,2]=0.1*srandu(self.nwalkers)
        guess[:,3]=0.1*srandu(self.nwalkers)

        guess[:,4] = self.T_guess*(1 + 0.1*srandu(self.nwalkers))
        guess[:,5] = self.counts_guess*(1 + 0.1*srandu(self.nwalkers))

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
            gsens=self._get_lensfit_gsens(pars)
            self._result['gsens']=gsens

        if self.do_pqr:
            P,Q,R = self._get_PQR()
            self._result['P']=P
            self._result['Q']=Q
            self._result['R']=R

        T,T_err,T_s2n=self._get_T_stats(self._result['pars'],
                                        self._result['pcov'])
        self._result['T'] = T
        self._result['T_err'] = T_err
        self._result['T_s2n'] = T_s2n

        flux,flux_err,flux_s2n=self._get_flux_stats(self._result['pars'],
                                                    self._result['pcov'])
        Fs2n=flux/flux_err
        self._result['flux'] = flux
        self._result['flux_err'] = flux_err
        self._result['flux_s2n'] = flux_s2n


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


    def _get_T_stats(self, pars, pcov):
        """
        Simple model only
        """
        T     = pars[4]
        T_err = numpy.sqrt(pcov[4,4])
        T_s2n = T/T_err
        return T, T_err, T_s2n

    def _get_flux_stats(self, pars, pcov):
        """
        Simple model only
        """
        counts     = pars[5]
        counts_err = numpy.sqrt(pcov[5,5])
        counts_s2n = counts/counts_err
        return counts, counts_err, counts_s2n


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

    # psf
    mc_psf=MCMCGaussPSF(im_psf, wt_psf, j,
                        T=Tsky_psf, counts=counts_sky_psf)
    mc_psf.go()

    psf_fit=mc_psf.get_gmix()
    res_psf=mc_psf.get_result()
    print_pars(res_psf['pars'], front='pars_psf:', stream=stderr)
    print_pars(res_psf['perr'], front='perr_psf:', stream=stderr)
    print 'Tpix: %.4g +/- %.4g' % (res_psf['pars'][4]/jfac2, res_psf['perr'][4]/jfac2)

    imfit_psf=psf_fit.make_image(im_psf.shape, jacobian=j)
    images.compare_images(im_psf, imfit_psf, label1='psf',label2='fit')
    mcmc.plot_results(mc_psf.get_trials())

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


def test_model_priors(model, counts_sky=100.0, noise_sky=0.001, nimages=1, jfac=0.27):
    """
    testing jacobian stuff
    """
    import images
    import mcmc

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
    # priors
    #

    cen_prior=priors.CenPrior(0.0, 0.0, 0.1, 0.1)
    T_prior=priors.LogNormal(Tsky_obj, 0.1*Tsky_obj)
    counts_prior=priors.LogNormal(counts_sky_obj, 0.1*counts_sky_obj)
    g_prior = priors.GPriorBA(0.3)

    #
    # fitting
    #

    # psf
    mc_psf=MCMCGaussPSF(im_psf, wt_psf, j,
                        T=Tsky_psf, counts=counts_sky_psf)
    mc_psf.go()

    psf_fit=mc_psf.get_gmix()
    res_psf=mc_psf.get_result()
    print_pars(res_psf['pars'], front='pars_psf:', stream=stderr)
    print_pars(res_psf['perr'], front='perr_psf:', stream=stderr)
    print 'Tpix: %.4g +/- %.4g' % (res_psf['pars'][4]/jfac2, res_psf['perr'][4]/jfac2)

    imfit_psf=psf_fit.make_image(im_psf.shape, jacobian=j)
    images.compare_images(im_psf, imfit_psf, label1='psf',label2='fit')
    mcmc.plot_results(mc_psf.get_trials())

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
                      g_prior=g_prior)
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

