import numpy
from . import gmix
from . import jacobian
from .jacobian import Jacobian, UnitJacobian
from .point2d import Point2D

from .priors import srandu

from .gexceptions import GMixRangeError

import time

LOWVAL=-9999.0e47

class FitterBase(object):
    """
    Base for other fitters
    """
    def __init__(self, image, weight, model, **keys):
        self.keys=keys

        self.model=gmix.get_model_num(model)
        self.npars=gmix.get_model_npars(self.model)

        self.im_list=_get_as_list(image)
        self.wt_list=_get_as_list(weight)
        self.psf_list=_get_as_list(keys.get('psf',None))
        self._set_jacob_list()
        self.nimages=len(self.im_list)

        self.g_prior = keys.get('g_prior',None)
        self.cen_prior = keys.get('cen_prior',None)
        self.T_prior = keys.get('T_prior',None)

        self.totpix=self.verify()

        self.make_plots=keys.get('make_plots',False)

        self._gmix_list=None

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
        return self._result

    def _set_jacob_list(self):
        from copy import copy
        jlist = _get_as_list(self.keys.get('jacob',None))
        if jlist is None:
            cenlist=_get_as_list(self.keys.get('cen',None))
            if cenlist is None:
                raise ValueError("send either cen= or jacob=")

            jlist=[UnitJacobian(cen.row, cen.col) for cen in cenlist]

        self.jacob_list=jlist        

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



class MCMCBase(FitterBase):
    def __init__(self, image, weight, model, **keys):
        super(MCMCBase,self).__init__(image, weight, model, **keys)

        self.nwalkers=keys.get('nwalkers',20)
        self.nstep=keys.get('nstep',200)
        self.burnin=keys.get('burnin',400)
        self.mca_a=keys.get('mca_a',2.0)

        self.draw_g_prior=keys.get('draw_g_prior',True)


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

        return sampler

    def _get_sampler(self):
        import emcee
        sampler = emcee.EnsembleSampler(self.nwalkers, 
                                        self.npars, 
                                        self._calc_lnprob,
                                        a=self.mca_a)
        return sampler

    def _get_from_lists(self, i, gmix_list):
        return gmix_list[i], self.im_list[i], self.wt_list[i], self.jacob_list[i]


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

        pars,pcov,g,gcov,gsens=self._get_trial_stats_with_lensfit()
 
        arates = self.sampler.acceptance_fraction
        arate = arates.mean()

        self._result={'flags':0,
                      'model':self.model,
                      'g':g,
                      'gcov':gcov,
                      'gsens':gsens,
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
        Tmean,Terr=self._get_T_stats(pars,pcov)
        Ts2n=Tmean/Terr

        flux,flux_err=self._get_flux_stats(pars,pcov)
        Fs2n=flux/flux_err

        self._result={'flags':0,
                      'model':self.model,
                      'g':g,
                      'gcov':gcov,
                      'gsens':gsens,
                      'pars':pars,
                      'perr':sqrt(diag(pcov)),
                      'pcov':pcov,
                      'Tmean':Tmean,
                      'Terr':Terr,
                      'Ts2n':Ts2n,
                      'flux':flux,
                      'flux_err':flux_err,
                      'flux_s2n':Fs2n,
                      'arate':arate}

        if self.do_pqr:
            P,Q,R = self._get_PQR()
            self._result['P']=P
            self._result['Q']=Q
            self._result['R']=R

        self._result.update(stats)
        '''


    def _get_trial_stats_with_lensfit(self):

        if self.g_prior is not None:
            prior = self.g_prior.get_prob_array2d(g1vals,g2vals)
            pars,pcov = extract_mcmc_stats(self.trials,weights=prior)
            g = pars[2:4].copy()
            gcov = pcov[2:4, 2:4].copy()

            g1vals=self.trials[:,2]
            g2vals=self.trials[:,3]

            dpri_by_g1 = self.g_prior.dbyg1_array(g1vals,g2vals)
            dpri_by_g2 = self.g_prior.dbyg2_array(g1vals,g2vals)

            psum = prior.sum()

            g1diff = g[0]-g1vals
            g2diff = g[1]-g2vals

            gsens = numpy.zeros(2)
            gsens[0]= 1.-(g1diff*dpri_by_g1).sum()/psum
            gsens[1]= 1.-(g2diff*dpri_by_g2).sum()/psum
        else:
            pars,pcov = extract_mcmc_stats(self.trials)
            g = pars[2:4].copy()
            gcov = pcov[2:4, 2:4].copy()

            gsens=numpy.array([1.,1.])

        return pars, pcov, g, gcov, gsens


    def _get_guess(self):
        raise RuntimeError("over-ride me")

class MCMCGaussPSF(MCMCBase):
    def __init__(self, image, weight, **keys):
        """
        We demand a good centroid guess
        """
        model=gmix.GMIX_GAUSS
        super(MCMCGaussPSF,self).__init__(image, weight, model, **keys)

    def _get_median_counts(self):
        """
        median of the counts across all input images
        """
        clist=numpy.zeros(self.nimages)
        for i,im in enumerate(self.im_list):
            clist[i] = im.sum()
        
        return numpy.median(clist)

    def _get_guess(self):
        """
        The counts guess is stupid unless you have a well-trimmed
        PSF image
        """
        
        counts_guess=self._get_median_counts()
        T_guess=4.0

        guess=numpy.zeros( (self.nwalkers,self.npars) )

        # center
        guess[:,0]=0.1*srandu(self.nwalkers)
        guess[:,1]=0.1*srandu(self.nwalkers)

        guess[:,2]=0.1*srandu(self.nwalkers)
        guess[:,3]=0.1*srandu(self.nwalkers)

        guess[:,4] = T_guess*(1 + 0.1*srandu(self.nwalkers))
        guess[:,5] = counts_guess*(1 + 0.1*srandu(self.nwalkers))

        self._guess=guess
        return guess

    def _get_T_stats(self, pars, pcov):
        """
        Simple model only
        """
        return pars[4], sqrt(pcov[4,4])
    def _get_flux_stats(self, pars, pcov):
        """
        Simple model only
        """
        return pars[5], sqrt(pcov[5,5])



def _get_as_list(arg):
    if arg is None:
        return None

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


def test_gauss_psf_graph(counts=100.0, noise=0.1, nimages=10, n=10, groups=True):
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
    cen=Point2D(dims[0]/2., dims[1]/2.)

    g1=0.1
    g2=0.05
    T=8.0

    pars = [cen.row, cen.col, g1, g2, T, counts]
    gm=gmix.GMixModel(pars, "gauss")

    im=gm.make_image(dims)

    im[:,:] += noise*numpy.random.randn(im.size).reshape(im.shape)
    wt=numpy.zeros(im.shape) + 1./noise**2

    imlist=[im]*nimages
    wtlist=[wt]*nimages
    cenlist=[cen]*nimages

    # one run to warm up the jit compiler
    mc=MCMCGaussPSF(im, wt, cen=cen)
    mc.go()

    with PyCallGraph(config=config, output=graphviz):
        for i in xrange(n):
            #mc=MCMCGaussPSF(im, wt, cen=cen)
            mc=MCMCGaussPSF(imlist, wtlist, cen=cenlist)
            mc.go()

            res=mc.get_result()

            print res['g']

def test_gauss_psf(counts=100.0, noise=0.001, n=10, nimages=10):
    #import pylab
    import time


    dims=[25,25]
    cen=Point2D(dims[0]/2., dims[1]/2.)

    g1=0.1
    g2=0.05
    T=8.0

    pars = [cen.row, cen.col, g1, g2, T, counts]
    gm=gmix.GMixModel(pars, "gauss")

    im=gm.make_image(dims)

    im[:,:] += noise*numpy.random.randn(im.size).reshape(im.shape)
    wt=numpy.zeros(im.shape) + 1./noise**2

    imlist=[im]*nimages
    wtlist=[wt]*nimages
    cenlist=[cen]*nimages


    #pylab.imshow(im, cmap=pylab.gray(), interpolation='nearest')
    #pylab.show()

    # one run to warm up the jit compiler
    mc=MCMCGaussPSF(im, wt, cen=cen)
    mc.go()

    t0=time.time()
    for i in xrange(n):
        mc=MCMCGaussPSF(imlist, wtlist, cen=cenlist)
        mc.go()

        res=mc.get_result()

        print res['g']

    sec=time.time()-t0
    secper=sec/n
    print secper,'seconds per'

    return sec,secper
