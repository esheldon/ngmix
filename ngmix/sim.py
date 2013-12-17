"""
Simulate images and fit them, currently MCMC only

additional dependence on
    emcee for MCMC fitting and
    fitsio if checkpointing

example sim config file in yaml format
--------------------------------------
name: "nsim-dg01"

psf_model: "gauss"
psf_T: 4.0
psf_shape: [0.0, 0.0]

obj_model: "dev"
obj_T_mean: 16.0
obj_T_sigma_frac: 0.3

obj_counts_mean: 100.0
obj_counts_sigma_frac: 0.3

shear: [0.01,0.0]

nsub: 16

example run config file in yaml format
--------------------------------------
run: "ngmix-dg01r33"
sim: "nsim-dg01"

fit_model: "dev"

nwalkers: 40
burnin:   400
nstep:    200
mca_a:    3.0

# we normalize splits by split for is2n==0
desired_err: 2.0e-05
nsplit0: 60000

s2n_vals: [ 15, 21, 30, 42, 60, 86, 122, 174, 247, 352, 500] 

"""

import os
from sys import stderr
import time

import numpy
from numpy.random import random as randu
from numpy.random import randn

import ngmix
from .gexceptions import GMixRangeError, GMixMaxIterEM

# region over which to render images and calculate likelihoods
NSIGMA_RENDER=5.0

# minutes
DEFAULT_CHECKPOINTS=[5,30,60,100]

class TryAgainError(Exception):
    def __init__(self, message):

        # Call the base class constructor with the parameters it needs
        Exception.__init__(self, message)

class NGMixSim(dict):
    def __init__(self, sim_conf, run_conf, s2n, npairs, **keys):
        """
        Simulate and fit the requested number of pairs at
        the specified s/n
        """

        self.set_config(sim_conf, run_conf)

        self.s2n=s2n
        self.npairs=npairs

        self.update(self.conf)
        self.update(keys)

        self.shear=self.simc['shear']
        self.nsub=self.simc['nsub']

        self.obj_model=self.simc['obj_model']

        self.setup_checkpoints(**keys)

        if self.data is None:
            self.make_struct()

        self.set_priors()
        self.make_psf()
        self.set_noise()

    def set_config(self, sim_conf, run_conf):
        """
        Check and set the configurations
        """
        if sim_conf['name'] != run_conf['sim']:
            err="sim name in run config '%s' doesn't match sim name '%s'"
            raise ValueError(err % (run_conf['sim'],sim_conf['name']))

        self.simc=sim_conf
        self.conf=run_conf

    def get_data(self):
        """
        Get a ref to the data array with the fit results
        """
        return self.data

    def run_sim(self):
        """
        Run the simulation, fitting psf and all pairs
        """
        self.fit_psf()

        self.start_timer()

        i=0
        npairs=self.npairs
        for ipair in xrange(npairs):
            print >>stderr,'%s/%s' % (ipair+1,npairs)

            if self.data['processed'][i]:
                i += 2 # skip the pair
            else:
                while True:
                    try:
                        reslist=self.process_pair()
                        break
                    except TryAgainError as err:
                        print >>stderr,str(err)

                self.copy_to_output(reslist[0], i)
                i += 1
                self.copy_to_output(reslist[1], i)
                i += 1

            self.set_elapsed_time()
            self.try_checkpoint()

        self.set_elapsed_time()
        print >>stderr,'time minutes:',self.tm_minutes
        print >>stderr,'time per image sec:',self.tm/(2*npairs)

    def start_timer(self):
        """
        Set the elapsed time so far
        """
        self.tm0 = time.time()

    def set_elapsed_time(self):
        """
        Set the elapsed time so far
        """

        self.tm = time.time()-self.tm0
        self.tm_minutes = self.tm/60.0

    def process_pair(self):
        """
        Create a simulated image pair and perform the fit
        """

        imdicts = self.get_noisy_image_pair()
        reslist=[]
        for key in imdicts:
            res=self.fit_galaxy(imdicts[key])
            if res['flags'] != 0:
                raise TryAgainError("failed at %s" % key)

            reslist.append(res)
            self.print_res(res)

        return reslist

    def fit_galaxy(self, imdict):
        """
        Fit the model to the galaxy
        """

        full_guess=self.get_guess(imdict)
        fitter=ngmix.fitting.MCMCSimple(imdict['image'],
                                        imdict['wt'],
                                        imdict['jacobian'],
                                        self['fit_model'],

                                        cen_prior=self.cen_prior,
                                        g_prior=self.g_prior,
                                        T_prior=self.T_prior,
                                        counts_prior=self.counts_prior,

                                        full_guess=full_guess,

                                        psf=self.psf_gmix_fit,
                                        nwalkers=self['nwalkers'],
                                        nstep=self['nstep'],
                                        burnin=self['burnin'],
                                        mca_a=self['mca_a'],
                                        do_pqr=True,
                                        do_lensfit=True)
        fitter.go()
        #fitter.make_plots(show=True)
        return fitter.get_result()

    def get_guess(self, imdict):
        """
        Get a guess centered on the truth
        """
        pars=imdict['pars']
        nwalkers=self['nwalkers']
        guess=numpy.zeros( (nwalkers, pars.size) )

        guess[:,0] = 0.01*srandu(nwalkers)
        guess[:,1] = 0.01*srandu(nwalkers)
        guess_shape=self.get_shape_guess(pars[2],pars[3],nwalkers)
        guess[:,2]=guess_shape[:,0]
        guess[:,3]=guess_shape[:,1]
        guess[:,4] = self.get_positive_guess(pars[4],nwalkers)
        guess[:,5] = self.get_positive_guess(pars[5],nwalkers)

        return guess

    def get_shape_guess(self, g1, g2, n):
        """
        Get guess, making sure in range
        """
        guess=numpy.zeros( (n, 2) )
        shape=ngmix.Shape(g1, g2)

        for i in xrange(n):

            while True:
                try:
                    g1_offset,g2_offset=0.01*srandu(2)
                    shape_new=shape.copy()
                    shape_new.shear(g1_offset, g2_offset)
                    break
                except GMixRangeError:
                    pass

            guess[i,0] = shape_new.g1
            guess[i,1] = shape_new.g2

        return guess

    def get_positive_guess(self, val, n):
        """
        Get guess, making sure positive
        """
        if val <= 0.0:
            raise GMixRangeError("val <= 0: %s" % val)

        vals=numpy.zeros(n)-9999.0
        while True:
            w,=numpy.where(vals <= 0)
            if w.size == 0:
                break
            else:
                vals[w] = val*(1.0 + 0.01*srandu(w.size))

        return vals

    def print_res(self,res):
        """
        print some stats
        """
        print >>stderr,'    arate:',res['arate']
        ngmix.fitting.print_pars(res['pars'],front='    pars: ',stream=stderr)
        ngmix.fitting.print_pars(res['perr'],front='    perr: ',stream=stderr)

    def fit_psf(self):
        """
        Fit the pixelized psf to a model
        """

        print >>stderr,'fitting psf'
        imsky,sky=ngmix.em.prep_image(self.psf_image)

        em=ngmix.em.GMixEM(imsky)
        guess=self.psf_gmix_true.copy()
        print 'psf guess:'
        print guess
        em.go(guess, sky, tol=1.e-5)

        self.psf_gmix_fit=em.get_gmix()
        print 'psf fit:'
        print self.psf_gmix_fit

    def set_priors(self):
        """
        Set all the priors
        """

        print >>stderr,"setting priors"
        T=self.simc['obj_T_mean']
        T_sigma = self.simc['obj_T_sigma_frac']*T
        counts=self.simc['obj_counts_mean']
        counts_sigma = self.simc['obj_counts_sigma_frac']*counts

        self.g_prior=ngmix.priors.GPriorBA(0.3)
        self.cen_prior=ngmix.priors.CenPrior(0.0, 0.0, 0.1, 0.1)
        self.T_prior=ngmix.priors.LogNormal(T, T_sigma)
        self.counts_prior=ngmix.priors.LogNormal(counts, counts_sigma)

    def make_psf(self):
        """
        make the psf gaussian mixture model
        """

        print >>stderr,"making psf"

        self.psf_dims, self.psf_cen=self.get_dims_cen(self.simc['psf_T'])

        pars=[self.psf_cen[0],
              self.psf_cen[1],
              self.simc['psf_shape'][0],
              self.simc['psf_shape'][1],
              self.simc['psf_T'],
              1.0]
        self.psf_gmix_true=ngmix.gmix.GMixModel(pars, self.simc['psf_model'])
        
        self.psf_image=self.psf_gmix_true.make_image(self.psf_dims,
                                                     nsub=self.nsub)
    
    def set_noise(self):
        """
        Find gaussian noise that when added to the image 
        produces the requested s/n.  Use a matched filter.

         sum(pix^2)
        ------------ = S/N^2
          skysig^2

        thus
            
        sum(pix^2)
        ---------- = skysig^2
          (S/N)^2
        """
        

        print >>stderr,"setting noise"

        imdict=self.get_image_pair(random=False)
        im=imdict['im1']['image']
        skysig2 = (im**2).sum()/self.s2n**2
        skysig = numpy.sqrt(skysig2)

        noise_image = skysig*randn(im.size).reshape(im.shape)
        new_im = im + noise_image

        s2n_check = numpy.sqrt( (im**2).sum()/skysig**2 )
        print >>stderr,"S/N goal:",self.s2n,"found:",s2n_check

        self.skysig=skysig
        self.ivar=1.0/skysig**2


    def get_noisy_image_pair(self, random=True):
        """
        Get an image pair, with noise added
        """
        imdict=self.get_image_pair(random=random)
        self.add_noise(imdict['im1']['image'])
        self.add_noise(imdict['im2']['image'])

        wt=numpy.zeros(imdict['im1']['image'].shape) + self.ivar
        imdict['im1']['wt']=wt
        imdict['im2']['wt']=wt
        return imdict

    def add_noise(self, im):
        """
        Add gaussian random noise
        """

        im[:,:] += self.skysig*randn(im.size).reshape(im.shape)

    def get_image_pair(self, random=True):
        """
        get a model image

        If random is True, use draw random values from the priors.
        Otherwise use the mean of the priors
        """

        cen_offset, shape1, shape2, T, counts=self.get_pair_pars(random=random)

        # center is just placeholder for now
        pars1=[0.0, 0.0, shape1.g1, shape1.g2, T, counts]
        pars2=[0.0, 0.0, shape2.g1, shape2.g2, T, counts]

        gm1_pre=ngmix.gmix.GMixModel(pars1, self.obj_model)
        gm2_pre=ngmix.gmix.GMixModel(pars2, self.obj_model)

        gm1  = gm1_pre.convolve(self.psf_gmix_true)
        gm2  = gm2_pre.convolve(self.psf_gmix_true)

        T = gm1.get_T()
        dims, cen = self.get_dims_cen(T)

        # jacobian is at center before offset
        j=ngmix.jacobian.UnitJacobian(cen[0], cen[1])

        cen[0] += cen_offset[0]
        cen[1] += cen_offset[1]

        gm1.set_cen(cen[0], cen[1])
        gm2.set_cen(cen[0], cen[1])

        nsub = self.nsub
        im1=gm1.make_image(dims, nsub=nsub)
        im2=gm2.make_image(dims, nsub=nsub)

        pars_true1=numpy.array(pars1)
        pars_true2=numpy.array(pars2)
        pars_true1[0] += cen_offset[0]
        pars_true1[1] += cen_offset[1]
        pars_true2[0] += cen_offset[0]
        pars_true2[1] += cen_offset[1]

        out={'im1':{'pars':pars_true1,'gm_pre':gm1_pre,'gm':gm1,'image':im1,'jacobian':j},
             'im2':{'pars':pars_true2,'gm_pre':gm2_pre,'gm':gm2,'image':im2,'jacobian':j}}
        return out

    def get_pair_pars(self, random=True):
        """
        Get pair parameters
        """

        if random:
            cen_offset=self.cen_prior.sample()
            g = self.g_prior.sample1d(1)
            g=g[0]
            rangle1 = randu()*2*numpy.pi
            rangle2 = rangle1 + numpy.pi/2.0
            g1_1 = g*numpy.cos(2*rangle1)
            g2_1 = g*numpy.sin(2*rangle1)
            g1_2 = g*numpy.cos(2*rangle2)
            g2_2 = g*numpy.sin(2*rangle2)

            T=self.T_prior.sample()
            counts=self.counts_prior.sample()
        else:
            cen_offset=[0.0, 0.0]
            g1_1=0.0
            g2_1=0.0
            g1_2=0.0
            g2_2=0.0
            T=self.T_prior.mean
            counts=self.counts_prior.mean

        shape1=ngmix.shape.Shape(g1_1, g2_1)
        shape2=ngmix.shape.Shape(g1_2, g2_2)

        shear=self.shear
        shape1.shear(shear[0], shear[1])
        shape2.shear(shear[0], shear[1])

        return cen_offset, shape1, shape2, T, counts

    def get_dims_cen(self, T):
        """
        Based on T, get the required dimensions and a center
        """
        sigma=numpy.sqrt(T/2.)
        dims = [2.*sigma*NSIGMA_RENDER]*2
        cen = [(dims[0]-1.)/2.]*2

        return dims, cen

    def setup_checkpoints(self, **keys):
        """
        Set up checkpoint times, file, and sent data
        """

        self.checkpoints     = keys.get('checkpoints',DEFAULT_CHECKPOINTS)
        self.n_checkpoint    = len(self.checkpoints)
        self.checkpointed    = [False]*self.n_checkpoint

        self.checkpoint_file=keys.get('checkpoint_file',None)
        self.set_checkpoint_data(**keys)

        if self.checkpoint_file is not None:
            self.do_checkpoint=True
        else:
            self.do_checkpoint=False


    def set_checkpoint_data(self, **keys):
        """
        Look for checkpoint data, file etc.
        """
        self.data=None

        checkpoint_data=keys.get('checkpoint_data',None)
        if checkpoint_data is not None:
            self.data=checkpoint_data


    def try_checkpoint(self):
        """
        If we should make a checkpoint, do so
        """

        should_checkpoint, icheck = self.should_checkpoint()

        if should_checkpoint:
            self.write_checkpoint()
            self.checkpointed[icheck]=True


    def should_checkpoint(self):
        """
        Should we write a checkpoint file?
        """

        should_checkpoint=False
        icheck=-1

        if self.do_checkpoint:
            for i in xrange(self.n_checkpoint):

                checkpoint=self.checkpoints[i]
                checkpointed=self.checkpointed[i]

                if self.tm_minutes > checkpoint and not checkpointed:
                    should_checkpoint=True
                    icheck=i

        return should_checkpoint, icheck


    def write_checkpoint(self):
        """
        Write the checkpoint file
        """
        import fitsio

        print >>stderr,'checkpointing at',self.tm_minutes,'minutes'
        print >>stderr,self.checkpoint_file

        with fitsio.FITS(self.checkpoint_file,'rw',clobber=True) as fobj:
            fobj.write(self.data)


    def copy_to_output(self, res, i):
        """
        Copy results into the output
        """
        d=self.data
        d['processed'][i] = 1
        d['pars'][i,:] = res['pars']
        d['pcov'][i,:,:] = res['pars_cov']
        d['P'][i] = res['P']
        d['Q'][i,:] = res['Q']
        d['R'][i,:,:] = res['R']
        d['g'][i,:] = res['g']
        d['gsens'][i,:] = res['g_sens']

    def make_struct(self):
        """
        Make the output array
        """
        dt=[('processed','i2'),
            ('pars','f8',6),
            ('pcov','f8',(6,6)),
            ('P','f8'),
            ('Q','f8',2),
            ('R','f8',(2,2)),
            ('g','f8',2),
            ('gsens','f8',2)]
        self.data=numpy.zeros(self.npairs*2, dtype=dt)

def srandu(num=None):
    """
    Generate random numbers in the symmetric distribution [-1,1]
    """
    return 2*(numpy.random.random(num)-0.5)


