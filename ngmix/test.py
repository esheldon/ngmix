from __future__ import print_function, absolute_import, division
import sys, os
import unittest
import numpy
from numpy import array, zeros, diag, exp
from numpy import sqrt, where, log, log10, isfinite, newaxis
from numpy.random import uniform as randu
from pprint import pprint

from . import stats
from .priors import srandu

from . import joint_prior
from .fitting import *
from .gexceptions import *
from .jacobian import Jacobian, UnitJacobian
from . import bootstrap

from . import em

def test():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFitting)
    unittest.TextTestRunner(verbosity=2).run(suite)

class TestFitting(unittest.TestCase):

    def setUp(self):
        self.T=4.0
        self.counts=100.0
        self.g1=0.1
        self.g2=-0.05

        self.psf_model='gauss'
        self.g1psf = 0.03
        self.g2psf = -0.01
        self.Tpsf = 4.0
        self.countspsf=1.0
        self.noisepsf=0.001

        self.seed=100
        self.rng=numpy.random.RandomState(self.seed)

    def get_obs_data(self, model, noise):

        rng=self.rng

        sigma=sqrt( (self.T + self.Tpsf)/2. )
        dims=[2.*5.*sigma]*2
        cen=[dims[0]/2., dims[1]/2.]

        j=UnitJacobian(
            row=cen[0],
            col=cen[1],
        )

        pars_psf = [0.0, 0.0, self.g1psf, self.g2psf, self.Tpsf, self.countspsf]
        gm_psf=gmix.GMixModel(pars_psf, self.psf_model)

        pars_obj = array([0.0, 0.0, self.g1, self.g2, self.T, self.counts])
        npars=pars_obj.size
        gm_obj0=gmix.GMixModel(pars_obj, model)

        gm=gm_obj0.convolve(gm_psf)

        im_psf=gm_psf.make_image(dims, jacobian=j)
        npsf=rng.normal(
            scale=self.noisepsf,
            size=im_psf.shape,
        )
        im_psf[:,:] += npsf
        wt_psf=zeros(im_psf.shape) + 1./self.noisepsf**2

        im_obj=gm.make_image(dims, jacobian=j)
        n=rng.normal(
            scale=noise,
            size=im_obj.shape,
        )
        im_obj[:,:] += n
        wt_obj=zeros(im_obj.shape) + 1./noise**2

        psf_obs = Observation(
            im_psf,
            weight=wt_psf,
            jacobian=j,
        )

        obs=Observation(
            im_obj,
            weight=wt_obj,
            jacobian=j,
        )

        return {
            'psf_obs':psf_obs,
            'obs':obs,
            'pars':pars_obj,
        }

    def testExp(self):

        print('\n')
        for noise in [0.001, 0.1, 1.0]:
            print('='*10)
            print('noise:',noise)
            mdict=self.get_obs_data('exp',noise)

            obs=mdict['obs']
            obs.set_psf(mdict['psf_obs'])

            pars=mdict['pars'].copy()
            pars[0] += randu(low=-0.1,high=0.1)
            pars[1] += randu(low=-0.1,high=0.1)
            pars[2] += randu(low=-0.1,high=0.1)
            pars[3] += randu(low=-0.1,high=0.1)
            pars[4] *= (1.0 + randu(low=-0.1,high=0.1))
            pars[5] *= (1.0 + randu(low=-0.1,high=0.1))

            max_pars={'method':'lm',
                      'lm_pars':{'maxfev':4000}}

            prior=joint_prior.make_uniform_simple_sep([0.0,0.0],     # cen
                                                      [0.1,0.1],     # g
                                                      [-10.0,3500.], # T
                                                      [-0.97,1.0e9]) # flux

            boot=bootstrap.Bootstrapper(obs)
            boot.fit_psfs('gauss', 4.0)
            boot.fit_max('exp', max_pars, pars, prior=prior)
            res=boot.get_max_fitter().get_result()

            print_pars(mdict['pars'],   front='pars true: ')
            print_pars(res['pars'],     front='pars meas: ')
            print_pars(res['pars_err'], front='pars err:  ')
            print('s2n:',res['s2n_w'])



