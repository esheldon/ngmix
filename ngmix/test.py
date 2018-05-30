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

try:
    import covmatrix
    have_covmatrix=True
except:
    have_covmatrix=False

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
        numpy.random.seed(self.seed)

    def get_obs_data(self, model, noise):
        obsdata=make_test_observations(
            model,
            g1_obj=self.g1,
            g2_obj=self.g2,
            T_obj=self.T,
            counts_obj=self.counts,
            noise_obj=noise,
            psf_model=self.psf_model,
            g1_psf=self.g1psf,
            g2_psf=self.g2psf,
            T_psf=self.Tpsf,
            counts_psf=self.countspsf,
            noise_psf=self.noisepsf,
        )

        return obsdata

    def testLM(self):

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

    def testMax(self):

        if not have_covmatrix:
            print("skipping max since covmatrix is not installed")
            return

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

            max_pars={
                'method':'Nelder-Mead',
                'options':{
                    'maxiter':2000,
                    'maxfev':4000,
                },
            }

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


    def testCM(self):

        if not have_covmatrix:
            print("skipping cm since covmatrix is not installed")
            return


        print('\n')
        for model in ['exp','dev','cm']:
            print("model:",model)
            for noise in [0.001]:
            #for noise in [0.001, 0.1, 1.0]:
                print('='*10)
                print('    noise:',noise)
                mdict=self.get_obs_data(model,noise)

                obs=mdict['obs']
                obs.set_psf(mdict['psf_obs'])

                max_pars={'method':'lm',
                          'lm_pars':{'maxfev':4000}}

                #prior=joint_prior.make_uniform_simple_sep([0.0,0.0],     # cen
                #                                          [0.1,0.1],     # g
                #                                          [-10.0,3500.], # T
                #                                          [-0.97,1.0e9]) # flux
                prior=None

                boot=bootstrap.CompositeBootstrapper(obs,verbose=True)
                boot.fit_psfs('gauss', 4.0)
                boot.fit_max('cm', max_pars, prior=prior)
                res=boot.get_max_fitter().get_result()

                print_pars(mdict['pars'],   front='    pars true: ')
                print_pars(res['pars'],     front='    pars meas: ')
                print_pars(res['pars_err'], front='    pars err:  ')
                print("    fracdev:",res['fracdev'], "TdByTe:",res['TdByTe'],"TdByTe_noclip:",res['TdByTe_noclip'])
                print('    s2n:',res['s2n_w'])


def make_test_observations(model, **kw):

    if model=='cm':
        imdata_dev=make_test_images('dev', **kw)
        imdata_exp=make_test_images('exp', **kw)

        imdata={}
        imdata.update(imdata_dev)
        imdata['im'] = (imdata_dev['im'] + imdata_exp['im'])*0.5
        imdata['wt'] = 1.0/(1.0/imdata_dev['wt'] + 1.0/imdata_exp['wt'])
    else:
        imdata=make_test_images(model, **kw)

    psf_obs = Observation(
        imdata['psf'],
        jacobian=imdata['jacobian'],
    )

    obs=Observation(
        imdata['im'],
        weight=imdata['wt'],
        jacobian=imdata['jacobian'],
    )

    return {
        'psf_obs':psf_obs,
        'obs':obs,
        'pars':imdata['pars'],
    }

def make_test_images(model,
                     g1_obj=0.1,
                     g2_obj=0.05,
                     T_obj=16.0,
                     counts_obj=100.0,
                     noise_obj=0.001,
                     psf_model="gauss",
                     g1_psf=0.0,
                     g2_psf=0.0,
                     T_psf=4.0,
                     counts_psf=100.0,
                     noise_psf=0.001):


    sigma=sqrt( (T_obj + T_psf)/2. )
    dims=[2.*5.*sigma]*2
    cen=[dims[0]/2., dims[1]/2.]

    j=UnitJacobian(row=cen[0],col=cen[1])

    pars_psf = [0.0, 0.0, g1_psf, g2_psf, T_psf, counts_psf]
    gm_psf=gmix.GMixModel(pars_psf, psf_model)

    pars_obj = array([0.0, 0.0, g1_obj, g2_obj, T_obj, counts_obj])
    npars=pars_obj.size
    gm_obj0=gmix.GMixModel(pars_obj, model)

    gm=gm_obj0.convolve(gm_psf)

    im_psf=gm_psf.make_image(dims, jacobian=j)
    npsf=noise_psf*numpy.random.randn(im_psf.size).reshape(im_psf.shape)
    im_psf[:,:] += npsf
    wt_psf=zeros(im_psf.shape) + 1./noise_psf**2

    im_obj=gm.make_image(dims, jacobian=j)
    n=noise_obj*numpy.random.randn(im_obj.size).reshape(im_obj.shape)
    im_obj[:,:] += n
    wt_obj=zeros(im_obj.shape) + 1./noise_obj**2

    return {
        'pars':pars_obj,
        'psf':im_psf,
        'im':im_obj,
        'wt':wt_obj,
        'jacobian':j,
    }

