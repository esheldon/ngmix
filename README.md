ngmix
=====

Gaussian mixture models for 2d images, implemented in python

Notes
    - this code is under heavy development.  No stable API is yet provided.

examples
--------
```python
import ngmix
from ngmix.observation import Observation, ObsList, MultiBandObsList
import numpy
from numpy.random import uniform as urand

#
# make a gaussian mixture, convolve it with a PSF and
# render an image
#

# The code is designed to work on multiple images simulaneously, we always
# fit in "sky coordinates".  We do this by making a jacobian to represent
# the transformation, and make sure the center is at our best guess of the
# object location.  In this case we make a unit jacobian.  for non-unit,
# see the ngmix.Jacobian class. 

gal_jacob=ngmix.UnitJacobian(16.0, 16.0)
psf_jacob=ngmix.UnitJacobian(12.0, 12.0)

# pars correspond to
#     [cen1, cen2, g1, g2, T, flux]
#
# note T=ixx+iyy, and the center corresponds to an offset relative to the
# jacobian center.  Use cen1,cen2=0,0 for no offset

pars=[0.0, 0.0, 0.2, -0.1, 16.0, 100.0]

# PSF is a single gaussian
psf_pars=[0.0, 0.0, -0.03, 0.02, 4.0, 1.0]

gmix0=ngmix.GMixModel(pars,"exp")
psf_gmix=ngmix.GMixModel(psf_pars,"gauss")

# convolution with the PSF
gmix=gmix0.convolve(psf_gmix)

# render the image, integrating the model over the pixels
# with a sub-pixel grid 16x16

dims=[32,32]
image=gmix.make_image(dims, nsub=16, jacobian=gal_jacob)

# add some noise

sigma=0.01
noise = sigma*numpy.random.randn(image.size).reshape(image.shape)
noisy_image = image + noise


#
# fit the data
#

# fit the PSF using an EM algorithm
psf_dims=[24,24]
psf_im=psf_gmix.make_image(psf_dims, nsub=16, jacobian=psf_jacob)

# For EM, image must have non-zero sky and no negative pixel values
imsky,sky=ngmix.em.prep_image(psf_im)

psf_obs=Observation(imsky, jacobian=psf_jacob)

em=ngmix.em.GMixEM(psf_obs)
# for simplicity, guess pars before pixelization
guess=psf_gmix.copy()
em.go(guess, sky, tol=1.e-5)

psf_gmix_fit=em.get_gmix()

# set the gmix; needed for galaxy fitting later
psf_obs.set_gmix(psf_gmix_fit)

# fit the galaxy.
# the weight image for the fit; can be complex in principle
weight=numpy.zeros(image.shape) + 1/sigma**2


# When constructing the Observation we include a weight map and a psf
# observation

obs = Observation(image, weight=weight, jacobian=gal_jacob, psf=psf_obs)

# Use MCMCSimple to fit using a "simple" model, either "exp" or "dev"
# currently. You can also send a prior= keyword that takes parameters
# and returns the log(probability).  Send use_logpars to work in
# log(T) and log(flux) space

model="exp"
nwalkers=80
burnin=400
nstep=400
fitter=ngmix.fitting.MCMCSimple(obs, model, nwalkers=nwalkers)

# guess should be an array [nwalkers, npars].  It is good to
# guess random points around your best estimate of the galaxy
# parameters

eps=0.01
guess=numpy.zeros( (nwalkers, len(pars)) )
guess[:,0] = urand(size=nwalkers, low=-eps, high=eps)
guess[:,1] = urand(size=nwalkers, low=-eps, high=eps)
guess[:,2] = pars[2] + urand(size=nwalkers, low=-eps, high=eps).clip(min=-0.5, max=0.5)
guess[:,3] = pars[3] + urand(size=nwalkers, low=-eps, high=eps).clip(min=-0.5, max=0.5)
guess[:,4] = pars[4]*(1.0 + urand(size=nwalkers, low=-eps, high=eps))
guess[:,5] = pars[5]*(1.0 + urand(size=nwalkers, low=-eps, high=eps))

pos=fitter.run_mcmc(guess, burnin)
pos=fitter.run_mcmc(pos, nstep)

fitter.calc_result()

res=fitter.get_result()

ngmix.print_pars(pars, front="truth:")
ngmix.print_pars(res['pars'],front="meas: ")
ngmix.print_pars(res['pars_err'],front="err:  ")
#print res['pars_cov'] # full covariance matrix

# note the trials can be gotten with get_trials()


# Fit multiple images of the same object. Send an ObsList to the fitter.
# create an ObsList by appending Observation objects

obs_list = ObsList()
obs_list.append( obs1 )
obs_list.append( obs2 )
obs_list.append( obs3 )

fitter=ngmix.fitting.MCMCSimple(obs_list, model, nwalkers=nwalkers)

# you can also fit multiple bands/filters at once with fixed structural
# parameters but different fluxes in each band

mb_obs_list = MultiBandObsList()
mb_obs_list.append( obs_list_g ) 
mb_obs_list.append( obs_list_r ) 
mb_obs_list.append( obs_list_i ) 
mb_obs_list.append( obs_list_z ) 

fitter=ngmix.fitting.MCMCSimple(mb_obs_list, model, nwalkers=nwalkers)
```
dependencies
------------

* numpy
* scipy: optional needed for generating random samples from shape PDFs.
* emcee: optional for doing MCMC fitting: http://dan.iel.fm/emcee/current/ Affine invariant MCMC sampler.
* statsmodels: optional for importance sampling (multivariate student
    T distribution)
