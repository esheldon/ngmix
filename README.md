ngmix
=====

Gaussian mixture models for 2d images, implemented in python

Notes
    - this code is under heavy development.  No stable API is yet provided.

examples
--------
```python
# some imports constants we need
import ngmix
from ngmix.observation import Observation, ObsList, MultiBandObsList
from ngmix.fitting import LMSimple
import numpy
from numpy import array
from numpy.random import uniform, normal

eps=0.01
numpy.random.seed(8381)

#
# make a simlated gaussian mixture, convolve it with a PSF and
# render a simulated image
#

# The code is designed to work on multiple images simulaneously, we always
# fit in "sky coordinates".  We do this by making a jacobian to represent
# the transformation, and make sure the center is at our best guess of the
# object location.  In this case we make a unit jacobian.  for non-unit,
# see the ngmix.Jacobian class. 

gal_jacob=ngmix.UnitJacobian(row=16.0, col=16.0)
psf_jacob=ngmix.UnitJacobian(row=12.0, col=12.0)

# parameters for the gaussian mixture
#
# pars for a simple gmix correspond to
#     [cen1, cen2, g1, g2, T, flux]
#
# note T=ixx+iyy, and the center corresponds to an offset relative to the
# jacobian center.  Use cen1,cen2=0,0 for no offset

# exponential disk approximated by gaussians
pars=[0.0, 0.0, 0.2, -0.1, 16.0, 100.0]
gmix0=ngmix.GMixModel(pars,"exp")

# PSF is a single gaussian
psf_pars=[0.0, 0.0, -0.03, 0.02, 4.0, 1.0]
psf_gmix=ngmix.GMixModel(psf_pars,"gauss")

# convolution with the PSF
gmix=gmix0.convolve(psf_gmix)

# render a simulated image, integrating the model over the pixels
# using 10-point Gauss-Legendre integration

dims=[32,32]
image0=gmix.make_image(dims, npoints=10, jacobian=gal_jacob)

psf_dims=[24,24]
psf_im=psf_gmix.make_image(psf_dims, npoints=10, jacobian=psf_jacob)

# add some noise to the galaxy image

sigma=0.01
noise = numpy.random.normal(scale=sigma,size=image0.shape)
image = image0 + noise

#
# fit the data using a maximum likelihood fitter
# we use the LM method, Levenberg-Marquardt
#


# make an observation for the psf image
psf_obs=Observation(psf_im, jacobian=psf_jacob)

# Simple means one of the 6 parameter models
pfitter=LMSimple(psf_obs,'gauss')

# for simplicity, guess pars before pixelization
guess=array(psf_pars)
guess[0] += uniform(low=-eps,high=eps)
guess[1] += uniform(low=-eps,high=eps)
guess[2] += uniform(low=-eps, high=eps)
guess[3] += uniform(low=-eps, high=eps)
guess[4] *= (1.0 + uniform(low=-eps, high=eps))
guess[5] *= (1.0 + uniform(low=-eps, high=eps))

pfitter.go(guess)


psf_gmix_fit=pfitter.get_gmix()

# set the gmix; needed for galaxy fitting later
psf_obs.set_gmix(psf_gmix_fit)

# fit the galaxy.
# Set the weight image for the fit; can be a complex weight
# map in principle

weight=numpy.zeros(image.shape) + 1/sigma**2

# When constructing the Observation we include a weight map and a psf
# observation

obs = Observation(image, weight=weight, jacobian=gal_jacob, psf=psf_obs)

fitter=LMSimple(obs,'exp')

guess=array(pars)
guess[0] += uniform(low=-eps, high=eps)
guess[1] += uniform(low=-eps, high=eps)
guess[2] += uniform(low=-eps, high=eps)
guess[3] += uniform(low=-eps, high=eps)
guess[4] *= (1.0 + uniform(low=-eps, high=eps))
guess[5] *= (1.0 + uniform(low=-eps, high=eps))

fitter.go(guess)

res=fitter.get_result()

ngmix.print_pars(pars, front="truth:")
ngmix.print_pars(res['pars'],front="meas: ")
ngmix.print_pars(res['pars_err'],front="err:  ")


# Fit multiple images of the same object. Send an ObsList to the fitter.
# create an ObsList by appending Observation objects

obs_list = ObsList()
obs_list.append( obs1 )
obs_list.append( obs2 )
obs_list.append( obs3 )

fitter=LMSimple(obs_list, model)

# you can also fit multiple bands/filters at once with fixed structural
# parameters but different fluxes in each band

mb_obs_list = MultiBandObsList()
mb_obs_list.append( obs_list_g ) 
mb_obs_list.append( obs_list_r ) 
mb_obs_list.append( obs_list_i ) 
mb_obs_list.append( obs_list_z ) 

fitter=LMSimple(mb_obs_list, model)
```
dependencies
------------

* numpy
* scipy: optional needed for generating random samples from shape PDFs.
* emcee: optional for doing MCMC fitting: http://dan.iel.fm/emcee/current/ Affine invariant MCMC sampler.
* statsmodels: optional for importance sampling (multivariate student
    T distribution)
