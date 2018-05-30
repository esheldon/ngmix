ngmix
=====

Gaussian mixture models and other code for working with for 2d images,
implemented in python.   The code is made fast by using the numba package.
Note the old c-extension based code is still available in the tag v0.9.5

For some examples, please see [the wiki](https://github.com/esheldon/ngmix/wiki).

dependencies
------------

* numpy
* numba

optional dependencies
---------------------
* scipy: optional needed for image fitting using the Levenberg-Marquardt fitter
* galsim: optional for doing metacalibration operations.
* skikit-learn:  for sampling multivariate PDFs
* statsmodels: optional for importance sampling (multivariate student
    T distribution)
* emcee: optional for doing MCMC fitting: http://dan.iel.fm/emcee/current/ Affine invariant MCMC sampler.
