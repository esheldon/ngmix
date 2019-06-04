ngmix
=====

Gaussian mixture models and other code for working with for 2d images,
implemented in python.   The code is made fast using the numba package.  Note
the old c-extension based code is still available in the tag v0.9.5

For some examples, please see [the wiki](https://github.com/esheldon/ngmix/wiki).

dependencies
------------

* numpy
* numba >= 0.43

optional dependencies
---------------------
* scipy: optional needed for image fitting using the Levenberg-Marquardt fitter
* galsim: optional for doing metacalibration operations.
* skikit-learn:  for sampling multivariate PDFs
* statsmodels: optional for importance sampling (multivariate student
    T distribution)
* emcee: optional for doing MCMC fitting: http://dan.iel.fm/emcee/current/ Affine invariant MCMC sampler.

installation
------------
```bash
python setup.py install
```

installing numba
----------------
```bash
# by far the easiest way is using anaconda.
conda install numba
```

TODO
----
Make numba optional.  There are many useful data structures
in ngmix that do not require numba, it would be nice to make them
available without the numba dependency.  To implement this would
be straightforward:  wrap some of the imports in try/except blocks
