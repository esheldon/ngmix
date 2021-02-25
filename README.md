ngmix
=====

[![Build Status](https://travis-ci.com/esheldon/ngmix.svg?branch=master)](https://travis-ci.com/esheldon/ngmix)

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
* scipy: for image fitting using the Levenberg-Marquardt fitter
* galsim: for performing metacalibration operations.
* scikit-learn: for sampling multivariate PDFs

installation
------------
```bash
# using conda.  This also installs numba and numpy
conda install -c conda-forge ngmix

# from source. In this case you need to install numba yourself
python setup.py install
conda install numba
```
