ngmix
=====

[![Build Status](https://travis-ci.com/esheldon/ngmix.svg?branch=master)](https://travis-ci.com/esheldon/ngmix)

Gaussian mixture models and other tools for working with 2d images, implemented
in python. The code is made fast using the numba package.

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

Notes on versions
-----------------

The api for fitting routines and "bootstrapping" code was rewritten for the
ngmix version 2 release.  This is a "breaking change", so if you have existing
code that uses the ngmix version 1 apis you most likely will need to update it.
You can also install version 1.3.8 to get the old api.

The wiki has been updated to reflect the new usage patterns.
