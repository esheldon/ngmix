import numpy as np

# Shim to support numpy >= 2 and < 2.0.0.
if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
    copy_if_needed = None
else:
    copy_if_needed = False

# default values
PDEF = -9.999e9  # parameter defaults
CDEF = 9.999e9  # covariance or error defaults

# for priors etc.
LOWVAL = -np.inf
BIGVAL = 9999.0e47

DEFAULT_LM_PARS = {"maxfev": 4000, "ftol": 1.0e-5, "xtol": 1.0e-5}
