import numpy as np


# default values
PDEF = -9.999e9  # parameter defaults
CDEF = 9.999e9  # covariance or error defaults

# for priors etc.
LOWVAL = -np.inf
BIGVAL = 9999.0e47

DEFAULT_LM_PARS = {"maxfev": 4000, "ftol": 1.0e-5, "xtol": 1.0e-5}
