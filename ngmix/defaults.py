import numpy as np


# default values
PDEF = -9.999e9  # parameter defaults
CDEF = 9.999e9  # covariance or error defaults

# for priors etc.
LOWVAL = -np.inf
BIGVAL = 9999.0e47

DEFAULT_LM_PARS = {"maxfev": 4000, "ftol": 1.0e-5, "xtol": 1.0e-5}

ADMOM_RESULT_DTYPE = [
    ('flags', 'i4'),
    ('numiter', 'i4'),
    ('nimage', 'i4'),
    ('npix', 'i4'),
    ('wsum', 'f8'),

    ('sums', 'f8', 6),
    # ('sums_cov','f8', 36),
    ('sums_cov', 'f8', (6, 6)),
    ('pars', 'f8', 6),
    # temporary
    ('F', 'f8', 6),
]
