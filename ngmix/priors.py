"""
Convention is that all priors should have peak ln(prob)==0. This
helps use in priors for LM fitting
"""
import math

import numpy
from numpy import where, array, exp, log, sqrt, cos, sin, zeros, diag
from numpy import pi

from .gexceptions import GMixRangeError

LOWVAL = -numpy.inf
BIGVAL = 9999.0e47


def lognorm_convert_old(mean, sigma):
    logmean = log(mean) - 0.5 * log(1 + sigma ** 2 / mean ** 2)
    logvar = log(1 + sigma ** 2 / mean ** 2)
    logsigma = sqrt(logvar)

    return logmean, logsigma


def lognorm_convert(mean, sigma, base=math.e):
    from math import log

    lbase = log(base)

    logmean = log(mean, base) - 0.5 * lbase * log(
        1 + sigma ** 2 / mean ** 2, base
    )
    logvar = log(1 + sigma ** 2 / mean ** 2, base)
    logsigma = sqrt(logvar)

    return logmean, logsigma


def scipy_to_lognorm(shape, scale):
    """
    Wrong?
    """
    srat2 = numpy.exp(shape ** 2) - 1.0
    # srat2 = numpy.exp( shape ) - 1.0

    meanx = scale * numpy.exp(0.5 * numpy.log(1.0 + srat2))
    sigmax = numpy.sqrt(srat2 * meanx ** 2)

    return meanx, sigmax


_g_cosmos_t = array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.001,
        0.15,
        0.2,
        0.4,
        0.5,
        0.7,
        0.8,
        0.85,
        0.86,
        0.865,
        0.89,
        0.9,
        0.91,
        0.92,
        0.925,
        0.93,
        0.94,
        0.95,
        0.96,
        0.97,
        0.98,
        0.999,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
)
_g_cosmos_c = array(
    [
        0.00000000e00,
        4.16863836e01,
        1.08898496e03,
        1.11759878e03,
        1.08252322e03,
        8.37179646e02,
        5.01288706e02,
        2.85169581e02,
        1.82341644e01,
        9.86159479e00,
        2.35167895e00,
        -1.16749781e-02,
        2.18333603e-03,
        6.23967745e-03,
        4.25839705e-03,
        5.30413464e-03,
        4.89754389e-03,
        5.11359296e-03,
        4.87944452e-03,
        5.13580057e-03,
        4.78950152e-03,
        5.37471368e-03,
        4.00911261e-03,
        7.30958026e-03,
        -4.03798531e-20,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
    ]
)
_g_cosmos_k = 3
