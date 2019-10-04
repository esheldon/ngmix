import numpy as np
from numba import njit


def _make_exp_lookup(minval=-15, maxval=0):
    """
    lookup array in range [minval,0] inclusive
    """
    nlook = abs(maxval-minval)+1
    expvals = np.zeros(nlook, dtype='f8')

    ivals = np.arange(minval, maxval+1, dtype='i4')

    expvals[:] = np.exp(ivals)

    return ivals, expvals


FASTEXP_MAX_CHI2 = 25.0

# we limit to chi squared of 25, which means an argument of
# -0.5*25. Use -15 to be safe
_EXP_IVALS, _EXP_LOOKUP = _make_exp_lookup(
    minval=-15,
    maxval=0,
)
_EXP_I0 = _EXP_IVALS[0]


@njit
def exp3(x):
    """
    fast exponential

    in the range -15, 0 the relative error is at worst about -0.004

    no range checking is done here, do it at the caller

    Parameters
    ----------
    x: number
        a number.  You should check it is in the valid range for
        the lookup table
    """
    ival = int(x-0.5)
    f = x - ival
    index = ival - _EXP_I0
    expval = _EXP_LOOKUP[index]
    expval *= (6+f*(6+f*(3+f)))*0.16666666

    return expval


@njit
def exp4(x):
    """
    fast exponential

    in the range -15, 0 the relative error is at worst about -0.0002

    no range checking is done here, do it at the caller

    Parameters
    ----------
    x: number
        a number.  You should check it is in the valid range for
        the lookup table
    """

    ival = int(x-0.5)
    f = x - ival
    index = ival - _EXP_I0
    expval = _EXP_LOOKUP[index]
    expval *= (24+f*(24+f*(12+f*(4+f))))*0.041666666

    return expval


@njit
def exp5(x):
    """
    fast exponential

    in the range -15, 0 the relative error is at worst about -4.0e-5

    no range checking is done here, do it at the caller

    Parameters
    ----------
    x: number
        a number.  You should check it is in the valid range for
        the lookup table
    """

    ival = int(x-0.5)
    f = x - ival
    index = ival - _EXP_I0
    expval = _EXP_LOOKUP[index]
    expval *= (120+f*(120+f*(60+f*(20+f*(5+f)))))*0.0083333333

    return expval
