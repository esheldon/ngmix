from __future__ import print_function
import numpy
from numba import jit, njit

from .fastexp import make_exp_lookup

_exp3_ivals, _exp3_lookup = make_exp_lookup(
    minval=-300,
    maxval=0,
)
_exp3_i0 = _exp3_ivals[0]

@njit(cache=True)
def exp3(x):
    """
    fast exponential

    no range checking is done here, do it at the caller

    x: number
        any number
    """
    ival = int(x-0.5)
    f = x - ival
    index = ival-_exp3_i0
    expval = _exp3_lookup[index]
    expval *= (6+f*(6+f*(3+f)))*0.16666666

    return expval
