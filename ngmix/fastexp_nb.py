import numpy
from numba import jit, njit

# will check > -26 and < 0.0 so these are not actually necessary
_exp3_ivals = numpy.array([
    -26, -25, -24, -23, -22, -21, 
    -20, -19, -18, -17, -16, -15, -14,
    -13, -12, -11, -10,  -9,  -8,  -7,
    -6,  -5,  -4,  -3,  -2,  -1,   0,
])

_exp3_i0=-26

_exp3_lookup = numpy.array([
    5.10908903e-12,   1.38879439e-11,   3.77513454e-11,
    1.02618796e-10,   2.78946809e-10,   7.58256043e-10,
    2.06115362e-09,   5.60279644e-09,   1.52299797e-08,
    4.13993772e-08,   1.12535175e-07,   3.05902321e-07,
    8.31528719e-07,   2.26032941e-06,   6.14421235e-06,
    1.67017008e-05,   4.53999298e-05,   1.23409804e-04,
    3.35462628e-04,   9.11881966e-04,   2.47875218e-03,
    6.73794700e-03,   1.83156389e-02,   4.97870684e-02,
    1.35335283e-01,   3.67879441e-01,   1.00000000e+00,
])

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



