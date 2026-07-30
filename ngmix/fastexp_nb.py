import numpy as np
from numba import njit, vectorize


def _make_exp_lookup(minval=-15, maxval=0):
    """
    lookup array in range [minval,0] inclusive
    """
    nlook = abs(maxval-minval)+1
    expvals = np.zeros(nlook, dtype='f8')

    ivals = np.arange(minval, maxval+1, dtype='i4')

    expvals[:] = np.exp(ivals)

    return ivals, expvals


def _make_smooth_exp_coeffs():
    """
    coefficients of the degree 5 polynomial P that minimizes the
    relative error P(f)/exp(f) - 1 on [-1/2, 1/2] in least squares,
    subject to the constraints

        P(1/2) = e P(-1/2)
        P'(1/2) = e P'(-1/2)
        P''(1/2) = e P''(-1/2)

    With these, neighboring pieces exp(i) * P(x - i) and their first
    two derivatives match exactly at the half integer break points,
    so the piecewise function used by exp5_smooth is C2.  The least
    squares fit makes the relative error oscillate in sign, so its
    mean over an interval is ~1e-9 and rendered models are not
    scaled by the mean error.

    This generates the hard coded coefficients in exp5_smooth; it is
    kept for documentation and for testing
    """
    from numpy.polynomial import chebyshev

    ncoef = 6
    ident = np.eye(ncoef)

    # fit the relative error on a grid, using a chebyshev basis in
    # t = 2 f for conditioning; rows are exp(-f) T_k(t), target 1
    fvals = np.linspace(-0.5, 0.5, 1001)
    tvals = 2 * fvals
    amat = np.empty((fvals.size, ncoef))
    for k in range(ncoef):
        amat[:, k] = chebyshev.chebval(tvals, ident[k])
    amat *= np.exp(-fvals)[:, None]
    bvec = np.ones(fvals.size)

    # constraint rows: the m-th f derivative at f = 1/2 minus e times
    # the same at f = -1/2, noting d/df = 2 d/dt
    ncon = 3
    cmat = np.zeros((ncon, ncoef))
    for m in range(ncon):
        for k in range(ncoef):
            dcoef = chebyshev.chebder(ident[k], m)
            cmat[m, k] = 2.0 ** m * (
                chebyshev.chebval(1.0, dcoef)
                - np.exp(1.0) * chebyshev.chebval(-1.0, dcoef)
            )

    # solve the constrained least squares problem via the KKT system
    kkt = np.zeros((ncoef + ncon, ncoef + ncon))
    kkt[:ncoef, :ncoef] = 2 * amat.T @ amat
    kkt[:ncoef, ncoef:] = cmat.T
    kkt[ncoef:, :ncoef] = cmat
    rhs = np.zeros(ncoef + ncon)
    rhs[:ncoef] = 2 * amat.T @ bvec

    ccheb = np.linalg.solve(kkt, rhs)[:ncoef]

    # chebyshev in t to monomial in t to monomial in f = t/2
    return chebyshev.cheb2poly(ccheb) * 2.0 ** np.arange(ncoef)


FASTEXP_MAX_CHI2 = 25.0

# gaussian evaluations are apodized rather than cut: apod_window
# is 1 below FASTEXP_APOD_CHI2 and rolls off smoothly to 0 at
# FASTEXP_MAX_CHI2
FASTEXP_APOD_CHI2 = 20.0
_APOD_IWIDTH = 1.0 / (FASTEXP_MAX_CHI2 - FASTEXP_APOD_CHI2)

# we limit to chi squared of 25, which means an argument of
# -0.5*25. Use -15 to be safe
_EXP_IVALS, _EXP_LOOKUP = _make_exp_lookup(
    minval=-15,
    maxval=0,
)
_EXP_I0 = _EXP_IVALS[0]


@njit
def apod_window(chi2):
    """
    smooth apodization window for the chi^2 truncation of gaussian
    evaluations

    A quintic smoothstep going from 1 at FASTEXP_APOD_CHI2 to 0 at
    FASTEXP_MAX_CHI2 with zero first and second derivatives at both
    ends, so an apodized gaussian is C2 in its parameters where the
    plain truncation has a step at the boundary

    no range checking is done here: only apply the window for chi2
    in [FASTEXP_APOD_CHI2, FASTEXP_MAX_CHI2]

    Parameters
    ----------
    chi2: number
        the chi^2 argument of the gaussian
    """
    u = (FASTEXP_MAX_CHI2 - chi2) * _APOD_IWIDTH
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


@njit
def apod_window_deriv(chi2):
    """
    derivative of apod_window with respect to chi^2

    no range checking is done here: only apply the window for chi2
    in [FASTEXP_APOD_CHI2, FASTEXP_MAX_CHI2]

    Parameters
    ----------
    chi2: number
        the chi^2 argument of the gaussian
    """
    u = (FASTEXP_MAX_CHI2 - chi2) * _APOD_IWIDTH
    umu = u * (1.0 - u)
    return -30.0 * umu * umu * _APOD_IWIDTH


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


# generated by _make_smooth_exp_coeffs
_EXP5_SMOOTH_COEFFS = np.array([
    1.0000011318561302,
    0.999993601071577,
    0.49992478810274166,
    0.16674612720799442,
    0.042330947141114836,
    0.008197933236258961,
])


@njit
def exp5_smooth(x):
    """
    fast exponential with continuous value and derivatives

    Like exp5 this evaluates exp(i) * P(f) with x = i + f split at
    half integers, but P is a constrained fit to exp(f) on
    [-1/2, 1/2] built so that neighboring pieces and their first two
    derivatives match exactly at the break points (see
    _make_smooth_exp_coeffs).  Unlike exp3/exp4/exp5 the value and
    first two derivatives have no steps at half integer arguments.

    in the range -15, 0 the relative error is at worst about 2.0e-6
    in the value and 2.8e-5 in the first derivative, and its mean
    is ~1e-9 so rendered models are not scaled by the mean error

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
    expval *= 1.0000011318561302 + f*(
        0.999993601071577 + f*(
            0.49992478810274166 + f*(
                0.16674612720799442 + f*(
                    0.042330947141114836 + f*0.008197933236258961
                )
            )
        )
    )

    return expval


fexp = exp5_smooth


@vectorize
def fexp_arr(x):
    return fexp(x)
