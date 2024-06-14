"""
Constrained multivariate least-squares optimization

Copied from https://github.com/jjhelmus/leastsqbound-scipy

Placing a copy in this repo because that code is no longer maintained
"""

import numpy as np
from numpy import diag
import logging
import warnings

from numpy import take, eye, triu, transpose, dot, finfo
from numpy import empty_like, sqrt, cos, sin, arcsin, asarray
from numpy import atleast_1d, shape, issubdtype, dtype, inexact
from scipy.optimize import _minpack, leastsq
from ..util import print_pars
from ..flags import (
    ZERO_DOF,
    DIV_ZERO,
    LM_SINGULAR_MATRIX,
    LM_NEG_COV_EIG,
    LM_NEG_COV_DIAG,
    EIG_NOTFINITE,
    LM_FUNC_NOTFINITE,
)
from ..defaults import PDEF, CDEF

LOGGER = logging.getLogger(__name__)


def run_leastsq(func, guess, n_prior_pars, **keys):
    """
    run leastsq from scipy.optimize.  Deal with certain types of errors.  Allow
    the user to input boundaries for parameters, dealt with in a robust way
    using the leastsqbound code that wraps leastsq

    Parameters
    ----------
    func:
        the function to minimize
    guess:
        guess at pars
    n_prior_pars:
        number of slots in fdiff for priors
    bounds : list, optional
        ``(min, max)`` pairs for each element in ``x``, defining
        the bounds on that parameter. Use None for one of ``min`` or
        ``max`` when there is no bound in that direction.

    some useful keywords
    maxfev:
        maximum number of function evaluations. e.g. 1000
    epsfcn:
        Step for jacobian estimation (derivatives). 1.0e-6
    ftol:
        Relative error desired in sum of squares, 1.0e06
    xtol:
        Relative error desired in solution. 1.0e-6
    """

    npars = guess.size
    k_space = keys.pop("k_space", False)

    res = {}
    try:
        lm_tup = leastsqbound(func, guess, full_output=1, **keys)

        pars, pcov0, infodict, errmsg, ier = lm_tup

        if ier == 0:
            # wrong args, this is a bug
            raise ValueError(errmsg)

        flags = 0
        if ier > 4:
            flags |= 2 ** (ier - 5)
            pars, pcov, perr = _get_def_stuff(npars)
            LOGGER.debug(errmsg)

        elif pcov0 is None:
            # why on earth is this not in the flags?
            flags |= LM_SINGULAR_MATRIX
            errmsg = "singular covariance"
            LOGGER.debug(errmsg)
            print_pars(pars, front="    pars at singular:", logger=LOGGER)
            junk, pcov, perr = _get_def_stuff(npars)
        else:
            # Scale the covariance matrix returned from leastsq; this will
            # recover the covariance of the parameters in the right units.
            fdiff = func(pars)

            # npars: to remove priors

            if k_space:
                dof = (fdiff.size - n_prior_pars) // 2 - npars
            else:
                dof = fdiff.size - n_prior_pars - npars

            if dof == 0:
                junk, pcov, perr = _get_def_stuff(npars)
                flags |= ZERO_DOF
            else:
                s_sq = (fdiff[n_prior_pars:] ** 2).sum() / dof
                pcov = pcov0 * s_sq

                cflags = _test_cov(pcov)
                if cflags != 0:
                    flags |= cflags
                    errmsg = "bad covariance matrix"
                    LOGGER.debug(errmsg)
                    junk1, junk2, perr = _get_def_stuff(npars)
                else:
                    # only if we reach here did everything go well
                    perr = sqrt(diag(pcov))

        res["flags"] = flags
        res["nfev"] = infodict["nfev"]
        res["ier"] = ier
        res["errmsg"] = errmsg

        res["pars"] = pars
        res["pars_err"] = perr
        res["pars_cov0"] = pcov0
        res["pars_cov"] = pcov

    except ValueError as e:
        serr = str(e)
        if "NaNs" in serr or "infs" in serr:
            pars, pcov, perr = _get_def_stuff(npars)

            res["pars"] = pars
            res["pars_cov0"] = pcov
            res["pars_cov"] = pcov
            res["nfev"] = -1
            res["flags"] = LM_FUNC_NOTFINITE
            res["errmsg"] = "not finite"
            LOGGER.debug("not finite")
        else:
            raise e

    except ZeroDivisionError:
        pars, pcov, perr = _get_def_stuff(npars)

        res["pars"] = pars
        res["pars_cov0"] = pcov
        res["pars_cov"] = pcov
        res["nfev"] = -1

        res["flags"] = DIV_ZERO
        res["errmsg"] = "zero division"
        LOGGER.debug("zero division")

    return res


def _get_def_stuff(npars):
    pars = np.zeros(npars) + PDEF
    cov = np.zeros((npars, npars)) + CDEF
    err = np.zeros(npars) + CDEF
    return pars, cov, err


def _test_cov(pcov):
    flags = 0
    try:
        e, v = np.linalg.eig(pcov)
        (weig,) = np.where(e < 0)
        if weig.size > 0:
            flags |= LM_NEG_COV_EIG

        (wneg,) = np.where(np.diag(pcov) < 0)
        if wneg.size > 0:
            flags |= LM_NEG_COV_DIAG

    except np.linalg.linalg.LinAlgError:
        flags |= EIG_NOTFINITE

    return flags


def _internal2external_grad(xi, bounds):
    """
    Calculate the internal (unconstrained) to external (constained)
    parameter gradiants.
    """
    grad = empty_like(xi)
    for i, (v, bound) in enumerate(zip(xi, bounds)):
        lower, upper = bound
        if lower is None and upper is None:  # No constraints
            grad[i] = 1.0
        elif upper is None:     # only lower bound
            grad[i] = v / sqrt(v * v + 1.)
        elif lower is None:     # only upper bound
            grad[i] = -v / sqrt(v * v + 1.)
        else:   # lower and upper bounds
            grad[i] = (upper - lower) * cos(v) / 2.
    return grad


def _internal2external_func(bounds):
    """
    Make a function which converts between internal (unconstrained) and
    external (constrained) parameters.
    """
    ls = [_internal2external_lambda(b) for b in bounds]

    def convert_i2e(xi):
        xe = empty_like(xi)
        xe[:] = [lam(p) for lam, p in zip(ls, xi)]
        return xe

    return convert_i2e


def _internal2external_lambda(bound):
    """
    Make a lambda function which converts a single internal (uncontrained)
    parameter to a external (constrained) parameter.
    """
    lower, upper = bound

    if lower is None and upper is None:  # no constraints
        return lambda x: x
    elif upper is None:     # only lower bound
        return lambda x: lower - 1. + sqrt(x * x + 1.)
    elif lower is None:     # only upper bound
        return lambda x: upper + 1. - sqrt(x * x + 1.)
    else:
        return lambda x: lower + ((upper - lower) / 2.) * (sin(x) + 1.)


def _external2internal_func(bounds):
    """
    Make a function which converts between external (constrained) and
    internal (unconstrained) parameters.
    """
    ls = [_external2internal_lambda(b) for b in bounds]

    def convert_e2i(xe):
        xi = empty_like(xe)
        xi[:] = [lam(p) for lam, p in zip(ls, xe)]
        return xi

    return convert_e2i


def _external2internal_lambda(bound):
    """
    Make a lambda function which converts an single external (constrained)
    parameter to a internal (unconstrained) parameter.
    """
    lower, upper = bound

    if lower is None and upper is None:  # no constraints
        return lambda x: x
    elif upper is None:     # only lower bound
        return lambda x: sqrt((x - lower + 1.) ** 2 - 1)
    elif lower is None:     # only upper bound
        return lambda x: sqrt((upper - x + 1.) ** 2 - 1)
    else:
        return lambda x: arcsin((2. * (x - lower) / (upper - lower)) - 1.)


def _check_func(checker, argname, thefunc, x0, args, numinputs,
                output_shape=None):
    res = atleast_1d(thefunc(*((x0[:numinputs],) + args)))
    if (output_shape is not None) and (shape(res) != output_shape):
        if (output_shape[0] != 1):
            if len(output_shape) > 1:
                if output_shape[1] == 1:
                    return shape(res)
            msg = "%s: there is a mismatch between the input and output " \
                  "shape of the '%s' argument" % (checker, argname)
            func_name = getattr(thefunc, '__name__', None)
            if func_name:
                msg += " '%s'." % func_name
            else:
                msg += "."
            raise TypeError(msg)
    if issubdtype(res.dtype, inexact):
        dt = res.dtype
    else:
        dt = dtype(float)
    return shape(res), dt


def leastsqbound(func, x0, args=(), bounds=None, Dfun=None, full_output=0,
                 col_deriv=0, ftol=1.49012e-8, xtol=1.49012e-8,
                 gtol=0.0, maxfev=0, epsfcn=None, factor=100, diag=None):
    """
    Bounded minimization of the sum of squares of a set of equations.

    ::

        x = arg min(sum(func(y)**2,axis=0))
                 y

    Parameters
    ----------
    func : callable
        should take at least one (possibly length N vector) argument and
        returns M floating point numbers.
    x0 : ndarray
        The starting estimate for the minimization.
    args : tuple
        Any extra arguments to func are placed in this tuple.
    bounds : list
        ``(min, max)`` pairs for each element in ``x``, defining
        the bounds on that parameter. Use None for one of ``min`` or
        ``max`` when there is no bound in that direction.
    Dfun : callable
        A function or method to compute the Jacobian of func with derivatives
        across the rows. If this is None, the Jacobian will be estimated.
    full_output : bool
        non-zero to return all optional outputs.
    col_deriv : bool
        non-zero to specify that the Jacobian function computes derivatives
        down the columns (faster, because there is no transpose operation).
    ftol : float
        Relative error desired in the sum of squares.
    xtol : float
        Relative error desired in the approximate solution.
    gtol : float
        Orthogonality desired between the function vector and the columns of
        the Jacobian.
    maxfev : int
        The maximum number of calls to the function. If zero, then 100*(N+1) is
        the maximum where N is the number of elements in x0.
    epsfcn : float
        A suitable step length for the forward-difference approximation of the
        Jacobian (for Dfun=None). If epsfcn is less than the machine precision,
        it is assumed that the relative errors in the functions are of the
        order of the machine precision.
    factor : float
        A parameter determining the initial step bound
        (``factor * || diag * x||``). Should be in interval ``(0.1, 100)``.
    diag : sequence
        N positive entries that serve as a scale factors for the variables.

    Returns
    -------
    x : ndarray
        The solution (or the result of the last iteration for an unsuccessful
        call).
    cov_x : ndarray
        Uses the fjac and ipvt optional outputs to construct an
        estimate of the jacobian around the solution.  ``None`` if a
        singular matrix encountered (indicates very flat curvature in
        some direction).  This matrix must be multiplied by the
        residual standard deviation to get the covariance of the
        parameter estimates -- see curve_fit.
    infodict : dict
        a dictionary of optional outputs with the key s::

            - 'nfev' : the number of function calls
            - 'fvec' : the function evaluated at the output
            - 'fjac' : A permutation of the R matrix of a QR
                     factorization of the final approximate
                     Jacobian matrix, stored column wise.
                     Together with ipvt, the covariance of the
                     estimate can be approximated.
            - 'ipvt' : an integer array of length N which defines
                     a permutation matrix, p, such that
                     fjac*p = q*r, where r is upper triangular
                     with diagonal elements of nonincreasing
                     magnitude. Column j of p is column ipvt(j)
                     of the identity matrix.
            - 'qtf'  : the vector (transpose(q) * fvec).

    mesg : str
        A string message giving information about the cause of failure.
    ier : int
        An integer flag.  If it is equal to 1, 2, 3 or 4, the solution was
        found.  Otherwise, the solution was not found. In either case, the
        optional output variable 'mesg' gives more information.

    Notes
    -----
    "leastsq" is a wrapper around MINPACK's lmdif and lmder algorithms.

    cov_x is a Jacobian approximation to the Hessian of the least squares
    objective function.
    This approximation assumes that the objective function is based on the
    difference between some observed target data (ydata) and a (non-linear)
    function of the parameters `f(xdata, params)` ::

           func(params) = ydata - f(xdata, params)

    so that the objective function is ::

           min   sum((ydata - f(xdata, params))**2, axis=0)
         params

    Contraints on the parameters are enforced using an internal parameter list
    with appropiate transformations such that these internal parameters can be
    optimized without constraints. The transfomation between a given internal
    parameter, p_i, and a external parameter, p_e, are as follows:

    With ``min`` and ``max`` bounds defined ::

        p_i = arcsin((2 * (p_e - min) / (max - min)) - 1.)
        p_e = min + ((max - min) / 2.) * (sin(p_i) + 1.)

    With only ``max`` defined ::

        p_i = sqrt((max - p_e + 1.)**2 - 1.)
        p_e = max + 1. - sqrt(p_i**2 + 1.)

    With only ``min`` defined ::

        p_i = sqrt((p_e - min + 1.)**2 - 1.)
        p_e = min - 1. + sqrt(p_i**2 + 1.)

    These transfomations are used in the MINUIT package, and described in
    detail in the section 1.3.1 of the MINUIT User's Guide.

    When a parameter being optimized takes an on values near one of the
    imposed bounds the optimization can become blocked and the solution
    returned may be non-optimal.  In addition, near the limits cov_x and
    related returned variables do not contain meanful results.

    For best results, bounds of parameters should be limited to only those
    which are absolutly necessary, limits should be made wide enough to avoid
    parameters taking values near these limits and the optimization should be
    repeated without limits after a satisfactory minimum has been found for
    best error analysis. See section 1.3 and 5.3 of the MINUIT User's Guide
    for addition discussion of this topic.

    To Do
    -----
    Currently the ``factor`` and ``diag`` parameters scale the
    internal parameter list, but should scale the external parameter list.

    The `qtf` vector in the infodic dictionary reflects internal parameter
    list, it should be correct to reflect the external parameter list.

    References
    ----------
    * F. James and M. Winkler. MINUIT User's Guide, July 16, 2004.

    """
    # use leastsq if no bounds are present
    if bounds is None:
        return leastsq(func, x0, args, Dfun, full_output, col_deriv,
                       ftol, xtol, gtol, maxfev, epsfcn, factor, diag)

    # create function which convert between internal and external parameters
    i2e = _internal2external_func(bounds)
    e2i = _external2internal_func(bounds)

    x0 = asarray(x0).flatten()
    i0 = e2i(x0)
    n = len(x0)
    if len(bounds) != n:
        raise ValueError('length of x0 != length of bounds')
    if not isinstance(args, tuple):
        args = (args,)
    shape, dtype = _check_func('leastsq', 'func', func, x0, args, n)
    m = shape[0]
    if n > m:
        raise TypeError('Improper input: N=%s must not exceed M=%s' % (n, m))
    if epsfcn is None:
        epsfcn = finfo(dtype).eps

    # define a wrapped func which accept internal parameters, converts them
    # to external parameters and calls func
    def wfunc(x, *args):
        return func(i2e(x), *args)

    if Dfun is None:
        if (maxfev == 0):
            maxfev = 200 * (n + 1)
        retval = _minpack._lmdif(wfunc, i0, args, full_output, ftol, xtol,
                                 gtol, maxfev, epsfcn, factor, diag)
    else:
        if col_deriv:
            _check_func('leastsq', 'Dfun', Dfun, x0, args, n, (n, m))
        else:
            _check_func('leastsq', 'Dfun', Dfun, x0, args, n, (m, n))
        if (maxfev == 0):
            maxfev = 100 * (n + 1)

        def wDfun(x, *args):  # wrapped Dfun
            scale = _internal2external_grad(x, bounds)
            if col_deriv == 1:
                scale = scale.reshape(len(x), 1)
            return Dfun(i2e(x), *args)*scale

        retval = _minpack._lmder(wfunc, wDfun, i0, args, full_output,
                                 col_deriv, ftol, xtol, gtol, maxfev,
                                 factor, diag)

    errors = {0: ["Improper input parameters.", TypeError],
              1: ["Both actual and predicted relative reductions "
                  "in the sum of squares\n  are at most %f" % ftol, None],
              2: ["The relative error between two consecutive "
                  "iterates is at most %f" % xtol, None],
              3: ["Both actual and predicted relative reductions in "
                  "the sum of squares\n  are at most %f and the "
                  "relative error between two consecutive "
                  "iterates is at \n  most %f" % (ftol, xtol), None],
              4: ["The cosine of the angle between func(x) and any "
                  "column of the\n  Jacobian is at most %f in "
                  "absolute value" % gtol, None],
              5: ["Number of calls to function has reached "
                  "maxfev = %d." % maxfev, ValueError],
              6: ["ftol=%f is too small, no further reduction "
                  "in the sum of squares\n  is possible.""" % ftol,
                  ValueError],
              7: ["xtol=%f is too small, no further improvement in "
                  "the approximate\n  solution is possible." % xtol,
                  ValueError],
              8: ["gtol=%f is too small, func(x) is orthogonal to the "
                  "columns of\n  the Jacobian to machine "
                  "precision." % gtol, ValueError],
              'unknown': ["Unknown error.", TypeError]}

    info = retval[-1]    # The FORTRAN return value

    if (info not in [1, 2, 3, 4] and not full_output):
        if info in [5, 6, 7, 8]:
            warnings.warn(errors[info][0], RuntimeWarning)
        else:
            try:
                raise errors[info][1](errors[info][0])
            except KeyError:
                raise errors['unknown'][1](errors['unknown'][0])

    mesg = errors[info][0]
    x = i2e(retval[0])  # internal params to external params

    if full_output:
        # convert fjac from internal params to external
        grad = _internal2external_grad(retval[0], bounds)
        retval[1]['fjac'] = (retval[1]['fjac'].T / take(grad,
                             retval[1]['ipvt'] - 1)).T
        cov_x = None
        if info in [1, 2, 3, 4]:
            from numpy.linalg import inv
            from numpy.linalg import LinAlgError
            perm = take(eye(n), retval[1]['ipvt'] - 1, 0)
            r = triu(transpose(retval[1]['fjac'])[:n, :])
            R = dot(r, perm)
            try:
                cov_x = inv(dot(transpose(R), R))
            except (LinAlgError, ValueError):
                pass
        return (x, cov_x) + retval[1:-1] + (mesg, info)
    else:
        return (x, info)
