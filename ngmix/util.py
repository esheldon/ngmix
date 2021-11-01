from sys import stdout
import numpy as np


def print_pars(pars, fmt="%8.3g", front=None, stream=stdout, logger=None):
    """
    print the parameters with a uniform width

    Parameters
    ----------
    pars: array/sequence
        The parameters to print
    fmt: string
        The format string for each number
    front: string
        A string to put at the front
    stream: optional
        Default stdout
    logger: logger
        If True, use the logger to print a debug statement than the stream
    """
    txt = ""
    if front is not None:
        txt += front
        txt += " "
    if pars is None:
        txt += "%s" % None
    else:
        s = format_pars(pars, fmt=fmt)
        txt += s

    if logger is not None:
        logger.debug(txt)
    else:
        stream.write(txt + '\n')


def format_pars(pars, fmt="%8.3g"):
    """
    get a nice string of the pars with no line breaks

    Parameters
    ----------
    pars: array/sequence
        The parameters to print
    fmt: string
        The format string for each number

    Returns
    --------
    the string
    """
    fmt = " ".join([fmt + " "] * len(pars))
    return fmt % tuple(pars)


def get_ratio_var(a, b, var_a, var_b, cov_ab):
    """
    get (a/b)**2 and variance in mean of (a/b)
    """

    if np.any(b == 0):
        raise ValueError("zero in denominator")

    rsq = (a/b)**2

    var = rsq * (var_a/a**2 + var_b/b**2 - 2*cov_ab/(a*b))
    return var


def get_ratio_error(a, b, var_a, var_b, cov_ab):
    """
    get a/b and error on a/b
    """

    var = get_ratio_var(a, b, var_a, var_b, cov_ab)

    var = np.clip(var, 0.0, np.inf)
    error = np.sqrt(var)
    return error
