from sys import stdout


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
        stream.write(txt)


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
