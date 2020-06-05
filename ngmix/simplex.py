"""
stolen from scipy to avoid dependency
"""
import numpy

# standard status messages of optimizers
_status_message = {'success': 'Optimization terminated successfully.',
                   'maxfev': 'Maximum number of function evaluations has '
                              'been exceeded.',
                   'maxiter': 'Maximum number of iterations has been '
                              'exceeded.',
                   'pr_loss': 'Desired error not necessarily achieved due '
                              'to precision loss.'}



def minimize_neldermead(func, x0,
                        xtol=1.0e-3,
                        ftol=1.0e-3,
                        maxiter=None,
                        maxfev=None,
                        disp=False,
                        **unused_kw):
    """
    stolen from scipy.optimize, modified in a number of ways, including
    different tolerances for each parameter and using relative
    tolerances

    Minimization of scalar function of one or more variables using the
    Nelder-Mead algorithm.

    Options for the Nelder-Mead algorithm are:
        disp : bool
            Set to True to print convergence messages.
        xtol : float
            Relative difference in simplex vertex locations in consecutive
            iterations that signals convergence.
        ftol : float
            Relative difference in function value at vertex locations in
            consecutive iterations that signals convergence.
        maxiter : int
            Maximum number of iterations to perform.
        maxfev : int
            Maximum number of function evaluations to make.
    """

    fcalls = 0
    x0 = numpy.asfarray(x0).flatten()
    N = len(x0)
    rank = len(x0.shape)
    if not -1 < rank < 2:
        raise ValueError("Initial guess must be a scalar or rank-1 sequence.")
    if maxiter is None:
        maxiter = N * 200
    if maxfev is None:
        maxfev = maxiter

    rho = 1
    chi = 2
    psi = 0.5
    sigma = 0.5
    one2np1 = list(range(1, N + 1))

    if rank == 0:
        sim = numpy.zeros((N + 1,), dtype=x0.dtype)
    else:
        sim = numpy.zeros((N + 1, N), dtype=x0.dtype)
    fsim = numpy.zeros((N + 1,), float)
    sim[0] = x0

    fsim[0] = func(x0)
    fcalls += 1
    nonzdelt = 0.05
    zdelt = 0.00025
    for k in range(0, N):
        y = numpy.array(x0, copy=True)
        if y[k] != 0:
            y[k] = (1 + nonzdelt)*y[k]
        else:
            y[k] = zdelt

        sim[k + 1] = y
        f = func(y)
        fcalls += 1
        fsim[k + 1] = f

    ind = numpy.argsort(fsim)
    fsim = numpy.take(fsim, ind, 0)
    # sort so sim[0,:] has the lowest function value
    sim = numpy.take(sim, ind, 0)

    iterations = 1

    while (fcalls < maxfev and iterations < maxiter):

        xdiff = numpy.abs(sim[1:, :] - sim[0,:])
        xvt = numpy.abs(sim[0,:])*xtol
        xcomp = (xdiff - xvt).max()

        fdiff = numpy.abs( (fsim[0] - fsim[1:]) ).max()
        fvt = numpy.abs(fsim[0])*ftol
        fcomp = fdiff - fvt

        if xcomp < 0. and fcomp < 0.:
            break

        xbar = numpy.add.reduce(sim[:-1], 0) / N
        xr = (1 + rho) * xbar - rho * sim[-1]
        fxr = func(xr)
        fcalls += 1
        doshrink = 0

        if fxr < fsim[0]:
            xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
            fxe = func(xe)
            fcalls += 1

            if fxe < fxr:
                sim[-1] = xe
                fsim[-1] = fxe
            else:
                sim[-1] = xr
                fsim[-1] = fxr
        else:  # fsim[0] <= fxr
            if fxr < fsim[-2]:
                sim[-1] = xr
                fsim[-1] = fxr
            else:  # fxr >= fsim[-2]
                # Perform contraction
                if fxr < fsim[-1]:
                    xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                    fxc = func(xc)
                    fcalls += 1

                    if fxc <= fxr:
                        sim[-1] = xc
                        fsim[-1] = fxc
                    else:
                        doshrink = 1
                else:
                    # Perform an inside contraction
                    xcc = (1 - psi) * xbar + psi * sim[-1]
                    fxcc = func(xcc)
                    fcalls += 1

                    if fxcc < fsim[-1]:
                        sim[-1] = xcc
                        fsim[-1] = fxcc
                    else:
                        doshrink = 1

                if doshrink:
                    for j in one2np1:
                        sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                        fsim[j] = func(sim[j])
                        fcalls += 1

        ind = numpy.argsort(fsim)
        sim = numpy.take(sim, ind, 0)
        fsim = numpy.take(fsim, ind, 0)

        iterations += 1

    x = sim[0]
    fval = numpy.min(fsim)
    warnflag = 0

    if fcalls >= maxfev:
        warnflag = 1
        msg = _status_message['maxfev']
        if disp:
            print('Warning: ' + msg)
    elif iterations >= maxiter:
        warnflag = 2
        msg = _status_message['maxiter']
        if disp:
            print('Warning: ' + msg)
    else:
        msg = _status_message['success']
        if disp:
            print(msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % iterations)
            print("         Function evaluations: %d" % fcalls)

    result={'x':x,
            'fun':fval,
            'nfev':fcalls,
            'nit':iterations,
            'status':warnflag,
            'success':(warnflag == 0),
            'message':msg}

    return result

minimize_neldermead_rel = minimize_neldermead


def minimize_neldermead_old(func, x0,
                        xtol=1e-4,
                        ftol=1e-4,
                        maxiter=None,
                        maxfev=None,
                        disp=False,
                        **unused_kw):
    """
    stolen from scipy.optimize, slightly modified

    Minimization of scalar function of one or more variables using the
    Nelder-Mead algorithm.

    Options for the Nelder-Mead algorithm are:
        disp : bool
            Set to True to print convergence messages.
        xtol : float
            Absolute difference in simplex vertex locations in consecutive
            iterations that signals convergence.
        ftol : float
            Absolute difference in function value at vertex locations in
            consecutive iterations that signals convergence.
        maxiter : int
            Maximum number of iterations to perform.
        maxfev : int
            Maximum number of function evaluations to make.
    """

    fcalls = 0
    x0 = numpy.asfarray(x0).flatten()
    N = len(x0)
    rank = len(x0.shape)
    if not -1 < rank < 2:
        raise ValueError("Initial guess must be a scalar or rank-1 sequence.")
    if maxiter is None:
        maxiter = N * 200
    if maxfev is None:
        maxfev = maxiter

    rho = 1
    chi = 2
    psi = 0.5
    sigma = 0.5
    one2np1 = list(range(1, N + 1))

    if rank == 0:
        sim = numpy.zeros((N + 1,), dtype=x0.dtype)
    else:
        sim = numpy.zeros((N + 1, N), dtype=x0.dtype)
    fsim = numpy.zeros((N + 1,), float)
    sim[0] = x0

    fsim[0] = func(x0)
    fcalls += 1
    nonzdelt = 0.05
    zdelt = 0.00025
    for k in range(0, N):
        y = numpy.array(x0, copy=True)
        if y[k] != 0:
            y[k] = (1 + nonzdelt)*y[k]
        else:
            y[k] = zdelt

        sim[k + 1] = y
        f = func(y)
        fcalls += 1
        fsim[k + 1] = f

    ind = numpy.argsort(fsim)
    fsim = numpy.take(fsim, ind, 0)
    # sort so sim[0,:] has the lowest function value
    sim = numpy.take(sim, ind, 0)

    iterations = 1

    while (fcalls < maxfev and iterations < maxiter):
        if (numpy.max(numpy.ravel(numpy.abs(sim[1:] - sim[0]))) <= xtol and
                numpy.max(numpy.abs(fsim[0] - fsim[1:])) <= ftol):
            break

        xbar = numpy.add.reduce(sim[:-1], 0) / N
        xr = (1 + rho) * xbar - rho * sim[-1]
        fxr = func(xr)
        fcalls += 1
        doshrink = 0

        if fxr < fsim[0]:
            xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
            fxe = func(xe)
            fcalls += 1

            if fxe < fxr:
                sim[-1] = xe
                fsim[-1] = fxe
            else:
                sim[-1] = xr
                fsim[-1] = fxr
        else:  # fsim[0] <= fxr
            if fxr < fsim[-2]:
                sim[-1] = xr
                fsim[-1] = fxr
            else:  # fxr >= fsim[-2]
                # Perform contraction
                if fxr < fsim[-1]:
                    xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                    fxc = func(xc)
                    fcalls += 1

                    if fxc <= fxr:
                        sim[-1] = xc
                        fsim[-1] = fxc
                    else:
                        doshrink = 1
                else:
                    # Perform an inside contraction
                    xcc = (1 - psi) * xbar + psi * sim[-1]
                    fxcc = func(xcc)
                    fcalls += 1

                    if fxcc < fsim[-1]:
                        sim[-1] = xcc
                        fsim[-1] = fxcc
                    else:
                        doshrink = 1

                if doshrink:
                    for j in one2np1:
                        sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                        fsim[j] = func(sim[j])
                        fcalls += 1

        ind = numpy.argsort(fsim)
        sim = numpy.take(sim, ind, 0)
        fsim = numpy.take(fsim, ind, 0)

        iterations += 1

    x = sim[0]
    fval = numpy.min(fsim)
    warnflag = 0

    if fcalls >= maxfev:
        warnflag = 1
        msg = _status_message['maxfev']
        if disp:
            print('Warning: ' + msg)
    elif iterations >= maxiter:
        warnflag = 2
        msg = _status_message['maxiter']
        if disp:
            print('Warning: ' + msg)
    else:
        msg = _status_message['success']
        if disp:
            print(msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % iterations)
            print("         Function evaluations: %d" % fcalls)

    result={'x':x,
            'fun':fval,
            'nfev':fcalls,
            'nit':iterations,
            'status':warnflag,
            'success':(warnflag == 0),
            'message':msg}

    return result

