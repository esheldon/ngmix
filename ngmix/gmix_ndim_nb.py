import numpy
from numba import njit


@njit
def gmixnd_get_prob(log_pnorms,
                    means,
                    icovars,
                    pars,
                    xdiff,
                    tmp_lnprob,
                    dolog):
    """
    evaluate the gaussian mixture

    parameters
    ----------
    log_pnorms: array
        array of size number of gaussians
    means: array
        array of shape [n_gauss, n_dim]
    icovars: array
        array of shape [n_gauss, n_dim, n_dim]
    pars: array
        array of shape [n_dim]
    xdiff: array
        scratch array of shape [n_dim]
    tmp_lnprob: array
        scratch array of shape [n_gauss]
    dolog: int
        0 if the return value should be linear
    """

    n_dim = means.shape[1]
    n_gauss = log_pnorms.size
    lnpmax=-9.99e9

    for i in range(n_gauss):

        logpnorm = log_pnorms[i]

        for idim1 in range(n_dim):
            par = pars[idim1]
            mean = means[i,idim1]

            xdiff[idim1] = par-mean

        chi2=0.0
        for idim1 in range(n_dim):
            for idim2 in range(n_dim):
                icov = icovars[i,idim1,idim2]

                chi2 += xdiff[idim1]*xdiff[idim2]*icov

        lnp = -0.5*chi2 + logpnorm
        if lnp > lnpmax:
            lnpmax=lnp

        tmp_lnprob[i] = lnp

    p=0.0
    for i in range(n_gauss):
        p += numpy.exp(tmp_lnprob[i] - lnpmax)

    if dolog:
        retval = numpy.log(p) + lnpmax
    else:
        retval = p*numpy.exp(lnpmax)

    return retval

