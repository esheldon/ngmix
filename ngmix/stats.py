import numpy
from numpy import zeros

def calc_mcmc_stats(data, sigma_clip=False, weights=None, **kw):
    """
    Get the mean and covariance for the input mcmc trials

    parameters
    ----------
    data: array
        [N, npars] array
    weights: array, optional
        [N] array of weights
    """

    send_data=data
    send_weights=weights
    if sigma_clip:
        keep, ok = get_sigma_clipped_indices(data, weights=weights, **kw)
        if ok:
            send_data=data[keep]
            if weights is not None:
                send_weights=weights[keep]

    if weights is not None:
        return _calc_weighted_stats(send_data, send_weights)
    else:
        return _calc_stats(send_data)

def get_sigma_clipped_indices(data, weights=None, **kw):
    import esutil as eu
    npoints = data.shape[0]
    npar = data.shape[1]
    keep=numpy.arange(npoints)
    ok=True
    for i in range(npar):
        if weights is None:
            send_weights=None
        else:
            send_weights=weights[keep]

        tmean,tstd,tind = eu.stat.sigma_clip(data[keep,i],
                                             weights=send_weights,
                                             get_indices=True,
                                             **kw)

        if tind.size < 4:
            print(
                'warning: sigma clip removed too many points, '
                'skipping sigma clip'
            )
            ok=False
            break
        keep = keep[tind]

    return keep, ok

def _calc_stats(data):
    ntrials=data.shape[0]
    npar = data.shape[1]

    means = zeros(npar,dtype='f8')
    cov = zeros( (npar,npar), dtype='f8')

    for i in range(npar):
        means[i] = data[:, i].mean()

    num=ntrials

    for i in range(npar):
        idiff = data[:,i]-means[i]
        for j in range(i,npar):
            if i == j:
                jdiff = idiff
            else:
                jdiff = data[:,j]-means[j]

            cov[i,j] = (idiff*jdiff).sum()/(num-1)

            if i != j:
                cov[j,i] = cov[i,j]

    return means, cov

def _calc_weighted_stats(data, weights):
    if weights.size != data.shape[0]:
        raise ValueError("weights not same size as data")

    npar = data.shape[1]

    wsum = weights.sum()

    if wsum <= 0.0:
        for i in range(data.shape[0]//100):
            print(i,data[i,:])
        raise ValueError("wsum <= 0: %s" % wsum)

    means = zeros(npar,dtype='f8')
    cov = zeros( (npar,npar), dtype='f8')

    for i in range(npar):
        dsum = (data[:, i]*weights).sum()
        means[i] = dsum/wsum

    for i in range(npar):
        idiff = data[:,i]-means[i]
        for j in range(i,npar):
            if i == j:
                jdiff = idiff
            else:
                jdiff = data[:,j]-means[j]

            wvar = ( weights*idiff*jdiff ).sum()/wsum
            cov[i,j] = wvar

            if i != j:
                cov[j,i] = cov[i,j]

    return means, cov


