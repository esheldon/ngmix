from __future__ import print_function

try:
    xrange = xrange
    # We have Python 2
except:
    xrange = range
    # We have Python 3

from numpy import zeros

def calc_mcmc_stats(data, weights=None):
    """
    Get the mean and covariance for the input mcmc trials

    parameters
    ----------
    data: array
        [N, npars] array
    weights: array, optional
        [N] array of weights
    """
    if weights is not None:
        return _calc_weighted_stats(data, weights)
    else:
        return _calc_stats(data)

def _calc_stats(data):
    ntrials=data.shape[0]
    npar = data.shape[1]

    means = zeros(npar,dtype='f8')
    cov = zeros( (npar,npar), dtype='f8')

    for i in xrange(npar):
        means[i] = data[:, i].mean()

    num=ntrials

    for i in xrange(npar):
        idiff = data[:,i]-means[i]
        for j in xrange(i,npar):
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
        for i in xrange(data.shape[0]/100):
            print(i,data[i,:])
        raise ValueError("wsum <= 0: %s" % wsum)

    means = zeros(npar,dtype='f8')
    cov = zeros( (npar,npar), dtype='f8')

    for i in xrange(npar):
        dsum = (data[:, i]*weights).sum()
        means[i] = dsum/wsum

    for i in xrange(npar):
        idiff = data[:,i]-means[i]
        for j in xrange(i,npar):
            if i == j:
                jdiff = idiff
            else:
                jdiff = data[:,j]-means[j]

            wvar = ( weights*idiff*jdiff ).sum()/wsum
            cov[i,j] = wvar

            if i != j:
                cov[j,i] = cov[i,j]

    return means, cov


