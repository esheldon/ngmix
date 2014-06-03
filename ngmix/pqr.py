"""
class PQR
"""
from __future__ import print_function

try:
    xrange = xrange
    # We have Python 2
except:
    xrange = range
    # We have Python 3


import numpy
from numpy import array, where, zeros
from .gexceptions import GMixRangeError, GMixFatalError

_def_shear_expand=array([0.0, 0.0])

def calc_pqr(g, g_prior, shear_expand=_def_shear_expand, remove_prior=False):
    """
    calculate the P,Q,R terms from Bernstein & Armstrong

    parameters
    ----------
    g: 2-d array
        g values [N,2]
    g_prior:
        The g prior object.
    shear_expand: 2-element sequence
        Shear to expand about.  Default [0.0,0.0]
    remove_prior: bool, optional
        Remove the prior value from the Q,R terms.  This is needed
        if the prior was used in likelihood exploration.
    """

    o=PQR(g, g_prior, shear_expand=shear_expand, remove_prior=remove_prior)
    P,Q,R = o.get_pqr()

    return P,Q,R

def calc_shear(P, Q, R, get_sums=False):
    """
    Extract a shear estimate from the p,q,r values from
    Bernstein & Armstrong

    parameters
    ----------
    P: array[nobj]
        Prior times jacobian
    Q: array[nobj,2]
        gradient of P with respect to shear
    R: array[nobj,2,2]
        gradient of gradient

    output
    ------
    [g1,g2]: array

    notes
    -----
    If done on a single object, the operations would look simpler

    QQ = numpy.outer(Q,Q)
    Cinv = QQ/P**2 - R/P
    C = numpy.linalg.inv(Cinv)
    g1g2 = numpy.dot(C,Q/P)

    """

    P_sum, Q_sum, Cinv_sum = calc_pqr_sums(P,Q,R)

    g1g2, C = combine_pqr_sums(P_sum, Q_sum, Cinv_sum)

    if get_sums:
        return g1g2, C, Q_sum, Cinv_sum
    else:
        return g1g2, C

class PQR(object):
    def __init__(self, g, g_prior,
                 shear_expand=_def_shear_expand, remove_prior=False):
        """
        A class to calculate the P,Q,R terms from Bernstein & Armstrong

        parameters
        ----------
        g: 2-d array
            g values [N,2]
        g_prior:
            The g prior object.
        shear_expand: 2-element sequence
            Shear to expand about.  Default [0.0,0.0]
        remove_prior: bool, optional
            Remove the prior value from the Q,R terms.  This is needed
            if the prior was used in likelihood exploration.
        """

        self._g=g
        self._g_prior=g_prior

        self._shear_expand=array(shear_expand)
        assert self._shear_expand.size==2,"shear expand should have two elements"

        self._remove_prior=remove_prior

        self._calc_pqr()

    def get_pqr(self):
        """
        get P,Q,R
        """
        return self._P,self._Q,self._R
    
    def get_nuse(self):
        """
        get number of points used.  Will be less than the input number
        of points if remove_prior is set and some prior values were zero
        """
        return self._nuse

    def _calc_pqr(self):
        """
        get the P,Q,R
        """

        g_prior=self._g_prior

        g1=self._g[:,0]
        g2=self._g[:,1]

        sh=self._shear_expand
        print("        expanding pqr about:",sh)
        if hasattr(g_prior,'get_pqr_expand'):
            Pi,Qi,Ri = g_prior.get_pqr_expand(g1,g2, sh[0], sh[1])
        else:
            Pi,Qi,Ri = g_prior.get_pqr_num(g1, g2, s1=sh[0], s2=sh[1])

        self._P,self._Q,self._R = self._get_mean_pqr(Pi,Qi,Ri)

    def _get_mean_pqr(self, Pi, Qi, Ri):
        """
        Get the mean P,Q,R marginalized over priors.  Optionally weighted for
        importance sampling
        """

        if self._remove_prior:
            # We measured the posterior surface.  But the integrals are over
            # the likelihood.  So divide by the prior.
            #
            # Also note the p we divide by is in principle different from the
            # Pi above, which are evaluated at the shear expansion value

            print("        undoing prior for pqr")

            g1=self._g[:,0]
            g2=self._g[:,1]
            prior_vals = self._g_prior.get_prob_array2d(g1,g2)

            w,=numpy.where(prior_vals > 0.0)

            Pinv = 1.0/prior_vals[w]
            Pinv_sum=Pinv.sum()

            Pi = Pi[w]
            Qi = Qi[w,:]
            Ri = Ri[w,:,:]

            # this is not unity if expanding about some shear
            Pi *= Pinv
            Qi[:,0] *= Pinv 
            Qi[:,1] *= Pinv

            Ri[:,0,0] *= Pinv
            Ri[:,0,1] *= Pinv
            Ri[:,1,0] *= Pinv
            Ri[:,1,1] *= Pinv

            P = Pi.sum()/Pinv_sum
            Q = Qi.sum(axis=0)/Pinv_sum
            R = Ri.sum(axis=0)/Pinv_sum

        else:
            P = Pi.mean()
            Q = Qi.mean(axis=0)
            R = Ri.mean(axis=0)

        self._nuse=Pi.size

        return P,Q,R


def calc_pqr_sums(P,Q,R):
    """
    Create the sums used to calculate shear from BA13 PQR
    """
    from numpy import zeros,where

    n=P.size

    wbad, = where(P <= 0)
    if wbad.size != 0:
        raise ValueError('Found P <= 0: %s/%s' % (wbad.size,n) )

    QQ       = zeros( (n,2,2), dtype=Q.dtype)
    Cinv_all = zeros( (n,2,2), dtype=Q.dtype)
    QbyP     = zeros( (n,2),   dtype=Q.dtype)

    # outer product
    QQ[:,0,0] = Q[:,0]*Q[:,0]
    QQ[:,0,1] = Q[:,0]*Q[:,1]
    QQ[:,1,0] = Q[:,1]*Q[:,0]
    QQ[:,1,1] = Q[:,1]*Q[:,1]

    Pinv = 1/P
    P2inv = Pinv*Pinv

    # QQ/P**2 - R/P
    Cinv_all[:,0,0] = QQ[:,0,0]*P2inv - R[:,0,0]*Pinv
    Cinv_all[:,0,1] = QQ[:,0,1]*P2inv - R[:,0,1]*Pinv
    Cinv_all[:,1,0] = QQ[:,1,0]*P2inv - R[:,1,0]*Pinv
    Cinv_all[:,1,1] = QQ[:,1,1]*P2inv - R[:,1,1]*Pinv

    P_sum = P.sum()

    Cinv_sum = Cinv_all.sum(axis=0)

    QbyP[:,0] = Q[:,0]*Pinv
    QbyP[:,1] = Q[:,1]*Pinv
    Q_sum = QbyP.sum(axis=0)

    return P_sum, Q_sum, Cinv_sum

def combine_pqr_sums(P_sum, Q_sum, Cinv_sum):
    """
    Combine the sums from calc_pqr_sums to
    get a shear and covariance matrix
    """

    # linalg doesn't support f16 if that is the type of above
    # arguments
    C = numpy.linalg.inv(Cinv_sum.astype('f8')).astype(P_sum.dtype)
    g1g2 = numpy.dot(C,Q_sum)

    return g1g2, C


def pqr_jackknife(P, Q, R,
                  chunksize=1,
                  get_sums=False,
                  get_shears=False,
                  progress=False,
                  show=False,
                  eps=None,
                  png=None):
    """
    Get the shear covariance matrix using jackknife resampling.

    The trick is that this must be done in pairs for ring tests

    chunksize is the number of *pairs* to remove for each chunk
    """

    if progress:
        import progressbar
        pg=progressbar.ProgressBar(width=70)

    ntot = P.size
    if ( (ntot % 2) != 0 ):
        raise  ValueError("expected factor of two, got %d" % ntot)
    npair = ntot/2

    # some may not get used
    nchunks = npair/chunksize

    print('getting overall sums')
    P_sum, Q_sum, Cinv_sum = calc_pqr_sums(P,Q,R)
    C = numpy.linalg.inv(Cinv_sum)
    shear = numpy.dot(C,Q_sum)

    print('doing jackknife')
    shears = numpy.zeros( (nchunks, 2) )
    for i in xrange(nchunks):

        beg = i*chunksize*2
        end = (i+1)*chunksize*2
        
        if progress:
            frac=float(i+1)/nchunks
            pg.update(frac=frac)

        Ptmp = P[beg:end]
        Qtmp = Q[beg:end,:]
        Rtmp = R[beg:end,:,:]

        P_sum, Q_sum_tmp, Cinv_sum_tmp = \
                calc_pqr_sums(Ptmp,Qtmp,Rtmp)
        
        Q_sum_tmp    = Q_sum - Q_sum_tmp
        Cinv_sum_tmp = Cinv_sum - Cinv_sum_tmp

        Ctmp = numpy.linalg.inv(Cinv_sum_tmp)
        shear_tmp = numpy.dot(C,Q_sum_tmp)

        shears[i, :] = shear_tmp

    shear_cov = numpy.zeros( (2,2) )
    fac = (nchunks-1)/float(nchunks)

    shear = shears.mean(axis=0)

    shear_cov[0,0] = fac*( ((shear[0]-shears[:,0])**2).sum() )
    shear_cov[0,1] = fac*( ((shear[0]-shears[:,0]) * (shear[1]-shears[:,1])).sum() )
    shear_cov[1,0] = shear_cov[0,1]
    shear_cov[1,1] = fac*( ((shear[1]-shears[:,1])**2).sum() )

    if show or eps or png:
        _plot_shears(shears, show=show, eps=eps, png=png)

    if get_sums:
        return shear, shear_cov, Q_sum, Cinv_sum
    elif get_shears:
        return shear, shear_cov, shears
    else:
        return shear, shear_cov

def pqr_in_chunks(P, Q, R, chunksize):
    """
    Get the mean shear in chunks.  They will be in order
    """


    ntot = P.size
    if ( (ntot % 2) != 0 ):
        raise  ValueError("expected factor of two, got %d" % ntot)
    npair = ntot/2

    # some may not get used
    nchunks = npair/chunksize

    shears = numpy.zeros( (nchunks, 2) )
    covs = numpy.zeros( (nchunks, 2, 2) )
    for i in xrange(nchunks):
        print('%d/%d' % (i+1, nchunks))

        beg = i*chunksize*2
        end = (i+1)*chunksize*2

        Ptmp = P[beg:end]
        Qtmp = Q[beg:end,:]
        Rtmp = R[beg:end,:,:]

        sh, C = calc_pqr_shear(Ptmp, Qtmp, Rtmp)

        shears[i, :] = sh
        covs[i, :, :] = C

    return shears, covs


def pqr_bootstrap(P, Q, R, nsamples, verbose=False, show=False, eps=None, png=None):
    """
    Get the shear covariance matrix using boot resampling.

    The trick is that this must be done in pairs
    """

    if verbose:
        import progressbar
        pg=progressbar.ProgressBar(width=70)

    ntot = P.size
    if ( (ntot % 2) != 0 ):
        raise  ValueError("expected factor of two, got %d" % ntot)

    npair = ntot/2

    Pboot = P.copy()
    Qboot = Q.copy()
    Rboot = R.copy()

    rind1 = numpy.zeros(npair, dtype='i8')
    rind2 = numpy.zeros(npair, dtype='i8')
    rind = numpy.zeros(ntot, dtype='i8')

    shears = numpy.zeros( (nsamples, 2) )

    for i in xrange(nsamples):
        if verbose:
            frac=float(i+1)/nsamples
            pg.update(frac=frac)

        # first of the pair
        rind1[:] = 2*numpy.random.randint(low=0,high=npair,size=npair)
        # second of the pair
        rind2[:] = rind1[:]+1

        rind[0:npair] = rind1
        rind[npair:]  = rind2

        Pboot[:] = P[rind]
        Qboot[:,:] = Q[rind,:]
        Rboot[:,:,:] = R[rind,:,:]

        sh, C_not_used =  calc_pqr_shear(Pboot, Qboot, Rboot)

        shears[i, :] = sh

    shear = shears.mean(axis=0)

    shear_cov = numpy.zeros( (2,2) )

    shear_cov[0,0] = ( (shears[:,0]-shear[0])**2 ).sum()/(nsamples-1)
    shear_cov[0,1] = ( (shears[:,0]-shear[0])*(shears[:,1]-shear[1]) ).sum()/(nsamples-1)
    shear_cov[1,0] = shear_cov[0,1]
    shear_cov[1,1] = ( (shears[:,1]-shear[1])**2 ).sum()/(nsamples-1)

    if show or eps or png:
        _plot_shears(shears, show=show, eps=eps, png=png)

    return shear, shear_cov



def _plot_shears(shears, show=True, eps=None, png=None):
    import biggles
    import esutil as eu
    tab=biggles.Table(2,1)
    std=shears.std(axis=0)

    plt1=eu.plotting.bhist(shears[:,0], binsize=0.2*std[0],
                           color='blue',show=False,
                           xlabel=r'$\gamma_1$')
    plt2=eu.plotting.bhist(shears[:,1], binsize=0.2*std[1],
                           color='red',show=False,
                           xlabel=r'$\gamma_2$')
    tab[0,0] = plt1
    tab[1,0] = plt2

    if png is not None:
        print(png)
        tab.write_img(800,800,png)
    if eps is not None:
        print(eps)
        tab.write_eps(eps)


    if show:
        tab.show()


