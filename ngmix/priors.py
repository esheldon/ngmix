import math
import numpy
import numba
from numba import jit, autojit, float64

class LogNormal(object):
    """
    Lognormal distribution

    parameters
    ----------
    mean:
        such that <x> in linear space is mean.  This implies the mean in log(x)
        is 
            <log(x)> = log(mean) - 0.5*log( 1 + sigma**2/mean**2 )
    sigma:
        such than the variace in linear space is sigma**2.  This implies
        the variance in log space is
            var(log(x)) = log( 1 + sigma**2/mean**2 )
    norm: optional
        When calling eval() the return value will be norm*prob(x)


    methods
    -------
    sample(nrand):
        Get nrand random deviates from the distribution
    lnprob(x):
        Get the natural logarithm of the probability of x.  x can
        be an array
    prob(x):
        Get the probability of x.  x can be an array
    """
    def __init__(self, mean, sigma):
        from numpy import log,exp,sqrt,pi
        mean=numpy.float64(mean)
        sigma=numpy.float64(sigma)

        self.dist="LogNormal"

        if mean <= 0:
            raise ValueError("mean %s is < 0" % mean)

        self.mean=mean
        self.sigma=sigma

        self.logmean = numpy.float64( log(mean) - 0.5*log( 1 + sigma**2/mean**2 ) )
        self.logvar = numpy.float64( log(1 + sigma**2/mean**2 ) )
        self.logsigma = sqrt(self.logvar)
        self.logivar = numpy.float64( 1./self.logvar )

        #self.nconst = numpy.float64( 1/sqrt(2*pi*self.logvar) )
        #self.logofnconst = log(self.nconst)

        self.mode=exp(self.logmean - self.logvar)
        self.maxval_lnprob = self.get_lnprob_scalar(self.mode)
        self.maxval = exp(self.maxval_lnprob)

    def get_dist_name(self):
        """
        Get the name of this distribution
        """
        return self.dist
 
    def get_mean(self):
        """
        Get the mean of the distribution
        """
        return self.mean

    def get_sigma(self):
        """
        Get the width sigma of the distribution
        """
        return self.sigma

    def get_mode(self):
        """
        Get the location of the peak
        """
        return self.mode

    def get_max(self):
        """
        Get maximum value of this distribution
        """
        return self.maxval

    def get_max_lnprob(self):
        """
        Get maximum value ln(prob) of this distribution
        """
        return self.maxval_lnprob

    def get_lnprob_scalar(self, x):
        """
        This one no error checking
        """
        return _lognorm_lnprob(self.logmean, self.logivar, x)

    def get_lnprob_array(self, x):
        """
        This one no error checking
        """
        logx = numpy.log(x)

        chi2 = self.logivar*(logx-self.logmean)**2

        #lnprob = self.logofnconst - 0.5*chi2 - logx
        lnprob = -0.5*chi2 - logx
        return lnprob


    def get_prob(self, x):
        """
        Get the probability of x.
        """
        if isinstance(x,numpy.ndarray):
            return self.get_prob_array(x)
        else:
            return self.get_prob_scalar(x)

    def get_prob_scalar(self, x):
        """
        Get the probability of x.  x can be an array
        """
        if x <= 0:
            raise ValueError("values of x must be > 0")
        lnp=self.get_lnprob_scalar(x)
        return math.exp(lnp)

    def get_prob_array(self, x):
        """
        Get the probability of x.  x can be an array
        """
        if numpy.any(x <= 0):
            raise ValueError("values of x must be > 0")
        lnp=self.get_lnprob_array(x)
        return numpy.exp(lnp)


    def sample(self, nrand=None):
        """
        Get nrand random deviates from the distribution

        If z is drawn from a normal random distribution, then exp(logmean+logsigma*z)
        is drawn from lognormal
        """
        if nrand is None:
            z=numpy.random.randn()
        else:
            z=numpy.random.randn(nrand)
        return exp(self.logmean + self.logsigma*z)

@jit(argtypes=[float64,float64,float64],
     restype=float64)
def _lognorm_lnprob(logmean, logivar, x):
    """
    chi2 = self.logivar*(logx-self.logmean)**2
    lnprob = - 0.5*chi2 - logx
    """
    logx = numpy.log(x)
    lnp = logx
    lnp -= logmean 
    lnp *= lnp
    lnp *= logivar
    lnp *= (-0.5)
    lnp -= logx

    return lnp


class CenPrior:
    def __init__(self, cen, sigma):
        self.cen=numpy.array(cen, dtype='f8')
        self.sigma=numpy.array(sigma, dtype='f8')

        one=numpy.float64(1.0)
        self.cen1 = numpy.float64(cen[0])
        self.cen2 = numpy.float64(cen[1])
        self.sigma1=numpy.float64(sigma[0])
        self.sigma2=numpy.float64(sigma[1])
        self.s2inv1=one/self.sigma1**2
        self.s2inv2=one/self.sigma2**2

        self.minusofive = numpy.float64(-0.5)

    def get_max(self):
        return 1.0

    def get_lnprob(self,pos):
        return _cen_lnprob(self.cen1,
                           self.cen2,
                           self.s2inv1,
                           self.s2inv2,
                           pos[0],
                           pos[1])

    def sample(self, n=None):
        """
        Get a single sample
        """
        if n is None:
            rand=self.cen + self.sigma*numpy.random.randn(2)
        else:
            rand = numpy.random.randn(n,2).reshape(n,2)
            rand[:,0] *= self.sigma[0]
            rand[:,0] += self.cen[0]

            rand[:,1] *= self.sigma[1]
            rand[:,1] += self.cen[1]

        return rand

@jit(argtypes=[float64,float64,float64,float64,float64,float64],
     restype=float64)
def _cen_lnprob(cen1,cen2,s2inv1,s2inv2,p1,p2):
    d1=cen1-p1
    d2=cen2-p2
    lnp = -0.5*d1*d1*s2inv1
    lnp -= 0.5*d2*d2*s2inv2

    return lnp
