import numpy
from numpy.random import random as randu
import numba
from numba import jit, autojit, float64, void

class GPrior(object):
    """
    This is the base class.  You need to over-ride a few of
    the functions, see below
    """
    def __init__(self, pars):
        self.pars = numpy.array(pars, dtype='f8')

        # sub-class may want to over-ride this, see GPriorExp
        self.gmax=numpy.float64(1.0)

    def get_lnprob_scalar2d(self, g1, g2):
        """
        Get the 2d log prob
        """
        raise RuntimeError("over-ride me")

    def get_prob_scalar2d(self, g1, g2):
        """
        Get the 2d prob
        """
        raise RuntimeError("over-ride me")

    def get_prob_array2d(self, g1arr, g2arr):
        """
        Get the 2d prior for the array inputs
        """
        output=numpy.zeros(garr1.size)
        self.fill_prob_array2d(g1arr, g2arr, output)
        return output

    def get_prob_scalar1d(self, g):
        """
        Get the 1d prob
        """
        raise RuntimeError("over-ride me")

    def get_prob_array1d(self, garr):
        """
        Get the 1d prior for the array inputs
        """
        output=numpy.zeros(garr.size)
        self.fill_prob_array1d(garr, output)
        return output


    def dbyg1(self, g1, g2, h=1.e-6):
        """
        Derivative with respect to g1 at the input g1,g2 location
        Used for lensfit.

        Uses central difference and a small enough step size
        to use just two points

        Can over-ride with a jit method for speed
        """
        ff = self.get_prob_scalar2d(g1+h/2, g2)
        fb = self.get_prob_scalar2d(g1-h/2, g2)

        return (ff - fb)/h

    def dbyg2(self, g1, g2, h=1.e-6):
        """
        Derivative with respect to g2 at the input g1,g2 location

        Uses central difference and a small enough step size
        to use just two points

        Can over-ride with a jit method for speed
        """
        ff = self.get_prob_scalar2d(g1, g2+h/2)
        fb = self.get_prob_scalar2d(g1, g2-h/2)
        return (ff - fb)/h


 
    def get_pqr_num(self, g1in, g2in, h=1.e-6):
        """
        Evaluate 
            P
            Q
            R
        From Bernstein & Armstrong

        P is this prior times the jacobian at shear==0

        Q is the gradient of P*J evaluated at shear==0

            [ d(P*J)/dg1, d(P*J)/dg2]_{g=0}

        R is grad of grad of P*J at shear==0
            [ d(P*J)/dg1dg1  d(P*J)/dg1dg2 ]
            [ d(P*J)/dg1dg2  d(P*J)/dg2dg2 ]_{g=0}

        Derivatives are calculated using finite differencing
        """
        if numpy.isscalar(g1in):
            isscalar=True
        else:
            isscalar=False

        g1 = numpy.array(g1in, dtype='f8', ndmin=1, copy=False)
        g2 = numpy.array(g2in, dtype='f8', ndmin=1, copy=False)
        h2=1./(2.*h)
        hsq=1./h**2

        P=self.get_pj(g1, g2, 0.0, 0.0)

        Q1_p = self.get_pj(g1, g2, +h, 0.0)
        Q1_m = self.get_pj(g1, g2, -h, 0.0)
        Q2_p = self.get_pj(g1, g2, 0.0, +h)
        Q2_m = self.get_pj(g1, g2, 0.0, -h)
        R12_pp = self.get_pj(g1, g2, +h, +h)
        R12_mm = self.get_pj(g1, g2, -h, -h)

        Q1 = (Q1_p - Q1_m)*h2
        Q2 = (Q2_p - Q2_m)*h2

        R11 = (Q1_p - 2*P + Q1_m)*hsq
        R22 = (Q2_p - 2*P + Q2_m)*hsq
        R12 = (R12_pp - Q1_p - Q2_p + 2*P - Q1_m - Q2_m + R12_mm)*hsq*0.5

        np=g1.size
        Q = numpy.zeros( (np,2) )
        R = numpy.zeros( (np,2,2) )

        Q[:,0] = Q1
        Q[:,1] = Q2
        R[:,0,0] = R11
        R[:,0,1] = R12
        R[:,1,0] = R12
        R[:,1,1] = R22

        if isscalar:
            P = P[0]
            Q = Q[0,:]
            R = R[0,:,:]

        return P, Q, R

    def get_pj(self, g1, g2, s1, s2):
        """
        PJ = p(g,-shear)*jacob

        where jacob is d(es)/d(eo) and
        es=eo(+)(-g)
        """
        import lensing

        # note sending negative shear to jacob
        s1m=-s1
        s2m=-s2
        J=lensing.shear.dgs_by_dgo_jacob(g1, g2, s1m, s2m)

        # evaluating at negative shear
        g1new,g2new=lensing.shear.gadd(g1, g2, s1m, s2m)
        P=self.get_prob_scalar2d(g1new,g2new)

        return P*J

    def sample2d_pj(self, nrand, s1, s2):
        """
        Get random g1,g2 values from an approximate
        sheared distribution

        parameters
        ----------
        nrand: int
            Number to generate
        """

        maxval2d = self.get_prob_scalar2d(0.0,0.0)
        g1,g2=numpy.zeros(nrand),numpy.zeros(nrand)

        ngood=0
        nleft=nrand
        while ngood < nrand:

            # generate on cube [-1,1,h]
            g1rand=srandu(nleft)
            g2rand=srandu(nleft)

            # a bit of padding since we are modifying the distribution
            fac=1.3
            h = fac*maxval2d*randu(nleft)

            pjvals = self.get_pj(g1rand,g2rand,s1,s2)
            
            w,=numpy.where(h < pjvals)
            if w.size > 0:
                g1[ngood:ngood+w.size] = g1rand[w]
                g2[ngood:ngood+w.size] = g2rand[w]
                ngood += w.size
                nleft -= w.size
   
        return g1,g2



    def sample1d(self, nrand):
        """
        Get random |g| from the 1d distribution

        Set self.gmax appropriately

        parameters
        ----------
        nrand: int
            Number to generate
        """

        if not hasattr(self,'maxval1d'):
            self.set_maxval1d()

        g = numpy.zeros(nrand)

        ngood=0
        nleft=nrand
        while ngood < nrand:

            # generate total g in [0,1)
            grand = self.gmax*randu(nleft)

            # now the height from [0,maxval)
            h = self.maxval1d*randu(nleft)

            pvals = self.get_prob_array1d(grand)

            w,=numpy.where(h < pvals)
            if w.size > 0:
                g[ngood:ngood+w.size] = grand[w]
                ngood += w.size
                nleft -= w.size
   
        return g


    def sample2d(self, nrand):
        """
        Get random g1,g2 values by first drawing
        from the 1-d distribution

        parameters
        ----------
        nrand: int
            Number to generate
        """

        grand=self.sample1d(nrand)
        rangle = randu(nrand)*2*numpy.pi
        g1rand = grand*numpy.cos(rangle)
        g2rand = grand*numpy.sin(rangle)
        return g1rand, g2rand

    def sample2d_brute(self, nrand):
        """
        Get random g1,g2 values using 2-d brute
        force method

        parameters
        ----------
        nrand: int
            Number to generate
        """

        maxval2d = self.get_prob_scalar2d(0.0,0.0)
        g1,g2=numpy.zeros(nrand),numpy.zeros(nrand)

        ngood=0
        nleft=nrand
        while ngood < nrand:
            
            # generate on cube [-1,1,h]
            g1rand=srandu(nleft)
            g2rand=srandu(nleft)

            # a bit of padding since we are modifying the distribution
            h = maxval2d*randu(nleft)

            vals = self.get_prob_array2d(g1rand,g2rand)

            w,=numpy.where(h < vals)
            if w.size > 0:
                g1[ngood:ngood+w.size] = g1rand[w]
                g2[ngood:ngood+w.size] = g2rand[w]
                ngood += w.size
                nleft -= w.size
   
        return g1,g2



    def set_maxval1d(self):
        """
        Use a simple minimizer to find the max value of the 1d 
        distribution
        """
        import scipy.optimize

        (minvalx, fval, iterations, fcalls, warnflag) \
                = scipy.optimize.fmin(self.get_prob_scalar1d_neg,
                                      0.1,
                                      full_output=True, 
                                      disp=False)
        if warnflag != 0:
            raise ValueError("failed to find min: warnflag %d" % warnflag)
        self.maxval1d = -fval

    def get_prob_scalar1d_neg(self, g, *args):
        """
        So we can use the minimizer
        """
        return -self.get_prob_scalar1d(g)





@jit
class GPriorBA(GPrior):
    """
    g prior from Bernstein & Armstrong 2013
    """
    @void(float64)
    def __init__(self, sigma):
        """
        pars are scalar gsigma from B&A 
        """
        self.sigma=sigma
        self.sig2inv = 1./sigma**2

        self.gmax=numpy.float64(1.0)

        self.h=numpy.float64(1.e-6)
        self.hhalf=numpy.float64(0.5*self.h)
        self.hinv = numpy.float64(1./self.h)

    @float64(float64,float64)
    def get_lnprob_scalar2d(self, g1, g2):
        """
        Get the 2d log prob for the input g value
        """
        lnp=-9999.e9
        gsq = g1*g1 + g2*g2
        omgsq = 1.0 - gsq
        if omgsq > 0.0:
            lnp = 2*numpy.log(omgsq) -0.5*gsq*self.sig2inv
        return lnp

    @float64(float64,float64)
    def get_prob_scalar2d(self, g1, g2):
        """
        Get the 2d prob for the input g value
        (1-g^2)^2 * exp(-0.5*g^2/sigma^2)
        """
        p=0.0

        gsq=g1*g1 + g2*g2
        omgsq=1.0-gsq
        if omgsq > 0.0:
            omgsq *= omgsq

            expval = numpy.exp(-0.5*gsq*self.sig2inv)

            p = omgsq*expval
        return p

    @void(float64[:],float64[:],float64[:])
    def fill_prob_array2d(self, g1arr, g2arr, output):
        """
        Fill the output with the 2d prob for the input g value
        """
        n=g1arr.size
        for i in xrange(n):
            p=0.0

            g1=g1arr[i]
            g2=g2arr[i]

            gsq=g1*g1 + g2*g2
            omgsq=1.0-gsq
            if omgsq >= 0.0:
                omgsq *= omgsq
                expval = numpy.exp(-0.5*gsq*self.sig2inv)
                p = omgsq*expval
            output[i] = p

    @float64(float64)
    def get_prob_scalar1d(self, g):
        """
        Get the 1d prior for the input |g| value
        """
        p=0.0

        gsq=g*g
        omgsq=1.0-gsq
        
        if omgsq > 0.0:
            omgsq *= omgsq

            expval = numpy.exp(-0.5*gsq*self.sig2inv)

            p = omgsq*expval

            p *= 2*numpy.pi*g
        return p


    @void(float64[:],float64[:])
    def fill_prob_array1d(self, garr, output):
        """
        Fill the output with the 1d prior for the input g value
        """

        n=garr.size
        for i in xrange(n):
            p = 0.0

            g=garr[i]

            gsq=g*g
            omgsq=1.0-gsq

            if omgsq > 0.0:
                omgsq *= omgsq
                expval = numpy.exp(-0.5*gsq*self.sig2inv)
                p = omgsq*expval
                # can't use pi in the if statement
                #p = 2*numpy.pi*omgsq*expval
            output[i] = 2*numpy.pi*p

    @float64(float64,float64)
    def dbyg1(self, g1, g2):
        """
        Derivative with respect to g1 at the input g1,g2 location
        Used for lensfit.

        Uses central difference and a small enough step size
        to use just two points
        """
        ff = self.get_prob_scalar2d(g1+self.hhalf, g2)
        fb = self.get_prob_scalar2d(g1-self.hhalf, g2)

        return (ff - fb)*self.hinv

    @float64(float64,float64)
    def dbyg2(self, g1, g2):
        """
        Derivative with respect to g2 at the input g1,g2 location
        Used for lensfit.

        Uses central difference and a small enough step size
        to use just two points
        """
        ff = self.get_prob_scalar2d(g1,g2+self.hhalf)
        fb = self.get_prob_scalar2d(g1,g2-self.hhalf)

        return (ff - fb)*self.hinv

'''
class GPriorBAOld(GPrior):
    """
    g prior from Bernstein & Armstrong 2013
    """
    def __init__(self, sigma):
        """
        pars are scalar gsigma from B&A 
        """
        self.sigma=numpy.float64(sigma)
        self.sig2inv = numpy.float64( 1./sigma**2 )
        self.gmax=numpy.float64(1.0)

    def get_max(self):
        return 1.0

    def get_prob_scalar2d(self, g1, g2):
        """
        Get the 2d prior for the input g1,g2 value
        """
        return _ba13_prob_scalar2d(self.sig2inv, g1, g2)

    def get_prob_array2d(self, g1, g2):
        """
        Get the 2d prior for the input g1,g2 value(s)
        """
        output=numpy.zeros(g1.size)
        _ba13_prob_array2d(self.sig2inv, g1, g2, output)
        return output

    def get_prob_scalar1d(self, g):
        """
        Get the 1d prior for the input |g| value
        """
        return _ba13_prob_scalar1d(self.sig2inv, g)
    def get_prob_array1d(self, g):
        """
        Get the 1d prior for the input |g| value(s)
        """
        output=numpy.zeros(g.size)
        _ba13_prob_array1d(self.sig2inv, g, output)
        return output
     
    def get_lnprob_scalar2d(self, g1, g2):
        """
        Get the 2d prior for the input |g| value(s)
        """
        
        return _ba13_lnprob_scalar2d(self.sig2inv, g1, g2)


@jit(argtypes=[float64,float64,float64],restype=float64)
def _ba13_prob_scalar2d(sig2inv, g1, g2):
    """
    (1-gsq)**2*exp(-0.5*gsq/sigma**2)
    """
    gsq=g1*g1 + g2*g2
    fac=1.0-gsq
    fac *= fac

    expval = numpy.exp(-0.5*gsq*sig2inv)

    p = fac*expval
    return p

@jit(argtypes=[ float64,float64[:],float64[:],float64[:] ])
def _ba13_prob_array2d(sig2inv, g1arr, g2arr, output):
    """
    (1-gsq)**2*exp(-0.5*gsq/sigma**2)
    """

    n=g1arr.size
    for i in xrange(n):
        g1=g1arr[i]
        g2=g2arr[i]

        gsq=g1*g1 + g2*g2
        fac=1.0-gsq
        fac *= fac

        expval = numpy.exp(-0.5*gsq*sig2inv)

        output[i] = fac*expval

@jit(argtypes=[float64,float64],restype=float64)
def _ba13_prob_scalar1d(sig2inv, g):
    gsq=g*g
    fac=1.0-gsq
    fac *= fac

    expval = numpy.exp(-0.5*gsq*sig2inv)

    p = fac*expval

    p *= 2*numpy.pi*g
    return p

@jit(argtypes=[ float64,float64[:],float64[:] ])
def _ba13_prob_array1d(sig2inv, garr, output):
    """
    (1-gsq)**2*exp(-0.5*gsq/sigma**2)
    """

    n=garr.size
    for i in xrange(n):
        g=garr[i]

        gsq=g*g
        fac=1.0-gsq
        fac *= fac

        expval = numpy.exp(-0.5*gsq*sig2inv)

        p = fac*expval
        p *= 2*numpy.pi*g
        output[i] = p


@jit(argtypes=[float64,float64,float64],restype=float64)
def _ba13_lnprob_scalar2d(sig2inv, g1, g2):
    """
    p = (1-gsq)**2*exp(-0.5*gsq/sigma**2)

    log(p) = 2*log(1-gsq) -0.5*gsq/sigma**2
    """
    gsq = g1*g1 + g2*g2
    omgsq = 1.0 - gsq
    lnp = 2*numpy.log(omgsq) -0.5*gsq*sig2inv
    return lnp
'''

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
        return numpy.exp(lnp)

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


class CenPrior(object):
    """
    Independent gaussians in each dimension
    """
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

def srandu(num=None):
    """
    Generate random numbers in the symmetric distribution [-1,1]
    """
    return 2*(numpy.random.random(num)-0.5)


