import numpy
from numpy.random import random as randu
import numba
from numba import jit, autojit, float64, void

from .gexceptions import GMixRangeError

class GPriorBase(object):
    """
    This is the base class.  You need to over-ride a few of
    the functions, see below
    """
    def __init__(self, pars):
        self.pars = numpy.array(pars, dtype='f8')

        # sub-class may want to over-ride this, see GPriorExp
        self.gmax=1.0

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
        output=numpy.zeros(g1arr.size)
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


    def dbyg1_array(self, g1, g2, h=1.e-6):
        """
        Derivative with respect to g1 at the input g1,g2 location
        Used for lensfit.

        Uses central difference and a small enough step size
        to use just two points

        Can over-ride with a jit method for speed
        """
        ff = self.get_prob_array2d(g1+self.hhalf, g2)
        fb = self.get_prob_array2d(g1-self.hhalf, g2)

        return (ff - fb)*self.hinv

    def dbyg2_array(self, g1, g2, h=1.e-6):
        """
        Derivative with respect to g2 at the input g1,g2 location

        Uses central difference and a small enough step size
        to use just two points

        Can over-ride with a jit method for speed
        """
        ff = self.get_prob_array2d(g1, g2+h/2)
        fb = self.get_prob_array2d(g1, g2-h/2)
        return (ff - fb)*self.hinv


 
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



class GPriorBABase(GPriorBase):
    """
    non-jitted methods
    """
    def get_pqr(self, g1in, g2in):
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
        """

        if numpy.isscalar(g1in):
            isscalar=True
        else:
            isscalar=False

        g1 = numpy.array(g1in, dtype='f8', ndmin=1, copy=False)
        g2 = numpy.array(g2in, dtype='f8', ndmin=1, copy=False)

        # these are the same
        #P=self.get_pj(g1, g2, 0.0, 0.0)
        P=self.get_prob_array2d(g1, g2)

        sig2 = self.sig2
        sig4 = self.sig4
        sig2inv = self.sig2inv
        sig4inv = self.sig4inv

        gsq = g1**2 + g2**2
        omgsq = 1. - gsq

        fac = numpy.exp(-0.5*gsq*sig2inv)*omgsq**2

        Qf = fac*(omgsq + 8*sig2)*sig2inv

        Q1 = g1*Qf
        Q2 = g2*Qf

        R11 = (fac * (g1**6 + g1**4*(-2 + 2*g2**2 - 19*sig2) + (1 + g2**2)*sig2*(-1 + g2**2 - 8*sig2) + g1**2*(1 + g2**4 + 20*sig2 + 72*sig4 - 2*g2**2*(1 + 9*sig2))))*sig4inv
        R22 = (fac * (g2**6 + g2**4*(-2 + 2*g1**2 - 19*sig2) + (1 + g1**2)*sig2*(-1 + g1**2 - 8*sig2) + g2**2*(1 + g1**4 + 20*sig2 + 72*sig4 - 2*g1**2*(1 + 9*sig2))))*sig4inv

        R12 = fac * g1*g2 * (80 + omgsq**2*sig4inv + 20*omgsq*sig2inv)

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


#@jit(argtypes=[float64, float64, float64, float64, float64, float64])
@autojit
def _gprior2d_exp_scalar(A, a, g0sq, gmax, g, gsq):

    if g > gmax:
        return 0.0

    numer = A*(1-numpy.exp( (g-gmax)/a ))
    denom = (1+g)*numpy.sqrt(gsq + g0sq)

    prior=numer/denom

    return prior


@autojit
class GPriorExp(GPriorBase):
    def __init__(self, pars):
        """
        [A, a, g0, gmax]
        """
        self.pars=pars

        self.A    = pars[0]
        self.a    = pars[1]
        self.g0   = pars[2]
        self.g0sq = self.g0**2
        self.gmax = pars[3]

        self.h     = 1.e-6
        self.hhalf = 0.5*self.h
        self.hinv  = 1./self.h

    @float64(float64, float64)
    def get_prob_scalar2d(self, g1, g2):
        """
        Get the 2d prob for the input g value
        (1-g^2)^2 * exp(-0.5*g^2/sigma^2)
        """

        gsq = g1**2 + g2**2
        g = numpy.sqrt(gsq)
        return _gprior2d_exp_scalar(self.A, self.a, self.g0sq, self.gmax, g, gsq)

    @float64(float64, float64)
    def get_lnprob_scalar2d(self, g1, g2):
        """
        Get the 2d prob for the input g value
        (1-g^2)^2 * exp(-0.5*g^2/sigma^2)
        """

        gsq = g1**2 + g2**2
        g = numpy.sqrt(gsq)
        p=_gprior2d_exp_scalar(self.A, self.a, self.g0sq, self.gmax, g, gsq)

        if p <= 0.0:
            raise GMixRangeError("g too big: %s" % g)

        lnp=numpy.log(p)
        return lnp


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
            g = numpy.sqrt(gsq)

            p=_gprior2d_exp_scalar(self.A, self.a, self.g0sq, self.gmax, g, gsq)
            output[i] = p


    @float64(float64)
    def get_prob_scalar1d(self, g):
        """
        Get the 1d prior for the input |g| value
        """

        gsq=g**2

        p=_gprior2d_exp_scalar(self.A, self.a, self.g0sq, self.gmax, g, gsq)

        p *= 2*numpy.pi*g
        
        return p


    def fill_prob_array1d(self, garr, output):
        """
        Fill the output with the 1d prob for the input g value
        """
        n=garr.size
        for i in xrange(n):
            p=0.0

            g=garr[i]
            gsq=g**2

            p=_gprior2d_exp_scalar(self.A, self.a, self.g0sq, self.gmax, g, gsq)

            p *= 2*numpy.pi*g

            output[i] = p

    @float64(float64,float64)
    def dbyg1_scalar(self, g1, g2):
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
    def dbyg2_scalar(self, g1, g2):
        """
        Derivative with respect to g2 at the input g1,g2 location
        Used for lensfit.

        Uses central difference and a small enough step size
        to use just two points
        """
        ff = self.get_prob_scalar2d(g1,g2+self.hhalf)
        fb = self.get_prob_scalar2d(g1,g2-self.hhalf)

        return (ff - fb)*self.hinv



@jit
class GPriorBA(GPriorBABase):
    """
    g prior from Bernstein & Armstrong 2013
    """
    @void(float64)
    def __init__(self, sigma):
        """
        pars are scalar gsigma from B&A 
        """
        self.sigma   = sigma
        self.sig2 = self.sigma**2
        self.sig4 = self.sigma**4
        self.sig2inv = 1./self.sig2
        self.sig4inv = 1./self.sig4

        self.gmax=1.0

        self.h=1.e-6
        self.hhalf=0.5*self.h
        self.hinv = 1./self.h

    @float64(float64,float64)
    def get_lnprob_scalar2d(self, g1, g2):
        """
        Get the 2d log prob for the input g value
        """
        gsq = g1*g1 + g2*g2
        omgsq = 1.0 - gsq
        if omgsq <= 0.0:
            raise GMixRangeError("g^2 too big: %s" % gsq)
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
    def dbyg1_scalar(self, g1, g2):
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
    def dbyg2_scalar(self, g1, g2):
        """
        Derivative with respect to g2 at the input g1,g2 location
        Used for lensfit.

        Uses central difference and a small enough step size
        to use just two points
        """
        ff = self.get_prob_scalar2d(g1,g2+self.hhalf)
        fb = self.get_prob_scalar2d(g1,g2-self.hhalf)

        return (ff - fb)*self.hinv



class LogNormalBase(object):
    """
    Lognormal distribution Base, holds non-jitted methods
    """
    def get_lnprob_array(self, x):
        """
        This one no error checking
        """
        if numpy.any(x <= 0):
            raise GMixRangeError("values of x must be > 0")
        logx = numpy.log(x)

        chi2 = self.logivar*(logx-self.logmean)**2

        lnprob = -0.5*chi2 - logx
        return lnprob

    def get_prob_array(self, x):
        """
        Get the probability of x.  x can be an array
        """
        if numpy.any(x <= 0):
            raise GMixRangeError("values of x must be > 0")
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


@jit
class LogNormal(LogNormalBase):
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
    @void(float64,float64)
    def __init__(self, mean, sigma):

        if mean <= 0:
            raise ValueError("mean %s is < 0" % mean)

        self.mean=mean
        self.sigma=sigma

        logmean  = numpy.log(self.mean) - 0.5*numpy.log( 1 + self.sigma**2/self.mean**2 )
        logvar   = numpy.log(1 + self.sigma**2/self.mean**2 )
        logsigma = numpy.sqrt(logvar)
        logivar  = 1./logvar

        self.logmean  = logmean
        self.logvar   = logvar
        self.logsigma = logsigma
        self.logivar  = logivar

    @float64(float64)
    def get_lnprob_scalar(self, x):
        """
        This one no error checking
        """
        if x <= 0:
            raise GMixRangeError("values of x must be > 0")
        logx = numpy.log(x)
        lnp = logx
        lnp -= self.logmean 
        lnp *= lnp
        lnp *= self.logivar
        lnp *= (-0.5)
        lnp -= logx

        return lnp

    @float64(float64)
    def get_prob_scalar(self, x):
        """
        Get the probability of x.
        """
        if x <= 0:
            raise GMixRangeError("values of x must be > 0")
        logx = numpy.log(x)
        lnp = logx
        lnp -= self.logmean 
        lnp *= lnp
        lnp *= self.logivar
        lnp *= (-0.5)
        lnp -= logx
        p=numpy.exp(lnp)
        return p


class CenPriorBase(object):
    """
    Base class provides non-jitted methods
    """
    def sample(self, n=None):
        """
        Get a single sample or arrays
        """
        if n is None:
            rand1=self.cen1 + self.sigma1*numpy.random.randn()
            rand2=self.cen2 + self.sigma2*numpy.random.randn()
        else:
            rand1=self.cen1 + self.sigma1*numpy.random.randn(n)
            rand2=self.cen2 + self.sigma2*numpy.random.randn(n)

        return rand1, rand2


@jit
class CenPrior(CenPriorBase):
    """
    Independent gaussians in each dimension
    """
    @void(float64, float64, float64, float64)
    def __init__(self, cen1, cen2, sigma1, sigma2):

        self.cen1 = cen1
        self.cen2 = cen2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.s2inv1 = 1./self.sigma1**2
        self.s2inv2 = 1./self.sigma2**2


    @float64(float64,float64)
    def get_lnprob(self,p1,p2):
        d1 = self.cen1-p1
        d2 = self.cen2-p2
        lnp = -0.5*d1*d1*self.s2inv1
        lnp -= 0.5*d2*d2*self.s2inv2

        return lnp


def srandu(num=None):
    """
    Generate random numbers in the symmetric distribution [-1,1]
    """
    return 2*(numpy.random.random(num)-0.5)


