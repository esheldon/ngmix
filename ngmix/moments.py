from __future__ import print_function
import numpy
from ._gmix import GMixRangeError
from . import shape

class Deriv(object):
    """
    class to calculate derivatives of moments with respect to shear

    use numerical derivatives

    parameters
    ----------
    M1: float or array
        M1 values
    M2: float or array
        M2 values
    T:  float or array
        T values
    h:  float, optional
        step size.  not currently used since we only
        have the zero shear methods which are all analytic
    """
    def __init__(self, M1, M2, T, shear=None,h=1.0e-3):

        self.h=h
        self.h2inv = 1/(2*h)

        self.hsqinv=1./h**2

        self.set_moms(M1, M2, T)

    #
    # derivatives at shear==0
    #

    def dTds1z(self):
        """
        derivative of T with respect to shear1 at zero shear
        """

        #h=self.h
        #_,_,Tp0=get_sheared_moments(self.M1, self.M2, self.T, +h, 0.0)
        #_,_,Tm0=get_sheared_moments(self.M1, self.M2, self.T, -h, 0.0)

        #return (Tp0-Tm0)*self.h2inv

        # at zero shear
        return 2*self.M1

    def dTds2z(self):
        """
        derivative of T with respect to shear2 at zero shear
        """
        #h=self.h
        #_,_,T0p=get_sheared_moments(self.M1, self.M2, self.T, 0.0, +h)
        #_,_,T0m=get_sheared_moments(self.M1, self.M2, self.T, 0.0, -h)

        #return (T0p-T0m)*self.h2inv

        # at zero shear
        return 2*self.M2

    def d2Tds1ds1z(self):
        """
        2nd derivative of T with respect to shear1 and shear1 at zero shear
        """

        #h=self.h
        #_,_,Tp0=get_sheared_moments(self.M1, self.M2, self.T, +h, 0.0)
        #_,_,Tm0=get_sheared_moments(self.M1, self.M2, self.T, -h, 0.0)

        #return (Tp0 - 2*self.T + Tm0)*self.hsqinv

        # at zero shear
        return 4*self.T

    def d2Tds1ds2z(self):
        """
        2nd derivative of T with respect to shear1 and shear2 at zero shear
        """

        #h=self.h
        #_,_,Tp0=get_sheared_moments(self.M1, self.M2, self.T, +h, 0.0)
        #_,_,Tm0=get_sheared_moments(self.M1, self.M2, self.T, -h, 0.0)
        #_,_,T0p=get_sheared_moments(self.M1, self.M2, self.T, 0.0, +h)
        #_,_,T0m=get_sheared_moments(self.M1, self.M2, self.T, 0.0, -h)

        #_,_,Tpp=get_sheared_moments(self.M1, self.M2, self.T, +h, +h)
        #_,_,Tmm=get_sheared_moments(self.M1, self.M2, self.T, -h, -h)

        #return (Tpp - Tp0 - T0p + 2*self.T - Tm0 - T0m + Tmm)*self.hsqinv*0.5

        # at zero shear
        return self.zero

    def d2Tds2ds2z(self):
        """
        2nd derivative of T with respect to shear2 and shear2 at zero shear
        """

        #h=self.h
        #_,_,T0p=get_sheared_moments(self.M1, self.M2, self.T, 0.0, +h)
        #_,_,T0m=get_sheared_moments(self.M1, self.M2, self.T, 0.0, -h)

        #return (T0p - 2*self.T + T0m)*self.hsqinv

        # at zero shear
        return 4*self.T

    def dM1ds1z(self):
        """
        derivative of M1 with respect to shear1 at zero shear
        """

        #h=self.h
        #Mp0,_,_=get_sheared_moments(self.M1, self.M2, self.T, +h, 0.0)
        #Mm0,_,_=get_sheared_moments(self.M1, self.M2, self.T, -h, 0.0)

        #return (Mp0-Mm0)*self.h2inv

        # at zero shear
        return 2*self.T


    def dM1ds2z(self):
        """
        derivative of M1 with respect to shear2 at zero shear
        """
        #h=self.h
        #M0p,_,_=get_sheared_moments(self.M1, self.M2, self.T, 0.0, +h)
        #M0m,_,_=get_sheared_moments(self.M1, self.M2, self.T, 0.0, -h)

        #return (M0p-M0m)*self.h2inv

        # zero shear
        return self.zero

    def d2M1ds1ds1z(self):
        """
        2nd derivative of M1 with respect to shear1 and shear1 at zero shear
        """
        #h=self.h
        #Mp0,_,_=get_sheared_moments(self.M1, self.M2, self.T, +h, 0.0)
        #Mm0,_,_=get_sheared_moments(self.M1, self.M2, self.T, -h, 0.0)

        #return (Mp0 - 2*self.M1 + Mm0)*self.hsqinv
        # for zero shear seems to always be 4*M1?

        return 4*self.M1

    def d2M1ds1ds2z(self):
        """
        2nd derivative of M1 with respect to shear1 and shear2 at zero shear
        """

        #h=self.h
        #Mp0,_,_=get_sheared_moments(self.M1, self.M2, self.T, +h, 0.0)
        #Mm0,_,_=get_sheared_moments(self.M1, self.M2, self.T, -h, 0.0)
        #M0p,_,_=get_sheared_moments(self.M1, self.M2, self.T, 0.0, +h)
        #M0m,_,_=get_sheared_moments(self.M1, self.M2, self.T, 0.0, -h)

        #Mpp,_,_=get_sheared_moments(self.M1, self.M2, self.T, +h, +h)
        #Mmm,_,_=get_sheared_moments(self.M1, self.M2, self.T, -h, -h)

        #return (Mpp - Mp0 - M0p + 2*self.M1 - Mm0 - M0m + Mmm)*self.hsqinv*0.5

        # zero shear always 2*M2?
        return 2*self.M2

    def d2M1ds2ds2z(self):
        """
        2nd derivative of M1 with respect to shear2 and shear2 at zero shear
        """

        #h=self.h
        #M0p,_,_=get_sheared_moments(self.M1, self.M2, self.T, 0.0, +h)
        #M0m,_,_=get_sheared_moments(self.M1, self.M2, self.T, 0.0, -h)

        #return (M0p - 2*self.M1 + M0m)*self.hsqinv

        # at zero shear
        return self.zero


    def dM2ds1z(self):
        """
        derivative of M2 with respect to shear1 at zero shear
        """
        #h=self.h
        #_,Mp0,_=get_sheared_moments(self.M1, self.M2, self.T, +h, 0.0)
        #_,Mm0,_=get_sheared_moments(self.M1, self.M2, self.T, -h, 0.0)

        #return (Mp0-Mm0)*self.h2inv

        # zero shear
        return self.zero

    def dM2ds2z(self):
        """
        derivative of M2 with respect to shear2 at zero shear
        """

        #h=self.h
        #_,M0p,_=get_sheared_moments(self.M1, self.M2, self.T, 0.0, +h)
        #_,M0m,_=get_sheared_moments(self.M1, self.M2, self.T, 0.0, -h)

        #return (M0p-M0m)*self.h2inv
        # zero shear 2*T
        return 2*self.T

    def d2M2ds1ds1z(self):
        """
        2nd derivative of M2 with respect to shear1 and shear1 at zero shear
        """

        #h=self.h
        #_,Mp0,_=get_sheared_moments(self.M1, self.M2, self.T, +h, 0.0)
        #_,Mm0,_=get_sheared_moments(self.M1, self.M2, self.T, -h, 0.0)

        # for zero shear, always nearly zero?
        #return (Mp0 - 2*self.M2 + Mm0)*self.hsqinv

        # at zero shear
        return self.zero

    def d2M2ds1ds2z(self):
        """
        2nd derivative of M2 with respect to shear1 and shear2 at zero shear
        """
        #h=self.h
        #_,Mp0,_=get_sheared_moments(self.M1, self.M2, self.T, +h, 0.0)
        #_,Mm0,_=get_sheared_moments(self.M1, self.M2, self.T, -h, 0.0)
        #_,M0p,_=get_sheared_moments(self.M1, self.M2, self.T, 0.0, +h)
        #_,M0m,_=get_sheared_moments(self.M1, self.M2, self.T, 0.0, -h)

        #_,Mpp,_=get_sheared_moments(self.M1, self.M2, self.T, +h, +h)
        #_,Mmm,_=get_sheared_moments(self.M1, self.M2, self.T, -h, -h)

        #return (Mpp - Mp0 - M0p + 2*self.M2 - Mm0 - M0m + Mmm)*self.hsqinv*0.5

        # zero shear always 2*M1?
        return 2*self.M1


    def d2M2ds2ds2z(self):
        """
        2nd derivative of M2 with respect to shear2 and shear2 at zero shear
        """
        #h=self.h
        #_,M0p,_=get_sheared_moments(self.M1, self.M2, self.T, 0.0, +h)
        #_,M0m,_=get_sheared_moments(self.M1, self.M2, self.T, 0.0, -h)

        #return (M0p - 2*self.M2 + M0m)*self.hsqinv

        # zero shear 4*M2
        return 4*self.M2

    def set_moms(self, M1, M2, T):
        """
        set the moments and derived shape parameters

        bounds checking is done here
        """

        self.M1=M1
        self.M2=M2
        self.T=T

        # T > 0 checked here
        e1,e2 = moms2e1e2(M1, M2, T)

        # shape boundaries checked here
        g1,g2 = shape.e1e2_to_g1g2(e1,e2)

        self.zero = numpy.abs(g1)*0

class OldDeriv(object):
    """
    class to calculate derivatives of moments with respect to shear
    """
    def __init__(self, M1, M2, T):
        self.set_moms(M1, M2, T)

    #
    # derivatives at shear==0
    #

    def dTds1z(self):
        """
        derivative of T with respect to shear1 at zero shear

        This is 2*M1 I realized
        """

        g1=self.g1
        g2sq=self.g2sq
        g1cub=self.g1cub
        Tround = self.Tround
        omgsq_inv2 = self.omgsq_inv2

        #val=-((4*(g1**3 + g1*(-1 + g2**2))*Tround)/(-1 + g1**2 + g2**2)**2)
        val=-4*Tround*(g1cub + g1*(-1 + g2sq))*omgsq_inv2

        return val

    def dTds2z(self):
        """
        derivative of T with respect to shear2 at zero shear
        """

        g2=self.g2
        g1sq=self.g1sq
        g2cub=self.g2cub
        Tround = self.Tround
        omgsq_inv2 = self.omgsq_inv2

        #val=-((4*((-1 + g1**2)*g2 + g2**3)*Tround)/(-1 + g1**2 + g2**2)**2)
        val=-4*Tround*(g2cub + g2*(-1 + g1sq))*omgsq_inv2

        return val

    def d2Tds1ds1z(self):
        """
        2nd derivative of T with respect to shear1 and shear1 at zero shear
        """

        g1sq=self.g1sq
        g2sq=self.g2sq
        g1q=self.g1q
        g2q=self.g2q
        omgsq_inv2 = self.omgsq_inv2
        Tround = self.Tround

        #val= (8*(-g1**2 + g1**4 - g2**2*(1 + g2**2))*Tround)/(-1 + g1**2 + g2**2)**2
        #val= 8*Tround*(-g1sq + g1q - g2sq*(1 + g2sq))*omgsq_inv2
        val= 8*Tround*(-g1sq + g1q - g2sq - g2q)*omgsq_inv2

        return val

    def d2Tds1ds2z(self):
        """
        2nd derivative of T with respect to shear1 and shear2 at zero shear
        """

        g1=self.g1
        g2=self.g2
        g1sq=self.g1sq
        g2sq=self.g2sq
        omgsq_inv2 = self.omgsq_inv2
        Tround = self.Tround

        #val=(16*g1*g2*(g1**2 + g2**2)*Tround)/(-1 + g1**2 + g2**2)**2

        val= 16*Tround*g1*g2*(g1sq + g2sq)*omgsq_inv2

        return val

    def d2Tds2ds2z(self):
        """
        2nd derivative of T with respect to shear2 and shear2 at zero shear
        """

        g1sq=self.g1sq
        g2sq=self.g2sq
        g1q=self.g1q
        g2q=self.g2q
        omgsq_inv2 = self.omgsq_inv2
        Tround = self.Tround

        #val=-((8*(g1**2 + g1**4 + g2**2 - g2**4)*Tround)/(-1 + g1**2 + g2**2)**2)
        val=-8*Tround*(g1sq + g1q + g2sq - g2q)*omgsq_inv2

        return val


    def dM1ds1z(self):
        """
        derivative of M1 with respect to shear1 at zero shear
        """

        #val = -((2*(1 + g1**2 + g2**2)*Tround)/(-1 + g1**2 + g2**2))
        return 2*self.T

    def dM1ds2z(self):
        """
        derivative of M1 with respect to shear2 at zero shear
        """
        return self.zero

    def d2M1ds1ds1z(self):
        """
        2nd derivative of M1 with respect to shear1 and shear1 at zero shear
        """
        g1=self.g1

        g1sq=self.g1sq
        g2sq=self.g2sq
        g1q=self.g1q
        g2q=self.g2q
        omgsq_inv2 = self.omgsq_inv2
        Tround = self.Tround
 
        #val = (4*g1*(-1 + g1**4 - 4*g2**2 + 2*g1**2*g2**2 + g2**4)*Tround)/(-1 + g1**2 + g2**2)**2
        val = 4*g1*Tround*(-1 + g1q - 4*g2sq + 2*g1sq*g2sq + g2q)*omgsq_inv2
        return val

    def d2M1ds1ds2z(self):
        """
        2nd derivative of M1 with respect to shear1 and shear2 at zero shear
        """
        g2=self.g2

        g1sq=self.g1sq
        g2sq=self.g2sq
        g1q=self.g1q

        Tround = self.Tround

        omgsq_inv2 = self.omgsq_inv2

        #val=(4*g2*(g1**4 + g2**2 * (-1 + g2**2) + g1**2 * (3 + 2*g2**2))*Tround)/(-1 + g1**2 + g2**2)**2 
        val=4*g2*Tround*(g1q + g2sq * (-1 + g2sq) + g1sq * (3 + 2*g2sq))*omgsq_inv2
        return val

    def d2M1ds2ds2z(self):
        """
        2nd derivative of M1 with respect to shear2 and shear2 at zero shear
        """
        g1=self.g1

        g1sq=self.g1sq
        g2sq=self.g2sq
        g1q=self.g1q

        omgsq_inv2 = self.omgsq_inv2
        Tround = self.Tround
 
        #val=-((4*g1*(g1**4 + (-1 + g2**2)**2 + 2*g1**2 *(1 + g2**2))*Tround)/(-1 + g1**2 + g2**2)**2)
        val=-4*g1*Tround*(g1q + (-1 + g2sq)**2 + 2*g1sq *(1 + g2sq))*omgsq_inv2
        return val



    def dM2ds1z(self):
        """
        derivative of M2 with respect to shear1 at zero shear
        """
        return self.zero

    def dM2ds2z(self):
        """
        derivative of M2 with respect to shear2 at zero shear
        """

        # -((2 (1 + g1^2 + g2^2) Tround)/(-1 + g1^2 + g2^2))
        return 2*self.T

    def d2M2ds1ds1z(self):
        """
        2nd derivative of M2 with respect to shear1 and shear1 at zero shear
        """
        g2=self.g2

        g1sq=self.g1sq
        g2sq=self.g2sq
        g1q=self.g1q
        omgsq_inv2 = self.omgsq_inv2
        Tround = self.Tround
 
        #val= -((4*g2*(g1**4 + 2*g1**2 * (-1 + g2**2) + (1 + g2**2)**2) * Tround)/(-1 + g1**2 + g2**2)**2)
        val= -4*Tround*g2*((g1q + 2*g1sq * (-1 + g2sq) + (1 + g2sq)**2) )*omgsq_inv2
        return val

    def d2M2ds1ds2z(self):
        """
        2nd derivative of M2 with respect to shear1 and shear2 at zero shear
        """
        g1=self.g1

        g1sq=self.g1sq
        g2sq=self.g2sq
        g1q=self.g1q
 
        omgsq_inv2 = self.omgsq_inv2
        Tround = self.Tround

        #val=(4*g1*(g1**4 + g2**2 * (3 + g2**2) + g1**2 * (-1 + 2 * g2**2)) * Tround)/(-1 + g1**2 + g2**2)**2
        val=4*g1*Tround*(g1q + g2sq * (3 + g2sq) + g1sq * (-1 + 2 * g2sq))*omgsq_inv2
        return val

    def d2M2ds2ds2z(self):
        """
        2nd derivative of M2 with respect to shear2 and shear2 at zero shear
        """
        g2=self.g2

        g1sq=self.g1sq
        g2sq=self.g2sq
        g1q=self.g1q
        g2q=self.g2q
 

        omgsq_inv2 = self.omgsq_inv2
        Tround = self.Tround
 
        #val=(4*g2*(-1 + g1**4 + g2**4 + 2*g1**2 * (-2 + g2**2)) * Tround)/(-1 + g1**2 + g2**2)**2
        val=4*g2*Tround*(-1 + g1q + g2q + 2*g1sq * (-2 + g2sq))*omgsq_inv2
        return val



    def set_moms(self, M1, M2, T):
        """
        set the moments and derived shape parameters

        bounds checking is done here
        """

        self.M1=M1
        self.M2=M2
        self.T=T

        # T > 0 checked here
        e1,e2 = moms2e1e2(M1, M2, T)

        # shape boundaries checked here
        g1,g2 = shape.e1e2_to_g1g2(e1,e2)

        self.e1=e1
        self.e2=e2
        self.g1=g1
        self.g2=g2

        self.g1sq=self.g1**2
        self.g2sq=self.g2**2
        self.g1cub=self.g1**3
        self.g2cub=self.g2**3
        self.g1q=self.g1**4
        self.g2q=self.g2**4

        self.gsq = g1**2 + g2**2
        self.omgsq = 1.0-self.gsq
        self.omgsq_inv = 1.0/self.omgsq
        self.omgsq_inv2 = self.omgsq_inv**2

        self.Tround = get_Tround(T, g1, g2)

        self.zero = numpy.abs(g1)*0



class PQRMomTemplatesBase(object):
    """
    calculate pqr from the input moments and a
    likelihood function using the templates as
    the priors

    random centers will be randomly placed in 
    a radius 
    """
    def __init__(self,
                 templates,
                 cen_dist, # pdf for cen prior
                 nrand_cen,
                 nsigma=5.0,
                 nmin=100,
                 neff_max=100.0):

        self.seed=numpy.random.randint(0,1000000)

        self.nsigma=nsigma
        self.templates_orig=templates
        self.cen_dist=cen_dist
        self.nrand_cen=nrand_cen

        self.nmin=nmin
        self.neff_max=neff_max

        self._set_templates()
        self._set_deriv()

    def _set_templates(self):
        """
        set the templates, trimming to the good ones
        """
        from .priors import ZDisk2D
        templates_orig = self.templates_orig


        M1 = templates_orig[:,2]
        M2 = templates_orig[:,3]
        T  = templates_orig[:,4]

        w,=numpy.where(T > 0)

        print("using %d/%d with T > 0" % (w.size,T.size))
        if w.size == 0:
            raise ValueError("none with T > 0")
        
        Tinv = 1.0/T[w]
        e1 = M1[w]*Tinv
        e2 = M2[w]*Tinv
        e=numpy.sqrt(e1**2 + e2**2)
        w2,=numpy.where(e < 1.0)
        
        print("using %d/%d with e < 1" % (w2.size,T.size))
        if w2.size == 0:
            raise ValueError("none with e < 1")

        w=w[w2]

        templates_keep = templates_orig[w,:]

        nkeep,ndim = templates_keep.shape

        nrand_cen=self.nrand_cen
        ntot = nkeep*nrand_cen

        print("replicating translated centers")
        print("total templates:",ntot)

        # note this will guarantee correct byte ordering for C code
        templates = numpy.zeros( (ntot, ndim) )

        cen_dist = self.cen_dist
        for irand in xrange(nrand_cen):
            beg=irand*nkeep
            end=(irand+1)*nkeep

            # replicate the moments
            templates[beg:end,:] = templates_keep[:,:]

            # randomized centers
            cen1,cen2 = cen_dist.sample2d(nkeep)
            templates[beg:end,0] = cen1
            templates[beg:end,1] = cen2

        print("shuffling")
        numpy.random.shuffle(templates)

        self.templates = templates

    def _set_deriv(self):

        templates = self.templates
        M1 = templates[:,2]
        M2 = templates[:,3]
        T  = templates[:,4]

        print("creating Deriv object")

        deriv = Deriv(M1, M2, T)


        print("copying shear derivatives")

        nt=T.size
        # correct byte ordering for C code
        Qderiv = numpy.zeros( (nt, 3, 2) )
        Rderiv = numpy.zeros( (nt, 3, 2, 2) )

        # first derivatives shear1
        Qderiv[:,0,0] = deriv.dM1ds1z()
        Qderiv[:,1,0] = deriv.dM2ds1z()
        Qderiv[:,2,0] = deriv.dTds1z()

        # first derivatives shear2
        Qderiv[:,0,1] = deriv.dM1ds2z()
        Qderiv[:,1,1] = deriv.dM2ds2z()
        Qderiv[:,2,1] = deriv.dTds2z()

        # 2nd deriv 11
        Rderiv[:,0,0,0] = deriv.d2M1ds1ds1z()
        Rderiv[:,1,0,0] = deriv.d2M2ds1ds1z()
        Rderiv[:,2,0,0] = deriv.d2Tds1ds1z()

        # 2nd deriv 12
        Rderiv[:,0,0,1] = deriv.d2M1ds1ds2z()
        Rderiv[:,1,0,1] = deriv.d2M2ds1ds2z()
        Rderiv[:,2,0,1] = deriv.d2Tds1ds2z()

        # 2nd deriv 21
        Rderiv[:,0,1,0] = Rderiv[:,0,0,1]
        Rderiv[:,1,1,0] = Rderiv[:,1,0,1]
        Rderiv[:,2,1,0] = Rderiv[:,2,0,1]

        # 2nd deriv 22
        Rderiv[:,0,1,1] = deriv.d2M1ds2ds2z()
        Rderiv[:,1,1,1] = deriv.d2M2ds2ds2z()
        Rderiv[:,2,1,1] = deriv.d2Tds2ds2z()

        self.Qderiv = Qderiv
        self.Rderiv = Rderiv

        del deriv

class PQRMomTemplatesGauss(PQRMomTemplatesBase):
    """
    calculate pqr from the input moments and a
    likelihood function using the templates as
    the priors

    Assumes multi-variate gaussian for the likelihoods
    """

    def get_result(self):
        """
        get the result dict.  You need to run calc_pqr first
        """
        return self._result

    def calc_pqr(self, mom, mom_cov):
        """
        calculate pqr sums assuming multivariate gaussian likelihood,
        equation 36 B&A 2014
        """
        from ._gmix import mvn_calc_pqr_templates
        from numpy import sqrt

        self._set_likelihood(mom,mom_cov)
        dist=self.dist

        P = numpy.zeros(1)
        Q = numpy.zeros(2)
        R = numpy.zeros((2,2))

        ierrors = 1.0/numpy.sqrt( numpy.diag(mom_cov) )

        #nmin = self.nmin*self.nrand_cen
        #neff_max = self.neff_max*self.nrand_cen
        nmin = self.nmin
        neff_max = self.neff_max

        nuse,neff=mvn_calc_pqr_templates(dist.mean,
                                         dist.icov,
                                         ierrors,
                                         dist.norm,
                                         self.nsigma,
                                         nmin,
                                         neff_max,
                                         self.templates,
                                         self.Qderiv,
                                         self.Rderiv,
                                         self.seed,
                                         P,Q,R)

        # only seed the first call
        self.seed=-1
        #neff /= self.nrand_cen
        #nuse /= self.nrand_cen

        P=P[0]

        self._result={'mom':mom,
                      'mom_cov':mom_cov,
                      'P':P,
                      'Q':Q,
                      'R':R,
                      'nuse':nuse,
                      'neff':neff}

    def get_slow_result(self):
        """
        get the result dict.  You need to run calc_pqr first
        """
        return self._slow_result


    def calc_pqr_slow(self, mom, mom_cov):
        """
        calculate pqr sums assuming multivariate gaussian likelihood,
        equation 36 B&A 2014
        """
        from numpy import dot

        self._set_likelihood(mom,mom_cov)

        dist=self.dist
        likes = dist.get_prob(self.templates, nsigma=self.nsigma)

        Pmax = likes.max()

        P = likes.sum()
        Q = numpy.zeros(2)
        R = numpy.zeros( (2,2) )

        neff=P/Pmax

        Qderiv=self.Qderiv
        Rderiv=self.Rderiv

        templates=self.templates
        n=templates.shape[0]

        mean=dist.mean[2:2+3]
        icov=dist.icov[2:2+3, 2:2+3]

        nuse=0
        for i in xrange(n):

            like = likes[i]
            if like > 0:
                nuse += 1
                datamu=templates[i,2:2+3]
                Qd=Qderiv[i,:,:]
                Rd=Rderiv[i,:,:,:]

                xdiff = mean-datamu

                icov_dot_Qd_1 = dot(icov, Qd[:,0])
                icov_dot_Qd_2 = dot(icov, Qd[:,1])

                Qsum1 = dot(xdiff, icov_dot_Qd_1)
                Qsum2 = dot(xdiff, icov_dot_Qd_2)

                Q[0] += Qsum1*like
                Q[1] += Qsum2*like

                icov_dot_Rd_11 = dot(icov, Rd[:,0,0])
                icov_dot_Rd_12 = dot(icov, Rd[:,0,1])
                icov_Rd_dot21 = icov_dot_Rd_12
                icov_dot_Rd_22 = dot(icov, Rd[:,1,1])

                R11sum_1 = dot(xdiff, icov_dot_Rd_11)
                R12sum_1 = dot(xdiff, icov_dot_Rd_12)
                R21sum_1 = R12sum_1
                R22sum_1 = dot(xdiff, icov_dot_Rd_22)

                # this is the term one gets if the templates are
                # integrated over all space
                R11sum_2 = dot(Qd[:,0], icov_dot_Qd_1)
                R12sum_2 = dot(Qd[:,0], icov_dot_Qd_2)
                R21sum_2 = R12sum_2
                R22sum_2 = dot(Qd[:,1], icov_dot_Qd_2)

                R[0,0] += (R11sum_1 + R11sum_2)*like
                R[0,1] += (R12sum_1 + R12sum_2)*like
                R[1,0] += (R21sum_1 + R21sum_2)*like
                R[1,1] += (R22sum_1 + R22sum_2)*like


        self._slow_result={'mom':mom,
                           'mom_cov':mom_cov,
                           'P':P,
                           'Q':Q,
                           'R':R,
                           'nuse':nuse,
                           'neff':neff}

    def _set_likelihood(self, mom, mom_cov):
        """
        set the likelihood based on the input moment means
        and covariance
        """
        from .priors import MultivariateNormal

        self.dist = MultivariateNormal(mom, mom_cov)

        self.mom=mom
        self.mom_cov=mom_cov

        # M1,M2,T subset
        self.mom_sub = mom[2:2+3]
        self.mom_cov_sub=mom_cov[2:2+3, 2:2+3]


 
def moms2e1e2(M1, M2, T):
    """
    convert M1, M2, T to e1,e2

    parameters
    -----------
    M1,M2,T: float or array
        M1 is <x^2 - y^2>
        M2 is <2*x*y>
        T  is <x^2 + y^2>

    returns
    -------
    e1,e2:
        M1/T M2/T
    """
    if isinstance(T, numpy.ndarray):
        w,=numpy.where(T <= 0.0)
        if w.size > 0:
            raise GMixRangeError("%d T were <= 0.0" % w.size)
    else:
        if T <= 0.0:
            raise GMixRangeError("T <= 0.0: %g" % g)

    Tinv=1.0/T
    e1 = M1*Tinv
    e2 = M2*Tinv

    return e1,e2

def get_Tround(T, g1, g2):
    """
    get the round T

    parameters
    ----------
    T: float
        <x^2> + <y^2>
    g1,g2: float
        The reduced shear style shape
    """
    gsq = g1**2 + g2**2
    return T*(1-gsq)/(1+gsq)

def get_T(Tround, g1, g2):
    """
    get elliptical T for given Tround and g1,g2

    parameters
    ----------
    Tround: float
        <x^2> + <y^2>
    g1,g2: float
        The reduced shear style shape
    """
    gsq = g1**2 + g2**2
    return Tround*(1+gsq)/(1-gsq)

def get_sheared_moments(M1, M2, T, s1, s2):
    """
    Get sheared moments

    parameters
    ----------
    M1: float or array
        <x^2 - y^2>
    M2: float or array
        <2*xy>
    T:  float or array
        <x^2 + y^2>

    returns
    -------
    sheared M1, M2, T
    """

    e1,e2 = moms2e1e2(M1, M2, T)
    g1,g2 = shape.e1e2_to_g1g2(e1,e2)

    g1s, g2s = shape.shear_reduced(g1, g2, s1, s2)

    Tround = get_Tround(T, g1, g2)

    Ts = get_T(Tround, g1s, g2s)

    e1s,e2s = shape.g1g2_to_e1e2(g1s, g2s)

    M1s = Ts*e1s
    M2s = Ts*e2s

    return M1s, M2s, Ts

def test_mom():

    from ngmix import print_pars
    from numpy import linspace, array

    e1=array([-0.2, -0.2, 0.2, 0.2,  0.1,  0.0, 0.1, 0.0])
    e2=array([ 0.1,  0.0, 0.1, 0.0, -0.2, -0.2, 0.2, 0.2])
    T=array([16.0]*e1.size)

    M1 = T*e1
    M2 = T*e2

    md=Deriv(M1, M2, T)

    print_pars(M1, front="M1:         ")
    print_pars(M2, front="M2:         ")
    print_pars(T,  front="T:          ")
    print_pars(e1, front="e1:         ")
    print_pars(e2, front="e2:         ")

    print()
    print_pars(md.dTds1z(),      front="dTds1z:     ")
    print_pars(md.dTds2z(),      front="dTds2z:     ")
    print_pars(md.dM1ds1z(),     front="dM1ds1z:    ")
    print_pars(md.dM1ds2z(),     front="dM1ds2z:    ")
    print_pars(md.dM2ds1z(),     front="dM2ds1z:    ")
    print_pars(md.dM2ds2z(),     front="dM2ds2z:    ")

    print()
    print_pars(md.d2Tds1ds1z(),  front="d2Tds1ds1z: ")
    print_pars(md.d2Tds1ds2z(),  front="d2Tds1ds2z: ")
    print_pars(md.d2Tds2ds2z(),  front="d2Tds2ds2z: ")
    print()
    print_pars(md.d2M1ds1ds1z(), front="d2M1ds1ds1z:")
    print_pars(md.d2M1ds1ds2z(), front="d2M1ds1ds2z:")
    print_pars(md.d2M1ds2ds2z(), front="d2M1ds2ds2z:")
    print()
    print_pars(md.d2M2ds1ds1z(), front="d2M2ds1ds1z:")
    print_pars(md.d2M2ds1ds2z(), front="d2M2ds1ds2z:")
    print_pars(md.d2M2ds2ds2z(), front="d2M2ds2ds2z:")

def test_pqr_moments(ntemplate=10000, seed=None, cen_radius=2.0, nrand_cen=10):
    """

    note testing with neff_max=infinity in comparing to slow, so that we don't
    get hit by randomness as badly

    """
    from numpy import array, diag
    from .priors import MultivariateNormal, ZDisk2D
    import time

    numpy.random.seed(seed)

    mean=array([0.0, 0.0, 2.0, 1.5, 16.0, 100.0])
    cov = diag([0.1**2, 0.1**2,
                0.5**2, 0.5**2, 4.0**2,
                10.0**2])

    mvn = MultivariateNormal(mean, cov)

    templates = mvn.sample(ntemplate)

    cen_dist = ZDisk2D(cen_radius)
    pqrt = PQRMomTemplatesGauss(templates,
                                cen_dist,
                                nrand_cen,
                                neff_max=1.0e9)

    tm0=time.time()
    pqrt.calc_pqr(mean, cov)
    pqr_res = pqrt.get_result()
    tm=time.time()-tm0

    tm0=time.time()
    pqrt.calc_pqr_slow(mean, cov)
    slow_res = pqrt.get_slow_result()
    tmc=time.time()-tm0

    print("P,Pc")
    print(pqr_res['P'])
    print(slow_res['P'])
    print("Q,Qc")
    print(pqr_res['Q'])
    print(slow_res['Q'])
    print("R,Rc")
    print(pqr_res['R'])
    print(slow_res['R'])


    print("nuse,nusec")
    print(pqr_res['nuse'],slow_res['nuse'])
    print("neff,neffc:",pqr_res['neff'],slow_res['neff'])
    print("time: ",tm)
    print("timec:",tmc)
