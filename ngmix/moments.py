from __future__ import print_function
import numpy
from ._gmix import GMixRangeError
from . import shape

class Deriv(object):
    def __init__(self, M1, M2, T):
        self.set_moms(M1, M2, T)

    #
    # derivatives at shear==0
    #

    def dTds1z(self):
        """
        derivative of T with respect to shear1 at zero shear
        """

        g1=self.g1
        g2=self.g2
        Tround = self.Tround

        #val=-((4*(g1**3 + g1*(-1 + g2**2))*Tround)/(-1 + g1**2 + g2**2)**2)
        val=-4*Tround*(self.g1cub + g1*(-1 + self.g2sq))*self.omgsq_inv2

        return val

    def dTds2z(self):
        """
        derivative of T with respect to shear2 at zero shear
        """

        g1=self.g1
        g2=self.g2
        Tround = self.Tround

        #val=-((4*((-1 + g1**2)*g2 + g2**3)*Tround)/(-1 + g1**2 + g2**2)**2)
        val=-4*Tround*(self.g2cub + g2*(-1 + self.g1sq))*self.omgsq_inv2

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

def moms2e1e2(M1, M2, T):
    """
    convert M1, M2, T to e1,e2

    parameters
    -----------
    M1,M2,T: float or array
        M1 is <row^2> - <col^2>
        M2 is <2*row*col>
        T  is <row^2 + col^2>

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
