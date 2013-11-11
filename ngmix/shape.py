import numpy
import numba
from numba import float64, int64, autojit, jit

from .gexceptions import GMixRangeError, GMixFatalError

class Shape(object):
    """
    Shape object
    """
    def __init__(self):
        self.g1 = numpy.float64(0.0)
        self.g2 = numpy.float64(0.0)
        self.e1 = numpy.float64(0.0)
        self.e1 = numpy.float64(0.0)
        self.eta1 = numpy.float64(0.0)
        self.eta1 = numpy.float64(0.0)

    def set_e1e2(self, e1, e2):
        """
        Set e1,e2
        """
        e1=numpy.float64(e1)
        e2=numpy.float64(e2)
        etot = numpy.sqrt(e1**2 + e2**2)

        if etot >= 1:
            mess="e values must be < 1, found %.16g (%.16g,%.16g)"
            mess = mess % (etot,e1,e2)
            raise GMixRangeError(mess)

        if etot==0:
            self.eta1=numpy.float64(0.0)
            self.eta2=numpy.float64(0.0)
            self.g1=numpy.float64(0.0)
            self.g2=numpy.float64(0.0)
            self.e1=numpy.float64(0.0)
            self.e2=numpy.float64(0.0)
            return 

        eta = numpy.atanh(etot)
        gtot = numpy.tanh(eta/2)

        cos2theta = e1/etot
        sin2theta = e2/etot

        self.e1   = e1
        self.e2   = e2
        self.g1   = gtot*cos2theta
        self.g2   = gtot*sin2theta
        self.eta1 = eta*cos2theta
        self.eta2 = eta*sin2theta

    def set_g1g2(self, g1, g2):
        g1=numpy.float64(g1)
        g2=numpy.float64(g2)

        gtot = sqrt(g1**2 + g2**2)
        if gtot >= 1:
            mess="g values must be < 1, found %.16g (%.16g,%.16g)"
            mess = mess % (gtot,g1,g2)
            raise GMixRangeError(mess)

        if gtot==0:
            self.eta1=numpy.float64(0.0)
            self.eta2=numpy.float64(0.0)
            self.g1=numpy.float64(0.0)
            self.g2=numpy.float64(0.0)
            self.e1=numpy.float64(0.0)
            self.e2=numpy.float64(0.0)
            return 

        eta = 2*atanh(gtot)
        etot = tanh(eta)

        cos2theta = g1/gtot
        sin2theta = g2/gtot

        self.g1   = g1
        self.g2   = g2
        self.e1   = etot*cos2theta
        self.e2   = etot*sin2theta
        self.eta1 = eta*cos2theta
        self.eta2 = eta*sin2theta

    def set_eta1eta2(self, eta1, eta2):
        eta1 = numpy.float64(eta1)
        eta2 = numpy.float64(eta2)

        eta=sqrt(eta1**2 + eta2**2)

        if eta==0.:
            self.eta1=numpy.float64(0.0)
            self.eta2=numpy.float64(0.0)
            self.g1=numpy.float64(0.0)
            self.g2=numpy.float64(0.0)
            self.e1=numpy.float64(0.0)
            self.e2=numpy.float64(0.0)
            return

        etot=tanh(eta)
        gtot=tanh(eta/2.)

        if etot >= 1.0:
            mess="e values must be < 1, found %.16g" % etot
            raise GMixRangeError(mess)
        if gtot >= 1.0:
            mess="g values must be < 1, found %.16g" % gtot
            raise GMixRangeError(mess)

        cos2theta = eta1/eta
        sin2theta = eta2/eta

        e1=etot*cos2theta
        e2=etot*sin2theta

        g1=gtot*cos2theta
        g2=gtot*sin2theta

        self.eta1,self.eta2=eta1,eta2
        self.e1,self.e2=e1,e2
        self.g1,self.g2=g1,g2

    def copy(self):
        """
        Make a new Shape object with the same ellipticity parameters
        """
        s = Shape()
        s.set_g1g2(self.g1,self.g2)
        return s

    def shear(self, s):
        """
        shear the shape.
        """
        if not isinstance(s,Shape):
            raise ValueError("Only Shape objects can be added")

        g1,g2 = shear_reduced(self.g1,self.g2,s.g1,s.g2)
        self.set_g1g2(g1, g2)

    def __repr__(self):
        return '(%.16g, %.16g)' % (self.g1,self.g2)

@jit(argtypes=[ float64, float64 ])
def g1g2_to_e1e2(g1, g2):
    g=numpy.sqrt(g1*g1 + g2*g2)

    if g >= 1.:
        raise GMixRangeError("g out of bounds: %s" % g)

    e = numpy.tanh(2*numpy.arctanh(g))
    if e >= 1.:
        # round off?
        e = 0.99999999

    fac = e/g

    e1 = fac*g1
    e2 = fac*g2
    
    return e1,e2

@jit(argtypes=[float64,float64,float64,float64])
def shear_reduced(g1, g2, s1, s2):
    """
    addition formula for reduced shear
    """

    A = 1 + g1*s1 + g2*s2
    B = g2*s1 - g1*s2
    denom_inv = 1./(A*A + B*B)

    g1o = A*(g1 + s1) + B*(g2 + s2)
    g2o = A*(g2 + s2) - B*(g1 + s1)

    g1o *= denom_inv
    g2o *= denom_inv

    return g1o,g2o

