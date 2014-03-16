import numpy
import numba
from numba import float64, int64, void, autojit, jit

from .gexceptions import GMixRangeError, GMixFatalError

#@jit(argtypes=[float64,float64,float64,float64])
@autojit
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

class ShapeBase(object):
    """
    This base class provides non-jitted methods.
    """
    def copy(self):
        """
        Make a new Shape object with the same ellipticity parameters
        """
        s = Shape(self.g1, self.g2)
        return s

    def __repr__(self):
        return '(%.16g, %.16g)' % (self.g1,self.g2)

@jit
class Shape(ShapeBase):
    """
    Shape object.  Currently only for reduced shear style shapes

    This version is jitted, but inherits non-jitted methods
    from ShapeBase
    """
    @void(float64, float64)
    def __init__(self, g1, g2):
        self.g1 = g1
        self.g2 = g2

        # can't call the other jitted methods
        g=numpy.sqrt(g1*g1 + g2*g2)
        if g >= 1.0:
            raise GMixRangeError("g out of range: %.16g" % g)

    @void(float64,float64)
    def set_g1g2(self, g1, g2):
        """
        Set reduced shear style ellipticity
        """
        self.g1=g1
        self.g2=g2

        g=numpy.sqrt(g1*g1 + g2*g2)
        if g >= 1.0:
            raise GMixRangeError("g out of range: %.16g" % g)

    @void(float64, float64)
    def shear(self, s1, s2):
        """
        shear the shape.
        """
        g1,g2 = shear_reduced(self.g1,self.g2, s1, s2)
        self.set_g1g2(g1, g2)

    @void(float64)
    def rotate(self, theta_radians):
        """
        Rotate the shape by the input angle
        """
        twotheta = 2.0*theta_radians

        cos2angle = numpy.cos(twotheta)
        sin2angle = numpy.sin(twotheta)
        g1rot =  self.g1*cos2angle + self.g2*sin2angle
        g2rot = -self.g1*sin2angle + self.g2*cos2angle

        self.set_g1g2(g1rot, g2rot)


@jit(argtypes=[ float64, float64 ])
def g1g2_to_e1e2(g1, g2):
    """
    convert reduced shear g1,g2 to standard ellipticity
    parameters e1,e2
    """
    g=numpy.sqrt(g1*g1 + g2*g2)

    if g >= 1.:
        raise GMixRangeError("g out of bounds: %s" % g)
    if g == 0.0:
        return (0.0, 0.0)

    eta = 2*numpy.arctanh(g)
    e = numpy.tanh(eta)
    if e >= 1.:
        # round off?
        e = 0.99999999

    fac = e/g

    e1 = fac*g1
    e2 = fac*g2
    
    return e1,e2

@jit(argtypes=[float64,float64])
def e1e2_to_g1g2(e1, e2):

    e = numpy.sqrt(e1*e1 + e2*e2)
    if e >= 1.:
        raise GMixRangeError("e out of bounds: %s" % e)
    if e == 0.0:
        return (0.0, 0.0)
    
    eta=numpy.arctanh(e)
    g = numpy.tanh(0.5*eta)
    if g >= 1.:
        # round off?
        g = 0.99999999

    fac = g/e

    g1 = fac*e1
    g2 = fac*e2

    return g1,g2


def g1g2_to_eta1eta2_array(g1, g2):
    """
    convert reduced shear g1,g2 to eta
    """
    n=g1.size

    g=numpy.sqrt(g1*g1 + g2*g2)

    eta1=numpy.zeros(n) -9999.0
    eta2=eta1.copy()
    good = numpy.zeros(n, dtype='i1')

    w,=numpy.where(g < 1.0)

    if w.size > 0:

        eta_w = 2*numpy.arctanh(g[w])

        fac = eta_w/g[w]

        eta1[w] = fac*g1[w]
        eta2[w] = fac*g2[w]

        good[w] = 1

    return eta1,eta2, good


@jit(argtypes=[float64,float64])
def g1g2_to_eta1eta2(g1, g2):
    """
    convert reduced shear g1,g2 to eta
    """
    g=numpy.sqrt(g1*g1 + g2*g2)

    if g >= 1.:
        raise GMixRangeError("g out of bounds: %s" % g)
    if g == 0.0:
        return (0.0, 0.0)

    eta = 2*numpy.arctanh(g)

    fac = eta/g

    eta1 = fac*g1
    eta2 = fac*g2
    
    return eta1,eta2

def eta1eta2_to_g1g2_array(eta1,eta2):
    """
    Perform the conversion for all elements in an array
    """
    n=eta1.size
    g1=numpy.zeros(n) - 9999
    g2=numpy.zeros(n) - 9999
    good = numpy.zeros(n, dtype='i1')

    eta=numpy.sqrt(eta1*eta1 + eta2*eta2)

    g = numpy.tanh(0.5*eta)

    w,=numpy.where( g < 1.0 )
    if w.size > 0:
        fac = g[w]/eta[w]

        g1 = fac*eta1[w]
        g2 = fac*eta2[w]
        good[w] = 1

    return g1,g2,good

@jit(argtypes=[float64,float64])
def eta1eta2_to_g1g2(eta1,eta2):
    """
    convert from eta to reduced style 
    """
    eta=numpy.sqrt(eta1*eta1 + eta2*eta2)

    g = numpy.tanh(0.5*eta)

    if g >= 1.:
        raise GMixRangeError("g out of bounds: %s" % g)
    if g == 0.0:
        return (0.0, 0.0)

    fac = g/eta

    g1 = fac*eta1
    g2 = fac*eta2

    return g1,g2



#@jit(argtypes=[float64,float64,float64,float64])
@autojit
def dgs_by_dgo_jacob(g1, g2, s1, s2):
    """
    jacobian of the transformation
        |dgs/dgo|_{-shear}

    parameters
    ----------
    g1,g2: numbers or arrays
        shape pars for "observed" image
    s1,s2: numbers or arrays
        shape pars for shear, applied negative
    """

    ssq = s1*s1 + s2*s2
    num = (ssq - 1)**2
    denom=(1 + 2*g1*s1 + 2*g2*s2 + g1**2*ssq + g2**2*ssq)**2

    jacob = num/denom
    return jacob


