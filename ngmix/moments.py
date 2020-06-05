import numpy
from .gexceptions import GMixRangeError
from . import shape

def sigma_to_fwhm(sigma):
    """
    convert sigma to fwhm for a gaussian
    """
    return sigma*2.3548200450309493

def T_to_fwhm(T):
    """
    convert sigma to fwhm for a gaussian
    """
    sigma=numpy.sqrt(T/2.0)
    return sigma_to_fwhm(sigma)

def fwhm_to_sigma(fwhm):
    """
    convert fwhm to sigma for a gaussian
    """
    return fwhm/2.3548200450309493

def fwhm_to_T(fwhm):
    """
    convert fwhm to T for a gaussian
    """
    sigma = fwhm_to_sigma(fwhm)
    return 2*sigma**2


def r50_to_sigma(r50):
    """
    convert r50, the half light radius, to sigma for a gaussian

    half light radius is the radius that contains half the total light.
    For a gaussian this is fwhm/2
    """
    fwhm = 2.0*r50
    return fwhm_to_sigma(fwhm)

def sigma_to_r50(sigma):
    """
    convert sigma to r50 for a gaussian
    """
    fwhm = sigma_to_fwhm(sigma)
    r50 = fwhm/2.0
    return r50

def r50_to_T(r50):
    """
    convert r50, the half light radius, to T=2*sigma**2 for a gaussian

    half light radius is the radius that contains half the total light.
    For a gaussian this is fwhm/2
    """

    sigma = r50_to_sigma(r50)
    T = 2*sigma**2
    return T

def T_to_r50(T):
    """
    convert T=2*sigma**2 to r50 for a gaussian
    """
    sigma = numpy.sqrt(T/2.0)
    return sigma_to_r50(sigma)

def moms_to_e1e2(M1, M2, T):
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
            raise GMixRangeError("T <= 0.0: %g" % T)

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

def get_sheared_M1M2T(M1, M2, T, s1, s2):
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

    e1,e2 = moms_to_e1e2(M1, M2, T)
    g1,g2 = shape.e1e2_to_g1g2(e1,e2)

    g1s, g2s = shape.shear_reduced(g1, g2, s1, s2)

    Tround = get_Tround(T, g1, g2)

    Ts = get_T(Tround, g1s, g2s)

    e1s,e2s = shape.g1g2_to_e1e2(g1s, g2s)

    M1s = Ts*e1s
    M2s = Ts*e2s

    return M1s, M2s, Ts

def get_sheared_g1g2T(g1,g2,T, s1, s2):

    g1s,g2s = shape.shear_reduced(g1,g2,s1,s2)

    Tround = get_Tround(T, g1, g2)
    Ts = get_T(Tround, g1s, g2s)

    return g1s, g2s, Ts

def get_sheared_moments(irr, irc, icc, s1, s2):
    g1,g2,T=mom2g(irr, irc, icc)
    g1s,g2s,Ts = get_sheared_g1g2T(g1,g2,T,s1,s2)
    irr_s, irc_s, icc_s = g2mom(g1s, g2s, Ts)
    return irr_s, irc_s, icc_s


def mom2e(Irr, Irc, Icc):
    T = Irr+Icc
    e1=(Icc-Irr)/T
    e2=2.0*Irc/T

    return e1,e2,T

def mom2g(Irr, Irc, Icc):
    e1,e2,T = mom2e(Irr, Irc, Icc)
    g1,g2=shape.e1e2_to_g1g2(e1,e2)

    return g1,g2,T

def e2mom(e1,e2,T):

    Irc = e2*T/2.0
    Icc = (1+e1)*T/2.0
    Irr = (1-e1)*T/2.0

    return Irr, Irc, Icc

def g2mom(g1, g2, T):

    e1,e2=shape.g1g2_to_e1e2(g1,g2)
    Irc = e2*T/2.0
    Icc = (1+e1)*T/2.0
    Irr = (1-e1)*T/2.0

    return Irr, Irc, Icc
