from __future__ import print_function
import numpy
from ._gmix import GMixRangeError
from . import shape

class Deriv(object):
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
    """
    def __init__(self, templates, nsigma=5.0):
        self.nsigma=nsigma
        self.templates_orig=templates

        self._set_templates()
        self._set_deriv()

    def _set_templates(self):
        """
        set the templates, trimming to the good ones
        """

        templates = self.templates_orig

        M1 = templates[:,2]
        M2 = templates[:,3]
        T  = templates[:,4]

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

        self.deriv = Deriv(M1[w], M2[w], T[w])
        self.templates = templates[w,:]

    def _set_deriv(self):
        print("setting shear derivatives")
        deriv = self.deriv
        nt=self.templates.shape[0]

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


class PQRMomTemplatesGauss(PQRMomTemplatesBase):
    """
    calculate pqr from the input moments and a
    likelihood function using the templates as
    the priors

    Assumes multi-variate gaussian for the likelihoods
    """

    def calc_pqr(self, mom, mom_cov):
        """
        calculate pqr sums assuming multivariate gaussian likelihood,
        equation 36 B&A 2014
        """
        from ._gmix import mvn_calc_pqr_templates

        self._set_likelihood(mom,mom_cov)

        P = numpy.zeros(1)
        Q = numpy.zeros(2)
        R = numpy.zeros((2,2))

        dist=self.dist
        mvn_calc_pqr_templates(dist.mean,
                               dist.icov,
                               dist.norm,
                               self.nsigma,
                               self.templates,
                               self.Qderiv,
                               self.Rderiv,
                               P,Q,R)

        P=P[0]

        return P,Q,R

    def calc_pqr_slow(self, mom, mom_cov):
        """
        calculate pqr sums assuming multivariate gaussian likelihood,
        equation 36 B&A 2014
        """
        from numpy import dot

        self._set_likelihood(mom,mom_cov)

        dist=self.dist
        likes = dist.get_prob(self.templates, nsigma=self.nsigma)

        P = likes.sum()
        Q = numpy.zeros(2)
        R = numpy.zeros( (2,2) )

        Qderiv=self.Qderiv
        Rderiv=self.Rderiv

        templates=self.templates
        n=templates.shape[0]

        mean=dist.mean[2:2+3]
        icov=dist.icov[2:2+3, 2:2+3]

        for i in xrange(n):

            like = likes[i]
            if like > 0:
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

                R11sum_2 = dot(Qd[:,0], icov_dot_Qd_1)
                R12sum_2 = dot(Qd[:,0], icov_dot_Qd_2)
                R21sum_2 = R12sum_2
                R22sum_2 = dot(Qd[:,1], icov_dot_Qd_2)
                R11sum_2 = 0.0
                R12sum_2 = 0.0
                R21sum_2 = 0.0
                R22sum_2 = 0.0


                R[0,0] += (R11sum_1 + R11sum_2)*like
                R[0,1] += (R12sum_1 + R12sum_2)*like
                R[1,0] += (R21sum_1 + R21sum_2)*like
                R[1,1] += (R22sum_1 + R22sum_2)*like

        return P,Q,R


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

def test_pqr_moments(ntemplate=100):
    from numpy import array, diag
    from .priors import MultivariateNormal
    import time
    mean=array([0.0, 0.0, 2.0, 1.5, 16.0, 100.0])
    cov = diag([0.1, 0.1, 0.5, 0.4, 2.0, 10.0])

    mvn = MultivariateNormal(mean, cov)

    templates = mvn.sample(ntemplate)

    pqrt = PQRMomTemplatesGauss(templates)

    tm0=time.time()
    P,Q,R = pqrt.calc_pqr(mean, cov)
    tm=time.time()-tm0

    tm0=time.time()
    Pc,Qc,Rc = pqrt.calc_pqr_slow(mean, cov)
    tmc=time.time()-tm0

    print(P)
    print(Pc)
    print(Q)
    print(Qc)
    print(R)
    print(Rc)


    print("time: ",tm)
    print("timec:",tmc)
