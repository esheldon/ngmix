from __future__ import print_function

import numpy
from numpy import where, log10, zeros, exp, sqrt
from numpy.random import random as randu

from . import shape
from .shape import g1g2_to_eta1eta2, eta1eta2_to_g1g2_array, g1g2_to_eta1eta2_array
from .gexceptions import GMixRangeError

from . import priors
from .priors import LOWVAL
from . import gmix
from .gmix import GMixND


class JointPriorBDF(GMixND):
    """
    Joint prior in g1,g2,T,Fb,Fd

    The prior is actually in eta1,eta2,logT,logFb,logFd as a sum of gaussians.
    When sampling the values are converted to the linear ones.  Also when
    getting the log probability, the input linear values are converted into log
    space.

    """
    def __init__(self,
                 weights,
                 means,
                 covars,
                 logT_bounds=[-1.3, 1.0],
                 logFlux_bounds=[-3.0, 2.0]):

        super(JointPriorBDF,self).__init__(weights, means, covars)

        self.logT_bounds=logT_bounds
        self.logFlux_bounds=logFlux_bounds
        self.eta_max = 28.0

        self._make_gmm()

    def get_prob_scalar(self, pars, throw=True):
        """
        Additional checks on bounds over GMixND.  Scalar input.
        """
        if not self.check_bounds_scalar(pars, throw=throw):
            return LOWVAL
        else:
            return super(JointPriorBDF,self).get_prob_scalar(pars)

    def get_lnprob_scalar(self, pars, throw=True):
        """
        Additional checks on bounds over GMixND.  Scalar input.
        """
        if not self.check_bounds_scalar(pars, throw=throw):
            return LOWVAL
        else:
            return super(JointPriorBDF,self).get_lnprob_scalar(pars)

    def get_prob_array(self, pars):
        """
        Additional checks on bounds over GMixND.  Array input.
        """
        p=zeros(pars.shape[0])
        w=self.check_bounds_array(pars)
        if w.size > 0:
            p[w]=super(JointPriorBDF,self).get_prob_array(pars[w,:])
        return p

    def get_lnprob_array(self, pars):
        """
        Additional checks on bounds over GMixND.  Array input.
        """
        lnp=zeros(pars.shape[0]) + LOWVAL
        w=self.check_bounds_array(pars)
        if w.size > 0:
            lnp[w]=super(JointPriorBDF,self).get_lnprob_array(pars[w,:])
        return lnp

    def sample(self, n=None):
        """
        Get samples in log space
        """

        if n is None:
            is_scalar=True
            n=1
        else:
            is_scalar=False

        samples=zeros( (n,self.ndim) )
        nleft=n
        ngood=0
        while nleft > 0:
            tsamples=self.gmm.sample(nleft)

            w=self.check_bounds_array(tsamples)

            if w.size > 0:
                first=ngood
                last=ngood+w.size

                samples[first:last,:] = tsamples[w,:]

                ngood += w.size
                nleft -= w.size

        if is_scalar:
            samples = samples[0,:]
        return samples


    def get_pqr_num(self, pars, s1=0.0, s2=0.0, h=1.e-6, **keys):
        """

        linear parameters input

        Evaluate 
            P
            Q
            R
        From Bernstein & Armstrong

        P is this prior times the jacobian at shear==0

        Q is the gradient of P*J evaluated at shear==0

            [ d(P*J)/ds1, d(P*J)/ds2]_{s=0}

        R is grad of grad of P*J at shear==0
            [ d(P*J)/dg1ds1  d(P*J)/dg1ds2 ]
            [ d(P*J)/dg1ds2  d(P*J)/dg2ds2 ]_{s=0}

        Derivatives are calculated using finite differencing
        """

        npoints=pars.shape[0]

        h2=1./(2.*h)
        hsq=1./h**2

        P=self.get_pj(pars, s1, s2, **keys)

        Q1_p   = self.get_pj(pars, s1+h, s2, **keys)
        Q1_m   = self.get_pj(pars, s1-h, s2, **keys)
        Q2_p   = self.get_pj(pars, s1,   s2+h, **keys)
        Q2_m   = self.get_pj(pars, s1,   s2-h, **keys)

        R12_pp = self.get_pj(pars, s1+h, s2+h, **keys)
        R12_mm = self.get_pj(pars, s1-h, s2-h, **keys)

        Q1 = (Q1_p - Q1_m)*h2
        Q2 = (Q2_p - Q2_m)*h2

        R11 = (Q1_p - 2*P + Q1_m)*hsq
        R22 = (Q2_p - 2*P + Q2_m)*hsq
        R12 = (R12_pp - Q1_p - Q2_p + 2*P - Q1_m - Q2_m + R12_mm)*hsq*0.5

        Q = zeros( (npoints,2) )
        R = zeros( (npoints,2,2) )

        Q[:,0] = Q1
        Q[:,1] = Q2
        R[:,0,0] = R11
        R[:,0,1] = R12
        R[:,1,0] = R12
        R[:,1,1] = R22

        return P, Q, R

    def get_pj(self, pars, s1, s2, **keys):
        """
        PJ = p(g,-shear)*jacob

        where jacob is d(es)/d(eo) and
        es=eo(+)(-g)
        """

        # note sending negative shear to jacob
        s1m=-s1
        s2m=-s2

        eta1,eta2 = pars[:,0], pars[:,1]

        J=zeros(eta1.size)
        for i in xrange(J.size):
            J[i]=shape.detas_by_detao_jacob_num(eta1[i],
                                                eta2[i],
                                                s1m,
                                                s2m,
                                                1.0e-6)

        # evaluating at negative shear
        eta1new,eta2new=shape.shear_eta(eta1,eta2,s1m,s2m)

        newpars=pars.copy()
        newpars[:,0] = eta1new
        newpars[:,1] = eta2new

        P = self.get_prob_array(newpars, **keys)

        return P*J

    def check_bounds_scalar(self, pars, throw=True):
        """
        Check bounds on T Flux
        """

        logT_bounds=self.logT_bounds
        logFlux_bounds=self.logFlux_bounds

        eta=sqrt(pars[0]**2 + pars[1]**2)
        logT=pars[2]
        logFb=pars[3]
        logFd=pars[4]
        if (eta > self.eta_max
            or logT < logT_bounds[0]
            or logT > logT_bounds[1]
            or logFb < logFlux_bounds[0]
            or logFb > logFlux_bounds[1]
            or logFd < logFlux_bounds[0]
            or logFd > logFlux_bounds[1]):

            if throw:
                raise GMixRangeError("g or T or F out of range")
            else:
                return False
        else:
            return True

    def check_bounds_array(self,pars):
        """
        Check bounds on T Flux
        """
        logT_bounds=self.logT_bounds
        logFlux_bounds=self.logFlux_bounds

        eta=sqrt(pars[:,0]**2 + pars[:,1]**2)
        logT=pars[:,2]
        logFb=pars[:,3]
        logFd=pars[:,4]

        wgood, = where(  (eta < self.eta_max)
                       & (logT > logT_bounds[0])
                       & (logT < logT_bounds[1])
                       & (logFb > logFlux_bounds[0])
                       & (logFb < logFlux_bounds[1])
                       & (logFd > logFlux_bounds[0])
                       & (logFd < logFlux_bounds[1]) )
        return wgood


    def _make_gmm(self):
        """
        Make a GMM object for sampling
        """
        from sklearn.mixture import GMM

        # these numbers are not used because we set the means, etc by hand
        ngauss=self.weights.size
        gmm=GMM(n_components=self.ngauss,
                n_iter=10000,
                min_covar=1.0e-12,
                covariance_type='full')
        gmm.means_ = self.means
        gmm.covars_ = self.covars
        gmm.weights_ = self.weights

        self.gmm=gmm 

    def test_pqr_shear_recovery(self, smin, smax, nshear,
                                npair=10000, h=1.e-6, eps=None):
        """
        Test how well we recover the shear with no noise.

        parameters
        ----------
        smin: float
            min shear to test
        smax: float
            max shear to test
        nshear:
            number of shear values to test
        npair: integer, optional
            Number of pairs to use at each shear test value
        """
        import lensing
        from .shape import Shape, shear_reduced

        shear1_true=numpy.linspace(smin, smax, nshear)
        shear2_true=numpy.zeros(nshear)

        shear1_meas=numpy.zeros(nshear)
        shear2_meas=numpy.zeros(nshear)
        
        # _te means expanded around truth
        shear1_meas_te=numpy.zeros(nshear)
        shear2_meas_te=numpy.zeros(nshear)
 
        theta=numpy.pi/2.0
        twotheta = 2.0*theta
        cos2angle = numpy.cos(twotheta)
        sin2angle = numpy.sin(twotheta)

        g1=numpy.zeros(npair*2)
        g2=numpy.zeros(npair*2)

        samples=numpy.zeros( (npair*2, self.ndim) )

        for ishear in xrange(nshear):
            s1=shear1_true[ishear]
            s2=shear2_true[ishear]
            print(s1,s2)

            tsamples=self.sample(npair)
            #print(self.get_pj(tsamples, 0.0, 0.0) )
            #print(self.get_pj(tsamples, s1, s2) )
            #stop

            eta1,eta2 = tsamples[:,0], tsamples[:,1]

            g1[0:npair],g2[0:npair],good = eta1eta2_to_g1g2_array(eta1,eta2)
            wbad,=numpy.where(good != 1)
            if wbad.size > 0:
                raise ValueError("bad")

            g1[npair:] =  g1[0:npair]*cos2angle + g2[0:npair]*sin2angle
            g2[npair:] = -g1[0:npair]*sin2angle + g2[0:npair]*cos2angle

            g1s, g2s = shear_reduced(g1, g2, s1, s2)

            eta1s, eta2s,good = g1g2_to_eta1eta2_array(g1s, g2s)
            wbad,=numpy.where(good != 1)
            if wbad.size > 0:
                raise ValueError("bad")


            samples[:, 0] = eta1s
            samples[:, 1] = eta2s
            samples[0:npair, 2:] = tsamples[:, 2:]
            samples[npair:, 2:] = tsamples[:, 2:]

            P,Q,R=self.get_pqr_num(samples, h=h)
            P_te,Q_te,R_te=self.get_pqr_num(samples, s1=s1, s2=s2, h=h)

            g1g2, C = lensing.pqr.get_pqr_shear(P,Q,R)
            g1g2_te, C_te = lensing.pqr.get_pqr_shear(P_te,Q_te,R_te)

            g1g2_te[0] += s1
            g1g2_te[0] += s2
            shear1_meas[ishear] = g1g2[0]
            shear2_meas[ishear] = g1g2[1]

            shear1_meas_te[ishear] = g1g2_te[0]
            shear2_meas_te[ishear] = g1g2_te[1]

            mess='true: %.6f,%.6f meas: %.6f,%.6f expand true: %.6f,%.6f'
            print(mess % (s1,s2,g1g2[0],g1g2[1],g1g2_te[0],g1g2_te[1]))

        fracdiff=shear1_meas/shear1_true-1
        fracdiff_te=shear1_meas_te/shear1_true-1
        if eps is not None:
            import biggles
            biggles.configure('default','fontsize_min',3)
            plt=biggles.FramedPlot()
            #plt.xlabel=r'$\gamma_{true}$'
            #plt.ylabel=r'$\Delta \gamma/\gamma$'
            plt.xlabel=r'$g_{true}$'
            plt.ylabel=r'$\Delta g/g$'
            plt.aspect_ratio=1.0

            plt.add( biggles.FillBetween([0.0,smax], [0.004,0.004], 
                                         [0.0,smax], [0.000,0.000],
                                          color='grey90') )
            plt.add( biggles.FillBetween([0.0,smax], [0.002,0.002], 
                                         [0.0,smax], [0.000,0.000],
                                          color='grey80') )


            psize=2.25
            pts=biggles.Points(shear1_true, fracdiff,
                               type='filled circle',size=psize,
                               color='blue')
            pts.label='expand shear=0'
            plt.add(pts)

            pts_te=biggles.Points(shear1_true, fracdiff_te,
                                  type='filled diamond',size=psize,
                                  color='red')
            pts_te.label='expand shear=true'
            plt.add(pts_te)

            coeffs=numpy.polyfit(shear1_true, fracdiff, 2)
            poly=numpy.poly1d(coeffs)

            curve=biggles.Curve(shear1_true, poly(shear1_true), type='solid',
                                color='black')
            #curve.label=r'$\Delta \gamma/\gamma~\propto~\gamma^2$'
            curve.label=r'$\Delta g/g = 1.9 g^2$'
            plt.add(curve)

            plt.add( biggles.PlotKey(0.1, 0.9, [pts,pts_te,curve], halign='left') )

            print('writing:',eps)
            plt.write_eps(eps)

            print(poly)




class JointPriorSimpleLinPars(GMixND):
    """
    Joint prior in g1,g2,T,Fb,Fd

    The prior is actually in eta1,eta2,logT,logFb,logFd as a sum of gaussians.
    When sampling the values are converted to the linear ones.  Also when
    getting the log probability, the input linear values are converted into log
    space.

    """
    def __init__(self,
                 weights,
                 means,
                 covars,
                 T_bounds,
                 Flux_bounds):

        self.T_bounds=T_bounds
        self.Flux_bounds=Flux_bounds

        super(JointPriorSimpleLinPars,self).__init__(weights, means, covars)

        self._make_gmm()


    def get_prob_scalar1d(self, pars, throw=False):
        """
        Use a numba compile function
        """
        if not self.check_bounds_scalar(pars,throw=throw):
            return 0.0

        return super(JointPriorSimpleLinPars,self).get_prob_scalar(pars)

    def get_lnprob_scalar1d(self, pars, throw=False):
        """
        (x-xmean) icovar (x-xmean)
        """
        if not self.check_bounds_scalar(pars,throw=throw):
            return LOWVAL

        return super(JointPriorSimpleLinPars,self).get_lnprob_scalar(pars)

    def get_prob_array1d(self, pars, **keys):
        """
        array input
        """

        p=zeros(pars.shape[0])

        w=self.check_bounds_array(pars)
        if w.size > 0:
            p[w]=super(JointPriorSimpleLinPars,self).get_prob_array(pars[w,:])

        return p

    def get_lnprob_array1d(self, pars, **keys):
        """
        array input
        """
        lnp=zeros(pars.shape[0]) + LOWVAL

        w=self.check_bounds_array(pars)
        if w.size > 0:
            lnp[w]=super(JointPriorSimpleLinPars,self).get_lnprob_array(pars[w,:])

        return lnp


    def get_prob_scalar2d(self, pars2d, **keys):
        """
        scalar input
        """

        pars1d=self.get_pars1d_scalar(pars2d)
        return self.get_prob_scalar1d(pars1d, **keys)

    def get_lnprob_scalar2d(self, pars2d, **keys):
        """
        scalar input
        """

        pars1d=self.get_pars1d_scalar(pars2d)
        return self.get_lnprob_scalar1d(pars1d, **keys)


    def get_prob_array2d(self, pars2d, **keys):
        """
        array input
        """

        pars1d=self.get_pars1d_array(pars2d)
        return self.get_prob_array1d(pars1d, **keys)

    def get_lnprob_array2d(self, pars2d, **keys):
        """
        array input
        """

        pars1d=self.get_pars1d_array(pars2d)
        return self.get_lnprob_array1d(pars1d, **keys)


    def get_pars1d_array(self, pars2d):
        """
        Convert 2d g pars to 1d g pars
        """
        pars1d=zeros( (pars2d.shape[0], self.ndim) )
        pars1d[:,0] = sqrt( pars2d[:,0]**2 + pars2d[:,1]**2 )
        pars1d[:,1:] = pars2d[:, 2:]
        return pars1d

    def get_pars1d_scalar(self, pars2d):
        """
        Convert 2d g pars to 1d g pars
        """
        pars1d = zeros(self.ndim)

        pars1d[0] = sqrt( pars2d[0]**2 + pars2d[1]**2 )
        pars1d[1:] = pars2d[2:]
        return pars1d


    def sample1d(self, n=None):
        """
        Get samples
        """

        if n is None:
            is_scalar=True
            n=1
        else:
            is_scalar=False

        samples=zeros( (n,self.ndim) )

        nleft=n
        ngood=0
        while nleft > 0:

            tsamples=self.gmm.sample(nleft)
            w=self.check_bounds_array(tsamples)

            if w.size > 0:
                first=ngood
                last=ngood+w.size

                samples[first:last,:] = tsamples[w,:]

                ngood += w.size
                nleft -= w.size

        if is_scalar:
            samples = samples[0,:]
        return samples

    def sample2d(self, n=None):
        """
        Get samples in 2d g space
        """
        if n is None:
            is_scalar=True
            n=1
        else:
            is_scalar=False

        samples1d = self.sample1d(n=n)

        grand=samples1d[:,0]

        theta = randu(n)*2*numpy.pi
        twotheta = 2*theta

        g1rand = grand*numpy.cos(twotheta)
        g2rand = grand*numpy.sin(twotheta)

        samples2d = zeros( (n, self.ndim+1) )

        samples2d[:,0] = g1rand
        samples2d[:,1] = g2rand
        samples2d[:,2:] = samples1d[:,1:]

        if is_scalar:
            samples2d = samples2d[0,:]
        return samples2d


    def get_pqr_num(self, pars, s1=0.0, s2=0.0, h=1.e-6, throw=True):
        """

        linear parameters input

        Evaluate 
            P
            Q
            R
        From Bernstein & Armstrong

        P is this prior times the jacobian at shear==0

        Q is the gradient of P*J evaluated at shear==0

            [ d(P*J)/ds1, d(P*J)/ds2]_{s=0}

        R is grad of grad of P*J at shear==0
            [ d(P*J)/dg1ds1  d(P*J)/dg1ds2 ]
            [ d(P*J)/dg1ds2  d(P*J)/dg2ds2 ]_{s=0}

        Derivatives are calculated using finite differencing
        """

        npoints=pars.shape[0]

        h2=1./(2.*h)
        hsq=1./h**2

        P=self.get_pj(pars, s1, s2, throw=throw)

        Q1_p   = self.get_pj(pars, s1+h, s2, throw=throw)
        Q1_m   = self.get_pj(pars, s1-h, s2, throw=throw)
        Q2_p   = self.get_pj(pars, s1,   s2+h, throw=throw)
        Q2_m   = self.get_pj(pars, s1,   s2-h, throw=throw)

        R12_pp = self.get_pj(pars, s1+h, s2+h, throw=throw)
        R12_mm = self.get_pj(pars, s1-h, s2-h, throw=throw)

        Q1 = (Q1_p - Q1_m)*h2
        Q2 = (Q2_p - Q2_m)*h2

        R11 = (Q1_p - 2*P + Q1_m)*hsq
        R22 = (Q2_p - 2*P + Q2_m)*hsq
        R12 = (R12_pp - Q1_p - Q2_p + 2*P - Q1_m - Q2_m + R12_mm)*hsq*0.5

        Q = zeros( (npoints,2) )
        R = zeros( (npoints,2,2) )

        Q[:,0] = Q1
        Q[:,1] = Q2
        R[:,0,0] = R11
        R[:,0,1] = R12
        R[:,1,0] = R12
        R[:,1,1] = R22

        return P, Q, R

    def get_pj(self, pars, s1, s2, throw=True):
        """
        PJ = p(g,-shear)*jacob

        where jacob is d(es)/d(eo) and
        es=eo(+)(-g)
        """

        # note sending negative shear to jacob
        s1m=-s1
        s2m=-s2

        g1,g2 = pars[:,0], pars[:,1]

        J=shape.dgs_by_dgo_jacob(g1, g2, s1m, s2m)

        # evaluating at negative shear
        # will get converted to log pars (eta)in get_prob_array
        g1new,g2new=shape.shear_reduced(g1, g2, s1m, s2m)

        newpars=pars.copy()
        newpars[:,0] = g1new
        newpars[:,1] = g2new

        P = self.get_prob_array2d(newpars, throw=throw)

        return P*J

    def _make_gmm(self):
        """
        Make a GMM object for sampling
        """
        from sklearn.mixture import GMM

        # these numbers are not used because we set the means, etc by hand
        ngauss=self.weights.size
        gmm=GMM(n_components=self.ngauss,
                n_iter=10000,
                min_covar=1.0e-12,
                covariance_type='full')
        gmm.means_ = self.means.copy()
        gmm.covars_ = self.covars.copy()
        gmm.weights_ = self.weights.copy()

        self.gmm=gmm 

    def check_bounds_scalar(self, pars, throw=True):
        """
        Check bounds on T Flux only
        """

        T_bounds=self.T_bounds
        Flux_bounds=self.Flux_bounds

        g=pars[0]
        T=pars[1]
        F=pars[2]
        if (g < 0.0
            or g >= 1.0
            or T < T_bounds[0]
            or T > T_bounds[1]
            or F < Flux_bounds[0]
            or F > Flux_bounds[1]):

            if throw:
                raise GMixRangeError("g or T or F out of range")
            else:
                return False
        else:
            return True

    def check_bounds_array(self,pars):
        """
        Check bounds on T Flux only
        """
        T_bounds=self.T_bounds
        Flux_bounds=self.Flux_bounds

        g=pars[:,0]
        T=pars[:,1]
        F=pars[:,2]

        wgood, = where(  (g >= 0.)
                       & (g < 1.0)
                       & (T > T_bounds[0])
                       & (T < T_bounds[1])
                       & (F > Flux_bounds[0])
                       & (F < Flux_bounds[1]) )
        return wgood

    def test_pqr_shear_recovery(self, smin, smax, nshear,
                                npair=10000, h=1.e-6, eps=None,
                                expand_shear=None):
        """
        Test how well we recover the shear with no noise.

        parameters
        ----------
        smin: float
            min shear to test
        smax: float
            max shear to test
        nshear:
            number of shear values to test
        npair: integer, optional
            Number of pairs to use at each shear test value
        """
        import lensing
        from .shape import Shape, shear_reduced

        shear1_true=numpy.linspace(smin, smax, nshear)
        shear2_true=numpy.zeros(nshear)

        shear1_meas=numpy.zeros(nshear)
        shear2_meas=numpy.zeros(nshear)
        
        # _te means expanded around truth
        shear1_meas_te=numpy.zeros(nshear)
        shear2_meas_te=numpy.zeros(nshear)

        theta=numpy.pi/2.0
        twotheta = 2.0*theta
        cos2angle = numpy.cos(twotheta)
        sin2angle = numpy.sin(twotheta)

        # extra dim because working in 2d
        samples=numpy.zeros( (npair*2, self.ndim+1) )
        g=numpy.zeros(npair)
        g1=numpy.zeros(npair*2)
        g2=numpy.zeros(npair*2)

        for ishear in xrange(nshear):
            s1=shear1_true[ishear]
            s2=shear2_true[ishear]

            tsamples = self.sample2d(npair)

            g1samp = tsamples[:,0]
            g2samp = tsamples[:,1]
            g1[0:npair]=g1samp
            g2[0:npair]=g2samp

            # ring test, rotated by pi/2
            g1[npair:] =  g1samp*cos2angle + g2samp*sin2angle
            g2[npair:] = -g1samp*sin2angle + g2samp*cos2angle

            # now shear all
            g1s, g2s = shear_reduced(g1, g2, s1, s2)

            #g=numpy.sqrt(g1s**2 + g2s**2)
            #print("gmin:",g.min(),"gmax:",g.max())

            samples[:,0] = g1s
            samples[:,1] = g2s
            samples[0:npair,2:] = tsamples[:,2:]
            samples[npair:, 2:] = tsamples[:,2:]

            P,Q,R=self.get_pqr_num(samples, h=h)

            if expand_shear is not None:
                s1expand=expand_shear[0]
                s2expand=expand_shear[1]
            else:
                s1expand=s1
                s2expand=s2
            P_te,Q_te,R_te=self.get_pqr_num(samples, s1=s1expand, s2=s2expand, h=h)


            g1g2, C = lensing.pqr.get_pqr_shear(P,Q,R)
            g1g2_te, C_te = lensing.pqr.get_pqr_shear(P_te,Q_te,R_te)

            #g1g2_te[0] += s1
            #g1g2_te[1] += s2

            shear1_meas[ishear] = g1g2[0]
            shear2_meas[ishear] = g1g2[1]

            shear1_meas_te[ishear] = g1g2_te[0]
            shear2_meas_te[ishear] = g1g2_te[1]

            mess='true: %.6f,%.6f meas: %.6f,%.6f expand true: %.6f,%.6f'
            print(mess % (s1,s2,g1g2[0],g1g2[1],g1g2_te[0],g1g2_te[1]))

        fracdiff=shear1_meas/shear1_true-1
        fracdiff_te=shear1_meas_te/shear1_true-1
        if eps:
            import biggles
            biggles.configure('default','fontsize_min',3)
            plt=biggles.FramedPlot()
            #plt.xlabel=r'$\gamma_{true}$'
            #plt.ylabel=r'$\Delta \gamma/\gamma$'
            plt.xlabel=r'$g_{true}$'
            plt.ylabel=r'$\Delta g/g$'
            plt.aspect_ratio=1.0

            plt.add( biggles.FillBetween([0.0,smax], [0.004,0.004], 
                                         [0.0,smax], [0.000,0.000],
                                          color='grey90') )
            plt.add( biggles.FillBetween([0.0,smax], [0.002,0.002], 
                                         [0.0,smax], [0.000,0.000],
                                          color='grey80') )


            psize=2.25
            pts=biggles.Points(shear1_true, fracdiff,
                               type='filled circle',size=psize,
                               color='blue')
            pts.label='expand shear=0'
            plt.add(pts)

            pts_te=biggles.Points(shear1_true, fracdiff_te,
                                  type='filled diamond',size=psize,
                                  color='red')
            pts_te.label='expand shear=true'
            plt.add(pts_te)

            if nshear > 1:
                coeffs=numpy.polyfit(shear1_true, fracdiff, 2)
                poly=numpy.poly1d(coeffs)

                curve=biggles.Curve(shear1_true, poly(shear1_true), type='solid',
                                    color='black')
                #curve.label=r'$\Delta \gamma/\gamma~\propto~\gamma^2$'
                curve.label=r'$\Delta g/g = 1.9 g^2$'
                plt.add(curve)

                plt.add( biggles.PlotKey(0.1, 0.9, [pts,pts_te,curve], halign='left') )
                print(poly)

            print('writing:',eps)
            plt.write_eps(eps)

class JointPriorSimpleLogPars(JointPriorSimpleLinPars):
    """
    Joint prior in g1,g2,T,F

    The prior is actually in eta1,eta2,logT,logF as a sum of gaussians.  When
    sampling the values are converted to the linear ones.  Also when getting
    the log probability, the input linear values are converted into log space.
    """
    def __init__(self,
                 weights,
                 means,
                 covars,
                 logT_bounds=[-1.5, 0.5],
                 logFlux_bounds=[-0.7, 1.5]):

        print("JointPriorSimpleLogPars")

        self.eta_max = 28.0
        self.logT_bounds=logT_bounds
        self.logFlux_bounds=logFlux_bounds

        super(JointPriorSimpleLogPars,self).__init__(weights, means, covars, 0.0, 0.0)

        self._make_gmm()


    def check_bounds_scalar(self, pars, throw=True):
        """
        Check bounds on T Flux
        """

        logT_bounds=self.logT_bounds
        logFlux_bounds=self.logFlux_bounds

        eta=pars[0]
        logT=pars[1]
        logF=pars[2]
        if (eta > self.eta_max
            or logT < logT_bounds[0]
            or logT > logT_bounds[1]
            or logF < logFlux_bounds[0]
            or logF > logFlux_bounds[1]):

            if throw:
                raise GMixRangeError("g or T or F out of range")
            else:
                return False
        else:
            return True

    def check_bounds_array(self,pars):
        """
        Check bounds on T Flux
        """
        logT_bounds=self.logT_bounds
        logFlux_bounds=self.logFlux_bounds

        eta=pars[:,0]
        logT=pars[:,1]
        logF=pars[:,2]

        wgood, = where(  (eta < self.eta_max)
                       & (logT > logT_bounds[0])
                       & (logT < logT_bounds[1])
                       & (logF > logFlux_bounds[0])
                       & (logF < logFlux_bounds[1]) )
        return wgood



class JointPriorSimpleHybrid(JointPriorSimpleLinPars):
    """
    Joint prior in g1,g2,T,F

    The prior is actually in eta1,eta2,logT,logF as a sum of gaussians.  When
    sampling the values are converted to the linear ones.  Also when getting
    the log probability, the input linear values are converted into log space.
    """
    def __init__(self,
                 weights,
                 means,
                 covars,
                 g_prior,
                 logT_bounds=[-1.5, 0.5],
                 logFlux_bounds=[-0.7, 1.5]):

        print("JointPriorSimpleHybrid")

        self.logT_bounds=logT_bounds
        self.logFlux_bounds=logFlux_bounds
        self.g_prior=g_prior

        GMixND.__init__(self, weights, means, covars)

        self.ndim=2
        self._make_gmm()

    def get_prob_scalar(self, pars, **keys):
        """
        Use a numba compile function
        """
        if not self.check_bounds_scalar(pars, **keys):
            return 0.0

        return GMixND.get_prob_scalar(self,pars)

    def get_lnprob_scalar(self, pars, **keys):
        """
        (x-xmean) icovar (x-xmean)
        """
        if not self.check_bounds_scalar(pars, **keys):
            return LOWVAL

        return GMixND.get_lnprob_scalar(self,pars)

    def get_prob_array(self, pars, **keys):
        """
        array input
        """

        p=zeros(pars.shape[0])

        w=self.check_bounds_array(pars)
        if w.size > 0:
            p[w]=GMixND.get_prob_array(self,pars[w,:])

        return p

    def get_lnprob_array(self, pars, **keys):
        """
        array input
        """
        lnp=zeros(pars.shape[0]) + LOWVAL

        w=self.check_bounds_array(pars)
        if w.size > 0:
            lnp[w]=GMixND.get_lnprob_array(self,pars[w,:])

        return lnp


    def sample(self, n=None):
        """
        Get samples
        """

        if n is None:
            is_scalar=True
            n=1
        else:
            is_scalar=False

        samples=zeros( (n,self.ndim) )

        nleft=n
        ngood=0
        while nleft > 0:

            tsamples=self.gmm.sample(nleft)
            w=self.check_bounds_array(tsamples)

            if w.size > 0:
                first=ngood
                last=ngood+w.size

                samples[first:last,:] = tsamples[w,:]

                ngood += w.size
                nleft -= w.size

        if is_scalar:
            samples = samples[0,:]
        return samples


    def check_bounds_scalar(self, pars, throw=True):
        """
        Check bounds on T Flux
        """

        logT_bounds=self.logT_bounds
        logFlux_bounds=self.logFlux_bounds

        logT=pars[0]
        logF=pars[1]
        if (  logT < logT_bounds[0]
            or logT > logT_bounds[1]
            or logF < logFlux_bounds[0]
            or logF > logFlux_bounds[1]):

            if throw:
                raise GMixRangeError("g or T or F out of range")
            else:
                return False
        else:
            return True

    def check_bounds_array(self,pars):
        """
        Check bounds on T Flux
        """
        logT_bounds=self.logT_bounds
        logFlux_bounds=self.logFlux_bounds

        logT=pars[:,0]
        logF=pars[:,1]

        wgood, = where(  (logT > logT_bounds[0])
                       & (logT < logT_bounds[1])
                       & (logF > logFlux_bounds[0])
                       & (logF < logFlux_bounds[1]) )
        return wgood

class JointPriorBDFHybrid(JointPriorSimpleHybrid):
    """
    The flux bounds are applied separately to both
    Fb and Fd
    """
    def __init__(self,
                 weights,
                 means,
                 covars,
                 g_prior,
                 logT_bounds=[-1.5, 0.5],
                 logFlux_bounds=[-3.0, 1.0]):

        print("JointPriorBDFHybrid")

        self.logT_bounds=logT_bounds
        self.logFlux_bounds=logFlux_bounds
        self.g_prior=g_prior

        GMixND.__init__(self, weights, means, covars)

        self.ndim=3
        self._make_gmm()

    def check_bounds_scalar(self, pars, throw=True):
        """
        Check bounds on T Flux
        """

        logT_bounds=self.logT_bounds
        logFlux_bounds=self.logFlux_bounds

        logT=pars[0]
        logFb=pars[1]
        logFd=pars[2]

        if (   logT < logT_bounds[0]
            or logT > logT_bounds[1]
            or logFb < logFlux_bounds[0]
            or logFb > logFlux_bounds[1]
            or logFd < logFlux_bounds[0]
            or logFd > logFlux_bounds[1]) :

            if throw:
                raise GMixRangeError("T or Fb,Fd out of range")
            else:
                return False
        else:
            return True


    def check_bounds_array(self,pars):
        """
        Check bounds on T Flux
        """
        logT_bounds=self.logT_bounds
        logFlux_bounds=self.logFlux_bounds

        logT=pars[:,0]
        logFb=pars[:,1]
        logFd=pars[:,2]

        wgood, = where(  (logT > logT_bounds[0])
                       & (logT < logT_bounds[1])
                       & (logFb > logFlux_bounds[0])
                       & (logFb < logFlux_bounds[1])
                       & (logFd > logFlux_bounds[0])
                       & (logFd < logFlux_bounds[1]) )
        return wgood


