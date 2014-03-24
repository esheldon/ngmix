from __future__ import print_function

import numpy
from numpy import where, log10, zeros, exp, sqrt

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

        self.T_bounds=[10.0**logT_bounds[0], 10.0**logT_bounds[1]]
        self.Flux_bounds=[10.0**logFlux_bounds[0], 10.0**logFlux_bounds[1]]

        self._make_gmm()

    def get_lnprob_array(self, pars, throw=True):
        """
        using gmm

        the pars are in linear space
            [g1,g2,T,F1,....]
        """
        logpars,w=self._pars_to_logpars_array(pars, throw=throw)
        lnp = zeros(pars.shape[0]) + LOWVAL
        lnp[w] = self.gmm.score(logpars[w,:])
        return lnp

    def get_prob_array(self, pars, throw=True):
        """
        exp(lnprob)

        if throw==False and there are bad values ,they
        get prob==0
        """
        lnp = self.get_lnprob_array(pars, throw=throw)
        return exp(lnp)

    def get_lnprob_scalar(self, pars):
        """
        the pars are in linear space
            [g1,g2,T,Fb,Fd]
        """
        logpars=self._pars_to_logpars_scalar(pars)
        lnp = super(JointPriorBDF,self).get_lnprob_scalar(logpars)
        return lnp

    def get_prob_scalar(self, pars):
        """
        the pars are in linear space
            [g1,g2,T,Fb,Fd]
        """
        logpars=self._pars_to_logpars_scalar(pars)
        lnp = super(JointPriorBDF,self).get_prob_scalar(logpars)
        return lnp



    def sample(self, n=None):
        """
        Get samples in linear space
        """

        if n is None:
            is_scalar=True
            n=1
        else:
            is_scalar=False

        logT_bounds = self.logT_bounds
        logFlux_bounds = self.logFlux_bounds

        lin_samples=zeros( (n,self.ndim) )
        nleft=n
        ngood=0
        while nleft > 0:
            log_samples=self.gmm.sample(nleft)

            eta1,eta2 = log_samples[:,0], log_samples[:,1]
            g1,g2,g_good = eta1eta2_to_g1g2_array(eta1, eta2)

            logT = log_samples[:,2]
            logFb = log_samples[:,3]
            logFd = log_samples[:,4]
            
            w, = where(  (g_good == 1)
                       & (logT > logT_bounds[0])
                       & (logT < logT_bounds[1])
                       & (logFb > logFlux_bounds[0])
                       & (logFb < logFlux_bounds[1])
                       & (logFd > logFlux_bounds[0])
                       & (logFd < logFlux_bounds[1]) )

            if w.size > 0:
                first=ngood
                last=ngood+w.size

                T = 10.0**logT[w]
                Fb = 10.0**logFb[w]
                Fd = 10.0**logFd[w]

                lin_samples[first:last,0] = g1[w]
                lin_samples[first:last,1] = g2[w]
                lin_samples[first:last,2] = T
                lin_samples[first:last,3] = Fb
                lin_samples[first:last,4] = Fd

                ngood += w.size
                nleft -= w.size

        if is_scalar:
            lin_samples = lin_samples[0,:]
        return lin_samples



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

        raise RuntimeError("figure what do do with jacob in log pars")
        # note sending negative shear to jacob
        s1m=-s1
        s2m=-s2

        g1,g2 = pars[:,0], pars[:,1]

        eta1,eta2,good=g1g2_to_eta1eta2_array(g1,g2)
        J=zeros(eta1.size)
        for i in xrange(J.size):
            J[i]=shape.detas_by_detao_jacob_num(eta1[i],
                                                eta2[i],
                                                s1m,
                                                s2m,
                                                1.0e-6)

        #J=shape.dgs_by_dgo_jacob(g1, g2, s1m, s2m)

        # evaluating at negative shear
        # will get converted to log pars (eta)in get_prob_array
        g1new,g2new=shape.shear_reduced(g1, g2, s1m, s2m)

        newpars=pars.copy()
        newpars[:,0] = g1new
        newpars[:,1] = g2new

        n=g1.size
        P=zeros(n)

        P = self.get_prob_array(newpars, throw=throw)

        return P*J


    def _pars_to_logpars_scalar(self, pars):
        """
        convert the pars to log space
            [eta1,eta2,logT,logFb,logFd]
        """
        logpars=pars.copy()
        g1 = pars[0]
        g2 = pars[1]
        T  = pars[2]
        Fb = pars[3]
        Fd = pars[4]

        T_bounds=self.T_bounds
        Flux_bounds=self.Flux_bounds

        if ( (T < T_bounds[0])
                or (T > T_bounds[1])
                or (Fb < Flux_bounds[0])
                or (Fb > Flux_bounds[1])
                or (Fd < Flux_bounds[0])
                or (Fd > Flux_bounds[1]) ):
            raise GMixRangeError("T, Fb, Fd out of range")

        logpars[0],logpars[1] = g1g2_to_eta1eta2(g1,g2)
        logpars[2] = log10(T)
        logpars[3] = log10(Fb)
        logpars[4] = log10(Fd)

        return logpars

    def _pars_to_logpars_array(self, pars, throw=True):
        """
        convert the pars to log space
            [eta1,eta2,logT,logFb,logFd]

        If throw==False then the log values are just set to very
        negative values
        """

        num=pars.shape[0]

        logpars=pars.copy()
        logpars[:,:] = LOWVAL

        g1 = pars[:,0]
        g2 = pars[:,1]
        T  = pars[:,2]
        Fb = pars[:,3]
        Fd = pars[:,4]

        eta1,eta2,good = g1g2_to_eta1eta2_array(g1,g2)

        T_bounds=self.T_bounds
        Flux_bounds=self.Flux_bounds

        w,=where(  (good == 1)
                 & (T > T_bounds[0])
                 & (T < T_bounds[1])
                 & (Fb > Flux_bounds[0])
                 & (Fb < Flux_bounds[1])
                 & (Fd > Flux_bounds[0])
                 & (Fd < Flux_bounds[1]) )

        if w.size != num:
            mess='%s with T,Fb,Fd negative or bad e' % (num-w.size)
            if throw:
                raise GMixRangeError(mess)
            else:
                print(mess)

                wbad,=where(  (good != 1)
                            | (T <= T_bounds[0])
                            | (T >= T_bounds[1])
                            | (Fb <= Flux_bounds[0])
                            | (Fb >= Flux_bounds[1])
                            | (Fd <= Flux_bounds[0])
                            | (Fd >= Flux_bounds[1]) )

                n = 10 if wbad.size > 10 else wbad.size
                for i in xrange(n):
                    print(pars[wbad[i],:])

        logpars[w,0] = eta1[w]
        logpars[w,1] = eta2[w]
        logpars[w,2] = log10(T[w])
        logpars[w,3] = log10(Fb[w])
        logpars[w,4] = log10(Fd[w])

        return logpars, w


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


class JointPriorSimpleLogPars(JointPriorBDF):
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

        super(JointPriorSimple,self).__init__(weights,
                                              means,
                                              covars,
                                              logT_bounds=logT_bounds,
                                              logFlux_bounds=logFlux_bounds)

    def sample(self, n=None):
        """
        Get samples in linear space
        """

        if n is None:
            is_scalar=True
            n=1
        else:
            is_scalar=False

        logT_bounds = self.logT_bounds
        logFlux_bounds = self.logFlux_bounds

        lin_samples=zeros( (n,self.ndim) )
        nleft=n
        ngood=0
        while nleft > 0:
            log_samples=self.gmm.sample(nleft)

            eta1,eta2 = log_samples[:,0], log_samples[:,1]
            g1,g2,g_good = eta1eta2_to_g1g2_array(eta1, eta2)

            logT = log_samples[:,2]
            logF = log_samples[:,3]
            
            w, = where(  (g_good == 1)
                       & (logT > logT_bounds[0])
                       & (logT < logT_bounds[1])
                       & (logF > logFlux_bounds[0])
                       & (logF < logFlux_bounds[1]) )

            if w.size > 0:
                first=ngood
                last=ngood+w.size

                T = 10.0**logT[w]
                F = 10.0**logF[w]

                lin_samples[first:last,0] = g1[w]
                lin_samples[first:last,1] = g2[w]
                lin_samples[first:last,2] = T
                lin_samples[first:last,3] = F

                ngood += w.size
                nleft -= w.size

        if is_scalar:
            lin_samples = lin_samples[0,:]
        return lin_samples



    def _pars_to_logpars_scalar(self, pars):
        """
        convert the pars to log space
            [eta1,eta2,logT,logF]
        """
        logpars=pars.copy()
        g1 = pars[0]
        g2 = pars[1]
        T  = pars[2]
        F  = pars[3]

        T_bounds=self.T_bounds
        Flux_bounds=self.Flux_bounds

        if ( (T < T_bounds[0])
                or (T > T_bounds[1])
                or (F < Flux_bounds[0])
                or (F > Flux_bounds[1]) ):
            raise GMixRangeError("T, F out of range")

        # will raise if out of bounds
        logpars[0],logpars[1] = g1g2_to_eta1eta2(g1,g2)
        logpars[2] = log10(T)
        logpars[3] = log10(F)

        return logpars

    def _pars_to_logpars_array(self, pars, throw=True):
        """
        convert the pars to log space
            [eta1,eta2,logT,logF]

        If throw==False then the log values are just set to very
        negative values
        """

        num=pars.shape[0]

        logpars=pars.copy()
        logpars[:,:] = LOWVAL

        g1 = pars[:,0]
        g2 = pars[:,1]
        T  = pars[:,2]
        F  = pars[:,3]

        eta1,eta2,good = g1g2_to_eta1eta2_array(g1,g2)

        T_bounds=self.T_bounds
        Flux_bounds=self.Flux_bounds

        w,=where(  (good == 1)
                 & (T > T_bounds[0])
                 & (T < T_bounds[1])
                 & (F > Flux_bounds[0])
                 & (F < Flux_bounds[1]) )

        if w.size != num:
            mess='%s with T or F negative or bad e' % (num-w.size)
            if throw:
                raise GMixRangeError(mess)
            else:
                print(mess)

                wbad,=where(  (good != 1)
                            | (T <= T_bounds[0])
                            | (T >= T_bounds[1])
                            | (F <= Flux_bounds[0])
                            | (F >= Flux_bounds[1]) )

                n = 10 if wbad.size > 10 else wbad.size
                for i in xrange(n):
                    print(pars[wbad[i],:])

        logpars[w,0] = eta1[w]
        logpars[w,1] = eta2[w]
        logpars[w,2] = log10(T[w])
        logpars[w,3] = log10(F[w])

        return logpars, w


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
                 covars):

        super(JointPriorSimpleLinPars,self).__init__(weights, means, covars)

        self._make_gmm()

    def sample(self, n=None):
        """
        Get samples in linear space
        """

        if n is None:
            is_scalar=True
            n=1
        else:
            is_scalar=False


        nleft=n
        ngood=0
        while nleft > 0:
            samples=self.gmm.sample(nleft)

            g1,g2 = samples[:,0], samples[:,1]
            g=sqrt(g1**2 + g2**2)
            

            T = samples[:,2]
            F = samples[:,3]
            
            w, = where(  (g < 1)
                       & (T > 0)
                       & (F > 0) )

            if w.size > 0:
                first=ngood
                last=ngood+w.size

                samples[first:last,0] = g1[w]
                samples[first:last,1] = g2[w]
                samples[first:last,2] = T[w]
                samples[first:last,3] = F[w]

                ngood += w.size
                nleft -= w.size

        if is_scalar:
            samples = samples[0,:]
        return samples



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

        n=g1.size
        P=zeros(n)

        P = self.get_prob_array(newpars, throw=throw)

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
        gmm.means_ = self.means
        gmm.covars_ = self.covars
        gmm.weights_ = self.weights

        self.gmm=gmm 


