import numpy
from numpy import where, log10

from .shape import g1g2_to_eta1eta2, eta1eta2_to_g1g2_array
from .gexceptions import GMixRangeError

from . import gmix

class JointPriorBDF(gmix.GMixND):
    """
    Joint prior in g1,g2,T,Fb,Fd

    The prior is actually in eta1,eta2,logT,logFb,logFd
    as a sum of gaussians
    """
    def __init__(self,
                 weights,
                 means,
                 covars,
                 logT_bounds=[-1.6, 1.0],
                 logFlux_bounds=[-3.0, 2.0]):

        super(JointPriorBDF,self).__init__(weights, means, covars)

        self.logT_bounds=logT_bounds
        self.logFlux_bounds=logFlux_bounds
        self._make_gmm()

    def get_lnprob(self, pars):
        """
        the pars are in linear space
            [g1,g2,T,Fb,Fd]
        """
        logpars=self._pars_to_logpars(pars)
        lnp = super(JointPriorBDF,self).get_lnprob(logpars)
        return lnp

    def _pars_to_logpars(self, pars):
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

        if T <= 0 or Tb <= 0 or Fd <= 0:
            raise GMixRangeError("T, Fb, Fd must be positive")

        logpars[0],logpars[1] = g1g2_to_eta1eta2(g1,g2)
        logpars[2] = log10(T)
        logpars[3] = log10(Fb)
        logpars[4] = log10(Fd)

        return logpars

    def sample(self, n):
        """
        Get samples in linear space
        """

        logT_bounds = self.logT_bounds
        logFlux_bounds = self.logFlux_bounds

        lin_samples=numpy.zeros( (n,self.ndim) )
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

        return lin_samples


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



