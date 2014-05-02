"""
class PQR
"""

import numpy
from numpy import where, zeros
from .gexceptions import GMixRangeError, GMixFatalError

class PQR(object):
    def __init__(self, g, g_prior, shear_expand=[0.0,0.0], remove_prior=False):
        """
        A class to calculate the P,Q,R terms from Bernstein & Armstrong

        parameters
        ----------
        g: 2-d array
            g values [N,2]
        g_prior:
            The g prior object.
        remove_prior: bool, optional
            Remove the prior value from the Q,R terms.  This is needed
            if the prior was used in likelihood exploration.
        """

        self._g=g
        self._g_prior=g_prior

        self._shear_expand=array(shear_expand)
        assert self._shear_expand.size==2,"shear expand should have two elements"

        self._remove_prior=remove_prior

        self._calc_pqr()

    def get_pqr(self):
        """
        get P,Q,R
        """
        return self._P,self._Q,self._R
    
    def _calc_pqr(self):
        """
        get the P,Q,R
        """

        g_prior=self._g_prior

        g1=self._g[:,0]
        g2=self._g[:,1]

        sh=self._shear_expand
        print("        expanding pqr about:",sh)
        if hasattr(g_prior,'get_pqr_expand'):
            Pi,Qi,Ri = g_prior.get_pqr_expand(g1,g2, sh[0], sh[1])
        else:
            Pi,Qi,Ri = g_prior.get_pqr_num(g1, g2, s1=sh[0], s2=sh[1])

        self._P,self._Q,self._R = self._get_mean_pqr(Pi,Qi,Ri)

    def _get_mean_pqr(self, Pi, Qi, Ri):
        """
        Get the mean P,Q,R marginalized over priors.  Optionally weighted for
        importance sampling
        """

        if self._remove_prior:
            # We measured the posterior surface.  But the integrals are over
            # the likelihood.  So divide by the prior.
            #
            # Also note the p we divide by is in principle different from the
            # Pi above, which are evaluated at the shear expansion value

            print("        undoing prior for pqr")

            g1=self._g[:,0]
            g2=self._g[:,1]
            prior_vals = self._g_prior.get_prob_array2d(g1,g2)

            w,=numpy.where(prior_vals > 0.0)

            Pinv = 1.0/prior_vals[w]
            Pinv_sum=Pinv.sum()

            Pi = Pi[w]
            Qi = Qi[w,:]
            Ri = Ri[w,:,:]

            # this is not unity if expanding about some shear
            Pi *= Pinv
            Qi[:,0] *= Pinv 
            Qi[:,1] *= Pinv

            Ri[:,0,0] *= Pinv
            Ri[:,0,1] *= Pinv
            Ri[:,1,0] *= Pinv
            Ri[:,1,1] *= Pinv

            P = Pi.sum()/Pinv_sum
            Q = Qi.sum(axis=0)/Pinv_sum
            R = Ri.sum(axis=0)/Pinv_sum
        else:
            P = Pi.mean()
            Q = Qi.mean(axis=0)
            R = Ri.mean(axis=0)

        return P,Q,R


