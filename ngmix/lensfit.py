"""
class LensfitSensitivity
function calc_lensfit_shear
"""

import numpy
from numpy import where, zeros
from .gexceptions import GMixRangeError, GMixFatalError

def calc_sensitivity(g, g_prior, remove_prior=False):
    """
    parameters
    ----------
    g: 2-d array
        The g1,g2 values over the likelihood surface as a [N,2] array
    g_prior:
        The g prior object.
    remove_prior: bool, optional
        Remove the prior value from the Q,R terms.  This is needed
        if the prior was used in likelihood exploration.

    """

    ls=LensfitSensitivity(g, g_prior, remove_prior=remove_prior)
    gsens=ls.get_gsens()
    return gsens

def calc_shear(g, gsens):
    """
    Calculate shear from g and g sensitivity arrays

    parameters
    ----------
    g: array
        g1 and g2 as a [N,2] array
    gsens: array
        sensitivity as a [N,2] array
    """
    w=where(gsens > 0.0)
    if w[0].size != g.shape[0]:
        raise GMixFatalError("some gsens were <= 0.0")

    shear = g.mean(axis=0)/gsens.mean(axis=0)

    return shear

class LensfitSensitivity(object):
    def __init__(self, g, g_prior, remove_prior=False):
        """
        parameters
        ----------
        g1: 1-d array
            The g1 values over the likelihood surface
        g2: 1-d array
            The g2 values over the likelihood surface
        g_prior:
            The g prior object.
        remove_prior: bool, optional
            Remove the prior value from the Q,R terms.  This is needed
            if the prior was used in likelihood exploration.
        """

        self._g=g
        self._g_prior=g_prior
        self._remove_prior=remove_prior

        self._calc_gsens()


    def get_gsens(self):
        """
        get the g sensitivity values as a [N, 2] array
        """
        return self._gsens

    def get_nuse(self):
        """
        get number of points used.  Will be less than the input number
        of points if remove_prior is set and some prior values were zero
        """
        return self._nuse

    def _calc_gsens(self):
        """
        Calculate the sensitivity
        """

        g1=self._g[:,0]
        g2=self._g[:,1]

        dpri_by_g1 = self._g_prior.dbyg1_array(g1,g2)
        dpri_by_g2 = self._g_prior.dbyg2_array(g1,g2)

        prior=self._g_prior.get_prob_array2d(g1,g2)

        if self._remove_prior:
            w,=where( prior > 0.0 )
            if w.size == 0:
                raise GMixRangeError("no prior values > 0")
            g1mean=g1[w].mean()
            g2mean=g2[w].mean()

            self._indices=w
            self._nuse=w.size
        else:
            psum=prior.sum()
            g1mean = (g1*prior).sum()/psum
            g2mean = (g2*prior).sum()/psum
            self._nuse=g1.size

        g1diff = g1mean-g1
        g2diff = g2mean-g2

        gsens = zeros(2)

        if self._remove_prior:
            R1 = g1diff[w]*dpri_by_g1[w]/prior[w]
            R2 = g2diff[w]*dpri_by_g2[w]/prior[w]
        else:
            R1 = g1diff*dpri_by_g1
            R2 = g2diff*dpri_by_g2

        gsens[0]= 1.- R1.mean()
        gsens[1]= 1.- R2.mean()

        self._gsens=gsens


