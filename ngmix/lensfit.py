"""
class LensfitSensitivity
function calc_lensfit_shear
"""

from __future__ import print_function, absolute_import, division
import numpy
from numpy import where, zeros, ones, array, isfinite, newaxis
from .gexceptions import GMixRangeError, GMixFatalError

_default_h=1.0e-6

def calc_sensitivity(g, g_prior, remove_prior=False, h=_default_h):
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

    ls=LensfitSensitivity(g, g_prior, remove_prior=remove_prior, h=h)
    g_sens=ls.get_g_sens()
    return g_sens

def calc_shear(g, g_sens, weights=None):
    """
    Calculate shear from g and g sensitivity arrays

    parameters
    ----------
    g: array
        g1 and g2 as a [N,2] array
    g_sens: array
        sensitivity as a [N,2] array
    weights: array, optional
        Weights to apply as [N] array
    """

    if weights is None:
        shear = g.mean(axis=0)/g_sens.mean(axis=0)
    else:
        wa=weights[:,newaxis]
        shear = (g*wa).sum()/(g_sens*wa).sum()

    return shear

class LensfitSensitivity(object):
    def __init__(self,
                 g,
                 g_prior,
                 weights=None,
                 remove_prior=False,
                 response=None,
                 h=_default_h):
        """
        parameters
        ----------
        g: 2-d array
            g values [N,2]
        g_prior:
            The g prior object.
        weights: array
            Weights for each point in n-d space.  Cannot 
            use this with remove_prior=True
        remove_prior: bool, optional
            Remove the prior value from the Q,R terms.  This is needed
            if the prior was used in likelihood exploration.
        """

        self._g=g
        self._g_prior=g_prior
        self._weights=weights
        self._response=response

        self._remove_prior=remove_prior
        self._h=h

        self._calc_g_sens()


    def get_g_mean(self):
        """
        get the g mean
        """
        return self._g_mean

    def get_g_sens(self):
        """
        get the g sensitivity values as a [N, 2] array
        """
        return self._g_sens

    def get_nuse(self):
        """
        get number of points used.  Will be less than the input number
        of points if remove_prior is set and some prior values were zero
        """
        return self._nuse

    def _calc_g_sens(self):
        if self._remove_prior:
            self._calc_g_sens_remove_prior()
        else:
            self._calc_g_sens_no_remove_prior()

    def _calc_g_sens_remove_prior(self):
        """
        Calculate the sensitivity
        """

        response=self._response

        g_mean = zeros(2)
        g_sens = zeros(2)

        g1=self._g[:,0]
        g2=self._g[:,1]

        weights=self._weights
        if weights is None:
            weights=ones(g1.size)

        # derivative of log prior
        dpri_by_g1 = self._g_prior.dlnbyg1_array(g1,g2,h=self._h)
        dpri_by_g2 = self._g_prior.dlnbyg2_array(g1,g2,h=self._h)

        w,=where( isfinite(dpri_by_g1) & isfinite(dpri_by_g2) )

        if w.size == 0:
            raise GMixRangeError("no prior values > 0")
        elif w.size != g1.size:
            print("    found %d/%d finite" % (w.size, g1.size))

            weights = weights[w]
            g1 = g1[w]
            g2 = g2[w]
            dpri_by_g1 = dpri_by_g1[w]
            dpri_by_g2 = dpri_by_g2[w]

            if response is not None:
                response = response[w]

        wsum = weights.sum()

        g1mean = (g1*weights).sum()/wsum
        g2mean = (g2*weights).sum()/wsum

        g1diff = g1mean-g1
        g2diff = g2mean-g2

        R1 = g1diff*dpri_by_g1
        R2 = g2diff*dpri_by_g2

        if response is not None:
            R1sum = (response[:,0]*(1-R1)*weights).sum()
            R2sum = (response[:,1]*(1-R2)*weights).sum()
        else:
            R1sum = ((1-R1)*weights).sum()
            R2sum = ((1-R2)*weights).sum()

        #R1sum = (R1*weights).sum()
        #R2sum = (R2*weights).sum()

        g_mean[0] = g1mean
        g_mean[1] = g2mean
        g_sens[0] = R1sum/wsum
        g_sens[1] = R2sum/wsum
        #g_sens[0] = 1 - R1sum/wsum
        #g_sens[1] = 1 - R2sum/wsum

        self._nuse=w.size

        self._g_mean=g_mean
        self._g_sens=g_sens

    def _calc_g_sens_no_remove_prior(self):
        """
        Calculate the sensitivity
        """

        g_sens = zeros(2)

        g1=self._g[:,0]
        g2=self._g[:,1]

        dpri_by_g1 = self._g_prior.dbyg1_array(g1,g2,h=self._h)
        dpri_by_g2 = self._g_prior.dbyg2_array(g1,g2,h=self._h)

        if self._response is not None:
            dpri_by_g1 *= self._response[:,0]
            dpri_by_g2 *= self._response[:,1]

        prior=self._g_prior.get_prob_array2d(g1,g2)

        extra_weights=self._weights

        if extra_weights is not None:
            doweights=True
            weights = prior*extra_weights
        else:
            doweights=False
            weights = prior

        wsum = weights.sum()

        g1mean = (g1*weights).sum()/wsum
        g2mean = (g2*weights).sum()/wsum

        g1diff = g1mean-g1
        g2diff = g2mean-g2

        R1 = g1diff*dpri_by_g1
        R2 = g2diff*dpri_by_g2

        if doweights:
            # wsum is (w*prior).sum()
            #R1sum = (R1*extra_weights).sum()
            #R2sum = (R2*extra_weights).sum()
            R1 *= extra_weights
            R2 *= extra_weights
        else:
            # wsum is prior.sum()
            #R1sum = R1.sum()
            #R2sum = R2.sum()
            pass

        if self._response is not None:
            raise RuntimeError("check response here")
            res=self._response
            R1sum = ((res[:,0]-R1)*weights).sum()
            R2sum = ((res[:,1]-R2)*weights).sum()
        else:
            R1sum = ((1-R1)*weights).sum()
            R2sum = ((1-R2)*weights).sum()


        g_sens[0] = R1sum/wsum
        g_sens[1] = R2sum/wsum
        #g_sens[0] = 1 - R1sum/wsum
        #g_sens[1] = 1 - R2sum/wsum

        self._nuse=g1.size

        self._g_mean=array([g1mean, g2mean])
        self._g_sens=g_sens



    def _calc_g_sens_old(self):
        """
        Calculate the sensitivity
        """

        g_sens = zeros(2)

        g1=self._g[:,0]
        g2=self._g[:,1]

        dpri_by_g1 = self._g_prior.dbyg1_array(g1,g2,h=self._h)
        dpri_by_g2 = self._g_prior.dbyg2_array(g1,g2,h=self._h)

        prior=self._g_prior.get_prob_array2d(g1,g2)

        if self._remove_prior:
            print("        undoing prior for lensfit")

            w,=where( prior > 0.0 )
            if w.size == 0:
                raise GMixRangeError("no prior values > 0")
            g1mean=g1[w].mean()
            g2mean=g2[w].mean()

            g1diff = g1mean-g1
            g2diff = g2mean-g2

            R1 = g1diff[w]*dpri_by_g1[w]/prior[w]
            R2 = g2diff[w]*dpri_by_g2[w]/prior[w]

            g_sens[0] = 1- R1.mean()
            g_sens[1] = 1- R2.mean()

            self._nuse=w.size
        else:
            extra_weights=self._weights

            if extra_weights is not None:
                doweights=True
                weights = prior*extra_weights
                wsum = weights.sum()
            else:
                doweights=False
                weights = prior
                wsum = prior.sum()

            g1mean = (g1*weights).sum()/wsum
            g2mean = (g2*weights).sum()/wsum

            g1diff = g1mean-g1
            g2diff = g2mean-g2

            R1 = g1diff*dpri_by_g1
            R2 = g2diff*dpri_by_g2

            if doweights:
                # wsum is (w*prior).sum()
                R1sum = (R1*extra_weights).sum()
                R2sum = (R2*extra_weights).sum()
            else:
                # wsum is prior.sum()
                R1sum = R1.sum()
                R2sum = R2.sum()

            g_sens[0] = 1 - R1sum/wsum
            g_sens[1] = 1 - R2sum/wsum

            self._nuse=g1.size

        self._g_mean=array([g1mean, g2mean])
        self._g_sens=g_sens


def lensfit_jackknife(g, gsens, do_ring=False, **keys):
    """
    get the shear lensfit style

    for keywords, see help for _lensfit_jackknife

    parameters
    ----------
    g: array
        [N,2] shape measurements
    gsens: array
        [N,2] shape sensitivity measurements
    do_ring: bool, optional
        Was the data in a ring configuration?
    chunksize: int, optional
        chunksize for jackknifing
    get_sums: bool, optional
        Get the sums as well
    get_shears: bool, optional
        Get shears from jackknife sub samples
    weights: array, optional
        Weights to apply
    progress: bool, optional
        show a progress bar 
    show: bool, optional
        Show a plot of the jackknife sample values
    eps: string, optional
        eps file to write
    png: string, optional
        png file to write
    """
    if do_ring:
        return _lensfit_jackknife_ring(g, gsens,**keys)
    else:
        return _lensfit_jackknife(g, gsens, **keys)

def _lensfit_jackknife(g, gsens,
                       gsens_alt=None,
                       chunksize=1,
                       get_sums=False,
                       get_shears=False,
                       weights=None,
                       progress=False,
                       show=False,
                       eps=None,
                       png=None):
    """
    Get the shear covariance matrix using jackknife resampling.
    """

    if weights is None:
        weights=ones(g.shape[0])

    if progress:
        import progressbar
        pg=progressbar.ProgressBar(width=70)

    ntot = g.shape[0]

    # some may not get used
    nchunks = ntot//chunksize

    wsum = weights.sum()
    wa=weights[:,newaxis]

    g_sum = (g*wa).sum(axis=0)
    gsens_sum = (gsens*wa).sum(axis=0)

    shear = g_sum/gsens_sum

    if gsens_alt is not None:
        gsens_alt_sum = (gsens_alt*wa).sum(axis=0)
        
        gsens_alt_mean=gsens_alt_sum/wsum
        shear = shear/gsens_alt_mean

    shears = numpy.zeros( (nchunks, 2) )
    for i in xrange(nchunks):

        beg = i*chunksize
        end = (i+1)*chunksize
        
        if progress:
            frac=float(i+1)/nchunks
            pg.update(frac=frac)

        this_wts = (weights[beg:end])[:,newaxis]
        this_gsum = (g[beg:end,:]*this_wts).sum(axis=0)
        this_gsens_sum = (gsens[beg:end,:]*this_wts).sum(axis=0)

        j_g_sum     = g_sum     - this_gsum
        j_gsens_sum = gsens_sum - this_gsens_sum

        shears[i, :] = j_g_sum/j_gsens_sum

        if gsens_alt is not None:
            this_gsens_alt_sum = (gsens_alt[beg:end,:]*this_wts).sum(axis=0)
            wdiff = wsum - weights[beg:end].sum()
            j_gsens_alt_mean = (gsens_alt_sum - this_gsens_alt_sum)/wdiff
            shears[i,:] *= 1.0/j_gsens_alt_mean

    shear_cov = numpy.zeros( (2,2) )
    fac = (nchunks-1)/float(nchunks)

    shear_cov[0,0] = fac*( ((shear[0]-shears[:,0])**2).sum() )
    shear_cov[0,1] = fac*( ((shear[0]-shears[:,0]) * (shear[1]-shears[:,1])).sum() )
    shear_cov[1,0] = shear_cov[0,1]
    shear_cov[1,1] = fac*( ((shear[1]-shears[:,1])**2).sum() )

    if show or eps or png:
        from .pqr import _plot_shears
        _plot_shears(shears, show=show, eps=eps, png=png)

    if get_sums:
        if gsens_alt is not None:
            return shear, shear_cov, g_sum, gsens_sum, gsens_alt_sum
        else:
            return shear, shear_cov, g_sum, gsens_sum
    elif get_shears:
        return shear, shear_cov, shears
    else:
        return shear, shear_cov


def _lensfit_jackknife_ring(g, gsens,
                            gsens_alt=None,
                            chunksize=1,
                            get_sums=False,
                            get_shears=False,
                            weights=None,
                            progress=False,
                            show=False,
                            eps=None,
                            png=None):
    """
    Get the shear covariance matrix using jackknife resampling.

    The trick is that this must be done in pairs for ring tests

    chunksize is the number of *pairs* to remove for each chunk
    """

    if weights is None:
        weights=ones(g.shape[0])

    if progress:
        import progressbar
        pg=progressbar.ProgressBar(width=70)

    ntot = g.shape[0]
    if ( (ntot % 2) != 0 ):
        raise  ValueError("expected factor of two, got %d" % ntot)
    npair = ntot//2

    # some may not get used
    nchunks = npair//chunksize

    wsum = weights.sum()
    wa=weights[:,newaxis]

    g_sum = (g*wa).sum(axis=0)
    gsens_sum = (gsens*wa).sum(axis=0)

    shear = g_sum/gsens_sum

    if gsens_alt is not None:
        gsens_alt_sum = (gsens_alt*wa).sum(axis=0)
        
        gsens_alt_mean=gsens_alt_sum/wsum
        shear = shear/gsens_alt_mean

    shears = numpy.zeros( (nchunks, 2) )
    for i in xrange(nchunks):

        beg = i*chunksize*2
        end = (i+1)*chunksize*2
        
        if progress:
            frac=float(i+1)/nchunks
            pg.update(frac=frac)

        this_wts = (weights[beg:end])[:,newaxis]
        this_gsum = (g[beg:end,:]*this_wts).sum(axis=0)
        this_gsens_sum = (gsens[beg:end,:]*this_wts).sum(axis=0)


        j_g_sum     = g_sum     - this_gsum
        j_gsens_sum = gsens_sum - this_gsens_sum

        shears[i, :] = j_g_sum/j_gsens_sum

        if gsens_alt is not None:
            this_gsens_alt_sum = (gsens_alt[beg:end,:]*this_wts).sum(axis=0)
            wdiff = wsum - weights[beg:end].sum()
            j_gsens_alt_mean = (gsens_alt_sum - this_gsens_alt_sum)/wdiff
            shears[i,:] *= 1.0/j_gsens_alt_mean


    shear_cov = numpy.zeros( (2,2) )
    fac = (nchunks-1)/float(nchunks)

    shear_cov[0,0] = fac*( ((shear[0]-shears[:,0])**2).sum() )
    shear_cov[0,1] = fac*( ((shear[0]-shears[:,0]) * (shear[1]-shears[:,1])).sum() )
    shear_cov[1,0] = shear_cov[0,1]
    shear_cov[1,1] = fac*( ((shear[1]-shears[:,1])**2).sum() )

    if show or eps or png:
        from .pqr import _plot_shears
        _plot_shears(shears, show=show, eps=eps, png=png)

    if get_sums:
        if gsens_alt is not None:
            return shear, shear_cov, g_sum, gsens_sum, gsens_alt_sum
        else:
            return shear, shear_cov, g_sum, gsens_sum
    elif get_shears:
        return shear, shear_cov, shears
    else:
        return shear, shear_cov


