from __future__ import print_function

try:
    xrange = xrange
    # We have Python 2
except:
    xrange = range
    # We have Python 3

import numpy
from numpy import where, log10, zeros, ones, exp, sqrt
from numpy.random import random as randu

from . import shape
from .gexceptions import GMixRangeError

from . import priors
from .priors import LOWVAL
from . import gmix
from .gmix import GMixND


class JointPriorTF(GMixND):
    """
    Joint prior in T,F,  T and F could be linear or log, this
    code is agnostic.

    We may want the hard bounds because we fit with sums of gaussians those can
    allow very large values when drawing randoms.

    """
    def __init__(self,
                 weights,
                 means,
                 covars,
                 T_bounds=None,
                 F_bounds=None):

        print("JointPriorTF")

        self.T_bounds=T_bounds
        self.F_bounds=F_bounds

        GMixND.__init__(self, weights, means, covars)

        self._make_gmm()

    def get_prob_scalar(self, pars, **keys):
        """
        probability for scalar input (meaning one point)
        """

        if not self.check_bounds_scalar(pars, **keys):
            return 0.0

        return GMixND.get_prob_scalar(self,pars)

    def get_lnprob_scalar(self, pars, **keys):
        """
        log probability for scalar input (meaning one point)
        """
        if not self.check_bounds_scalar(pars, **keys):
            return LOWVAL

        return GMixND.get_lnprob_scalar(self,pars)

    def get_prob_array(self, pars, **keys):
        """
        probability for array input [N,ndims]
        """

        p=zeros(pars.shape[0])

        w=self.check_bounds_array(pars)
        if w.size > 0:
            p[w]=GMixND.get_prob_array(self,pars[w,:])

        return p

    def get_lnprob_array(self, pars, **keys):
        """
        log probability for array input [N,ndims]
        """
        lnp=zeros(pars.shape[0]) + LOWVAL

        w=self.check_bounds_array(pars)
        if w.size > 0:
            lnp[w]=GMixND.get_lnprob_array(self,pars[w,:])

        return lnp

    def sample(self, n=None):
        """
        Get samples for TF
        """

        if n is None:
            is_scalar=True
            n=1
        else:
            is_scalar=False

        samples=zeros( (n,2) )

        nleft=n
        ngood=0
        while nleft > 0:

            tsamples=self._gmm.sample(nleft)
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

        T=pars[0]
        F=pars[1]

        T_bounds=self.T_bounds
        if T_bounds is not None:
            if T < T_bounds[0] or T > T_bounds[1]:
                if throw:
                    raise GMixRangeError("T out of range")
                else:
                    return False

        F_bounds=self.F_bounds
        if F_bounds is not None:
            if F < F_bounds[0] or F > F_bounds[1]:
                if throw:
                    raise GMixRangeError("F out of range")
                else:
                    return False

        return True

    def check_bounds_array(self,pars):
        """
        Check bounds on T Flux
        """
        T_bounds=self.T_bounds
        F_bounds=self.F_bounds

        T=pars[:,0]
        F=pars[:,1]

        logic = ones(T.shape[0], dtype=bool)
        if T_bounds is not None:
            T_logic = ( (T > T_bounds[0]) & (T < T_bounds[1]) )
            logic = logic & T_logic

        if F_bounds is not None:
            F_logic = ((F > F_bounds[0]) & (F < F_bounds[1]) )
            logic = logic & F_logic

        wgood,=where(logic)
        return wgood


class JointPriorSimpleHybrid(GMixND):
    """
    Joint prior in g1,g2,T,F.  T and F are joint in log10 space

    Hybrid because g_prior and cen_prior are separate, but
    TF is joint.

    parameters
    ----------
    cen_prior:
        The center prior
    g_prior:
        The prior on g (g1,g2).
    TF_prior:
        Joint prior on T and F.
    """
    def __init__(self,
                 cen_prior,
                 g_prior,
                 TF_prior):

        print("JointPriorSimpleHybrid")

        self.cen_prior=cen_prior
        self.g_prior=g_prior
        self.TF_prior=TF_prior

    def get_widths(self, n=10000):
        """
        estimate the width in each dimension
        """
        if not hasattr(self, '_sigma_estimates'):
            samples=self.sample(n)
            sigmas = samples.std(axis=0)
            self._sigma_estimates=sigmas

            '''
            print("means  of TF:",samples.mean(axis=0))
            print("sigmas of TF:",sigmas)

            import esutil as eu
            eu.plotting.bhist(samples[:,4], binsize=0.2*sigmas[4], xlabel='log10(T)')
            eu.plotting.bhist(samples[:,5], binsize=0.2*sigmas[5], xlabel='log10(F)')
            '''
        return self._sigma_estimates

    def get_prob_scalar(self, pars, **keys):
        """
        probability for scalar input (meaning one point)
        """

        lnp = self.get_lnprob_scalar(pars, **keys)
        p = exp(lnp)
        return p

    def get_lnprob_scalar(self, pars, **keys):
        """
        log probability for scalar input (meaning one point)
        """

        lnp = self.cen_prior.get_lnprob_scalar(pars[0],pars[1])
        lnp += self.g_prior.get_lnprob_scalar2d(pars[2],pars[3])
        lnp += self.TF_prior.get_lnprob_scalar(pars[4:4+2], **keys)

        return lnp

    def get_prob_array(self, pars, **keys):
        """
        probability for array input [N,ndims]
        """

        lnp = self.get_lnprob_array(pars, **keys)
        p = exp(lnp)

        return p

    def get_lnprob_array(self, pars, **keys):
        """
        log probability for array input [N,ndims]
        """

        lnp = self.cen_prior.get_lnprob_array(pars[:,0], pars[:,1])
        lnp += self.g_prior.get_lnprob_array2d(pars[:,2],pars[:,3])
        lnp += self.TF_prior.get_lnprob_array(pars[:,4:4+2])

        return lnp

    def fill_fdiff(self, pars, fdiff, **keys):
        """
        set sqrt(-2ln(p)) ~ (model-data)/err
        """
        index=0

        #fdiff[index] = self.cen_prior.get_lnprob_scalar(pars[0],pars[1])

        lnp1,lnp2=self.cen_prior.get_lnprob_scalar_sep(pars[0],pars[1])

        fdiff[index] = lnp1
        index += 1
        fdiff[index] = lnp2
        index += 1

        fdiff[index] = self.g_prior.get_lnprob_scalar2d(pars[2],pars[3])
        index += 1
        fdiff[index] =  self.TF_prior.get_lnprob_scalar(pars[4:4+2], **keys)
        index += 1

        chi2 = -2*fdiff[0:index]
        chi2.clip(min=0.0, max=None, out=chi2)
        fdiff[0:index] = sqrt(chi2)

        return index



    def sample(self, n=None):
        """
        Get random samples
        """

        if n is None:
            is_scalar=True
            n=1
        else:
            is_scalar=False

        samples=zeros( (n,6) )

        cen1,cen2 = self.cen_prior.sample(n)
        g1,g2=self.g_prior.sample2d(n)
        TF=self.TF_prior.sample(n)

        samples[:,0] = cen1
        samples[:,1] = cen2
        samples[:,2] = g1
        samples[:,3] = g2
        samples[:,4] = TF[:,0]
        samples[:,5] = TF[:,1]

        if is_scalar:
            samples = samples[0,:]
        return samples


    def check_bounds_scalar(self, pars, throw=True):
        """
        Check bounds on T Flux
        """

        T_bounds=self.T_bounds
        F_bounds=self.F_bounds

        T=pars[4]
        F=pars[5]
        if (  T < T_bounds[0]
            or T > T_bounds[1]
            or F < F_bounds[0]
            or F > F_bounds[1]):

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
        T_bounds=self.T_bounds
        F_bounds=self.F_bounds

        T=pars[:,4]
        F=pars[:,5]

        wgood, = where(  (T > T_bounds[0])
                       & (T < T_bounds[1])
                       & (F > F_bounds[0])
                       & (F < F_bounds[1]) )
        return wgood


class JointPriorSersicHybrid(JointPriorSimpleHybrid):
    """
    Joint prior in cen1,cen2,g1,g2,T,F,n.

    parameters
    ----------
    cen_prior:
        The center prior
    g_prior:
        The prior on g (g1,g2).
    TF_prior:
        Joint prior on T and F.
    n_prior:
        Prior on sersic index n
    """
    def __init__(self,
                 cen_prior,
                 g_prior,
                 TF_prior,
                 n_prior,
                 logn_bounds=[log10(0.751), log10(5.99)],
                 **keys):

        super(JointPriorSersicHybrid,self).__init__(cen_prior,
                                                    g_prior,
                                                    TF_prior,
                                                    **keys)


        self.n_prior=n_prior
        self.logn_bounds=self.logn_bounds

    def get_lnprob_scalar(self, pars, **keys):
        """
        log probability for scalar input (meaning one point)
        """

        lnp=super(JointPriorSersicHybrid,self).get_lnprob_scalar(pars, **keys)
        lnp += self.n_prior.get_lnprob_scalar(pars[6])
        return lnp
        
    def get_lnprob_array(self, pars, **keys):
        """
        log probability for array input [N,ndims]
        """
        lnp = self.get_lnprob_array(pars, **keys)
        lnp += self.n_prior.get_lnprob_array(pars[:,6])
        return lnp

    def sample(self, n=None):
        """
        Get random samples
        """

        if n is None:
            is_scalar=True
            n=1
        else:
            is_scalar=False

        samples=zeros( (n,7) )

        tsamples = super(JointPriorSersicHybrid,self).sample(n)

        samples[:,0:0+6] = tsamples
        samples[:,6] = self.n_prior.sample(n)

        if is_scalar:
            samples = samples[0,:]
        return samples

    def check_bounds_scalar(self, pars, throw=True):
        """
        Check bounds on T Flux
        """

        T_bounds=self.T_bounds
        F_bounds=self.F_bounds
        logn_bounds=self.logn_bounds

        T=pars[4]
        F=pars[5]
        logn=pars[6]

        if (  T < T_bounds[0]
            or T > T_bounds[1]
            or F < F_bounds[0]
            or F > F_bounds[1]
            or logn < logn_bounds[0]
            or logn > logn_bounds[1]):

            if throw:
                raise GMixRangeError("T, F or n out of range")
            else:
                return False
        else:
            return True

    def check_bounds_array(self,pars):
        """
        Check bounds on T Flux
        """
        T_bounds=self.T_bounds
        F_bounds=self.F_bounds
        logn_bounds=self.logn_bounds

        T=pars[:,4]
        F=pars[:,5]
        logn=pars[:,6]

        wgood, = where(  (T > T_bounds[0])
                       & (T < T_bounds[1])
                       & (F > F_bounds[0])
                       & (F < F_bounds[1])
                       & (logn > logn_bounds[0])
                       & (logn < logn_bounds[1]) )
        return wgood


# not used currently

class JointPriorSimpleLinPars(GMixND):
    """
    Joint prior in T,F
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
        just do bound schecking and call the super
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

            tsamples=self._gmm.sample(nleft)
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

    The prior is actually in eta1,eta2,T,F as a sum of gaussians.  When
    sampling the values are converted to the linear ones.  Also when getting
    the log probability, the input linear values are converted into log space.
    """
    def __init__(self,
                 weights,
                 means,
                 covars,
                 T_bounds=[-1.5, 0.5],
                 F_bounds=[-0.7, 1.5]):

        print("JointPriorSimpleLogPars")

        self.eta_max = 28.0
        self.T_bounds=T_bounds
        self.F_bounds=F_bounds

        super(JointPriorSimpleLogPars,self).__init__(weights, means, covars, 0.0, 0.0)

        self._make_gmm()


    def check_bounds_scalar(self, pars, throw=True):
        """
        Check bounds on T Flux
        """

        T_bounds=self.T_bounds
        F_bounds=self.F_bounds

        eta=pars[0]
        T=pars[1]
        F=pars[2]
        if (eta > self.eta_max
            or T < T_bounds[0]
            or T > T_bounds[1]
            or F < F_bounds[0]
            or F > F_bounds[1]):

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
        T_bounds=self.T_bounds
        F_bounds=self.F_bounds

        eta=pars[:,0]
        T=pars[:,1]
        F=pars[:,2]

        wgood, = where(  (eta < self.eta_max)
                       & (T > T_bounds[0])
                       & (T < T_bounds[1])
                       & (F > F_bounds[0])
                       & (F < F_bounds[1]) )
        return wgood


def make_uniform_simple_sep(cen, cen_width, T_range, F_range):
    """
    For testing

    Make PriorSimpleSep uniform in all priors except the
    center, which is gaussian
    """

    cen_prior=priors.CenPrior(cen[0], cen[1], cen_width[0], cen_width[1])
    g_prior=priors.ZDisk2D(1.0)
    T_prior=priors.FlatPrior(T_range[0], T_range[1])
    F_prior=priors.FlatPrior(F_range[0], F_range[1])

    pr=PriorSimpleSep(cen_prior, g_prior, T_prior, F_prior)
    return pr

def make_erf_simple_sep(cen, cen_width, Tpars, Fpars):
    """
    For testing

    Make PriorSimpleSep uniform in all priors except the
    center, which is gaussian
    """

    cen_prior=priors.CenPrior(cen[0], cen[1], cen_width[0], cen_width[1])
    g_prior=priors.ZDisk2D(1.0)
    T_prior=priors.TwoSidedErf(*Tpars)
    F_prior=priors.TwoSidedErf(*Fpars)

    pr=PriorSimpleSep(cen_prior, g_prior, T_prior, F_prior)
    return pr


def make_cosmos_simple_sep(cen, cen_width, T_range, F_range):
    """
    For testing

    Make PriorSimpleSep uniform in all priors except the
    center, which is gaussian
    """

    cen_prior=priors.CenPrior(cen[0], cen[1], cen_width[0], cen_width[1])
    g_prior = priors.make_gprior_cosmos_sersic()
    T_prior=priors.FlatPrior(T_range[0], T_range[1])
    F_prior=priors.FlatPrior(F_range[0], F_range[1])

    pr=PriorSimpleSep(cen_prior, g_prior, T_prior, F_prior)
    return pr

def make_uniform_simple_eta_sep(cen, cen_width, T_range, F_range):
    """
    For testing

    Make PriorSimpleSep uniform in all priors except the
    center, which is gaussian
    """

    cen_prior=priors.CenPrior(cen[0], cen[1], cen_width[0], cen_width[1])
    g_prior=priors.FlatEtaPrior()
    T_prior=priors.FlatPrior(T_range[0], T_range[1])
    F_prior=priors.FlatPrior(F_range[0], F_range[1])

    pr=PriorSimpleSep(cen_prior, g_prior, T_prior, F_prior)
    return pr


class PriorSimpleSep(object):
    """
    Separate priors on each parameter

    parameters
    ----------
    cen_prior:
        The center prior
    g_prior:
        The prior on g (g1,g2).
    T_prior:
        Prior on T
    F_prior:
        Prior on Flux.  Can be a list for a multi-band prior.
    """

    def __init__(self,
                 cen_prior,
                 g_prior,
                 T_prior,
                 F_prior):

        #print("JointPriorSimpleSep")

        self.cen_prior=cen_prior
        self.g_prior=g_prior
        self.T_prior=T_prior

        if isinstance(F_prior,list):
            self.nband=len(F_prior)
        else:
            self.nband=1
            F_prior=[F_prior]

        self.F_priors=F_prior

    def get_widths(self, n=10000):
        """
        estimate the width in each dimension
        """
        if not hasattr(self, '_sigma_estimates'):
            import esutil as eu
            samples=self.sample(n)
            sigmas = samples.std(axis=0)

            # for e1,e2 we want to allow this a bit bigger
            # for very small objects.  Steps for MH could be
            # as large as half this
            sigmas[2] = 2.0
            sigmas[3] = 2.0

            self._sigma_estimates=sigmas

        return self._sigma_estimates


    def get_prob_scalar(self, pars, **keys):
        """
        probability for scalar input (meaning one point)
        """

        lnp = self.get_lnprob_scalar(pars, **keys)
        p = exp(lnp)
        return p

    def get_lnprob_scalar(self, pars, **keys):
        """
        log probability for scalar input (meaning one point)
        """

        lnp = self.cen_prior.get_lnprob_scalar(pars[0],pars[1])
        lnp += self.g_prior.get_lnprob_scalar2d(pars[2],pars[3])
        lnp += self.T_prior.get_lnprob_scalar(pars[4], **keys)

        #for i in xrange(self.nband):
        #    F_prior=self.F_priors[i]

        for i, F_prior in enumerate(self.F_priors):
            lnp += F_prior.get_lnprob_scalar(pars[5+i], **keys)

        return lnp


    def fill_fdiff(self, pars, fdiff, **keys):
        """
        set sqrt(-2ln(p)) ~ (model-data)/err
        """
        index=0

        #fdiff[index] = self.cen_prior.get_lnprob_scalar(pars[0],pars[1])

        lnp1,lnp2=self.cen_prior.get_lnprob_scalar_sep(pars[0],pars[1])

        fdiff[index] = lnp1
        index += 1
        fdiff[index] = lnp2
        index += 1

        fdiff[index] = self.g_prior.get_lnprob_scalar2d(pars[2],pars[3])
        index += 1
        fdiff[index] =  self.T_prior.get_lnprob_scalar(pars[4], **keys)
        index += 1

        for i in xrange(self.nband):
            F_prior=self.F_priors[i]
            fdiff[index] = F_prior.get_lnprob_scalar(pars[5+i], **keys)
            index += 1

        chi2 = -2*fdiff[0:index]
        chi2.clip(min=0.0, max=None, out=chi2)
        fdiff[0:index] = sqrt(chi2)

        return index


    def get_prob_array(self, pars, **keys):
        """
        probability for array input [N,ndims]
        """

        lnp = self.get_lnprob_array(pars, **keys)
        p = exp(lnp)

        return p

    def get_lnprob_array(self, pars, **keys):
        """
        log probability for array input [N,ndims]
        """

        lnp = self.cen_prior.get_lnprob_array(pars[:,0], pars[:,1])
        lnp += self.g_prior.get_lnprob_array2d(pars[:,2],pars[:,3])
        lnp += self.T_prior.get_lnprob_array(pars[:,4])

        for i in xrange(self.nband):
            F_prior=self.F_priors[i]
            lnp += F_prior.get_lnprob_array(pars[:,5+i])

        return lnp

    def sample(self, n=None, **unused_keys):
        """
        Get random samples
        """

        if n is None:
            is_scalar=True
            n=1
        else:
            is_scalar=False

        samples=zeros( (n,5+self.nband) )

        cen1,cen2 = self.cen_prior.sample(n)
        g1,g2=self.g_prior.sample2d(n)
        T=self.T_prior.sample(n)

        samples[:,0] = cen1
        samples[:,1] = cen2
        samples[:,2] = g1
        samples[:,3] = g2
        samples[:,4] = T

        for i in xrange(self.nband):
            F_prior=self.F_priors[i]
            F=F_prior.sample(n)
            samples[:,5+i] = F

        if is_scalar:
            samples=samples[0,:]
        return samples

    def __repr__(self):
        reps=[]
        reps += [str(self.cen_prior),
                 str(self.g_prior),
                 str(self.T_prior)]

        for p in self.F_priors:
            reps.append( str(p) )

        rep='\n'.join(reps)
        return rep

class PriorSimpleSepRound(PriorSimpleSep):
    """
    Separate priors on each parameter, no shape

    parameters
    ----------
    cen_prior:
        The center prior
    T_prior:
        Prior on T
    F_prior:
        Prior on Flux.  Can be a list for a multi-band prior.
    """

    def __init__(self,
                 cen_prior,
                 T_prior,
                 F_prior):

        self.cen_prior=cen_prior
        self.T_prior=T_prior

        if isinstance(F_prior,list):
            self.nband=len(F_prior)
        else:
            self.nband=1
            F_prior=[F_prior]

        self.F_priors=F_prior

    def get_widths(self, n=10000):
        """
        estimate the width in each dimension
        """
        if not hasattr(self, '_sigma_estimates'):
            import esutil as eu
            samples=self.sample(n)
            sigmas = samples.std(axis=0)

            self._sigma_estimates=sigmas

        return self._sigma_estimates


    def get_lnprob_scalar(self, pars, **keys):
        """
        log probability for scalar input (meaning one point)
        """

        lnp = self.cen_prior.get_lnprob_scalar(pars[0],pars[1])
        lnp += self.T_prior.get_lnprob_scalar(pars[2], **keys)

        for i, F_prior in enumerate(self.F_priors):
            lnp += F_prior.get_lnprob_scalar(pars[3+i], **keys)

        return lnp


    def fill_fdiff(self, pars, fdiff, **keys):
        """
        set sqrt(-2ln(p)) ~ (model-data)/err
        """
        index=0

        #fdiff[index] = self.cen_prior.get_lnprob_scalar(pars[0],pars[1])

        lnp1,lnp2=self.cen_prior.get_lnprob_scalar_sep(pars[0],pars[1])

        fdiff[index] = lnp1
        index += 1
        fdiff[index] = lnp2
        index += 1


        fdiff[index] =  self.T_prior.get_lnprob_scalar(pars[2], **keys)
        index += 1

        for i in xrange(self.nband):
            F_prior=self.F_priors[i]
            fdiff[index] = F_prior.get_lnprob_scalar(pars[3+i], **keys)
            index += 1

        chi2 = -2*fdiff[0:index]
        chi2.clip(min=0.0, max=None, out=chi2)
        fdiff[0:index] = sqrt(chi2)

        return index

    def get_lnprob_array(self, pars, **keys):
        """
        log probability for array input [N,ndims]
        """

        lnp = self.cen_prior.get_lnprob_array(pars[:,0], pars[:,1])
        lnp += self.T_prior.get_lnprob_array(pars[:,2])

        for i in xrange(self.nband):
            F_prior=self.F_priors[i]
            lnp += F_prior.get_lnprob_array(pars[:,3+i])

        return lnp

    def sample(self, n=None, **unused_keys):
        """
        Get random samples
        """

        if n is None:
            is_scalar=True
            n=1
        else:
            is_scalar=False

        samples=zeros( (n,3+self.nband) )

        cen1,cen2 = self.cen_prior.sample(n)
        g1,g2=self.g_prior.sample2d(n)
        T=self.T_prior.sample(n)

        samples[:,0] = cen1
        samples[:,1] = cen2
        samples[:,2] = T

        for i in xrange(self.nband):
            F_prior=self.F_priors[i]
            F=F_prior.sample(n)
            samples[:,3+i] = F

        if is_scalar:
            samples=samples[0,:]
        return samples

    def __repr__(self):
        reps=[]
        reps += [str(self.cen_prior),
                 str(self.T_prior)]

        for p in self.F_priors:
            reps.append( str(p) )

        rep='\n'.join(reps)
        return rep

class PriorSimpleSepFixT(object):
    """
    Separate priors on each parameter

    parameters
    ----------
    cen_prior:
        The center prior
    g_prior:
        The prior on g (g1,g2).
    F_prior:
        Prior on Flux.  Can be a list for a multi-band prior.
    """

    def __init__(self,
                 cen_prior,
                 g_prior,
                 F_prior):

        #print("JointPriorSimpleSep")

        self.cen_prior=cen_prior
        self.g_prior=g_prior

        if isinstance(F_prior,list):
            self.nband=len(F_prior)
        else:
            self.nband=1
            F_prior=[F_prior]

        self.F_priors=F_prior

    def get_widths(self, n=10000):
        """
        estimate the width in each dimension
        """
        if not hasattr(self, '_sigma_estimates'):
            import esutil as eu
            samples=self.sample(n)
            sigmas = samples.std(axis=0)

            # for e1,e2 we want to allow this a bit bigger
            # for very small objects.  Steps for MH could be
            # as large as half this
            sigmas[2] = 2.0
            sigmas[3] = 2.0

            self._sigma_estimates=sigmas

        return self._sigma_estimates


    def get_prob_scalar(self, pars, **keys):
        """
        probability for scalar input (meaning one point)
        """

        lnp = self.get_lnprob_scalar(pars, **keys)
        p = exp(lnp)
        return p

    def get_lnprob_scalar(self, pars, **keys):
        """
        log probability for scalar input (meaning one point)
        """

        lnp = self.cen_prior.get_lnprob_scalar(pars[0],pars[1])
        lnp += self.g_prior.get_lnprob_scalar2d(pars[2],pars[3])

        for i, F_prior in enumerate(self.F_priors):
            lnp += F_prior.get_lnprob_scalar(pars[4+i], **keys)

        return lnp


    def fill_fdiff(self, pars, fdiff, **keys):
        """
        set sqrt(-2ln(p)) ~ (model-data)/err
        """
        index=0

        #fdiff[index] = self.cen_prior.get_lnprob_scalar(pars[0],pars[1])

        lnp1,lnp2=self.cen_prior.get_lnprob_scalar_sep(pars[0],pars[1])

        fdiff[index] = lnp1
        index += 1
        fdiff[index] = lnp2
        index += 1

        fdiff[index] = self.g_prior.get_lnprob_scalar2d(pars[2],pars[3])
        index += 1

        for i in xrange(self.nband):
            F_prior=self.F_priors[i]
            fdiff[index] = F_prior.get_lnprob_scalar(pars[4+i], **keys)
            index += 1

        chi2 = -2*fdiff[0:index]
        chi2.clip(min=0.0, max=None, out=chi2)
        fdiff[0:index] = sqrt(chi2)

        return index


    def get_prob_array(self, pars, **keys):
        """
        probability for array input [N,ndims]
        """

        lnp = self.get_lnprob_array(pars, **keys)
        p = exp(lnp)

        return p

    def get_lnprob_array(self, pars, **keys):
        """
        log probability for array input [N,ndims]
        """

        lnp = self.cen_prior.get_lnprob_array(pars[:,0], pars[:,1])
        lnp += self.g_prior.get_lnprob_array2d(pars[:,2],pars[:,3])

        for i in xrange(self.nband):
            F_prior=self.F_priors[i]
            lnp += F_prior.get_lnprob_array(pars[:,5+i])

        return lnp

    def sample(self, n=None, **unused_keys):
        """
        Get random samples
        """

        if n is None:
            is_scalar=True
            n=1
        else:
            is_scalar=False

        samples=zeros( (n,5+self.nband) )

        cen1,cen2 = self.cen_prior.sample(n)
        g1,g2=self.g_prior.sample2d(n)

        samples[:,0] = cen1
        samples[:,1] = cen2
        samples[:,2] = g1
        samples[:,3] = g2

        for i in xrange(self.nband):
            F_prior=self.F_priors[i]
            F=F_prior.sample(n)
            samples[:,4+i] = F

        if is_scalar:
            samples=samples[0,:]
        return samples

    def __repr__(self):
        reps=[]
        reps += [str(self.cen_prior),
                 str(self.g_prior)]

        for p in self.F_priors:
            reps.append( str(p) )

        rep='\n'.join(reps)
        return rep

class PriorMomSep(object):
    """
    Separate priors on each parameter

    parameters
    ----------
    cen_prior:
        The center prior
    M1_prior:
        The prior on M1
    M2_prior:
        Prior on M2
    T_prior:
        Prior on T.
    F_prior
        Prior on Flux.  Can be a list for a multi-band prior.
    """

    def __init__(self,
                 cen1_prior,
                 cen2_prior,
                 M1_prior,
                 M2_prior,
                 T_prior,
                 F_prior):

        self.cen1_prior=cen1_prior
        self.cen2_prior=cen1_prior
        self.M1_prior=M1_prior
        self.M2_prior=M2_prior
        self.T_prior=T_prior

        if isinstance(F_prior,list):
            self.nband=len(F_prior)
        else:
            self.nband=1
            F_prior=[F_prior]

        self.F_priors=F_prior

    def get_prob_scalar(self, pars, **keys):
        """
        probability for scalar input (meaning one point)
        """

        lnp = self.get_lnprob_scalar(pars, **keys)
        p = exp(lnp)
        return p

    def get_lnprob_scalar(self, pars, **keys):
        """
        log probability for scalar input (meaning one point)
        """

        lnp  = self.cen1_prior.get_lnprob_scalar(pars[0])
        lnp += self.cen1_prior.get_lnprob_scalar(pars[1])
        lnp += self.M1_prior.get_lnprob_scalar(pars[2])
        lnp += self.M2_prior.get_lnprob_scalar(pars[3])
        lnp += self.T_prior.get_lnprob_scalar(pars[4])

        for i, F_prior in enumerate(self.F_priors):
            lnp += F_prior.get_lnprob_scalar(pars[5+i])

        return lnp

    def fill_fdiff(self, pars, fdiff, **keys):
        """
        set sqrt(-2ln(p)) ~ (model-data)/err
        """
        index=0


        fdiff[index] = self.cen1_prior.get_lnprob_scalar(pars[0])
        index += 1
        fdiff[index] = self.cen2_prior.get_lnprob_scalar(pars[0])
        index += 1

        fdiff[index] = self.M1_prior.get_lnprob_scalar(pars[2])
        index += 1
        fdiff[index] = self.M2_prior.get_lnprob_scalar(pars[3])
        index += 1
        fdiff[index] =  self.T_prior.get_lnprob_scalar(pars[4])
        index += 1

        for i in xrange(self.nband):
            F_prior=self.F_priors[i]
            fdiff[index] = F_prior.get_lnprob_scalar(pars[5+i], **keys)
            index += 1

        chi2 = -2*fdiff[0:index]
        chi2.clip(min=0.0, max=None, out=chi2)
        fdiff[0:index] = sqrt(chi2)

        return index


    def get_prob_array(self, pars, **keys):
        """
        probability for array input [N,ndims]
        """

        lnp = self.get_lnprob_array(pars, **keys)
        p = exp(lnp)

        return p

    def get_lnprob_array(self, pars, **keys):
        """
        log probability for array input [N,ndims]
        """

        lnp  = self.cen1_prior.get_lnprob_array2d(pars[:,0])
        lnp += self.cen2_prior.get_lnprob_array2d(pars[:,1])
        lnp += self.M1_prior.get_lnprob_array(pars[:,2])
        lnp += self.M2_prior.get_lnprob_array(pars[:,3])
        lnp += self.T_prior.get_lnprob_array(pars[:,4])

        for i in xrange(self.nband):
            F_prior=self.F_priors[i]
            lnp += F_prior.get_lnprob_array(pars[:,5+i])

        return lnp

    def __repr__(self):
        reps=[]
        reps += [str(self.cen1_prior),
                 str(self.cen2_prior),
                 str(self.M1_prior),
                 str(self.M2_prior),
                 str(self.T_prior)]

        for p in self.F_priors:
            reps.append( str(p) )

        rep='\n'.join(reps)
        return rep


