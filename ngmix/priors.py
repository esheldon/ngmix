"""
Convention is that all priors should have peak ln(prob)==0. This
helps use in priors for LM fitting

I haven't forced the max prob to be 1.0 yet, but should

"""
from __future__ import print_function

import math

import numpy
from numpy import where, array, exp, log, log10, sqrt, cos, sin, zeros, ones, diag
from numpy import pi
from numpy.random import random as randu
from numpy.random import randn

from . import gmix
from .gexceptions import GMixRangeError

from . import shape

LOWVAL=-9999.0e47
BIGVAL =9999.0e47

class GPriorBase(object):
    """
    This is the base class.  You need to over-ride a few of
    the functions, see below
    """
    def __init__(self, pars):
        self.pars = array(pars, dtype='f8')

        # sub-class may want to over-ride this, see GPriorM
        self.gmax=1.0

    def get_lnprob_scalar2d(self, g1, g2):
        """
        Get the 2d log prob
        """
        raise RuntimeError("over-ride me")

    def get_lnprob_array2d(self, g1arr, g2arr):
        """
        Get the 2d prior for the array inputs
        """

        g1arr=array(g1arr, dtype='f8', copy=False)
        g2arr=array(g2arr, dtype='f8', copy=False)

        output=numpy.zeros(g1arr.size)
        self.fill_lnprob_array2d(g1arr, g2arr, output)
        return output


    def get_prob_scalar2d(self, g1, g2):
        """
        Get the 2d prob
        """
        raise RuntimeError("over-ride me")

    def get_prob_array2d(self, g1arr, g2arr):
        """
        Get the 2d prior for the array inputs
        """

        g1arr=array(g1arr, dtype='f8', copy=False)
        g2arr=array(g2arr, dtype='f8', copy=False)

        output=numpy.zeros(g1arr.size)
        self.fill_prob_array2d(g1arr, g2arr, output)
        return output

    def get_prob_scalar1d(self, g):
        """
        Get the 1d prob
        """
        raise RuntimeError("over-ride me")

    def get_prob_array1d(self, garr):
        """
        Get the 1d prior for the array inputs
        """

        garr=array(garr, dtype='f8', copy=False)

        output=numpy.zeros(garr.size)
        self.fill_prob_array1d(garr, output)
        return output


    def dbyg1_array(self, g1, g2, h=1.e-6):
        """
        Derivative with respect to g1 at the input g1,g2 location
        Used for lensfit.

        Uses central difference and a small enough step size
        to use just two points

        """
        h2=1./(2.*h)

        ff = self.get_prob_array2d(g1+h, g2)
        fb = self.get_prob_array2d(g1-h, g2)

        return (ff - fb)*h2

    def dbyg2_array(self, g1, g2, h=1.e-6):
        """
        Derivative with respect to g2 at the input g1,g2 location

        Uses central difference and a small enough step size
        to use just two points

        """
        h2=1./(2.*h)

        ff = self.get_prob_array2d(g1, g2+h)
        fb = self.get_prob_array2d(g1, g2-h)

        return (ff - fb)*h2

 
    def get_pqr_num(self, g1in, g2in, s1=0.0, s2=0.0, h=1.e-6):
        """
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
        if numpy.isscalar(g1in):
            isscalar=True
        else:
            isscalar=False

        g1 = array(g1in, dtype='f8', ndmin=1, copy=False)
        g2 = array(g2in, dtype='f8', ndmin=1, copy=False)
        h2=1./(2.*h)
        hsq=1./h**2

        P=self.get_pj(g1, g2, s1, s2)

        Q1_p   = self.get_pj(g1, g2, s1+h, s2)
        Q1_m   = self.get_pj(g1, g2, s1-h, s2)
        Q2_p   = self.get_pj(g1, g2, s1,   s2+h)
        Q2_m   = self.get_pj(g1, g2, s1,   s2-h)

        R12_pp = self.get_pj(g1, g2, s1+h, s2+h)
        R12_mm = self.get_pj(g1, g2, s1-h, s2-h)

        Q1 = (Q1_p - Q1_m)*h2
        Q2 = (Q2_p - Q2_m)*h2

        R11 = (Q1_p - 2*P + Q1_m)*hsq
        R22 = (Q2_p - 2*P + Q2_m)*hsq
        R12 = (R12_pp - Q1_p - Q2_p + 2*P - Q1_m - Q2_m + R12_mm)*hsq*0.5

        np=g1.size
        Q = numpy.zeros( (np,2) )
        R = numpy.zeros( (np,2,2) )

        Q[:,0] = Q1
        Q[:,1] = Q2
        R[:,0,0] = R11
        R[:,0,1] = R12
        R[:,1,0] = R12
        R[:,1,1] = R22

        if isscalar:
            P = P[0]
            Q = Q[0,:]
            R = R[0,:,:]

        return P, Q, R

    def get_pj(self, g1, g2, s1, s2):
        """
        PJ = p(g,-shear)*jacob

        where jacob is d(es)/d(eo) and
        es=eo(+)(-g)
        """

        # note sending negative shear to jacob
        s1m=-s1
        s2m=-s2
        J=shape.dgs_by_dgo_jacob(g1, g2, s1m, s2m)

        # evaluating at negative shear
        g1new,g2new=shape.shear_reduced(g1, g2, s1m, s2m)
        if numpy.isscalar(g1):
            P=self.get_prob_scalar2d(g1new,g2new)
        else:
            P=self.get_prob_array2d(g1new,g2new)

        return P*J

    def sample2d_pj(self, nrand, s1, s2):
        """
        Get random g1,g2 values from an approximate
        sheared distribution

        parameters
        ----------
        nrand: int
            Number to generate
        """

        maxval2d = self.get_prob_scalar2d(0.0,0.0)
        g1,g2=numpy.zeros(nrand),numpy.zeros(nrand)

        ngood=0
        nleft=nrand
        while ngood < nrand:

            # generate on cube [-1,1,h]
            g1rand=srandu(nleft)
            g2rand=srandu(nleft)

            # a bit of padding since we are modifying the distribution
            fac=1.3
            h = fac*maxval2d*randu(nleft)

            pjvals = self.get_pj(g1rand,g2rand,s1,s2)
            
            w,=numpy.where(h < pjvals)
            if w.size > 0:
                g1[ngood:ngood+w.size] = g1rand[w]
                g2[ngood:ngood+w.size] = g2rand[w]
                ngood += w.size
                nleft -= w.size
   
        return g1,g2



    def sample1d(self, nrand):
        """
        Get random |g| from the 1d distribution

        Set self.gmax appropriately

        parameters
        ----------
        nrand: int
            Number to generate
        """

        if not hasattr(self,'maxval1d'):
            self.set_maxval1d()

        maxval1d = self.maxval1d * 1.1

        # don't go right up to the end
        #gmax=self.gmax
        gmax=self.gmax - 1.0e-4

        g = numpy.zeros(nrand)

        ngood=0
        nleft=nrand
        while ngood < nrand:

            # generate total g in [0,gmax)
            grand = gmax*randu(nleft)

            # now the height from [0,maxval)
            h = maxval1d*randu(nleft)

            pvals = self.get_prob_array1d(grand)

            w,=numpy.where(h < pvals)
            if w.size > 0:
                g[ngood:ngood+w.size] = grand[w]
                ngood += w.size
                nleft -= w.size
   
        return g


    def sample2d(self, nrand):
        """
        Get random g1,g2 values by first drawing
        from the 1-d distribution

        parameters
        ----------
        nrand: int
            Number to generate
        """

        grand=self.sample1d(nrand)
        theta = randu(nrand)*2*numpy.pi
        twotheta = 2*theta
        g1rand = grand*numpy.cos(twotheta)
        g2rand = grand*numpy.sin(twotheta)
        return g1rand, g2rand

    def sample2d_brute(self, nrand):
        """
        Get random g1,g2 values using 2-d brute
        force method

        parameters
        ----------
        nrand: int
            Number to generate
        """

        maxval2d = self.get_prob_scalar2d(0.0,0.0)
        g1,g2=numpy.zeros(nrand),numpy.zeros(nrand)

        ngood=0
        nleft=nrand
        while ngood < nrand:
            
            # generate on cube [-1,1,h]
            g1rand=srandu(nleft)
            g2rand=srandu(nleft)

            # a bit of padding since we are modifying the distribution
            h = maxval2d*randu(nleft)

            vals = self.get_prob_array2d(g1rand,g2rand)

            w,=numpy.where(h < vals)
            if w.size > 0:
                g1[ngood:ngood+w.size] = g1rand[w]
                g2[ngood:ngood+w.size] = g2rand[w]
                ngood += w.size
                nleft -= w.size
   
        return g1,g2


    def set_maxval1d(self):
        """
        Use a simple minimizer to find the max value of the 1d 
        distribution
        """
        import scipy.optimize

        (minvalx, fval, iterations, fcalls, warnflag) \
                = scipy.optimize.fmin(self.get_prob_scalar1d_neg,
                                      0.1,
                                      full_output=True, 
                                      disp=False)
        if warnflag != 0:
            raise ValueError("failed to find min: warnflag %d" % warnflag)
        self.maxval1d = -fval
        self.maxval1d_loc = minvalx

    def get_prob_scalar1d_neg(self, g, *args):
        """
        So we can use the minimizer
        """
        return -self.get_prob_scalar1d(g)

    def test_pqr_shear_recovery(self,
                                smin,
                                smax,
                                nshear,
                                doring=True,
                                npair=10000,
                                shear_expand=None,
                                h=1.e-6,
                                show=False,
                                eps=None):
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
        shear1_meas_err=numpy.zeros(nshear)
        shear2_meas_err=numpy.zeros(nshear)
        
        # _te means expanded around truth
        shear1_meas_te=numpy.zeros(nshear)
        shear2_meas_te=numpy.zeros(nshear)
        shear1_meas_err_te=numpy.zeros(nshear)
        shear2_meas_err_te=numpy.zeros(nshear)
 
        theta=numpy.pi/2.0
        twotheta = 2.0*theta
        cos2angle = numpy.cos(twotheta)
        sin2angle = numpy.sin(twotheta)

        g1=numpy.zeros(npair*2)
        g2=numpy.zeros(npair*2)
        for ishear in xrange(nshear):
            print("-"*70)
            s1=shear1_true[ishear]
            s2=shear2_true[ishear]

            if doring:

                g1[0:npair],g2[0:npair] = self.sample2d(npair)
                g1[npair:] =  g1[0:npair]*cos2angle + g2[0:npair]*sin2angle
                g2[npair:] = -g1[0:npair]*sin2angle + g2[0:npair]*cos2angle

                g1s, g2s = shear_reduced(g1, g2, s1, s2)

                P,Q,R=self.get_pqr_num(g1s, g2s, h=h)

                if shear_expand is None:
                    s1expand=s1
                    s2expand=s2
                else:
                    s1expand=shear_expand[0]
                    s2expand=shear_expand[1]
                P_te,Q_te,R_te=self.get_pqr_num(g1s, g2s,
                                                s1=s1expand, s2=s2expand, h=h)

                # watch for zeros in P for some priors
                first=numpy.arange(npair)
                second=first+npair

                Pscale = P/P.max()
                print("Pmin:",P.min(),"Pmax:",P.max())
                print("Pscale_min: %g" % Pscale.min())
                #w,=numpy.where( (Pscale[first] > 1.0e-6) & (Pscale[second] > 1.0e-6) )
                w,=numpy.where( (Pscale[first] > 0.) & (Pscale[second] > 0.) )
                w=numpy.concatenate( (first[w], second[w] ) )
                print("kept: %d/%d" % (w.size, npair*2))
                P=P[w]
                Q=Q[w,:]
                R=R[w,:,:]

                Pscale = P_te/P_te.max()
                print("P_te_min:",P_te.min(),"P_te_max:",P_te.max())
                print("P_te_scale_min: %g" % Pscale.min())
                w,=numpy.where( (Pscale[first] > 1.0e-6) & (Pscale[second] > 1.0e-6) )
                w=numpy.concatenate( (first[w], second[w] ) )
                print("kept: %d/%d" % (w.size, npair*2))
                P_te=P_te[w]
                Q_te=Q_te[w,:]
                R_te=R_te[w,:,:]
            else:
                ntot=npair
                g1,g2 = self.sample2d(ntot)

                g1s, g2s = shear_reduced(g1, g2, s1, s2)

                P,Q,R=self.get_pqr_num(g1s, g2s, h=h)
                P_te,Q_te,R_te=self.get_pqr_num(g1s, g2s, s1=s1, s2=s2, h=h)

                w,=numpy.where(P > 0.0)
                print("kept: %d/%d" % (w.size, ntot))
                P=P[w]
                Q=Q[w,:]
                R=R[w,:,:]

                w,=numpy.where(P_te > 0.0)
                print("kept: %d/%d" % (w.size, ntot))
                P_te=P_te[w]
                Q_te=Q_te[w,:]
                R_te=R_te[w,:,:]

            g1g2, C = lensing.pqr.get_pqr_shear(P,Q,R)
            g1g2_te, C_te = lensing.pqr.get_pqr_shear(P_te,Q_te,R_te)

            g1g2_te[0] += s1
            g1g2_te[0] += s2
            shear1_meas[ishear] = g1g2[0]
            shear2_meas[ishear] = g1g2[1]

            shear1_meas_te[ishear] = g1g2_te[0]
            shear2_meas_te[ishear] = g1g2_te[1]

            if doring:
                mess='true: %.6f,%.6f meas: %.6f,%.6f expand true: %.6f,%.6f'
                print(mess % (s1,s2,g1g2[0],g1g2[1],g1g2_te[0],g1g2_te[1]))
            else:
                mess='true: %.6f meas: %.6f +/- %.6f expand true: %.6f +/- %.6f'
                err=numpy.sqrt(C[0,0])
                err_te=numpy.sqrt(C_te[0,0])

                shear1_meas_err[ishear] = err
                shear1_meas_err_te[ishear] = err_te
                print(mess % (s1,g1g2[0],err,g1g2_te[0],err_te))

        fracdiff=shear1_meas/shear1_true-1
        fracdiff_err=shear1_meas_err/shear1_true
        fracdiff_te=shear1_meas_te/shear1_true-1
        fracdiff_err_te=shear1_meas_err_te/shear1_true

        if eps or show:
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

            if not doring:
                perr=biggles.SymmetricErrorBarsY(shear1_true, fracdiff, fracdiff_err,
                                                 color='blue')
                perr_te=biggles.SymmetricErrorBarsY(shear1_true, fracdiff_te, fracdiff_err_te,
                                                 color='blue')
                plt.add(perr,perr_te)

            coeffs=numpy.polyfit(shear1_true, fracdiff, 2)
            poly=numpy.poly1d(coeffs)

            curve=biggles.Curve(shear1_true, poly(shear1_true), type='solid',
                                color='black')
            curve.label=r'$\Delta g/g = %.1f g^2$' % coeffs[0]
            plt.add(curve)

            plt.add( biggles.PlotKey(0.1, 0.9, [pts,pts_te,curve], halign='left') )

            if eps:
                print('writing:',eps)
                plt.write_eps(eps)

            if show:
                plt.show()
            print(poly)


    def test_pqr_shear_recovery_iter(self, shear, npair, niter,
                                     shear_expand_start=None,
                                     doring=True,
                                     h=1.e-6):
        """
        Test how well we recover the shear with no noise and
        iteration on the measured shear

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
 
        theta=numpy.pi/2.0
        twotheta = 2.0*theta
        cos2angle = numpy.cos(twotheta)
        sin2angle = numpy.sin(twotheta)

        g1=numpy.zeros(npair*2)
        g2=numpy.zeros(npair*2)

        if shear_expand_start is not None:
            shear_expand=array(shear_expand_start)
        else:
            shear_expand=array([0.0, 0.0])

        for i in xrange(niter):

            if doring:

                g1[0:npair],g2[0:npair] = self.sample2d(npair)
                g1[npair:] =  g1[0:npair]*cos2angle + g2[0:npair]*sin2angle
                g2[npair:] = -g1[0:npair]*sin2angle + g2[0:npair]*cos2angle

            else:
                g1,g2 = self.sample2d(npair*2)


            g1s, g2s = shear_reduced(g1, g2, shear[0], shear[1])

            P,Q,R=self.get_pqr_num(g1s, g2s,
                                   s1=shear_expand[0],
                                   s2=shear_expand[1])

            shmeas, Cmeas = lensing.pqr.get_pqr_shear(P,Q,R)

            shmeas += shear_expand

            err=sqrt(diag(Cmeas))
            frac=shmeas/shear-1
            frac_err=err/shear

            if doring:
                mess='iter: %d true: %.6f, %.6f meas: %.6f, %.6f frac: %.6f %.6f'
                print(mess % (i+1,shear[0],shear[1],shmeas[0],shmeas[1],frac[0],frac[1]))
            else:
                mess='iter: %d true: %.6f,%.6f meas: %.6f +/- %.6f %.6f +/- %.6f frac: %.6f +/- %.6f %.6f +/- %.6f'
                print(mess % (i+1,shear[0],shear[1],shmeas[0],err[0],shmeas[1],err[1],
                              frac[0],frac_err[0],frac[1],frac_err[1]))

            shear_expand=shmeas

    def test_lensfit_shear_recovery(self,
                                    smin,
                                    smax,
                                    nshear,
                                    npair=10000,
                                    nsample=1000,
                                    sigma=0.1,
                                    h=1.e-6,
                                    show=False,
                                    eps=None):
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
        from . import lensfit

        # only shear in g1
        shear_true=zeros( (nshear,2) )
        shear_true[:,0]=numpy.linspace(smin, smax, nshear)/sqrt(2)
        shear_true[:,1]=numpy.linspace(smin, smax, nshear)/sqrt(2)


        shear_meas=numpy.zeros( (nshear,2) )
        shear_meas_err=numpy.zeros( (nshear,2) )

        fracdiff=numpy.zeros( (nshear,2) )
 
        theta=numpy.pi/2.0
        twotheta = 2.0*theta
        cos2angle = numpy.cos(twotheta)
        sin2angle = numpy.sin(twotheta)

        nshape = npair*2
        g1=numpy.zeros(nshape)
        g2=numpy.zeros(nshape)

        g_mean=zeros( (nshape,2) )
        g_sens=zeros( (nshape,2) )

        # for holding likelihood samples
        gsamples=numpy.zeros( (nsample, 2) )

        for ishear in xrange(nshear):
            print("-"*70)

            g1[0:npair],g2[0:npair] = self.sample2d(npair)
            g1[npair:] =  g1[0:npair]*cos2angle + g2[0:npair]*sin2angle
            g2[npair:] = -g1[0:npair]*sin2angle + g2[0:npair]*cos2angle

            g1s, g2s = shear_reduced(g1, g2, shear_true[ishear,0], shear_true[ishear,1])

            # add noise
            for i in xrange(nshape):
                #import esutil as eu
                glike=TruncatedGauss2D(g1s[i],g2s[i],sigma,sigma,1.0)
                rg1,rg2=glike.sample(nsample)
                #rg=sqrt(rg1**2 + rg2**2)
                #eu.plotting.bhist(rg1,binsize=0.01,title='g1')
                #eu.plotting.bhist(rg2,binsize=0.01,title='g2')
                #key=raw_input('hit a key: ')
                #if key.lower()=='q':
                #    stop

                gsamples[:,0] = rg1
                gsamples[:,1] = rg2
                lsobj = lensfit.LensfitSensitivity(gsamples, self, h=h)
                g_sens[i,:] = lsobj.get_g_sens()
                g_mean[i,:] = lsobj.get_g_mean()


            print("g_mean:     ",g_mean.mean(axis=0))
            print("g_sens mean:",g_sens.mean(axis=0))

            shear_meas[ishear,:] = g_mean.sum(axis=0)/g_sens.sum(axis=0)
            fracdiff[ishear,:] = shear_meas[ishear,:]/shear_true[ishear,:]-1

            mess='true: %.6f,%.6f meas: %.6f,%.6f frac: %.6g,%.6g'
            mess=mess % (shear_true[ishear,0],shear_true[ishear,1],
                         shear_meas[ishear,0],shear_meas[ishear,1],
                         fracdiff[ishear,0],fracdiff[ishear,1])
            print(mess)



class GPriorBA(GPriorBase):
    """
    g prior from Bernstein & Armstrong 2013

    automatically has max lnprob 0 and max prob 1
    """

    def __init__(self, sigma):
        """
        pars are scalar gsigma from B&A 
        """
        self.sigma   = sigma
        self.sig2 = self.sigma**2
        self.sig4 = self.sigma**4
        self.sig2inv = 1./self.sig2
        self.sig4inv = 1./self.sig4

        self.gmax=1.0

        self.h=1.e-6
        self.hhalf=0.5*self.h
        self.hinv = 1./self.h

    def get_lnprob_scalar2d(self, g1, g2):
        """
        Get the 2d log prob for the input g value
        """
        gsq = g1*g1 + g2*g2
        omgsq = 1.0 - gsq
        if omgsq <= 0.0:
            raise GMixRangeError("g^2 too big: %s" % gsq)
        lnp = 2*log(omgsq) -0.5*gsq*self.sig2inv
        return lnp

    def get_prob_scalar2d(self, g1, g2):
        """
        Get the 2d prob for the input g value
        (1-g^2)^2 * exp(-0.5*g^2/sigma^2)
        """
        p=0.0

        gsq=g1*g1 + g2*g2
        omgsq=1.0-gsq
        if omgsq > 0.0:
            omgsq *= omgsq

            expval = exp(-0.5*gsq*self.sig2inv)

            p = omgsq*expval
        return p

    def fill_prob_array2d(self, g1arr, g2arr, output):
        """
        Fill the output with the 2d prob for the input g value
        """

        gsq=g1arr*g1arr + g2arr*g2arr
        omgsq=1.0-gsq
        
        w,=where(omgsq > 0.0)
        if w.size > 0:
            omgsq *= omgsq

            expval = exp(-0.5*gsq[w]*self.sig2inv)

            output[w] = omgsq[w]*expval


    def fill_lnprob_array2d(self, g1arr, g2arr, output):
        """
        Fill the output with the 2d prob for the input g value
        """

        gsq=g1arr*g1arr + g2arr*g2arr
        omgsq = 1.0 - gsq
        w,=where(omgsq > 0.0)
        if w.size > 0:
            output[w] = 2*log(omgsq[w]) -0.5*gsq[w]*self.sig2inv


    def get_prob_scalar1d(self, g):
        """
        Get the 1d prior for the input |g| value
        """
        p=0.0

        gsq=g*g
        omgsq=1.0-gsq
        
        if omgsq > 0.0:
            omgsq *= omgsq

            expval = numpy.exp(-0.5*gsq*self.sig2inv)

            p = omgsq*expval

            p *= 2*numpy.pi*g
        return p

    def fill_prob_array1d(self, g, output):
        """
        Fill the output with the 1d prior for the input g value
        """

        gsq=g*g
        omgsq=1.0-gsq
        
        w,=where(omgsq > 0.0)
        if w.size > 0:
            omgsq *= omgsq

            expval = exp(-0.5*gsq[w]*self.sig2inv)

            output[w] = omgsq[w]*expval

            output[w] *= 2*numpy.pi*g

    def dbyg1_scalar(self, g1, g2):
        """
        Derivative with respect to g1 at the input g1,g2 location
        Used for lensfit.

        Uses central difference and a small enough step size
        to use just two points
        """
        ff = self.get_prob_scalar2d(g1+self.hhalf, g2)
        fb = self.get_prob_scalar2d(g1-self.hhalf, g2)

        return (ff - fb)*self.hinv

    def dbyg2_scalar(self, g1, g2):
        """
        Derivative with respect to g2 at the input g1,g2 location
        Used for lensfit.

        Uses central difference and a small enough step size
        to use just two points
        """
        ff = self.get_prob_scalar2d(g1,g2+self.hhalf)
        fb = self.get_prob_scalar2d(g1,g2-self.hhalf)

        return (ff - fb)*self.hinv


    def get_pqr(self, g1in, g2in):
        """
        Evaluate 
            P
            Q
            R
        From Bernstein & Armstrong

        P is this prior times the jacobian at shear==0

        Q is the gradient of P*J evaluated at shear==0

            [ d(P*J)/dg1, d(P*J)/dg2]_{g=0}

        R is grad of grad of P*J at shear==0
            [ d(P*J)/dg1dg1  d(P*J)/dg1dg2 ]
            [ d(P*J)/dg1dg2  d(P*J)/dg2dg2 ]_{g=0}
        """

        if numpy.isscalar(g1in):
            isscalar=True
        else:
            isscalar=False

        g1 = array(g1in, dtype='f8', ndmin=1, copy=False)
        g2 = array(g2in, dtype='f8', ndmin=1, copy=False)

        # these are the same for expanding about zero shear
        #P=self.get_pj(g1, g2, 0.0, 0.0)
        P=self.get_prob_array2d(g1, g2)

        sig2 = self.sig2
        sig4 = self.sig4
        sig2inv = self.sig2inv
        sig4inv = self.sig4inv

        gsq = g1**2 + g2**2
        omgsq = 1. - gsq

        fac = numpy.exp(-0.5*gsq*sig2inv)*omgsq**2

        Qf = fac*(omgsq + 8*sig2)*sig2inv

        Q1 = g1*Qf
        Q2 = g2*Qf

        R11 = (fac * (g1**6 + g1**4*(-2 + 2*g2**2 - 19*sig2) + (1 + g2**2)*sig2*(-1 + g2**2 - 8*sig2) + g1**2*(1 + g2**4 + 20*sig2 + 72*sig4 - 2*g2**2*(1 + 9*sig2))))*sig4inv
        R22 = (fac * (g2**6 + g2**4*(-2 + 2*g1**2 - 19*sig2) + (1 + g1**2)*sig2*(-1 + g1**2 - 8*sig2) + g2**2*(1 + g1**4 + 20*sig2 + 72*sig4 - 2*g1**2*(1 + 9*sig2))))*sig4inv

        R12 = fac * g1*g2 * (80 + omgsq**2*sig4inv + 20*omgsq*sig2inv)

        np=g1.size
        Q = numpy.zeros( (np,2) )
        R = numpy.zeros( (np,2,2) )

        Q[:,0] = Q1
        Q[:,1] = Q2
        R[:,0,0] = R11
        R[:,0,1] = R12
        R[:,1,0] = R12
        R[:,1,1] = R22

        if isscalar:
            P = P[0]
            Q = Q[0,:]
            R = R[0,:,:]

        return P, Q, R

    def get_pqr_expand(self, g1in, g2in, s1, s2):
        """
        Evaluate 
            P
            Q
            R
        From Bernstein & Armstrong

        P is this prior times the jacobian at shear==0

        Q is the gradient of P*J evaluated at shear==0

            [ d(P*J)/dg1, d(P*J)/dg2]_{g=0}

        R is grad of grad of P*J at shear==0
            [ d(P*J)/dg1dg1  d(P*J)/dg1dg2 ]
            [ d(P*J)/dg1dg2  d(P*J)/dg2dg2 ]_{g=0}
        """
        from numpy import exp

        if numpy.isscalar(g1in):
            isscalar=True
        else:
            isscalar=False

        g1 = array(g1in, dtype='f8', ndmin=1, copy=False)
        g2 = array(g2in, dtype='f8', ndmin=1, copy=False)

        P=self.get_pj(g1, g2, s1, s2)

        s1sq = s1**2
        s2sq = s2**2

        gsq = g1**2 + g2**2
        gsqmo = gsq - 1.0
        ssq = s1**2 + s2**2
        ssqmo = ssq - 1.0

        s2sqmo = s2sq - 1.0

        sigma=self.sigma

        sdet = (1 - 2*g1*s1 - 2*g2*s2 + g1**2*(s1**2 + s2**2) + g2**2*(s1**2 + s2**2))
        sdet2 = sdet**2
        sdet3 = sdet**3
        sdet4 = sdet**4
        sdet6 = sdet**6
        sdet7 = sdet**7

        expfac = exp((g1**2 + g2**2 - 2*g1*s1 + s1**2 - 2*g2*s2 + s2**2)/(2.*sdet*sigma**2))

        bigfac1 = (s2sqmo + 4*s1**4*sigma**2 + 4*s2**4*sigma**2 + s1**2*(1 + 8*s2**2*sigma**2))
        bigfac2 = (s2sqmo + 8*s2**2*sigma**2 + s1**2*(1 + 8*sigma**2))
        
        bigfac3 = (g2**2*s2 + (1 + g1**2 - 2*g1*s1)*s2 + g2*(-1 + s1**2 - s2**2))
        bigfac4 = (g1**2*s1 + s1*(1 + g2**2 - 2*g2*s2) + g1*(-1 - s1**2 + s2**2))

        Q_subfac = (ssqmo + 8*s1**2*sigma**2 + 8*s2**2*sigma**2)
        Qfac = gsqmo**2*ssqmo**3*((1 - s1**2 - s2**2 + 8*sigma**2 - 16*g1*s1*sigma**2 - 16*g2*s2*sigma**2 + g2**2*Q_subfac + g1**2*bigfac2))/ (expfac* sdet6*sigma**2)

        Q1 = bigfac4*Qfac
        Q2 = bigfac3*Qfac


        R11_fac1 = (5*s1**6 + 3*s1**4*(-7 + 3*s2**2) + 3*s1**2*(6 - 6*s2**2 + s2**4) - s2**2*(2 - 3*s2**2 + s2**4))
        R11_fac2 = (9 - 15*s2**2 + 6*s2**4 + s1**2*(-7 + 6*s2**2))

        R11 = ( (4*gsqmo**2*ssqmo**2*(-1 + 3*s1sq + s2**2 - 4*g2*s2*(-1 + 3*s1**2 + s2**2) - 2*g2**2*(1 - 9*s1**2 + 6*s1**4 + s2**2 - 2*s2**4) + 
            4*g2**3*s2*(1 + 6*s1**4 - s2**2 + s1**2*(-9 + 6*s2**2)) - 4*g1**3*s1*R11_fac2 + R11_fac1*(g1**4 + g2**4) - 
            4*g1*s1*(3 - s1**2 - 3*s2**2 + 2*g2*s2*(-3 + s1**2 + 3*s2**2) + g2**2*R11_fac2) + 
            2*g1**2*(5*g2**2*s1**6 - s2sqmo*(9 + 2*g2*s2 - 10*s2**2 + g2**2*s2**2*(-2 + s2**2)) + 3*s1**4*(-2 + 4*g2*s2 + g2**2*(-7 + 3*s2**2)) + 
               3*s1**2*(1 + g2*(-6*s2 + 4*s2**3) + g2**2*(6 - 6*s2**2 + s2**4)))))/(expfac*sdet6)
            + (gsqmo**2*ssqmo**2*((4*(-1 + 3*s1**2 + s2**2))/expfac+ (8*gsqmo*s1*ssqmo*bigfac4)/ (expfac* sdet2*sigma**2) + 
            (gsqmo*ssqmo**2*(gsqmo*bigfac4**2 - sdet*(-2*g1**3*s1*(3 + s1**2 - 3*s2**2) - 2*g1*g2**2*s1*(3 + s1**2 - 3*s2**2) + g1**4*(3*s1**2 - s2**2) + (1 + g2**2 - 2*g2*s2)*(-1 + 2*g2*s2 + g2**2*(3*s1**2 - s2**2)) + 
                    g1**2*(3 - 5*s2**2 - 2*g2**2*s2**2 + s1**2*(3 + 6*g2**2 - 6*g2*s2) + 2*g2*(s2 + s2**3)))*sigma**2))/
             (expfac*
               sdet4*sigma**4)))/sdet4 - 
       (8*gsqmo**2*ssqmo**2*(-2*g1*s2sqmo + g1**2*s1*(-2 + s1**2 + s2**2) + s1*(-1 + 2*g2*s2 + g2**2*(-2 + s1**2 + s2**2)))*
          (-(g1**3*(-s2sqmo**2 + 16*s1**2*s2**2*sigma**2 + s1**4*(1 + 16*sigma**2))) + g1**4*s1*bigfac1 + 
            g1*(s1**4 - s2sqmo**2 - 16*s1**2*sigma**2 + 32*g2*s1**2*s2*sigma**2 - g2**2*(-s2sqmo**2 + 16*s1**2*s2**2*sigma**2 + s1**4*(1 + 16*sigma**2))) + 
            2*g1**2*s1*(4*(3*s1**2 + s2**2)*sigma**2 - g2*s2*bigfac2 + 
               g2**2*bigfac1) + 
            s1*(1 - s1**2 - s2**2 + 4*sigma**2 + 8*g2**2*(s1**2 + 3*s2**2)*sigma**2 + 2*g2*s2*(ssqmo - 8*sigma**2) - 
               2*g2**3*s2*bigfac2 + g2**4*bigfac1)))/
        (expfac*
          sdet7*sigma**2) )



        R22_fac1 = (s1**6 - 18*s2**2 + 21*s2**4 - 5*s2**6 - 3*s1**4*(1 + s2**2) + s1**2*(2 + 18*s2**2 - 9*s2**4))
        R22_fac2 = (1 - 9*s2**2 + 6*s2**4 + s1**2*(-1 + 6*s2**2))

        R22 = ( (-4*gsqmo**2*ssqmo**2*(1 - s1**2 - 3*s2**2 - 4*g2*s2*(-3 + 3*s1**2 + s2**2) - 2*g2**2*(9 - 19*s1**2 + 10*s1**4 + 3*s2**2 - 6*s2**4) + 
            4*g2**3*s2*(9 + 6*s1**4 - 7*s2**2 + 3*s1**2*(-5 + 2*s2**2)) - 4*g1**3*s1*R22_fac2 + 
            R22_fac1*(g1**4 + g2**4) - 4*g1*s1*(1 - s1**2 - 3*s2**2 - 2*g2*s2*(-3 + 3*s1**2 + s2**2) + g2**2*R22_fac2) + 
            2*g1**2*(1 + g2**2*s1**6 - 9*s2**2 + 6*s2**4 - 2*g2*s2*(-9 + 7*s2**2) + g2**2*(-18*s2**2 + 21*s2**4 - 5*s2**6) - s1**4*(2 - 12*g2*s2 + 3*g2**2*(1 + s2**2)) + 
               s1**2*(1 + 6*g2*s2*(-5 + 2*s2**2) + g2**2*(2 + 18*s2**2 - 9*s2**4)))))/
        (expfac*
          sdet6) + 
       (gsqmo**2*ssqmo**2*((4*(-1 + s1**2 + 3*s2**2))/
             expfac + 
            (8*gsqmo*s2*ssqmo*bigfac3)/
             (expfac*
               sdet2*sigma**2) + 
            (gsqmo*ssqmo**2*(gsqmo*bigfac3**2 + 
                 sdet*
                  (1 + g1**4*(s1**2 - 3*s2**2) + g2**4*(s1**2 - 3*s2**2) + 2*g2**3*s2*(3 - 3*s1**2 + s2**2) - 2*g1**3*(s1 + s1**3 - 3*s1*s2**2) - 2*g1*s1*(2 + g2**2*(1 + s1**2 - 3*s2**2)) + 
                    g2**2*(5*s1**2 - 3*(1 + s2**2)) + g1**2*(1 - 3*s2**2 - 6*g2**2*s2**2 + s1**2*(5 + 2*g2**2 - 6*g2*s2) + 2*g2*s2*(3 + s2**2)))*sigma**2))/
             (expfac*
               sdet4*sigma**4)))/sdet4 - 
       (8*gsqmo**2*ssqmo**2*(-2*g2*(-1 + s1**2) + g2**2*s2*(-2 + s1**2 + s2**2) + s2*(-1 + 2*g1*s1 + g1**2*(-2 + s1**2 + s2**2)))*
          (g2**3*(1 + s1**4 - s2**4*(1 + 16*sigma**2) - 2*s1**2*(1 + 8*s2**2*sigma**2)) + g2**4*s2*bigfac1 + 
            g2*(-1 + 2*s1**2 - s1**4 + s2**4 - 16*s2**2*sigma**2 + 32*g1*s1*s2**2*sigma**2 + g1**2*(1 + s1**4 - s2**4*(1 + 16*sigma**2) - 2*s1**2*(1 + 8*s2**2*sigma**2))) + 
            2*g2**2*s2*(4*(s1**2 + 3*s2**2)*sigma**2 - g1*s1*bigfac2 + 
               g1**2*bigfac1) + 
            s2*(1 - s1**2 - s2**2 + 4*sigma**2 + 8*g1**2*(3*s1**2 + s2**2)*sigma**2 + 2*g1*s1*(ssqmo - 8*sigma**2) - 
               2*g1**3*s1*bigfac2 + g1**4*bigfac1)))/
        (expfac*
          sdet7*sigma**2) )


        R12_fac1 = (2*g1**4*s1*s2 + g1**3*s2*(-2 - 3*s1**2 + s2**2) + g1**2*s1*(4*s2 + 4*g2**2*s2 + g2*(-2 + s1**2 - 3*s2**2)) + g2*s1*(-1 + 4*g2*s2 + 2*g2**3*s2 + g2**2*(-2 + s1**2 - 3*s2**2)) + g1*(2*g2 - s2 + g2**2*s2*(-2 - 3*s1**2 + s2**2)))
        R12 = ( (gsqmo**2*ssqmo**2*(8*s1*s2 + (24*(-g1 + g1**2*s1 + g2**2*s1)*(-g2 + g1**2*s2 + g2**2*s2)*ssqmo**2)/sdet2 - 
           (16*(-g1 + g1**2*s1 + g2**2*s1)*ssqmo*bigfac3)/
            sdet2 - 
           (16*(-g2 + g1**2*s2 + g2**2*s2)*ssqmo*bigfac4)/
            sdet2 + 
           (8*bigfac3*bigfac4)/
            sdet2 - 
           (16*(-g1 + g1**2*s1 + g2**2*s1)*s2*ssqmo)/sdet - 
           (16*s1*(-g2 + g1**2*s2 + g2**2*s2)*ssqmo)/sdet + 
           (16*s1*bigfac3)/sdet + 
           (16*s2*bigfac4)/sdet - 
           (8*ssqmo*R12_fac1)/
            sdet2 + 
           (gsqmo**2*ssqmo**2*bigfac3*bigfac4)/
            (sdet4*sigma**4) - 
           (4*gsqmo*(-g1 + g1**2*s1 + g2**2*s1)*ssqmo**2*bigfac3)/
            (sdet3*sigma**2) - 
           (4*gsqmo*(-g2 + g1**2*s2 + g2**2*s2)*ssqmo**2*bigfac4)/
            (sdet3*sigma**2) + 
           (8*gsqmo*ssqmo*bigfac3*bigfac4)/
            (sdet3*sigma**2) + 
           (4*gsqmo*s1*ssqmo*bigfac3)/
            (sdet2*sigma**2) + 
           (4*gsqmo*s2*ssqmo*bigfac4)/
            (sdet2*sigma**2) - 
           (2*gsqmo*ssqmo**2*R12_fac1)/
            (sdet3*sigma**2)))/(expfac*sdet4) )



        np=g1.size
        Q = numpy.zeros( (np,2) )
        R = numpy.zeros( (np,2,2) )

        Q[:,0] = Q1
        Q[:,1] = Q2
        R[:,0,0] = R11
        R[:,0,1] = R12
        R[:,1,0] = R12
        R[:,1,1] = R22

        if isscalar:
            P = P[0]
            Q = Q[0,:]
            R = R[0,:,:]

        return P, Q, R



#
# these don't have the 2*pi*g in them
def _gprior2d_exp_scalar(A, a, g0sq, gmax, g, gsq):

    if g > gmax:
        return 0.0

    numer = A*(1-numpy.exp( (g-gmax)/a ))
    denom = (1+g)*numpy.sqrt(gsq + g0sq)

    prior=numer/denom

    return prior

def _gprior2d_exp_array(A, a, g0sq, gmax, g, gsq, output):

    w,=where(g < gmax)
    if w.size == 0:
        return

    numer = A*(1-exp( (g[w]-gmax)/a ))
    denom = (1+g)*sqrt(gsq[w] + g0sq)

    output[w]=numer/denom


class FlatPriorBase(object):
    def sample(self, n=None):
        from numpy.random import random as randu
        if n is None:
            is_scalar=True
            n=1
        else:
            is_scalar=False

        rvals = self.minval + (self.maxval-self.minval)*randu(n)

        if is_scalar:
            rvals=rvals[0]

        return rvals

class FlatPrior(FlatPriorBase):
    def __init__(self, minval, maxval):
        self.minval=minval
        self.maxval=maxval

    def get_prob_scalar(self, val):
        retval=1.0
        if val < self.minval or val > self.maxval:
            raise GMixRangeError("value %s out of range: "
                                 "[%s,%s]" % (val, self.minval, self.maxval))
        return retval

    def get_lnprob_scalar(self, val):
        retval=0.0
        if val < self.minval or val > self.maxval:
            raise GMixRangeError("value %s out of range: "
                                 "[%s,%s]" % (val, self.minval, self.maxval))
        return retval
       

class GPriorGreat3Exp(GPriorBase):
    """
    This doesn't fit very well
    """
    def __init__(self, pars):
        self.A = pars[0]
        self.a = pars[1]
        self.g0 = pars[2]
        self.g0_sq = self.g0**2
        self.gmax = 1.0

        self.p_mode=self.get_prob_scalar2d(0.0, 0.0)
        self.lnprob_mode = self.get_lnprob_scalar2d(0.0, 0.0)

    def get_prob_scalar2d(self, g1, g2):
        """
        Get the 2d prob for the input g value
        """

        g = sqrt(g1**2 + g2**2)
        if g >= 1.0:
            return 0.0

        return self._get_prob2d(g)

    def get_prob_array2d(self, g1, g2):
        """
        array input
        """

        n=g1.size
        p=zeros(n)

        g = sqrt(g1**2 + g2**2)
        w,=numpy.where(g < 1.0)
        if w.size > 0:
            p[w] = self._get_prob2d(g[w])

        return p

    def get_prob_scalar1d(self, g):
        """
        Has the 2*pi*g in it
        """

        if g >= 1.0:
            return 0.0

        return 2*numpy.pi*g*self._get_prob2d(g)

    def get_prob_array1d(self, g):
        """
        has 2*pi*g in it array input
        """

        n=g.size
        p=zeros(n)

        w,=numpy.where(g < 1.0)
        if w.size > 0:
            p[w] = 2*numpy.pi*g*self._get_prob2d(g[w])

        return p

    def get_lnprob_scalar2d(self, g1, g2):
        """
        Get the 2d prob for the input g value
        """

        p=self.get_prob_scalar2d(g1,g2)
        if p <= 0.0:
            raise GMixRangeError("g too big: %s" % g)

        lnp=numpy.log(p)
        return lnp

    def _get_prob2d(self, g):
        """
        no error checking

        no 2*pi*g
        """
        gsq = g**2
        omgsq=1.0-gsq
        omgsq_sq = omgsq*omgsq

        A=self.A
        a=self.a
        g0_sq=self.g0_sq
        gmax=self.gmax

        numer = A*(1-exp( (g-gmax)/a )) * omgsq_sq
        denom = (1+g)*sqrt(gsq + g0_sq)

        prob=numer/denom

        return prob


def make_gprior_great3_exp():
    """
    from fitting exp to the real galaxy deep data
    """
    pars=[0.0126484, 1.64042, 0.0554674]
    print("great3 simple joint exp g pars:",pars)
    return GPriorGreat3Exp(pars)

def make_gprior_great3_bdf():
    """
    from fitting bdf to the real galaxy deep data
    """
    pars=[0.014,1.85,0.058]
    print("great3 bdf joint bdf g pars:",pars)
    return GPriorGreat3Exp(pars)




def make_gprior_cosmos_exp():
    """
    From fitting exp to cosmos galaxies
    """
    pars=[560.0, 1.11, 0.052, 0.791]
    return GPriorM(pars)

def make_gprior_cosmos_dev():
    """
    From fitting devto cosmos galaxies
    """
    pars=[560.0, 1.28, 0.088, 0.887]
    return GPriorM(pars)

def make_gprior_cosmos_sersic(type='erf'):
    """
    Fitting to Lackner ellipticities
    A:        0.0250834 +/- 0.00198797
    a:        1.98317 +/- 0.187965
    g0:       0.0793992 +/- 0.00256969
    gmax:     0.706151 +/- 0.00414383
    gsigma:   0.124546 +/- 0.00611789
    """
    if type=='spline':
        return GPriorCosmosSersicSpline()
    else:
        pars=[0.0250834, 1.98317, 0.0793992, 0.706151, 0.124546]
        print("great3 sersic cgc g pars:",pars)
        return GPriorMErf(pars)


class GPriorM(GPriorBase):
    def __init__(self, pars):
        """
        [A, a, g0, gmax]

        From Miller et al.
        """
        self.pars=pars

        self.A    = pars[0]
        self.a    = pars[1]
        self.g0   = pars[2]
        self.g0sq = self.g0**2
        self.gmax = pars[3]

        self.h     = 1.e-6
        self.hhalf = 0.5*self.h
        self.hinv  = 1./self.h

        self.lnprob_mode = self.get_lnprob_scalar2d(0.0, 0.0)

    def get_prob_scalar2d(self, g1, g2):
        """
        Get the 2d prob for the input g value
        """

        gsq = g1**2 + g2**2
        g = numpy.sqrt(gsq)
        return _gprior2d_exp_scalar(self.A, self.a, self.g0sq, self.gmax, g, gsq)

    def get_lnprob_scalar2d(self, g1, g2):
        """
        Get the 2d prob for the input g value
        """

        gsq = g1**2 + g2**2
        g = numpy.sqrt(gsq)
        p=_gprior2d_exp_scalar(self.A, self.a, self.g0sq, self.gmax, g, gsq)

        if p <= 0.0:
            raise GMixRangeError("g too big: %s" % g)

        lnp=numpy.log(p)
        lnp -= self.lnprob_mode
        return lnp

    def fill_prob_array2d(self, g1arr, g2arr, output):
        """
        Fill the output with the 2d prob for the input g value
        """

        _gprior2d_exp_array(self.A, self.a, self.g0sq, self.gmax, g, gsq, output)

    def get_prob_scalar1d(self, g):
        """
        Get the 1d prior for the input |g| value
        """

        gsq=g**2

        p=_gprior2d_exp_scalar(self.A, self.a, self.g0sq, self.gmax, g, gsq)

        p *= 2*numpy.pi*g
        
        return p


    def fill_prob_array1d(self, garr, output):
        """
        Fill the output with the 1d prob for the input g value
        """
        _gprior2d_exp_array(self.A, self.a, self.g0sq, self.gmax, garr, gsq, output)

        output *= 2*numpy.pi*garr

    def dbyg1_scalar(self, g1, g2):
        """
        Derivative with respect to g1 at the input g1,g2 location
        Used for lensfit.

        Uses central difference and a small enough step size
        to use just two points
        """
        ff = self.get_prob_scalar2d(g1+self.hhalf, g2)
        fb = self.get_prob_scalar2d(g1-self.hhalf, g2)

        return (ff - fb)*self.hinv

    def dbyg2_scalar(self, g1, g2):
        """
        Derivative with respect to g2 at the input g1,g2 location
        Used for lensfit.

        Uses central difference and a small enough step size
        to use just two points
        """
        ff = self.get_prob_scalar2d(g1,g2+self.hhalf)
        fb = self.get_prob_scalar2d(g1,g2-self.hhalf)

        return (ff - fb)*self.hinv


class GPriorMErf(GPriorBase):
    def __init__(self, pars):
        """
        [A, a, g0, gmax, gsigma]

        From Miller et al. multiplied by an erf
        """
        self.pars=pars

        self.A    = pars[0]
        self.a    = pars[1]
        self.g0   = pars[2]
        self.g0sq = self.g0**2
        self.gmax_func = pars[3]
        self.gsigma = pars[4]

        self.gmax = 1.0

    def get_prob_scalar1d(self, g):
        """
        Get the prob for the input g value
        """

        if g < 0.0 or g >= 1.0:
            raise GMixRangeError("g out of range")

        return self._get_prob_nocheck_prefac(g)

    def get_prob_scalar2d(self, g1, g2):
        """
        Get the 2d prob for the input g1,g2 values
        """

        gsq = g1**2 + g2**2
        g = sqrt(gsq)

        if g < 0.0 or g >= 1.0:
            raise GMixRangeError("g out of range")

        return self._get_prob_nocheck(g)

    def get_lnprob_scalar2d(self, g1, g2):
        """
        Get the 2d log prob for the input g1,g2 values
        """
        
        p=self.get_prob_scalar2d(g1,g2)
        return log(p)

    def get_prob_array1d(self, g):
        """
        Get the prob for the input g values
        """

        p=zeros(g.size)
        w,=where( (g >= 0.0) & (g < 1.0) )
        if w.size > 0:
            p[w]=self._get_prob_nocheck_prefac(g[w])
        return p

    def get_prob_array2d(self, g1, g2):
        """
        Get the 2d prob for the input g1,g2 values
        """

        gsq = g1**2 + g2**2
        g = sqrt(gsq)

        p=zeros(g1.size)
        w,=where( (g >= 0.0) & (g < 1.0) )
        if w.size > 0:
            p[w]=self._get_prob_nocheck(g[w])
        return p


    def get_lnprob_array2d(self, g1, g2):
        """
        Get the 2d log prob for the input g1,g2 values
        """

        gsq = g1**2 + g2**2
        g = sqrt(gsq)

        lnp=zeros(g1.size)+LOWVAL
        w,=where( (g >= 0.0) & (g < 1.0) )
        if w.size > 0:
            p=self._get_prob_nocheck(g[w])
            lnp[w] = log(p)
        return lnp



    def _get_prob_nocheck_prefac(self, g):
        """
        With the 2*pi*g
        """
        return 2*pi*g*self._get_prob_nocheck(g)

    def _get_prob_nocheck(self, g):
        """
        workhorse function

        this does not include the 2*pi*g
        """
        from scipy.special import erf

        #numer1 = 2*pi*g*self.A*(1-exp( (g-1.0)/self.a ))
        numer1 = self.A*(1-exp( (g-1.0)/self.a ))

        gerf=0.5*(1.0+erf((self.gmax_func-g)/self.gsigma))
        numer =  numer1 * gerf

        denom = (1+g)*sqrt(g**2 + self.g0sq)

        model=numer/denom

        return model



class Normal(object):
    """
    This class provides a uniform interface consistent with LogNormal
    """
    def __init__(self, cen, sigma):

        self.cen = array(cen, copy=False)
        self.sigma = sigma
        self.s2inv = 1./self.sigma**2

    def sample(self, *args):
        """
        Get samples.  Send no args to get a scalar.
        """
        rand=self.cen + self.sigma*numpy.random.randn(*args)
        return rand

    def get_lnprob(self, p):
        """
        log probability
        """
        diff = self.cen-p
        lnp = -0.5*diff*diff*self.s2inv
        return lnp

    # aliases
    get_lnprob_array  = get_lnprob
    get_lnprob_scalar = get_lnprob

    def get_prob(self,p):
        """
        probability
        """
        lnp = self.get_lnprob(p)
        return numpy.exp(lnp)

    # aliases
    get_prob_array  = get_prob
    get_prob_scalar = get_prob



class LogNormal(object):
    """
    Lognormal distribution

    parameters
    ----------
    mean:
        such that <x> in linear space is mean.  This implies the mean in log(x)
        is 
            <log(x)> = log(mean) - 0.5*log( 1 + sigma**2/mean**2 )
    sigma:
        such than the variace in linear space is sigma**2.  This implies
        the variance in log space is
            var(log(x)) = log( 1 + sigma**2/mean**2 )
    norm: optional
        When calling eval() the return value will be norm*prob(x)


    methods
    -------
    sample(nrand):
        Get nrand random deviates from the distribution
    lnprob(x):
        Get the natural logarithm of the probability of x.  x can
        be an array
    prob(x):
        Get the probability of x.  x can be an array
    """

    def __init__(self, mean, sigma):

        if mean <= 0:
            raise ValueError("mean %s is < 0" % mean)

        self.mean=mean
        self.sigma=sigma

        logmean  = numpy.log(self.mean) - 0.5*numpy.log( 1 + self.sigma**2/self.mean**2 )
        logvar   = numpy.log(1 + self.sigma**2/self.mean**2 )
        logsigma = numpy.sqrt(logvar)
        logivar  = 1./logvar

        self.logmean  = logmean
        self.logvar   = logvar
        self.logsigma = logsigma
        self.logivar  = logivar

        self.mode = numpy.exp(self.logmean - self.logvar)


    def get_lnprob_scalar(self, x):
        """
        This one has error checking
        """
        if x <= 0:
            raise GMixRangeError("values of x must be > 0")

        logx = numpy.log(x)
        chi2   = self.logivar*(logx-self.logmean)**2

        # subtract mode to make max 0.0
        lnprob = -0.5*chi2 - logx

        return lnprob

    def get_lnprob_array(self, x):
        """
        This one no error checking
        """

        x=numpy.array(x, dtype='f8', copy=False)

        w,=where(x <= 0)
        if w.size > 0:
            raise GMixRangeError("values of x must be > 0")

        logx = numpy.log(x)
        chi2   = self.logivar*(logx-self.logmean)**2

        # subtract mode to make max 0.0
        lnprob = -0.5*chi2 - logx - self.lnprob_mode

        return lnprob

    def get_prob_scalar(self, x):
        """
        Get the probability of x.
        """

        lnprob=self.get_lnprob_scalar(x)
        return numpy.exp(lnprob)

    def get_prob_array(self, x):
        """
        Get the probability of x.  x can be an array
        """

        lnp = self.get_lnprob_array(x)
        return numpy.exp(lnp)


    def sample(self, nrand=None):
        """
        Get nrand random deviates from the distribution

        If z is drawn from a normal random distribution, then exp(logmean+logsigma*z)
        is drawn from lognormal
        """
        if nrand is None:
            z=numpy.random.randn()
        else:
            z=numpy.random.randn(nrand)
        return numpy.exp(self.logmean + self.logsigma*z)

    def sample_brute(self, nrand=None, maxval=None):
        """
        Get nrand random deviates from the distribution using brute force

        This is really to check that our probabilities are being calculated
        correctly, by comparing to the regular sampler
        """

        if maxval is None:
            maxval=self.mean + 10*self.sigma


        if nrand is None:
            is_scalar=True
            nrand=1
        else:
            is_scalar=False

        samples=numpy.zeros(nrand)

        ngood=0
        nleft=nrand
        while ngood < nrand:
            
            rvals = maxval*randu(nleft)

            # between 0,1 which is max for prob
            h = randu(nleft)

            pvals = self.get_prob_array(rvals)

            w,=numpy.where(h < pvals)
            if w.size > 0:
                samples[ngood:ngood+w.size] = rvals[w]
                ngood += w.size
                nleft -= w.size
 
        if is_scalar:
            samples=samples[0]

        return samples

def lognorm_convert_old(mean, sigma):
    logmean  = log(mean) - 0.5*log( 1 + sigma**2/mean**2 )
    logvar   = log(1 + sigma**2/mean**2 )
    logsigma = sqrt(logvar)

    return logmean, logsigma

def lognorm_convert(mean, sigma, base=math.e):
    from math import log
    lbase=log(base)

    logmean  = log(mean,base) - 0.5*lbase*log( 1 + sigma**2/mean**2, base )
    logvar   = log(1 + sigma**2/mean**2, base )
    logsigma = sqrt(logvar)

    return logmean, logsigma


class BFracBase(object):
    """
    Base class for bulge fraction distribution
    """
    def sample(self, nrand=None):
        if nrand is None:
            return self.sample_one()
        else:
            return self.sample_many(nrand)

    def sample_one(self):
        while True:
            t=numpy.random.random()
            if t < self.bd_frac:
                r=self.bd_sigma*randn()
            else:
                r=1.0 + self.dev_sigma*randn()
            if r >= 0.0 and r <= 1.0:
                break
        return r

    def sample_many(self, nrand):
        r=numpy.zeros(nrand)-9999
        ngood=0
        nleft=nrand
        while ngood < nrand:
            tr=numpy.zeros(nleft)

            tmpu=randu(nleft)
            w,=numpy.where( tmpu < self.bd_frac )
            if w.size > 0:
                tr[0:w.size] = self.bd_loc  + self.bd_sigma*randn(w.size)
            if w.size < nleft:
                tr[w.size:]  = self.dev_loc + self.dev_sigma*randn(nleft-w.size)

            wg,=numpy.where( (tr >= 0.0) & (tr <= 1.0) )

            nkeep=wg.size
            if nkeep > 0:
                r[ngood:ngood+nkeep] = tr[wg]
                ngood += nkeep
                nleft -= nkeep
        return r

    def get_lnprob_array(self, bfrac):
        """
        Get the ln(prob) for the input b frac value
        """
        bfrac=numpy.array(bfrac, dtype='f8', copy=False)
        n=bfrac.size

        lnp=numpy.zeros(n)

        for i in xrange(n):
            lnp[i] = self.get_lnprob_scalar(bfrac[i])

        return lnp

class BFrac(BFracBase):
    """
    Bulge fraction

    half gaussian at zero width 0.1 for bulge+disk galaxies.

    smaller half gaussian at 1.0 with width 0.01 for bulge-only
    galaxies
    """
    def __init__(self):
        sq2pi=numpy.sqrt(2*numpy.pi)

        # bd is half-gaussian centered at 0.0
        self.bd_loc=0.0
        # fraction that are bulge+disk
        self.bd_frac=0.9
        # width of bd prior
        self.bd_sigma=0.1
        self.bd_ivar=1.0/self.bd_sigma**2
        self.bd_norm = self.bd_frac/(sq2pi*self.bd_sigma)


        # dev is half-gaussian centered at 1.0
        self.dev_loc=1.0
        # fraction of objects that are pure dev
        self.dev_frac=1.0-self.bd_frac
        # width of prior for dev-only region
        self.dev_sigma=0.01
        self.dev_ivar=1.0/self.dev_sigma**2
        self.dev_norm = self.dev_frac/(sq2pi*self.dev_sigma)

    def get_lnprob_scalar(self, bfrac):
        """
        Get the ln(prob) for the input b frac value
        """
        if bfrac < 0.0 or bfrac > 1.0:
            raise GMixRangeError("bfrac out of range")
        
        bd_diff  = bfrac-self.bd_loc
        dev_diff = bfrac-self.dev_loc

        p_bd  = self.bd_norm*numpy.exp(-0.5*bd_diff*bd_diff*self.bd_ivar)
        p_dev = self.dev_norm*numpy.exp(-0.5*dev_diff*dev_diff*self.dev_ivar)

        lnp = numpy.log(p_bd + p_dev)
        return lnp

class TruncatedGaussian(object):
    """
    Truncated gaussian
    """
    def __init__(self, mean, sigma, minval, maxval):
        self.mean=mean
        self.sigma=sigma
        self.ivar=1.0/sigma**2
        self.minval=minval
        self.maxval=maxval

    def get_lnprob_scalar(self, x):
        """
        just raise error if out of rang
        """
        if x < self.minval or x > self.maxval:
            raise GMixRangeError("value out of range")
        diff=x-self.mean
        return -0.5*diff*diff*self.ivar

class TruncatedGaussianPolar(object):
    """
    Truncated gaussian on a circle
    """
    def __init__(self, mean1, mean2, sigma1, sigma2, maxval):
        self.mean1=mean1
        self.mean2=mean2
        self.sigma1=sigma1
        self.sigma2=sigma2
        self.ivar1=1.0/sigma1**2
        self.ivar2=1.0/sigma2**2

        self.maxval=maxval
        self.maxval2=maxval**2

    def sample(self, nrand):
        """
        Sample from truncated gaussian
        """
        x1=numpy.zeros(nrand)
        x2=numpy.zeros(nrand)

        nleft=nrand
        ngood=0
        while nleft > 0:
            r1=self.mean1 + self.sigma1*randn(nleft)
            r2=self.mean2 + self.sigma2*randn(nleft)

            rsq=r1**2 + r2**2

            w,=numpy.where(rsq < self.maxval2)
            nkeep=w.size
            if nkeep > 0:
                x1[ngood:ngood+nkeep] = r1[w]
                x2[ngood:ngood+nkeep] = r2[w]
                nleft -= nkeep
                ngood += nkeep

        return x1,x2

    def get_lnprob_scalar(self, x1, x2):
        """
        ln(p) for scalar inputs
        """
        xsq=x1**2 + x2**2
        if xsq > self.maxval2:
            raise GMixRangeError("square value out of range: %s" % xsq)
        diff1=x1-self.mean1
        diff2=x2-self.mean2
        return - 0.5*diff1*diff1*self.ivar1 - 0.5*diff2*diff2*self.ivar2 

    def get_lnprob_array(self, x1, x2):
        """
        ln(p) for a array inputs
        """
        x1=numpy.array(x1, dtype='f8', copy=False)
        x2=numpy.array(x2, dtype='f8', copy=False)

        x2=x1**2 + x2**2
        w,=numpy.where(x2 > self.maxval2)
        if w.size > 0:
            raise GMixRangeError("values out of range")
        diff1=x1-self.mean1
        diff2=x2-self.mean2
        return - 0.5*diff1*diff1*self.ivar1 - 0.5*diff2*diff2*self.ivar2 

class Student(object):
    def __init__(self, mean, sigma):
        """
        sigma is not std(x)
        """
        self.reset(mean,sigma)

    def reset(self, mean, sigma):
        """
        complete reset
        """
        import scipy.stats
        self.mean=mean
        self.sigma=sigma

        self.tdist = scipy.stats.t(1.0, loc=mean, scale=sigma)

    def sample(self, nrand):
        """
        Draw samples from the distribution
        """
        return self.tdist.rvs(nrand)

    def get_lnprob_array(self, x):
        """
        get ln(prob)
        """
        return self.tdist.logpdf(x)

    get_lnprob_scalar=get_lnprob_array

class StudentPositive(Student):
    def sample(self, nrand):
        """
        Draw samples from the distribution
        """
        vals=numpy.zeros(nrand)

        nleft=nrand
        ngood=0
        while nleft > 0:
            r = super(StudentPositive,self).sample(nleft)

            w,=numpy.where(r > 0.0)
            nkeep=w.size
            if nkeep > 0:
                vals[ngood:ngood+nkeep] = r[w]
                nleft -= nkeep
                ngood += nkeep

        return vals

    def get_lnprob_scalar(self, x):
        """
        get ln(prob)
        """

        if x <= 0:
            raise GMixRangeError("value less than zero")
        return super(StudentPositive,self).get_lnprob_scalar(x)

    def get_lnprob_array(self, x):
        """
        get ln(prob)
        """
        x=numpy.array(x, dtype='f8', copy=False)
        
        w,=numpy.where(x <= 0)
        if w.size != 0:
            raise GMixRangeError("values less than zero")
        return super(StudentPositive,self).get_lnprob_array(x)



class Student2D(object):
    def __init__(self, mean1, mean2, sigma1, sigma2):
        """
        sigma is not std(x)
        """
        self.reset(mean1, mean2, sigma1, sigma2)

    def reset(self, mean1, mean2, sigma1, sigma2):
        """
        complete reset
        """
        import scipy.stats
        self.mean1=mean1
        self.sigma1=sigma1
        self.mean2=mean2
        self.sigma2=sigma2

        self.tdist1 = Student(mean1, sigma1)
        self.tdist2 = Student(mean2, sigma2)

    def sample(self, nrand):
        """
        Draw samples from the distribution
        """
        r1 = self.tdist1.sample(nrand)
        r2 = self.tdist2.sample(nrand)

        return r1, r2

    def get_lnprob_array(self, x1, x2):
        """
        get ln(prob)
        """
        lnp1 = self.tdist1.get_lnprob_array(x1)
        lnp2 = self.tdist2.get_lnprob_array(x2)

        return lnp1 + lnp2

    get_lnprob_scalar=get_lnprob_array


class TruncatedStudentPolar(Student2D):
    """
    Truncated gaussian on a circle
    """
    def __init__(self, mean1, mean2, sigma1, sigma2, maxval):
        """
        sigma is not std(x)
        """
        self.reset(mean1, mean2, sigma1, sigma2, maxval)

    def reset(self, mean1, mean2, sigma1, sigma2, maxval):
        super(TruncatedStudentPolar,self).reset(mean1, mean2, sigma1, sigma2)
        self.maxval=maxval
        self.maxval2=maxval**2

    def sample(self, nrand):
        """
        Sample from truncated gaussian
        """
        x1=numpy.zeros(nrand)
        x2=numpy.zeros(nrand)

        nleft=nrand
        ngood=0
        while nleft > 0:
            r1,r2 = super(TruncatedStudentPolar,self).sample(nleft)

            rsq=r1**2 + r2**2

            w,=numpy.where(rsq < self.maxval2)
            nkeep=w.size
            if nkeep > 0:
                x1[ngood:ngood+nkeep] = r1[w]
                x2[ngood:ngood+nkeep] = r2[w]
                nleft -= nkeep
                ngood += nkeep

        return x1,x2

    def get_lnprob_scalar(self, x1, x2):
        """
        ln(p) for scalar inputs
        """

        x2=x1**2 + x2**2
        if x2 > self.maxval2:
            raise GMixRangeError("square value out of range: %s" % x2)
        lnp = super(TruncatedStudentPolar,self).get_lnprob_scalar(x1,x2)
        return lnp


    def get_lnprob_array(self, x1, x2):
        """
        ln(p) for a array inputs
        """

        x1=numpy.array(x1, dtype='f8', copy=False)
        x2=numpy.array(x2, dtype='f8', copy=False)

        x2=x1**2 + x2**2
        w,=numpy.where(x2 > self.maxval2)
        if w.size > 0:
            raise GMixRangeError("values out of range")

        lnp = super(TruncatedStudentPolar,self).get_lnprob_array(x1,x2)
        return lnp

def scipy_to_lognorm(shape, scale):
    """
    Wrong?
    """
    srat2 = numpy.exp( shape**2 ) - 1.0
    #srat2 = numpy.exp( shape ) - 1.0

    meanx = scale*numpy.exp( 0.5*numpy.log( 1.0 + srat2 ) )
    sigmax = numpy.sqrt( srat2*meanx**2)

    return meanx, sigmax

class CenPrior(object):
    """
    Independent gaussians in each dimension
    """
    def __init__(self, cen1, cen2, sigma1, sigma2):

        self.cen1 = cen1
        self.cen2 = cen2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.s2inv1 = 1./self.sigma1**2
        self.s2inv2 = 1./self.sigma2**2

    def get_lnprob(self,p1, p2):
        """
        log probability
        """
        d1 = self.cen1-p1
        d2 = self.cen2-p2
        lnp = -0.5*d1*d1*self.s2inv1 - 0.5*d2*d2*self.s2inv2

        return lnp

    get_lnprob_array=get_lnprob

    def get_prob(self,p1, p2):
        """
        probability, max 1
        """
        lnp = self.get_lnprob(p1, p2)
        return exp(lnp)

    get_prob_array=get_prob

    def sample(self, n=None):
        """
        Get a single sample or arrays
        """
        if n is None:
            rand1=self.cen1 + self.sigma1*numpy.random.randn()
            rand2=self.cen2 + self.sigma2*numpy.random.randn()
        else:
            rand1=self.cen1 + self.sigma1*numpy.random.randn(n)
            rand2=self.cen2 + self.sigma2*numpy.random.randn(n)

        return rand1, rand2

    def sample_brute(self, nrand=None):
        """
        Get nrand random deviates from the distribution using brute force

        This is really to check that our probabilities are being calculated
        correctly, by comparing to the regular sampler
        """

        # 10 sigma
        maxval1 = self.cen1 + 5.0*self.sigma1
        maxval2 = self.cen2 + 5.0*self.sigma2

        if nrand is None:
            is_scalar=True
            nrand=1
        else:
            is_scalar=False

        r1=numpy.zeros(nrand)
        r2=numpy.zeros(nrand)

        ngood=0
        nleft=nrand
        while ngood < nrand:
            
            rvals1 = maxval1*srandu(nleft)
            rvals2 = maxval2*srandu(nleft)

            # between 0,1 which is max for prob
            h = randu(nleft)

            pvals = self.get_prob_array(rvals1, rvals2)

            w,=numpy.where(h < pvals)
            if w.size > 0:
                r1[ngood:ngood+w.size] = rvals1[w]
                r2[ngood:ngood+w.size] = rvals2[w]
                ngood += w.size
                nleft -= w.size
 
        if is_scalar:
            r1=r1[0]
            r2=r2[0]

        return r1,r2





class TruncatedGauss2D(object):
    """
    Independent gaussians in each dimension, with a specified
    maximum length
    """

    def __init__(self, cen1, cen2, sigma1, sigma2, maxval):

        self.cen1 = cen1
        self.cen2 = cen2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.s2inv1 = 1./self.sigma1**2
        self.s2inv2 = 1./self.sigma2**2

        self.maxval=maxval
        self.maxval_sq = maxval**2

    def get_lnprob_nothrow(self,p1, p2):
        """
        log probability
        """

        psq = p1**2 + p2**2
        if psq >= self.maxval_sq:
            lnp=LOWVAL
        else:
            d1 = self.cen1-p1
            d2 = self.cen2-p2
            lnp = -0.5*d1*d1*self.s2inv1 - 0.5*d2*d2*self.s2inv2

        return lnp

    def get_lnprob(self,p1, p2):
        """
        log probability
        """

        psq = p1**2 + p2**2
        if psq >= self.maxval_sq:
            raise GMixRangeError("value too big")

        d1 = self.cen1-p1
        d2 = self.cen2-p2
        lnp = -0.5*d1*d1*self.s2inv1 - 0.5*d2*d2*self.s2inv2

        return lnp

    def get_prob(self,p1, p2):
        """
        linear probability
        """
        lnp = self.get_lnprob(p1, p2)
        return numpy.exp(lnp)

    def sample(self, nrand=None):
        """
        Get nrand random deviates from the distribution
        """

        if nrand is None:
            is_scalar=True
            nrand=1
        else:
            is_scalar=False

        r1=numpy.zeros(nrand)
        r2=numpy.zeros(nrand)

        ngood=0
        nleft=nrand
        while ngood < nrand:
            
            rvals1=self.cen1 + self.sigma1*randn(nleft)
            rvals2=self.cen2 + self.sigma2*randn(nleft)

            rsq = rvals1**2 + rvals2**2

            w,=numpy.where(rsq < self.maxval_sq)
            if w.size > 0:
                r1[ngood:ngood+w.size] = rvals1[w]
                r2[ngood:ngood+w.size] = rvals2[w]
                ngood += w.size
                nleft -= w.size
 
        if is_scalar:
            r1=r1[0]
            r2=r2[0]

        return r1,r2


JOINT_N_ITER=1000
JOINT_MIN_COVAR=1.0e-6
JOINT_COVARIANCE_TYPE='full'


class TFluxPriorCosmosBase(object):
    """
    T is in arcsec**2
    """
    def __init__(self,
                 T_bounds=[1.e-6,100.0],
                 flux_bounds=[1.e-6,100.0]):

        self.set_bounds(T_bounds=T_bounds, flux_bounds=flux_bounds)

        self.make_gmm()
        self.make_gmix()

    def set_bounds(self, T_bounds=None, flux_bounds=None):

        if len(T_bounds) != 2:
            raise ValueError("T_bounds must be len==2")
        if len(flux_bounds) != 2:
            raise ValueError("flux_bounds must be len==2")

        self.T_bounds    = array(T_bounds, dtype='f8')
        self.flux_bounds = array(flux_bounds,dtype='f8')


    def get_flux_mode(self):
        return self.flux_mode

    def get_T_near(self):
        return self.T_near

    def get_lnprob_slow(self, T_and_flux_input):
        """
        ln prob for linear variables
        """

        nd=len( T_and_flux_input.shape )

        if nd == 1:
            T_and_flux = T_and_flux_input.reshape(1,2)
        else:
            T_and_flux = T_and_flux_input

        T_bounds=self.T_bounds
        T=T_and_flux[0,0]
        if T < T_bounds[0] or T > T_bounds[1]:
            return LOWVAL

        flux_bounds=self.flux_bounds
        flux=T_and_flux[0,1]
        if flux < flux_bounds[0] or flux > flux_bounds[1]:
            return LOWVAL

        logvals = numpy.log10( T_and_flux )
        
        return self.gmm.score(logvals)

    def get_lnprob_many(self, T_and_flux):
        """
        The fast array one using a GMix
        """

        n=T_and_flux.shape[0]
        lnp=numpy.zeros(n)

        for i in xrange(n):
            lnp[i] = self.get_lnprob_one(T_and_flux[i,:])

        return lnp

    def get_lnprob_one(self, T_and_flux):
        """
        The fast scalar one using a GMix
        """

        # bounds cuts happen in pixel space (if scale!=1)
        T_bounds=self.T_bounds
        flux_bounds=self.flux_bounds

        T=T_and_flux[0]
        if T < T_bounds[0] or T > T_bounds[1]:
            return LOWVAL

        flux=T_and_flux[1]
        if flux < flux_bounds[0] or flux > flux_bounds[1]:
            return LOWVAL

        logpars=numpy.log10(T_and_flux)

        return self.gmix.get_loglike_rowcol(logpars[0], logpars[1])

    def sample(self, n):
        """
        Sample the linear variables
        """

        T_bounds = self.T_bounds
        flux_bounds = self.flux_bounds

        lin_samples=numpy.zeros( (n,2) )
        nleft=n
        ngood=0
        while nleft > 0:
            tsamples=self.gmm.sample(nleft)
            tlin_samples = 10.0**tsamples

            Tvals = tlin_samples[:,0]
            flux_vals=tlin_samples[:,1]
            w,=numpy.where(  (Tvals > T_bounds[0])
                           & (Tvals < T_bounds[1])
                           & (flux_vals > flux_bounds[0])
                           & (flux_vals < flux_bounds[1]) )

            if w.size > 0:
                lin_samples[ngood:ngood+w.size,:] = tlin_samples[w,:]
                ngood += w.size
                nleft -= w.size

        return lin_samples

    def make_gmm(self):
        """
        Make a GMM object from the inputs
        """
        from sklearn.mixture import GMM
        # we will over-ride values, pars here shouldn't matter except
        # for consistency

        ngauss=self.weights.size
        gmm=GMM(n_components=ngauss,
                n_iter=JOINT_N_ITER,
                min_covar=JOINT_MIN_COVAR,
                covariance_type=JOINT_COVARIANCE_TYPE)
        gmm.means_ = self.means
        gmm.covars_ = self.covars
        gmm.weights_ = self.weights

        self.gmm=gmm 

    def make_gmix(self):
        """
        Make a GMix object for speed
        """
        from . import gmix

        ngauss=self.weights.size
        pars=numpy.zeros(6*ngauss)
        for i in xrange(ngauss):
            index = i*6
            pars[index + 0] = self.weights[i]
            pars[index + 1] = self.means[i,0]
            pars[index + 2] = self.means[i,1]

            pars[index + 3] = self.covars[i,0,0]
            pars[index + 4] = self.covars[i,0,1]
            pars[index + 5] = self.covars[i,1,1]

        self.gmix = gmix.GMix(ngauss=ngauss, pars=pars)

class TFluxPriorCosmosExp(TFluxPriorCosmosBase):
    """
    joint T-flux distribution based on fits to cosmos data
    using an Exp model

    pars from 
    ~/lensing/galsim-cosmos-data/gmix-fits/001/dist/gmix-cosmos-001-exp-joint-dist.fits
    """

    flux_mode=0.121873372203

    # T_near is mean T near the flux mode
    T_near=0.182461907543  # arcsec**2

    weights=array([ 0.24725964,  0.12439838,  0.10301993,
                   0.07903986,  0.16064439, 0.01162365,
                   0.15587558,  0.11813856])
    means=array([[-0.63771027, -0.55703495],
                 [-1.09910453, -0.79649937],
                 [-0.89605713, -0.9432781 ],
                 [-1.01262357, -0.05649636],
                 [-0.26622288, -0.16742824],
                 [-0.50259072, -0.10134068],
                 [-0.11414387, -0.49449105],
                 [-0.39625801, -0.83781264]])

    covars=array([[[ 0.17219003, -0.00650028],
                   [-0.00650028,  0.05053097]],

                  [[ 0.09800749,  0.00211059],
                   [ 0.00211059,  0.01099591]],

                  [[ 0.16110231,  0.00194853],
                   [ 0.00194853,  0.00166609]],

                  [[ 0.07933064,  0.08535596],
                   [ 0.08535596,  0.20289928]],

                  [[ 0.17805248,  0.1356711 ],
                   [ 0.1356711 ,  0.24728999]],

                  [[ 1.07112875, -0.08777646],
                   [-0.08777646,  0.26131025]],

                  [[ 0.06570913,  0.04912536],
                   [ 0.04912536,  0.094739  ]],

                  [[ 0.0512232 ,  0.00519115],
                   [ 0.00519115,  0.00999365]]])



class TFluxPriorCosmosDev(TFluxPriorCosmosBase):
    """
    joint T-flux distribution based on fits to cosmos data
    using an Dev model

    pars from
    ~/lensing/galsim-cosmos-data/gmix-fits/001/dist/gmix-cosmos-001-dev-joint-dist.fits
    """

    flux_mode=0.241579121406777
    # T_near is mean T near the flux mode
    T_near=2.24560320009

    weights=array([ 0.13271808,  0.10622821,  0.09982766,
                   0.29088236,  0.1031488 , 0.10655095,
                   0.01631938,  0.14432457])
    means=array([[ 0.22180692,  0.05099562],
                 [ 0.92589409, -0.61785194],
                 [-0.10599071, -0.82198516],
                 [ 1.06182382, -0.29997484],
                 [ 1.57670969,  0.25244698],
                 [-0.1807241 , -0.35644187],
                 [ 0.73288177,  0.07028779],
                 [-0.14674281, -0.65843109]])
    covars=array([[[ 0.35760805,  0.17652397],
                   [ 0.17652397,  0.28479473]],

                  [[ 0.14969479,  0.03008289],
                   [ 0.03008289,  0.01369154]],

                  [[ 0.37153864,  0.03293217],
                   [ 0.03293217,  0.00476308]],

                  [[ 0.27138008,  0.08628813],
                   [ 0.08628813,  0.0883718 ]],

                  [[ 0.20242911,  0.12374718],
                   [ 0.12374718,  0.22626528]],

                  [[ 0.30836137,  0.07141098],
                   [ 0.07141098,  0.05870399]],

                  [[ 2.13662397,  0.04301596],
                   [ 0.04301596,  0.17006638]],

                  [[ 0.40151762,  0.05215542],
                   [ 0.05215542,  0.01733399]]])


'''
class TPriorCosmosBase(object):
    """
    From fitting double gaussians to log(T) (actually log10
    but we ignore the constant of prop.)

    dN/dT = (1/T)dN/dlog(T)
    """
    def get_prob_scalar(self, T):
        """
        prob for a scalar
        """
        return _2gauss_prob_scalar(self.means, self.ivars, self.weights, T)

    def get_prob_array(self, Tarr):
        """
        prob for an array
        """
        Tarr=numpy.array(Tarr, copy=False)
        output=numpy.zeros(Tarr.size)
        _2gauss_prob_array(self.means, self.ivars, self.weights, Tarr, output)
        return output
    
    def get_lnprob_scalar(self, T):
        """
        ln prob for a scalar
        """
        return _2gauss_lnprob_scalar(self.means, self.ivars, self.weights, T)


class TPriorCosmosExp(TPriorCosmosBase):
    __doc__=TPriorCosmosBase.__doc__
    def __init__(self):
        self.means   = numpy.array([-0.40019405, -1.13597665])
        self.sigmas  = numpy.array([ 0.40682263,  0.29353975])
        self.ivars   = 1./self.sigmas**2
        self.weights = numpy.array([ 0.75845293,  0.24154707])
class TPriorCosmosDev(TPriorCosmosBase):
    __doc__=TPriorCosmosBase.__doc__
    def __init__(self):
        self.means   = numpy.array([-0.23898365,  0.99036254])
        self.sigmas  = numpy.array([ 0.56108066,  0.6145333 ])
        self.ivars   = 1./self.sigmas**2
        self.weights = numpy.array([ 0.34782633,  0.65217367])


@autojit
def _2gauss_prob_scalar(means, ivars, weights, val):
    """
    calculate the double gaussian dN/dlog10(T)

    Then multiply by 1/T to put back in linear space
    """
    if val <= 0:
        raise GMixRangeError("values must be > 0")
    lnv=numpy.log10(val)

    diff1=lnv-means[0]
    diff1sq = diff1*diff1

    diff2=lnv-means[1]
    diff2sq = diff2*diff2

    p1 = weights[0]*numpy.exp( -0.5*ivars[0]*diff1sq )
    p2 = weights[1]*numpy.exp( -0.5*ivars[1]*diff2sq )
    
    # the 1/val puts us back into dN/val space
    p = (p1+p2)/val
    return p

@autojit
def _2gauss_lnprob_scalar(means, ivars, weights, val):
    p=_2gauss_prob_scalar(means, ivars, weights, val) 
    return numpy.log(p)

@autojit
def _2gauss_prob_array(means, ivars, weights, vals, output):
    """
    calculate the double gaussian dN/dlog10(T)

    Then multiply by 1/T to put back in linear space
    """

    n=vals.size
    for i in xrange(n):
        val=vals[i]
        output[i] = _2gauss_prob_scalar(val)
'''


def srandu(num=None):
    """
    Generate random numbers in the symmetric distribution [-1,1]
    """
    return 2*(numpy.random.random(num)-0.5)



class GPriorCosmosSersicSpline(GPriorMErf):
    def __init__(self):
        self.gmax=1.0

    def _get_prob_nocheck(self, g):
        """
        Remove the 2*pi*g
        """
        g_arr=numpy.atleast_1d(g)
        w,=where(g_arr > 0)
        if w.size != g_arr.size:
            raise GMixRangeError("zero g found")

        p_prefac=self._get_prob_nocheck_prefac(g_arr)
        p=zeros(g_arr.size)
        p = p_prefac/(2.*pi*g_arr)
        return p

    def _get_prob_nocheck_prefac(self, g):
        """
        Input should be in bounds and an array
        """
        from scipy.interpolate import fitpack

        g_arr=numpy.atleast_1d(g)
        p,err=fitpack._fitpack._spl_(g_arr,
                                     0,
                                     _g_cosmos_t,
                                     _g_cosmos_c,
                                     _g_cosmos_k,
                                     0)
        if err:
            raise RuntimeError("error occurred in g interp")

        if not isinstance(g,numpy.ndarray):
            return p[0]
        else:
            return p


_g_cosmos_t = array([ 0.   ,  0.   ,  0.   ,  0.   ,  0.001,  0.15 ,  0.2  ,  0.4  ,
                             0.5  ,  0.7  ,  0.8  ,  0.85 ,  0.86 ,  0.865,  0.89 ,  0.9  ,
                             0.91 ,  0.92 ,  0.925,  0.93 ,  0.94 ,  0.95 ,  0.96 ,  0.97 ,
                             0.98 ,  0.999,  1.   ,  1.   ,  1.   ,  1.   ])
_g_cosmos_c = array([  0.00000000e+00,   4.16863836e+01,   1.08898496e+03,
                              1.11759878e+03,   1.08252322e+03,   8.37179646e+02,
                              5.01288706e+02,   2.85169581e+02,   1.82341644e+01,
                              9.86159479e+00,   2.35167895e+00,  -1.16749781e-02,
                              2.18333603e-03,   6.23967745e-03,   4.25839705e-03,
                              5.30413464e-03,   4.89754389e-03,   5.11359296e-03,
                              4.87944452e-03,   5.13580057e-03,   4.78950152e-03,
                              5.37471368e-03,   4.00911261e-03,   7.30958026e-03,
                             -4.03798531e-20,   0.00000000e+00,   0.00000000e+00,
                              0.00000000e+00,   0.00000000e+00,   0.00000000e+00])
_g_cosmos_k= 3


'''
_g_cosmos_t = array([0.,0.,0.,0.,0.001,0.15,0.2,0.4,0.5,
                     0.7,0.8,0.9,0.92,0.999,1.,1.,1.,1.])

_g_cosmos_c= array([0.,41.56646313,1089.28265178,1117.24827112, 
          1083.02107936,836.29580332,502.71408304,282.19129159, 
          -14.58689825,4.83572205,-3.331918,2.16668349, 
          0.,0.,0.,0., 0.,0.])

_g_cosmos_k = 3
'''


class Disk2D(object):
    """
    uniform over a disk, [0,r)
    """
    def __init__(self, cen, radius):

        self.cen = array(cen, copy=False)
        if self.cen.size != 2:
            raise ValueError("cen should have two elements")
        self.radius = radius
        self.radius_sq = radius**2

    def sample1d(self, n):
        """
        Get samples in 1-d radius
        """
        r2 = self.radius_sq*randu(n)

        r = sqrt(r2)
        return r

    def sample2d(self, n):
        """
        Get samples.  Send no args to get a scalar.
        """

        radius=self.sample1d(n)

        theta=2.0*numpy.pi*randu(n)

        x=radius*cos(theta)
        y=radius*sin(theta)

        x += self.cen[0]
        y += self.cen[1]

        return x,y

    def get_lnprob_scalar1d(self, r):
        """
        log probability 0.0 inside of disk, outside raises
        an exception
        """
        if r >= self.radius:
            raise GMixRangeError("position out of disk")
        return 0.0

    def get_lnprob_array1d(self, r):
        """
        log probability 0.0 inside of disk, outside raises
        an exception
        """

        w,=numpy.where(r >= self.radius)
        if w.size > 0:
            raise GMixRangeError("some positions were out of disk")

        return zeros(x.size)

    def get_prob_scalar1d(self, r):
        """
        probability, 1.0 inside disk, outside raises exception

        does not raise an exception
        """
        if r >= self.radius:
            p=0.0
        else:
            p=1.0
        return p

    def get_prob_array1d(self, r):
        """
        probability, 1.0 inside disk, outside raises exception

        does not raise an exception
        """

        p = ones(x.size)
        w,=where(r >= self.radius)
        if w.size > 0:
            p[w]=0.0
        return p

    def get_lnprob_scalar2d(self, x, y):
        """
        log probability 0.0 inside of disk, outside raises
        an exception
        """

        r2 = x**2 + y**2
        if r2 >= self.radius_sq:
            p=0.0
        else:
            p=1.0
        return p


        return self.get_lnprob_scalar1d(r)


    def get_lnprob_array2d(self, x, y):
        """
        log probability 0.0 inside of disk, outside raises
        an exception
        """

        r = sqrt(x**2 + y**2)
        return self.get_lnprob_array1d(r)


    def get_prob_scalar2d(self, x, y):
        """
        probability, 1.0 inside disk, outside raises exception

        does not raise an exception
        """
        r = sqrt(x**2 + y**2)
        return self.get_prob_scalar1d(r)

    def get_prob_array2d(self, x, y):
        """
        probability, 1.0 inside disk, outside raises exception

        does not raise an exception
        """
        r = sqrt(x**2 + y**2)
        return self.get_prob_array1d(r)



