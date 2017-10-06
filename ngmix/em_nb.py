import numpy
from numpy import nan
from numba import njit

from .gexceptions import GMixRangeError
from .gmix_nb import gauss2d_set_norm, gmix_get_e1e2T
from .fastexp_nb import exp3

try:
    xrange
except:
    xrange=range

@njit(cache=True)
def em_run(conf, pixels, sums, gmix):
    """
    run the EM algorithm

    parameters
    ----------
    conf: array
        Should have fields

            sky_guess: guess for the sky
            counts: counts in the image
            tol: tolerance for stopping
            maxiter: maximum number of iterations
            pixel_scale: pixel scale

    pixels: pixel array
        for the image/jacobian
    gmix: gaussian mixture
        Initialized to the starting guess
    """

    area = pixels.size*conf['pixel_scale']*conf['pixel_scale']

    nsky = conf['sky_guess']/conf['counts']
    psky = conf['sky_guess']/(conf['counts']/area)

    T_last = e1_last = e2_last = -9999.0
    for i in xrange(conf['maxiter']):
        skysum=0.0
        clear_sums(sums)

        for pixel in pixels:

            gtot = do_scratch_sums(pixel, gmix, sums)

            gtot += nsky
            if gtot==0.0:
                raise GMixRangeError("gtot == 0")

            imnorm = pixel['val']/counts
            skysum += nsky*imnorm/gtot

            igrat = imnorm/gtot
            do_sums(sums, igrat)

        gmix_set_from_sums(gmix, sums)

        psky = skysum
        nsky = psky/area

        e1,e2,T=gmix_get_e1e2T(gmix)

        frac_diff = abs((T-T_last)/T)
        e1diff    = abs(e1-e1_last)
        e2diff    = abs(e2-e2_last)

        if ( frac_diff < tol and e1diff < tol and e2diff < tol ):
            break

        T_last, e1_last, e2_last = T, e1, e2

    numiter=i+1
    return numiter, frac_diff

@njit(cache=True)
def do_scratch_sums(pixel, gmix, sums):
    """
    do the basic sums for this pixel, using
    scratch space in the sums struct
    """
    n_gauss = gmix.size

    gtot = 0.0
    for igauss in xrange(n_gauss):
        gauss = gmix[igauss]
        tsums = sums[igauss]

        vdiff = pixel['v']-gauss['row']
        udiff = pixel['u']-gauss['col']

        u2 = udiff*udiff
        v2 = vdiff*vdiff
        uv = udiff*vdiff

        chi2 = \
            gauss['dcc']*v2 + gauss['drr']*u2 - 2.0*gauss['drc']*uv

        if chi2 < 25.0 and chi2 >= 0.0:
            tsums['gi'] = gauss['pnorm']*exp3( -0.5*chi2 )

        gtot += tsums['gi']
        tsums['trowsum'] = v*tsums['gi']
        tsums['tcolsum'] = u*tsums['gi']
        tsums['tv2sum']  = v2*tsums['gi']
        tsums['tuvsum']  = uv*tsums['gi']
        tsums['tu2sum']  = u2*tsums['gi']

    return gtot

@njit(cache=True)
def do_sums(sums, igrat):
    """
    do the sums based on the scratch values
    """

    n_gauss=sums.size
    for igauss in xrange(n_gauss):
        tsums = sums[igauss]

        # wtau is gi[pix]/gtot[pix]*imnorm[pix]
        # which is Dave's tau*imnorm = wtau
        wtau = tsums['gi']*igrat

        tsums['pnew'] += wtau

        # row*gi/gtot*imnorm
        tsums['rowsum'] += tsums['trowsum']*igrat
        tsums['colsum'] += tsums['tcolsum']*igrat
        tsums['u2sum']  += tsums['tu2sum']*igrat
        tsums['uvsum']  += tsums['tuvsum']*igrat
        tsums['v2sum']  += tsums['tv2sum']*igrat


@njit(cache=True)
def gmix_set_from_sums(gmix, sums):
    """
    fill the gaussian mixture from the em sums
    """

    n_gauss=gmix.size
    for i in xrange(n_gauss):

        tsums = sums[i]
        gauss = gmix[i]

        p = tsum['pnew']
        pinv=1.0/p

        gauss2d_set(
            gauss,
            p,
            tsums['rowsum']*pinv,
            tsums['colsum']*pinv,
            tsums['v2sum']*pinv,
            tsums['uvsum']*pinv,
            tsums['u2sum']*pinv,
        )

        gauss2d_set_norm(gauss)

@njit(cache=True)
def clear_sums(sums):
    """
    set all sums to zero
    """
    sums.fill(0)
