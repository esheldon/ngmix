import numpy
from numpy import nan
from numba import jit, njit
from .fastexp_nb import exp3

try:
    xrange
except:
    xrange=range

# need to make this a pure python exception
from .gexceptions import GMixRangeError

@njit(cache=True)
def gauss2d_eval_pixel_fast(gauss, pixel, max_chi2=25.0):
    """
    evaluate a 2-d gaussian at the specified location, using
    the fast exponential

    parameters
    ----------
    gauss2d: gauss2d structure
        row,col,dcc,drr,drc,pnorm... See gmix.py
    v,u: numbers
        location in v,u plane (row,col for simple transforms)
    """
    model_val=0.0

    # v->row, u->col in gauss
    vdiff = pixel['v'] - gauss['row']
    udiff = pixel['u'] - gauss['col']

    chi2 = (      gauss['dcc']*vdiff*vdiff
            +     gauss['drr']*udiff*udiff
            - 2.0*gauss['drc']*vdiff*udiff )

    if chi2 < max_chi2 and chi2 >= 0.0:
        model_val = gauss['pnorm']*exp3( -0.5*chi2 )

    return model_val

@njit(cache=True)
def gmix_eval_pixel_fast(gmix, pixel, max_chi2=25.0):
    """
    evaluate a single gaussian mixture, using the
    fast exponential
    """
    model_val=0.0
    for igauss in xrange(gmix.size):

        model_val += gauss2d_eval_pixel_fast(
            gmix[igauss],
            pixel,
            max_chi2,
        )


    return model_val


@njit(cache=True)
def gauss2d_eval_pixel(gauss, pixel):
    """
    evaluate a 2-d gaussian at the specified location

    parameters
    ----------
    gauss2d: gauss2d structure
        row,col,dcc,drr,drc,pnorm... See gmix.py
    pixel: struct with coods
        should have fields v,u
    """
    model_val=0.0

    # v->row, u->col in gauss
    vdiff = pixel['v'] - gauss['row']
    udiff = pixel['u'] - gauss['col']

    chi2 = (      gauss['dcc']*vdiff*vdiff
            +     gauss['drr']*udiff*udiff
            - 2.0*gauss['drc']*vdiff*udiff )

    model_val = gauss['pnorm']*numpy.exp( -0.5*chi2 )

    return model_val

@njit(cache=True)
def gmix_eval_pixel(gmix, pixel):
    """
    evaluate a single gaussian mixture
    """
    model_val=0.0
    for igauss in xrange(gmix.size):

        model_val += gauss2d_eval_pixel(
            gmix[igauss],
            pixel,
        )

    return model_val



@njit(cache=True)
def gmix_get_cen(gmix):
    """
    get the center of the gaussian mixture, as well as
    the psum
    """
    row=0.0
    col=0.0
    psum=0.0

    n_gauss=gmix.size
    for i in xrange(n_gauss):
        gauss=gmix[i]

        p = gauss['p']
        row += p*gauss['row']
        col += p*gauss['col']
        psum += p

    row /= psum
    col /= psum

    return row, col, psum

@jit(cache=True)
def gmix_get_e1e2T(gmix):
    """
    get e1,e2,T for the gaussian mixture
    """

    row, col, psum0 = gmix_get_cen(gmix)

    if psum0 == 0.0:
        raise GMixRangeError("cannot calculate T due to zero psum")

    n_gauss=gmix.size
    psum=0.0
    irr_sum=0.0
    irc_sum=0.0
    icc_sum=0.0

    for i in xrange(n_gauss):
        gauss = gmix[i]

        p = gauss['p']

        rowdiff = gauss['row']-row
        coldiff = gauss['col']-col

        irr_sum += p*(gauss['irr'] + rowdiff*rowdiff)
        irc_sum += p*(gauss['irc'] + rowdiff*coldiff)
        icc_sum += p*(gauss['icc'] + coldiff*coldiff)

        psum += p


    T_sum = irr_sum + icc_sum
    T = T_sum/psum

    e1 = (icc_sum - irr_sum)/T_sum
    e2 = 2.0*irc_sum/T_sum

    return e1, e2, T

@njit(cache=True)
def gmix_set_norms(gmix):
    """
    set all norms for gaussians in the input gaussian mixture

    parameters
    ----------
    gmix:
       gaussian mixture

    returns
    -------
    status. 0 for failure
    """
    for gauss in gmix:
        status=gauss2d_set_norm(gauss)
        if status != 1:
            break

    return status

@njit(cache=True)
def gauss2d_set_norm(gauss):
    """
    set the normalization, and nromalized variances

    A GMixRangeError is raised if the determinant is too small

    parameters
    ----------
    gauss: a 2-d gaussian structure
        See gmix.py
    returns
    -------
    status. 0 for failure
    """

    if gauss['det'] > 1.0e-200:

        idet=1.0/gauss['det']
        gauss['drr'] = gauss['irr']*idet
        gauss['drc'] = gauss['irc']*idet
        gauss['dcc'] = gauss['icc']*idet
        gauss['norm'] = 1./(2*numpy.pi*numpy.sqrt(gauss['det']))

        gauss['pnorm'] = gauss['p']*gauss['norm']

        gauss['norm_set']=1
        status=1
    else:
        status=0

    return status

@njit(cache=True)
def gauss2d_set(gauss,
                p,
                row, col,
                irr, irc, icc):
    """
    set the gaussian, clearing normalizations
    """
    gauss['norm_set']=0
    gauss['drr']=nan
    gauss['drc']=nan
    gauss['dcc']=nan
    gauss['norm']=nan
    gauss['pnorm']=nan

    gauss['p']=p
    gauss['row']=row
    gauss['col']=col
    gauss['irr']=irr
    gauss['irc']=irc
    gauss['icc']=icc

    gauss['det'] = irr*icc - irc*irc


@njit(cache=True)
def gmix_fill_simple(gmix, pars, fvals, pvals):
    """
    fill a simple (6 parameter) gaussian mixture model

    no error checking done here
    """

    row  = pars[0]
    col  = pars[1]
    g1   = pars[2]
    g2   = pars[3]
    T    = pars[4]
    flux = pars[5]

    e1, e2, status = g1g2_to_e1e2(g1, g2)
    if status == 0:
        return 0

    n_gauss=gmix.size
    for i in xrange(n_gauss):

        gauss = gmix[i]

        T_i_2 = 0.5*T*fvals[i]
        flux_i=flux*pvals[i]

        gauss2d_set(
            gauss,
            flux_i,
            row,
            col, 
            T_i_2*(1-e1), 
            T_i_2*e2,
            T_i_2*(1+e1),
        )

    return 1

@njit(cache=True)
def g1g2_to_e1e2(g1, g2):
    """
    convert g to e
    """

    g=numpy.sqrt(g1*g1 + g2*g2)

    if g >= 1:
        return -9999.0, -9999.0, 0

    if g == 0.0:
        e1=0
        e2=0
    else:

        eta = 2*numpy.arctanh(g)
        e = numpy.tanh(eta)
        if e >= 1.:
            e = 0.99999999

        fac = e/g

        e1 = fac*g1
        e2 = fac*g2

    return e1, e2, 1

