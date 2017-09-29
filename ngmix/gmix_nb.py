import numpy
from numba import jit, njit
from .fastexp_nb import exp3, exp3_extended

try:
    xrange
except:
    xrange=range

# need to make this a pure python exception
from .gexceptions import GMixRangeError

@njit(cache=True)
def gauss2d_eval_pixel(gauss, pixel):
    """
    evaluate a 2-d gaussian at the specified location

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

    if chi2 < 25.0 and chi2 >= 0.0:
        model_val = gauss['pnorm']*exp3( -0.5*chi2 )

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
def gauss2d_eval_pixel_extended(gauss, pixel):
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

    if chi2 < 300.0 and chi2 >= 0.0:
        model_val = gauss['pnorm']*exp3_extended( -0.5*chi2 )

    return model_val

@njit(cache=True)
def gmix_eval_pixel_extended(gmix, pixel):
    """
    evaluate a single gaussian mixture
    """
    model_val=0.0
    for igauss in xrange(gmix.size):

        model_val += gauss2d_eval_pixel_extended(
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

@njit
def gauss2d_set(gauss,
                p,
                row, col,
                irr, irc, icc):
    """
    set the gaussian, clearing normalizations
    """
    gauss['norm_set']=0
    gauss['drr']=0
    gauss['drc']=0
    gauss['dcc']=0
    gauss['norm']=0
    gauss['pnorm']=0

    gauss['p']=p
    gauss['row']=row
    gauss['col']=col
    gauss['irr']=irr
    gauss['irc']=irc
    gauss['icc']=icc

    gauss['det'] = irr*icc - irc*irc


