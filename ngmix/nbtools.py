from __future__ import print_function
import os
import numpy
import time
import numba
from numba import njit, prange

try:
    xrange
except:
    xrange=range

# will check > -26 and < 0.0 so these are not actually necessary
_exp3_ivals = numpy.array([
    -26, -25, -24, -23, -22, -21, 
    -20, -19, -18, -17, -16, -15, -14,
    -13, -12, -11, -10,  -9,  -8,  -7,
    -6,  -5,  -4,  -3,  -2,  -1,   0,
])

_exp3_i0=-26

_exp3_lookup = numpy.array([
    5.10908903e-12,   1.38879439e-11,   3.77513454e-11,
    1.02618796e-10,   2.78946809e-10,   7.58256043e-10,
    2.06115362e-09,   5.60279644e-09,   1.52299797e-08,
    4.13993772e-08,   1.12535175e-07,   3.05902321e-07,
    8.31528719e-07,   2.26032941e-06,   6.14421235e-06,
    1.67017008e-05,   4.53999298e-05,   1.23409804e-04,
    3.35462628e-04,   9.11881966e-04,   2.47875218e-03,
    6.73794700e-03,   1.83156389e-02,   4.97870684e-02,
    1.35335283e-01,   3.67879441e-01,   1.00000000e+00,
])

@njit(cache=True)
def exp3(x):
    """
    fast exponential

    x: number
        any number
    """
    ival = int(x-0.5)
    f = x - ival
    index = ival-_exp3_i0
    expval = _exp3_lookup[index]
    expval *= (6+f*(6+f*(3+f)))*0.16666666

    return expval

@njit(cache=True)
def jacobian_get_vu(jacob, row, col):
    """
    convert row,col to v,u using the input jacobian
    """

    rowdiff = row - jacob['row0'][0]
    coldiff = col - jacob['col0'][0]

    u = jacob['dudrow'][0]*rowdiff + jacob['dudcol'][0]*coldiff
    v = jacob['dvdrow'][0]*rowdiff + jacob['dvdcol'][0]*coldiff

    return v,u
 
@njit(cache=True)
def fill_pixels(pixels, image, weight, jacob):
    """
    store v,u image value, and 1/err for each pixel

    store into 1-d pixels array

    parameters
    ----------
    pixels: array
        1-d array of pixel structures, u,v,val,ierr
    image: 2-d array
        2-d image array
    weight: 2-d array
        2-d image array same shape as image
    jacob: jacobian structure
        row0,col0,dvdrow,dvdcol,dudrow,dudcol,...
    """
    nrow, ncol = image.shape

    ipixel=0
    for row in xrange(nrow):
        for col in xrange(ncol):

            pixel = pixels[ipixel]

            u,v = jacobian_get_vu(jacob,row,col)

            pixel['u'] = u
            pixel['v'] = v

            pixel['val'] = image[row,col]
            ivar = weight[row,col]

            if ivar < 0.0:
                ivar = 0.0

            pixel['ierr'] = numpy.sqrt(ivar)

            ipixel += 1

@njit(cache=True)
def gauss2d_set_norm(gauss2d,):
    """
    set the normalization, and nromalized variances

    parameters
    ----------
    gauss2d: a 2-d gaussian structure
        See gmix.py
    """
    status=0

    if gauss2d['det'] > 1.0e-200:

        idet=1.0/gauss2d['det']
        gauss2d['drr'] = gauss2d['irr']*idet
        gauss2d['drc'] = gauss2d['irc']*idet
        gauss2d['dcc'] = gauss2d['icc']*idet
        gauss2d['norm'] = 1./(2*numpy.pi*numpy.sqrt(gauss2d['det']))

        gauss2d['pnorm'] = gauss2d['p']*gauss2d['norm']

        gauss2d['norm_set']=1
        status=1

    return status

@njit(cache=True)
def gmix_set_norms(gmix,):
    """
    set all norms for gaussians in the input gaussian mixture
    """
    status=0
    n_gauss=gmix.shape[0]

    for i in xrange(n_gauss):

        status=gauss2d_set_norm(gmix[i])
        if status != 1:
            break

    return status



@njit(cache=True)
def gauss2d_eval(gauss, v, u,):
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
    vdiff = v - gauss['row']
    udiff = u - gauss['col']

    chi2 = (      gauss['dcc']*vdiff*vdiff
            +     gauss['drr']*udiff*udiff
            - 2.0*gauss['drc']*vdiff*udiff )

    if chi2 < 25.0 and chi2 >= 0.0:
        model_val = gauss['pnorm']*exp3( -0.5*chi2 )

    return model_val

@njit(cache=True)
def gmix_eval(gmix, pixel):
    """
    evaluate a single gaussian mixture
    """
    model_val=0.0
    for igauss in xrange(gmix.shape[0]):

        model_val += gauss2d_eval(
            gmix[igauss],
            pixel['v'],
            pixel['u'],
        )


    return model_val

@njit(cache=True)
def get_loglike(gmix, pixels):
    """
    get the log likelihood

    parameters
    ----------
    gmix: gaussian mixture
        See gmix.py
    pixels: array if pixel structs
        u,v,val,ierr
    """

    loglike = 0.0

    n_pixels = pixels.shape[0]
    for ipixel in xrange(n_pixels):
        pixel = pixels[ipixel]

        model_val = gmix_eval(gmix, pixel)

        diff = model_val-pixel['val']
        loglike += diff*diff*pixel['ierr']*pixel['ierr']

    loglike *= (-0.5);

    return loglike

@njit(cache=True)
def fill_fdiff(gmix, pixels, fdiff, start):
    """
    fill fdiff array (model-data)/err

    parameters
    ----------
    gmix: gaussian mixture
        See gmix.py
    pixels: array if pixel structs
        u,v,val,ierr
    fdiff: array
        Array to fill, should be same length as pixels
    """

    if gmix['norm_set'][0] != 1:
        status=gmix_set_norms(gmix)
    else:
        status=1

    if status != 0:
        n_pixels = pixels.shape[0]
        for ipixel in xrange(n_pixels):
            pixel = pixels[ipixel]

            model_val = gmix_eval(gmix, pixel)
            fdiff[start+ipixel] = (model_val-pixel['val'])*pixel['ierr']

    return status

