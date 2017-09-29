from __future__ import print_function
import os
import numpy
import time
import numba
from numba import jit, njit, prange

from .gmix_nb import gmix_eval_pixel
from .jacobian_nb import jacobian_get_vu

try:
    xrange
except:
    xrange=range

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

        model_val = gmix_eval_pixel(gmix, pixel)

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

    n_pixels = pixels.shape[0]
    for ipixel in xrange(n_pixels):
        pixel = pixels[ipixel]

        model_val = gmix_eval_pixel(gmix, pixel)
        fdiff[start+ipixel] = (model_val-pixel['val'])*pixel['ierr']
