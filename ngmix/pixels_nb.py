import numpy
from numba import njit

from .jacobian_nb import jacobian_get_vu

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

            v,u = jacobian_get_vu(jacob,row,col)

            pixel['v'] = v
            pixel['u'] = u

            pixel['val'] = image[row,col]
            ivar = weight[row,col]

            if ivar < 0.0:
                ivar = 0.0

            pixel['ierr'] = numpy.sqrt(ivar)

            ipixel += 1


@njit(cache=True)
def fill_coords(coords, nrow, ncol, jacob):
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

    icoord=0
    for row in xrange(nrow):
        for col in xrange(ncol):

            coord = coords[icoord]

            v,u = jacobian_get_vu(jacob,row,col)

            coord['v'] = v
            coord['u'] = u

            icoord += 1



