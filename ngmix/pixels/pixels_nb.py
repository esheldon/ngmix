import numpy
from numba import njit
from ..jacobian.jacobian_nb import jacobian_get_vu, jacobian_get_area


@njit
def fill_pixels(pixels, image, weight, jacob, ignore_zero_weight=True):
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
    ignore_zero_weight: bool
        If set, zero or negative weight pixels are ignored.
        In this case it verified that the input pixels
        are equal in length to the set of positive weight
        pixels in the weight image.  Default True.
    """
    nrow, ncol = image.shape
    pixel_area = jacobian_get_area(jacob)

    ipixel = 0
    for row in range(nrow):
        for col in range(ncol):

            ivar = weight[row, col]
            if ignore_zero_weight and ivar <= 0.0:
                continue

            pixel = pixels[ipixel]

            v, u = jacobian_get_vu(jacob, row, col)

            pixel['v'] = v
            pixel['u'] = u
            pixel['area'] = pixel_area

            pixel['val'] = image[row, col]

            if ivar < 0.0:
                ivar = 0.0

            pixel['ierr'] = numpy.sqrt(ivar)

            ipixel += 1

    if ipixel != pixels.size:
        raise RuntimeError('some pixels were not filled')


@njit
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

    pixel_area = jacobian_get_area(jacob)

    icoord = 0
    for row in range(nrow):
        for col in range(ncol):

            coord = coords[icoord]

            v, u = jacobian_get_vu(jacob, row, col)

            coord['v'] = v
            coord['u'] = u
            coord['area'] = pixel_area

            icoord += 1
