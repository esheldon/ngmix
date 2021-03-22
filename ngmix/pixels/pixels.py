__all__ = ['make_pixels', 'make_coords']
import numpy
from ..gexceptions import GMixFatalError


def make_pixels(image, weight, jacob, ignore_zero_weight=True):
    """
    make a pixel array from the image and weight

    stores v,u image value, and 1/err for each pixel


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
        If set, zero or negative weight pixels are ignored.  In this case the
        returned pixels array is equal in length to the set of positive weight
        pixels in the weight image.  Default True.

    returns
    -------
    1-d pixels array
    """
    from .pixels_nb import fill_pixels

    if ignore_zero_weight:
        w = numpy.where(weight > 0.0)
        if w[0].size == 0:
            raise GMixFatalError("no weights > 0")
        npixels = w[0].size
    else:
        npixels = image.size

    pixels = numpy.zeros(npixels, dtype=_pixels_dtype)

    fill_pixels(
        pixels,
        image,
        weight,
        jacob._data,
        ignore_zero_weight=ignore_zero_weight,
    )

    return pixels


def make_coords(dims, jacob):
    """
    make a coords array
    """
    from .pixels_nb import fill_coords

    nrow, ncol = dims

    coords = numpy.zeros(nrow * ncol, dtype=_coords_dtype)

    fill_coords(
        coords, nrow, ncol, jacob._data,
    )

    return coords


_pixels_dtype = [
    ("u", "f8"),
    ("v", "f8"),
    ("area", "f8"),
    ("val", "f8"),
    ("ierr", "f8"),
    ("fdiff", "f8"),
]


_coords_dtype = [
    ("u", "f8"),
    ("v", "f8"),
    ("area", "f8"),
]
