import numpy
from .gexceptions import GMixFatalError

def make_pixels(image, weight, jacob, trim=False):
    """
    make a pixel array from the image and weight
    """
    from .pixels_nb import fill_pixels

    if trim:
        w=numpy.where(weight > 0.0)
        if w[0].size == 0:
            raise GMixFatalError('no weights > 0')
        npixels = w[0].size
    else:
        npixels = image.size

    pixels = numpy.zeros(npixels, dtype=_pixels_dtype)

    fill_pixels(
        pixels,
        image,
        weight,
        jacob._data,
        trim=trim,
    )

    return pixels

def make_coords(dims, jacob):
    """
    make a coords array
    """
    from .pixels_nb import fill_coords

    nrow, ncol = dims

    coords = numpy.zeros(nrow*ncol, dtype=_coords_dtype)

    fill_coords(
        coords,
        nrow,
        ncol,
        jacob._data,
    )

    return coords

_pixels_dtype=[
    ('u','f8'),
    ('v','f8'),
    ('val','f8'),
    ('ierr','f8'),
    ('fdiff','f8'),
]


_coords_dtype=[
    ('u','f8'),
    ('v','f8'),
]
