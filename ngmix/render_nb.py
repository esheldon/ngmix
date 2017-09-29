import numpy
from numba import jit, njit
from .gmix_nb import gmix_eval_pixel_extended

try:
    xrange
except:
    xrange=range

@njit(cache=True)
def render(gmix, coords, image):
    """
    render the gaussian mixture in the image

    parameters
    ----------
    gmix:
        The gaussian mixture.  norm is not checked
    coords:  array of coords
        The coords, holding location information
    image:
        the image to fill, should be unraveled
    """

    n_coords = coords.shape[0]
    for icoord in xrange(n_coords):
        image[icoord] = gmix_eval_pixel_extended(gmix, coords[icoord])
