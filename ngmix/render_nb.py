from numba import njit
from .gmix_nb import (
    gmix_eval_pixel,
    gmix_eval_pixel_fast,
    gmix_set_norms,
)

try:
    xrange
except NameError:
    xrange=range

@njit
def render(gmix, coords, image, fast_exp=0, max_chi2=300.0):
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
    fast_exp: integer, optional
        1 for fast
    max_chi2: float, optional
        If fast_exp is 1, this is the maximum chi^2 to
        be evaluated
    """

    if gmix['norm_set'][0] == 0:
        gmix_set_norms(gmix)

    n_coords = coords.shape[0]

    if fast_exp:
        for icoord in xrange(n_coords):
            image[icoord] += gmix_eval_pixel_fast(
                gmix,
                coords[icoord],
                max_chi2=max_chi2,
            )
    else:
        for icoord in xrange(n_coords):
            image[icoord] += gmix_eval_pixel(gmix, coords[icoord])
