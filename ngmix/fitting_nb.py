from numba import njit

from .gmix_nb import gmix_eval_pixel

try:
    xrange
except:
    xrange=range

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
