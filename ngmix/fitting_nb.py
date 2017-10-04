from numba import njit

from .gmix_nb import gmix_eval_pixel_fast

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

    returns
    -------
    a tuple of

    loglike: float
        log likelihood
    s2n_numer: float
        numerator for s/n
    s2n_demon: float
        will use sqrt(s2n_denom) for denominator for s/n
    npix: int
        number of pixels used
    """

    npix = 0
    loglike = s2n_numer = s2n_denom = 0.0

    n_pixels = pixels.shape[0]
    for ipixel in xrange(n_pixels):
        pixel = pixels[ipixel]

        model_val = gmix_eval_pixel_fast(gmix, pixel)

        ivar = pixel['ierr']*pixel['ierr']
        val  = pixel['val']
        diff = model_val-val

        loglike += diff*diff*ivar

        s2n_numer += val * model_val * ivar
        s2n_denom += model_val * model_val * ivar
        npix += 1

    loglike *= (-0.5)

    return loglike, s2n_numer, s2n_denom, npix

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

        model_val = gmix_eval_pixel_fast(gmix, pixel)
        fdiff[start+ipixel] = (model_val-pixel['val'])*pixel['ierr']
