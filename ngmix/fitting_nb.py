from numba import njit

from .gmix_nb import (
    gmix_eval_pixel_fast,
    gmix_set_norms,
)

@njit
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

    if gmix['norm_set'][0] == 0:
        gmix_set_norms(gmix)

    npix = 0
    loglike = s2n_numer = s2n_denom = 0.0

    n_pixels = pixels.shape[0]
    for ipixel in range(n_pixels):
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

@njit
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

    if gmix['norm_set'][0] == 0:
        gmix_set_norms(gmix)

    n_pixels = pixels.shape[0]
    for ipixel in range(n_pixels):
        pixel = pixels[ipixel]

        model_val = gmix_eval_pixel_fast(gmix, pixel)
        fdiff[start+ipixel] = (model_val-pixel['val'])*pixel['ierr']

@njit
def finish_fdiff(pixels, fdiff, start):
    """
    fill fdiff array (model-data)/err

    parameters
    ----------
    fdiff: gaussian mixture
        this is assumed to corrently hold the model value, will
        be converted to (model-data)/err
    pixels: array if pixel structs
        u,v,val,ierr
    fdiff: array
        Array to fill, should be same length as pixels
    """

    n_pixels = pixels.shape[0]
    for ipixel in range(n_pixels):
        pixel = pixels[ipixel]

        model_val = fdiff[start+ipixel]

        fdiff[start+ipixel] = (model_val-pixel['val'])*pixel['ierr']


@njit
def update_model_array(gmix, pixels, arr, start):
    """
    fill 1d array, adding to existing pixels

    parameters
    ----------
    gmix: gaussian mixture
        See gmix.py
    pixels: array if pixel structs
        u,v,val,ierr
    arr: array
        Array to fill
    """

    if gmix['norm_set'][0] == 0:
        gmix_set_norms(gmix)

    n_pixels = pixels.shape[0]
    for ipixel in range(n_pixels):
        pixel = pixels[ipixel]

        model_val = gmix_eval_pixel_fast(gmix, pixel)
        arr[start+ipixel] += model_val

@njit
def get_model_s2n_sum(gmix, pixels):
    """
    get the model s/n sum.

    The s/n is then sqrt(s2n_sum)

    parameters
    ----------
    gmix: gaussian mixture
        See gmix.py
    pixels: array if pixel structs
        u,v,val,ierr

    returns
    -------
    s2n_sum: float
        sum to calculate s/n
    """

    if gmix['norm_set'][0] == 0:
        gmix_set_norms(gmix)

    n_pixels = pixels.shape[0]
    s2n_sum = 0.0

    for ipixel in range(n_pixels):
        pixel = pixels[ipixel]

        model_val = gmix_eval_pixel_fast(gmix, pixel)
        ivar = pixel['ierr']*pixel['ierr']

        s2n_sum += model_val*model_val*ivar

    return s2n_sum


