import numpy as np
from numba import njit

from ..gmix.gmix_nb import (
    gmix_set_norms,
    gmix_eval_pixel_fast,
    GMIX_LOW_DETVAL,
)

import ngmix.flags


@njit
def admom(confarray, wt, pixels, resarray):
    """
    run the adaptive moments algorithm

    parameters
    ----------
    conf: admom config struct
        See admom._admom_conf_dtype
    """
    # to simplify notation
    conf = confarray[0]
    res = resarray[0]

    roworig = wt['row'][0]
    colorig = wt['col'][0]

    min_ierr = get_min_ierr(pixels)
    if min_ierr == np.inf:
        res['flags'] = ngmix.flags.NO_GOOD_PIXELS

    e1old = e2old = Told = np.nan
    for i in range(conf['maxiter']):

        if wt['det'][0] < GMIX_LOW_DETVAL:
            res['flags'] = ngmix.flags.LOW_DET
            break

        # due to check above, this should not raise an exception
        gmix_set_norms(wt)

        if i == 0:
            # first run with min_err=0 ignores zero weight pixels
            use_min_ierr = 0
        else:
            use_min_ierr = min_ierr

        admom_set_flux(wt, pixels, res, use_min_ierr)
        if res['flux_flags'] != 0:
            break

        clear_result(res)
        admom_censums(wt, pixels, res)

        if res['sums'][5] <= 0.0:
            res['flags'] = ngmix.flags.NONPOS_FLUX
            break

        wt['row'][0] = res['sums'][0]/res['sums'][5]
        wt['col'][0] = res['sums'][1]/res['sums'][5]

        if (abs(wt['row'][0]-roworig) > conf['shiftmax']
                or abs(wt['col'][0]-colorig) > conf['shiftmax']):
            res['flags'] = ngmix.flags.CEN_SHIFT
            break

        clear_result(res)
        admom_momsums(wt, pixels, res, min_ierr)

        if res['sums'][5] <= 0.0:
            res['flags'] = ngmix.flags.NONPOS_FLUX
            break

        # look for convergence
        finv = 1.0/res['sums'][5]
        M1 = res['sums'][2]*finv
        M2 = res['sums'][3]*finv
        T = res['sums'][4]*finv

        Irr = 0.5*(T - M1)
        Icc = 0.5*(T + M1)
        Irc = 0.5*M2

        if T <= 0.0:
            res['flags'] = ngmix.flags.NONPOS_SIZE
            break

        e1 = (Icc - Irr)/T
        e2 = 2*Irc/T

        if ((abs(e1-e1old) < conf['etol'])
                and (abs(e2-e2old) < conf['etol'])
                and (abs(T/Told-1.) < conf['Ttol'])):

            res['pars'][0] = wt['row'][0]
            res['pars'][1] = wt['col'][0]
            res['pars'][2] = wt['icc'][0] - wt['irr'][0]
            res['pars'][3] = 2.0*wt['irc'][0]
            res['pars'][4] = wt['icc'][0] + wt['irr'][0]
            res['pars'][5] = 1.0

            break

        else:
            # deweight moments and go to the next iteration

            if not conf['cenonly']:
                deweight_moments(wt, Irr, Irc, Icc, res)
                if res['flags'] != 0:
                    break

            e1old = e1
            e2old = e2
            Told = T

    res['numiter'] = i+1
    admom_set_flux_err(wt, pixels, res)

    if res['numiter'] == conf['maxiter']:
        res['flags'] = ngmix.flags.MAXITER
    if res['flux_flags'] != 0:
        res['flags'] |= res['flux_flags']


@njit
def get_min_ierr(pixels):
    """
    get the min 1/err for pixels with non zero 1/err
    """
    min_ierr = np.inf

    n_pixels = pixels.size
    for i in range(n_pixels):
        pixel = pixels[i]
        ierr = pixel['ierr']
        if ierr > 0 and ierr < min_ierr:
            min_ierr = ierr

    return min_ierr


@njit
def admom_set_flux(wt, pixels, res, min_ierr):
    """
    calculate a tempalte flux using the weight, and filling
    in pixels with the model as needed

    Parameters
    ----------
    wt: gmix
        The weight gmix
    pixels: pixels array
        Array of pixels, possibly with zero 1/ierr
    res: admom result
        The result struct
    min_ierr: float
        The minimum non zero 1/err

    Side Effects
    -----------
    the flux for the weight function and in the res are set
    """
    min_ivar = min_ierr**2

    xcorr_sum = 0.0
    msq_sum = 0.0

    oflux = wt['p'][0]
    if oflux == 0.0:
        oflux = 1.0
        wt['p'][0] = oflux
        gmix_set_norms(wt)

    iflux = 1 / oflux

    n_pixels = pixels.size
    for i in range(n_pixels):

        pixel = pixels[i]

        if pixel['ierr'] <= 0 and min_ierr <= 0:
            # this is most likely a first run at setting the flux,
            # ignore zero weight pixels
            continue

        if pixel['ierr'] < min_ierr:
            ivar = min_ivar
        else:
            ivar = pixel['ierr']**2

        model = gmix_eval_pixel_fast(wt, pixel) * iflux

        if pixel['ierr'] <= 0:
            val = model * oflux
        else:
            val = pixel['val']

        xcorr_sum += model * val * ivar
        msq_sum += model * model * ivar

    if msq_sum <= 0:
        res['flux_flags'] = ngmix.flags.NONPOS_FLUX
        res['flux_flags'] = ngmix.flags.DIV_ZERO
    else:
        flux = xcorr_sum / msq_sum
        if flux <= 0:
            res['flux_flags'] = ngmix.flags.NONPOS_FLUX
        else:
            res['flux'] = flux

            wt['p'][0] = flux
            gmix_set_norms(wt)


@njit
def admom_set_flux_err(wt, pixels, res):
    """
    calculate the tempalte flux error

    Parameters
    ----------
    wt: gmix
        The weight gmix
    pixels: pixels array
        Array of pixels, possibly with zero 1/ierr
    res: admom result
        The result struct

    Side Effects
    -----------
    the flux_err for the res is set, as well as possibly
    flux_flags
    """

    totpix = pixels.size

    chi2sum = 0.0
    msq_sum = 0.0
    flux = wt['p'][0]
    if flux <= 0:
        res['flux_flags'] = ngmix.flags.NONPOS_FLUX
        return

    iflux = 1 / flux

    n_pixels = pixels.size
    for i in range(n_pixels):

        pixel = pixels[i]
        if pixel['ierr'] <= 0:
            # Don't use bad pixels for the error estimate, because we filled in
            # with the model.  Errors will reflect zero information from these
            # pixels, which may be an overestimate
            continue

        ivar = pixel['ierr']**2

        model = gmix_eval_pixel_fast(wt, pixel)

        if pixel['ierr'] <= 0:
            val = model
        else:
            val = pixel['val']

        chi2sum += (model - val)**2 * ivar
        msq_sum += model * model * ivar

    if msq_sum <= 0 or totpix == 1:
        res['flux_flags'] = ngmix.flags.DIV_ZERO
    else:
        # normalize the model squared sum
        msq_sum *= iflux**2
        arg = chi2sum / msq_sum / (totpix - 1)
        if arg >= 0.0:
            res['flux_err'] = np.sqrt(arg)
        else:
            res['flux_flags'] = ngmix.flags.NONPOS_FLUX
            res['flux_err'] = np.nan


@njit
def admom_censums(wt, pixels, res):
    """
    do sums for determining the center
    """

    n_pixels = pixels.size
    for i in range(n_pixels):

        pixel = pixels[i]

        weight = gmix_eval_pixel_fast(wt, pixel)

        # fill in bad pixels with the model
        if pixel['ierr'] <= 0.0:
            val = weight
        else:
            val = pixel['val']

        wdata = weight * val

        res['npix'] += 1
        res['sums'][0] += wdata*pixel['v']
        res['sums'][1] += wdata*pixel['u']
        res['sums'][5] += wdata


@njit
def admom_momsums(wt, pixels, res, min_ierr):
    """
    do sums for calculating the weighted moments
    """

    min_ivar = min_ierr**2

    vcen = wt['row'][0]
    ucen = wt['col'][0]
    F = res['F']

    n_pixels = pixels.size
    for i_pixel in range(n_pixels):

        pixel = pixels[i_pixel]
        weight = gmix_eval_pixel_fast(wt, pixel)

        if pixel['ierr'] < min_ierr:
            var = 1/min_ivar
        else:
            var = 1.0/pixel['ierr']**2

        vmod = pixel['v']-vcen
        umod = pixel['u']-ucen

        # fill in with the model
        if pixel['ierr'] <= 0:
            val = weight
        else:
            val = pixel['val']

        wdata = weight * val
        w2 = weight*weight

        F[0] = pixel['v']
        F[1] = pixel['u']
        F[2] = umod*umod - vmod*vmod
        F[3] = 2*vmod*umod
        F[4] = umod*umod + vmod*vmod
        F[5] = 1.0

        res['wsum'] += weight
        res['npix'] += 1

        for i in range(6):
            res['sums'][i] += wdata*F[i]
            for j in range(6):
                res['sums_cov'][i, j] += w2*var*F[i]*F[j]


@njit
def deweight_moments(wt, Irr, Irc, Icc, res):
    """
    deweight a set of weighted moments

    parameters
    ----------
    wt: gaussian mixture
        The weight used to measure the moments
    Irr, Irc, Icc:
        The weighted moments
    res: admom result struct
        the flags field will be set on error
    """
    # measured moments
    detm = Irr*Icc - Irc*Irc
    if detm <= GMIX_LOW_DETVAL:
        res['flags'] = ngmix.flags.LOW_DET
        return

    Wrr = wt['irr'][0]
    Wrc = wt['irc'][0]
    Wcc = wt['icc'][0]
    detw = Wrr*Wcc - Wrc*Wrc
    if detw <= GMIX_LOW_DETVAL:
        res['flags'] = ngmix.flags.LOW_DET
        return

    idetw = 1.0/detw
    idetm = 1.0/detm

    # Nrr etc. are actually of the inverted covariance matrix
    Nrr = Icc*idetm - Wcc*idetw
    Ncc = Irr*idetm - Wrr*idetw
    Nrc = -Irc*idetm + Wrc*idetw
    detn = Nrr*Ncc - Nrc*Nrc

    if detn <= GMIX_LOW_DETVAL:
        res['flags'] = ngmix.flags.LOW_DET
        return

    # now set from the inverted matrix
    idetn = 1./detn
    wt['irr'][0] = Ncc*idetn
    wt['icc'][0] = Nrr*idetn
    wt['irc'][0] = -Nrc*idetn
    wt['det'][0] = (
        wt['irr'][0]*wt['icc'][0] - wt['irc'][0]*wt['irc'][0]
    )


@njit
def clear_result(res):
    """
    clear some fields in the result structure
    """
    res['npix'] = 0
    res['wsum'] = 0.0
    res['sums'][:] = 0.0
    res['sums_cov'][:, :] = 0.0
    res['pars'][:] = np.nan

    # res['flags']=0
    # res['numiter']=0
    # res['nimage']=0
    # res['F'][:]=0.0
