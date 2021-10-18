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

    e1old = e2old = Told = np.nan
    for i in range(conf['maxiter']):

        if wt['det'][0] < GMIX_LOW_DETVAL:
            res['flags'] = ngmix.flags.LOW_DET
            break

        # due to check above, this should not raise an exception
        gmix_set_norms(wt)

        clear_result(res)
        admom_censums(wt, pixels, res)

        if res['sums'][5] <= 0.0:
            res['flags'] = ngmix.flags.NONPOS_FLUX
            break

        wt['row'][0] = res['sums'][0]/res['sums'][5]
        wt['col'][0] = res['sums'][1]/res['sums'][5]

        if (abs(wt['row'][0]-roworig) > conf['shiftmax']
                or abs(wt['col'][0]-colorig) > conf['shiftmax']):
            res['flags'] = ngmix.flags.BIG_CEN_SHIFT
            break

        clear_result(res)
        admom_momsums(wt, pixels, res)

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

    if res['numiter'] == conf['maxiter']:
        res['flags'] = ngmix.flags.MAXITER


@njit
def admom_censums(wt, pixels, res):
    """
    do sums for determining the center
    """

    n_pixels = pixels.size
    for i in range(n_pixels):

        pixel = pixels[i]
        weight = gmix_eval_pixel_fast(wt, pixel)

        wdata = weight*pixel['val']

        res['npix'] += 1
        res['sums'][0] += wdata*pixel['v']
        res['sums'][1] += wdata*pixel['u']
        res['sums'][5] += wdata


@njit
def admom_momsums(wt, pixels, res):
    """
    do sums for calculating the weighted moments
    """

    vcen = wt['row'][0]
    ucen = wt['col'][0]
    F = res['F']

    n_pixels = pixels.size
    for i_pixel in range(n_pixels):

        pixel = pixels[i_pixel]
        weight = gmix_eval_pixel_fast(wt, pixel)

        var = 1.0/(pixel['ierr']*pixel['ierr'])

        vmod = pixel['v']-vcen
        umod = pixel['u']-ucen

        wdata = weight*pixel['val']
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
