import numpy
from numba import njit

try:
    xrange
except:
    xrange=range

from .gmix_nb import gmix_eval_pixel_fast, GMIX_LOW_DETVAL

ADMOM_EDGE   = 0x1
ADMOM_SHIFT  = 0x2
ADMOM_FAINT  = 0x4
ADMOM_SMALL  = 0x8
ADMOM_DET    = 0x10
ADMOM_MAXIT  = 0x20

@njit(cache=True)
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
    res  = resarray[0]

    roworig=wt['row'][0]
    colorig=wt['col'][0]

    for i in xrange(conf['maxit']):

        if wt['det'][0] < GMIX_LOW_DETVAL:
            res['flags'] = ADMOM_DET
            break

        # due to check above, this should not raise an exception
        gmix_set_norms(wt)

        clear_result(res)
        admom_censums(wt, pixels, res)

        if res['sums'][5] <= 0.0:
            res['flags'] = ADMOM_FAINT
            break

        wt['row'][0] = res['sums'][0]/res['sums'][5]
        wt['col'][0] = res['sums'][1]/res['sums'][5]

        if ( abs(wt['row'][0]-roworig) > conf['shiftmax']
                 or abs(wt['col'][0]-colorig) > conf['shiftmax'] ):
            res['flags'] = ADMOM_SHIFT
            break

        clear_result(res)
        admom_momsums(wt, pixels, res)

        if res['sums'][5] <= 0.0:
            res['flags'] = ADMOM_FAINT
            break

        # look for convergence
        finv = 1.0/res['sums'][5]
        M1 = res['sums'][2]*finv
        M2 = res['sums'][3]*finv
        T  = res['sums'][4]*finv

        Irr = 0.5*(T - M1)
        Icc = 0.5*(T + M1)
        Irc = 0.5*M2

        if T <= 0.0:
            res['flags'] = ADMOM_SMALL
            break

        e1 = (Icc - Irr)/T
        e2 = 2*Irc/T

        if ( 
                ( abs(e1-e1old) < self['etol'])
              and
                ( abs(e2-e2old) < self['etol'])
              and
                ( abs(T/Told-1.) < self['Ttol'])  ):

            res['pars'][0] = wt['row'][0]
            res['pars'][1] = wt['col'][0]
            res['pars'][2] = wt['icc'][0] - wt['irr'][0]
            res['pars'][3] = 2.0*wt['irc'][0]
            res['pars'][4] = wt['icc'][0] + wt['irr'][0]
            res['pars'][5] = 1.0

            break

        else:
            # deweight moments and go to the next iteration

            deweight_moments(wt, Irr, Irc, Icc, res)
            if res['flags'] != 0:
                break

            e1old=e1
            e2old=e2
            Told=T

    res['numiter'] = i

    if res['numiter'] == conf['maxit']:
        res['flags'] = ADMOM_MAXIT


@njit(cache=True)
def admom_censums(wt, pixels, res):
    """
    do sums for determining the center
    """

    n_pixels = pixels.size
    for i in xrange(n_pixels):

        pixel = pixels[i]
        weight = gmix_eval_pixel_fast(wt, pixel)

        wdata = weight*pixel['val']

        res['npix'] += 1
        res['sums'][0] += wdata*pixel['v']
        res['sums'][1] += wdata*pixel['u']
        res['sums'][5] += wdata

@njit(cache=True)
def admom_momsums(wt, pixels, res):
    """
    do sums for calculating the weighted moments
    """

    vcen = wt['row'][0]
    ucen = wt['col'][0]
    F = res['F']

    n_pixels = pixels.size
    for i_pixel in xrange(n_pixels):

        pixel = pixels[i_pixel]
        weight = gmix_eval_pixel_fast(wt, pixel)

        var = 1.0/(pixel['ierr']*pixel['ierr'])

        wdata = weight*pixel['val']
        w2 = weight*weight

        res['npix'] += 1
        res['sums'][0] += wdata*pixel['v']
        res['sums'][1] += wdata*pixel['u']
        res['sums'][5] += wdata
        res['wsum']    += weight

        vmod = pixel['v']-vcen
        umod = pixel['u']-ucen

        F[0] = pixel['v']
        F[1] = pixel['u']
        F[2] = umod*umod - vmod*vmod
        F[3] = 2*vmod*umod
        F[4] = umod*umod + vmod*vmod
        F[5] = 1.0

        for i in xrange(6):
            res['sums'][i] += wdata*F[i]
            for j in xrange(6):
                res['sums_cov'][i,j] += w2*var*F[i]*F[j]


@njit(cache=True)
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
        res['flags'] = ADMOM_DET
        return

    Wrr = wt['irr'][0]
    Wrc = wt['irc'][0]
    Wcc = wt['icc'][0]
    detw = Wrr*Wcc - Wrc*Wrc
    if detw <= GMIX_LOW_DETVAL:
        res['flags']=ADMOM_DET
        return

    idetw=1.0/detw
    idetm=1.0/detm

    # Nrr etc. are actually of the inverted covariance matrix
    Nrr =  Icc*idetm - Wcc*idetw
    Ncc =  Irr*idetm - Wrr*idetw
    Nrc = -Irc*idetm + Wrc*idetw
    detn = Nrr*Ncc - Nrc*Nrc

    if detn <= GMIX_LOW_DETVAL:
        res['flags']=ADMOM_DET
        return

    # now set from the inverted matrix
    idetn=1./detn
    wt['irr'][0] =  Ncc*idetn
    wt['icc'][0] =  Nrr*idetn
    wt['irc'][0] = -Nrc*idetn
    wt['det'][0] = (
        wt['irr'][0]*wt['icc'][0] - wt['irc'][0]*wt['irc'][0]
    )


@njit(cache=True)
def clear_result(res):
    """
    clear some fields in the result structure
    """
    res['npix']=0
    res['wsum']=0.0
    res['sums'][:] = 0.0
    res['sums_cov'][:,:] = 0.0
    res['pars'][:] = -9999.0
