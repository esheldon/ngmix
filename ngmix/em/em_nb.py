import numpy as np
from numba import njit
from ..gexceptions import GMixRangeError
from ..gmix.gmix_nb import (
    gauss2d_set,
    gmix_set_norms,
    gmix_get_cen,
    gmix_convolve_fill,
    gmix_eval_pixel_fast,
    GMIX_LOW_DETVAL,
)
from ..fastexp_nb import fexp


@njit
def em_run(conf,
           pixels,
           sums,
           gmix,
           gmix_psf,
           gmix_conv,
           fill_zero_weight=False):
    """
    run the EM algorithm

    Parameters
    ----------
    conf: array
        Should have fields

            tol: The fractional change in the log likelihood that implies
                convergence
            miniter: minimum number of iterations
            maxiter: maximum number of iterations
            sky: the sky, or guess for sky if fitting for it
            vary_sky: True if fitting for the sky

    pixels: pixel array
        for the image/jacobian
    sums: array with fields
        The sums array, a type _sums_dtype
    gmix: gauss2d array
        The initial mixture.  The final result is also stored in this array.
    gmix_psf: gauss2d array
        Single gaussian psf
    gmix_conv: gauss2d array
        Convolved gmix
    fill_zero_weight: bool
        If True, fill the zero weight pixels with the model on
        each iteration

    Returns
    -------
    numiter, frac_diff, sky
        number of iterations, fractional difference in log likelihood between
        last two steps, and sky.  The sky may have been fit for.
    """

    gmix_set_norms(gmix_conv)
    ngauss_psf = gmix_psf.size

    taudata = np.zeros(gmix_conv.size, dtype=_tau_dtype)

    tol = conf['tol']

    npix = pixels.size

    sky = conf['sky']

    elogL_last = -9999.9e9

    for i in range(conf['maxiter']):

        elogL = 0.0
        skysum = 0.0

        clear_sums(sums)
        set_logtau_logdet(gmix_conv, taudata)

        if fill_zero_weight:
            fill_zero_weight_pixels(gmix_conv, pixels, sky)

        for pixel in pixels:

            gsum, tlogL = do_scratch_sums(
                pixel, gmix_conv, sums, ngauss_psf,
                taudata,
            )

            gtot = gsum + sky
            if gtot == 0.0:
                raise GMixRangeError('gtot == 0')

            elogL += tlogL

            skysum += sky*pixel['val']/gtot

            do_sums(sums, pixel, gtot)

        gmix_set_from_sums(
            gmix,
            gmix_psf,
            gmix_conv,
            sums,
        )

        if conf['vary_sky']:
            sky = skysum/npix

        numiter = i+1
        if numiter >= conf['miniter']:

            if elogL == 0.0:
                raise GMixRangeError('elogL == 0')

            frac_diff = abs((elogL - elogL_last)/elogL)
            if frac_diff < tol:
                break

        elogL_last = elogL

    # we have modified the mixture and not set the norms, and we don't want to
    # set them for the pre-psf mixture

    gmix['norm_set'][:] = 0

    return numiter, frac_diff, sky


@njit
def clear_sums(sums):
    """
    set all sums to zero

    Parameters
    ----------
    sums: array
        Array with dtype _sums_dtype
    """
    sums['gi'][:] = 0.0

    sums['tusum'][:] = 0.0
    sums['tvsum'][:] = 0.0

    sums['tu2sum'][:] = 0.0
    sums['tuvsum'][:] = 0.0
    sums['tv2sum'][:] = 0.0

    # sums over all pixels
    sums['pnew'][:] = 0.0

    sums['usum'][:] = 0.0
    sums['vsum'][:] = 0.0

    sums['u2sum'][:] = 0.0
    sums['uvsum'][:] = 0.0
    sums['v2sum'][:] = 0.0


@njit
def do_scratch_sums(pixel, gmix_conv, sums, ngauss_psf, taudata):
    """
    do the basic sums for this pixel, using scratch space in the sums struct

    we may have multiple components per "object" so we update the
    sums accordingly

    Parameters
    ----------
    pixel: pixel structure
        Dtype should be ngmix.pixels._pixels_dtype
    gmix_conv: ngmix.GMix
        The current mixture, convolved by the PSF
    sums: sums structure
        With dtype _sums_dtype
    ngauss_psf: int
        Number of gaussians in psf
    taudata: tau data struct
        With dtype _tau_dtype

    Returns
    -------
    gsum, logL:
        The total of the gaussians evaluated in the pixel and the
        log likelihood
    """

    v = pixel['v']
    u = pixel['u']

    gsum = 0.0
    logL = 0.0

    ngauss = gmix_conv.size//ngauss_psf

    for ii in range(ngauss):
        tsums = sums[ii]
        tsums['gi'] = 0.0
        tsums['tvsum'] = 0.0
        tsums['tusum'] = 0.0
        tsums['tv2sum'] = 0.0
        tsums['tuvsum'] = 0.0
        tsums['tu2sum'] = 0.0

        start = ii*ngauss_psf
        end = (ii+1)*ngauss_psf

        for i in range(start, end):
            gauss = gmix_conv[i]
            ttau = taudata[i]

            # in practice we have co-centric gaussians even after psf
            # convolution, so we could move these to the outer loop

            vdiff = v - gauss['row']
            udiff = u - gauss['col']

            u2 = udiff*udiff
            v2 = vdiff*vdiff
            uv = udiff*vdiff

            chi2 = gauss['dcc']*v2 + gauss['drr']*u2 - 2.0*gauss['drc']*uv

            if chi2 < 25.0 and chi2 >= 0.0:
                val = gauss['pnorm']*fexp(-0.5*chi2) * pixel['area']
            else:
                val = 0.0

            tsums['gi'] += val
            gsum += val

            tsums['tvsum'] += v*val
            tsums['tusum'] += u*val

            tsums['tv2sum'] += v2*val
            tsums['tuvsum'] += uv*val
            tsums['tu2sum'] += u2*val

            logL += val*(ttau['logtau'] - 0.5*ttau['logdet'] - 0.5*chi2)

    if gsum == 0.0:
        logL = 0.0
    else:
        logL *= 1.0/gsum

    return gsum, logL


@njit
def do_sums(sums, pixel, gtot):
    """
    do the sums based on the scratch values

    Parameters
    ----------
    sums: sums structure
        With dtype _sums_dtype
    pixel: pixel structure
        Dtype should be ngmix.pixels._pixels_dtype
    gtot: float
        The sum over gaussian values
    """

    factor = pixel['val']/gtot

    n_gauss = sums.size
    for i in range(n_gauss):
        tsums = sums[i]

        wtau = tsums['gi']*factor

        tsums['pnew'] += wtau

        # row*gi/gtot*imnorm

        tsums['usum'] += tsums['tusum']*factor
        tsums['vsum'] += tsums['tvsum']*factor

        tsums['u2sum'] += tsums['tu2sum']*factor
        tsums['uvsum'] += tsums['tuvsum']*factor
        tsums['v2sum'] += tsums['tv2sum']*factor


@njit
def gmix_set_from_sums(gmix,
                       gmix_psf,
                       gmix_conv,
                       sums):
    """
    fill the gaussian mixture from the em sums

    Parameters
    ----------
    gmix: ngmix.GMix
        The gaussian mixture before psf convolution
    gmix_psf: ngmix.GMix
        The gaussian mixture of the PSF (can be zero size for no psf)
    gmix_conv: ngmix.GMix
        The final convolved mixture
    sums: sums structure
        With dtype _sums_dtype
    """

    # minval = 1.0e-4
    # minval = -1.0e-1

    _, _, psf_irr, psf_irc, psf_icc, _ = gmix_get_moms(gmix_psf)
    psf_T = psf_irr + psf_icc
    minval = -0.5*psf_T/2

    n_gauss = gmix.size
    for i in range(n_gauss):

        tsums = sums[i]
        gauss = gmix[i]

        p = tsums['pnew']
        pinv = 1.0/p

        # update for convolved gaussian
        v = tsums['vsum']*pinv
        u = tsums['usum']*pinv
        irr = tsums['v2sum']*pinv
        irc = tsums['uvsum']*pinv
        icc = tsums['u2sum']*pinv

        # get pre-psf moments, only works if the psf gaussians
        # are all centered
        irr = irr - psf_irr
        irc = irc - psf_irc
        icc = icc - psf_icc

        # currently are forcing the sizes of pre-psf gaussians to
        # be positive.  We may be able to relax this

        # if irr < 0.0 or icc < 0.0:
        if irr < minval or icc < minval:
            irr, irc, icc = minval, 0.0, minval

        # this causes oscillations in likelihood
        det = irr*icc - irc**2
        if det < GMIX_LOW_DETVAL:
            T = irr + icc
            irr = icc = T/2
            irc = 0.0

        gauss2d_set(
            gauss,
            p,
            v,
            u,
            irr,
            irc,
            icc,
        )

    gmix_convolve_fill(gmix_conv, gmix, gmix_psf)
    gmix_set_norms(gmix_conv)


@njit
def em_run_fixcen(conf,
                  pixels,
                  sums,
                  gmix,
                  gmix_psf,
                  gmix_conv,
                  fill_zero_weight=False):
    """
    run the EM algorithm with fixed positions

    Parameters
    ----------
    conf: array
        Should have fields

            tol: The fractional change in the log likelihood that implies
                convergence
            miniter: minimum number of iterations
            maxiter: maximum number of iterations
            sky: the sky, or guess for sky if fitting for it
            vary_sky: True if fitting for the sky

    pixels: pixel array
        for the image/jacobian
    sums: array with fields
        The sums array, a type _sums_dtype_fixcen
    gmix: gauss2d array
        The initial mixture.  The final result is also stored in this array.
    gmix_psf: gauss2d array
        Single gaussian psf
    gmix_conv: gauss2d array
        Convolved gmix
    fill_zero_weight: bool
        If True, fill the zero weight pixels with the model on
        each iteration

    Returns
    -------
    numiter, frac_diff, sky
        number of iterations, fractional difference in log likelihood between
        last two steps, and sky.  The sky may have been fit for.
    """

    gmix_set_norms(gmix_conv)
    ngauss_psf = gmix_psf.size

    taudata = np.zeros(gmix_conv.size, dtype=_tau_dtype)

    tol = conf['tol']

    npix = pixels.size

    sky = conf['sky']

    elogL_last = -9999.9e9

    for i in range(conf['maxiter']):

        elogL = 0.0
        skysum = 0.0

        clear_sums_fixcen(sums)
        set_logtau_logdet(gmix_conv, taudata)

        if fill_zero_weight:
            fill_zero_weight_pixels(gmix_conv, pixels, sky)

        for pixel in pixels:

            gsum, tlogL = do_scratch_sums_fixcen(
                pixel, gmix_conv, sums, ngauss_psf,
                taudata,
            )

            gtot = gsum + sky
            if gtot == 0.0:
                raise GMixRangeError('gtot == 0')

            elogL += tlogL

            skysum += sky*pixel['val']/gtot

            do_sums_fixcen(sums, pixel, gtot)

        gmix_set_from_sums_fixcen(
            gmix,
            gmix_psf,
            gmix_conv,
            sums,
        )

        if conf['vary_sky']:
            sky = skysum/npix

        numiter = i+1
        if numiter >= conf['miniter']:

            if elogL == 0.0:
                raise GMixRangeError('elogL == 0')

            frac_diff = abs((elogL - elogL_last)/elogL)
            if frac_diff < tol:
                break

        elogL_last = elogL

    # we have modified the mixture and not set the norms, and we don't want to
    # set them for the pre-psf mixture

    gmix['norm_set'][:] = 0

    return numiter, frac_diff, sky


@njit
def do_scratch_sums_fixcen(pixel, gmix_conv, sums, ngauss_psf, taudata):
    """
    do the basic sums for this pixel, using scratch space in the sums struct

    we may have multiple components per "object" so we update the
    sums accordingly

    Parameters
    ----------
    pixel: pixel structure
        Dtype should be ngmix.pixels._pixels_dtype
    gmix_conv: ngmix.GMix
        The current mixture, convolved by the PSF
    sums: sums structure
        With dtype _sums_dtype_fixcen
    ngauss_psf: int
        Number of gaussians in psf
    taudata: tau data struct
        With dtype _tau_dtype

    Returns
    -------
    gsum, logL:
        The total of the gaussians evaluated in the pixel and the
        log likelihood
    """

    v = pixel['v']
    u = pixel['u']

    gsum = 0.0
    logL = 0.0

    ngauss = gmix_conv.size//ngauss_psf

    for ii in range(ngauss):
        tsums = sums[ii]
        tsums['gi'] = 0.0
        tsums['tv2sum'] = 0.0
        tsums['tuvsum'] = 0.0
        tsums['tu2sum'] = 0.0

        start = ii*ngauss_psf
        end = (ii+1)*ngauss_psf

        for i in range(start, end):
            gauss = gmix_conv[i]
            ttau = taudata[i]

            # in practice we have co-centric gaussians even after psf
            # convolution, so we could move these to the outer loop

            vdiff = v - gauss['row']
            udiff = u - gauss['col']

            u2 = udiff*udiff
            v2 = vdiff*vdiff
            uv = udiff*vdiff

            chi2 = gauss['dcc']*v2 + gauss['drr']*u2 - 2.0*gauss['drc']*uv

            if chi2 < 25.0 and chi2 >= 0.0:
                val = gauss['pnorm']*fexp(-0.5*chi2) * pixel['area']
            else:
                val = 0.0

            tsums['gi'] += val
            gsum += val

            tsums['tv2sum'] += v2*val
            tsums['tuvsum'] += uv*val
            tsums['tu2sum'] += u2*val

            logL += val*(ttau['logtau'] - 0.5*ttau['logdet'] - 0.5*chi2)

    if gsum == 0.0:
        logL = 0.0
    else:
        logL *= 1.0/gsum

    return gsum, logL


@njit
def do_sums_fixcen(sums, pixel, gtot):
    """
    do the sums based on the scratch values

    Parameters
    ----------
    sums: sums structure
        With dtype _sums_dtype_fixcen
    pixel: pixel structure
        Dtype should be ngmix.pixels._pixels_dtype
    gtot: float
        The sum over gaussian values
    """

    factor = pixel['val']/gtot

    n_gauss = sums.size
    for i in range(n_gauss):
        tsums = sums[i]

        wtau = tsums['gi']*factor

        tsums['pnew'] += wtau

        # row*gi/gtot*imnorm
        tsums['u2sum'] += tsums['tu2sum']*factor
        tsums['uvsum'] += tsums['tuvsum']*factor
        tsums['v2sum'] += tsums['tv2sum']*factor


@njit
def gmix_set_from_sums_fixcen(gmix,
                              gmix_psf,
                              gmix_conv,
                              sums):
    """
    fill the gaussian mixture from the em sums

    Parameters
    ----------
    gmix: ngmix.GMix
        The gaussian mixture before psf convolution
    gmix_psf: ngmix.GMix
        The gaussian mixture of the PSF (can be zero size for no psf)
    gmix_conv: ngmix.GMix
        The final convolved mixture
    sums: sums structure
        With dtype _sums_dtype_fixcen
    """

    minval = 1.0e-4

    _, _, psf_irr, psf_irc, psf_icc, _ = gmix_get_moms(gmix_psf)

    n_gauss = gmix.size
    for i in range(n_gauss):

        tsums = sums[i]
        gauss = gmix[i]

        p = tsums['pnew']
        pinv = 1.0/p

        # update for convolved gaussian
        irr = tsums['v2sum']*pinv
        irc = tsums['uvsum']*pinv
        icc = tsums['u2sum']*pinv

        # get pre-psf moments, only works if the psf gaussians
        # are all centered
        irr = irr - psf_irr
        irc = irc - psf_irc
        icc = icc - psf_icc

        # currently are forcing the sizes of pre-psf gaussians to
        # be positive.  We may be able to relax this

        if irr < 0.0 or icc < 0.0:
            irr, irc, icc = minval, 0.0, minval

        # this causes oscillations in likelihood
        det = irr*icc - irc**2
        if det < GMIX_LOW_DETVAL:
            T = irr + icc
            irr = icc = T/2
            irc = 0.0

        gauss2d_set(
            gauss,
            p,
            gauss['row'],
            gauss['col'],
            irr,
            irc,
            icc,
        )

    gmix_convolve_fill(gmix_conv, gmix, gmix_psf)
    gmix_set_norms(gmix_conv)


@njit
def set_logtau_logdet(gmix, sums):
    """
    set log(tau) and log(det) for every gaussian

    Parameters
    -----------
    gmix: ngmix.Gmix
        gaussian Mixture
    sums: array
        Array with sums
    """

    for i in range(gmix.size):
        gauss = gmix[i]
        tsums = sums[i]
        tsums['logtau'] = np.log(gauss['p'])
        tsums['logdet'] = np.log(gauss['det'])


@njit
def clear_sums_fixcen(sums):
    """
    set all sums to zero

    Parameters
    ----------
    sums: array
        Array with dtype _sums_dtype_fixcen
    """
    sums['gi'][:] = 0.0

    sums['tu2sum'][:] = 0.0
    sums['tuvsum'][:] = 0.0
    sums['tv2sum'][:] = 0.0

    # sums over all pixels
    sums['pnew'][:] = 0.0
    sums['u2sum'][:] = 0.0
    sums['uvsum'][:] = 0.0
    sums['v2sum'][:] = 0.0


# start fixcov
@njit
def em_run_fixcov(
    conf,
    pixels,
    sums,
    gmix,
    gmix_psf,
    gmix_conv,
    fill_zero_weight=False,
):
    """
    run the EM algorithm

    Parameters
    ----------
    conf: array
        Should have fields

            tol: The fractional change in the log likelihood that implies
                convergence
            miniter: minimum number of iterations
            maxiter: maximum number of iterations
            sky: the sky, or guess for sky if fitting for it
            vary_sky: True if fitting for the sky

    pixels: pixel array
        for the image/jacobian
    sums: array with fields
        The sums array, a type _sums_dtype_fixcov
    gmix: gauss2d array
        The initial mixture.  The final result is also stored in this array.
    gmix_psf: gauss2d array
        Single gaussian psf
    gmix_conv: gauss2d array
        Convolved gmix
    fill_zero_weight: bool
        If True, fill the zero weight pixels with the model on
        each iteration

    Returns
    -------
    numiter, frac_diff, sky
        number of iterations, fractional difference in log likelihood between
        last two steps, and sky.  The sky may have been fit for.
    """

    gmix_set_norms(gmix_conv)
    ngauss_psf = gmix_psf.size

    taudata = np.zeros(gmix_conv.size, dtype=_tau_dtype)

    tol = conf['tol']

    npix = pixels.size

    sky = conf['sky']

    elogL_last = -9999.9e9

    for i in range(conf['maxiter']):

        elogL = 0.0
        skysum = 0.0

        clear_sums_fixcov(sums)
        set_logtau_logdet(gmix_conv, taudata)

        if fill_zero_weight:
            fill_zero_weight_pixels(gmix_conv, pixels, sky)

        for pixel in pixels:

            gsum, tlogL = do_scratch_sums_fixcov(
                pixel, gmix_conv, sums, ngauss_psf,
                taudata,
            )

            gtot = gsum + sky
            if gtot == 0.0:
                raise GMixRangeError('gtot == 0')

            elogL += tlogL

            skysum += sky*pixel['val']/gtot

            do_sums_fixcov(sums, pixel, gtot)

        gmix_set_from_sums_fixcov(
            gmix,
            gmix_psf,
            gmix_conv,
            sums,
        )

        if conf['vary_sky']:
            sky = skysum/npix

        numiter = i+1
        if numiter >= conf['miniter']:

            if elogL == 0.0:
                raise GMixRangeError('elogL == 0')

            frac_diff = abs((elogL - elogL_last)/elogL)
            if frac_diff < tol:
                break

        elogL_last = elogL

    # we have modified the mixture and not set the norms, and we don't want to
    # set them for the pre-psf mixture

    gmix['norm_set'][:] = 0

    return numiter, frac_diff, sky


@njit
def clear_sums_fixcov(sums):
    """
    set all sums to zero

    Parameters
    ----------
    sums: array
        Array with dtype _sums_dtype
    """
    sums['gi'][:] = 0.0

    sums['tusum'][:] = 0.0
    sums['tvsum'][:] = 0.0

    # sums over all pixels
    sums['pnew'][:] = 0.0

    sums['usum'][:] = 0.0
    sums['vsum'][:] = 0.0


@njit
def do_scratch_sums_fixcov(pixel, gmix_conv, sums, ngauss_psf, taudata):
    """
    do the basic sums for this pixel, using scratch space in the sums struct

    we may have multiple components per "object" so we update the
    sums accordingly

    Parameters
    ----------
    pixel: pixel structure
        Dtype should be ngmix.pixels._pixels_dtype
    gmix_conv: ngmix.GMix
        The current mixture, convolved by the PSF
    sums: sums structure
        With dtype _sums_dtype
    ngauss_psf: int
        Number of gaussians in psf
    taudata: tau data struct
        With dtype _tau_dtype

    Returns
    -------
    gsum, logL:
        The total of the gaussians evaluated in the pixel and the
        log likelihood
    """

    v = pixel['v']
    u = pixel['u']

    gsum = 0.0
    logL = 0.0

    ngauss = gmix_conv.size//ngauss_psf

    for ii in range(ngauss):
        tsums = sums[ii]
        tsums['gi'] = 0.0
        tsums['tvsum'] = 0.0
        tsums['tusum'] = 0.0

        start = ii*ngauss_psf
        end = (ii+1)*ngauss_psf

        for i in range(start, end):
            gauss = gmix_conv[i]
            ttau = taudata[i]

            # in practice we have co-centric gaussians even after psf
            # convolution, so we could move these to the outer loop

            vdiff = v - gauss['row']
            udiff = u - gauss['col']

            u2 = udiff*udiff
            v2 = vdiff*vdiff
            uv = udiff*vdiff

            chi2 = gauss['dcc']*v2 + gauss['drr']*u2 - 2.0*gauss['drc']*uv

            if chi2 < 25.0 and chi2 >= 0.0:
                val = gauss['pnorm']*fexp(-0.5*chi2) * pixel['area']
            else:
                val = 0.0

            tsums['gi'] += val
            gsum += val

            tsums['tvsum'] += v*val
            tsums['tusum'] += u*val

            logL += val*(ttau['logtau'] - 0.5*ttau['logdet'] - 0.5*chi2)

    if gsum == 0.0:
        logL = 0.0
    else:
        logL *= 1.0/gsum

    return gsum, logL


@njit
def do_sums_fixcov(sums, pixel, gtot):
    """
    do the sums based on the scratch values

    Parameters
    ----------
    sums: sums structure
        With dtype _sums_dtype
    pixel: pixel structure
        Dtype should be ngmix.pixels._pixels_dtype
    gtot: float
        The sum over gaussian values
    """

    factor = pixel['val']/gtot

    n_gauss = sums.size
    for i in range(n_gauss):
        tsums = sums[i]

        wtau = tsums['gi']*factor

        tsums['pnew'] += wtau

        # row*gi/gtot*imnorm

        tsums['usum'] += tsums['tusum']*factor
        tsums['vsum'] += tsums['tvsum']*factor


@njit
def gmix_set_from_sums_fixcov(
    gmix,
    gmix_psf,
    gmix_conv,
    sums,
):
    """
    fill the gaussian mixture from the em sums

    Parameters
    ----------
    gmix: ngmix.GMix
        The gaussian mixture before psf convolution
    gmix_psf: ngmix.GMix
        The gaussian mixture of the PSF (can be zero size for no psf)
    gmix_conv: ngmix.GMix
        The final convolved mixture
    sums: sums structure
        With dtype _sums_dtype
    """

    n_gauss = gmix.size
    for i in range(n_gauss):

        tsums = sums[i]
        gauss = gmix[i]

        p = tsums['pnew']
        pinv = 1.0/p

        # update for convolved gaussian
        v = tsums['vsum']*pinv
        u = tsums['usum']*pinv

        gauss2d_set(
            gauss,
            p,
            v,
            u,
            gauss['irr'],
            gauss['irc'],
            gauss['icc'],
        )

    gmix_convolve_fill(gmix_conv, gmix, gmix_psf)
    gmix_set_norms(gmix_conv)

# end fixcov


@njit
def em_run_fluxonly(
    conf,
    pixels,
    sums,
    gmix,
    gmix_psf,
    gmix_conv,
    fill_zero_weight=False,
):
    """
    run the EM algorithm, allowing only fluxes to vary

    Parameters
    ----------
    conf: array
        Should have fields
            tol: The fractional change in the log likelihood that implies
                convergence
            miniter: minimum number of iterations
            maxiter: maximum number of iterations
            sky: the sky, or guess for sky if fitting for it
            vary_sky: True if fitting for the sky

    pixels: pixel array
        for the image/jacobian
    sums: array with fields
        The sums array, a type _sums_dtype_fluxonly
    gmix: gauss2d array
        The initial mixture.  The final result is also stored in this array.
    gmix_psf: gauss2d array
        Single gaussian psf
    gmix_conv: gauss2d array
        Convolved gmix
    fill_zero_weight: bool
        If True, fill the zero weight pixels with the model on
        each iteration

    Returns
    -------
    numiter, frac_diff, sky
        number of iterations, fractional difference in log likelihood between
        last two steps, and sky.  The sky may have been fit for.
    """

    gmix_set_norms(gmix_conv)
    ngauss_psf = gmix_psf.size

    tol = conf['tol']

    npix = pixels.size

    sky = conf['sky']

    p_last = gmix['p'].sum()

    for i in range(conf['maxiter']):
        skysum = 0.0
        clear_sums_fluxonly(sums)

        if fill_zero_weight:
            fill_zero_weight_pixels(gmix_conv, pixels, sky)

        for pixel in pixels:

            # this fills some fields of sums, as well as return
            gsum = do_scratch_sums_fluxonly(pixel, gmix_conv, sums, ngauss_psf)

            gtot = gsum + sky
            if gtot == 0.0:
                raise GMixRangeError("gtot == 0")

            skysum += sky*pixel['val']/gtot

            do_sums_fluxonly(sums, pixel, gtot)

        gmix_set_from_sums_fluxonly(
            gmix,
            gmix_psf,
            gmix_conv,
            sums,
        )

        if conf['vary_sky']:
            sky = skysum/npix

        psum = gmix['p'].sum()

        numiter = i+1
        if numiter >= conf['miniter']:
            frac_diff = abs(psum/p_last-1)
            if frac_diff < tol:
                break

        p_last = psum

    # we have modified the mixture and not set the norms, and we don't want to
    # set them for the pre-psf mixture

    gmix['norm_set'][:] = 0

    return numiter, frac_diff, sky


@njit
def do_scratch_sums_fluxonly(pixel, gmix_conv, sums, ngauss_psf):
    """
    do the basic sums for this pixel, using
    scratch space in the sums struct

    Parameters
    ----------
    pixel: pixel structure
        Dtype should be ngmix.pixels._pixels_dtype
    gmix_conv: ngmix.GMix
        The current mixture, convolved by the PSF
    sums: sums structure
        With dtype _sums_dtype_fluxonly
    ngauss_psf: int
        Number of gaussians in psf

    Returns
    -------
    gsum, logL:
        The total of the gaussians evaluated in the pixel and the
        log likelihood
    """

    v = pixel['v']
    u = pixel['u']

    gsum = 0.0

    ngauss = gmix_conv.size//ngauss_psf

    for ii in range(ngauss):
        tsums = sums[ii]
        tsums['gi'] = 0.0

        start = ii*ngauss_psf
        end = (ii+1)*ngauss_psf

        for i in range(start, end):
            gauss = gmix_conv[i]

            # in practice we have co-centric gaussians even after psf
            # convolution, so we could move these to the outer loop

            vdiff = v-gauss['row']
            udiff = u-gauss['col']

            u2 = udiff*udiff
            v2 = vdiff*vdiff
            uv = udiff*vdiff

            chi2 = gauss['dcc']*v2 + gauss['drr']*u2 - 2.0*gauss['drc']*uv

            if chi2 < 25.0 and chi2 >= 0.0:
                val = gauss['pnorm']*fexp(-0.5*chi2) * pixel['area']
            else:
                val = 0.0

            tsums['gi'] += val
            gsum += val

    return gsum


@njit
def do_sums_fluxonly(sums, pixel, gtot):
    """
    do the sums based on the scratch values

    Parameters
    ----------
    sums: sums structure
        With dtype _sums_dtype_fluxonly
    pixel: pixel structure
        Dtype should be ngmix.pixels._pixels_dtype
    gtot: float
        The sum over gaussian values
    """

    factor = pixel['val']/gtot

    n_gauss = sums.size
    for i in range(n_gauss):
        tsums = sums[i]

        # gi*image_val/sum(gi)
        wtau = tsums['gi']*factor

        tsums['pnew'] += wtau


@njit
def gmix_set_from_sums_fluxonly(
    gmix,
    gmix_psf,
    gmix_conv,
    sums,
):
    """
    fill the gaussian mixture from the em sums

    Parameters
    ----------
    gmix: ngmix.GMix
        The gaussian mixture before psf convolution
    gmix_psf: ngmix.GMix
        The gaussian mixture of the PSF (can be zero size for no psf)
    gmix_conv: ngmix.GMix
        The final convolved mixture
    sums: sums structure
        With dtype _sums_dtype_fluxonly
    """

    n_gauss = gmix.size
    for i in range(n_gauss):

        tsums = sums[i]
        gauss = gmix[i]

        p = tsums['pnew']

        gauss2d_set(
            gauss,
            p,
            gauss['row'],
            gauss['col'],
            gauss['irr'],
            gauss['irc'],
            gauss['icc'],
        )

    gmix_convolve_fill(gmix_conv, gmix, gmix_psf)
    gmix_set_norms(gmix_conv)


@njit
def clear_sums_fluxonly(sums):
    """
    set all sums to zero

    Parameters
    ----------
    sums: array
        Array with dtype _sums_dtype_fluxonly
    """
    sums['gi'][:] = 0.0

    # sums over all pixels
    sums['pnew'][:] = 0.0


@njit
def gmix_get_moms(gmix):
    """
    get row, col, irr, irc, icc, psum

    Parameters
    ----------
    gmix: ngmix.Gmix
        Gaussian mixture

    Returns
    -------
    return row, col, irr, irc, icc, psum
    """

    row, col, psum = gmix_get_cen(gmix)

    irr = irc = icc = 0.0

    for i in range(gmix.size):
        gauss = gmix[i]

        rowdiff = gauss['row'] - row
        coldiff = gauss['col'] - col

        p = gauss['p']
        irr += p*(gauss['irr'] + rowdiff**2)
        irc += p*(gauss['irc'] + rowdiff*coldiff)
        icc += p*(gauss['icc'] + coldiff**2)

    irr /= psum
    irc /= psum
    icc /= psum

    return row, col, irr, irc, icc, psum


@njit
def fill_zero_weight_pixels(gmix, pixels, sky):
    """
    fill zero weight pixels with the model

    Paramters
    ---------
    gmix: ngmix.Gmix
        The mixture with which to evaluate in the zero weight pixels
    pixels: array of pixels
        The pixels to be modified
    sky: float
        Value of the sky in the pixels
    """

    for pixel in pixels:
        if pixel['ierr'] <= 0.0:
            val = gmix_eval_pixel_fast(gmix, pixel)
            pixel['val'] = sky + val


_tau_dtype = [
    ('logtau', 'f8'),
    ('logdet', 'f8'),
]
_tau_dtype = np.dtype(_tau_dtype)
