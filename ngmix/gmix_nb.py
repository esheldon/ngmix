import numpy
from numpy import array, nan
from numba import njit
from .fastexp_nb import exp3

# need to make this a pure python exception
from .gexceptions import GMixRangeError

GMIX_LOW_DETVAL=1.0e-200

@njit
def gmix_eval_pixel_fast(gmix, pixel, max_chi2=25.0):
    """
    evaluate a single gaussian mixture, using the
    fast exponential
    """
    model_val=0.0
    for igauss in xrange(gmix.size):

        model_val += gauss2d_eval_pixel_fast(
            gmix[igauss],
            pixel,
            max_chi2,
        )


    return model_val

@njit
def gauss2d_eval_pixel_fast(gauss, pixel, max_chi2=25.0):
    """
    evaluate a 2-d gaussian at the specified location, using
    the fast exponential

    parameters
    ----------
    gauss2d: gauss2d structure
        row,col,dcc,drr,drc,pnorm... See gmix.py
    v,u: numbers
        location in v,u plane (row,col for simple transforms)
    """
    model_val=0.0

    # v->row, u->col in gauss
    vdiff = pixel['v'] - gauss['row']
    udiff = pixel['u'] - gauss['col']

    chi2 = (      gauss['dcc']*vdiff*vdiff
            +     gauss['drr']*udiff*udiff
            - 2.0*gauss['drc']*vdiff*udiff )

    if chi2 < max_chi2 and chi2 >= 0.0:
        model_val = gauss['pnorm']*exp3( -0.5*chi2 )

    return model_val

@njit
def gauss2d_eval_pixel(gauss, pixel):
    """
    evaluate a 2-d gaussian at the specified location

    parameters
    ----------
    gauss2d: gauss2d structure
        row,col,dcc,drr,drc,pnorm... See gmix.py
    pixel: struct with coods
        should have fields v,u
    """
    model_val=0.0

    # v->row, u->col in gauss
    vdiff = pixel['v'] - gauss['row']
    udiff = pixel['u'] - gauss['col']

    chi2 = (      gauss['dcc']*vdiff*vdiff
            +     gauss['drr']*udiff*udiff
            - 2.0*gauss['drc']*vdiff*udiff )

    model_val = gauss['pnorm']*numpy.exp( -0.5*chi2 )

    return model_val

@njit
def gmix_eval_pixel(gmix, pixel):
    """
    evaluate a single gaussian mixture
    """
    model_val=0.0
    for igauss in xrange(gmix.size):

        model_val += gauss2d_eval_pixel(
            gmix[igauss],
            pixel,
        )

    return model_val



@njit
def gmix_get_cen(gmix):
    """
    get the center of the gaussian mixture, as well as
    the psum
    """
    row=0.0
    col=0.0
    psum=0.0

    n_gauss=gmix.size
    for i in xrange(n_gauss):
        gauss=gmix[i]

        p = gauss['p']
        row += p*gauss['row']
        col += p*gauss['col']
        psum += p

    row /= psum
    col /= psum

    return row, col, psum

@njit
def gmix_get_e1e2T(gmix):
    """
    get e1,e2,T for the gaussian mixture
    """

    row, col, psum0 = gmix_get_cen(gmix)

    if psum0 == 0.0:
        raise GMixRangeError("cannot calculate T due to zero psum")

    n_gauss=gmix.size
    psum=0.0
    irr_sum=0.0
    irc_sum=0.0
    icc_sum=0.0

    for i in xrange(n_gauss):
        gauss = gmix[i]

        p = gauss['p']

        rowdiff = gauss['row']-row
        coldiff = gauss['col']-col

        irr_sum += p*(gauss['irr'] + rowdiff*rowdiff)
        irc_sum += p*(gauss['irc'] + rowdiff*coldiff)
        icc_sum += p*(gauss['icc'] + coldiff*coldiff)

        psum += p


    T_sum = irr_sum + icc_sum
    T = T_sum/psum

    if T_sum <= 0.0:
        raise GMixRangeError("T <= 0.0")

    e1 = (icc_sum - irr_sum)/T_sum
    e2 = 2.0*irc_sum/T_sum

    return e1, e2, T

@njit
def gmix_set_norms(gmix):
    """
    set all norms for gaussians in the input gaussian mixture

    parameters
    ----------
    gmix:
       gaussian mixture
    """
    for gauss in gmix:
        gauss2d_set_norm(gauss)

@njit
def gauss2d_set_norm(gauss):
    """
    set the normalization, and nromalized variances

    A GMixRangeError is raised if the determinant is too small

    parameters
    ----------
    gauss: a 2-d gaussian structure
        See gmix.py
    """

    if gauss['det'] < GMIX_LOW_DETVAL:
        raise GMixRangeError("det too low")

    T=gauss['irr']+gauss['icc']
    if T <= GMIX_LOW_DETVAL:
        raise GMixRangeError("T too low")

    idet=1.0/gauss['det']
    gauss['drr'] = gauss['irr']*idet
    gauss['drc'] = gauss['irc']*idet
    gauss['dcc'] = gauss['icc']*idet
    gauss['norm'] = 1./(2*numpy.pi*numpy.sqrt(gauss['det']))

    gauss['pnorm'] = gauss['p']*gauss['norm']

    gauss['norm_set']=1

@njit
def gauss2d_set(gauss,
                p,
                row, col,
                irr, irc, icc):
    """
    set the gaussian, clearing normalizations
    """
    gauss['norm_set']=0
    gauss['drr']=nan
    gauss['drc']=nan
    gauss['dcc']=nan
    gauss['norm']=nan
    gauss['pnorm']=nan

    gauss['p']=p
    gauss['row']=row
    gauss['col']=col
    gauss['irr']=irr
    gauss['irc']=irc
    gauss['icc']=icc

    gauss['det'] = irr*icc - irc*irc

_pvals_exp = array([
    0.00061601229677880041, 
    0.0079461395724623237, 
    0.053280454055540001, 
    0.21797364640726541, 
    0.45496740582554868, 
    0.26521634184240478,
])

_fvals_exp = array([
    0.002467115141477932, 
    0.018147435573256168, 
    0.07944063151366336, 
    0.27137669897479122, 
    0.79782256866993773, 
    2.1623306025075739,
])

_pvals_dev = array([
    6.5288960012625658e-05,
    0.00044199216814302695, 
    0.0020859587871659754, 
    0.0075913681418996841, 
    0.02260266219257237, 
    0.056532254390212859, 
    0.11939049233042602, 
    0.20969545753234975, 
    0.29254151133139222, 
    0.28905301416582552,
])

_fvals_dev = array([
    3.068330909892871e-07,
    3.551788624668698e-06,
    2.542810833482682e-05,
    0.0001466508940804874,
    0.0007457199853069548,
    0.003544702600428794,
    0.01648881157673708,
    0.07893194619504579,
    0.4203787615506401,
    3.055782252301236,
])

_pvals_turb = array([
    0.596510042804182,
    0.4034898268889178,
    1.303069003078001e-07,
])

_fvals_turb = array([
    0.5793612389470884,
    1.621860687127999,
    7.019347162356363,
])

_pvals_gauss = array([1.0])
_fvals_gauss = array([1.0])


@njit
def gmix_fill_simple(gmix, pars, fvals, pvals):
    """
    fill a simple (6 parameter) gaussian mixture model

    no error checking done here
    """

    row  = pars[0]
    col  = pars[1]
    g1   = pars[2]
    g2   = pars[3]
    T    = pars[4]
    flux = pars[5]

    e1, e2 = g1g2_to_e1e2(g1, g2)

    n_gauss=gmix.size
    for i in xrange(n_gauss):

        gauss = gmix[i]

        T_i_2 = 0.5*T*fvals[i]
        flux_i=flux*pvals[i]

        gauss2d_set(
            gauss,
            flux_i,
            row,
            col, 
            T_i_2*(1-e1), 
            T_i_2*e2,
            T_i_2*(1+e1),
        )

@njit
def gmix_fill_exp(gmix, pars):
    """
    fill an exponential model
    """
    gmix_fill_simple(gmix, pars, _fvals_exp, _pvals_exp)

@njit
def gmix_fill_dev(gmix, pars):
    """
    fill a dev model
    """
    gmix_fill_simple(gmix, pars, _fvals_dev, _pvals_dev)

@njit
def gmix_fill_turb(gmix, pars):
    """
    fill a turbulent psf model
    """
    gmix_fill_simple(gmix, pars, _fvals_turb, _pvals_turb)

@njit
def gmix_fill_gauss(gmix, pars):
    """
    fill a gaussian model
    """
    gmix_fill_simple(gmix, pars, _fvals_gauss, _pvals_gauss)


@njit
def gmix_fill_coellip(gmix, pars):
    """
    fill a coelliptical model

    [cen1,cen2,g1,g2,T1,T2,...,F1,F2...]
    """

    row=pars[0]
    col=pars[1]
    g1=pars[2]
    g2=pars[3]

    e1,e2 = g1g2_to_e1e2(g1, g2)

    n_gauss=gmix.size

    for i in xrange(n_gauss):
        T = pars[4+i]
        Thalf=0.5*T
        flux=pars[4+n_gauss+i]

        gauss2d_set(
            gmix[i],
            flux,
            row,
            col, 
            Thalf*(1-e1), 
            Thalf*e2,
            Thalf*(1+e1),
        )


@njit
def gmix_fill_full(gmix, pars):
    """
    fill a "full" gmix model, parameters are specified
    for each gaussian independently
    """

    n_gauss=gmix.size
    for i in xrange(n_gauss):
        beg=i*6

        gauss2d_set(
            gmix[i],
            pars[beg+0],
            pars[beg+1],
            pars[beg+2],
            pars[beg+3],
            pars[beg+4],
            pars[beg+5],
        )


@njit
def gmix_fill_cm(gmix, fracdev, TdByTe, Tfactor, pars):
    """
    fill a composite model
    """

    row  = pars[0]
    col  = pars[1]
    g1   = pars[2]
    g2   = pars[3]
    T    = pars[4] * Tfactor
    flux = pars[5]


    ifracdev = 1.0-fracdev

    e1, e2 = g1g2_to_e1e2(g1, g2)

    for i in xrange(16):
        if i < 6:
            p = _pvals_exp[i] * ifracdev
            f = _fvals_exp[i]
        else:
            p = _pvals_dev[i-6] * fracdev
            f = _fvals_dev[i-6] * TdByTe

        T_i_2  = 0.5*T*f
        flux_i = flux*p

        gauss2d_set(
            gmix[i],
            flux_i,
            row,
            col, 
            T_i_2*(1-e1), 
            T_i_2*e2,
            T_i_2*(1+e1),
        )

@njit
def gmix_fill_bdf(gmix, pars):
    """
    fill a composite model with fixed Td/Te=1 but fracdev
    varying
    """

    TdByTe=1.0

    row     = pars[0]
    col     = pars[1]
    g1      = pars[2]
    g2      = pars[3]
    T       = pars[4]
    fracdev = pars[5]
    flux    = pars[6]

    Tfactor  = get_cm_Tfactor(fracdev, TdByTe)
    T = T*Tfactor

    ifracdev = 1.0-fracdev

    e1, e2 = g1g2_to_e1e2(g1, g2)

    for i in xrange(16):
        if i < 6:
            p = _pvals_exp[i] * ifracdev
            f = _fvals_exp[i]
        else:
            p = _pvals_dev[i-6] * fracdev
            f = _fvals_dev[i-6] * TdByTe

        T_i_2  = 0.5*T*f
        flux_i = flux*p

        gauss2d_set(
            gmix[i],
            flux_i,
            row,
            col, 
            T_i_2*(1-e1), 
            T_i_2*e2,
            T_i_2*(1+e1),
        )


@njit
def get_cm_Tfactor(fracdev, TdByTe):
    """
    get the factor needed to convert T to the T needed
    for using in filling a cmodel gaussian mixture

    parameters
    ----------
    fracdev: float
        fraction of flux in the dev component
    TdByTe: float
        T_{dev}/T_{exp}
    """

    ifracdev = 1.0-fracdev

    Tfactor = 0.0

    for i in xrange(6):
        p = _pvals_exp[i] * ifracdev
        f = _fvals_exp[i]

        Tfactor += p*f

    for i in xrange(10):
        p = _pvals_dev[i] * fracdev
        f = _fvals_dev[i] * TdByTe

        Tfactor += p*f

    Tfactor = 1.0/Tfactor

    return Tfactor

_gmix_fill_functions={
    'exp': gmix_fill_exp,
    'dev': gmix_fill_dev,
    'turb': gmix_fill_turb,
    'gauss': gmix_fill_gauss,
    'cm': gmix_fill_cm,
    'bdf': gmix_fill_bdf,
    'coellip': gmix_fill_coellip,
    'full':gmix_fill_full,
}

@njit
def gmix_convolve_fill(self, gmix, psf):
    """
    fill the gaussian mixture with the convolution of gmix0,
    the unconvolved mixture, and the psf

    parameters
    ----------
    self: gaussian mixture
        The convolved mixture, to be filled
    gmix: gaussian mixture
        The unconvolved mixture
    psf: gaussian mixture
        The psf with which to convolve
    """

    psf_rowcen, psf_colcen, psf_psum = gmix_get_cen(psf)

    psf_ipsum   = 1.0/psf_psum
    n_gauss     = gmix.size
    psf_n_gauss = psf.size

    itot=0
    for iobj in xrange(n_gauss):
        obj_gauss = gmix[iobj]

        for ipsf in xrange(psf_n_gauss):
            psf_gauss=psf[ipsf]

            p = obj_gauss['p']*psf_gauss['p']*psf_ipsum

            row = obj_gauss['row'] + (psf_gauss['row']-psf_rowcen)
            col = obj_gauss['col'] + (psf_gauss['col']-psf_colcen)

            irr = obj_gauss['irr'] + psf_gauss['irr']
            irc = obj_gauss['irc'] + psf_gauss['irc']
            icc = obj_gauss['icc'] + psf_gauss['icc']

            gauss2d_set(self[itot],
                        p, row, col, irr, irc, icc)

            itot += 1

@njit
def g1g2_to_e1e2(g1, g2):
    """
    convert g to e
    """

    g=numpy.sqrt(g1*g1 + g2*g2)

    if g >= 1:
        raise GMixRangeError("g >= 1")

    if g == 0.0:
        e1=0.0
        e2=0.0
    else:

        eta = 2*numpy.arctanh(g)
        e = numpy.tanh(eta)
        if e >= 1.:
            e = 0.99999999

        fac = e/g

        e1 = fac*g1
        e2 = fac*g2

    return e1, e2



@njit
def get_weighted_sums(wt, pixels, res, maxrad):
    """
    do sums for calculating the weighted moments
    """

    maxrad2=maxrad**2

    vcen = wt['row'][0]
    ucen = wt['col'][0]
    F = res['F']

    n_pixels = pixels.size
    for i_pixel in xrange(n_pixels):

        pixel = pixels[i_pixel]

        vmod = pixel['v']-vcen
        umod = pixel['u']-ucen

        rad2 = umod*umod + vmod*vmod
        if rad2 < maxrad2:

            weight = gmix_eval_pixel(wt, pixel)
            var = 1.0/(pixel['ierr']*pixel['ierr'])

            wdata = weight*pixel['val']
            w2 = weight*weight


            F[0] = pixel['v']
            F[1] = pixel['u']
            F[2] = umod*umod - vmod*vmod
            F[3] = 2*vmod*umod
            F[4] = rad2
            F[5] = 1.0

            res['wsum'] += weight
            res['npix'] += 1

            for i in xrange(6):
                res['sums'][i] += wdata*F[i]
                for j in xrange(6):
                    res['sums_cov'][i,j] += w2*var*F[i]*F[j]



