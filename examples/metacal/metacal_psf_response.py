"""
Metacalibration (https://arxiv.org/abs/1702.02600, https://arxiv.org/abs/1702.02601)

In this example we perform basic metacalibration with no object detection or
object selections, same as metacal.py, but with GalsimFitter and PSF response
correction implemented. The PSF used for this example is the SuperBIT optical design.

We use a bootstrapper to run measurements on the object, with GalsimFitter
(exponential model) for galaxy fitting and adaptive moments for PSF shape
estimation. PSF response correction (R_psf * e_psf) follows Huff & Mandelbaum
(2017) (https://arxiv.org/abs/1702.02600).

The psf parameter controls the reconvolution PSF:
    'dilate'    : dilates the original PSF (default)
    'fitgauss'  : fits and dilates a Gaussian to the original PSF
    'gauss'     : deterministic round Gaussian, slower but more conservative

Types of metacal images generated:
    noshear       : deconvolved/reconvolved, no shear — used for shear estimator
    1p / 1m       : +/- g1 shear — used for shear response R11
    1p_psf/1m_psf : +/- g1 PSF shear — used for PSF response R11_psf

m and c are estimated from a weighted linear fit over 5 input shear values:
    shear_rec = (1 + m) * shear_true + c

Expected output (low noise, no blending, ntrial=1000, psf='dilate'):
    S/N: 73510.3
    R11: 0.999571
    R11_psf: 3.32287e-06
    m: 0.000384931 +/- 0.00038159 (99.7% conf)
    c: 6.24475e-07 +/- 7.63179e-06 (99.7% conf)
"""
import numpy as np
import ngmix
import galsim

# SuperBIT optical design (fixed instrument parameters)
_SBIT_LAM_NM      = 475      # observing wavelength [nm]
_SBIT_TEL_DIAM_M  = 0.5      # telescope diameter [m]
_SBIT_OBSCURATION = 0.380    # central obscuration (fraction of diameter)
_SBIT_NSTRUTS     = 4
_SBIT_STRUT_THICK = 0.087
_SBIT_JITTER_FWHM = 0.467    # pointing-jitter FWHM [arcsec]

# Zernike aberrations in the Noll convention (index 0 ignored; index 1 = piston,
# which has no effect on the PSF image; first shape-changing term is [4]=defocus).
_SBIT_ABERRATIONS = np.zeros(38)
_SBIT_ABERRATIONS[4]  = -0.02474205   # defocus
_SBIT_ABERRATIONS[11] = -0.01544329   # primary spherical
_SBIT_ABERRATIONS[22] =  0.00199235   # secondary spherical
_SBIT_ABERRATIONS[26] =  0.00000017
_SBIT_ABERRATIONS[37] =  0.00000004


def main():
    args = get_args()
    shear_true_vals = [-0.02, 0.02]
    step = 0.02 
    
    psf = bit_psf(psf_g1=-0.02, psf_g2=-0.02)

    rng = np.random.RandomState(args.seed)

    prior   = get_prior(rng=rng, scale=0.141)
    guesser = ngmix.guessers.TFluxAndPriorGuesser(rng=rng, T=0.2, flux=1.0, prior=prior)
    fitter  = ngmix.fitting.GalsimFitter(model='exp', prior=prior)
    runner  = ngmix.runners.Runner(fitter=fitter, guesser=guesser, ntry=20)

    boot = ngmix.metacal.MetacalBootstrapper(
        runner=runner, psf_runner=None, rng=rng,
        psf=args.psf,
        types=['noshear', '1p', '1m', '1p_psf', '1m_psf'],
    )

    shear_rec     = []
    shear_rec_err = []

    for shear_val in shear_true_vals:
        print(f"Running for shear_true = {shear_val:+.2f} ...")
        dlist, g_psf = [], []

        for i in progress(args.ntrial, miniters=10):
            obs = make_data(rng=rng, noise=args.noise, shear=[shear_val, 0.0], psf=psf)
            resdict, obsdict = boot.go(obs)

            for stype, sres in resdict.items():
                dlist.append(make_struct(res=sres, obs=obsdict[stype], shear_type=stype))

            res_psf = get_admoms(obs.psf, rng)
            if res_psf['flags'] == 0:
                g_psf.append([res_psf['e1'], res_psf['e2']])

        print()
        data  = np.hstack(dlist)
        g_psf = np.asarray(g_psf)

        # selection masks
        w        = select(data=data, shear_type='noshear')
        w_1p     = select(data=data, shear_type='1p')
        w_1m     = select(data=data, shear_type='1m')
        w_1p_psf = select(data=data, shear_type='1p_psf')
        w_1m_psf = select(data=data, shear_type='1m_psf')

        # shear response
        R11     = (data['g'][w_1p,     0].mean() - data['g'][w_1m,     0].mean()) / step
        R11_psf = (data['g'][w_1p_psf, 0].mean() - data['g'][w_1m_psf, 0].mean()) / step

        # calibrated shear
        g    = data['g'][w].mean(axis=0)
        gerr = data['g'][w].std(axis=0) / np.sqrt(w.size)

        c1 = (data['g'][w_1p, 0].mean() + data['g'][w_1m, 0].mean()) / 2 - g[0]

        shear     = (g - g_psf.mean(axis=0) * R11_psf - c1) / R11
        shear_err = gerr / R11

        shear_rec.append(shear[0])
        shear_rec_err.append(shear_err[0])

    x  = np.array(shear_true_vals)        # [-0.02, +0.02]
    y  = np.array(shear_rec)
    ye = np.array(shear_rec_err)

    dx = x[1] - x[0]                       # = 0.04

    slope = (y[1] - y[0]) / dx
    c     = (y[1] + y[0]) / 2              # intercept = mean, since x is symmetric about 0
    m     = slope - 1

    merr = np.sqrt(ye[0]**2 + ye[1]**2) / dx
    cerr = np.sqrt(ye[0]**2 + ye[1]**2) / 2

    s2n = data['s2n'][w].mean()
    print('S/N: %g'                     % s2n)
    print('R11: %g'                     % R11)
    print('R11_psf: %g'                 % R11_psf)
    print('m: %g +/- %g (99.7%% conf)' % (m, merr * 3))
    print('c: %g +/- %g (99.7%% conf)' % (c, cerr * 3))

def select(data, shear_type):
    """
    select the data by shear type and size

    Parameters
    ----------
    data: array
        The array with fields shear_type and T
    shear_type: str
        e.g. 'noshear', '1p', etc.

    Returns
    -------
    array of indices
    """

    w, = np.where(
        (data['flags'] == 0) & (data['shear_type'] == shear_type)
    )
    return w


def make_struct(res, obs, shear_type):
    """
    make the data structure

    Parameters
    ----------
    res: dict
        With keys 's2n', 'e', and 'T'
    obs: ngmix.Observation
        The observation for this shear type
    shear_type: str
        The shear type

    Returns
    -------
    1-element array with fields
    """
    dt = [
        ('flags', 'i4'),
        ('shear_type', 'U7'),
        ('s2n', 'f8'),
        ('g', 'f8', 2),
        ('hlr', 'f8'),
    ]
    data = np.zeros(1, dtype=dt)
    data['shear_type'] = shear_type
    data['flags'] = res['flags']
    if res['flags'] == 0:
        data['s2n'] = res['s2n_r']
        # for moments we are actually measureing e, the elliptity
        data['g'] = res['g']
        data['hlr'] = res['pars'][4]
    else:
        data['s2n'] = np.nan
        data['g'] = np.nan
        data['T'] = np.nan

    return data


def get_prior(*, rng, scale, T_range=None, F_range=None, nband=None):
    """
    get a prior for use with the maximum likelihood fitter

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    scale: float
        Pixel scale
    T_range: (float, float), optional
        The range for the prior on T
    F_range: (float, float), optional
        Fhe range for the prior on flux
    nband: int, optional
        number of bands
    """
    if T_range is None:
        T_range = [-1.0, 1.e3]
    if F_range is None:
        F_range = [-100.0, 1.e9]

    g_prior = ngmix.priors.GPriorBA(sigma=0.1, rng=rng)
    cen_prior = ngmix.priors.CenPrior(
        cen1=0, cen2=0, sigma1=scale, sigma2=scale, rng=rng,
    )
    T_prior = ngmix.priors.FlatPrior(minval=T_range[0], maxval=T_range[1], rng=rng)
    F_prior = ngmix.priors.FlatPrior(minval=F_range[0], maxval=F_range[1], rng=rng)

    if nband is not None:
        F_prior = [F_prior]*nband

    prior = ngmix.joint_prior.PriorSimpleSep(
        cen_prior=cen_prior,
        g_prior=g_prior,
        T_prior=T_prior,
        F_prior=F_prior,
    )

    return prior

def get_admoms(obs, rng):
    gm = ngmix.gaussmom.GaussMom(1.2).go(obs)
    am = ngmix.admom.AdmomFitter(rng=rng)
    res = am.go(obs, guess=gm["T"])
    e1, e2, T = res["e1"], res["e2"], res["T"]
    e1, e2 = ngmix.shape.e1e2_to_g1g2(e1, e2)  
    return {"e1": e1, "e2": e2, "T": T, "flags": res["flags"]}  

def bit_psf(psf_g1, psf_g2):
    """
    Build the SuperBIT PSF: optical model (with fixed aberrations) convolved
    with Gaussian pointing jitter, then sheared by (psf_g1, psf_g2).

    Parameters
    ----------
    psf_g1, psf_g2 : float
        Applied PSF shear components.

    Returns
    -------
    galsim.GSObject
        The sheared PSF.
    """
    jitter = galsim.Gaussian(flux=1.0, fwhm=_SBIT_JITTER_FWHM)
    optics = galsim.OpticalPSF(
        lam=_SBIT_LAM_NM,
        diam=_SBIT_TEL_DIAM_M,
        obscuration=_SBIT_OBSCURATION,
        nstruts=_SBIT_NSTRUTS,
        strut_angle=90 * galsim.degrees,
        strut_thick=_SBIT_STRUT_THICK,
        aberrations=_SBIT_ABERRATIONS,
    )
    return galsim.Convolve([jitter, optics]).shear(g1=psf_g1, g2=psf_g2)

def make_data(rng, noise, shear, psf):
    """
    simulate an exponential object with moffat psf

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    noise: float
        Noise for the image
    shear: (g1, g2)
        The shear in each component
    psf: galsim.GSObject
        The PSF to convolve with the galaxy

    Returns
    -------
    ngmix.Observation
    """
    scale = 0.141
    npix_psf = 128

    gal_hlr = 0.5
    dy, dx = rng.uniform(low=-scale/2, high=scale/2, size=2)

    obj0 = galsim.Exponential(
        half_light_radius=gal_hlr,
    ).shear(
        g1=shear[0],
        g2=shear[1],
    ).shift(
        dx=dx,
        dy=dy,
    )

    obj = galsim.Convolve(psf, obj0)

    psf_im = psf.drawImage(nx=npix_psf, ny=npix_psf, scale=scale).array
    im = obj.drawImage(scale=scale).array

    im += rng.normal(scale=noise, size=im.shape)

    cen = (np.array(im.shape)-1.0)/2.0
    psf_cen = (np.array(psf_im.shape)-1.0)/2.0

    jacobian = ngmix.DiagonalJacobian(
        row=cen[0] + dy/scale, col=cen[1] + dx/scale, scale=scale,
    )
    psf_jacobian = ngmix.DiagonalJacobian(
        row=psf_cen[0], col=psf_cen[1], scale=scale,
    )

    wt = im*0 + 1.0/noise**2

    psf_obs = ngmix.Observation(
        psf_im,
        jacobian=psf_jacobian,
    )

    obs = ngmix.Observation(
        im,
        weight=wt,
        jacobian=jacobian,
        psf=psf_obs,
    )

    return obs


def progress(total, miniters=1):
    last_print_n = 0
    last_printed_len = 0
    sl = str(len(str(total)))
    mf = '%'+sl+'d/%'+sl+'d %3d%%'
    for i in range(total):
        yield i

        num = i+1
        if i == 0 or num == total or num - last_print_n >= miniters:
            meter = mf % (num, total, 100*float(num) / total)
            nspace = max(last_printed_len-len(meter), 0)

            print('\r'+meter+' '*nspace, flush=True, end='')
            last_printed_len = len(meter)
            if i > 0:
                last_print_n = num

    print(flush=True)


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=31415,
                        help='seed for rng')
    parser.add_argument('--ntrial', type=int, default=100,
                        help='number of trials')
    parser.add_argument('--noise', type=float, default=1.0e-6,
                        help='noise for images')
    parser.add_argument('--psf', default='dilate',
                        help='psf for reconvolution')
    return parser.parse_args()


if __name__ == '__main__':
    main()