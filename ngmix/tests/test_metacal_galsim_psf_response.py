import numpy as np
import ngmix
import galsim
import pytest

@pytest.mark.parametrize('psf_g1', [-0.02, 0.02])
@pytest.mark.parametrize('psf_g2', [-0.02, 0.02])
@pytest.mark.parametrize('reconv_psf', ['dilate', 'azgauss'])
def test_metacal_galsim_psf_response(psf_g1, psf_g2, reconv_psf):

    ntrial = 100
    seed = 31415
    noise = 1e-6
    
    print(f"Running test_metacal_galsim_psf_response with psf_g1={psf_g1}, psf_g2={psf_g2}, reconv_psf={reconv_psf} ...")

    psf = bit_psf(psf_g1=psf_g1, psf_g2=psf_g2)

    shear_true_vals = [-0.02, 0.02]
    step = 0.02

    rng = np.random.RandomState(seed)

    prior   = get_prior(rng=rng, scale=0.141)
    guesser = ngmix.guessers.TFluxAndPriorGuesser(rng=rng, T=0.2, flux=1.0, prior=prior)
    fitter  = ngmix.fitting.GalsimFitter(model='exp', prior=prior)
    runner  = ngmix.runners.Runner(fitter=fitter, guesser=guesser, ntry=20)

    boot = ngmix.metacal.MetacalBootstrapper(
        runner=runner, psf_runner=None, rng=rng,
        psf=reconv_psf,
        types=['noshear', '1p', '1m', '1p_psf', '1m_psf'] if reconv_psf=='dilate' else ['noshear', '1p', '1m']
    )

    shear_rec     = []
    shear_rec_err = []

    for shear_val in shear_true_vals:
        print(f"Running for shear_true = {shear_val:+.2f} ...")
        dlist, g_psf = [], []

        for i in progress(ntrial, miniters=10):
            obs = _make_data(rng=rng, noise=noise, shear=[shear_val, 0.0], psf=psf)
            resdict, obsdict = boot.go(obs)

            for stype, sres in resdict.items():
                dlist.append(_make_struct(res=sres, obs=obsdict[stype], shear_type=stype))

            res_psf = get_admoms(obs.psf, rng)
            if res_psf['flags'] == 0:
                g_psf.append([res_psf['e1'], res_psf['e2']])

        print()
        data  = np.hstack(dlist)
        g_psf = np.asarray(g_psf)

        # selection masks
        w        = _select(data=data, shear_type='noshear')
        w_1p     = _select(data=data, shear_type='1p')
        w_1m     = _select(data=data, shear_type='1m')


        # shear response
        R11     = (data['g'][w_1p,     0].mean() - data['g'][w_1m,     0].mean()) / step
        if reconv_psf == 'dilate':
            w_1p_psf = _select(data=data, shear_type='1p_psf')
            w_1m_psf = _select(data=data, shear_type='1m_psf')
            R11_psf = (data['g'][w_1p_psf, 0].mean() - data['g'][w_1m_psf, 0].mean()) / step

        # calibrated shear
        g    = data['g'][w].mean(axis=0)
        gerr = data['g'][w].std(axis=0) / np.sqrt(w.size)

        c1 = (data['g'][w_1p, 0].mean() + data['g'][w_1m, 0].mean()) / 2 - g[0]

        correction = c1
        if reconv_psf=='dilate':
            correction += g_psf.mean(axis=0) * R11_psf
        shear_b = g - correction
        shear     = shear_b / R11
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
    if reconv_psf == 'dilate':
        print('R11_psf: %g'             % R11_psf)
    print('m: %g +/- %g (99.7%% conf)' % (m, merr * 3))
    print('c: %g +/- %g (99.7%% conf)' % (c, cerr * 3))

    assert abs(m) < 1.0e-3
    assert abs(c) < 1.0e-5


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

def get_admoms(obs, rng):
    gm = ngmix.gaussmom.GaussMom(1.2).go(obs)
    am = ngmix.admom.AdmomFitter(rng=rng)
    res = am.go(obs, guess=gm["T"])
    e1, e2, T = res["e1"], res["e2"], res["T"]
    e1, e2 = ngmix.shape.e1e2_to_g1g2(e1, e2)  
    return {"e1": e1, "e2": e2, "T": T, "flags": res["flags"]}  


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

def _make_data(rng, noise, shear, psf):
    """
    simulate an exponential object with superbit optical psf

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

    psf_npix = 128
    npix = 128
    
    scale = 0.141

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

    psf_im = psf.drawImage(nx=psf_npix, ny=psf_npix, scale=scale).array
    im = obj.drawImage(nx=npix, ny=npix, scale=scale).array

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

def _make_struct(res, obs, shear_type):
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
        data['g'] = res['g']
        data['hlr'] = res['pars'][4]
    else:
        data['s2n'] = np.nan
        data['g'] = np.nan
        data['hlr'] = np.nan

    return data

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

def _select(data, shear_type):
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