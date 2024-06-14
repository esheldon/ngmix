import numpy as np
import ngmix
import galsim
import pytest


@pytest.mark.parametrize('psf', ['gauss', 'fitgauss', 'galsim_obj'])
def test_metacal_accuracy(psf):

    ntrial = 100
    seed = 99

    if psf == 'galsim_obj':
        psf = galsim.Gaussian(fwhm=1.05)

    # Wacky WCS
    wcs_g1 = 0.1
    wcs_g2 = 0.0
    wcs = galsim.ShearWCS(0.263, galsim.Shear(g1=wcs_g1, g2=wcs_g2)).jacobian()

    print("WCS:", wcs.getDecomposition())
    print()
    print()

    shear_true = 0.02
    rng = np.random.RandomState(seed)

    # We will measure moments with a fixed gaussian weight function
    weight_fwhm = 1.2
    fitter = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)
    psf_fitter = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)

    # these "runners" run the measurement code on observations
    psf_runner = ngmix.runners.PSFRunner(fitter=psf_fitter)
    runner = ngmix.runners.Runner(fitter=fitter)

    # this "bootstrapper" runs the metacal image shearing as well as both psf
    # and object measurements
    boot = ngmix.metacal.MetacalBootstrapper(
        runner=runner, psf_runner=psf_runner,
        rng=rng,
        psf=psf,
        types=['noshear', '1p', '1m'],
    )
    dlist = []
    for i in range(ntrial):
        im, psf_im, obs = _make_data(rng=rng, shear=shear_true, wcs=wcs)
        resdict, obsdict = boot.go(obs)
        for stype, sres in resdict.items():
            st = _make_struct(res=sres, obs=obsdict[stype], shear_type=stype)
            dlist.append(st)

    print()
    data = np.hstack(dlist)

    w = _select(data=data, shear_type='noshear')
    w_1p = _select(data=data, shear_type='1p')
    w_1m = _select(data=data, shear_type='1m')

    g1 = data['g'][w, 0].mean()
    g1err = data['g'][w, 0].std() / np.sqrt(w.size)
    g1_1p = data['g'][w_1p, 0].mean()
    g1_1m = data['g'][w_1m, 0].mean()

    R11 = (g1_1p - g1_1m)/0.02

    shear = g1 / R11
    shear_err = g1err / R11
    m = shear / shear_true - 1
    merr = shear_err / shear_true

    s2n = data['s2n'][w].mean()
    print('S/N: %g' % s2n)
    print('R11: %g' % R11)
    print('m: %g +/- %g (99.7%% conf)' % (m, merr*3))

    assert np.abs(m - 0.00034) < 1.0e-4


def _make_struct(res, obs, shear_type):  # pragma: no cover
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
        ('T', 'f8'),
        ('Tpsf', 'f8'),
    ]
    data = np.zeros(1, dtype=dt)
    data['shear_type'] = shear_type
    data['flags'] = res['flags']
    if res['flags'] == 0:
        data['s2n'] = res['s2n']
        # for moments we are actually measureing e, the elliptity
        data['g'] = res['e']
        data['T'] = res['T']
    else:
        data['s2n'] = np.nan
        data['g'] = np.nan
        data['T'] = np.nan
        data['Tpsf'] = np.nan
    # we only have one epoch and band, so we can get the psf T from the
    # observation rather than averaging over epochs/bands
    data['Tpsf'] = obs.psf.meta['result']['T']
    return data


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
    # raw moments, so the T is the post-psf T.  This the
    # selection is > 1.2 rather than something smaller like 0.5
    # for pre-psf T from one of the maximum likelihood fitters
    wtype, = np.where(
        (data['shear_type'] == shear_type) &
        (data['flags'] == 0)
    )
    w, = np.where(data['T'][wtype]/data['Tpsf'][wtype] > 1.2)
    print('%s kept: %d/%d' % (shear_type, w.size, wtype.size))
    w = wtype[w]
    return w


def _make_data(rng, shear, wcs):
    """
    simulate an exponential object with moffat psf
    the hlr of the exponential is drawn from a gaussian
    with mean 0.4 arcseconds and sigma 0.2
    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    noise: float
        Noise for the image
    shear: (g1, g2)
        The shear in each component
    Returns
    -------
    ngmix.Observation
    """
    noise = 1.0e-8
    psf_noise = 1.0e-8
    stamp_size = 91

    psf_fwhm = 0.9
    gal_hlr = 0.5
    psf = galsim.Moffat(
        beta=2.5,
        fwhm=psf_fwhm,
    ).shear(
        g1=0.02,
        g2=-0.01,
    )
    obj0 = galsim.Exponential(
        half_light_radius=gal_hlr,
    ).shear(
        g1=shear,
    )
    obj = galsim.Convolve(psf, obj0)

    psf_im = psf.drawImage(nx=stamp_size, ny=stamp_size, wcs=wcs).array
    im = obj.drawImage(nx=stamp_size, ny=stamp_size, wcs=wcs).array

    psf_im += rng.normal(scale=psf_noise, size=psf_im.shape)
    im += rng.normal(scale=noise, size=im.shape)

    cen = (np.array(im.shape)-1.0)/2.0
    psf_cen = (np.array(psf_im.shape)-1.0)/2.0

    jacobian = ngmix.Jacobian(
        x=cen[1], y=cen[0], wcs=wcs.jacobian(
            image_pos=galsim.PositionD(cen[1], cen[0])
        ),
    )
    psf_jacobian = ngmix.Jacobian(
        x=psf_cen[1], y=psf_cen[0], wcs=wcs.jacobian(
            image_pos=galsim.PositionD(psf_cen[1], psf_cen[0])
        ),
    )

    wt = im*0 + 1.0/noise**2
    psf_wt = psf_im*0 + 1.0/psf_noise**2
    psf_obs = ngmix.Observation(
        psf_im,
        weight=psf_wt,
        jacobian=psf_jacobian,
    )
    obs = ngmix.Observation(
        im,
        weight=wt,
        jacobian=jacobian,
        psf=psf_obs,
    )
    return im, psf_im, obs
