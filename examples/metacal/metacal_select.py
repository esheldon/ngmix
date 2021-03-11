"""
An example of metacal with selections.

Use a metacal bootstrapper with gaussian moments.  Simple weighted moments are
used for measurement.  In this example we perform no detection and make no
selections.

In this example, we set two parameters for the metacal run: the psf and the
types of images.  These are set when constructing the MetacalBootstrapper

the psf
    We deconvolve, shear the image, then reconvolve.  Setting psf to
    'fitgauss' means we reconvolve by a round gaussian psf, based on
    fitting the original psf with a gaussian and dilating it appropriately.

    Setting it simply to 'gauss' uses a deterministic algorithm to create a
    psf that is round and larger than the original.  This algorithm is
    slower and can result in a slightly noisier measurement, because it is
    more conservative.

    The default is 'gauss'

the types
    types is the types of images to produce.  Here we just use minimal set
    of shears to speed up this example, where we only calculate the
    response of the g1 measurement to a shear in g1

        noshear: the deconvolved/reconvolved image but without shear.  This image
          is used to measure the shear estimator and other quantities.
        1p: sheared +g1
        1m: sheared -g1
            1p/1m are are used to calculate the response and selection effects.

    standard default set would also includes shears in g2 (2p, 2m)
"""
import numpy as np
import ngmix
import galsim


def main():
    args = get_args()

    shear_true = [0.01, 0.00]
    rng = np.random.RandomState(args.seed)

    # measure moments with a fixed gaussian weight function, no psf correction
    weight_fwhm = 1.2
    fitter = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)
    psf_fitter = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)

    # these run the moments
    psf_runner = ngmix.runners.PSFRunner(fitter=psf_fitter)
    runner = ngmix.runners.Runner(fitter=fitter)

    # this runs metacal as well as both psf and object measurements
    boot = ngmix.metacal_bootstrap.MetacalBootstrapper(
        runner=runner, psf_runner=psf_runner,
        rng=rng,
        psf=args.psf,
        types=['noshear', '1p', '1m'],
    )

    # let's just do R11 for simplicity and to speed up this example; typically
    # the off diagonal terms are negligible, and R11 and R22 are usually
    # consistent

    dlist = []

    for i in progress(args.ntrial, miniters=10):
        obs = make_data(rng=rng, noise=args.noise, shear=shear_true)

        boot.go(obs)
        res = boot.result
        obsdict = boot.obsdict

        # keep any that pass the flags
        for stype, sres in res.items():
            if sres['flags'] == 0:
                st = make_struct(res=sres, obs=obsdict[stype], shear_type=stype)
                dlist.append(st)

    print()
    data = np.hstack(dlist)

    w = select(data=data, shear_type='noshear')
    w_1p = select(data=data, shear_type='1p')
    w_1m = select(data=data, shear_type='1m')

    g = data['g'][w].mean(axis=0)
    gerr = data['g'][w].std(axis=0) / np.sqrt(w.size)
    g1_1p = data['g'][w_1p, 0].mean()
    g1_1m = data['g'][w_1m, 0].mean()
    R11 = (g1_1p - g1_1m)/0.02

    shear = g / R11
    shear_err = gerr / R11

    m = shear[0]/shear_true[0]-1
    merr = shear_err[0]/shear_true[0]

    s2n = data['s2n'][w].mean()

    print('s2n: %g' % s2n)
    print('R11: %g' % R11)
    print('m: %g +/- %g (99.7%% conf)' % (m, merr*3))
    print('c: %g +/- %g (99.7%% conf)' % (shear[1], shear_err[1]*3))


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
        ('shear_type', 'U7'),
        ('s2n', 'f8'),
        ('g', 'f8', 2),
        ('T', 'f8'),
        ('Tpsf', 'f8'),
    ]
    data = np.zeros(1, dtype=dt)
    data['shear_type'] = shear_type
    data['s2n'] = res['s2n']
    # for moments we are actually measureing e, the elliptity
    data['g'] = res['e']
    data['T'] = res['T']

    # we only have one epoch and band, so we can get the psf T from the
    # observation rather than averaging over epochs/bands
    data['Tpsf'] = obs.psf.meta['result']['T']
    return data


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
    # raw moments, so the T is the post-psf T.  This the
    # selection is > 1.2 rather than something smaller like 0.5
    # for pre-psf T from one of the maximum likelihood fitters

    wtype, = np.where(data['shear_type'] == shear_type)
    w, = np.where(data['T'][wtype]/data['Tpsf'][wtype] > 1.2)

    w = wtype[w]
    print('%s kept %d/%d' % (shear_type, w.size, wtype.size))
    return w


def make_data(rng, noise, shear):
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
    psf_noise = 1.0e-6

    scale = 0.263

    psf_fwhm = 0.9
    gal_hlr = rng.normal(loc=0.4, scale=0.2)
    dy, dx = rng.uniform(low=-scale/2, high=scale/2, size=2)

    psf = galsim.Moffat(
        beta=2.5, fwhm=psf_fwhm,
    ).shear(
        g1=0.02,
        g2=-0.01,
    )

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

    psf_im = psf.drawImage(scale=scale).array
    im = obj.drawImage(scale=scale).array

    psf_im += rng.normal(scale=psf_noise, size=psf_im.shape)
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
    parser.add_argument('--ntrial', type=int, default=1000,
                        help='number of trials')
    parser.add_argument('--noise', type=float, default=1.0e-6,
                        help='noise for images')
    parser.add_argument('--psf', default='gauss',
                        help='psf for reconvolution')
    return parser.parse_args()


if __name__ == '__main__':
    main()
