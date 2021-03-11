"""
Example using a bootstrapper to run metacal, using simple
weighted moments for the shape measurement.  The simulation
is of a Moffat psf and exponential disk galaxy.

No detection is performed, so there is no associated shear-dependent
detection bias.  Thus we use "normal" metacal with perfect detection.
"""
import numpy as np
import ngmix
import galsim


def main():
    """
    Use a metacal bootstrapper with gaussian moments

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
        types is the types of images to produce.  Here we just use minimal set of
        shears to speed this example, where we only calculate the response
        for the g1 measured component to a g1 shear

            noshear: the deconvolved/reconvolved image but without shear.  This image
              to get the shear estimator.
            1p: sheared +g1
            1m: sheared -g1
                1p/1m are are used to calculate the response.

        standard default set would also includes shears in g2 (2p, 2m)
    """

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

    gvals = np.zeros((args.ntrial, 2))
    s2n = np.zeros(args.ntrial)
    R11vals = np.zeros(args.ntrial)

    for i in range(args.ntrial):
        if (i == 0) or (i+1) % 100 == 0:
            print('.', end='', flush=True)

        obs = make_data(rng=rng, noise=args.noise, shear=shear_true)

        boot.go(obs)
        resdict = boot.get_result()

        gvals[i, :] = resdict['noshear']['e']

        s2n[i] = resdict['noshear']['s2n']

        res1p = resdict['1p']
        res1m = resdict['1m']

        R11vals[i] = (res1p['e'][0] - res1m['e'][0])/0.02

    R11 = R11vals.mean()

    shear = gvals.mean(axis=0)/R11
    shear_err = gvals.std(axis=0)/np.sqrt(args.ntrial)/R11

    m = shear[0]/shear_true[0]-1
    merr = shear_err[0]/shear_true[0]

    print()
    print('s2n: %g' % s2n.mean())
    print('R11: %g' % R11)
    print('m: %g +/- %g (99.7%% conf)' % (m, merr*3))
    print('c: %g +/- %g (99.7%% conf)' % (shear[1], shear_err[1]*3))


def make_data(rng, noise, shear):

    psf_noise = 1.0e-6

    scale = 0.263

    psf_fwhm = 0.9
    gal_hlr = 0.5
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
