"""
example fitting an exponential model.  The psf is fit
using a set of coelliptical gaussians

Despite this being a fit to single image we use the full fitting framework.  We
use a generic bootstrapper to "bootstrap" the process, first fitting the psf
and then a psf flux for the object.  Then the object is fit including the PSF,
so the inferred parameters are "pre-psf".  The guess for the fit is made
based on the psf flux fit and a generic rough guess for size.

To faciliate this bootstrapping process we define the fitters for psf and
object as well as objects to provide guesses.

Bootstrappers are especially useful when you will perform the same fit on many
objects.

A run of the code should produce output something like thid

    > python fitting_bd_empsf.py

    S/N: 920.5078121454815
    true flux: 100 meas flux: 95.3763 +/- 0.653535 (99.7% conf)
    true g1: 0.05 meas g1: 0.0508 +/- 0.00960346 (99.7% conf)
    true g2: -0.02 meas g2: -0.0261123 +/- 0.0095837 (99.7% conf)
    true fracdev: 0.5 meas fracdev: 0.514028 +/- 0.011873 (99.7% conf)
"""
import numpy as np
import galsim
import ngmix


def main():

    args = get_args()
    rng = np.random.RandomState(args.seed)

    obs, obj_pars = make_data(rng=rng, noise=args.noise)

    # fit the object to an exponential disk
    prior = get_prior(rng=rng, scale=obs.jacobian.scale)
    # fit using the levenberg marquards algorithm
    fitter = ngmix.fitting.Fitter(model='exp', prior=prior)
    # make parameter guesses based on a psf flux and a rough T
    guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(
        rng=rng,
        T=0.25,
        prior=prior,
    )

    # psf fitting with coelliptical gaussians
    psf_ngauss = 5
    psf_fitter = ngmix.fitting.CoellipFitter(ngauss=psf_ngauss)
    # special guesser for coelliptical gaussians
    psf_guesser = ngmix.guessers.CoellipPSFGuesser(rng=rng, ngauss=psf_ngauss)

    # this runs the fitter. We set ntry=2 to retry the fit if it fails
    psf_runner = ngmix.runners.PSFRunner(
        fitter=psf_fitter, guesser=psf_guesser,
        ntry=2,
    )
    runner = ngmix.runners.Runner(
        fitter=fitter, guesser=guesser,
        ntry=2,
    )

    # this bootstraps the process, first fitting psfs then the object
    boot = ngmix.bootstrap.Bootstrapper(
        runner=runner,
        psf_runner=psf_runner,
    )

    res = boot.go(obs)

    print()
    print('S/N:', res['s2n'])
    print('true flux: %g meas flux: %g +/- %g (99.7%% conf)' % (
        obj_pars['flux'], res['flux'], res['flux_err']*3,
    ))
    print('true g1: %g meas g1: %g +/- %g (99.7%% conf)' % (
        obj_pars['g1'], res['g'][0], res['g_err'][0]*3,
    ))
    print('true g2: %g meas g2: %g +/- %g (99.7%% conf)' % (
        obj_pars['g2'], res['g'][1], res['g_err'][1]*3,
    ))

    if args.show:
        try:
            import images
        except ImportError:
            from espy import images

        imfit = res.make_image()

        images.compare_images(obs.image, imfit)


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


def make_data(rng, noise, g1=0.05, g2=-0.02, flux=100.0):
    """
    simulate an exponential object with moffat psf

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    noise: float
        Noise for the image
    g1: float
        object g1, default 0.05
    g2: float
        object g2, default -0.02
    flux: float, optional
        default 100

    Returns
    -------
    ngmix.Observation, pars dict
    """

    psf_noise = 1.0e-6

    scale = 0.263

    psf_fwhm = 0.9
    gal_hlr = 0.5
    dy, dx = rng.uniform(low=-scale/2, high=scale/2, size=2)

    psf = galsim.Moffat(
        beta=2.5, fwhm=psf_fwhm,
    ).shear(
        g1=-0.01,
        g2=-0.01,
    )

    obj0 = galsim.Exponential(
        half_light_radius=gal_hlr,
        flux=flux,
    ).shear(
        g1=g1,
        g2=g2,
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

    obj_pars = {
        'g1': g1,
        'g2': g2,
        'flux': flux,
    }
    return obs, obj_pars


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='seed for rng')
    parser.add_argument('--show', action='store_true',
                        help='show plot comparing model and data')
    parser.add_argument('--noise', type=float, default=0.01,
                        help='noise for images')
    return parser.parse_args()


if __name__ == '__main__':
    main()
