"""
example of fitting two gaussians with a psf.  The resulting
mixture is the gaussians *before* convolution by a psf
"""
import numpy as np
import ngmix


def main():

    args = get_args()
    rng = np.random.RandomState(args.seed)

    # psf is set in this function
    obs, gm = make_data(rng=rng, noise=args.noise)

    guess = gm.copy()
    randomize_gmix(rng=rng, gmix=guess, pixel_scale=obs.jacobian.scale)

    res = ngmix.em.run_em(obs=obs, guess=guess)

    gmfit = res.get_gmix()
    print('true gm:')
    print(gm)
    print('fit gm:')
    print(gmfit)

    if args.show:
        try:
            import images
        except ImportError:
            from espy import images

        imfit = res.make_image()

        images.compare_images(obs.image, imfit)

    return gmfit


def randomize_gmix(rng, gmix, pixel_scale):
    gm_data = gmix.get_data()
    for gauss in gm_data:
        gauss["p"] *= rng.uniform(low=0.9, high=1.1)
        gauss["row"] += rng.uniform(low=-pixel_scale, high=pixel_scale)
        gauss["col"] += rng.uniform(low=-pixel_scale, high=pixel_scale)
        gauss["irr"] += 0.1 * pixel_scale**2 * rng.uniform(low=-1, high=1)
        gauss["irc"] += 0.1 * pixel_scale**2 * rng.uniform(low=-1, high=1)
        gauss["icc"] += 0.1 * pixel_scale**2 * rng.uniform(low=-1, high=1)


def make_data(
    rng,
    counts=100.0,
    g1=0.0,
    g2=0.0,
    noise=0.0,
):

    pixel_scale = 0.263
    psf_fwhm = 0.9

    dims = [32]*2
    cen = (np.array(dims)-1)/2

    dvdrow, dvdcol, dudrow, dudcol = pixel_scale, -0.02, 0.01, pixel_scale
    jacobian = ngmix.Jacobian(
        row=cen[0],
        col=cen[1],
        dvdrow=dvdrow,
        dvdcol=dvdcol,
        dudrow=dudrow,
        dudcol=dudcol,
    )

    T1 = 0.2
    T2 = 0.05

    pars1 = [
        -3.25*pixel_scale, -3.25*pixel_scale, 0.1, 0.05, T1, 0.4*counts,
    ]
    pars2 = [
        3.0*pixel_scale, 0.5*pixel_scale, -0.2, -0.1, T2, 0.6*counts,
    ]

    gm1 = ngmix.GMixModel(pars1, "gauss")
    gm2 = ngmix.GMixModel(pars2, "gauss")

    full_pars = np.hstack([gm1.get_full_pars(), gm2.get_full_pars()])

    gm0 = ngmix.GMix(pars=full_pars)

    Tpsf = ngmix.moments.fwhm_to_T(psf_fwhm)
    psf_pars = [0.0, 0.0, 0.01, -0.005, Tpsf, 1.0]
    psf = ngmix.GMixModel(pars=psf_pars, model="turb")
    psf_im = psf.make_image(dims, jacobian=jacobian)

    gm = gm0.convolve(psf)

    im0 = gm.make_image(dims, jacobian=jacobian)

    im = im0 + rng.normal(size=im0.shape, scale=noise)

    # to fit with psfs, the psf observation must be set and it
    # must have a gaussian mixture set
    psf_obs = ngmix.Observation(
        image=psf_im, jacobian=jacobian, gmix=psf,
    )

    obs = ngmix.Observation(image=im, jacobian=jacobian, psf=psf_obs)

    return obs, gm0


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='seed for rng')
    parser.add_argument('--show', action='store_true',
                        help='show plot comparing model and data')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='noise for images')
    return parser.parse_args()


if __name__ == '__main__':
    main()
