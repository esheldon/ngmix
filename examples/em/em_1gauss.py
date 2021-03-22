"""
example fitting a single gaussian
"""
import numpy as np
import ngmix


def main():

    args = get_args()
    rng = np.random.RandomState(args.seed)

    obs, gm = make_data(rng=rng, noise=args.noise)

    # use a psf guesser
    guesser = ngmix.guessers.GMixPSFGuesser(rng=rng, ngauss=1)
    guess = guesser(obs=obs)
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


def make_data(
    rng,
    counts=100.0,
    fwhm=1.2,
    g1=0.0,
    g2=0.0,
    noise=0.0,
):

    pixel_scale = 0.263

    T = ngmix.moments.fwhm_to_T(fwhm)
    sigma = np.sqrt(T / 2)
    dim = int(2 * 5 * sigma/pixel_scale)
    dims = [dim] * 2
    cen = [dims[0] / 2.0, dims[1] / 2.0]

    dvdrow, dvdcol, dudrow, dudcol = pixel_scale, -0.02, 0.01, pixel_scale
    jacobian = ngmix.Jacobian(
        row=cen[0],
        col=cen[1],
        dvdrow=dvdrow,
        dvdcol=dvdcol,
        dudrow=dudrow,
        dudcol=dudcol,
    )

    pars = [0.0, 0.0, g1, g2, T, counts]
    gm = ngmix.GMixModel(pars, "gauss")

    im0 = gm.make_image(dims, jacobian=jacobian)

    im = im0 + rng.normal(size=im0.shape, scale=noise)

    obs = ngmix.Observation(image=im, jacobian=jacobian)

    return obs, gm


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
