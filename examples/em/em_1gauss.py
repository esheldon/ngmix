import numpy as np
import ngmix


def main():

    args = get_args()
    rng = np.random.RandomState(args.seed)

    obs, gm = make_data(rng=rng, noise=args.noise)

    guesser = ngmix.guessers.GMixPSFGuesser(rng=rng, ngauss=1)
    guess = guesser(obs=obs)
    fitter = ngmix.em.fit_em(obs=obs, guess=guess)

    gmfit = fitter.get_gmix()
    print('true gm:')
    print(gm)
    print('fit gm:')
    print(gmfit)

    if args.show:
        try:
            import images
        except ImportError:
            from espy import images

        imfit = fitter.make_image()

        images.compare_images(obs.image, imfit)

    return gmfit


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--noise', type=float, default=0.0)
    return parser.parse_args()


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


if __name__ == '__main__':
    main()
