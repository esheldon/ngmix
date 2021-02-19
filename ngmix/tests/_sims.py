import numpy as np

from ngmix import DiagonalJacobian, GMix, GMixModel
from ngmix import Observation, ObsList

PIXEL_SCALE = 0.263
TPSF = 0.27


def get_ngauss_obs(*, rng, ngauss, noise=0.0, with_psf=False, psf_model='turb'):

    counts = 100.0
    dims = [25, 25]
    cen = (np.array(dims) - 1.0) / 2.0
    jacob = DiagonalJacobian(scale=PIXEL_SCALE, row=cen[0], col=cen[1])

    T_1 = 0.55  # arcsec**2
    if with_psf:
        T_1 = T_1 - TPSF

    e1_1 = 0.1
    e2_1 = 0.05

    irr_1 = T_1 / 2.0 * (1 - e1_1)
    irc_1 = T_1 / 2.0 * e2_1
    icc_1 = T_1 / 2.0 * (1 + e1_1)

    cen1 = [-3.25*PIXEL_SCALE, -3.25*PIXEL_SCALE]

    if ngauss == 2:

        frac1 = 0.4
        frac2 = 0.6

        cen2 = [3.0*PIXEL_SCALE, 0.5*PIXEL_SCALE]

        T_2 = T_1/2

        e1_1 = 0.1
        e2_1 = 0.05
        e1_2 = -0.2
        e2_2 = -0.1

        counts_1 = frac1 * counts
        counts_2 = frac2 * counts

        irr_1 = T_1 / 2.0 * (1 - e1_1)
        irc_1 = T_1 / 2.0 * e2_1
        icc_1 = T_1 / 2.0 * (1 + e1_1)

        irr_2 = T_2 / 2.0 * (1 - e1_2)
        irc_2 = T_2 / 2.0 * e2_2
        icc_2 = T_2 / 2.0 * (1 + e1_2)

        pars = [
            counts_1,
            cen1[0],
            cen1[1],
            irr_1,
            irc_1,
            icc_1,
            counts_2,
            cen2[0],
            cen2[1],
            irr_2,
            irc_2,
            icc_2,
        ]

    elif ngauss == 1:

        pars = [
            counts,
            cen1[0],
            cen1[1],
            irr_1,
            irc_1,
            icc_1,
        ]

    gm = GMix(pars=pars)

    if with_psf:
        psf_ret = get_psf_obs(rng=rng, model=psf_model)
        gmconv = gm.convolve(psf_ret['gmix'])

        im0 = gmconv.make_image(dims, jacobian=jacob)
        psf_obs = psf_ret['obs']
    else:
        im0 = gm.make_image(dims, jacobian=jacob)
        psf_obs = None

    im = im0 + rng.normal(size=im0.shape, scale=noise)
    obs = Observation(im, jacobian=jacob, psf=psf_obs)

    ret = {
        'obs': obs,
        'gmix': gm,
    }

    if with_psf:
        ret['psf_gmix'] = psf_ret['gmix']

    return ret


def get_model_obs(
    *, rng, model,
    noise=0.0, psf_model='turb', set_psf_gmix=False, nepoch=None,
    star=False,
):

    if nepoch is not None:
        do_obslist = True
    else:
        do_obslist = False
        nepoch = 1

    if star:
        T = 0.0
    else:
        T = 0.27

    g1 = 0.1
    g2 = 0.05
    flux = 100.0
    off = 0.5

    # not offset from the jacobian center
    pars = [0.0, 0.0, g1, g2, T, flux]
    gm = GMixModel(pars, model)

    obslist = ObsList()
    for i in range(nepoch):

        off1_pix, off2_pix = rng.uniform(low=-off, high=off, size=2)
        dims = [32, 32]
        jcen = (np.array(dims) - 1.0) / 2.0
        jacob = DiagonalJacobian(
            scale=PIXEL_SCALE,
            row=jcen[0] + off1_pix,
            col=jcen[1] + off2_pix,
        )

        psf_ret = get_psf_obs(rng=rng, model=psf_model)
        if set_psf_gmix:
            psf_ret['obs'].set_gmix(psf_ret['gmix'])

        gmconv = gm.convolve(psf_ret['gmix'])

        im0 = gmconv.make_image(dims, jacobian=jacob)

        im = im0 + rng.normal(size=im0.shape, scale=noise)
        obs = Observation(im, jacobian=jacob, psf=psf_ret['obs'])

        obslist.append(obs)

    ret = {
        'gmix': gm,
        'pars': pars,
        'psf_data': psf_ret,
    }

    if not do_obslist:
        obs = obslist[0]
        ret['obs'] = obs
    else:
        ret['obslist'] = obslist

    return ret


def get_psf_obs(*, rng, T=TPSF, model="turb", noise=1.0e-6):
    dims = [25, 25]
    cen = (np.array(dims) - 1.0) / 2.0

    jacob = DiagonalJacobian(scale=PIXEL_SCALE, row=cen[0], col=cen[1])

    gm = GMixModel([0.0, 0.0, 0.0, 0.0, T, 1.0], model)
    im = gm.make_image(dims, jacobian=jacob)

    im += rng.normal(scale=noise, size=im.shape)
    weight = im*0 + 1.0/noise**2
    return {
        'obs': Observation(im, weight=weight, jacobian=jacob),
        'gmix': gm,
    }
