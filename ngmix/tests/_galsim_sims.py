import numpy as np
import ngmix
import galsim


def _get_obs(rng, set_noise_image=False, noise=1.0e-6):

    psf_noise = 1.0e-6

    scale = 0.263

    psf_fwhm = 0.9
    gal_fwhm = 0.7

    psf = galsim.Gaussian(fwhm=psf_fwhm)
    obj0 = galsim.Gaussian(fwhm=gal_fwhm)

    obj = galsim.Convolve(psf, obj0)

    psf_im = psf.drawImage(scale=scale).array
    im = obj.drawImage(scale=scale).array

    psf_im += rng.normal(scale=psf_noise, size=psf_im.shape)
    im += rng.normal(scale=noise, size=im.shape)

    cen = (np.array(im.shape)-1.0)/2.0
    psf_cen = (np.array(psf_im.shape)-1.0)/2.0

    j = ngmix.DiagonalJacobian(row=cen[0], col=cen[1], scale=scale)
    pj = ngmix.DiagonalJacobian(row=psf_cen[0], col=psf_cen[1], scale=scale)

    wt = im*0 + 1.0/noise**2
    psf_wt = psf_im*0 + 1.0/psf_noise**2

    psf_obs = ngmix.Observation(
        psf_im,
        weight=psf_wt,
        jacobian=pj,
    )

    if set_noise_image:
        nim = rng.normal(scale=noise, size=im.shape)
    else:
        nim = None

    bmask = np.zeros(im.shape, dtype='i2')
    obs = ngmix.Observation(
        im,
        weight=wt,
        bmask=bmask,
        noise=nim,
        jacobian=j,
        psf=psf_obs,
    )

    return obs
