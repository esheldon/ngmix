import galsim
import numpy as np
import ngmix


def _get_obs(rng):
    """
    obs with noise image included
    """
    noise = 0.1
    psf_noise = 1.0e-6
    scale = 0.263

    psf_fwhm = 0.9
    gal_fwhm = 0.7

    psf = galsim.Gaussian(fwhm=psf_fwhm)
    obj0 = galsim.Gaussian(fwhm=gal_fwhm)

    obj = galsim.Convolve(psf, obj0)

    psf_im = psf.drawImage(scale=scale).array
    im = obj.drawImage(scale=scale).array

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
    im += rng.normal(scale=noise, size=im.shape)
    psf_im += rng.normal(scale=psf_noise, size=psf_im.shape)
    nim = rng.normal(scale=noise, size=im.shape)

    obs = ngmix.Observation(
        im,
        weight=wt,
        noise=nim,
        jacobian=j,
        psf=psf_obs,
    )

    return obs


def test_metacal_fixnoise_noise_image():

    rng = np.random.RandomState(seed=100)
    obs = _get_obs(rng)
    noise_obs = _get_obs(rng)

    with noise_obs.writeable():
        nim = obs.noise.copy()
        noise_obs.image[:, :] = np.rot90(nim, k=1)

    mdict = ngmix.metacal.get_all_metacal(
        obs, psf='gauss', rng=rng, use_noise_image=True,
    )
    mdict_no_fixnoise = ngmix.metacal.get_all_metacal(
        obs, psf='gauss', rng=rng, fixnoise=False,
    )
    mdict_noise = ngmix.metacal.get_all_metacal(
        noise_obs, psf='gauss', rng=rng, fixnoise=False,
    )

    for key in mdict:
        im = mdict[key].image
        im_no_fixnoise = mdict_no_fixnoise[key].image
        noise_im = np.rot90(mdict_noise[key].image, k=3)

        assert np.all(im == im_no_fixnoise + noise_im)
        assert np.all(im != im_no_fixnoise)
