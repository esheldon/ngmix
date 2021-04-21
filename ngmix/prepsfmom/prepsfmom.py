import logging

import numpy as np

from ngmix import GMixModel
from ngmix.jacobian import Jacobian
from ngmix.observation import Observation
from ngmix.moments import fwhm_to_T


logger = logging.getLogger(__name__)


class PrePSFMom(object):
    """Measure a set of pre-PSF weighted moments of an obs.

    Parameters
    ----------
    fwhm : float
        The size of the kernel. This parameter has a slightly different meaning
        for each kernel. Roughly each parameter corresponds to the FWHM of the
        kernel.
    kernel : str, optional
        The kernel. Only supports 'gauss' currently. Default is 'gauss'.
    pad_factor : int, optional
        The factor by which to pad the FFTs used for the image. Default is 4.
    """
    def __init__(self, fwhm, kernel='gauss', pad_factor=4):
        self.kernel = kernel
        self.fwhm = fwhm
        self.pad_factor = pad_factor

    def go(self, obs, return_kernels=False):
        """Measure the pre-PSF moments

        Parameters
        ----------
        obs : Observation
            The observation to measure.
        return_kernels : bool, optional
            If True, return the kernels used for the flux and moments.
            Defaults to False.

        Returns
        -------
        result dictionary
        """
        if not isinstance(obs, Observation):
            raise ValueError("input obs must be an Observation")

        if not obs.has_psf():
            psf_obs = None
        else:
            psf_obs = obs.get_psf()

        if (
            obs.has_psf()
            and psf_obs.jacobian.get_galsim_wcs() != obs.jacobian.get_galsim_wcs()
        ):
            raise RuntimeError(
                "The PSF and observation must have the same WCS "
                "Jacobian for measuring pre-PSF moments."
            )

        # We need to pad the images to the same size.
        if psf_obs is not None:
            # if we have a PSF, pick the larger size
            if obs.image.shape[0] > psf_obs.image.shape[0]:
                target_dim = int(obs.image.shape[0] * self.pad_factor)
            else:
                target_dim = int(psf_obs.image.shape[0] * self.pad_factor)
        else:
            target_dim = int(obs.image.shape[0] * self.pad_factor)

        im, im_pad_offset = _zero_pad_image(obs.image.copy(), target_dim)
        wgt, _ = _zero_pad_image(obs.weight.copy(), target_dim)
        jac = obs.jacobian
        im_row0 = jac.row0 + im_pad_offset
        im_col0 = jac.col0 + im_pad_offset

        if psf_obs is not None:
            psf_im, psf_pad_offset = _zero_pad_image(
                psf_obs.image.copy(), target_dim
            )
            psf_row0 = psf_obs.jacobian.row0 + psf_pad_offset
            psf_col0 = psf_obs.jacobian.col0 + psf_pad_offset
            psf_row_offset = psf_row0 - im_row0
            psf_col_offset = psf_col0 - im_col0
        else:
            psf_im = None
            psf_pad_offset = None
            psf_row_offset = None
            psf_col_offset = None

        # now build the kernels
        if self.kernel == 'gauss':
            kres = _gauss_kernels(
                target_dim,
                self.fwhm,
                im_row0, im_col0,
                jac.dvdrow, jac.dvdcol, jac.dudrow, jac.dudcol,
                wgt,
            )
        else:
            raise RuntimeError(
                "Kernel '%s' is not allowed for pre-PSF moments!" % self.kernel
            )
        msk = wgt > 0
        res = _measure_moments_fft(
            im, np.sum(1.0/wgt[msk]),
            im_row0, im_col0,
            *kres,
            psf_im=psf_im,
            psf_row_offset=psf_row_offset,
            psf_col_offset=psf_col_offset,
        )
        if res['flags'] != 0:
            logger.debug("        pre-psf moments failed: %s" % res['flagstr'])

        if return_kernels:
            res["kernels"] = kres
            res["im"] = im

        return res


def _zero_pad_image(im, target_dim):
    """zero pad an image, returning it and the offset to the center"""
    twice_pad_width = target_dim - im.shape[0]
    if twice_pad_width % 2 == 0:
        pad_width_before = twice_pad_width // 2
        pad_width_after = pad_width_before
    else:
        pad_width_before = twice_pad_width // 2
        pad_width_after = pad_width_before + 1

    assert pad_width_before + pad_width_after == twice_pad_width

    im_padded = np.pad(
        im,
        (pad_width_before, pad_width_after),
        mode='constant',
        constant_values=0,
    )
    assert np.array_equal(
        im,
        im_padded[
            pad_width_before:im_padded.shape[0] - pad_width_after,
            pad_width_before:im_padded.shape[0] - pad_width_after
        ]
    )

    return im_padded, pad_width_before


def _gauss_kernels(
    dim,
    kernel_size,
    row0, col0,
    dvdrow, dvdcol, dudrow, dudcol,
    wgt,
):
    jac = Jacobian(
        row=row0,
        col=col0,
        dvdrow=dvdrow,
        dvdcol=dvdcol,
        dudrow=dudrow,
        dudcol=dudcol,
    )

    T = fwhm_to_T(kernel_size)

    weight = GMixModel(
        [0.0, 0.0, 0.0, 0.0, T, 1.0],
        'gauss',
    )

    weight.set_norms()
    norm = weight.get_data()['norm'][0]
    weight.set_flux(1.0/norm/jac.area)
    rkf = weight.make_image((dim, dim), jacobian=jac, fast_exp=True)
    mval = np.max(rkf)
    mind = np.unravel_index(np.argmax(rkf, axis=None), rkf.shape)

    x, y = np.meshgrid(np.arange(dim), np.arange(dim), indexing='xy')
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    y -= row0
    x -= col0
    v = dvdrow*y + dvdcol*x
    u = dudrow*y + dudcol*x

    rkf *= wgt
    rkf *= (mval/rkf[mind])

    rkxx = rkf * u**2
    rkxy = rkf * u * v
    rkyy = rkf * v**2

    fkf = np.fft.fftn(rkf)
    fkxx = np.fft.fftn(rkxx)
    fkxy = np.fft.fftn(rkxy)
    fkyy = np.fft.fftn(rkyy)

    return rkf, rkxx, rkxy, rkyy, fkf, fkxx, fkxy, fkyy


def _measure_moments_fft(
    im, tot_var, cen_row, cen_col,
    rkf, rkxx, rkxy, rkyy, fkf, fkxx, fkxy, fkyy,
    psf_im=None,
    psf_row_offset=None,
    psf_col_offset=None,
):
    flags = 0
    flagstr = ''

    imfft = np.fft.fftn(im)

    # we need to shift the FFT so that x = 0 is the center of the profile
    # this is a phase shift in fourier space
    # we have to ds it for the profile and the kernel
    f = np.fft.fftfreq(im.shape[0])

    fx = f.reshape(1, -1)
    fy = f.reshape(-1, 1)
    kcen = 2.0 * np.pi * (fy*cen_row + fx*cen_col)
    cen_phase = np.cos(kcen) + 1j*np.sin(kcen)
    # do it once profile and once for the kernel
    imfft *= cen_phase
    imfft *= cen_phase

    if psf_im is not None:
        # we also have to shift the center for the PSF (which could have a
        # different center)
        psf_imfft = np.fft.fftn(psf_im)
        fx = f.reshape(1, -1)
        fy = f.reshape(-1, 1)
        kcen = 2.0 * np.pi * (fy*psf_row_offset + fx*psf_col_offset)
        cen_phase = np.cos(kcen) + 1j*np.sin(kcen)
        psf_imfft *= cen_phase

        # make sure to zero out modes where we divide by zero
        psf_zero_msk = np.abs(psf_imfft) == 0
        if np.any(psf_zero_msk):
            psf_imfft[psf_zero_msk] = 1.0
            fkf[psf_zero_msk] = 0.0
            fkxx[psf_zero_msk] = 0.0
            fkxy[psf_zero_msk] = 0.0
            fkyy[psf_zero_msk] = 0.0

        # deconvolve!
        imfft /= psf_imfft

    df = f[1] - f[0]
    fnrm = np.sum(imfft * fkf)
    fxx = np.sum(imfft * fkxx) / fnrm
    fyy = np.sum(imfft * fkyy) / fnrm
    fxy = np.sum(imfft * fkxy) / fnrm

    flux = fnrm.real * df**2
    T = (fxx + fyy).real
    e1 = ((fxx - fyy) / (fxx + fyy)).real
    e2 = (2 * fxy / (fxx + fyy)).real

    flux_err = (
        np.sqrt(np.sum(tot_var * fkf * fkf)).real
        * df**2
    )
    # these are wrong
    # xx_err = (
    #     np.sqrt(np.sum(tot_var * fkxx * fkxx)).real
    #     * df**2
    # )
    # xy_err = (
    #     np.sqrt(np.sum(tot_var * fkxy * fkxy)).real
    #     * df**2
    # )
    # yy_err = (
    #     np.sqrt(np.sum(tot_var * fkyy * fkyy)).real
    #     * df**2
    # )

    return {
        "flags": flags,
        "flagstr": flagstr,
        "flux": flux,
        "flux_err": flux_err,
        "uu": fxx.real,
        "uv": fxy.real,
        "vv": fyy.real,
        "e1": e1,
        "e2": e2,
        "e": [e1, e2],
        "T": T,
        "pars": [0, 0, (fxx - fyy).real, (2 * fxy).real, T, flux],
    }
