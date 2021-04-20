import logging

import numpy as np

from ngmix.observation import Observation
from ngmix.shape import e1e2_to_g1g2

logger = logging.getLogger(__name__)


class PrePSFMom(object):
    """Measure a set of pre-PSF weighted moments of an obs.

    Parameters
    ----------
    kernel : str, optional
        The kernel. One of 'triweight'.
    kernel_size : float, optional
        The size of the kernel. This parameter has a slightly different meaning
        for each kernel. Roughly each parameter corresponds to the FWHM of the
        kernel.
    pad_factor : int, optional
        The factor by which to pad the FFTs used for the image.
    """
    def __init__(self, kernel='triweight', kernel_size=1.2, pad_factor=4):
        self.kernel = kernel
        self.kernel_size = kernel_size
        self.pad_factor = pad_factor

    def go(self, obs):
        """Measure the pre-PSF moments

        Parameters
        ----------
        obs : Observation
            The observation to measure.

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
        kres = _triweight_kernels(
            target_dim,
            self.kernel_size,
            im_row0, im_col0,
            jac.dvdrow, jac.dvdcol, jac.dudrow, jac.dudcol,
            wgt,
        )

        res = _measure_moments_fft(
            im, np.sum(1.0/wgt), *kres,
            psf_im=psf_im,
            psf_row_offset=psf_row_offset,
            psf_col_offset=psf_col_offset,
        )
        if res['flags'] != 0:
            logger.debug("        pre-psf moments failed: %s" % res['flagstr'])

        return res


def _zero_pad_image(im, target_dim):
    """zero pad an image, returning it and the offset to the center"""
    twice_pad_width = target_dim - im.shape[0]
    if twice_pad_width % 2 == 0:
        pad_width_before = twice_pad_width // 2
        pad_width_after = pad_width_before
    else:
        pad_width_before = twice_pad_width // 2
        pad_width_after = pad_width_before - 1

    im_padded = np.pad(
        im,
        (pad_width_before, pad_width_after),
        mode='constant',
        constant_values=0,
    )

    return im_padded, pad_width_before


def _triweight_kernels(
    dim,
    kernel_size,
    row0, col0,
    dvdrow, dvdcol, dudrow, dudcol,
    wgt,
):
    x, y = np.meshgrid(np.arange(dim), np.arrang(dim), indexing='xy')

    x = x.astype(np.float64)
    y = y.astype(np.float64)
    y -= row0
    x -= col0

    v = dvdrow*y + dvdcol*x
    u = dudrow*y + dudcol*x
    r2 = v**2 + u**2

    sc2 = kernel_size**2

    z = np.sqrt(r2/sc2)
    msk = z < 3
    rkf = np.zeros((dim, dim))
    rkf[msk] = 35 / 96 * (1 - (z[msk] / 3) ** 2) ** 3 / kernel_size

    rkf *= wgt
    rkf /= np.sum(rkf)

    rkxx = rkf * u**2
    rkxy = rkf * u * v
    rkyy = rkf * v**2

    fkf = np.fft.fftn(rkf)
    fkxx = np.fft.fftn(rkxx)
    fkxy = np.fft.fftn(rkxy)
    fkyy = np.fft.fftn(rkyy)

    return rkf, rkxx, rkxy, rkyy, fkf, fkxx, fkxy, fkyy


def _measure_moments_fft(
    im, tot_var,
    rkf, rkxx, rkxy, rkyy, fkf, fkxx, fkxy, fkyy,
    psf_im=None,
    psf_row_offset=None,
    psf_col_offset=None,
):
    flags = 0

    imfft = np.fft.fftn(im)

    f = np.fft.fftfreq(im.shape[0])
    if psf_im is not None:
        psf_imfft = np.fft.fftn(psf_im)
        if psf_row_offset is not None:
            fx = f.reshape(1, -1)
            fy = f.reshape(-1, 1)
            kcen = 2.0 * np.pi * (fy*psf_row_offset + fx*psf_col_offset)
            cen_phase = np.cos(kcen) - 1j*np.sin(kcen)
            psf_imfft *= cen_phase
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

    try:
        g1, g2 = e1e2_to_g1g2(e1, e2)
    except Exception:
        flags |= 2**0
        flagstr = (
            "Cannot convert (e1, e2) to (g1, g2)! "
            "Moments do not form a proper shear!"
        )
        g1 = -9999.0
        g2 = -9999.0

    flux_err = (
        np.sqrt(np.sum(tot_var * fkf * fkf)).real
        * df**2
    )
    xx_err = (
        np.sqrt(np.sum(tot_var * fkxx * fkxx)).real
        * df**2
    )
    xy_err = (
        np.sqrt(np.sum(tot_var * fkxy * fkxy)).real
        * df**2
    )
    yy_err = (
        np.sqrt(np.sum(tot_var * fkyy * fkyy)).real
        * df**2
    )

    return {
        "flags": flags,
        "flagstr": flagstr,
        # factor of df**2 is the k-space element volume
        "flux": flux,
        "flux_err": flux_err,
        "uu": fxx.real * df**2,
        "uu_err": xx_err,
        "uv": fxy.real * df**2,
        "uv_err": xy_err,
        "vv": fyy.real * df**2,
        "vv_err": yy_err,
        "e1": e1,
        "e2": e2,
        "T": T,
        "T_err": np.sqrt(xx_err**2 + yy_err**2),
        "g1": g1,
        "g2": g2,
        "pars": [0, 0, g1, g2, T, flux],
    }
