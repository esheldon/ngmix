import logging

import numpy as np

from ngmix import GMixModel
from ngmix.jacobian import Jacobian
from ngmix.observation import Observation
from ngmix.moments import fwhm_to_T
from ngmix.util import get_ratio_error


logger = logging.getLogger(__name__)


class PrePSFGaussMom(object):
    """Measure pre-PSF Gaussian weighted moments.

    If the fwhm of the weight function is of similar size to the PSF or smaller,
    then the object properties returned by this fitter will be very noisy.

    Parameters
    ----------
    fwhm : float
        The FWHM of the Gaussian kernel.
    pad_factor : int, optional
        The factor by which to pad the FFTs used for the image. Default is 4.
    kernel_trunc_fac : float, optional
        The kernel is truncated in Fourier-space at this factor times the kernel
        FWHM. Default is 1.
    """
    def __init__(self, fwhm, pad_factor=4, kernel_trunc_fac=1):
        self.fwhm = fwhm
        self.pad_factor = pad_factor
        self.kernel_trunc_fac = kernel_trunc_fac

    def go(self, obs, return_kernels=False):
        """Measure the pre-PSF moments.

        Parameters
        ----------
        obs : Observation
            The observation to measure.
        return_kernels : bool, optional
            If True, return the kernels used for the flux and moments.
            Only valid when `direct_deconv` is False. Defaults to False.

        Returns
        -------
        result dictionary
        """
        if not isinstance(obs, Observation):
            raise ValueError("input obs must be an Observation")

        if not obs.has_psf():
            raise RuntimeError("The PSF must be set to measure a pre-PSF moment!")

        psf_obs = obs.get_psf()

        if psf_obs.jacobian.get_galsim_wcs() != obs.jacobian.get_galsim_wcs():
            raise RuntimeError(
                "The PSF and observation must have the same WCS "
                "Jacobian for measuring pre-PSF moments."
            )

        return self._meas_fourier_only(obs, psf_obs, return_kernels)

    def _meas_fourier_only(self, obs, psf_obs, return_kernels):
        # pick the larger size
        if obs.image.shape[0] > psf_obs.image.shape[0]:
            target_dim = int(obs.image.shape[0] * self.pad_factor)
        else:
            target_dim = int(psf_obs.image.shape[0] * self.pad_factor)
        eff_pad_factor = target_dim / obs.image.shape[0]

        # pad the image and weight
        # compute new profile center
        im, im_pad_offset, _ = _zero_pad_image(obs.image.copy(), target_dim)
        wgt, _, _ = _zero_pad_image(obs.weight.copy(), target_dim)
        jac = obs.jacobian
        im_row0 = jac.row0 + im_pad_offset
        im_col0 = jac.col0 + im_pad_offset

        # if we have a PSF, we pad and get the offset of the PSF center from
        # the object center. this offset gets removed in the FFT so that objects
        # stay in the same spot.
        # We assume the Jacobian is centered at the object/PSF center.
        psf_im, psf_pad_offset, _ = _zero_pad_image(
            psf_obs.image.copy(), target_dim
        )
        psf_row0 = psf_obs.jacobian.row0 + psf_pad_offset
        psf_col0 = psf_obs.jacobian.col0 + psf_pad_offset
        psf_row_offset = psf_row0 - im_row0
        psf_col_offset = psf_col0 - im_col0

        # now build the kernels
        kres = _gauss_kernels(
            target_dim,
            self.fwhm,
            im_row0, im_col0,
            jac.dvdrow, jac.dvdcol, jac.dudrow, jac.dudcol,
            self.kernel_trunc_fac,
        )

        # compute the inverse of the weight map, not dividing by zero
        inv_wgt = np.zeros_like(wgt)
        msk = wgt > 0
        inv_wgt[msk] = 1.0 / wgt[msk]

        # run the actual measurements and return
        res = _measure_moments_fft(
            im, inv_wgt, eff_pad_factor,
            im_row0, im_col0,
            kres,
            psf_im,
            psf_row_offset,
            psf_col_offset,
        )
        if res['flags'] != 0:
            logger.debug("        pre-psf moments failed: %s" % res['flagstr'])

        if return_kernels:
            res["kernels"] = kres
            res["im"] = im
            res['wgt'] = wgt
            res["inv_wgt"] = inv_wgt

        return res


def _zero_pad_image(im, target_dim):
    """zero pad an image, returning it and the offset to the center"""
    twice_pad_width = target_dim - im.shape[0]
    # if the extra number of pixels we need is odd, we add those on the
    # second half
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

    return im_padded, pad_width_before, pad_width_after


def _gauss_kernels(
    dim,
    kernel_size,
    row0, col0,
    dvdrow, dvdcol, dudrow, dudcol,
    kernel_trunc_fac,
):
    """
    This function renders the kernel in real-space and then computes the right
    set of FFTs.

    It is possible to directly render the kernel in Fourier-space which may yield
    faster code. However this has not been done here.
    """
    # first we get the kernel from ngmix
    jac = Jacobian(
        row=row0,
        col=col0,
        dvdrow=dvdrow,
        dvdcol=dvdcol,
        dudrow=dudrow,
        dudcol=dudcol,
    )

    fft_fwhm = 1.0 / kernel_size

    T = fwhm_to_T(kernel_size)

    weight = GMixModel(
        [0.0, 0.0, 0.0, 0.0, T, 1.0],
        'gauss',
    )

    # make sure to set the peak of the kernel to 1 to get better fluxes
    weight.set_norms()
    norm = weight.get_data()['norm'][0]
    weight.set_flux(1.0/norm/jac.area)
    rkf = weight.make_image((dim, dim), jacobian=jac, fast_exp=True)

    # build u,v for each pixel to compute the moment kernels
    x, y = np.meshgrid(np.arange(dim), np.arange(dim), indexing='xy')
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    y -= row0
    x -= col0
    v = dvdrow*y + dvdcol*x
    u = dudrow*y + dudcol*x

    # now build the moment kernels and their FFTs
    rkxx = rkf * u**2
    rkxy = rkf * u * v
    rkyy = rkf * v**2

    fkf = np.fft.fftn(rkf)
    fkxx = np.fft.fftn(rkxx)
    fkxy = np.fft.fftn(rkxy)
    fkyy = np.fft.fftn(rkyy)

    # we truncate the kernels in fourier space
    f = np.fft.fftfreq(dim)
    fx = f.reshape(1, -1)
    fy = f.reshape(-1, 1)
    Atinv = np.linalg.inv([[dvdrow, dvdcol], [dudrow, dudcol]]).T
    fv = Atinv[0, 0] * fy + Atinv[0, 1] * fx
    fu = Atinv[1, 0] * fy + Atinv[1, 1] * fx
    msk = (fu**2 + fv**2) >= (fft_fwhm * kernel_trunc_fac)**2
    fkf[msk] = 0
    fkxx[msk] = 0
    fkxy[msk] = 0
    fkyy[msk] = 0

    # the linear combinations here measure the moments proportional to the size
    # and shears
    return dict(
        rkf=rkf,
        rkr=rkxx + rkyy,
        rkp=rkxx - rkyy,
        rkc=2.0 * rkxy,
        fkf=fkf,
        fkr=fkxx + fkyy,
        fkp=fkxx - fkyy,
        fkc=2.0 * fkxy,
    )


def _measure_moments_fft(
    im, inv_wgt, eff_pad_factor,
    cen_row, cen_col,
    kernels,
    psf_im,
    psf_row_offset,
    psf_col_offset,
    max_psf_frac=1e-12,
):
    flags = 0
    flagstr = ''

    imfft = np.fft.fftn(im)

    # we need to shift the FFT so that x = 0 is the center of the profile
    # this is a phase shift in fourier space
    # we have to do it for the profile and the kernel (and the PSF if needed)
    f = np.fft.fftfreq(im.shape[0])
    # this reshaping makes sure the arrays broadcast nicely
    fx = f.reshape(1, -1)
    fy = f.reshape(-1, 1)
    kcen = 2.0 * np.pi * (fy*cen_row + fx*cen_col)
    cen_phase = np.cos(kcen) + 1j*np.sin(kcen)
    # instead of adjusting the kernels, we will apply the shift twice to the image
    imfft *= cen_phase
    imfft *= cen_phase

    # first we shift the PSF to the object center
    psf_imfft = np.fft.fftn(psf_im)
    fx = f.reshape(1, -1)
    fy = f.reshape(-1, 1)
    kcen = 2.0 * np.pi * (fy*psf_row_offset + fx*psf_col_offset)
    psf_cen_phase = np.cos(kcen) + 1j*np.sin(kcen)
    psf_imfft *= psf_cen_phase

    # now we apply the shift as above
    psf_imfft *= cen_phase

    # now we remove the PSF
    # we truncate models below max_psf_frac of the maximum of the PSF profile
    # set the PSF to 1 there to make sure we don't divide by zero
    # set the kernels to zero there to ensure we do not use those modes
    abs_psfimfft = np.abs(psf_imfft)
    psf_zero_msk = abs_psfimfft <= max_psf_frac * np.max(abs_psfimfft)
    if np.any(psf_zero_msk):
        psf_imfft[psf_zero_msk] = 1.0
        for k in kernels:
            if k.startswith("f"):
                kernels[k][psf_zero_msk] = 0.0

    g = 1.0 / psf_imfft

    # deconvolve!
    imfft *= g

    # finally we build the kernels, moments and their errors
    df = f[1] - f[0]  # this is the area factor for the integral we are
    # doing in Fourier space

    # build the flux, radial, plus and cross kernels / moments
    fkf = kernels["fkf"]
    fkr = kernels["fkr"]
    fkp = kernels["fkp"]
    fkc = kernels["fkc"]
    mf = np.sum(imfft * fkf).real * df**2
    mr = np.sum(imfft * fkr).real * df**2
    mp = np.sum(imfft * fkp).real * df**2
    mc = np.sum(imfft * fkc).real * df**2

    # build a covariance matrix of the moments
    # here we assume each Fourier mode is independent and sum the variances
    # the variance in each mode is simply the total variance over the input image
    # we need a factor of the padding to correct for something...
    m_cov = np.zeros((4, 4))
    tot_var = np.sum(inv_wgt) * eff_pad_factor**2
    kerns = [fkf * g, fkr * g, fkp * g, fkc * g]
    for i in range(4):
        for j in range(i, 4):
            m_cov[i, j] = np.sum(
                tot_var
                * (kerns[i])
                * np.conj(kerns[j])
            ).real * df**4
            m_cov[j, i] = m_cov[i, j]

    # now finally build the outputs and their errors
    flux = mf
    T = mr / mf
    e1 = mp / mr
    e2 = mc / mr

    T_err = get_ratio_error(mr, mf, m_cov[1, 1], m_cov[0, 0], m_cov[0, 1])
    e_err = np.zeros(2)
    e_err[0] = get_ratio_error(mp, mr, m_cov[2, 2], m_cov[1, 1], m_cov[1, 2])
    e_err[1] = get_ratio_error(mc, mr, m_cov[3, 3], m_cov[1, 1], m_cov[1, 3])

    return {
        "flags": flags,
        "flagstr": flagstr,
        "flux": flux,
        "flux_err": np.sqrt(m_cov[0, 0]),
        "mom": np.array([mf, mr, mp, mc]),
        "mom_err": np.sqrt(np.diagonal(m_cov)),
        "mom_cov": m_cov,
        "e1": e1,
        "e2": e2,
        "e": [e1, e2],
        "e_err": e_err,
        "T": T,
        "T_err": T_err,
        "pars": [0, 0, mp/mf, mc/mf, T, flux],
    }
