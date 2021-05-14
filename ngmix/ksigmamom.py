import logging

import numpy as np

from ngmix.observation import Observation
from ngmix.moments import fwhm_to_sigma
from ngmix.util import get_ratio_error


logger = logging.getLogger(__name__)


class KSigmaMom(object):
    """Measure pre-PSF weighted real-sapce moments w/ the 'ksigma'
    Fourier-space kernels from Bernstein et al., arXiv:1508.05655.

    If the fwhm of the weight/kernel function is of similar size to the PSF or
    smaller, then the object properties returned by this fitter will be very noisy.

    Parameters
    ----------
    fwhm : float
        This parameter is the approximate real-space FWHM of the kernel.
    pad_factor : int, optional
        The factor by which to pad the FFTs used for the image. Default is 4.
    """
    def __init__(self, fwhm, pad_factor=4):
        self.fwhm = fwhm
        self.pad_factor = pad_factor

    def go(self, obs, return_kernels=False):
        """Measure the pre-PSF ksigma moments.

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

        # pad image, psf and weight map, get FFTs, apply cen_phases
        (
            kim, _, im_row, im_col, _, _
        ) = _zero_pad_and_compute_centroided_fft_and_cen_phase(
            obs.image.copy(), obs.jacobian.row0, obs.jacobian.col0, target_dim,
            apply_phase=False,
        )
        wgt = _zero_pad_image(obs.weight.copy(), target_dim)[0]

        psf_obs = obs.psf
        (
            kpsf_im, _, psf_im_row, psf_im_col, _, _
        ) = _zero_pad_and_compute_centroided_fft_and_cen_phase(
            psf_obs.image.copy(),
            psf_obs.jacobian.row0, psf_obs.jacobian.col0,
            target_dim,
            apply_phase=False,
        )

        # the final, deconvolved image we want is
        #
        #  deconv_im = kim * im_cen_phase / (kpsf_im * psf_imcen_phase)
        #
        # For efficiency we combine the phase comps to reduce sin and cos calls
        # like this
        #
        #  deconv_im = kim / kpsf_im * (im_cen_phase / psf_im_cen_phase)
        #
        # The phases are complex exponentials
        #
        #  exp(ik*cen)
        #
        # So we can compute one phase as
        #
        #  im_cen_phase / psf_im_cen_phase = exp(ik * (im_cen - psf_cen))
        #
        # and then multiply it into the image.
        #
        # This operation and the deconvolutiomn will be done
        # later in _measure_moments_fft

        # now build the kernels
        kres = _ksigma_kernels(
            target_dim,
            self.fwhm,
            obs.jacobian.dvdrow, obs.jacobian.dvdcol,
            obs.jacobian.dudrow, obs.jacobian.dudcol,
        )

        # compute the inverse of the weight map, not dividing by zero
        inv_wgt = np.zeros_like(wgt)
        msk = wgt > 0
        inv_wgt[msk] = 1.0 / wgt[msk]

        # run the actual measurements and return
        res = _measure_moments_fft(
            kim, kpsf_im, inv_wgt, eff_pad_factor, kres,
            im_row - psf_im_row, im_col - psf_im_col,
        )
        if res['flags'] != 0:
            logger.debug("ksigma pre-psf moments failed: %s" % res['flagstr'])

        if return_kernels:
            res["kernels"] = kres

        return res


def _measure_moments_fft(kim, kpsf_im, inv_wgt, eff_pad_factor, kernels, drow, dcol):
    flags = 0
    flagstr = ''

    # we only need to do things where the ksigma kernel is non-zero
    # this saves a bunch of CPU cycles
    msk = kernels["fkf"] != 0
    dim = kim.shape[0]

    # deconvolve PSF
    kim, kpsf_im, _ = _deconvolve_im_psf(
        kim[msk],
        kpsf_im[msk],
        # max amplitude is flux which is 0,0 in the standard FFT convention
        np.abs(kpsf_im[0, 0]),
    )

    # put in phase shift as described above
    # the sin and cos are expensive so we only compute them where we will
    # use the image which is in the msk
    cen_phase = _compute_cen_phase_shift(drow, dcol, dim, msk=msk)
    kim *= cen_phase

    # build the flux, radial, plus and cross kernels / moments
    # the inverse FFT in our convention has a factor of 1/n per dimension
    # the sums below are inverse FFTs but only computing the values at the
    # real-space center of the object (0 in our coordinate system).
    # thus we code the factor of 1/n by hand
    df = 1/dim

    # we only sum where the kernel is nonzero
    fkf = kernels["fkf"][msk]
    fkr = kernels["fkr"][msk]
    fkp = kernels["fkp"][msk]
    fkc = kernels["fkc"][msk]

    mf = np.sum(kim * fkf).real * df**2
    mr = np.sum(kim * fkr).real * df**2
    mp = np.sum(kim * fkp).real * df**2
    mc = np.sum(kim * fkc).real * df**2

    # build a covariance matrix of the moments
    # here we assume each Fourier mode is independent and sum the variances
    # the variance in each mode is simply the total variance over the input image
    # we need a factor of the padding to correct for something...
    m_cov = np.zeros((4, 4))
    tot_var = np.sum(inv_wgt) * eff_pad_factor**2
    kerns = [fkf / kpsf_im, fkr / kpsf_im, fkp / kpsf_im, fkc / kpsf_im]
    conj_kerns = [np.conj(k) for k in kerns]
    for i in range(4):
        for j in range(i, 4):
            m_cov[i, j] = np.sum(
                tot_var
                * kerns[i]
                * conj_kerns[j]
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
        "e_cov": np.diag(e_err**2),
        "T": T,
        "T_err": T_err,
        "pars": [0, 0, mp/mf, mc/mf, T, flux],
    }


def _zero_pad_image(im, target_dim):
    """zero pad an image, returning it and the offsets before and after
    the original image"""
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


def _compute_cen_phase_shift(cen_row, cen_col, dim, msk=None):
    """computes exp(i*2*pi*k*cen) for shifting the phases of FFTS.

    If you feed the centroid of a profile, then this factor times the raw FFT
    of that profile will result in an FFT centered at the profile.
    """
    f = np.fft.fftfreq(dim) * (2.0 * np.pi)
    # this reshaping makes sure the arrays broadcast nicely into a grid
    fx = f.reshape(1, -1)
    fy = f.reshape(-1, 1)
    kcen = fy*cen_row + fx*cen_col
    if msk is not None:
        return np.cos(kcen[msk]) + 1j*np.sin(kcen[msk])
    else:
        return np.cos(kcen) + 1j*np.sin(kcen)


def _compute_centroided_fft_and_cen_phase(im, cen_row, cen_col, apply_phase=True):
    """compute the FFT of an image, applying a phase shift to center things
    at the image center.

    Returns the fft **with the phase shift already applied** and the phase shift
    that was applied.
    """
    kim = np.fft.fftn(im)
    if apply_phase:
        cen_phase = _compute_cen_phase_shift(cen_row, cen_col, im.shape[0])
        kim *= cen_phase
        return kim, cen_phase
    else:
        return kim


def _zero_pad_and_compute_centroided_fft_and_cen_phase(
    im, cen_row, cen_col, target_dim, apply_phase=True,
):
    """zero pad, compute the FFT, and finally apply the phase shift to center
    the FFT at the object center.

    Returns the fft **with the phase shift already applied**, the phase shift
    that was applied, cen_row in the padded image, cen_col in the padded image,
    the padding before, and the padding after.
    """
    pim, pad_width_before, pad_width_after = _zero_pad_image(im, target_dim)
    pad_cen_row = cen_row + pad_width_before
    pad_cen_col = cen_col + pad_width_before
    if apply_phase:
        kpim, pad_cen_phase = _compute_centroided_fft_and_cen_phase(
            pim, pad_cen_row, pad_cen_col, apply_phase=apply_phase
        )
    else:
        kpim = _compute_centroided_fft_and_cen_phase(
            pim, pad_cen_row, pad_cen_col, apply_phase=apply_phase
        )
        pad_cen_phase = None

    return (
        kpim, pad_cen_phase,
        pad_cen_row, pad_cen_col,
        pad_width_before, pad_width_after,
    )


def _deconvolve_im_psf(kim, kpsf_im, max_amp, min_psf_frac=1e-5):
    """deconvolve the PSF from an image in place.

    Returns the deconvolved image, the kpsf_im used,
    and a bool mask marking PSF modes that were truncated
    """
    min_amp = min_psf_frac * max_amp
    abs_kpsf_im = np.abs(kpsf_im)
    msk = abs_kpsf_im <= min_amp
    kpsf_im_deconv = kpsf_im.copy()
    if np.any(msk):
        kpsf_im_deconv[msk] = kpsf_im_deconv[msk] / abs_kpsf_im[msk] * min_amp

    kim_deconv = kim/kpsf_im
    return kim_deconv, kpsf_im_deconv, msk


def _ksigma_kernels(
    dim,
    kernel_size,
    dvdrow, dvdcol, dudrow, dudcol,
):
    """This function builds a ksigma kernel in Fourier-space.

    It returns a dict of all of the kernels needed to measure moments in
    real-space by summing the kernel against the FFT of an image.
    """
    # we first get the Fourier modes in the u,v plane
    f = np.fft.fftfreq(dim) * (2.0 * np.pi)
    fx = f.reshape(1, -1)
    fy = f.reshape(-1, 1)
    Atinv = np.linalg.inv([[dvdrow, dvdcol], [dudrow, dudcol]]).T
    fv = Atinv[0, 0] * fy + Atinv[0, 1] * fx
    fu = Atinv[1, 0] * fy + Atinv[1, 1] * fx

    # now draw the kernels
    fft_dim = f.shape[0]
    fkf = np.zeros((fft_dim, fft_dim), dtype=np.complex128)
    fkr = np.zeros((fft_dim, fft_dim), dtype=np.complex128)
    fkp = np.zeros((fft_dim, fft_dim), dtype=np.complex128)
    fkc = np.zeros((fft_dim, fft_dim), dtype=np.complex128)

    # we are computing the Bernstein et al., arXiv:1508.05655. ksigma kernel which is
    # W(k) = (1 - (k*sigma/sqrt(2n))^2)^n for k < sqrt(2*n)/sigma
    # and zero otherwise. we follow them and set n = 4.
    n = 4
    sigma = fwhm_to_sigma(kernel_size)
    kmax2 = 2*n/sigma**2
    fmag2 = fu**2 + fv**2
    msk = fmag2 < kmax2
    karg = np.zeros_like(fmag2)
    karg2 = np.zeros_like(fmag2)
    karg3 = np.zeros_like(fmag2)
    karg4 = np.zeros_like(fmag2)
    karg[msk] = 1.0 - fmag2[msk]/kmax2
    karg2[msk] = karg[msk]*karg[msk]
    karg3[msk] = karg2[msk]*karg[msk]
    karg4[msk] = karg3[msk]*karg[msk]

    # we need to normalize the kernel to unity in real space at the object center
    # in our fourier conventions (angular frequency, non-unitary), the real-space
    # value at the pixel center is
    #
    # \frac{1}{(2\pi)^2} int_0^{\infty} 2\pi k W(k) dk
    #
    # where W(k) is the kernel profile and k = sqrt(kx^2 + ky^2) For the ksigma
    # kernel this expression is 2 * n / (sigma^2 * 10 * 2 * pi). We simplify this
    # to n / (sigma**2 * 10 * pi). Finally, we have to divide by this factor
    # to make the kernel have value 1.
    max_real_val = n / (sigma**2 * 10 * np.pi)

    # we also need a factor of the k-space area element so that when we
    # sum an image against this kernel, we get an integral
    detAtinv = np.abs(np.linalg.det(Atinv))

    # total factor is times the k-space area divided by the max_real_val
    # we multiply by this value
    knrm = detAtinv / max_real_val

    # now build the kernels
    # the flux kernel is easy since it is the kernel itself
    fkf[msk] = karg4[msk] * knrm

    # the moment kernels take a bit more work
    # product by u^2 in real space is -dk^2/dku^2 in Fourier space
    # same holds for v and cross deriv is -dk^2/dkudkv
    # in general
    #
    #   dWdkx = dWdk2 * dk2dx = 2kx * dWdk2
    #   dW^2dkx^2 = 2 dWdk2 + 4 kx^2 * dW^2dk2^2
    #
    # The other derivs are similar.
    # The math below has combined soem terms for efficiency, not that this
    # code is all that efficient anyways.
    two_knrm_dWdk2 = -knrm * 8.0 * karg3[msk] / kmax2
    four_knrm_dW2dk22 = knrm * 48 * karg2[msk] / kmax2**2

    # the linear combinations here measure the moments proportional to the size
    # and shears - see the Mf, Mr, M+, Mx moments in Bernstein et al., arXiv:1508.05655
    # fkr = fkxx + fkyy
    # fkp = fkxx - fkyy
    # fkc = 2 * fklxy
    fkr[msk] = -2 * two_knrm_dWdk2 - fmag2[msk] * four_knrm_dW2dk22
    fkp[msk] = -(fu[msk]**2 - fv[msk]**2) * four_knrm_dW2dk22
    fkc[msk] = -2 * fu[msk] * fv[msk] * four_knrm_dW2dk22

    return dict(
        fkf=fkf,
        fkr=fkr,
        fkp=fkp,
        fkc=fkc,
    )
