import logging
import functools

import numpy as np
import scipy.fft as fft
from numba import njit

from ngmix.observation import Observation
from ngmix.moments import fwhm_to_sigma, make_mom_result
from ngmix.gexceptions import FFTRangeError
from ngmix.fastexp_nb import FASTEXP_MAX_CHI2, fexp_arr


logger = logging.getLogger(__name__)


class PrePSFMom(object):
    """Measure pre-PSF weighted real-space moments.

    This class is not meant to be used directly. Instead use either `KSigmaMom`
    or `PGaussMom`.

    If the fwhm of the weight/kernel function is of similar size to the PSF or
    smaller, then the object properties returned by this fitter will be very noisy.

    Parameters
    ----------
    fwhm : float
        This parameter is the approximate real-space FWHM of the kernel. The units are
        whatever the Jacobian on the obs converts pixels units to. This is typically
        arcseconds.
    kernel : str
        The kernel to use. Either `ksigma` or `pgauss` or `gauss`. `gauss` and `pgauss`
        are aliases for the same thing.
    pad_factor : int, optional
        The factor by which to pad the FFTs used for the image. Default is 4.
    ap_rad : float, optional
        The apodization radius for the stamp in pixels. The default of 1.5 is likely
        fine for most ground based surveys.
    fwhm_smooth : float, optional
        If non-zero, this optional applies additional Gaussian smoothing to the
        object before computing the moments. Typically a non-zero value results
        in less shape noise.
    """
    def __init__(
        self, fwhm, kernel, pad_factor=4, ap_rad=1.5, fwhm_smooth=0
    ):
        self.fwhm = fwhm
        self.pad_factor = pad_factor
        self.kernel = kernel
        self.ap_rad = ap_rad
        self.fwhm_smooth = fwhm_smooth
        if self.kernel == "ksigma":
            self.kind = "ksigma"
        elif self.kernel in ["gauss", "pgauss"]:
            self.kind = "pgauss"
        else:
            raise ValueError(
                "The kernel '%s' for PrePSFMom is not recognized!" % self.kernel
            )

    def go(self, obs, return_kernels=False, no_psf=False):
        """Measure the pre-PSF ksigma moments.

        Parameters
        ----------
        obs : ngmix.Observation
            The observation to measure.  The image data must be square.
        return_kernels : bool, optional
            If True, return the kernels used for the flux and moments.
            Defaults to False.
        no_psf : bool, optional
            If True, allow inputs without a PSF observation. Defaults to False
            so that any input observation without a PSF will raise an error.

        Returns
        -------
        result dictionary
        """
        psf_obs = _check_obs_and_get_psf_obs(obs, no_psf)
        return self._meas(obs, psf_obs, return_kernels)

    def _meas(self, obs, psf_obs, return_kernels):
        # pick the larger size
        if psf_obs is not None:
            if obs.image.shape[0] > psf_obs.image.shape[0]:
                target_dim = int(obs.image.shape[0] * self.pad_factor)
            else:
                target_dim = int(psf_obs.image.shape[0] * self.pad_factor)
        else:
            target_dim = int(obs.image.shape[0] * self.pad_factor)
        eff_pad_factor = target_dim / obs.image.shape[0]

        # pad image, psf and weight map, get FFTs, apply cen_phases
        kim, im_row, im_col = _zero_pad_and_compute_fft_cached(
            obs.image, obs.jacobian.row0, obs.jacobian.col0, target_dim,
            self.ap_rad,
        )
        fft_dim = kim.shape[0]

        if psf_obs is not None:
            kpsf_im, psf_im_row, psf_im_col = _zero_pad_and_compute_fft_cached(
                psf_obs.image,
                psf_obs.jacobian.row0, psf_obs.jacobian.col0,
                target_dim,
                0,  # we do not apodize PSF stamps since it should not be needed
            )
        else:
            # delta function in k-space
            kpsf_im = np.ones_like(kim, dtype=np.complex128)
            psf_im_row = 0.0
            psf_im_col = 0.0

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
        if self.kernel == "ksigma":
            kernels = _ksigma_kernels(
                int(target_dim),
                float(self.fwhm),
                float(obs.jacobian.dvdrow), float(obs.jacobian.dvdcol),
                float(obs.jacobian.dudrow), float(obs.jacobian.dudcol),
                float(self.fwhm_smooth),
            )
        elif self.kernel in ["gauss", "pgauss"]:
            kernels = _gauss_kernels(
                int(target_dim),
                float(self.fwhm),
                float(obs.jacobian.dvdrow), float(obs.jacobian.dvdcol),
                float(obs.jacobian.dudrow), float(obs.jacobian.dudcol),
                float(self.fwhm_smooth),
            )
        else:
            raise ValueError(
                "The kernel '%s' for PrePSFMom is not recognized!" % self.kernel
            )

        # compute the total variance from weight map
        msk = obs.weight > 0
        tot_var = np.sum(1.0 / obs.weight[msk])

        # run the actual measurements and return
        mom, mom_cov, mom_norm = _measure_moments_fft(
            kim, kpsf_im, tot_var, eff_pad_factor, kernels,
            im_row - psf_im_row, im_col - psf_im_col,
        )
        res = make_mom_result(mom, mom_cov, sums_norm=mom_norm)
        if res['flags'] != 0:
            logger.debug("pre-psf moments failed: %s" % res['flagstr'])

        if return_kernels:
            # put the kernels back into their unpacked state
            full_kernels = {}
            for k in kernels:
                if k == "msk":
                    continue
                if k == "nrm":
                    full_kernels[k] = kernels[k]
                else:
                    full_kernels[k] = np.zeros((fft_dim, fft_dim), dtype=np.complex128)
                    full_kernels[k][kernels["msk"]] = kernels[k]
            res["kernels"] = full_kernels

        return res


class KSigmaMom(PrePSFMom):
    """Measure pre-PSF weighted real-space moments w/ the 'ksigma'
    Fourier-space kernels from Bernstein et al., arXiv:1508.05655.

    If the fwhm of the weight/kernel function is of similar size to the PSF or
    smaller, then the object properties returned by this fitter will be very noisy.

    Parameters
    ----------
    fwhm : float
        This parameter is the approximate real-space FWHM of the kernel. The units are
        whatever the Jacobian on the obs converts pixels units to. This is typically
        arcseconds.
    pad_factor : int, optional
        The factor by which to pad the FFTs used for the image. Default is 4.
    ap_rad : float, optional
        The apodization radius for the stamp in pixels. The default of 1.5 is likely
        fine for most ground based surveys.
    fwhm_smooth : float, optional
        If non-zero, this optional applies additional Gaussian smoothing to the
        object before computing the moments. Typically a non-zero value results
        in less shape noise.
    """
    def __init__(
        self, fwhm, pad_factor=4, ap_rad=1.5, fwhm_smooth=0
    ):
        super().__init__(
            fwhm, 'ksigma', pad_factor=pad_factor, ap_rad=ap_rad,
            fwhm_smooth=fwhm_smooth,
        )


class PGaussMom(PrePSFMom):
    """Measure pre-PSF weighted real-space moments w/ a Gaussian kernel.

    This fitter differs from `GaussMom` in that it deconvolves the PSF first.

    If the fwhm of the weight/kernel function is of similar size to the PSF or
    smaller, then the object properties returned by this fitter will be very noisy.

    Parameters
    ----------
    fwhm : float
        This parameter is the real-space FWHM of the kernel. The units are
        whatever the Jacobian on the obs converts pixels units to. This is typically
        arcseconds.
    pad_factor : int, optional
        The factor by which to pad the FFTs used for the image. Default is 4.
    ap_rad : float, optional
        The apodization radius for the stamp in pixels. The default of 1.5 is likely
        fine for most ground based surveys.
    fwhm_smooth : float, optional
        If non-zero, this optional applies additional Gaussian smoothing to the
        object before computing the moments. Typically a non-zero value results
        in less shape noise.
    """
    def __init__(
        self, fwhm, pad_factor=4, ap_rad=1.5, fwhm_smooth=0,
    ):
        super().__init__(
            fwhm, 'pgauss', pad_factor=pad_factor, ap_rad=ap_rad,
            fwhm_smooth=fwhm_smooth,
        )


# keep this here for API consistency
PrePSFGaussMom = PGaussMom


def _measure_moments_fft(
    kim, kpsf_im, tot_var, eff_pad_factor, kernels, drow, dcol,
):
    # we only need to do things where the kernel is non-zero
    # this saves a bunch of CPU cycles
    msk = kernels["msk"]
    dim = kim.shape[0]

    # deconvolve PSF
    kim, kpsf_im, _ = _deconvolve_im_psf_inplace(
        kim[msk],
        kpsf_im[msk],
        # max amplitude is flux which is 0,0 in the standard FFT convention
        np.abs(kpsf_im[0, 0]),
    )

    # put in phase shift as described above
    # the sin and cos are expensive so we only compute them where we will
    # use the image which is in the msk
    if drow != 0 or dcol != 0:
        cen_phase = _compute_cen_phase_shift(drow, dcol, dim, msk=msk)
        kim *= cen_phase

    fkf = kernels["fkf"]
    fkr = kernels["fkr"]
    fkp = kernels["fkp"]
    fkc = kernels["fkc"]

    mom_norm = kernels["fk00"]

    return _measure_moments_fft_numba(
        kim, kpsf_im, dim, eff_pad_factor, fkf, fkr, fkp, fkc, mom_norm, tot_var,
    )


@njit
def _measure_moments_fft_numba(
    kim, kpsf_im, dim, eff_pad_factor, fkf, fkr, fkp, fkc, mom_norm, tot_var,
):
    # build the flux, radial, plus and cross kernels / moments
    # the inverse FFT in our convention has a factor of 1/n per dimension
    # the sums below are inverse FFTs but only computing the values at the
    # real-space center of the object (0 in our coordinate system).
    # thus we code the factor of 1/n by hand
    df = 1/dim
    df2 = df * df
    df4 = df2 * df2

    mf = np.sum((kim * fkf).real) * df2
    mr = np.sum((kim * fkr).real) * df2
    mp = np.sum((kim * fkp).real) * df2
    mc = np.sum((kim * fkc).real) * df2

    # build a covariance matrix of the moments
    # here we assume each Fourier mode is independent and sum the variances
    # the variance in each mode is simply the total variance over the input image
    # we need a factor of the padding to correct for something...
    m_cov = np.zeros((6, 6))
    # TODO
    # FIXME
    # set these for real
    m_cov[0, 0] = 1
    m_cov[1, 1] = 1
    tot_var *= eff_pad_factor**2
    tot_var_df4 = tot_var * df4
    psf_kerns_fac = 1 / kpsf_im
    kerns = [
        fkp * psf_kerns_fac,
        fkc * psf_kerns_fac,
        fkr * psf_kerns_fac,
        fkf * psf_kerns_fac,
    ]
    conj_kerns = [np.conj(k) for k in kerns]
    for i in range(2, 6):
        for j in range(i, 6):
            # subtract two since kernels start at second moments
            m_cov[i, j] = np.sum((kerns[i-2] * conj_kerns[j-2]).real) * tot_var_df4
            m_cov[j, i] = m_cov[i, j]

    mom = np.array([np.nan, np.nan, mp, mc, mr, mf])

    return mom, m_cov, mom_norm


@njit
def _ap_kern_kern(x, m, h):
    # cumulative triweight kernel
    y = (x - m) / h + 3
    if y < -3:
        return 0
    elif y > 3:
        return 1
    else:
        val = (
            -5 * y ** 7 / 69984
            + 7 * y ** 5 / 2592
            - 35 * y ** 3 / 864
            + 35 * y / 96
            + 1 / 2
        )
        return val


@njit
def _build_square_apodization_mask(ap_rad, ap_mask):
    ap_range = int(6*ap_rad + 0.5)

    ny, nx = ap_mask.shape
    for y in range(min(ap_range+1, ny)):
        for x in range(nx):
            ap_mask[y, x] *= _ap_kern_kern(y, ap_range, ap_rad)
            ap_mask[ny-1 - y, x] *= _ap_kern_kern(y, ap_range, ap_rad)

    for y in range(ny):
        for x in range(min(ap_range+1, nx)):
            ap_mask[y, x] *= _ap_kern_kern(x, ap_range, ap_rad)
            ap_mask[y, nx - 1 - x] *= _ap_kern_kern(x, ap_range, ap_rad)


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

    im_padded = np.pad(
        im,
        (pad_width_before, pad_width_after),
        mode='constant',
        constant_values=0,
    )

    return im_padded, pad_width_before, pad_width_after


def _compute_cen_phase_shift(cen_row, cen_col, dim, msk=None):
    """computes exp(i*2*pi*k*cen) for shifting the phases of FFTS.

    If you feed the centroid of a profile, then this factor times the raw FFT
    of that profile will result in an FFT centered at the profile.
    """
    f = fft.fftfreq(dim) * (2.0 * np.pi)
    pxy = _compute_cen_phase_shift_numba(f, cen_row, cen_col)

    if msk is not None:
        pxy = pxy[msk]

    return pxy


@njit
def _compute_cen_phase_shift_numba(f, cen_row, cen_col):
    # this reshaping makes sure the arrays broadcast nicely into a grid
    fx = f.reshape(1, -1)
    fy = f.reshape(-1, 1)
    kcen_x = fx*cen_col
    kcen_y = fy*cen_row
    px = np.cos(kcen_x) + 1j*np.sin(kcen_x)
    py = np.cos(kcen_y) + 1j*np.sin(kcen_y)
    pxy = px * py
    return pxy


def _compute_cen_phase_shift_orig(cen_row, cen_col, dim, msk=None):
    """computes exp(i*2*pi*k*cen) for shifting the phases of FFTS.

    If you feed the centroid of a profile, then this factor times the raw FFT
    of that profile will result in an FFT centered at the profile.
    """
    f = fft.fftfreq(dim) * (2.0 * np.pi)
    # this reshaping makes sure the arrays broadcast nicely into a grid
    fx = f.reshape(1, -1)
    fy = f.reshape(-1, 1)
    kcen = fy*cen_row + fx*cen_col
    if msk is not None:
        return np.cos(kcen[msk]) + 1j*np.sin(kcen[msk])
    else:
        return np.cos(kcen) + 1j*np.sin(kcen)


def _zero_pad_and_compute_fft_impl(im, cen_row, cen_col, target_dim, ap_rad):
    """zero pad and compute the FFT

    Returns the fft, cen_row in the padded image, and cen_col in the padded image.
    """
    if ap_rad > 0:
        ap_mask = np.ones_like(im)
        _build_square_apodization_mask(ap_rad, ap_mask)
        im = im * ap_mask

    pim, pad_width_before, _ = _zero_pad_image(im, target_dim)
    pad_cen_row = cen_row + pad_width_before
    pad_cen_col = cen_col + pad_width_before
    kpim = fft.fftn(pim)
    return kpim, pad_cen_row, pad_cen_col


# see https://stackoverflow.com/a/52332109 for how this works
@functools.lru_cache(maxsize=128)
def _zero_pad_and_compute_fft_cached_impl(
    im_tuple, cen_row, cen_col, target_dim, ap_rad
):
    return _zero_pad_and_compute_fft_impl(
        np.array(im_tuple), cen_row, cen_col, target_dim, ap_rad
    )


@functools.wraps(_zero_pad_and_compute_fft_impl)
def _zero_pad_and_compute_fft_cached(im, cen_row, cen_col, target_dim, ap_rad):
    return _zero_pad_and_compute_fft_cached_impl(
        tuple(tuple(ii) for ii in im),
        float(cen_row), float(cen_col), int(target_dim), float(ap_rad)
    )


_zero_pad_and_compute_fft_cached.cache_info \
    = _zero_pad_and_compute_fft_cached_impl.cache_info
_zero_pad_and_compute_fft_cached.cache_clear \
    = _zero_pad_and_compute_fft_cached_impl.cache_clear


def _deconvolve_im_psf_inplace(kim, kpsf_im, max_amp, min_psf_frac=1e-5):
    """deconvolve the PSF from an image in place.

    Returns the deconvolved image, the kpsf_im used,
    and a bool mask marking PSF modes that were truncated
    """
    min_amp = min_psf_frac * max_amp
    abs_kpsf_im = np.abs(kpsf_im)
    msk = abs_kpsf_im <= min_amp
    if np.any(msk):
        kpsf_im[msk] = kpsf_im[msk] / abs_kpsf_im[msk] * min_amp

    kim /= kpsf_im
    return kim, kpsf_im, msk


def _get_fwhm_smooth_profile(fwhm_smooth, fmag2):
    sigma_smooth = fwhm_to_sigma(fwhm_smooth)
    chi2_2_smooth = sigma_smooth * sigma_smooth / 2 * fmag2
    exp_val_smooth = np.zeros_like(fmag2)
    msk_smooth = (chi2_2_smooth < FASTEXP_MAX_CHI2/2) & (chi2_2_smooth >= 0)
    exp_val_smooth[msk_smooth] = fexp_arr(-chi2_2_smooth[msk_smooth])
    return exp_val_smooth


@functools.lru_cache(maxsize=128)
def _ksigma_kernels(
    dim,
    kernel_size,
    dvdrow, dvdcol, dudrow, dudcol,
    fwhm_smooth,
):
    """This function builds a ksigma kernel in Fourier-space.

    It returns a dict of all of the kernels needed to measure moments in
    real-space by summing the kernel against the FFT of an image.
    """
    # we first get the Fourier modes in the u,v plane
    f = fft.fftfreq(dim) * (2.0 * np.pi)
    fx = f.reshape(1, -1)
    fy = f.reshape(-1, 1)
    Atinv = np.linalg.inv([[dvdrow, dvdcol], [dudrow, dudcol]]).T
    fv = Atinv[0, 0] * fy + Atinv[0, 1] * fx
    fu = Atinv[1, 0] * fy + Atinv[1, 1] * fx

    # now draw the kernels
    # we are computing the Bernstein et al., arXiv:1508.05655. ksigma kernel which is
    # W(k) = (1 - (k*sigma/sqrt(2n))^2)^n for k < sqrt(2*n)/sigma
    # and zero otherwise. we follow them and set n = 4.
    n = 4
    sigma = fwhm_to_sigma(kernel_size)
    kmax2 = 2*n/sigma**2
    fu2 = fu**2
    fv2 = fv**2
    fmag2 = fu2 + fv2
    msk = fmag2 < kmax2

    # from here we work with non-zero portion only
    fmag2 = fmag2[msk]
    fu = fu[msk]
    fu2 = fu2[msk]
    fv = fv[msk]
    fv2 = fv2[msk]

    karg = 1.0 - fmag2/kmax2
    karg2 = karg*karg
    karg3 = karg2*karg
    karg4 = karg3*karg

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
    fkf = karg4 * knrm

    # when the kernel support extends beyong the FFT region, we raise an error
    nrm = np.sum(fkf)/dim/dim
    if not np.allclose(nrm, 1.0, atol=1e-5, rtol=0):
        raise FFTRangeError(
            "FFT size appears to be too small for ksigma kernel size %f: "
            "norm = %f (should be 1)!" % (kernel_size, nrm)
        )

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
    two_knrm_dWdk2 = (-knrm * 8.0 / kmax2) * karg3
    four_knrm_dW2dk22 = (knrm * 48 / kmax2**2) * karg2

    # add smoothing after norm check above for kernel size
    if fwhm_smooth > 0:
        exp_val_smooth = _get_fwhm_smooth_profile(fwhm_smooth, fmag2)
        fkf *= exp_val_smooth
        two_knrm_dWdk2 *= exp_val_smooth
        four_knrm_dW2dk22 *= exp_val_smooth

    # the linear combinations here measure the moments proportional to the size
    # and shears - see the Mf, Mr, M+, Mx moments in Bernstein et al., arXiv:1508.05655
    # fkr = fkxx + fkyy
    # fkp = fkxx - fkyy
    # fkc = 2 * fkxy
    fkr = -2 * two_knrm_dWdk2 - fmag2 * four_knrm_dW2dk22
    fkp = -(fu2 - fv2) * four_knrm_dW2dk22
    fkc = -2 * fu * fv * four_knrm_dW2dk22

    return dict(
        fkf=fkf,
        fkr=fkr,
        fkp=fkp,
        fkc=fkc,
        msk=msk,
        nrm=nrm,
        fk00=knrm,
    )


@functools.lru_cache(maxsize=128)
def _gauss_kernels(
    dim,
    kernel_size,
    dvdrow, dvdcol, dudrow, dudcol,
    fwhm_smooth,
):
    """This function builds a Gaussian kernel in Fourier-space.

    It returns a dict of all of the kernels needed to measure moments in
    real-space by summing the kernel against the FFT of an image.
    """
    # we first get the Fourier modes in the u,v plane
    f = fft.fftfreq(dim) * (2.0 * np.pi)
    fx = f.reshape(1, -1)
    fy = f.reshape(-1, 1)
    Atinv = np.linalg.inv([[dvdrow, dvdcol], [dudrow, dudcol]]).T
    fv = Atinv[0, 0] * fy + Atinv[0, 1] * fx
    fu = Atinv[1, 0] * fy + Atinv[1, 1] * fx

    # now draw the kernels
    sigma = fwhm_to_sigma(kernel_size)
    sigma2 = sigma * sigma
    fu2 = fu**2
    fv2 = fv**2
    fmag2 = fu2 + fv2
    exp_fac = sigma2 / 2
    chi2_2 = exp_fac * fmag2
    msk = (chi2_2 < FASTEXP_MAX_CHI2/2) & (chi2_2 >= 0)

    # from here we work with non-zero portion only
    fmag2 = fmag2[msk]
    fu = fu[msk]
    fu2 = fu2[msk]
    fv = fv[msk]
    fv2 = fv2[msk]
    chi2_2 = chi2_2[msk]
    exp_val = fexp_arr(-chi2_2)

    # we need to normalize the kernel to unity in real space at the object center
    # we also need a factor of the k-space area element so that when we
    # sum an image against this kernel, we get an integral
    detAtinv = np.abs(np.linalg.det(Atinv))

    # the total factor is the k-space element times the right normalization in
    # fourier space for a unit peak kernel in real space
    # we multiply by this value
    knrm = detAtinv * np.pi * 2 * sigma2

    # now build the kernels
    # the flux kernel is easy since it is the kernel itself
    fkf = exp_val * knrm

    # when the kernel support extends beyong the FFT region, we raise an error
    nrm = np.sum(fkf)/dim/dim
    if not np.allclose(nrm, 1.0, atol=1e-5, rtol=0):
        raise FFTRangeError(
            "FFT size appears to be too small for gauss kernel size %f: "
            "norm = %f (should be 1)!" % (kernel_size, nrm)
        )

    # add smoothing after norm check above for kernel size
    if fwhm_smooth > 0:
        exp_val_smooth = _get_fwhm_smooth_profile(fwhm_smooth, fmag2)
        fkf *= exp_val_smooth

    # the moment kernels take a bit more work
    # product by u^2 in real space is -dk^2/dku^2 in Fourier space
    # same holds for v and cross deriv is -dk^2/dkudkv
    # in general
    #
    #   dWdkx = dWdk2 * dk2dx = 2kx * dWdk2
    #   dW^2dkx^2 = 2 dWdk2 + 4 kx^2 * dW^2dk2^2
    #
    # The other derivs are similar.
    # I've combined a lot of the math below.

    # the linear combinations here measure the moments proportional to the size
    # and shears - see the Mf, Mr, M+, Mx moments in Bernstein et al., arXiv:1508.05655
    # fkr = fkxx + fkyy
    # fkp = fkxx - fkyy
    # fkc = 2 * fkxy
    fkfac = 2 * exp_fac
    fkfac2 = 4 * exp_fac**2
    fkr = (2 * fkfac - fkfac2 * fmag2) * fkf
    fkp = fkfac2 * (fv2 - fu2) * fkf
    fkc = -2 * fkfac2 * fu * fv * fkf

    return dict(
        fkf=fkf,
        fkr=fkr,
        fkp=fkp,
        fkc=fkc,
        msk=msk,
        nrm=nrm,
        fk00=knrm,
    )


def _check_obs_and_get_psf_obs(obs, no_psf):
    if not isinstance(obs, Observation):
        raise ValueError("input obs must be an Observation")

    shape = obs.image.shape
    if shape[0] != shape[1]:
        raise ValueError(f'pre-psf moments require a square image, got {shape}')

    if not obs.has_psf() and not no_psf:
        raise RuntimeError("The PSF must be set to measure a pre-PSF moment!")

    if not no_psf:
        psf_obs = obs.get_psf()

        if psf_obs.jacobian.get_galsim_wcs() != obs.jacobian.get_galsim_wcs():
            raise RuntimeError(
                "The PSF and observation must have the same WCS "
                "Jacobian for measuring pre-PSF moments."
            )
    else:
        psf_obs = None

    return psf_obs
