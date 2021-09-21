import logging

import numpy as np
import scipy.fft as fft

from ngmix.observation import Observation
from ngmix.moments import fwhm_to_sigma
from ngmix.util import get_ratio_error
from ngmix.fastexp_nb import fexp_arr, FASTEXP_MAX_CHI2


logger = logging.getLogger(__name__)


class _PrePSFMom(object):
    """Measure pre-PSF weighted real-space moments.

    This class is not meant to be used directly. Instead use either `KSigmaMom`
    or `PrePSFGaussMom`.

    If the fwhm of the weight/kernel function is of similar size to the PSF or
    smaller, then the object properties returned by this fitter will be very noisy.

    Parameters
    ----------
    fwhm : float
        This parameter is the approximate real-space FWHM of the kernel. The units are
        whatever the Jacobian on the obs converts pixels units to. This is typically
        arcseconds.
    kernel : str
        The kernel to use. Either `ksigma` or `gauss`.
    pad_factor : int, optional
        The factor by which to pad the FFTs used for the image. Default is 4.
    """
    def __init__(self, fwhm, *, kernel, pad_factor=4):
        self.fwhm = fwhm
        self.pad_factor = pad_factor
        self.kernel = kernel

    def go(self, obs, return_kernels=False, no_psf=False):
        """Measure the pre-PSF ksigma moments.

        Parameters
        ----------
        obs : Observation
            The observation to measure.
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
        if not isinstance(obs, Observation):
            raise ValueError("input obs must be an Observation")

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
        kim, im_row, im_col = _zero_pad_and_compute_fft(
            obs.image, obs.jacobian.row0, obs.jacobian.col0, target_dim,
        )
        fft_dim = kim.shape[0]

        if psf_obs is not None:
            kpsf_im, psf_im_row, psf_im_col = _zero_pad_and_compute_fft(
                psf_obs.image,
                psf_obs.jacobian.row0, psf_obs.jacobian.col0,
                target_dim,
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
                target_dim,
                self.fwhm,
                obs.jacobian.dvdrow, obs.jacobian.dvdcol,
                obs.jacobian.dudrow, obs.jacobian.dudcol,
            )
        elif self.kernel == "gauss":
            kernels = _gauss_kernels(
                target_dim,
                self.fwhm,
                obs.jacobian.dvdrow, obs.jacobian.dvdcol,
                obs.jacobian.dudrow, obs.jacobian.dudcol,
            )
        else:
            raise ValueError(
                "The kernel '%s' _PrePSFMom is not recognized!" % self.kernel
            )

        # compute the total variance from weight map
        msk = obs.weight > 0
        tot_var = np.sum(1.0 / obs.weight[msk])

        # run the actual measurements and return
        mom, mom_cov = _measure_moments_fft(
            kim, kpsf_im, tot_var, eff_pad_factor, kernels,
            im_row - psf_im_row, im_col - psf_im_col,
        )
        res = _make_mom_res(mom, mom_cov)
        if res['flags'] != 0:
            logger.debug("ksigma pre-psf moments failed: %s" % res['flagstr'])

        if return_kernels:
            # put the kernels back into their unpacked state
            full_kernels = {}
            for k in kernels:
                if k == "msk":
                    continue
                full_kernels[k] = np.zeros((fft_dim, fft_dim), dtype=np.complex128)
                full_kernels[k][kernels["msk"]] = kernels[k]
            res["kernels"] = full_kernels

        return res


class KSigmaMom(_PrePSFMom):
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
    """
    def __init__(self, fwhm, pad_factor=4):
        super().__init__(fwhm, kernel='ksigma', pad_factor=pad_factor)


class PrePSFGaussMom(_PrePSFMom):
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
    """
    def __init__(self, fwhm, pad_factor=4):
        super().__init__(fwhm, kernel='gauss', pad_factor=pad_factor)


def _measure_moments_fft(kim, kpsf_im, tot_var, eff_pad_factor, kernels, drow, dcol):
    # we only need to do things where the ksigma kernel is non-zero
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

    # build the flux, radial, plus and cross kernels / moments
    # the inverse FFT in our convention has a factor of 1/n per dimension
    # the sums below are inverse FFTs but only computing the values at the
    # real-space center of the object (0 in our coordinate system).
    # thus we code the factor of 1/n by hand
    df = 1/dim
    df2 = df * df
    df4 = df2 * df2

    # we only sum where the kernel is nonzero
    fkf = kernels["fkf"]
    fkr = kernels["fkr"]
    fkp = kernels["fkp"]
    fkc = kernels["fkc"]

    mf = np.sum((kim * fkf).real) * df2
    mr = np.sum((kim * fkr).real) * df2
    mp = np.sum((kim * fkp).real) * df2
    mc = np.sum((kim * fkc).real) * df2

    # build a covariance matrix of the moments
    # here we assume each Fourier mode is independent and sum the variances
    # the variance in each mode is simply the total variance over the input image
    # we need a factor of the padding to correct for something...
    m_cov = np.zeros((4, 4))
    tot_var *= eff_pad_factor**2
    tot_var_df4 = tot_var * df4
    kerns = [fkf / kpsf_im, fkr / kpsf_im, fkp / kpsf_im, fkc / kpsf_im]
    conj_kerns = [np.conj(k) for k in kerns]
    for i in range(4):
        for j in range(i, 4):
            m_cov[i, j] = np.sum((kerns[i] * conj_kerns[j]).real) * tot_var_df4
            m_cov[j, i] = m_cov[i, j]

    mom = np.array([mf, mr, mp, mc])

    return mom, m_cov


def _make_mom_res(mom, mom_cov):
    # now finally build the outputs and their errors
    res = {}
    res["flags"] = 0
    res["flagstr"] = ""
    res["flux"] = mom[0]
    res["mom"] = mom
    res["mom_cov"] = mom_cov
    res["flux_flags"] = 0
    res["flux_flagstr"] = ""
    res["T_flags"] = 0
    res["T_flagstr"] = ""

    # we fill these in later if T > 0 and flux cov is positive
    res["flux_err"] = 9999.0
    res["T"] = -9999.0
    res["T_err"] = 9999.0
    res["s2n"] = -9999.0
    res["e1"] = 9999.0
    res["e2"] = 9999.0
    res["e"] = np.array([-9999.0, -9999.0])
    res["e_err"] = np.array([9999.0, 9999.0])
    res["e_cov"] = np.diag([9999.0, 9999.0])
    res["mom_err"] = np.ones(4) * 9999.0

    # handle flux-only
    if np.diagonal(mom_cov)[0] > 0:
        res["flux_err"] = np.sqrt(mom_cov[0, 0])
        res["s2n"] = res["flux"] / res["flux_err"]
    else:
        res["flux_flags"] |= 0x40
        res["flux_flagstr"] += 'zero or neg flux var;'

    # handle flux+T only
    if np.all(np.diagonal(mom_cov)[0:2] > 0):
        if mom[0] > 0:
            res["T"] = mom[1] / mom[0]
            res["T_err"] = get_ratio_error(
                mom[1], mom[0],
                mom_cov[1, 1], mom_cov[0, 0], mom_cov[0, 1]
            )
        else:
            # flux <= 0.0
            res["T_flags"] |= 0x4
            res["T_flagstr"] += "flux <= 0.0;"
    else:
        res["T_flags"] |= 0x40
        res["T_flagstr"] += 'zero or neg flux/T var;'

    # now handle full flags
    if np.all(np.diagonal(mom_cov) > 0):
        res["mom_err"] = np.sqrt(np.diagonal(mom_cov))
    else:
        res["flags"] |= 0x40
        res["flagstr"] += 'zero or neg moment var;'

    if res["flags"] == 0:
        if mom[0] > 0:
            if res["T"] > 0:
                res["pars"] = np.array([
                    0, 0,
                    mom[2]/mom[0],
                    mom[3]/mom[0],
                    mom[1]/mom[0],
                    mom[0],
                ])
                res["e1"] = mom[2] / mom[1]
                res["e2"] = mom[3] / mom[1]
                res["e"] = np.array([res["e1"], res["e2"]])
                e_err = np.zeros(2)
                e_err[0] = get_ratio_error(
                    mom[2], mom[1],
                    mom_cov[2, 2], mom_cov[1, 1], mom_cov[1, 2]
                )
                e_err[1] = get_ratio_error(
                    mom[3], mom[1],
                    mom_cov[3, 3], mom_cov[1, 1], mom_cov[1, 3]
                )
                if np.all(np.isfinite(e_err)):
                    res["e_err"] = e_err
                    res["e_cov"] = np.diag(e_err**2)
                else:
                    # bad e_err
                    res["flags"] |= 0x100
                    res["flagstr"] += "non-finite shape errors;"
            else:
                # T <= 0.0
                res["flags"] |= 0x8
                res["flagstr"] += "T <= 0.0;"
        else:
            # flux <= 0.0
            res["flags"] |= 0x4
            res["flagstr"] += "flux <= 0.0;"

    return res


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
    # this reshaping makes sure the arrays broadcast nicely into a grid
    fx = f.reshape(1, -1)
    fy = f.reshape(-1, 1)
    kcen = fy*cen_row + fx*cen_col
    if msk is not None:
        return np.cos(kcen[msk]) + 1j*np.sin(kcen[msk])
    else:
        return np.cos(kcen) + 1j*np.sin(kcen)


def _zero_pad_and_compute_fft(im, cen_row, cen_col, target_dim):
    """zero pad and compute the FFT

    Returns the fft, cen_row in the padded image, and cen_col in the padded image.
    """
    pim, pad_width_before, _ = _zero_pad_image(im, target_dim)
    pad_cen_row = cen_row + pad_width_before
    pad_cen_col = cen_col + pad_width_before
    kpim = fft.fftn(pim)
    return kpim, pad_cen_row, pad_cen_col


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

    # when the kernel support extends beyong the FFt region, we have to do more
    # thus we have to normalize the discrete FFT to unit peak in real-space
    # for kernels much smaller than the image size, this comes out fine
    # for kernels much bigger than the image size, you need an extra factor to
    # correct for the truncated aperture
    nrm = np.sum(fkf)/dim/dim
    fkf /= nrm
    knrm /= nrm

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
    )


def _gauss_kernels(
    dim,
    kernel_size,
    dvdrow, dvdcol, dudrow, dudcol,
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
    chi2 = exp_fac * fmag2
    msk = chi2 < FASTEXP_MAX_CHI2

    # from here we work with non-zero portion only
    fmag2 = fmag2[msk]
    fu = fu[msk]
    fu2 = fu2[msk]
    fv = fv[msk]
    fv2 = fv2[msk]
    chi2 = chi2[msk]
    exp_val = fexp_arr(-chi2)

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

    # when the kernel support extends beyong the FFt region, we have to do more
    # thus we have to normalize the discrete FFT to unit peak in real-space
    # for kernels much smaller than the image size, this comes out fine
    # for kernels much bigger than the image size, you need an extra factor to
    # correct for the truncated aperture
    nrm = np.sum(fkf)/dim/dim
    fkf /= nrm
    knrm /= nrm

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
    fkr = (fkfac2 * fmag2 - 2 * fkfac) * fkf
    fkp = fkfac2 * (fu2 - fv2) * fkf
    fkc = 2 * fkfac2 * fu * fv * fkf

    return dict(
        fkf=fkf,
        fkr=fkr,
        fkp=fkp,
        fkc=fkc,
        msk=msk,
    )
