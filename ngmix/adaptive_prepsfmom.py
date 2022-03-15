import logging

import numpy as np
import scipy.fft as fft

from ngmix.moments import make_mom_result, e2mom, fwhm_to_T, get_Tround
from ngmix.fastexp_nb import FASTEXP_MAX_CHI2, fexp_arr
from ngmix.prepsfmom import (
    _measure_moments_fft,
    _check_obs_and_get_psf_obs,
    _zero_pad_and_compute_fft,
)
from ngmix.moments import mom2e, T_to_fwhm, MOMENTS_NAME_MAP
from ngmix.prepsfmom import _ap_kern_kern
from ngmix.gmix.gmix_nb import GMIX_LOW_DETVAL
import ngmix.flags
from ngmix.admom.admom import (
    DEFAULT_MAXITER,
    DEFAULT_TTOL,
)


logger = logging.getLogger(__name__)


def _mom2e1e2fwhm(Irr, Irc, Icc):
    e1, e2, T = mom2e(Irr, Irc, Icc)
    return e1, e2, T_to_fwhm(T)


def _scale_e1e2T(e1, e2, T, minT, deltaT, efac):
    fac = _ap_kern_kern(get_Tround(T, e1, e2), minT + 6*deltaT, deltaT)
    return fac*e1, fac*e2, T * fac + minT * (1-fac)


def _truncate_e2mom(e1, e2, T, minT, deltaT):
    efac = 1.0 + 2.0 * np.sqrt(e1**2 + e2**2)
    e1, e2, T = _scale_e1e2T(e1, e2, T, minT, deltaT, efac)
    Wrr, Wrc, Wcc = e2mom(e1, e2, T)
    return Wrr, Wrc, Wcc


def _deweight_moments(Irr, Irc, Icc, Wrr, Wrc, Wcc, minT, deltaT):
    # measured moments
    detm = Irr*Icc - Irc*Irc
    if detm <= GMIX_LOW_DETVAL:
        we1, we2, wT = mom2e(Wrr, Wrc, Wcc)
        return _truncate_e2mom(we1, we2, wT, minT, deltaT)

    detw = Wrr*Wcc - Wrc*Wrc
    if detw <= GMIX_LOW_DETVAL:
        we1, we2, wT = mom2e(Wrr, Wrc, Wcc)
        return _truncate_e2mom(we1, we2, wT, minT, deltaT)

    idetw = 1.0/detw
    idetm = 1.0/detm

    # Nrr etc. are actually of the inverted covariance matrix
    Nrr = Icc*idetm - Wcc*idetw
    Ncc = Irr*idetm - Wrr*idetw
    Nrc = -Irc*idetm + Wrc*idetw
    detn = Nrr*Ncc - Nrc*Nrc

    if detn <= GMIX_LOW_DETVAL:
        we1, we2, wT = mom2e(Wrr, Wrc, Wcc)
        return _truncate_e2mom(we1, we2, wT, minT, deltaT)

    # now set from the inverted matrix
    idetn = 1./detn
    Wrr = Ncc*idetn
    Wcc = Nrr*idetn
    Wrc = -Nrc*idetn

    we1, we2, wT = mom2e(Wrr, Wrc, Wcc)
    return _truncate_e2mom(we1, we2, wT, minT, deltaT)


def _comp_rel_err(p1, p2, tol):
    return np.abs(p1-p2) < tol * np.abs(p2)


class PrePSFAdmom(object):
    """Measure pre-PSF adaptive moments.

    If the minimum fwhm of the weight/kernel function is of similar size to the PSF or
    smaller, then the object properties returned by this fitter will be very noisy.

    Parameters
    ----------
    min_fwhm : float
        The minimum allowed FWHM for the moments. The weight kernel will slowy go
        to a round kernel as the FWHM approaches this minimum.
    delta_fwhm : float, optional.
        The shape of the weight kernel will go to round over region of 6*delta_fwhm.
        Default is 0.01 and tends to work well.
    pad_factor : int, optional
        The factor by which to pad the FFTs used for the image. Default is 4.
    ap_rad : float, optional
        The apodization radius for the stamp in pixels. The default of 1.5 is likely
        fine for most ground based surveys.
    maxiter: integer, optional
        Maximum number of iterations, default 200
    Ttol: float, optional
        Relative tolerance in the moments T <x^2> + <y^2> to determine
        convergence. Default is 1.0e-3.
    rng: np.random.RandomState or None, optional
        Random state for creating starting guess.
    """
    def __init__(
        self, min_fwhm, delta_fwhm=0.01, pad_factor=4, ap_rad=1.5,
        maxiter=DEFAULT_MAXITER, Ttol=DEFAULT_TTOL,
        rng=None,
    ):
        self.min_fwhm = min_fwhm
        self.delta_fwhm = delta_fwhm
        self.pad_factor = pad_factor
        self.ap_rad = ap_rad
        self.kernel = "pam"
        self.kind = "pam"
        self.maxiter = maxiter
        self.Ttol = Ttol
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

    def go(self, obs, return_kernels=False, no_psf=False):
        """Measure the pre-PSF adaptive moments.

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
        d = self._prep_data(obs, psf_obs)
        return self._run_admom(obs, psf_obs, return_kernels, d)

    def _run_admom(self, obs, psf_obs, return_kernels, d):
        minT = fwhm_to_T(self.min_fwhm)
        # this is dT/dFWHM|_{min_fwhm} * delta_fwhm
        deltaT = 4.0 * self.min_fwhm * self.delta_fwhm

        mf_ind = MOMENTS_NAME_MAP["MF"]
        mt_ind = MOMENTS_NAME_MAP["MT"]
        m1_ind = MOMENTS_NAME_MAP["M1"]
        m2_ind = MOMENTS_NAME_MAP["M2"]

        guessT = minT + 6*deltaT + self.rng.uniform(low=0.0, high=2*deltaT)

        wrr, wrc, wcc = _truncate_e2mom(
            self.rng.uniform(low=-0.1, high=0.1),
            self.rng.uniform(low=-0.1, high=0.1),
            guessT,
            minT,
            deltaT,
        )
        Iccold = Irrold = Ircold = np.nan
        am_flags = 0

        for i in range(self.maxiter):
            mom, mom_cov, kernels = self._meas_mom(wrr, wrc, wcc, obs, d)

            if mom[mf_ind] <= 0:
                am_flags |= ngmix.flags.NONPOS_FLUX
                break

            MT = mom[mt_ind] / mom[mf_ind]
            M1 = mom[m1_ind] / mom[mf_ind]
            M2 = mom[m2_ind] / mom[mf_ind]
            Irc = M2 / 2.0
            Icc = (MT + M1) / 2.0
            Irr = (MT - M1) / 2.0

            we1, we2, wT = mom2e(wrr, wrc, wcc)

            if (
                (
                    _comp_rel_err(Icc, Iccold, self.Ttol)
                    and _comp_rel_err(Irr, Irrold, self.Ttol)
                    and _comp_rel_err(Irc, Ircold, self.Ttol)
                ) or (
                    np.allclose(we1, 0)
                    and np.allclose(we2, 0)
                    and np.allclose(wT, minT)

                )
            ):
                break
            else:
                Iccold = Icc
                Irrold = Irr
                Ircold = Irc
                wrr, wrc, wcc = _deweight_moments(
                    Irr, Irc, Icc, wrr, wrc, wcc, minT, deltaT
                )

        res = make_mom_result(mom, mom_cov)
        e1, e2, T = mom2e(wrr, wrc, wcc)
        res["weight_e1e2T"] = {"e1": e1, "e2": e2, "T": T}

        res["numiter"] = i+1
        if i+1 == self.maxiter:
            am_flags |= ngmix.flags.MAXITER

        res["flags"] |= am_flags
        res["flux_flags"] |= am_flags
        res["T_flags"] |= am_flags

        res["flagstr"] = ngmix.flags.get_flags_str(res["flags"])
        res["T_flagstr"] = ngmix.flags.get_flags_str(res["T_flags"])
        res["flux_flagstr"] = ngmix.flags.get_flags_str(res["flux_flags"])

        if res['flags'] != 0:
            logger.debug("pre-psf adaptive moments failed: %s" % res['flagstr'])

        if return_kernels:
            # put the kernels back into their unpacked state
            full_kernels = {}
            for k in kernels:
                if k == "msk":
                    continue
                if k == "nrm":
                    full_kernels[k] = kernels[k]
                else:
                    full_kernels[k] = np.zeros(
                        (d["fft_dim"], d["fft_dim"]), dtype=np.complex128,
                    )
                    full_kernels[k][kernels["msk"]] = kernels[k]
            res["kernels"] = full_kernels

        return res

    def _meas_mom(self, wrr, wrc, wcc, obs, d):
        # now build the kernels
        e1, e2, fwhm = _mom2e1e2fwhm(wrr, wrc, wcc)
        kernels = _gauss_shape_kernels(
            d["target_dim"],
            fwhm,
            e1,
            e2,
            obs.jacobian.dvdrow, obs.jacobian.dvdcol,
            obs.jacobian.dudrow, obs.jacobian.dudcol,
        )

        # run the actual measurements and return
        mom, mom_cov = _measure_moments_fft(
            d["kim"].copy(), d["kpsf_im"].copy(), d["tot_var"], d["eff_pad_factor"],
            kernels,
            d["im_row"] - d["psf_im_row"], d["im_col"] - d["psf_im_col"],
        )
        return mom, mom_cov, kernels

    def _prep_data(self, obs, psf_obs):
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
            self.ap_rad,
        )
        fft_dim = kim.shape[0]

        if psf_obs is not None:
            kpsf_im, psf_im_row, psf_im_col = _zero_pad_and_compute_fft(
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

        # compute the total variance from weight map
        msk = obs.weight > 0
        tot_var = np.sum(1.0 / obs.weight[msk])

        return dict(
            target_dim=target_dim,
            eff_pad_factor=eff_pad_factor,
            tot_var=tot_var,
            fft_dim=fft_dim,
            im_row=im_row,
            im_col=im_col,
            psf_im_row=psf_im_row,
            psf_im_col=psf_im_col,
            kim=kim,
            kpsf_im=kpsf_im,
        )


def _gauss_shape_kernels(
    dim,
    kernel_size, e1, e2,
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
    Irr, Irc, Icc = e2mom(e1, e2, fwhm_to_T(kernel_size))
    detSigma = np.sqrt(np.abs(Irr * Icc - Irc*Irc))
    fu2 = fu**2
    fv2 = fv**2
    chi2_2 = (fu2*Icc + 2*fv*fu*Irc + fv2*Irr)/2
    msk = (chi2_2 < FASTEXP_MAX_CHI2/2) & (chi2_2 >= 0)

    # from here we work with non-zero portion only
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
    knrm = detAtinv * np.pi * 2 * detSigma

    # now build the kernels
    # the flux kernel is easy since it is the kernel itself
    fkf = exp_val * knrm

    # when the kernel support extends beyong the FFT region, we need to normalize
    nrm = np.sum(fkf)/dim/dim
    fkf /= nrm
    # if not np.allclose(nrm, 1.0, atol=1e-5, rtol=0):
    #     raise FFTRangeError(
    #         "FFT size appears to be too small for gauss kernel size %f: "
    #         "norm = %f (should be 1)!" % (kernel_size, nrm)
    #     )

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
    fkx = -fkf * (2*fu*Icc + 2*fv*Irc)/2
    fkxx = fkx * (-1.0) * (2*fu*Icc + 2*fv*Irc)/2 + fkf * (-1.0) * 2.0 * Icc / 2.0

    fky = -fkf * (2*fv*Irr + 2*fu*Irc)/2
    fkyy = fky * (-1.0) * (2*fv*Irr + 2*fu*Irc)/2 + fkf * (-1.0) * 2.0 * Irr / 2.0

    fkxy = fky * (-1.0) * (2*fu*Icc + 2*fv*Irc)/2 + fkf * (-1.0) * 2.0 * Irc / 2.0

    fkr = -1 * (fkxx + fkyy)
    fkp = -1 * (fkxx - fkyy)
    fkc = -2 * fkxy

    return dict(
        fkf=fkf,
        fkr=fkr,
        fkp=fkp,
        fkc=fkc,
        msk=msk,
        nrm=nrm,
    )
