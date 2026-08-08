"""
k-space data preparation for the pre-PSF adaptive moments machinery

prep_epoch turns an Observation into an epoch dict: the image and
psf are FFT'd, the psf is deconvolved, the common smoothing is
applied, and the k grids, mode selection and noise information are
stored.  The epoch dicts are consumed by the numba kernels
(admom_ksums, admom_finalize) together with get_phase_angles, by the
fitter, and by external users such as deblenders.

choose_fwhm_smooth picks the common smoothing fwhm from the psfs of
a set of observations when it is not specified explicitly.
"""
import functools

import numpy as np
import scipy.fft as fft

from ..observation import get_mb_obs
from ..moments import fwhm_to_T, T_to_fwhm
from ..gexceptions import FFTRangeError
from ..fastexp_nb import FASTEXP_MAX_CHI2
from .. import prepsfmom
from ..prepsfmom import (
    _zero_pad_image,
    _build_square_apodization_mask,
    _deconvolve_im_psf_inplace,
    _check_obs_and_get_psf_obs,
    _pixel_fft,
)

__all__ = ['choose_fwhm_smooth', 'prep_epoch']

DEFAULT_SMOOTH_FAC = 1.05


def choose_fwhm_smooth(
    obs, fwhm_smooth=None, smooth_fac=DEFAULT_SMOOTH_FAC,
    no_psf=False, rng=None,
):
    """
    choose the common smoothing fwhm for a set of observations

    An explicit fwhm_smooth is returned as sent; otherwise the choice
    is smooth_fac times the largest psf fwhm among the observations,
    fitting the psfs with adaptive moments where no psf gmix is set.

    Parameters
    ----------
    obs: Observation, ObsList, or MultiBandObsList
        The observation(s); each must have a psf set unless
        fwhm_smooth is sent.
    fwhm_smooth: float, optional
        An explicit smoothing fwhm, returned unchanged if not None.
    smooth_fac: float, optional
        Factor applied to the largest psf fwhm, default 1.05.
    no_psf: bool, optional
        If True there are no psfs to measure, and fwhm_smooth must be
        sent.
    rng: np.random.RandomState, optional
        Used for the psf fits.

    Returns
    -------
    fwhm_smooth: float
    """
    if fwhm_smooth is not None:
        return fwhm_smooth

    if no_psf:
        raise ValueError(
            'send fwhm_smooth= (0 to disable) when no_psf=True'
        )

    from ..admom import run_admom

    if rng is None:
        rng = np.random.RandomState()

    mb_obs = get_mb_obs(obs)
    Tmax = 0.0
    for obslist in mb_obs:
        for tobs in obslist:
            if not tobs.has_psf():
                raise RuntimeError(
                    "The PSF must be set to measure a pre-PSF moment!"
                )
            psf_obs = tobs.psf
            if psf_obs.has_gmix():
                T = psf_obs.gmix.get_T()
            else:
                scale = psf_obs.jacobian.get_scale()
                T = None
                for fac in [3.0, 1.5, 6.0]:
                    Tguess = fwhm_to_T(fac * scale)
                    pres = run_admom(psf_obs, guess=Tguess, rng=rng)
                    if pres['flags'] == 0:
                        T = pres['T']
                        break
                if T is None:
                    raise RuntimeError(
                        'could not fit PSF to choose the smoothing; '
                        'send fwhm_smooth= explicitly'
                    )
            Tmax = max(Tmax, T)

    return smooth_fac * T_to_fwhm(Tmax)


def _pad_center(shape, cen_row, cen_col, target_dim):
    """the center location in the zero-padded frame (the pads of
    prepsfmom._zero_pad_image, without building the padded image)"""
    pad_row_before = (target_dim - shape[0]) // 2
    pad_col_before = (target_dim - shape[1]) // 2
    return cen_row + pad_row_before, cen_col + pad_col_before


def prep_epoch_scalars(
    obs, band=0, fwhm_smooth=0.0, pad_factor=4, no_psf=False,
):
    """
    the scalar half of prep_epoch, computable without any FFTs:
    dims, pad phase centers, weights and jacobian factors.  Used
    by prep_epoch itself and by device-prep paths that compute
    the k arrays elsewhere and only need the epoch scalars on the
    host (the array entries of the returned epoch are None).

    Returns
    -------
    epoch, aux
        epoch is the prep_epoch dict with kim/iy/ix/kv/ku/
        err_fac2/fold/ktransfer set to None; aux carries the
        intermediates the array computation needs (Tsmooth,
        psf_obs, target_dim, tot_var, eff_pad_factor).
    """
    Tsmooth = fwhm_to_T(fwhm_smooth) if fwhm_smooth > 0 else 0.0

    psf_obs = _check_obs_and_get_psf_obs(obs, no_psf)

    wmsk = obs.weight > 0
    if not np.any(wmsk):
        raise ValueError('no positive weight pixels in observation')
    tot_var = np.sum(1.0 / obs.weight[wmsk])

    if psf_obs is not None:
        max_dim = max(obs.image.shape + psf_obs.image.shape)
    else:
        max_dim = max(obs.image.shape)
    target_dim = int(max_dim * pad_factor)
    eff_pad_factor = target_dim / np.sqrt(
        obs.image.shape[0] * obs.image.shape[1]
    )
    dim = target_dim

    im_row, im_col = _pad_center(
        obs.image.shape, obs.jacobian.row0, obs.jacobian.col0,
        target_dim,
    )
    if psf_obs is not None:
        psf_row, psf_col = _pad_center(
            psf_obs.image.shape,
            psf_obs.jacobian.row0, psf_obs.jacobian.col0,
            target_dim,
        )
    else:
        psf_row = 0.0
        psf_col = 0.0

    jac = obs.jacobian
    Atinv = np.linalg.inv(
        [[jac.dvdrow, jac.dvdcol], [jac.dudrow, jac.dudcol]]
    ).T
    detAtinv = np.abs(np.linalg.det(Atinv))

    epoch = {
        'band': band,
        'kim': None,
        'iy': None,
        'ix': None,
        'dim': dim,
        'kv': None,
        'ku': None,
        'Atinv': Atinv,
        'drow': im_row - psf_row,
        'dcol': im_col - psf_col,
        'detAtinv': detAtinv,
        'df2': 1.0 / dim ** 2,
        'err_fac2': None,
        'fold': None,
        'ktransfer': None,
        'weight': 1.0 / (tot_var * eff_pad_factor ** 2),
    }
    aux = {
        'Tsmooth': Tsmooth,
        'psf_obs': psf_obs,
        'target_dim': target_dim,
        'tot_var': tot_var,
        'eff_pad_factor': eff_pad_factor,
    }
    return epoch, aux


def prep_epoch(
    obs, band=0, fwhm_smooth=0.0, pad_factor=4, ap_rad=1.5,
    use_noise_image=False, no_psf=False, store_transfer=False,
):
    """
    prepare the k-space data for one observation: FFT the image and
    psf, deconvolve the psf, apply the smoothing, and store the k
    grids and noise information

    Parameters
    ----------
    obs: Observation
        The observation; must have a psf set unless no_psf=True.
        Non-square images are zero padded to square internally.
    band: int, optional
        The band index stored in the epoch, default 0.
    fwhm_smooth: float, optional
        The fwhm of the common round gaussian smoothing (see
        choose_fwhm_smooth); 0 disables smoothing.  Default 0.
    pad_factor: int, optional
        The factor by which to pad the FFTs, default 4.
    ap_rad: float, optional
        The apodization radius for the stamp in pixels, default 1.5.
    use_noise_image: bool, optional
        If True, measure the per mode noise power from the noise
        image attached to the observation (obs.noise); see
        PAdmomFitter.  Default False.
    no_psf: bool, optional
        If True, allow an observation without a psf; only the pixel
        window function is deconvolved.  Default False.
    store_transfer: bool, optional
        If True, store the linear image->kim transfer at the
        retained modes ('ktransfer'), needed by the full_errors
        influence-kernel machinery; costs ~25 percent of the prep
        time and one complex array per epoch, so it is opt-in.
        Requires ap_rad=0 (ktransfer is None otherwise).
        Default False.

    Returns
    -------
    epoch: dict with entries
        band: int
            the band index
        kim: complex array
            the deconvolved, smoothed rfft values at the retained
            modes, folded with the conjugate symmetry weights
        iy, ix: int arrays
            the retained mode indices on the rfft half plane
        dim: int
            the padded fft dimension
        kv, ku: arrays
            the sky coordinate frequencies at the retained modes
        Atinv: (2, 2) array
            the inverse transpose jacobian matrix, mapping sky
            offsets to pixel offsets
        detAtinv: float
            its determinant, the k-space area factor
        drow, dcol: float
            the phase offsets of the image relative to the psf
            jacobian centers in the padded frames
        df2: float
            1/dim^2, the k-space measure of the sums
        err_fac2: array
            the per mode noise power times |smooth/kpsf|^2, for the
            noise propagation
        fold: array
            the smoothing profile folded with the conjugate
            symmetry weights at the retained modes, the same factor
            already applied to kim; model templates built against
            kim must carry it too
        weight: float
            the epoch combination weight, from the weight map
    Callers may attach additional entries; the fitting machinery
    reads only the ones above.
    """
    # the scalar half (dims, pad centers, weights) is shared with
    # the device-prep path; the array computations below fill in
    # the rest
    epoch, aux = prep_epoch_scalars(
        obs, band=band, fwhm_smooth=fwhm_smooth,
        pad_factor=pad_factor, no_psf=no_psf,
    )
    Tsmooth = aux['Tsmooth']
    psf_obs = aux['psf_obs']
    target_dim = aux['target_dim']
    tot_var = aux['tot_var']
    # the square of this is the ratio of padded to unpadded pixel
    # counts, used to scale the noise
    eff_pad_factor = aux['eff_pad_factor']

    # the image is real, so we work with the rfft half plane;
    # conjugate modes are folded in with the symmetry weights
    kim, im_row, im_col = _zero_pad_and_compute_rfft(
        obs.image, obs.jacobian.row0, obs.jacobian.col0, target_dim,
        ap_rad,
    )
    dim = kim.shape[0]

    if psf_obs is not None:
        kpsf, psf_row, psf_col = _zero_pad_and_compute_rfft(
            psf_obs.image,
            psf_obs.jacobian.row0, psf_obs.jacobian.col0,
            target_dim,
            0,  # we do not apodize PSF stamps
        )
    else:
        kpsf = _pixel_fft(dim)[:, :dim // 2 + 1]
        psf_row = 0.0
        psf_col = 0.0

    max_amp = np.abs(kpsf[0, 0])

    # the k grids, mode selection and smoothing profile are shared
    # between fits with the same stamp geometry, so they are cached
    jac = obs.jacobian
    grids = _get_kspace_grids(
        dim, jac.dvdrow, jac.dvdcol, jac.dudrow, jac.dudcol,
        float(Tsmooth),
    )

    # extract the masked modes
    kim = kim[grids['iy'], grids['ix']]
    kpsf = kpsf[grids['iy'], grids['ix']]

    kim, kpsf, _ = _deconvolve_im_psf_inplace(kim, kpsf, max_amp)

    # fold holds the smoothing profile times the conjugate symmetry
    # weights; fold2 has the squared smoothing for the noise, where
    # the symmetry weight enters once
    kim *= grids['fold']

    # the noise power per mode, white from the weight map or
    # measured from the attached noise realization.  In both cases
    # the eff_pad_factor boost accounts for treating the modes of
    # the zero padded image as independent.  The noise is not
    # apodized: the taper is part of the padding window, whose
    # effect on the noise in the moments vanishes for kernels
    # supported in the stamp interior, and the boost corresponds to
    # the plain zero pad window
    if use_noise_image:
        if not obs.has_noise():
            raise ValueError(
                'obs.noise must be set when use_noise_image=True'
            )
        knoise, _, _ = _zero_pad_and_compute_rfft(
            obs.noise, obs.jacobian.row0, obs.jacobian.col0,
            target_dim, 0,
        )
        pnoise = np.abs(knoise[grids['iy'], grids['ix']]) ** 2
        pnoise *= eff_pad_factor ** 2
    else:
        pnoise = tot_var * eff_pad_factor ** 2

    # factor for noise propagation: the effective kernels act on
    # the raw image fft, so include the noise power, the smoothing
    # and the deconvolution
    err_fac2 = grids['fold2'] * pnoise / np.abs(kpsf) ** 2

    # the linear image->kim transfer at the retained modes: the
    # coefficient of image pixel (0, 0), with other pixels
    # differing by the mode phases.  Used by the influence-kernel
    # error propagation (full_errors).  Only exact without
    # apodization, where the map is diagonal in k up to the
    # placement phase
    if ap_rad > 0 or not store_transfer:
        ktransfer = None
    else:
        f1d = fft.fftfreq(dim) * (2.0 * np.pi)
        r00 = im_row - obs.jacobian.row0
        c00 = im_col - obs.jacobian.col0
        ktransfer = grids['fold'] * np.exp(
            -1j * (
                f1d[grids['iy']] * r00 + f1d[grids['ix']] * c00
            )
        ) / kpsf

    return {
        'band': band,
        'kim': kim,
        'iy': grids['iy'],
        'ix': grids['ix'],
        'dim': dim,
        'kv': grids['kv'],
        'ku': grids['ku'],
        'Atinv': grids['Atinv'],
        # the phase shift putting the effective center at the image
        # jacobian center; the psf centering phase cancels in the
        # deconvolution except for this difference
        'drow': im_row - psf_row,
        'dcol': im_col - psf_col,
        'detAtinv': grids['detAtinv'],
        'df2': 1.0 / dim ** 2,
        'err_fac2': err_fac2,
        'fold': grids['fold'],
        'ktransfer': ktransfer,
        # the relative epoch weights always come from the weight
        # maps, even when the noise power comes from a noise image
        'weight': 1.0 / (tot_var * eff_pad_factor ** 2),
    }


@functools.lru_cache(maxsize=16)
def _get_kspace_grids(dim, dvdrow, dvdcol, dudrow, dudcol, Tsmooth):
    """
    get the k grids in sky coordinates over the rfft half plane, the
    selection of modes where the smoothing kernel is significant, and
    the smoothing profile folded with the conjugate symmetry weights.
    These only depend on the stamp geometry and smoothing, so they are
    cached and shared between fits; the returned arrays are marked
    read-only.
    """
    half = dim // 2 + 1
    f1d = fft.fftfreq(dim) * (2.0 * np.pi)
    fx = f1d[:half].reshape(1, -1)
    fy = f1d.reshape(-1, 1)
    Atinv = np.linalg.inv(
        [[dvdrow, dvdcol], [dudrow, dudcol]]
    ).T
    kv = Atinv[0, 0] * fy + Atinv[0, 1] * fx
    ku = Atinv[1, 0] * fy + Atinv[1, 1] * fx
    detAtinv = np.abs(np.linalg.det(Atinv))

    kmag2 = kv * kv + ku * ku

    # we only keep modes where the smoothing kernel is significant
    if Tsmooth > 0:
        chi2_2 = 0.25 * Tsmooth * kmag2
        msk = chi2_2 < FASTEXP_MAX_CHI2 / 2
    else:
        msk = np.ones((dim, half), dtype=bool)

    iy, ix = np.where(msk)

    kmag2 = kmag2[msk]
    kv = kv[msk]
    ku = ku[msk]

    # conjugate symmetry weights: modes with 0 < kx < nyquist appear
    # twice in the full plane
    wsym = np.ones(ix.size)
    if dim % 2 == 0:
        wsym[(ix > 0) & (ix < dim // 2)] = 2.0
    else:
        wsym[ix > 0] = 2.0

    if Tsmooth > 0:
        smooth = np.exp(-0.25 * Tsmooth * kmag2)

        # check that the smoothing kernel is contained in the FFT
        # region; the converged weight is at least this large so this
        # covers the worst case.  the tolerance is loose since small
        # truncations only produce correspondingly small biases in the
        # moments
        nrm = np.sum(wsym * smooth) * detAtinv * np.pi * Tsmooth / dim**2
        if not np.allclose(nrm, 1.0, atol=1e-3, rtol=0):
            raise FFTRangeError(
                'FFT size appears too small for smoothing fwhm %g: '
                'norm = %f (should be 1)' % (T_to_fwhm(Tsmooth), nrm)
            )

        fold = wsym * smooth
        fold2 = wsym * smooth * smooth
    else:
        fold = wsym
        fold2 = wsym

    grids = {
        'kv': kv,
        'ku': ku,
        'iy': iy,
        'ix': ix,
        'fold': fold,
        'fold2': fold2,
        'Atinv': Atinv,
        'detAtinv': detAtinv,
    }
    for key in ['kv', 'ku', 'iy', 'ix', 'fold', 'fold2', 'Atinv']:
        grids[key].flags.writeable = False
    return grids


def _zero_pad_and_compute_rfft_impl(im, cen_row, cen_col, target_dim, ap_rad):
    """
    zero pad and compute the real fft, returning the fft and the center
    location in the padded image
    """
    if ap_rad > 0:
        ap_mask = np.ones_like(im)
        _build_square_apodization_mask(ap_rad, ap_mask)
        im = im * ap_mask

    pim, pad_row_before, pad_col_before = _zero_pad_image(im, target_dim)
    pad_cen_row = cen_row + pad_row_before
    pad_cen_col = cen_col + pad_col_before
    kpim = fft.rfftn(pim)
    return kpim, pad_cen_row, pad_cen_col


@functools.lru_cache(maxsize=128)
def _zero_pad_and_compute_rfft_cached(
    im_tuple, cen_row, cen_col, target_dim, ap_rad
):
    return _zero_pad_and_compute_rfft_impl(
        np.array(im_tuple), cen_row, cen_col, target_dim, ap_rad
    )


@functools.wraps(_zero_pad_and_compute_rfft_impl)
def _zero_pad_and_compute_rfft(im, cen_row, cen_col, target_dim, ap_rad):
    # respect the fft caching switch in prepsfmom
    if prepsfmom.USE_FFT_CACHE:
        return _zero_pad_and_compute_rfft_cached(
            tuple(tuple(ii) for ii in im),
            float(cen_row), float(cen_col), int(target_dim), float(ap_rad),
        )
    else:
        return _zero_pad_and_compute_rfft_impl(
            im, cen_row, cen_col, target_dim, ap_rad,
        )
