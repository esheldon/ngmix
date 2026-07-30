"""
Sandwich covariance for LM fits under stationary correlated noise,
using the per-mode noise power measured from noise images attached
to the observations, in the same spirit as use_noise_image for the
pre-psf moment fitters.

The LM machinery reports pars_cov0 * chi2/dof, which only adapts to
the overall pixel variance.  Under stationary correlated noise the
covariance of the weighted least squares estimator is the sandwich

    Cov = A^-1 B A^-1

where A^-1 = pars_cov0 is the curvature at the solution (including
any prior curvature, which shapes the estimator's response) and

    B_ab = sum_epochs sum_q conj(G_a) G_b |n~(q)|^2 / N^2

with G_a = fft2(weight * dmodel/dp_a) the influence kernel of
parameter a and n~ the fft of the epoch's noise image.  For white
noise at the weight-map level B reduces to the data block of A and
the sandwich returns pars_cov0.

The noise image attached to each observation must be an independent
realization of the noise, in the same frame as the image.
"""
__all__ = ['calc_noise_cov', 'apply_noise_cov']

import numpy as np

from .. import gmix
from ..gexceptions import GMixRangeError
from .leastsqbound import _test_cov, _get_def_stuff

# absolute floors for the numerical derivative steps; the
# centroid pars are in sky units, T in arcsec^2, flux in
# image units
STEP_CEN = 1.0e-3
STEP_SHAPE = 1.0e-4
STEP_STRUCT_MIN = 1.0e-4
STEP_FLUX_MIN = 1.0e-6
STEP_FRAC = 1.0e-3


def apply_noise_cov(fit_model, result):
    """
    Replace the chi^2-scaled LM covariance in the result with the
    noise-power sandwich covariance.  No-op unless the fit
    succeeded and the curvature is usable.  On a bad sandwich
    covariance the covariance test flags are set and the errors
    are set to the defaults, as in run_leastsq.

    Parameters
    ----------
    fit_model: FitModel
        The fit model holding the observations, model and band
        parameter mapping
    result: dict
        The result dict from run_leastsq, modified in place
    """
    if result['flags'] != 0:
        return
    pcov0 = result.get('pars_cov0')
    if pcov0 is None or not np.all(np.isfinite(pcov0)):
        return

    npars = result['pars'].size
    try:
        cov = calc_noise_cov(
            fit_model=fit_model, pars=result['pars'],
            pars_cov0=pcov0,
        )
    except GMixRangeError:
        # the central-difference step left the valid gmix domain:
        # the solution sits at a parameter boundary (T near the
        # domain edge, |g| near 1), where the sandwich is not
        # evaluable
        cov = np.full((npars, npars), np.nan)

    if not np.all(np.isfinite(cov)):
        cflags = _test_cov(np.diag(np.full(npars, -1.0)))
    else:
        cflags = _test_cov(cov)

    if cflags != 0:
        result['flags'] |= cflags
        result['errmsg'] = 'bad noise covariance matrix'
        _, result['pars_cov'], result['pars_err'] = (
            _get_def_stuff(npars)
        )
    else:
        result['pars_cov'] = cov
        result['pars_err'] = np.sqrt(np.diag(cov))


def calc_noise_cov(fit_model, pars, pars_cov0):
    """
    Calculate the sandwich covariance pars_cov0 B pars_cov0 with B
    accumulated from the per-mode noise power of every epoch's
    attached noise image.

    Parameters
    ----------
    fit_model: FitModel
        The fit model holding the observations, model and band
        parameter mapping
    pars: array
        Parameters at the solution
    pars_cov0: array
        The unscaled covariance (J^T J)^-1 from the fit

    Returns
    -------
    cov: (npars, npars) array
    """
    npars = pars.size
    nband = fit_model.nband
    nshape = npars - nband

    B = np.zeros((npars, npars))
    for band in range(nband):
        kpars = list(range(nshape)) + [nshape + band]
        for obs in fit_model.obs[band]:
            dimages = _dmodel_images(
                fit_model=fit_model, pars=pars, band=band,
                obs=obs, kpars=kpars,
            )
            kernels = [
                np.fft.fft2(obs.weight * dim)
                for dim in dimages
            ]
            p = np.abs(np.fft.fft2(obs.noise)) ** 2
            n = obs.image.size
            for ia in range(len(kpars)):
                for ib in range(ia, len(kpars)):
                    val = np.sum(
                        np.conj(kernels[ia]) * kernels[ib] * p
                    ).real / n ** 2
                    B[kpars[ia], kpars[ib]] += val
                    if ib != ia:
                        B[kpars[ib], kpars[ia]] += val

    return pars_cov0 @ B @ pars_cov0


# the simple models whose band parameters are
# [cen1, cen2, g1, g2, T, flux] with a shared shape across the
# fixed mixture components, for which the derivative images are
# analytic; other models use the central differences
_ANALYTIC_MODELS = ('gauss', 'exp', 'dev')


def _dmodel_images(fit_model, pars, band, obs, kpars,
                   force_fd=False):
    """the derivative images of the convolved model for the
    given parameter indices: a single analytic pass for the
    simple models (every derivative of a rendered gaussian
    shares the render's exponential, replacing the twelve
    renders of the stepped evaluation), central differences
    otherwise"""
    if force_fd \
            or fit_model.model_name not in _ANALYTIC_MODELS:
        return [
            _dmodel(
                fit_model=fit_model, pars=pars, ipar=a,
                band=band, obs=obs,
            )
            for a in kpars
        ]

    from .noise_cov_nb import deriv_images

    band_pars = fit_model.get_band_pars(pars=pars, band=band)
    g1, g2, T, flux = band_pars[2:6]

    gm0 = gmix.make_gmix_model(band_pars, fit_model.model)
    if obs.has_psf_gmix():
        gmc = gm0.convolve(obs.psf.gmix)
        npsf = len(obs.psf.gmix)
    else:
        gmc = gm0
        npsf = 1
    gpars = gmc.get_full_pars().reshape(-1, 6)
    # the model-component covariances, aligned with the
    # model-major composed ordering of convolve
    modcov = np.repeat(
        gm0.get_full_pars().reshape(-1, 6)[:, 3:6], npsf,
        axis=0,
    )

    # d(e1, e2)/d(g1, g2) for the distortion e = 2 g / (1 + g^2)
    gsq = g1 * g1 + g2 * g2
    f = 2.0 / (1.0 + gsq)
    dfac = -f / (1.0 + gsq)
    de1dg1 = f + 2.0 * g1 * g1 * dfac
    de1dg2 = 2.0 * g1 * g2 * dfac
    de2dg1 = de1dg2
    de2dg2 = f + 2.0 * g2 * g2 * dfac

    # dSigma_k/dg_i = (T_k / 2) [[-de1, de2], [de2, de1]]/dg_i
    # and dSigma_k/dT = Sigma_k / T, from
    # Sigma_k = (T_k / 2) [[1 - e1, e2], [e2, 1 + e1]]
    Tk = modcov[:, 0] + modcov[:, 2]
    dcov = np.zeros((gpars.shape[0], 3, 3))
    for i, (de1, de2) in enumerate(
        ((de1dg1, de2dg1), (de1dg2, de2dg2)),
    ):
        dcov[:, i, 0] = -0.5 * Tk * de1
        dcov[:, i, 1] = 0.5 * Tk * de2
        dcov[:, i, 2] = 0.5 * Tk * de1
    dcov[:, 2, :] = modcov / T

    dims = obs.image.shape
    rows, cols = np.mgrid[0:dims[0], 0:dims[1]]
    vv, uu = obs.jacobian.get_vu(
        row=rows.ravel().astype('f8'),
        col=cols.ravel().astype('f8'),
    )
    out = np.zeros((6, vv.size))
    deriv_images(
        gpars, dcov, vv, uu, obs.jacobian.area, out,
    )

    # out rows: value, cen1, cen2, g1, g2, T; the flux
    # derivative is the value image over the flux (the model is
    # linear in flux).  kpars is the shape parameters followed
    # by this band's flux
    images = [
        out[k].reshape(dims) for k in (1, 2, 3, 4, 5)
    ]
    if flux != 0.0:
        images.append(out[0].reshape(dims) / flux)
    else:
        images.append(_dmodel(
            fit_model=fit_model, pars=pars, ipar=kpars[-1],
            band=band, obs=obs,
        ))
    return images


def _dmodel(fit_model, pars, ipar, band, obs):
    """central difference derivative image of the convolved model
    with respect to one parameter"""
    step = _get_step(pars=pars, ipar=ipar, nband=fit_model.nband)

    ims = []
    for sign in (1, -1):
        p = pars.copy()
        p[ipar] += sign * step
        band_pars = fit_model.get_band_pars(pars=p, band=band)
        gm = gmix.make_gmix_model(band_pars, fit_model.model)
        if obs.has_psf_gmix():
            gm = gm.convolve(obs.psf.gmix)
        # the same fast exp and clip as the fdiff renders,
        # consistent with the fit's objective
        ims.append(gm.make_image(
            obs.image.shape, jacobian=obs.jacobian,
            fast_exp=True,
        ))
    return (ims[0] - ims[1]) / (2 * step)


def _get_step(pars, ipar, nband):
    npars = pars.size
    nshape = npars - nband
    if ipar < 2:
        return STEP_CEN
    elif ipar < 4:
        return STEP_SHAPE
    elif ipar < nshape:
        return max(STEP_STRUCT_MIN, STEP_FRAC * abs(pars[ipar]))
    else:
        return max(STEP_FLUX_MIN, STEP_FRAC * abs(pars[ipar]))
