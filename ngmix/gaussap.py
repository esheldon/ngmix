"""
calculate gaussian aperture fluxes for a catalog of parameters
"""
import logging
import numpy as np
from .gmix import GMixModel, GMixCM
from .gexceptions import GMixRangeError

DEFAULT_FLUX = -9999.0
NO_ATTEMPT = 2**0
RANGE_ERROR = 2**1

logger = logging.getLogger(__name__)


def get_gaussap_flux(pars,
                     model,
                     weight_fwhm,
                     fracdev=None,
                     TdByTe=None,
                     mask=None,
                     verbose=True):
    """
    Measure synthesized gaussian weighted apertures for a ngmix
    models

    parameters
    ----------
    pars: array
        Shape [nobj, 6]
    model: string
        e.g. exp,dev,gauss,cm
    weight_fwhm: float
        FWHM of the weight function in the same units as the
        pixel scale.
    fracdev: array
        Send for model 'cm'
    TdByTe: array
        Send for model 'cm'
    verbose: bool
        If True, print otu progress
    """

    fracdev, TdByTe, pars, mask = _prepare(
        pars,
        model,
        weight_fwhm,
        fracdev=fracdev,
        TdByTe=TdByTe,
        mask=mask,
    )

    nband = _get_nband(pars, model)

    flags = np.zeros((pars.shape[0], nband), dtype='i4')
    gap_flux = np.zeros((pars.shape[0], nband))
    gap_flux[:, :] = DEFAULT_FLUX

    nobj = pars.shape[0]
    for i in range(nobj):

        if verbose and ((i+1) % 1000) == 0:
            logger.info("%d/%d" % (i+1, nobj))
        else:
            logger.debug("%d/%d" % (i+1, nobj))

        if not mask[i]:
            flags[i] = NO_ATTEMPT
            continue

        for band in range(nband):
            tflux, tflags = _do_gap(
                weight_fwhm,
                fracdev,
                TdByTe,
                pars,
                model,
                i,
                band,
            )
            gap_flux[i, band] = tflux
            flags[i, band] = tflags

    return gap_flux, flags


def _do_gap(weight_fwhm, fracdev, TdByTe, pars, model, i, band):

    flux = DEFAULT_FLUX
    flags = RANGE_ERROR

    try:

        tpars = _get_band_pars(pars, model, i, band)

        if model == 'cm':
            gm = GMixCM(
                fracdev[i],
                TdByTe[i],
                tpars,
            )
        else:
            gm = GMixModel(tpars, model)

        flux = gm.get_gaussap_flux(fwhm=weight_fwhm)
        flags = 0

    except GMixRangeError as err:
        logger.debug(str(err))

    return flux, flags


def _prepare(pars,
             model,
             weight_fwhm,
             fracdev=None,
             TdByTe=None,
             mask=None):

    pars = np.array(pars, dtype='f8', ndmin=2, copy=False)

    if mask is not None:
        mask = np.array(mask, dtype=np.bool_, ndmin=1, copy=False)
        assert mask.shape[0] == pars.shape[0], \
            'mask and pars must be same length'
    else:
        mask = np.ones(pars.shape[0], dtype=np.bool_)

    if len(pars.shape) == 1:
        oldpars = pars
        pars = np.zeros((1, pars.shape[0]), dtype='f8')
        pars[0, :] = oldpars

    if model == 'cm':

        fracdev = np.array(fracdev, dtype='f8', ndmin=1, copy=False)
        TdByTe = np.array(TdByTe, dtype='f8', ndmin=1, copy=False)
        assert fracdev.size == pars.shape[0], 'fracdev/pars must be same size'
        assert TdByTe.size == pars.shape[0], 'TdByTe/pars must be same length'

    return fracdev, TdByTe, pars, mask


def _get_band_pars(pars, model, index, band):

    npars = _get_band_npars(model)

    flux_start = npars-1

    tpars = np.zeros(npars)
    tpars[0:npars-1] = pars[index, 0:npars-1]
    tpars[-1] = pars[index, flux_start+band]

    tpars[4] = tpars[4].clip(min=0.0001)

    return tpars


def _get_nband(pars, model):
    if model == 'bdf':
        nband = len(pars[0])-7+1
    else:
        nband = len(pars[0])-6+1

    return nband


def _get_band_npars(model):
    if model == 'bdf':
        nband = 7
    else:
        nband = 6

    return nband
