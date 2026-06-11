import numpy as np
import logging
from ..gexceptions import GMixRangeError, BootPSFFailure
from .. import moments

logger = logging.getLogger(__name__)


def get_fitgauss_target_psf(psfobs, rng, psf_flux=None):
    """
    do the gaussian fit.

    try the following in order
        - adaptive moments
        - maximim likelihood
        - see if there is already a gmix object


    if the above all fail, rase BootPSFFailure
    """
    import galsim

    from ..admom import AdmomFitter
    from ..guessers import GMixPSFGuesser, SimplePSFGuesser
    from ..runners import run_psf_fitter
    from ..fitting import Fitter

    ntry = 4
    guesser = GMixPSFGuesser(rng=rng, ngauss=1)

    # try adaptive moments first
    fitter = AdmomFitter(rng=rng)

    res = run_psf_fitter(obs=psfobs, fitter=fitter, guesser=guesser, ntry=ntry)

    if res['flags'] == 0:
        e1, e2 = res['e']
        T = res['T']
    else:
        # try maximum likelihood

        lm_pars = {
            'maxfev': 2000,
            'ftol': 1.0e-05,
            'xtol': 1.0e-05,
        }

        fitter = Fitter(model='gauss', fit_pars=lm_pars)
        guesser = SimplePSFGuesser(rng=rng)

        res = run_psf_fitter(
            obs=psfobs,
            fitter=fitter,
            guesser=guesser,
            ntry=ntry,
            set_result=False,
        )

        if res['flags'] == 0:
            psf_gmix = res.get_gmix()
        else:
            # see if there was already a gmix that we might use instead
            if psfobs.has_gmix() and len(psfobs.gmix) == 1:
                psf_gmix = psfobs.gmix.copy()
            else:
                # ok, just raise and exception
                raise BootPSFFailure(
                    'failed to fit psf for MetacalFitGaussPSF'
                )
        try:
            e1, e2, T = psf_gmix.get_e1e2T()
        except GMixRangeError as err:
            logger.info('%s', err)
            raise BootPSFFailure(
                'could not get e1,e2 from psf fit for MetacalFitGaussPSF'
            )

    dilation = _get_ellip_dilation(e1, e2, T)
    T_dilated = T * dilation
    sigma = np.sqrt(T_dilated / 2.0)

    if psf_flux is None:
        psf_flux = psfobs.image.sum()

    return galsim.Gaussian(
        sigma=sigma,
        flux=psf_flux,
    )


def _get_ellip_dilation(e1, e2, T):
    """
    when making a new image after shearing, we need to dilate the PSF to hide
    modes that get exposed
    """
    irr, irc, icc = moments.e2mom(e1, e2, T)

    mat = np.zeros((2, 2))
    mat[0, 0] = irr
    mat[0, 1] = irc
    mat[1, 0] = irc
    mat[1, 1] = icc

    eigs = np.linalg.eigvals(mat)

    dilation = eigs.max() / (T / 2.0)
    dilation = np.sqrt(dilation)

    dilation = 1.0 + 2 * (dilation - 1.0)

    if dilation > 1.1:
        dilation = 1.1

    return dilation
