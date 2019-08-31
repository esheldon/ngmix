import numpy as np
import logging
import ngmix
from .observation import get_mb_obs
from .shape import e1e2_to_g1g2
from .gexceptions import GMixRangeError

logger = logging.getLogger(__name__)


class GaussMom(object):
    """
    measure gaussian weighted moments

    parameters
    ----------
    obs: Observation, ObsList or MultiBandObsList
        The observations to fit. Note that if an ObsList or a MultiBandObsList
        is passed, the observations are coadded assuming perfect registration.
    fwhm: float
        The FWHM of the Gaussian weight function.
    rng: np.random.RandomState, optional
        If not None, the RNG to use. Otherwise a new RNG will be made.
    """
    def __init__(self, obs, fwhm, rng=None):
        self.rng = rng if rng is not None else np.random.RandomState()
        self.fwhm = fwhm
        self.mbobs = get_mb_obs(obs)
        self._set_mompars()

    def go(self):
        """
        run moments measurements on all objects
        """
        obs = self._do_coadd_maybe(self.mbobs)
        res = self._measure_moments(obs)

        if res['flags'] != 0:
            logger.debug("        moments failed: %s" % res['flagstr'])

        self.result = res

    def get_result(self):
        """
        get the result
        """

        if not hasattr(self, 'result'):
            raise RuntimeError("run go() first")

        return self.result

    def _do_coadd_maybe(self, mbobs):
        """
        coadd all images and psfs.  Assume perfect registration and
        same wcs
        """

        # note here assuming we can re-use the wcs etc.
        new_obs = mbobs[0][0].copy()

        if len(mbobs) == 1 and len(mbobs[0]) == 1:
            return new_obs

        first = True
        wsum = 0.0
        for obslist in mbobs:
            for obs in obslist:
                tim = obs.image
                twt = obs.weight

                medweight = np.median(twt)
                noise = np.sqrt(1.0/medweight)

                tnim = self.rng.normal(size=tim.shape, scale=noise)

                wsum += medweight

                if first:
                    im = tim*medweight
                    nim = tnim * medweight

                    first = False
                else:
                    im += tim*medweight
                    nim += tnim * medweight

        fac = 1.0/wsum
        im *= fac

        nim *= fac

        noise_var = nim.var()

        wt = np.zeros(im.shape) + 1.0/noise_var

        new_obs.set_image(im, update_pixels=False)
        new_obs.set_weight(wt)

        return new_obs

    def _measure_moments(self, obs):
        """
        measure weighted moments
        """

        res = self.weight.get_weighted_moments(obs=obs, maxrad=1.e9)

        if res['flags'] != 0:
            return res

        res['numiter'] = 1
        try:
            res['g'] = np.array(e1e2_to_g1g2(res['e'][0], res['e'][1]))
        except GMixRangeError:
            res['g'] = np.array([-9999, -9999])
            # we don't have a valid shape, set flags
            res['flags'] = 0x80
            res['flagstr'] = (
                'e is out of range for a proper ellipticity - '
                'could not convert to a shear!')

        return res

    def _set_mompars(self):
        T = ngmix.moments.fwhm_to_T(self.fwhm)

        # the weight is always centered at 0, 0 or the
        # center of the coordinate system as defined
        # by the jacobian

        weight = ngmix.GMixModel(
            [0.0, 0.0, 0.0, 0.0, T, 1.0],
            'gauss',
        )

        # make the max of the weight 1.0 to get better
        # fluxes

        weight.set_norms()
        norm = weight.get_data()['norm'][0]
        weight.set_flux(1.0/norm)

        self.weight = weight
