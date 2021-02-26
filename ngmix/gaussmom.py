import logging
import ngmix

logger = logging.getLogger(__name__)


class GaussMom(object):
    """
    measure gaussian weighted moments

    parameters
    ----------
    fwhm: float
        The FWHM of the Gaussian weight function.
    """
    def __init__(self, fwhm):
        self.fwhm = fwhm
        self._set_mompars()

    def go(self, obs):
        """
        run moments measurements on all objects

        Parameters
        ----------
        obs: Observation, ObsList or MultiBandObsList
            The observations to fit. Note that if an ObsList or a MultiBandObsList
            is passed, the observations are coadded assuming perfect registration.
        """
        res = self._measure_moments(obs=obs)

        if res['flags'] != 0:
            logger.debug("        moments failed: %s" % res['flagstr'])

        self._result = res

    def get_result(self):
        """
        get the result
        """

        if not hasattr(self, '_result'):
            raise RuntimeError("run go() first")

        return self._result

    def _measure_moments(self, obs):
        """
        measure weighted moments
        """

        res = self.weight.get_weighted_moments(obs=obs, maxrad=1.e9)

        if res['flags'] != 0:
            return res

        res['numiter'] = 1
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
