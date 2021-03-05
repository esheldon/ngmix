from .metacal import get_all_metacal
from .bootstrap import bootstrap


class MetacalBootstrapper(object):
    """
    Make metacal sheared images and run a fitter/measurment, possibly
    bootstrapping the fit based on information inferred from the data or the
    psf model

    Parameters
    ----------
    runner: fit runner for object
        Must have go(obs=obs) method
    psf_runner: fit runner for psfs
        Must have go(obs=obs, set_result=) method
    ignore_failed_psf: bool, optional
        If set to True, remove observations where the psf fit fails, and
        only fit the remaining.  Default True.
    rng: numpy.random.RandomState
        Random state for generating noise fields.  Not needed if metacal if
        using the noise field in the observations
    **metacal_kws:  keywords
        Keywords to send to get_all_metacal
    """
    def __init__(self, runner, psf_runner, ignore_failed_psf=True,
                 rng=None,
                 **metacal_kws):
        self.runner = runner
        self.psf_runner = psf_runner
        self.ignore_failed_psf = ignore_failed_psf
        self.metacal_kws = metacal_kws
        self.rng = rng

    def go(self, obs):
        """
        Run the runners on the input observation(s)

        Parameters
        ----------
        obs: ngmix Observation(s)
            Observation, ObsList, or MultiBandObsList
        """
        self._resdict = metacal_bootstrap(
            obs=obs,
            runner=self.runner,
            psf_runner=self.psf_runner,
            ignore_failed_psf=self.ignore_failed_psf,
            rng=self.rng,
            **self.metacal_kws
        )

    @property
    def fitter(self):
        """
        get a reference to the fitter
        """
        return self.runner.fitter

    def get_result(self):
        """
        get the result dict for the last fit
        """
        if not hasattr(self, '_resdict'):
            raise RuntimeError('run go first')

        return self._resdict

    @property
    def result(self):
        """
        get the result dict for the last fit
        """
        return self.get_result()


def metacal_bootstrap(
    obs,
    runner,
    psf_runner=None,
    ignore_failed_psf=True,
    rng=None,
    **metacal_kws
):
    """
    Make metacal sheared images and run a fitter/measurment, possibly
    bootstrapping the fit based on information inferred from the data or the
    psf model

    Parameters
    ----------
    obs: ngmix Observation(s)
        Observation, ObsList, or MultiBandObsList
    runner: ngmix Runner
        Must have go(obs=obs) method
    psf_runner: ngmix PSFRunner, optional
        Must have go(obs=obs, set_result=) method
    ignore_failed_psf: bool, optional
        If set to True, remove observations where the psf fit fails, and
        only fit the remaining.  Default True.
    rng: numpy.random.RandomState
        Random state for generating noise fields.  Not needed if metacal if
        using the noise field in the observations
    **metacal_kws:  keywords
        Keywords to send to get_all_metacal

    Side effects
    ------------
    the obs.psf.meta['result'] and the obs.psf.gmix may be set if a psf runner
    is sent and the internal fitter has a get_gmix method.  gmix are only set
    for successful fits
    """

    obsdict = get_all_metacal(obs=obs, rng=rng, **metacal_kws)

    resdict = {}

    for key, tobs in obsdict.items():
        bootstrap(
            obs=tobs, runner=runner, ignore_failed_psf=ignore_failed_psf,
        )
        resdict[key] = runner.get_result()

    return resdict
