"""
TODO

    - make a tester for it
    - test it in nsim
    - make it possible to specify the guess type (not just psf)

"""
import logging

from .observation import Observation, ObsList, MultiBandObsList
from .gexceptions import BootPSFFailure

BOOT_S2N_LOW = 2 ** 0
BOOT_R2_LOW = 2 ** 1
BOOT_R4_LOW = 2 ** 2
BOOT_TS2N_ROUND_FAIL = 2 ** 3
BOOT_ROUND_CONVOLVE_FAIL = 2 ** 4
BOOT_WEIGHTS_LOW = 2 ** 5

logger = logging.getLogger(__name__)


class Bootstrapper(object):
    """
    bootstrap fits to psf and object

    Parameters
    ----------
    runner: fit runner for object
        Must have go(obs=obs) method
    psf_runner: fit runner for psfs
        Must have go(obs=obs) method
    ignore_failed_psf: bool, optional
        If set to True, remove observations where the psf fit fails, and
        only fit the remaining.  Default True.
    """
    def __init__(self, runner, psf_runner=None, ignore_failed_psf=True):
        self.runner = runner
        self.psf_runner = psf_runner
        self.ignore_failed_psf = ignore_failed_psf

    def go(self, obs):
        """
        Run the runners on the input observation(s)

        Parameters
        ----------
        obs: ngmix Observation(s)
            Observation, ObsList, or MultiBandObsList
        """
        return bootstrap(
            obs=obs,
            runner=self.runner,
            psf_runner=self.psf_runner,
            ignore_failed_psf=self.ignore_failed_psf,
        )

    @property
    def fitter(self):
        """
        get a reference to the fitter
        """
        return self.runner.fitter


def bootstrap(
    obs,
    runner,
    psf_runner=None,
    ignore_failed_psf=True,
):
    """
    Run a fitter on the input observations, possibly bootstrapping the fit
    based on information inferred from the data or the psf model

    Parameters
    ----------
    obs: ngmix Observation(s)
        Observation, ObsList, or MultiBandObsList
    runner: ngmix Runner
        Must have go(obs=obs) method
    psf_runner: ngmix PSFRunner, optional
        Must have go(obs=obs) method
    ignore_failed_psf: bool, optional
        If set to True, remove observations where the psf fit fails, and
        only fit the remaining.  Default True.

    Side effects
    ------------
    the obs.psf.meta['result'] and the obs.psf.gmix may be set if a psf runner
    is sent and the internal fitter has a get_gmix method.  gmix are only set
    for successful fits
    """

    if psf_runner is not None:
        psf_runner.go(obs=obs)

        if ignore_failed_psf:
            obs = remove_failed_psf_obs(obs=obs)

    return runner.go(obs=obs)


def remove_failed_psf_obs(obs):
    """
    remove observations from the input that failed

    Parameters
    ----------
    obs: observation(s)
        Observation, ObsList, or MultiBandObsList

    Returns
    --------
    obs: observation(s)
        new observations, same type as input
    """
    if isinstance(obs, MultiBandObsList):
        new_mbobs = MultiBandObsList(meta=obs.meta)
        for tobslist in obs:

            new_obslist = ObsList(meta=tobslist.meta)
            for tobs in tobslist:
                if tobs.psf.meta['result']['flags'] == 0:
                    new_obslist.append(tobs)

            if len(new_obslist) == 0:
                raise BootPSFFailure('no good psf fits')

            new_mbobs.append(new_obslist)

        return new_mbobs

    elif isinstance(obs, ObsList):
        new_obslist = ObsList(meta=obs.meta)
        for tobs in obs:
            if tobs.psf.meta['result']['flags'] == 0:
                new_obslist.append(tobs)

        if len(new_obslist) == 0:
            raise BootPSFFailure('no good psf fits')

        return new_obslist
    elif isinstance(obs, Observation):
        if obs.psf.meta['result']['flags'] != 0:
            raise BootPSFFailure('no good psf fits')
        return obs
    else:
        mess = (
            'got obs input type: "%s", should be '
            'Observation, ObsList, or MulitiBandObsList' % type(obs)
        )
        raise ValueError(mess)
