"""
TODO

    - make a tester for it
    - test it in nsim
    - make it possible to specify the guess type (not just psf)

"""
import logging
import numpy as np

from .observation import Observation, ObsList, MultiBandObsList
from .gexceptions import BootPSFFailure
from copy import deepcopy

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


def replace_masked_pixels(
    mb_obs_list, inplace=False, method="best-fit", fitter=None, add_noise=False
):
    """
    replaced masked pixels

    The original image is stored for each Observation as .image_orig

    parameters
    ----------
    mb_obs_list: MultiBandObsList
        The original observations
    inplace: bool
        If True, modify the data in place.  Default False; a full
        copy is made.
    method: string, optional
        Method for replacement.  Supported methods are 'best-fit'.
        Default is 'best-fit'
    fitter:
        when method=='best-fit', a fitter from fitting.py

    add_noise: bool
        If True, add noise to the replaced pixels based on the median
        noise in the image, derived from the weight map
    """

    assert method == "best-fit", "only best-fit replacement is supported"
    assert fitter is not None, "fitter required"

    if inplace:
        mbo = mb_obs_list
    else:
        mbo = deepcopy(mb_obs_list)

    nband = len(mbo)

    for band in range(nband):
        olist = mbo[band]
        for iobs, obs in enumerate(olist):

            im = obs.image

            if obs.has_bmask():
                bmask = obs.bmask
            else:
                bmask = None

            if hasattr(obs, "weight_raw"):
                weight = obs.weight_raw
            else:
                weight = obs.weight

            if bmask is not None:
                w = np.where((bmask != 0) | (weight == 0.0))
            else:
                w = np.where(weight == 0.0)

            if w[0].size > 0:
                logger.debug(
                    "        replacing %d/%d masked or zero weight "
                    "pixels" % (w[0].size, im.size)
                )
                obs.image_orig = obs.image.copy()
                gm = fitter.get_convolved_gmix(band=band, obsnum=iobs)

                im = obs.image
                model_image = gm.make_image(im.shape, jacobian=obs.jacobian)

                im[w] = model_image[w]

                if add_noise:
                    wgood = np.where(weight > 0.0)
                    if wgood[0].size > 0:
                        median_err = np.median(1.0 / weight[wgood])

                        noise_image = np.random.normal(
                            loc=0.0, scale=median_err, size=im.shape
                        )

                        im[w] += noise_image[w]

            else:
                obs.image_orig = None

    return mbo
