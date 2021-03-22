__all__ = ['get_all_metacal']

import copy
import numpy as np
import logging
from .defaults import DEFAULT_STEP
from .. import simobs
from ..observation import Observation, ObsList, MultiBandObsList
from .metacal import (
    MetacalDilatePSF, MetacalGaussPSF, MetacalFitGaussPSF, MetacalAnalyticPSF,
)

logger = logging.getLogger(__name__)


def get_all_metacal(
    obs,
    psf='gauss',
    step=DEFAULT_STEP,
    fixnoise=True,
    rng=None,
    use_noise_image=False,
    types=None,
):
    """
    Get all combinations of metacal images in a dict

    parameters
    ----------
    obs: Observation, ObsList, or MultiBandObsList
        The values in the dict correspond to these
    psf: string or galsim object, optional
        PSF to use for metacal.  Default 'gauss'.  Note 'fitgauss'
        will usually produce a smaller psf, but it can fail.

            'gauss': reconvolve gaussian that is larger than
                the original and round.
            'fitgauss': fit a gaussian to the PSF and make
                use round, dilated version for reconvolution
            galsim object: any arbitrary galsim object
                Use the exact input object for the reconvolution kernel; this
                psf gets convolved by thye pixel
            'dilate': dilate the origial psf
                just dilate the original psf; the resulting psf is not round,
                so you need to calculate the _psf terms and make an explicit
                correction
    step: float, optional
        The shear step value to use for metacal.  Default 0.01
    fixnoise: bool, optional
        If set to True, add a compensating noise field to cancel the effect of
        the sheared, correlated noise component.  Default True
    rng: np.random.RandomState
        A random number generator; this is required if fixnoise is True and
        use_noise_image is False.  It is also required when psf= is sent, in
        order to add a small amount of noise to the rendered image of the
        psf.
    use_noise_image: bool, optional
        If set to True, use the .noise attribute of the observation
        for fixing the noise when fixnoise=True.
    types: list, optional
        If psf='gauss' or 'fitgauss', then the default set is the minimal
        set ['noshear','1p','1m','2p','2m']

        Otherwise, the default is the full possible set listed in
        ['noshear','1p','1m','2p','2m',
         '1p_psf','1m_psf','2p_psf','2m_psf']

    returns
    -------
    A dictionary with all the relevant metacaled images
        dict keys:
            1p -> ( shear, 0)
            1m -> (-shear, 0)
            2p -> ( 0, shear)
            2m -> ( 0, -shear)
        simular for 1p_psf etc.
    """

    if fixnoise:
        odict = _get_all_metacal_fixnoise(
            obs, step=step, rng=rng,
            use_noise_image=use_noise_image,
            psf=psf,
            types=types,
        )
    else:
        logger.debug("    not doing fixnoise")
        odict = _get_all_metacal(
            obs, step=step, rng=rng,
            psf=psf,
            types=types,
        )

    return odict


def _get_all_metacal(
    obs,
    step=DEFAULT_STEP,
    rng=None,
    psf=None,
    types=None,
):
    """
    internal routine

    get all metacal
    """
    if isinstance(obs, Observation):

        if psf == 'dilate':
            m = MetacalDilatePSF(obs)
        else:

            if psf == 'gauss':
                m = MetacalGaussPSF(obs=obs, rng=rng)
            elif psf == 'fitgauss':
                m = MetacalFitGaussPSF(obs=obs, rng=rng)
            else:
                m = MetacalAnalyticPSF(obs=obs, psf=psf, rng=rng)

        odict = m.get_all(step=step, types=types)

    elif isinstance(obs, MultiBandObsList):
        odict = _make_metacal_mb_obs_list_dict(
            mb_obs_list=obs, step=step, rng=rng,
            psf=psf,
            types=types,
        )
    elif isinstance(obs, ObsList):
        odict = _make_metacal_obs_list_dict(
            obs, step, rng=rng,
            psf=psf,
            types=types,
        )
    else:
        raise ValueError("obs must be Observation, ObsList, "
                         "or MultiBandObsList")

    return odict


def _make_metacal_mb_obs_list_dict(mb_obs_list, step, rng=None, **kw):

    new_dict = None
    for obs_list in mb_obs_list:
        odict = _make_metacal_obs_list_dict(
            obs_list=obs_list, step=step, rng=rng, **kw,
        )

        if new_dict is None:
            new_dict = _init_mb_obs_list_dict(odict.keys())

        for key in odict:
            new_dict[key].append(odict[key])

    return new_dict


def _make_metacal_obs_list_dict(obs_list, step, rng=None, **kw):
    odict = None
    for obs in obs_list:

        todict = _get_all_metacal(obs, step=step, rng=rng, **kw)

        if odict is None:
            odict = _init_obs_list_dict(todict.keys())

        for key in odict:
            odict[key].append(todict[key])

    return odict


def _init_obs_list_dict(keys):
    odict = {}
    for key in keys:
        odict[key] = ObsList()
    return odict


def _init_mb_obs_list_dict(keys):
    odict = {}
    for key in keys:
        odict[key] = MultiBandObsList()
    return odict


def _get_all_metacal_fixnoise(
    obs,
    step=DEFAULT_STEP,
    rng=None,
    use_noise_image=False,
    psf=None,
    types=None,
):
    """
    internal routine
    Add a sheared noise field to cancel the correlated noise
    """

    # Using None for the model means we get just noise
    if use_noise_image:
        noise_obs = _replace_image_with_noise(obs)
        logger.debug("    Doing fixnoise with input noise image")
    else:
        noise_obs = simobs.simulate_obs(gmix=None, obs=obs, rng=rng)

    # rotate by 90
    _rotate_obs_image_square(noise_obs, k=1)

    obsdict = _get_all_metacal(
        obs, step=step, rng=rng,
        psf=psf,
        types=types,
    )
    noise_obsdict = _get_all_metacal(
        noise_obs, step=step, rng=rng,
        psf=psf,
        types=types,
    )

    for type in obsdict:

        imbobs = obsdict[type]
        nmbobs = noise_obsdict[type]

        # rotate back, which is 3 more rotations
        _rotate_obs_image_square(nmbobs, k=3)

        if isinstance(imbobs, Observation):
            _doadd_single_obs(imbobs, nmbobs)

        elif isinstance(imbobs, ObsList):
            for iobs in range(len(imbobs)):

                obs = imbobs[iobs]
                nobs = nmbobs[iobs]

                _doadd_single_obs(obs, nobs)

        elif isinstance(imbobs, MultiBandObsList):
            for imb in range(len(imbobs)):
                iolist = imbobs[imb]
                nolist = nmbobs[imb]

                for iobs in range(len(iolist)):

                    obs = iolist[iobs]
                    nobs = nolist[iobs]

                    _doadd_single_obs(obs, nobs)

    return obsdict


def _rotate_obs_image_square(obs, k=1):
    """
    rotate the image.  internal routine just for fixnoise with rotnoise=True
    """

    if isinstance(obs, Observation):
        obs.set_image(np.rot90(obs.image, k=k))
    elif isinstance(obs, ObsList):
        for tobs in obs:
            _rotate_obs_image_square(tobs, k=k)
    elif isinstance(obs, MultiBandObsList):
        for obslist in obs:
            _rotate_obs_image_square(obslist, k=k)


def _doadd_single_obs(obs, nobs):
    obs.image_orig = obs.image.copy()
    obs.weight_orig = obs.weight.copy()

    # the weight and image can be modified in the context, and update_pixels is
    # automatically called upon exit

    with obs.writeable():
        obs.image += nobs.image

        wpos = np.where(
            (obs.weight != 0.0) &
            (nobs.weight != 0.0)
        )
        if wpos[0].size > 0:
            tvar = obs.weight*0
            # add the variances
            tvar[wpos] = (
                1.0/obs.weight[wpos] +
                1.0/nobs.weight[wpos]
            )
            obs.weight[wpos] = 1.0/tvar[wpos]


def _replace_image_with_noise(obs):
    """
    copy the observation and copy the .noise parameter
    into the image position
    """

    noise_obs = copy.deepcopy(obs)

    if isinstance(noise_obs, Observation):
        noise_obs.image = noise_obs.noise
    elif isinstance(noise_obs, ObsList):
        for nobs in noise_obs:
            nobs.image = nobs.noise
    else:
        for obslist in noise_obs:
            for nobs in obslist:
                nobs.image = nobs.noise

    return noise_obs
