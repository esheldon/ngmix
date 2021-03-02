import numpy
from numpy import where, sqrt, zeros
from .observation import Observation, ObsList, MultiBandObsList
from .gmix import GMix

from copy import deepcopy
import logging

LOGGER = logging.getLogger(__name__)


def simulate_obs(
    gmix, obs,
    add_noise=True,
    rng=None,
    add_all=True,
    noise_factor=None,
    use_raw_weight=True,
    convolve_psf=True,
):
    """
    Simulate the observation(s) using the input gaussian mixture

    parameters
    ----------
    gmix: GMix or subclass
        The gaussian mixture or None
    obs: observation(s)
        One of Observation, ObsList, or MultiBandObsList
    convolve_psf: bool, optional
        If True, convolve by the PSF.  Default True.
    add_noise: bool, optional
        If True, add noise according to the weight map.  Default True.

        The noise image is monkey-patched in as obs.noise_image
    use_raw_weight: bool, optional
        If True, look for a .weight_raw attribute for generating
        the noise.  Often one is using modified weight map to simplify
        masking neighbors, but may want to use the raw map for
        adding noise.  Default True
    add_all: bool, optional
        If True, add noise to zero-weight pixels as well.  For max like methods
        this makes no difference, but if the image is being run through an FFT
        it might be important. Default is True.
    """

    if isinstance(obs, MultiBandObsList):
        return _simulate_mbobs(
            gmix_list=gmix, mbobs=obs,
            add_noise=add_noise,
            rng=rng,
            add_all=add_all,
            noise_factor=noise_factor,
            use_raw_weight=use_raw_weight,
            convolve_psf=convolve_psf,
        )

    else:

        if gmix is not None and not isinstance(gmix, GMix):
            raise ValueError("input gmix must be a gaussian mixture")

        elif isinstance(obs, ObsList):
            return _simulate_obslist(
                gmix, obs,
                add_noise=add_noise,
                rng=rng,
                add_all=add_all,
                noise_factor=noise_factor,
                use_raw_weight=use_raw_weight,
                convolve_psf=convolve_psf,
            )

        elif isinstance(obs, Observation):
            return _simulate_obs(
                gmix, obs,
                add_noise=add_noise,
                rng=rng,
                add_all=add_all,
                noise_factor=noise_factor,
                use_raw_weight=use_raw_weight,
                convolve_psf=convolve_psf,
            )

        else:
            raise ValueError(
                "obs should be an Observation, " "ObsList, or MultiBandObsList"
            )


def _simulate_mbobs(
    gmix_list, mbobs,
    add_noise=True,
    rng=None,
    add_all=True,
    noise_factor=None,
    use_raw_weight=True,
    convolve_psf=True,
):

    if gmix_list is not None:
        if not isinstance(gmix_list, list):
            raise ValueError(
                "for simulating MultiBandObsLists, the "
                "input must be a list of gaussian mixtures"
            )

        if not isinstance(gmix_list[0], GMix):
            raise ValueError("input must be gaussian mixtures")

        if not len(gmix_list) == len(mbobs):

            mess = "len(mbobs)==%d but len(gmix_list)==%d"
            mess = mess % (len(mbobs), len(gmix_list))
            raise ValueError(mess)

    new_mbobs = MultiBandObsList()
    nband = len(mbobs)
    for i in range(nband):
        if gmix_list is None:
            gmix = None
        else:
            gmix = gmix_list[i]

        ol = mbobs[i]
        new_obslist = _simulate_obslist(
            gmix=gmix, obslist=ol,
            add_noise=add_noise,
            rng=rng,
            add_all=add_all,
            noise_factor=noise_factor,
            use_raw_weight=use_raw_weight,
            convolve_psf=convolve_psf,
        )
        new_mbobs.append(new_obslist)

    return new_mbobs


def _simulate_obslist(
    gmix, obslist,
    add_noise=True,
    rng=None,
    add_all=True,
    noise_factor=None,
    use_raw_weight=True,
    convolve_psf=True,
):
    new_obslist = ObsList()
    for o in obslist:
        newobs = simulate_obs(
            gmix=gmix, obs=o,
            add_noise=add_noise,
            rng=rng,
            add_all=add_all,
            noise_factor=noise_factor,
            use_raw_weight=use_raw_weight,
            convolve_psf=convolve_psf,
        )
        new_obslist.append(newobs)

    return new_obslist


def _simulate_obs(
    gmix, obs,
    add_noise=True,
    rng=None,
    add_all=True,
    noise_factor=None,
    use_raw_weight=True,
    convolve_psf=True,
):

    sim_image = _get_simulated_image(gmix, obs, convolve_psf=convolve_psf)

    if add_noise:
        sim_image, noise_image = _get_noisy_image(
            obs, sim_image, rng=rng, add_all=add_all,
            noise_factor=noise_factor,
            use_raw_weight=use_raw_weight,
        )
    else:
        noise_image = None

    if not obs.has_psf():
        psf = None
    else:
        psf = deepcopy(obs.psf)

    weight = obs.weight.copy()

    if noise_factor is not None:
        LOGGER.debug(
            "Modding weight with noise factor: %s" % noise_factor
        )
        weight *= 1.0 / noise_factor ** 2

    new_obs = Observation(
        sim_image, weight=weight, jacobian=obs.jacobian, psf=psf
    )

    new_obs.noise_image = noise_image
    return new_obs


def _get_simulated_image(gmix, obs, convolve_psf=True):
    if gmix is None:
        return zeros(obs.image.shape)

    if convolve_psf:
        psf_gmix = _get_psf_gmix(obs)

        gm = gmix.convolve(psf_gmix)
    else:
        gm = gmix

    sim_image = gm.make_image(obs.image.shape, jacobian=obs.jacobian)

    return sim_image


def _get_noisy_image(obs, sim_image, rng, add_all=True, noise_factor=None,
                     use_raw_weight=True):
    """
    create a noise image from the weight map
    """

    # often we are using a modified weight map for fitting,
    # to simplify masking of neighbors.  The user can request
    # to use an attribute called `weight_raw` instead, which
    # would have the unmodified weight map, good for adding the
    # correct noise

    if hasattr(obs, "weight_raw") and use_raw_weight:
        weight = obs.weight_raw
    else:
        weight = obs.weight

    noise_image = get_noise_image(
        weight=weight, rng=rng, add_all=add_all, noise_factor=noise_factor,
    )
    return sim_image + noise_image, noise_image


BIGNOISE = 1.0e15


def get_noise_image(weight, rng, add_all=True, noise_factor=None):
    """
    get a noise image based on the input weight map

    If add_all, we set weight==0 pixels with the median noise.  This should not
    be a problem for algorithms that use the weight map
    """

    assert rng is not None

    noise_image = rng.normal(loc=0.0, scale=1.0, size=weight.shape,)

    err = zeros(weight.shape)
    w = where(weight > 0)
    if w[0].size > 0:
        err[w] = sqrt(1.0 / weight[w])

        if add_all and (w[0].size != weight.size):
            # there were some zero weight pixels, and we
            # want to add noise there anyway
            median_err = numpy.median(err[w])

            wzero = where(weight <= 0)
            err[wzero] = median_err

        if noise_factor is not None:
            LOGGER.debug("Adding noise factor: %s" % noise_factor)
            err *= noise_factor

    else:
        LOGGER.debug("All weight is zero!  Setting noise to %s" % BIGNOISE)
        err[:, :] = BIGNOISE

    noise_image *= err
    return noise_image


def _get_psf_gmix(obs):
    if not obs.has_psf():
        raise RuntimeError(
            "You requested to convolve by the psf, "
            "but the observation has no psf observation set"
        )

    psf = obs.get_psf()
    if not psf.has_gmix():
        raise RuntimeError(
            "You requested to convolve by the psf, "
            "but the observation has no psf gmix set"
        )

    return psf.gmix
