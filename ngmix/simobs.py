import numpy
from numpy import where, sqrt, zeros
from .observation import Observation, ObsList, MultiBandObsList
from .gmix import GMix

from copy import deepcopy

def simulate_obs(gmix, obs, **kw):
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
        return _simulate_mobs(gmix, obs, **kw)
    else:

        if gmix is not None and not isinstance(gmix, GMix):
            raise ValueError("input gmix must be a gaussian mixture")

        elif isinstance(obs, ObsList):
            return _simulate_obslist(gmix, obs, **kw)

        elif isinstance(obs, Observation):
            return _simulate_obs(gmix, obs, **kw)

        else:
            raise ValueError("obs should be an Observation, "
                             "ObsList, or MultiBandObsList")

def _simulate_mobs(gmix_list, mobs, **kw):
    if gmix_list is not None:
        if not isinstance(gmix_list, list):
            raise ValueError("for simulating MultiBandObsLists, the "
                             "input must be a list of gaussian mixtures")

        if not isinstance(gmix_list[0], GMix):
            raise ValueError("input must be gaussian mixtures")

        if not len(gmix_list)==len(mobs):

            mess="len(mobs)==%d but len(gmix_list)==%d"
            mess=mess % (len(mobs),len(gmix_list))
            raise ValueError(mess)

    new_mobs=MultiBandObsList()
    nband = len(mobs)
    for i in range(nband):
        if gmix_list is None:
            gmix=None
        else:
            gmix = gmix_list[i]

        ol = mobs[i]
        new_obslist=_simulate_obslist(gmix, ol, **kw)
        new_mobs.append(new_obslist)

    return new_mobs

def _simulate_obslist(gmix, obslist, **kw):
    new_obslist=ObsList()
    for o in obslist:
        newobs = simulate_obs(gmix, o, **kw)
        new_obslist.append( newobs )

    return new_obslist

def _simulate_obs(gmix, obs, **kw):
    sim_image = _get_simulated_image(gmix, obs, **kw)

    add_noise = kw.get('add_noise',True)
    if add_noise:
        sim_image, noise_image = _get_noisy_image(obs, sim_image, **kw)
    else:
        noise_image=None

    if not obs.has_psf():
        psf=None
    else:
        psf=deepcopy( obs.psf )

    weight=obs.weight.copy()

    noise_factor=kw.get("noise_factor",None)
    if noise_factor is not None:
        print("    Modding weight with noise factor:",noise_factor)
        weight *= (1.0/noise_factor**2)

    new_obs = Observation(
        sim_image,
        weight=weight,
        jacobian=obs.jacobian,
        psf=psf
    )

    new_obs.noise_image = noise_image
    return new_obs

def _get_simulated_image(gmix, obs, **kw):
    if gmix is None:
        return zeros(obs.image.shape)

    convolve_psf=kw.get('convolve_psf',True)
    if convolve_psf:
        psf_gmix = _get_psf_gmix(obs)

        gm = gmix.convolve(psf_gmix)
    else:
        gm=gmix

    sim_image= gm.make_image(obs.image.shape,
                             jacobian=obs.jacobian)

    return sim_image

def _get_noisy_image(obs, sim_image, **kw):
    """
    create a noise image from the weight map
    """

    # often we are using a modified weight map for fitting,
    # to simplify masking of neighbors.  The user can request
    # to use an attribute called `weight_raw` instead, which
    # would have the unmodified weight map, good for adding the
    # correct noise

    use_raw_weight=kw.get('use_raw_weight',True)
    if hasattr(obs, 'weight_raw') and use_raw_weight:
        #print("        using weight raw for simobs noise")
        weight = obs.weight_raw
    else:
        weight = obs.weight

    noise_image = get_noise_image(weight, **kw)
    return sim_image + noise_image, noise_image

BIGNOISE=1.0e15
def get_noise_image(weight, **kw):
    """
    get a noise image based on the input weight map

    If add_all, we set weight==0 pixels with the median noise.  This should not
    be a problem for algorithms that use the weight map
    """
    add_all=kw.get('add_all',True)

    if 'rng' in kw:
        randn=kw['rng'].normal
    else:
        randn=numpy.random.normal

    noise_image = randn(
        loc=0.0,
        scale=1.0,
        size=weight.shape,
    )

    err = zeros(weight.shape)
    w=where(weight > 0)
    if w[0].size > 0:
        err[w] = sqrt(1.0/weight[w])

        if add_all and (w[0].size != weight.size):
            #print("adding noise to all")
            # there were some zero weight pixels, and we
            # want to add noise there anyway
            median_err = numpy.median(err[w])

            wzero=where(weight <= 0)
            err[wzero] = median_err

        noise_factor=kw.get("noise_factor",None)
        if noise_factor is not None:
            print("    Adding noise factor:",noise_factor)
            err *= noise_factor

    else:
        print("    All weight is zero!  Setting noise to",BIGNOISE)
        err[:,:] = BIGNOISE


    noise_image *= err
    return noise_image


def _get_psf_gmix(obs):
    if not obs.has_psf():
        raise RuntimeError("You requested to convolve by the psf, "
                           "but the observation has no psf observation set")

    psf = obs.get_psf()
    if not psf.has_gmix():
        raise RuntimeError("You requested to convolve by the psf, "
                           "but the observation has no psf gmix set")

    return psf.gmix
