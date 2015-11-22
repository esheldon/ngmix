from __future__ import print_function

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
        The gaussian mixture
    obs: observation(s)
        One of Observation, ObsList, or MultiBandObsList
    convolve_psf: bool
        If True, convolve by the PSF.  Default True.
    add_noise: bool
        If True, add noise according to the weight map.  Default True.

        The noise image is monkey-patched in as obs.noise_image
    """

    if isinstance(obs, MultiBandObsList):
        return _simulate_mobs(gmix, obs, **kw)
    else:

        if not isinstance(gmix, GMix):
            raise ValueError("input gmix must be a gaussian mixture")

        elif isinstance(obs, ObsList):
            return _simulate_obslist(gmix, obs, **kw)

        elif isinstance(obs, Observation):
            return _simulate_obs(gmix, obs, **kw)

        else:
            raise ValueError("obs should be an Observation, "
                             "ObsList, or MultiBandObsList")

def _simulate_mobs(gmix_list, mobs, **kw):
    if not isinstance(gmix_list, list):
        raise ValueError("for simulating MultiBandObsLists, the "
                         "input must be a list of gaussian mixtures")

    if not isinstance(gmix_list[0], GMix):
        raise ValueError("input must be gaussian mixtures")

    if not len(gmix_list)==len(mobs):

        mess="len(obs)==%d but len(gmix_list)==%d"
        mess=mess % (len(obs),len(gmix_list))
        raise ValueError(mess)

    new_mobs=MultiBandObsList()
    nband = len(mobs)
    for i in xrange(nband):
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
        #print("    adding noise")
        sim_image, noise_image = _get_noisy_image(obs, sim_image)
    else:
        noise_image=None
        #print("    not adding noise")

    if not obs.has_psf():
        psf=None
    else:
        psf=deepcopy( obs.psf )

    new_obs = Observation(
        sim_image,
        weight=obs.weight.copy(),
        jacobian=obs.jacobian.copy(),
        psf=psf
    )

    new_obs.noise_image = noise_image
    return new_obs

def _get_simulated_image(gmix, obs, **kw):
    convolve_psf=kw.get('convolve_psf',True)
    if convolve_psf:
        #print("    convolving psf")
        psf_gmix = _get_psf_gmix(obs)

        gm = gmix.convolve(psf_gmix)
    else:
        gm=gmix

    sim_image= gm.make_image(obs.image.shape,
                             jacobian=obs.jacobian)

    return sim_image

def _get_noisy_image(obs, sim_image):
    noise_image = get_noise_image(obs.weight)
    return sim_image + noise_image, noise_image

def get_noise_image(weight):
    """
    get a noise image based on the input weight map
    """
    noise_image = numpy.random.normal(loc=0.0,
                                      scale=1.0,
                                      size=weight.shape)

    err = zeros(weight.shape)
    w=where(weight > 0)
    if w[0].size > 0:
        err[w] = sqrt(1.0/weight[w])

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
