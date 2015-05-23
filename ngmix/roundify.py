import numpy
from .observation import Observation,ObsList,MultiBandObsList


def get_round_mb_obs_list(mb_obs_list_in, sim_image=True):
    """
    Get roundified version of the MultiBandObsList

    The observations must have
        - a gmix set, representing the pre-psf object
        - a psf observation set, with it's own gmix set

    By default the image is simulated to have the round model convolved with
    the round psf, and noise is added according to the weight map.
    """


    mb_obs_list=MultiBandObsList()
    for obs_list in mb_obs_list_in:
        new_obs_list = get_round_obs_list(obs_list,sim_image=sim_image)
        mb_obs_list.append( new_obs_list )

    return mb_obs_list


def get_round_obs_list(obs_list_in, sim_image=True):
    """
    Get roundified version of the ObsList

    The observations must have
        - a gmix set, representing the pre-psf object
        - a psf observation set, with it's own gmix set

    By default the image is simulated to have the round model convolved with
    the round psf, and noise is added according to the weight map.
    """

    obs_list=ObsList()
    for obs in obs_list_in:
        new_obs = get_round_obs(obs,sim_image=sim_image)
        obs_list.append( new_obs )

    return obs_list

def get_round_obs(obs_in, sim_image=True):
    """
    Get roundified version of the observation.

    The observation must have
        - a gmix set, representing the pre-psf object
        - a psf observation set, with it's own gmix set

    By default the image is simulated to have the round model convolved with
    the round psf, and noise is added according to the weight map.
    """

    assert obs_in.has_gmix(),"observation must have a gmix set"
    assert obs_in.has_psf(),"observation must have a psf set"
    assert obs_in.psf.has_gmix(),"psf observation must have a gmix set"

    psf_round = obs_in.psf.gmix.make_round()

    #pgm=obs_in.psf.gmix
    #g1,g2,T=pgm.get_g1g2T()
    #g1n,g2n,Tn=psf_round.get_g1g2T()
    #print("    psf_e1,e2,T:",g1,g2,T,"round:",g1n,g2n,Tn)

    gm0_round = obs_in.gmix.make_round()

    gm_round = gm0_round.convolve(psf_round)


    weight = obs_in.weight.copy()
    if sim_image:
        noise_image = numpy.random.normal(size=weight.shape)
        w=numpy.where(weight > 0)

        if w[0].size > 0:
            noise_image[w] *= 1/numpy.sqrt(weight[w])

        imuse = gm_round.make_image(weight.shape,
                                    jacobian=obs_in.jacobian)
        imuse += noise_image

        psf_imuse = psf_round.make_image(obs_in.psf.image.shape,
                                         jacobian=obs_in.psf.jacobian)
    else:
        imuse=obs_in.image.copy()
        psf_imuse=obs_in.psf.image.copy()

    psf_obs = Observation(psf_imuse,
                          jacobian=obs_in.psf.jacobian,
                          gmix=psf_round)
    obs=Observation(imuse,
                    weight=weight,
                    jacobian=obs_in.jacobian,
                    gmix=gm_round,
                    psf=psf_obs)
    return obs

def get_simple_sub_pars(pars_in):
    """
    for pars laid out like [cen1,cen2,g1,g2,...]
    get the subset [cen1,cen2,...]
    """

    pars_sub=numpy.zeros(pars_in.size-2)
    pars_sub[0:0+2] = pars_in[0:0+2]
    pars_sub[2:] = pars_in[4:]
    return pars_sub
