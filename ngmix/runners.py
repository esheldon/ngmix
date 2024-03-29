from .observation import (
    Observation,
    ObsList,
    MultiBandObsList,
)


class RunnerBase(object):
    """
    Run a fitter and guesser on observations

    Parameters
    ----------
    fitter: ngmix fitter or measurer
        An object to perform measurements, must have a go(obs=obs, guess=guess)
        method.
    guesser: ngmix guesser object, optional
        Must be a callable returning an array of parameters.
    ntry: int, optional
        Number of times to try if there is failure
    """
    def __init__(self, fitter, guesser=None, ntry=1):
        self.fitter = fitter
        self.guesser = guesser
        self.ntry = ntry


class Runner(RunnerBase):
    """
    Run a fitter and guesser on observations

    Parameters
    ----------
    fitter: ngmix fitter or measurer
        An object to perform measurements, must have a go(obs=obs, guess=guess)
        method.
    guesser: ngmix guesser object, optional
        Must be a callable returning an array of parameters.
    ntry: int, optional
        Number of times to try if there is failure
    """
    def go(self, obs):
        """
        Run the fitter on the input observation(s), possibly multiple times
        using guesses generated from the guesser

        Parameters
        ----------
        obs: ngmix Observation(s)
            Observation, ObsList, or MultiBandObsList

        Returns
        -------
        result dictionary
        """

        return run_fitter(
            obs=obs, fitter=self.fitter, guesser=self.guesser, ntry=self.ntry,
        )


class PSFRunner(RunnerBase):
    """
    Run a fitter on each psf observation.

    Parameters
    ----------
    fitter: ngmix fitter or measurer
        An object to perform measurements, must have a go(obs=obs, guess=guess)
        method.
    guesser: ngmix guesser object
        Must be a callable returning an array of parameters
    ntry: int, optional
        Number of times to try if there is failure
    set_result: bool
        If True, set the result and possibly a gmix in the observation.
        Default True
    """
    def __init__(self, fitter, guesser=None, ntry=1, set_result=True):
        self.fitter = fitter
        self.guesser = guesser
        self.ntry = ntry
        self.set_result = set_result

    def go(self, obs):
        """
        Run the fitter on the psf observations associated with the input
        observation(s), possibly multiple times using guesses generated from
        the guesser.

        Parameters
        ----------
        obs: ngmix Observation(s)
            Observation, ObsList, or MultiBandObsList
        ntry: int, optional
            Number of times to try if there is failure

        Returns
        --------
        result if obs is an Observation
        result list if obs is an ObsList
        list of result lists if obs is a MultiBandObsList

        Side Effects
        ------------
        if set_result is True, then obs.meta['result'] is set to the fit result
        and the .gmix attribuite is set for each successful fit, if appropriate
        """

        return run_psf_fitter(
            obs=obs, fitter=self.fitter, guesser=self.guesser, ntry=self.ntry,
            set_result=self.set_result,
        )


def run_fitter(obs, fitter, guesser=None, ntry=1):
    """
    run a fitter multiple times if needed, with guesses generated from the
    input guesser

    Parameters
    ----------
    obs: ngmix Observation(s)
        Observation, ObsList, or MultiBandObsList
    fitter: ngmix fitter or measurer
        An object to perform measurements, must have a go(obs=obs, guess=guess)
        method.
    guesser: ngmix guesser object, optional
        Must be a callable returning an array of parameters
    ntry: int, optional
        Number of times to try if there is failure

    Returns
    -------
    result dictionary
    """

    for i in range(ntry):

        if guesser is not None:
            guess = guesser(obs=obs)
            res = fitter.go(obs=obs, guess=guess)
        else:
            res = fitter.go(obs=obs)

        if res['flags'] == 0:
            break

    return res


def run_psf_fitter(obs, fitter, guesser=None, ntry=1, set_result=True):
    """
    run a fitter on each observation in the input observation(s).  The fitter
    will be run multiple times if needed, with guesses generated from the input
    guesser if one is sent.  If a psf obs is set that is fit rather than the
    primary observation.

    Parameters
    ----------
    obs: ngmix Observation(s)
        Observation, ObsList, or MultiBandObsList
    fitter: ngmix fitter or measurer
        An object to perform measurements, must have a go(obs=obs, guess=guess)
        method.
    guesser: ngmix guesser object, optional
        Must be a callable returning an array of parameters
    ntry: int, optional
        Number of times to try if there is failure
    set_result: bool
        If True, set the result and possibly a gmix in the observation.
        Default True

    Side Effects
    ------------
    if set_result is True, then obs.meta['result'] is set to the fit result
    and the .gmix attribuite is set for each successful fit, if appropriate
    """

    if isinstance(obs, MultiBandObsList):
        reslol = []
        for tobslist in obs:
            reslist = run_psf_fitter(
                obs=tobslist, fitter=fitter, guesser=guesser, ntry=ntry,
                set_result=set_result,
            )
            reslol.append(reslist)
        return reslol

    elif isinstance(obs, ObsList):
        reslist = []
        for tobs in obs:
            res = run_psf_fitter(
                obs=tobs, fitter=fitter, guesser=guesser, ntry=ntry,
                set_result=set_result,
            )
            reslist.append(res)
        return reslist

    elif isinstance(obs, Observation):

        if obs.has_psf():
            obs_to_fit = obs.psf
        else:
            obs_to_fit = obs

        res = run_fitter(
            obs=obs_to_fit, fitter=fitter, guesser=guesser, ntry=ntry,
        )

        if set_result:
            obs_to_fit.meta['result'] = res

            if res['flags'] == 0 and hasattr(res, 'get_gmix'):
                gmix = res.get_gmix()
                obs_to_fit.gmix = gmix

        return res

    else:
        raise ValueError(
            'obs must be an Observation, ObsList, or MultiBandObsList'
        )
