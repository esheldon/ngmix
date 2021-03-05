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

        run_fitter(
            obs=obs, fitter=self.fitter, guesser=self.guesser, ntry=self.ntry,
        )

    def get_result(self):
        """
        get the result dict
        """
        return self.fitter.get_result()


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
    """

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

        Side Effects
        ------------
        .meta['result'] is set to the fit result and the .gmix attribuite is
        set for each successful fit, if appropriate
        """

        run_psf_fitter(
            obs=obs, fitter=self.fitter, guesser=self.guesser, ntry=self.ntry,
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
            fitter.go(obs=obs, guess=guess)
        else:
            fitter.go(obs=obs)

        res = fitter.get_result()
        if res['flags'] == 0:
            break


def run_psf_fitter(obs, fitter, guesser=None, ntry=1):
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

    Side Effects
    ------------
    .meta['result'] is set to the fit result and the .gmix attribuite is set
    for each successful fit, if appropriate
    """

    if isinstance(obs, MultiBandObsList):
        for tobslist in obs:
            run_psf_fitter(
                obs=tobslist, fitter=fitter, guesser=guesser, ntry=ntry,
            )

    elif isinstance(obs, ObsList):
        for tobs in obs:
            run_psf_fitter(
                obs=tobs, fitter=fitter, guesser=guesser, ntry=ntry,
            )

    elif isinstance(obs, Observation):

        if obs.has_psf():
            obs_to_fit = obs.psf
        else:
            obs_to_fit = obs

        run_fitter(
            obs=obs_to_fit, fitter=fitter, guesser=guesser, ntry=ntry,
        )

        res = fitter.get_result()
        obs_to_fit.meta['result'] = res

        if res['flags'] == 0 and hasattr(fitter, 'get_gmix'):
            gmix = fitter.get_gmix()
            obs_to_fit.gmix = gmix

    else:
        raise ValueError(
            'obs must be an Observation, ObsList, or MultiBandObsList'
        )
