from .observation import (
    Observation,
    ObsList,
    MultiBandObsList,
)


class Runner(object):
    """
    Run a fitter and guesser on observations

    Parameters
    ----------
    fitter: ngmix fitter or measurer
        An object to perform measurements, must have a go(obs=obs, ntry=ntry)
        method.
    guesser: ngmix guesser object
        Must be a callable returning an array of parameters
    ntry: int, optional
        Number of times to try if there is failure
    """
    def __init__(self, *, fitter, guesser, ntry=1):
        self.fitter = fitter
        self.guesser = guesser
        self.ntry = ntry

    def go(self, *, obs):
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


class PSFRunner(Runner):
    """
    Run a fitter on each psf observation.

    Parameters
    ----------
    fitter: ngmix fitter or measurer
        An object to perform measurements, must have a go(obs=obs, ntry=ntry)
        method.
    guesser: ngmix guesser object
        Must be a callable returning an array of parameters
    ntry: int, optional
        Number of times to try if there is failure
    """

    def go(self, *, obs, set_result=False):
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
        set_result: bool, optional
            If set to True, the meta['result'] and the .gmix attribute

        Side Effects
        ------------
        If set_result is True then .meta['result'] is set to the fit result and the
        .gmix attribuite is set for each successful fit
        """

        run_psf_fitter(
            obs=obs, fitter=self.fitter, guesser=self.guesser, ntry=self.ntry,
            set_result=set_result,
        )


def run_fitter(*, obs, fitter, guesser, ntry=1):
    """
    run a fitter multiple times if needed, with guesses generated from the
    input guesser

    Parameters
    ----------
    obs: ngmix Observation(s)
        Observation, ObsList, or MultiBandObsList
    fitter: ngmix fitter or measurer
        An object to perform measurements, must have a go(obs=obs, ntry=ntry)
        method.
    guesser: ngmix guesser object
        Must be a callable returning an array of parameters
    ntry: int, optional
        Number of times to try if there is failure

    Returns
    -------
    result dictionary
    """

    for i in range(ntry):

        guess = guesser(obs=obs)
        fitter.go(obs=obs, guess=guess)

        res = fitter.get_result()
        if res['flags'] == 0:
            break


def run_psf_fitter(*, obs, fitter, guesser, ntry=1, set_result=False):
    """
    run a fitter on each observation in the input observation(s).  The
    fitter will be run multiple times if needed, with guesses generated from
    the input guesser.  If a psf obs is set that is fit rather than
    the primary observation.

    Parameters
    ----------
    obs: ngmix Observation(s)
        Observation, ObsList, or MultiBandObsList
    fitter: ngmix fitter or measurer
        An object to perform measurements, must have a go(obs=obs, ntry=ntry)
        method.
    guesser: ngmix guesser object
        Must be a callable returning an array of parameters
    ntry: int, optional
        Number of times to try if there is failure
    set_result: bool, optional
        If set to True, the meta['result'] and the .gmix attribute

    Side Effects
    ------------
    If set_result is True then .meta['result'] is set to the fit result and the
    .gmix attribuite is set for each successful fit
    """

    if isinstance(obs, MultiBandObsList):
        for tobslist in obs:
            run_psf_fitter(
                obs=tobslist, fitter=fitter, guesser=guesser, ntry=ntry,
                set_result=set_result,
            )

    elif isinstance(obs, ObsList):
        for tobs in obs:
            run_psf_fitter(
                obs=tobs, fitter=fitter, guesser=guesser, ntry=ntry,
                set_result=set_result,
            )

    elif isinstance(obs, Observation):

        if obs.has_psf():
            obs_to_fit = obs.psf
        else:
            obs_to_fit = obs

        run_fitter(
            obs=obs_to_fit, fitter=fitter, guesser=guesser, ntry=ntry,
        )

        if set_result:
            res = fitter.get_result()
            obs_to_fit.meta['result'] = res

            if res['flags'] == 0 and hasattr(fitter, 'get_gmix'):
                gmix = fitter.get_gmix()
                obs_to_fit.gmix = gmix

    else:
        raise ValueError(
            'obs must be an Observation, ObsList, or MultiBandObsList'
        )
