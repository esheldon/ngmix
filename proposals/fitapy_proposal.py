"""
schematic proposal for a new api for fitters, runners, and bootstrapping

bootstrapping is now just a function that runs the runners that are
input
"""
from .observation import (
    Observation,
    ObsList,
    MultiBandObsList,
)
from .gexceptions import BootPSFFailure


def bootstrap(
    *,
    obs,
    runner,
    psf_runner,
    remove_failed_psf=True,
):
    """
    Run a fitter on the input observations, possibly bootstrapping the fit
    based on information inferred from the data or the psf model

    Parameters
    ----------
    obs: ngmix Observation(s)
        Observation, ObsList, or MultiBandObsList
    runner: ngmix Runner
        A runner or measurement to run the fit.
    psf_runner: ngmix PSFRunner
        A runner to run the psf fits
    remove_failed_psf: bool, optional
        If set to True, remove observations where the psf fit fails, and
        only fit the remaining.  Default True.

    Returns
    -------
    result dict

    Side effects
    ------------
    the psf.meta['result'] and the .psf.gmix may be set if a psf runner is sent
    and the internal fitter has a get_gmix method.  gmix are only set for
    successful fits

    Note the only place where side effects should be occurring is
    in the call to store_psf_results.
    """

    if psf_runner is not None:
        psf_result = psf_runner.go(obs=obs)

        if remove_failed_psf:
            obs, psf_result = remove_failed_psf_obs(
                obs=obs, result=psf_result,
            )

        if result_has_gmix(psf_result):
            set_gmix = True
        else:
            set_gmix = False

        store_psf_results(
            obs=obs, result=psf_result, set_gmix=set_gmix,
        )

    return runner.go(obs=obs)


class MeasurementBase(object):
    """
    Perform measurements or fits on observations.  The only required
    method is go()
    """
    def go(self, *, obs):
        """
        Run a measurement or fit on the input observation(s)

        Parameters
        ----------
        obs: ngmix Observation(s)
            Observation, ObsList, or MultiBandObsList

        Returns
        -------
        None

        Side Effects
        -------------
        the result attribute is set
        """
        raise NotImplementedError('implement go method for fitter')


class Runner(MeasurementBase):
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

    Side effects
    ------------
    None
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

        return run_fitter(
            obs=obs, fitter=self.fitter, guesser=self.guesser, ntry=self.ntry,
        )


class PSFRunner(Runner):
    """
    Run a fitter on each psf observations.

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

    def go(self, *, obs):
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
        -------
        result dict, or list of, or list of list of, depending on the
        input type

        Side effects
        ------------
        None
        """

        return run_psf_fitter(
            obs=obs, fitter=self.fitter, guesser=self.guesser, ntry=self.ntry,
        )


class MaxlikeFitter(MeasurementBase):
    """
    Perform maximum likelihood fits on observations

    Parameters
    ----------
    prior: ngmix prior, optional
        A prior to use for the fit
    """

    def __init__(self, *, prior=None):
        self.prior = prior

    def go(self, *, obs, guess):
        """
        Run a measurement or fit on the input observation(s)

        Parameters
        ----------
        obs: ngmix Observation(s)
            Observation, ObsList, or MultiBandObsList
        guess: array-like
            Initial starting point for the fit

        Returns
        -------
        result dict

        Side effects
        ------------
        None
        """
        raise NotImplementedError('implement go method for fitter')

    def get_gmix(self, *, pars):
        raise NotImplementedError('implement get_gmix method for fitter')


class Moments(MeasurementBase):
    """
    Measure moments from observations

    Parameters
    ----------
    weight: gaussian mixture object
        The weight to use for the measurements
    """

    def __init__(self, *, weight):
        self.weight = weight

    def go(self, *, obs):
        """
        Measure weighted moments on the observation(s)

        Parameters
        ----------
        obs: ngmix Observation(s)
            Observation, ObsList, or MultiBandObsList

        Returns
        -------
        result dict

        Side effects
        ------------
        None
        """
        raise NotImplementedError('implement go method for fitter')


def remove_failed_psf_obs(*, obs, result):
    """
    remove observations from the input that failed

    Parameters
    ----------
    obs: observation(s)
        Observation, ObsList, or MultiBandObsList
    result: result dict, or list or list of lists
        The results for the psf fits

    Returns
    --------
    obs, result:
        new observations, same type as input, as well
        as new results
    """
    if isinstance(obs, MultiBandObsList):
        new_mbobs = MultiBandObsList(meta=obs.meta)
        new_mbobs_result = []
        for tobslist, treslist in zip(obs, result):

            new_obslist = ObsList(meta=tobslist.meta)
            new_obslist_result = []
            for tobs, tres in zip(tobslist, treslist):
                if tres['flags'] == 0:
                    new_obslist.append(tobs)
                    new_obslist_result.append(tres)

            if len(new_obslist) == 0:
                raise BootPSFFailure('no good psf fits')

            new_mbobs.append(new_obslist)
            new_mbobs_result.append(new_obslist_result)

        return new_mbobs, new_mbobs_result

    elif isinstance(obs, ObsList):
        new_obslist = ObsList(meta=tobslist.meta)
        new_obslist_result = []
        for tobs, tres in zip(obs, result):
            if tres['flags'] == 0:
                new_obslist.append(tobs)
                new_obslist_result.append(tres)

        if len(tobslist) == 0:
            raise BootPSFFailure('no good psf fits')

        return new_obslist, new_obslist_result
    else:
        if result['flags'] != 0:
            raise BootPSFFailure('no good psf fits')

        return obs, result


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

        guess = guesser()
        res = fitter.go(obs=obs, guess=guess)

        if res['flags'] == 0:
            break

    return res


def run_psf_fitter(*, obs, fitter, guesser, ntry=1):
    """
    run a fitter on each psf observation in the input observation(s).  The
    fitter will be run multiple times if needed, with guesses generated from
    the input guesser

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
    result dict, or list of, or list of list of, depending on the
    input type

    Side Effects
    ------------
    .meta['result'] is set to the fit result and the .gmix attribuite is set
    for each successful fit
    """

    if isinstance(obs, MultiBandObsList):
        result = []
        for tobslist in obs:
            tres = run_psf_fitter(
                obs=tobslist, fitter=fitter, guesser=guesser, ntry=ntry,
            )
            result.append(tres)

    elif isinstance(obs, ObsList):
        result = []
        for tobs in obs:
            tres = run_psf_fitter(
                obs=tobs, fitter=fitter, guesser=guesser, ntry=ntry,
            )
            result.append(tres)

    elif isinstance(obs, Observation):
        result = run_fitter(
            obs=obs.psf, fitter=fitter, guesser=guesser, ntry=ntry,
        )
        if result['flags'] == 0 and hasattr(fitter, 'get_gmix'):
            result['gmix'] = fitter.get_gmix(result['pars'])
    else:
        raise ValueError(
            'obs must be an Observation, ObsList, or MultiBandObsList'
        )

    return result


def result_has_gmix(result):
    """
    check if a gmix is present.  Not all are checked
    """
    try:
        if 'gmix' in result:
            return True
        else:
            return False
    except TypeError:
        return result_has_gmix(result[0])


def store_psf_results(*, obs, result, set_gmix=True):
    """
    store the psf results in the obs meta and possibly set the gmix
    for the psf

    Parameters
    ----------
    obs: ngmix Observation(s)
        Observation, ObsList, or MultiBandObsList
    result: result dict, list or list of lists
        The results to store
    set_gmix: bool, optional
        If set to True, the gmix is constructed for the psf and
        set

    Returns
    -------
    None, the result will be stored in the meta for each observation

    Side Effects
    ------------
    .meta['result'] is set to the fit result and the psf.gmix attribuite is set
    for each successful psf fit
    """

    if isinstance(obs, MultiBandObsList):
        for tobslist, tresult in zip(obs, result):
            store_psf_results(obs=tobslist, result=tresult, set_gmix=set_gmix)

    elif isinstance(obs, ObsList):
        for tobs, tresult in zip(obs, result):
            store_psf_results(obs=tobs, result=tresult, set_gmix=set_gmix)

    elif isinstance(obs, Observation):
        if result['flags'] != 0:
            raise ValueError('cannot set gmix for failed fit')

        obs.psf.meta['result'] = result
        if set_gmix:
            if 'gmix' not in result:
                raise ValueError('result has no gmix item')

            obs.psf.gmix = result['gmix']

    else:
        raise ValueError(
            'obs must be an Observation, ObsList, or MultiBandObsList'
        )
