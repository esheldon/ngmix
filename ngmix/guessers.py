from __future__ import print_function
import numpy
from numpy import log
from .fitting import print_pars
from .gexceptions import GMixRangeError
from .priors import srandu, LOWVAL

class GuesserBase(object):
    def _fix_guess(self, guess, prior, ntry=4):
        """
        Fix a guess for out-of-bounds values according the the input prior

        Bad guesses are replaced by a sample from the prior
        """

        n=guess.shape[0]
        for j in xrange(n):
            for itry in xrange(ntry):
                try:
                    lnp=prior.get_lnprob_scalar(guess[j,:])

                    if lnp <= LOWVAL:
                        dosample=True
                    else:
                        dosample=False
                except GMixRangeError as err:
                    dosample=True

                if dosample:
                    print_pars(guess[j,:], front="bad guess:")
                    guess[j,:] = prior.sample()
                else:
                    break


class TFluxGuesser(GuesserBase):
    """
    get full guesses from just T,fluxes

    parameters
    ----------
    T: float
        Center for T guesses
    fluxes: float or sequences
        Center for flux guesses
    scaling: string
        'linear' or 'log'
    prior: optional
        If sent, "fix-up" guesses if they are not allowed by the prior
    """
    def __init__(self, T, fluxes, prior=None, scaling='linear'):
        self.T=T

        if numpy.isscalar(fluxes):
            fluxes=numpy.array(fluxes, dtype='f8', ndmin=1)

        self.fluxes=fluxes
        self.prior=prior
        self.scaling=scaling

        self.log_T = log(T)
        self.log_fluxes = log(fluxes)

    def __call__(self, n=1, **keys):
        """
        center, shape are just distributed around zero
        """
        fluxes=self.fluxes
        nband=fluxes.size
        np = 5+nband

        guess=numpy.zeros( (n, np) )
        guess[:,0] = 0.01*srandu(n)
        guess[:,1] = 0.01*srandu(n)
        guess[:,2] = 0.1*srandu(n)
        guess[:,3] = 0.1*srandu(n)

        if self.scaling=='linear':
            guess[:,4] = self.T*(1.0 + 0.1*srandu(n))

            fluxes=self.fluxes
            for band in xrange(nband):
                guess[:,5+band] = fluxes[band]*(1.0 + 0.1*srandu(n))

        else:
            guess[:,4] = self.log_T + 0.1*srandu(n)

            for band in xrange(nband):
                guess[:,5+band] = self.log_fluxes[band] + 0.1*srandu(n)

        if self.prior is not None:
            self._fix_guess(guess, self.prior)

        if n==1:
            guess=guess[0,:]
        return guess

class TFluxAndPriorGuesser(GuesserBase):
    """
    Make guesses from the input T, fluxes and prior

    parameters
    ----------
    T: float
        Center for T guesses
    fluxes: float or sequences
        Center for flux guesses
    prior:
        cen, g drawn from this prior
    scaling: string
        'linear' or 'log'
    """
    def __init__(self, T, fluxes, prior, scaling='linear'):
        if numpy.isscalar(fluxes):
            fluxes=numpy.array(fluxes, dtype='f8', ndmin=1)

        self.T=T
        self.fluxes=fluxes
        self.prior=prior
        self.scaling=scaling

        self.log_T = log(T)
        self.log_fluxes = log(fluxes)

    def __call__(self, n=1, **keys):
        """
        center, shape are just distributed around zero
        """
        fluxes=self.fluxes
        log_fluxes=self.log_fluxes

        nband=fluxes.size
        np = 5+nband

        guess = self.prior.sample(n)

        if self.scaling=='linear':
            guess[:,4] = self.T*(1.0 + 0.1*srandu(n))
        else:
            guess[:,4] = self.log_T + 0.1*srandu(n)

        for band in xrange(nband):
            if self.scaling=='linear':
                guess[:,5+band] = fluxes[band]*(1.0 + 0.1*srandu(n))
            else:
                guess[:,5+band] = self.log_fluxes[band] + 0.1*srandu(n)

        self._fix_guess(guess, self.prior)

        if n==1:
            guess=guess[0,:]
        return guess

    def _fix_guess(self, guess, prior, ntry=4):
        """
        just fix T and flux
        """

        n=guess.shape[0]
        for j in xrange(n):
            for itry in xrange(ntry):
                try:
                    lnp=prior.get_lnprob_scalar(guess[j,:])

                    if lnp <= LOWVAL:
                        dosample=True
                    else:
                        dosample=False
                except GMixRangeError as err:
                    dosample=True

                if dosample:
                    print_pars(guess[j,:], front="bad guess:")
                    if itry < ntry:
                        tguess = prior.sample()
                        guess[j, 4:] = tguess[4:]
                    else:
                        # give up and just drawn a sample
                        guess[j,:] = prior.sample()
                else:
                    break

class RoundParsGuesser(GuesserBase):
    """
    pars do not include e1,2e
    """
    def __init__(self, pars, scaling='linear', prior=None, widths=None):
        self.pars=pars
        self.scaling=scaling
        self.prior=prior

        self.np = pars.size

        if widths is None:
            self.widths = pars*0 + 0.1
            self.widths[0:0+2] = 0.02
        else:
            self.widths = widths

    def __call__(self, n=None, **keys):
        """
        center, shape are just distributed around zero
        """

        if n is None:
            is_scalar=True
            n=1
        else:
            is_scalar=False

        pars=self.pars
        widths=self.widths

        guess=numpy.zeros( (n, self.np) )
        guess[:,0] = pars[0] + widths[0]*srandu(n)
        guess[:,1] = pars[1] + widths[1]*srandu(n)

        for i in xrange(2,self.np):
            if self.scaling=='linear':
                guess[:,i] = pars[i]*(1.0 + widths[i]*srandu(n))
            else:
                guess[:,i] = pars[i] + widths[i]*srandu(n)

        if self.prior is not None:
            self._fix_guess(guess, self.prior)

        if is_scalar:
            guess=guess[0,:]

        return guess


