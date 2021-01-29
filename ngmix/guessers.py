import numpy
from numpy import log
from .fitting import print_pars
from .gexceptions import GMixRangeError
from .priors import srandu, LOWVAL
from .shape import Shape

RNG = numpy.random


class GuesserBase(object):
    def _fix_guess(self, guess, prior, ntry=4):
        """
        Fix a guess for out-of-bounds values according the the input prior

        Bad guesses are replaced by a sample from the prior
        """

        n = guess.shape[0]
        for j in range(n):
            for itry in range(ntry):
                try:
                    lnp = prior.get_lnprob_scalar(guess[j, :])

                    if lnp <= LOWVAL:
                        dosample = True
                    else:
                        dosample = False
                except GMixRangeError:
                    dosample = True

                if dosample:
                    print_pars(guess[j, :], front="bad guess:")
                    guess[j, :] = prior.sample()
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

    def __init__(self, T, fluxes, prior=None, scaling="linear"):
        self.T = T

        if numpy.isscalar(fluxes):
            fluxes = numpy.array(fluxes, dtype="f8", ndmin=1)

        self.fluxes = fluxes
        self.prior = prior
        self.scaling = scaling

        if T <= 0.0:
            self.log_T = log(1.0e-10)
        else:
            self.log_T = log(T)

        lfluxes = fluxes.copy()
        (w,) = numpy.where(fluxes < 0.0)
        if w.size > 0:
            lfluxes[w[:]] = 1.0e-10
        self.log_fluxes = log(lfluxes)

    def __call__(self, n=1, **keys):
        """
        center, shape are just distributed around zero
        """
        fluxes = self.fluxes
        nband = fluxes.size
        np = 5 + nband

        guess = numpy.zeros((n, np))
        guess[:, 0] = 0.01 * srandu(n, rng=RNG)
        guess[:, 1] = 0.01 * srandu(n, rng=RNG)
        guess[:, 2] = 0.02 * srandu(n, rng=RNG)
        guess[:, 3] = 0.02 * srandu(n, rng=RNG)

        if self.scaling == "linear":
            guess[:, 4] = self.T * (1.0 + 0.1 * srandu(n, rng=RNG))

            fluxes = self.fluxes
            for band in range(nband):
                guess[:, 5 + band] = fluxes[band] * (1.0 + 0.1 * srandu(n, rng=RNG))

        else:
            guess[:, 4] = self.log_T + 0.1 * srandu(n, rng=RNG)

            for band in range(nband):
                guess[:, 5 + band] = self.log_fluxes[band] + 0.1 * srandu(n, rng=RNG)

        if self.prior is not None:
            self._fix_guess(guess, self.prior)

        if n == 1:
            guess = guess[0, :]
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

    def __init__(self, T, fluxes, prior, scaling="linear"):

        fluxes = numpy.array(fluxes, dtype="f8", ndmin=1)

        self.T = T
        self.fluxes = fluxes
        self.prior = prior
        self.scaling = scaling

        if T <= 0.0:
            self.log_T = log(1.0e-10)
        else:
            self.log_T = log(T)

        lfluxes = self.fluxes.copy()
        (w,) = numpy.where(self.fluxes < 0.0)
        if w.size > 0:
            lfluxes[w[:]] = 1.0e-10
        self.log_fluxes = log(lfluxes)

    def __call__(self, n=1, **keys):
        """
        center, shape are just distributed around zero
        """

        rng = self.prior.cen_prior.rng

        fluxes = self.fluxes

        nband = fluxes.size

        guess = self.prior.sample(n)

        r = rng.uniform(low=-0.1, high=0.1, size=n)
        if self.scaling == "linear":
            guess[:, 4] = self.T * (1.0 + r)
        else:
            guess[:, 4] = self.log_T + r

        for band in range(nband):
            r = rng.uniform(low=-0.1, high=0.1, size=n)
            if self.scaling == "linear":
                guess[:, 5 + band] = fluxes[band] * (1.0 + r)
            else:
                guess[:, 5 + band] = self.log_fluxes[band] + r

        self._fix_guess(guess, self.prior)

        if n == 1:
            guess = guess[0, :]
        return guess

    def _fix_guess(self, guess, prior, ntry=4):
        """
        just fix T and flux
        """

        n = guess.shape[0]
        for j in range(n):
            for itry in range(ntry):
                try:
                    lnp = prior.get_lnprob_scalar(guess[j, :])

                    if lnp <= LOWVAL:
                        dosample = True
                    else:
                        dosample = False
                except GMixRangeError:
                    dosample = True

                if dosample:
                    print_pars(guess[j, :], front="bad guess:")
                    if itry < ntry:
                        tguess = prior.sample()
                        guess[j, 4:] = tguess[4:]
                    else:
                        # give up and just drawn a sample
                        guess[j, :] = prior.sample()
                else:
                    break


class BDFGuesser(TFluxAndPriorGuesser):
    def __init__(self, Tguess, fluxes, prior):
        self.T = Tguess
        self.fluxes = numpy.array(fluxes, ndmin=1)
        self.prior = prior

    def __call__(self, n=1, **keys):
        """
        center, shape are just distributed around zero
        """
        rng = self.prior.cen_prior.rng

        fluxes = self.fluxes

        guess = self.prior.sample(n)

        nband = fluxes.size

        r = rng.uniform(low=-0.1, high=0.1, size=n)
        guess[:, 4] = self.T * (1.0 + r)

        # fracdev prior
        # guess[:,5] = rng.uniform(low=-0.1, high=0.1, size=n)
        # guess[:,5] = rng.uniform(low=0.2, high=0.4, size=n)
        # guess[:,5] = rng.uniform(low=0.1, high=0.3, size=n)
        guess[:, 5] = rng.uniform(low=0.4, high=0.6, size=n)
        # guess[:,5] = rng.uniform(low=0.9, high=0.99, size=n)

        for band in range(nband):
            r = rng.uniform(low=-0.1, high=0.1, size=n)
            guess[:, 6 + band] = fluxes[band] * (1.0 + r)

        if self.prior is not None:
            self._fix_guess(guess, self.prior)

        if n == 1:
            guess = guess[0, :]
        return guess


class BDGuesser(TFluxAndPriorGuesser):
    def __init__(self, Tguess, fluxes, prior):
        self.T = Tguess
        self.fluxes = numpy.array(fluxes, ndmin=1)
        self.prior = prior

    def __call__(self, n=1, **keys):
        """
        center, shape are just distributed around zero
        """
        rng = self.prior.cen_prior.rng

        fluxes = self.fluxes

        guess = self.prior.sample(n)

        nband = fluxes.size

        r = rng.uniform(low=-0.1, high=0.1, size=n)
        guess[:, 4] = self.T * (1.0 + r)

        # fracdev prior
        # guess[:,5] = rng.uniform(low=-0.1, high=0.1, size=n)
        # guess[:,5] = rng.uniform(low=0.2, high=0.4, size=n)
        # guess[:,5] = rng.uniform(low=0.1, high=0.3, size=n)
        guess[:, 5] = rng.uniform(low=0.4, high=0.6, size=n)
        # logTratio
        # guess[:,6] = rng.uniform(low=-0.01, high=0.01, size=n)
        # guess[:,5] = rng.uniform(low=0.9, high=0.99, size=n)

        for band in range(nband):
            r = rng.uniform(low=-0.1, high=0.1, size=n)
            guess[:, 7 + band] = fluxes[band] * (1.0 + r)

        if self.prior is not None:
            self._fix_guess(guess, self.prior)

        if n == 1:
            guess = guess[0, :]
        return guess


class ParsGuesser(GuesserBase):
    """
    pars include g1,g2
    """

    def __init__(self, pars, scaling="linear", prior=None, widths=None):
        self.pars = pars
        self.scaling = scaling
        self.prior = prior

        self.np = pars.size

        if widths is None:
            self.widths = pars * 0 + 0.1
            self.widths[0:0+2] = 0.02
        else:
            self.widths = widths

    def __call__(self, n=None, **keys):
        """
        center, shape are just distributed around zero
        """

        if n is None:
            is_scalar = True
            n = 1
        else:
            is_scalar = False

        pars = self.pars
        widths = self.widths

        guess = numpy.zeros((n, self.np))
        guess[:, 0] = pars[0] + widths[0] * srandu(n, rng=RNG)
        guess[:, 1] = pars[1] + widths[1] * srandu(n, rng=RNG)

        # prevent from getting too large
        guess_shape = get_shape_guess(
            pars[2], pars[3], n, widths[2:2+2], max=0.8
        )
        guess[:, 2] = guess_shape[:, 0]
        guess[:, 3] = guess_shape[:, 1]

        for i in range(4, self.np):
            if self.scaling == "linear":
                guess[:, i] = pars[i] * (1.0 + widths[i] * srandu(n, rng=RNG))
            else:
                guess[:, i] = pars[i] + widths[i] * srandu(n, rng=RNG)

        if self.prior is not None:
            self._fix_guess(guess, self.prior)

        if is_scalar:
            guess = guess[0, :]

        return guess


def get_shape_guess(g1, g2, n, width, max=0.99):
    """
    Get guess, making sure in range
    """

    g = numpy.sqrt(g1 ** 2 + g2 ** 2)
    if g > max:
        fac = max / g

        g1 = g1 * fac
        g2 = g2 * fac

    guess = numpy.zeros((n, 2))
    shape = Shape(g1, g2)

    for i in range(n):

        while True:
            try:
                g1_offset = width[0] * srandu(rng=RNG)
                g2_offset = width[1] * srandu(rng=RNG)
                shape_new = shape.get_sheared(g1_offset, g2_offset)
                break
            except GMixRangeError:
                pass

        guess[i, 0] = shape_new.g1
        guess[i, 1] = shape_new.g2

    return guess


class MomGuesser(GuesserBase):
    """
    pars are [cen1,cen2,M1,M2,T,I]
    """

    def __init__(self, pars, prior=None, widths=None):
        self.pars = pars
        self.prior = prior

        self.np = pars.size

        if widths is None:
            self.widths = pars * 0 + 0.1
        else:
            self.widths = widths

    def __call__(self, n=None, **keys):
        """
        center, shape are just distributed around zero
        """

        if n is None:
            is_scalar = True
            n = 1
        else:
            is_scalar = False

        pars = self.pars
        widths = self.widths

        guess = numpy.zeros((n, self.np))

        for i in range(self.np):
            guess[:, i] = pars[i] + widths[i] * srandu(n, rng=RNG)

        if self.prior is not None:
            self._fix_guess(guess, self.prior)

        if is_scalar:
            guess = guess[0, :]

        return guess


class R50FluxGuesser(object):
    """
    get full guesses from just r50 and fluxes

    parameters
    ----------
    r50: float
        Center for r50 (half light radius )guesses
    fluxes: float or sequences
        Center for flux guesses
    prior: optional
        If sent, "fix-up" guesses if they are not allowed by the prior
    """

    def __init__(self, r50, fluxes, prior=None, rng=None):

        if r50 < 0.0:
            raise GMixRangeError("r50 <= 0: %g" % r50)

        self.r50 = r50

        if numpy.isscalar(fluxes):
            fluxes = numpy.array(fluxes, dtype="f8", ndmin=1)

        self.fluxes = fluxes
        self.prior = prior

        if prior is not None and hasattr(prior, "rng"):
            rng = prior.rng

        if rng is None:
            rng = numpy.random.RandomState()

        self.rng = rng

    def __call__(self, n=1, **keys):
        """
        center, shape are just distributed around zero
        """

        rng = self.rng

        fluxes = self.fluxes
        nband = fluxes.size
        np = 5 + nband

        guess = numpy.zeros((n, np))
        guess[:, 0] = 0.01 * srandu(n, rng=rng)
        guess[:, 1] = 0.01 * srandu(n, rng=rng)
        guess[:, 2] = 0.02 * srandu(n, rng=rng)
        guess[:, 3] = 0.02 * srandu(n, rng=rng)

        guess[:, 4] = self.r50 * (1.0 + 0.1 * srandu(n, rng=rng))

        fluxes = self.fluxes
        for band in range(nband):
            guess[:, 5 + band] = fluxes[band] * (
                1.0 + 0.1 * srandu(n, rng=rng)
            )

        if self.prior is not None:
            self._fix_guess(guess, self.prior)

        if n == 1:
            guess = guess[0, :]
        return guess

    def _fix_guess(self, guess, prior, ntry=4):
        """
        Fix a guess for out-of-bounds values according the the input prior

        Bad guesses are replaced by a sample from the prior
        """

        n = guess.shape[0]
        for j in range(n):
            for itry in range(ntry):
                try:
                    lnp = prior.get_lnprob_scalar(guess[j, :])

                    if lnp <= LOWVAL:
                        dosample = True
                    else:
                        dosample = False
                except GMixRangeError:
                    dosample = True

                if dosample:
                    print_pars(guess[j, :], front="bad guess:")
                    guess[j, :] = prior.sample()
                else:
                    break


class PriorGuesser(object):
    def __init__(self, prior):
        self.prior = prior

    def __call__(self, n=None):
        return self.prior.sample(n)


class R50NuFluxGuesser(R50FluxGuesser):
    """
    get full guesses from just r50 spergel nu and fluxes

    parameters
    ----------
    r50: float
        Center for r50 (half light radius )guesses
    nu: float
        Index for the spergel function
    fluxes: float or sequences
        Center for flux guesses
    prior: optional
        If sent, "fix-up" guesses if they are not allowed by the prior
    """

    NUMIN = -0.99
    NUMAX = 3.5

    def __init__(self, r50, nu, fluxes, prior=None, rng=None):
        super(R50NuFluxGuesser, self).__init__(
            r50, fluxes, prior=prior, rng=rng,
        )

        if nu < self.NUMIN:
            nu = self.NUMIN
        elif nu > self.NUMAX:
            nu = self.NUMAX

        self.nu = nu

    def __call__(self, n=1, **keys):
        """
        center, shape are just distributed around zero
        """

        rng = self.rng

        fluxes = self.fluxes
        nband = fluxes.size
        np = 6 + nband

        guess = numpy.zeros((n, np))
        guess[:, 0] = 0.01 * srandu(n, rng=rng)
        guess[:, 1] = 0.01 * srandu(n, rng=rng)
        guess[:, 2] = 0.02 * srandu(n, rng=rng)
        guess[:, 3] = 0.02 * srandu(n, rng=rng)

        guess[:, 4] = self.r50 * (1.0 + 0.1 * srandu(n, rng=rng))

        for i in range(n):
            while True:
                nuguess = self.nu * (1.0 + 0.1 * srandu(rng=rng))
                if nuguess > self.NUMIN and nuguess < self.NUMAX:
                    break
            guess[i, 5] = nuguess

        fluxes = self.fluxes
        for band in range(nband):
            guess[:, 6 + band] = fluxes[band] * (
                1.0 + 0.1 * srandu(n, rng=rng)
            )

        if self.prior is not None:
            self._fix_guess(guess, self.prior)

        if n == 1:
            guess = guess[0, :]
        return guess
