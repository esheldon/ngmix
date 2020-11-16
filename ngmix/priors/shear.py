

class GPriorGauss(GPriorBase):
    """
    Gaussian shear prior.

    See `GPriorBase` for more documentation.

    parameters
    ----------
    pars: float
        The width of the Gaussian prior for g1, g2.
    rng: np.random.RandomState or None
        An RNG to use. If None, a new RNG is made using the numpy global RNG
        to generate a seed.
    """
    def __init__(self, *args, **kw):
        super(GPriorGauss, self).__init__(*args, **kw)
        self.sigma = float(self.pars)

    def sample1d(self, nrand, **kw):
        """
        Not implemented for Gaussian shear priors.
        """
        raise NotImplementedError("no 1d for gauss")

    def sample2d(self, nrand=None, **kw):
        """
        Get random g1,g2 values by first drawing
        from the 1-d distribution and assuming rotational symmetry.

        parameters
        ----------
        nrand: int
            Number to generate

        returns
        -------
        g1: array-like
            The generated g1 values.
        g2: array-like
            The generated g2 values.
        """
        if nrand is None:
            nrand = 1
            is_scalar = True
        else:
            is_scalar = False

        rng = self.rng

        gmax = self.gmax - 1.0e-4

        g1 = numpy.zeros(nrand)
        g2 = numpy.zeros(nrand)

        ngood = 0
        nleft = nrand
        while ngood < nrand:

            # generate total g in [0,gmax)
            g1rand = rng.normal(size=nleft, scale=self.sigma)
            g2rand = rng.normal(size=nleft, scale=self.sigma)
            grand = numpy.sqrt(g1rand ** 2 + g2rand ** 2)

            (w,) = numpy.where(grand < gmax)
            if w.size > 0:
                g1[ngood:ngood + w.size] = g1rand[w]
                g2[ngood:ngood + w.size] = g2rand[w]
                ngood += w.size
                nleft -= w.size

        if is_scalar:
            g1 = g1[0]
            g2 = g2[0]

        return g1, g2


class GPriorBA(GPriorBase):
    """
    Bernstein & Armstrong 2013 shear prior.

    Note this prior automatically has max lnprob 0 and max prob 1.

    See `GPriorBase` for more documentation.

    parameters
    ----------
    sigma: float, optional
        The overall width of the prior on |g|, matches `gsimga` from the paper.
        Default is 0.3.
    A: float, optional
        The overall amplitude of the prior. This is used for fitting, but not
        when evaluating lnprob. Default is 1.0.
    rng: np.random.RandomState or None
        An RNG to use. If None, a new RNG is made using the numpy global RNG
        to generate a seed.
    """
    def __init__(self, sigma=0.3, A=1.0, rng=None):
        PriorBase.__init__(self, rng=rng)

        self.set_pars([A, sigma])
        self.gmax = 1.0

    def sample1d(self, nrand, maxguess=None):
        """
        Get random |g| from the 1d distribution

        Set self.gmax appropriately.

        parameters
        ----------
        nrand: int
            Number to generate
        maxguess: float
            The guess for finding the maximum g value if it is needed.

        returns
        -------
        g: array-like
            The generated |g| values.
        """
        if maxguess is None:
            maxguess = self.sigma + 0.0001 * srandu(rng=self.rng)

        return super(GPriorBA, self).sample1d(nrand, maxguess=maxguess)

    def set_pars(self, pars):
        """
        Set parameters of the prior.

        This method is primarily used for fitting.

        parameters
        ----------
        pars: array-like, length 2
            The paraneters [`A`, and `sigma`].
        """
        # used for fitting
        self.A = pars[0]
        self.set_sigma(pars[1])

    def set_sigma(self, sigma):
        """
        Set sigma and its variants, sigma**2, sigma**4, etc.
        """
        # some of this is for the analytical pqr calculation
        self.sigma = sigma
        self.sig2 = self.sigma ** 2
        self.sig4 = self.sigma ** 4
        self.sig2inv = 1.0 / self.sig2
        self.sig4inv = 1.0 / self.sig4

    def get_fdiff(self, g1, g2):
        """
        Compute sqrt(-2ln(p)) ~ (data - mode)/err for using with LM fitters.
        """
        if isinstance(g1, numpy.ndarray):
            return self._get_fdiff_array(g1, g2)
        else:
            return self._get_fdiff_scalar(g1, g2)

    def _get_fdiff_scalar(self, g1, g2):
        # For use with LM fitter, which requires
        #     (model-data)/width
        # so we need to fake it
        # In this case the fake fdiff works OK because the prior
        # is on |g|, so the sign doesn't matter

        lnp = self.get_lnprob_scalar2d(g1, g2)
        chi2 = -2 * lnp
        if chi2 < 0.0:
            chi2 = 0.0
        fdiffish = sqrt(chi2)
        return fdiffish

    def _get_fdiff_array(self, g1, g2):
        # For use with LM fitter, which requires
        #     (model-data)/width
        # so we need to fake it
        # In this case the fake fdiff works OK because the prior
        # is on |g|, so the sign doesn't matter
        lnp = self.get_lnprob_scalar2d(g1, g2)
        chi2 = -2 * lnp
        chi2.clip(min=0.0, max=None, out=chi2)
        fdiffish = sqrt(chi2)
        return fdiffish

    def get_lnprob_scalar2d(self, g1, g2):
        """
        Get the 2d log prob for the input g value
        """
        gsq = g1 * g1 + g2 * g2
        omgsq = 1.0 - gsq
        if omgsq <= 0.0:
            raise GMixRangeError("g^2 too big: %s" % gsq)
        lnp = 2 * log(omgsq) - 0.5 * gsq * self.sig2inv
        return lnp

    def get_prob_scalar2d(self, g1, g2):
        """
        Get the 2d prob for the input g value: (1-g^2)^2 * exp(-0.5*g^2/sigma^2)
        """
        p = 0.0

        gsq = g1 * g1 + g2 * g2
        omgsq = 1.0 - gsq
        if omgsq > 0.0:
            omgsq *= omgsq

            expval = exp(-0.5 * gsq * self.sig2inv)

            p = omgsq * expval

        return self.A * p

    def fill_prob_array2d(self, g1arr, g2arr, output):
        """
        Fill the output with the 2d prob for the input g value
        """
        gsq = g1arr * g1arr + g2arr * g2arr
        omgsq = 1.0 - gsq

        (w,) = where(omgsq > 0.0)
        if w.size > 0:
            omgsq *= omgsq

            expval = exp(-0.5 * gsq[w] * self.sig2inv)

            output[w] = self.A * omgsq[w] * expval

    def fill_lnprob_array2d(self, g1arr, g2arr, output):
        """
        Fill the output with the 2d prob for the input g value
        """
        gsq = g1arr * g1arr + g2arr * g2arr
        omgsq = 1.0 - gsq
        (w,) = where(omgsq > 0.0)
        if w.size > 0:
            output[w] = 2 * log(omgsq[w]) - 0.5 * gsq[w] * self.sig2inv

    def get_prob_scalar1d(self, g):
        """
        Get the 1d prior for the input |g| value
        """
        p = 0.0

        gsq = g * g
        omgsq = 1.0 - gsq

        if omgsq > 0.0:
            omgsq *= omgsq

            expval = numpy.exp(-0.5 * gsq * self.sig2inv)

            p = omgsq * expval

            p *= 2 * numpy.pi * g
        return self.A * p

    def fill_prob_array1d(self, g, output):
        """
        Fill the output with the 1d prior for the input g value
        """

        gsq = g * g
        omgsq = 1.0 - gsq

        (w,) = where(omgsq > 0.0)
        if w.size > 0:
            omgsq *= omgsq

            expval = exp(-0.5 * gsq[w] * self.sig2inv)

            output[w] = omgsq[w] * expval

            output[w] *= self.A * 2 * numpy.pi * g[w]

    def _get_guess(self, num, n=None):
        Aguess = 1.3 * num * (self.xdata[1] - self.xdata[0])
        cen = [Aguess, 0.16]

        if n is None:
            n = 1
            is_scalar = True
        else:
            is_scalar = False

        guess = zeros((n, 2))

        guess[:, 0] = cen[0] * (1.0 + 0.2 * srandu(n, rng=self.rng))
        guess[:, 1] = cen[1] * (1.0 + 0.2 * srandu(n, rng=self.rng))

        if is_scalar:
            guess = guess[0, :]
        return guess


############################################
# MRB: I've gone through this file to here
# I am going to stop and write tests, then split this module into pieces
# and then do docs for the rest.

class GPriorGreat3Exp(GPriorBase):
    """
    This doesn't fit very well
    """

    def __init__(self, pars=None, gmax=1.0, rng=None):
        PriorBase.__init__(self, rng=rng)

        self.gmax = float(gmax)

        self.npars = 3
        if pars is not None:
            self.set_pars(pars)

    def set_pars(self, pars):
        pars = array(pars)

        assert pars.size == self.npars, "npars must be 3"

        self.pars = pars

        self.A = pars[0]
        self.a = pars[1]
        self.g0 = pars[2]
        self.g0_sq = self.g0 ** 2

    def get_prob_scalar2d(self, g1, g2):
        """
        Get the 2d prob for the input g value
        """

        g = sqrt(g1 ** 2 + g2 ** 2)
        if g < 0.0 or g >= self.gmax:
            return 0.0

        return self._get_prob2d(g)

    def get_prob_array2d(self, g1, g2):
        """
        array input
        """

        n = g1.size
        p = zeros(n)

        g = sqrt(g1 ** 2 + g2 ** 2)
        (w,) = numpy.where((g >= 0) & (g < self.gmax))
        if w.size > 0:
            p[w] = self._get_prob2d(g[w])

        return p

    def get_prob_scalar1d(self, g):
        """
        Has the 2*pi*g in it
        """

        if g < 0.0 or g >= self.gmax:
            return 0.0

        return 2 * numpy.pi * g * self._get_prob2d(g)

    def get_prob_array1d(self, g):
        """
        has 2*pi*g in it array input
        """

        n = g.size
        p = zeros(n)

        (w,) = numpy.where((g >= 0.0) & (g < self.gmax))
        if w.size > 0:
            p[w] = 2 * numpy.pi * g[w] * self._get_prob2d(g[w])

        return p

    def get_lnprob_scalar2d(self, g1, g2):
        """
        Get the 2d prob for the input g value
        """

        p = self.get_prob_scalar2d(g1, g2)
        if p <= 0.0:
            raise GMixRangeError("g too big")

        lnp = numpy.log(p)
        return lnp

    def get_lnprob_array2d(self, g1arr, g2arr):
        """
        Get the 2d prior for the array inputs
        """

        g1arr = array(g1arr, dtype="f8", copy=False)
        g2arr = array(g2arr, dtype="f8", copy=False)

        output = numpy.zeros(g1arr.size) + LOWVAL

        g = sqrt(g1arr ** 2 + g2arr ** 2)
        (w,) = numpy.where(g < self.gmax)
        # print("kept:",w.size,"for prog calc")
        if w.size > 0:
            p = self._get_prob2d(g[w])
            (ww,) = numpy.where(p > 0.0)
            if ww.size > 0:
                output[w[ww]] = log(p[ww])

        return output

    def _get_prob2d(self, g):
        """
        no error checking

        no 2*pi*g
        """
        gsq = g ** 2
        omgsq = 1.0 - gsq
        omgsq_sq = omgsq * omgsq

        A = self.A
        a = self.a
        g0_sq = self.g0_sq
        gmax = self.gmax

        numer = A * (1 - exp((g - gmax) / a)) * omgsq_sq
        denom = (1 + g) * sqrt(gsq + g0_sq)

        prob = numer / denom

        return prob

    def _get_guess(self, num, n=None):
        rng = self.rng
        Aguess = 1.3 * num * (self.xdata[1] - self.xdata[0])
        cen = [Aguess, 1.64042, 0.0554674]

        if n is None:
            return array(cen)
        else:
            guess = zeros((n, self.npars))

            guess[:, 0] = cen[0] * (1.0 + 0.2 * srandu(n, rng=rng))
            guess[:, 1] = cen[1] * (1.0 + 0.2 * srandu(n, rng=rng))
            guess[:, 2] = cen[2] * (1.0 + 0.2 * srandu(n, rng=rng))

            return guess


class GPriorGreatDES(GPriorGreat3Exp):
    """
    """

    def __init__(self, pars=None, gmax=1.0, rng=None):
        PriorBase.__init__(self, rng=rng)

        self.gmax = float(gmax)

        self.npars = 4
        if pars is not None:
            self.set_pars(pars)

    def set_pars(self, pars):
        pars = array(pars)

        assert pars.size == self.npars, "npars must be 3"

        self.pars = pars

        self.A = pars[0]
        self.a = pars[1]
        self.g0 = pars[2]
        self.g0_sq = self.g0 ** 2

        self.index = pars[3]
        # self.index2=pars[4]

    def _get_prob2d(self, g):
        """
        no error checking

        no 2*pi*g
        """
        gsq = g ** 2

        A = self.A
        a = self.a
        g0_sq = self.g0_sq
        gmax = self.gmax

        omgsq = gmax ** 2 - gsq

        omgsq_p = omgsq ** 2

        arg = (g - gmax) / a
        numer = A * (1 - exp(arg)) * omgsq_p * (0.01 + g) ** self.index

        denom = gsq + g0_sq

        prob = numer / denom

        return prob

    def _get_guess(self, num, n=None):
        rng = self.rng

        Aguess = 1.3 * num * (self.xdata[1] - self.xdata[0])
        cen = [Aguess, 1.64042, 0.0554674, 0.5]

        if n is None:
            n = 1
            is_scalar = True
        else:
            is_scalar = False

        guess = zeros((n, self.npars))

        guess[:, 0] = cen[0] * (1.0 + 0.2 * srandu(n, rng=rng))
        guess[:, 1] = cen[1] * (1.0 + 0.2 * srandu(n, rng=rng))
        guess[:, 2] = cen[2] * (1.0 + 0.2 * srandu(n, rng=rng))
        guess[:, 3] = cen[3] * (1.0 + 0.2 * srandu(n, rng=rng))
        # guess[:,4] = cen[4]*(1.0 + 0.2*srandu(n,rng=rng))

        if is_scalar:
            guess = guess[0, :]
        return guess


class GPriorGreatDES2(GPriorGreat3Exp):
    """
    """

    def __init__(self, pars=None, rng=None):
        """
        [A, g0, gmax, gsigma]

        From Miller et al. multiplied by an erf
        """
        PriorBase.__init__(self, rng=rng)

        self.npars = 4
        if pars is not None:
            self.set_pars(pars)

    def set_pars(self, pars):
        self.A = pars[0]
        self.g0 = pars[1]
        self.gmax_func = pars[2]
        self.gsigma = pars[3]

        self.g0sq = self.g0 ** 2
        self.gmax = 1.0

    def _get_prob2d(self, g):
        """
        no error checking

        no 2*pi*g
        """
        from ._gmix import erf_array

        # numer1 = self.A*(1-exp( (g-1.0)/self.a ))

        gerf0 = zeros(g.size)
        arg = (self.gmax_func - g) / self.gsigma
        erf_array(arg, gerf0)

        gerf = 0.5 * (1.0 + gerf0)

        gerf *= self.A

        denom = (1 + g) * sqrt(g ** 2 + self.g0sq)

        model = gerf / denom

        return model

    def _get_guess(self, num, n=None):

        rng = self.rng

        Aguess = 1.3 * num * (self.xdata[1] - self.xdata[0])
        cen = [Aguess, 0.0793992, 0.706151, 0.124546]

        if n is None:
            n = 1
            is_scalar = True
        else:
            is_scalar = False

        guess = zeros((n, self.npars))

        guess[:, 0] = cen[0] * (1.0 + 0.2 * srandu(n, rng=rng))
        guess[:, 1] = cen[1] * (1.0 + 0.2 * srandu(n, rng=rng))
        guess[:, 2] = cen[2] * (1.0 + 0.2 * srandu(n, rng=rng))
        guess[:, 3] = cen[3] * (1.0 + 0.2 * srandu(n, rng=rng))

        if is_scalar:
            guess = guess[0, :]
        return guess


class GPriorGreatDESNoExp(GPriorGreat3Exp):
    """
    """

    def __init__(self, pars=None, gmax=1.0, rng=None):
        PriorBase.__init__(self, rng=rng)

        self.gmax = float(gmax)

        self.npars = 3
        if pars is not None:
            self.set_pars(pars)

    def set_pars(self, pars):
        pars = array(pars)

        assert pars.size == self.npars, "npars must be 3"

        self.pars = pars

        self.A = pars[0]
        self.g0 = pars[1]
        self.g0_sq = self.g0 ** 2

        self.index = pars[2]

    def _get_prob2d(self, g):
        """
        no error checking

        no 2*pi*g
        """
        gsq = g ** 2

        A = self.A
        g0_sq = self.g0_sq
        gmax = self.gmax

        omgsq = gmax ** 2 - gsq

        omgsq_p = omgsq ** 2

        numer = A * omgsq_p * (0.01 + g) ** self.index

        denom = gsq + g0_sq

        prob = numer / denom

        return prob

    def _get_guess(self, num, n=None):

        rng = self.rng

        Aguess = 1.3 * num * (self.xdata[1] - self.xdata[0])
        cen = [Aguess, 0.0554674, 0.5]

        if n is None:
            n = 1
            is_scalar = True
        else:
            is_scalar = False

        guess = zeros((n, self.npars))

        guess[:, 0] = cen[0] * (1.0 + 0.2 * srandu(n, rng=rng))
        guess[:, 1] = cen[1] * (1.0 + 0.2 * srandu(n, rng=rng))
        guess[:, 2] = cen[2] * (1.0 + 0.2 * srandu(n, rng=rng))

        if is_scalar:
            guess = guess[0, :]
        return guess


def make_gprior_great3_exp():
    """
    from fitting exp to the real galaxy deep data
    """
    pars = [0.0126484, 1.64042, 0.0554674]
    print("great3 simple joint exp g pars:", pars)
    return GPriorGreat3Exp(pars)


def make_gprior_great3_bdf():
    """
    from fitting bdf to the real galaxy deep data
    """
    pars = [0.014, 1.85, 0.058]
    print("great3 bdf joint bdf g pars:", pars)
    return GPriorGreat3Exp(pars)


def make_gprior_cosmos_exp():
    """
    From fitting exp to cosmos galaxies
    """
    pars = [560.0, 1.11, 0.052, 0.791]
    return GPriorM(pars)


def make_gprior_cosmos_dev():
    """
    From fitting devto cosmos galaxies
    """
    pars = [560.0, 1.28, 0.088, 0.887]
    return GPriorM(pars)


def make_gprior_cosmos_sersic(type="erf"):
    """
    Fitting to Lackner ellipticities, see
    espydir/cosmos/lackner_fits.py
        fit_sersic
    A:        0.0250834 +/- 0.00198797
    a:        1.98317 +/- 0.187965
    g0:       0.0793992 +/- 0.00256969
    gmax:     0.706151 +/- 0.00414383
    gsigma:   0.124546 +/- 0.00611789
    """
    if type == "spline":
        return GPriorCosmosSersicSpline()
    elif type == "erf":
        pars = [0.0250834, 1.98317, 0.0793992, 0.706151, 0.124546]
        # print("great3 sersic cgc g pars:",pars)
        return GPriorMErf(pars)
    else:
        raise ValueError("bad cosmos g prior: %s" % type)


#
# does not have the 2*pi*g in them
def _gprior2d_exp_scalar(A, a, g0sq, gmax, g, gsq):

    if g > gmax:
        return 0.0

    numer = A * (1 - numpy.exp((g - gmax) / a))
    denom = (1 + g) * numpy.sqrt(gsq + g0sq)

    prior = numer / denom

    return prior


def _gprior2d_exp_array(A, a, g0sq, gmax, g, gsq, output):

    (w,) = where(g < gmax)
    if w.size == 0:
        return

    numer = A * (1 - exp((g[w] - gmax) / a))
    denom = (1 + g) * sqrt(gsq[w] + g0sq)

    output[w] = numer / denom


def _gprior1d_exp_scalar(A, a, g0sq, gmax, g, gsq, output):
    (w,) = where(g < gmax)
    if w.size == 0:
        return

    numer = A * (1 - exp((g[w] - gmax) / a))
    denom = (1 + g) * sqrt(gsq[w] + g0sq)

    output[w] = numer / denom
    output[w] *= numpy.pi * g[w]


class GPriorM(GPriorBase):
    def __init__(self, pars, rng=None):
        """
        [A, a, g0, gmax]

        From Miller et al.
        """

        PriorBase.__init__(self, rng=rng)

        self.pars = pars

        self.A = pars[0]
        self.a = pars[1]
        self.g0 = pars[2]
        self.g0sq = self.g0 ** 2
        self.gmax = pars[3]

    def get_prob_scalar2d(self, g1, g2):
        """
        Get the 2d prob for the input g value
        """

        gsq = g1 ** 2 + g2 ** 2
        g = numpy.sqrt(gsq)
        return _gprior2d_exp_scalar(
            self.A, self.a, self.g0sq, self.gmax, g, gsq
        )

    def get_lnprob_scalar2d(self, g1, g2, submode=True):
        """
        Get the 2d prob for the input g value
        """

        gsq = g1 ** 2 + g2 ** 2
        g = numpy.sqrt(gsq)
        p = _gprior2d_exp_scalar(self.A, self.a, self.g0sq, self.gmax, g, gsq)

        if p <= 0.0:
            raise GMixRangeError("g too big: %s" % g)

        lnp = numpy.log(p)

        return lnp

    def fill_prob_array2d(self, g1arr, g2arr, output):
        """
        Fill the output with the 2d prob for the input g value
        """
        raise NotImplementedError("implement")

    def get_prob_scalar1d(self, g):
        """
        Get the 1d prior for the input |g| value
        """

        gsq = g ** 2

        p = _gprior2d_exp_scalar(self.A, self.a, self.g0sq, self.gmax, g, gsq)

        p *= 2 * numpy.pi * g

        return p

    def fill_prob_array1d(self, garr, output):
        """
        Fill the output with the 1d prob for the input g value
        """
        gsq = garr ** 2
        _gprior2d_exp_array(
            self.A, self.a, self.g0sq, self.gmax, garr, gsq, output
        )

        output *= 2 * numpy.pi * garr


class GPriorMErf(GPriorBase):
    def __init__(self, pars=None, rng=None):
        """
        [A, a, g0, gmax, gsigma]

        From Miller et al. multiplied by an erf
        """

        PriorBase.__init__(self, rng=rng)

        self.npars = 5
        if pars is not None:
            self.set_pars(pars)

    def set_pars(self, pars):
        self.A = pars[0]
        self.a = pars[1]
        self.g0 = pars[2]
        self.g0sq = self.g0 ** 2
        self.gmax_func = pars[3]
        self.gsigma = pars[4]

        self.gmax = 1.0

    def get_prob_scalar1d(self, g):
        """
        Get the prob for the input g value
        """

        if g < 0.0 or g >= 1.0:
            raise GMixRangeError("g out of range")

        return self._get_prob_nocheck_prefac_scalar(g)

    def get_prob_scalar2d(self, g1, g2):
        """
        Get the 2d prob for the input g1,g2 values
        """

        gsq = g1 ** 2 + g2 ** 2
        g = sqrt(gsq)

        if g < 0.0 or g >= 1.0:
            raise GMixRangeError("g out of range")

        return self._get_prob_nocheck_scalar(g)

    def get_lnprob_scalar2d(self, g1, g2):
        """
        Get the 2d log prob for the input g1,g2 values
        """

        p = self.get_prob_scalar2d(g1, g2)
        return log(p)

    def get_prob_array1d(self, g):
        """
        Get the prob for the input g values
        """
        g = array(g, dtype="f8", ndmin=1)

        p = zeros(g.size)
        (w,) = where((g >= 0.0) & (g < 1.0))
        if w.size > 0:
            p[w] = self._get_prob_nocheck_prefac_array(g[w])
        return p

    def get_prob_array2d(self, g1, g2):
        """
        Get the 2d prob for the input g1,g2 values
        """

        g1 = array(g1, dtype="f8", ndmin=1)
        g2 = array(g2, dtype="f8", ndmin=1)

        gsq = g1 ** 2 + g2 ** 2
        g = sqrt(gsq)

        p = zeros(g1.size)
        (w,) = where((g >= 0.0) & (g < 1.0))
        if w.size > 0:
            p[w] = self._get_prob_nocheck_array(g[w])
        return p

    def get_lnprob_array2d(self, g1, g2):
        """
        Get the 2d log prob for the input g1,g2 values
        """

        g1 = array(g1, dtype="f8", ndmin=1)
        g2 = array(g2, dtype="f8", ndmin=1)

        gsq = g1 ** 2 + g2 ** 2
        g = sqrt(gsq)

        lnp = zeros(g1.size) + LOWVAL
        (w,) = where((g >= 0.0) & (g < 1.0))
        if w.size > 0:
            p = self._get_prob_nocheck_array(g[w])
            lnp[w] = log(p)
        return lnp

    def _get_prob_nocheck_prefac_scalar(self, g):
        """
        With the 2*pi*g
        """
        return 2 * pi * g * self._get_prob_nocheck_scalar(g)

    def _get_prob_nocheck_prefac_array(self, g):
        """
        With the 2*pi*g, must be an array
        """
        return 2 * pi * g * self._get_prob_nocheck_array(g)

    def _get_prob_nocheck_scalar(self, g):
        """
        workhorse function, must be scalar. Error checking
        is done in the argument parsing to erf

        this does not include the 2*pi*g
        """
        # from ._gmix import erf
        from math import erf

        # numer1 = 2*pi*g*self.A*(1-exp( (g-1.0)/self.a ))
        numer1 = self.A * (1 - exp((g - 1.0) / self.a))

        arg = (self.gmax_func - g) / self.gsigma
        gerf = 0.5 * (1.0 + erf(arg))
        numer = numer1 * gerf

        denom = (1 + g) * sqrt(g ** 2 + self.g0sq)

        model = numer / denom

        return model

    def _get_prob_nocheck_array(self, g):
        """
        workhorse function.  No error checking, must be an array

        this does not include the 2*pi*g
        """
        # from ._gmix import erf_array
        from math import erf

        # numer1 = 2*pi*g*self.A*(1-exp( (g-1.0)/self.a ))
        numer1 = self.A * (1 - exp((g - 1.0) / self.a))

        gerf0 = zeros(g.size)
        arg = (self.gmax_func - g) / self.gsigma
        for i in range(g.size):
            gerf0[i] = erf(arg[i])
        # erf_array(arg, gerf0)

        gerf = 0.5 * (1.0 + gerf0)

        numer = numer1 * gerf

        denom = (1 + g) * sqrt(g ** 2 + self.g0sq)

        model = numer / denom

        return model

    def _get_guess(self, num, n=None):
        Aguess = 1.3 * num * (self.xdata[1] - self.xdata[0])
        cen = [Aguess, 1.98317, 0.0793992, 0.706151, 0.124546]

        if n is None:
            n = 1
            is_scalar = True
        else:
            is_scalar = False

        guess = zeros((n, self.npars))

        guess[:, 0] = cen[0] * (1.0 + 0.2 * srandu(n))
        guess[:, 1] = cen[1] * (1.0 + 0.2 * srandu(n))
        guess[:, 2] = cen[2] * (1.0 + 0.2 * srandu(n))
        guess[:, 3] = cen[3] * (1.0 + 0.2 * srandu(n))
        guess[:, 4] = cen[4] * (1.0 + 0.2 * srandu(n))

        if is_scalar:
            guess = guess[0, :]
        return guess


class GPriorMErf2(GPriorMErf):
    def __init__(self, pars=None, rng=None):
        """
        [A, g0, gmax, gsigma]
        """
        PriorBase.__init__(self, rng=rng)

        self.npars = 4
        if pars is not None:
            self.set_pars(pars)

    def set_pars(self, pars):
        self.A = pars[0]
        self.g0 = pars[1]
        self.g0sq = self.g0 ** 2
        self.gmax_func = pars[2]
        self.gsigma = pars[3]

        self.gmax = 1.0
        self.a = 4500.0

    def _get_prob_nocheck_scalar(self, g):
        """
        workhorse function, must be scalar. Error checking
        is done in the argument parsing to erf

        this does not include the 2*pi*g
        """
        from ._gmix import erf

        # a fixed now
        numer1 = self.A * (1 - exp((g - 1.0) / self.a))

        arg = (self.gmax_func - g) / self.gsigma
        gerf = 0.5 * (1.0 + erf(arg))
        numer = numer1 * gerf

        denom = (1 + g) * sqrt(g ** 2 + self.g0sq)

        model = numer / denom

        return model

    def _get_prob_nocheck_array(self, g):
        """
        workhorse function.  No error checking, must be an array

        this does not include the 2*pi*g
        """
        from ._gmix import erf_array

        # a fixed now
        numer1 = self.A * (1 - exp((g - 1.0) / self.a))

        gerf0 = zeros(g.size)
        arg = (self.gmax_func - g) / self.gsigma
        erf_array(arg, gerf0)

        gerf = 0.5 * (1.0 + gerf0)

        numer = numer1 * gerf

        denom = (1 + g) * sqrt(g ** 2 + self.g0sq)

        model = numer / denom

        return model

    def _get_guess(self, num, n=None):
        rng = self.rng

        Aguess = 1.3 * num * (self.xdata[1] - self.xdata[0])
        cen = [Aguess, 0.0793992, 0.706151, 0.124546]

        if n is None:
            n = 1
            is_scalar = True
        else:
            is_scalar = False

        guess = zeros((n, self.npars))

        guess[:, 0] = cen[0] * (1.0 + 0.2 * srandu(n, rng=rng))
        guess[:, 1] = cen[1] * (1.0 + 0.2 * srandu(n, rng=rng))
        guess[:, 2] = cen[2] * (1.0 + 0.2 * srandu(n, rng=rng))
        guess[:, 3] = cen[3] * (1.0 + 0.2 * srandu(n, rng=rng))

        if is_scalar:
            guess = guess[0, :]
        return guess


class FlatEtaPrior(PriorBase):
    def __init__(self):
        self.max_eta_sq = 9.0 ** 2 + 9.0 ** 2

    def get_prob_scalar2d(self, eta1, eta2):
        """
        Get the 2d log prob
        """
        eta_sq = eta1 ** 2 + eta2 ** 2
        if eta_sq > self.max_eta_sq:
            raise GMixRangeError("eta^2 too big: %s" % eta_sq)

        return 1.0

    def get_lnprob_scalar2d(self, eta1, eta2):
        """
        Get the 2d log prob
        """
        eta_sq = eta1 ** 2 + eta2 ** 2
        if eta_sq > self.max_eta_sq:
            raise GMixRangeError("eta^2 too big: %s" % eta_sq)

        return 0.0


class GPriorCosmosSersicSpline(GPriorMErf):
    def __init__(self, rng=None):
        PriorBase.__init__(self, rng=rng)
        self.gmax = 1.0

    def _get_prob_nocheck(self, g):
        """
        Remove the 2*pi*g
        """
        g_arr = numpy.atleast_1d(g)
        (w,) = where(g_arr > 0)
        if w.size != g_arr.size:
            raise GMixRangeError("zero g found")

        p_prefac = self._get_prob_nocheck_prefac(g_arr)
        p = zeros(g_arr.size)
        p = p_prefac / (2.0 * pi * g_arr)
        return p

    def _get_prob_nocheck_prefac(self, g):
        """
        Input should be in bounds and an array
        """
        from scipy.interpolate import fitpack

        g_arr = numpy.atleast_1d(g)
        p, err = fitpack._fitpack._spl_(
            g_arr, 0, _g_cosmos_t, _g_cosmos_c, _g_cosmos_k, 0
        )
        if err:
            raise RuntimeError("error occurred in g interp")

        if not isinstance(g, numpy.ndarray):
            return p[0]
        else:
            return p
