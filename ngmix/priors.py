"""
Convention is that all priors should have peak ln(prob)==0. This
helps use in priors for LM fitting
"""
import math

import numpy
from numpy import where, array, exp, log, sqrt, cos, sin, zeros, diag
from numpy import pi

from .gexceptions import GMixRangeError

LOWVAL = -numpy.inf
BIGVAL = 9999.0e47


def make_rng(rng=None):
    """
    if the input rng is None, create a new RandomState
    but with a seed generated from current numpy state
    """
    if rng is None:
        seed = numpy.random.randint(0, 2 ** 30)
        rng = numpy.random.RandomState(seed)

    return rng


class PriorBase(object):
    """
    Base object for priors.

    parameters
    ----------
    bounds: 2-tuple of floats or None
        The bounds of the parameter. Default of None means no bounds.
    rng: np.random.RandomState or None
        An RNG to use. If None, a new RNG is made using the numpy global RNG
        to generate a seed.

    attributes
    ----------
    bounds: 2-tuple of floats or None
        The bounds of the parameter. Default of None means no bounds.
    rng: np.random.RandomState or None
        An RNG to use. If None, a new RNG is made using the numpy global RNG
        to generate a seed.

    methods
    -------
    has_bounds()
        Returns True if the object has bounds defined and they are non-None, False
        otherwise.
    """
    def __init__(self, bounds=None, rng=None):
        self.bounds = bounds
        self.rng = make_rng(rng=rng)

    def has_bounds(self):
        """
        returns True if the object has a bounds defined, False otherwise.
        """
        return hasattr(self, "bounds") and self.bounds is not None


class GPriorBase(PriorBase):
    """
    Base object for priors on shear.

    Note that depending on your purpose, you may need to override the following
    abstract methods:

        fill_prob_array1d
        fill_lnprob_array2d
        fill_prob_array2d
        get_lnprob_scalar2d
        get_prob_scalar2d
        get_prob_scalar1d

    parameters
    ----------
    pars: float or array-like
        Parameters for the prior.
    rng: np.random.RandomState or None
        An RNG to use. If None, a new RNG is made using the numpy global RNG
        to generate a seed.

    attributes
    ----------
    pars: float or array-like
        Parameters for the prior.
    gmax: float
        The maximum value of the shear.

    methods
    -------
    fill_prob_array1d(g, output)
        Fill the `output` array with the prob values at each `g`.
    fill_lnprob_array2d(g1arr, g2arr, output)
        Fill the `output` array with the lnprob values at each `g1`, `g2` pair.
    fill_prob_array2d(g1arr, g2arr, output)
        Fill the `output` array with the prob values at each `g1`, `g2` pair.
    get_lnprob_scalar2d(g1, g2)
        Get the 2d log prob
    get_lnprob_array2d(g1arr, g2arr)
        Get the 2d prior for the array inputs
    get_prob_scalar2d(g1, g2)
        Get the 2d prob
    get_prob_array2d(g1arr, g2arr)
        Get the 2d prior for the array inputs
    get_prob_scalar1d(g)
        Get the 1d prob
    get_prob_array1d(garr)
        Get the 1d prior for the array inputs
    sample1d(nrand, maxguess=0.1)
        Get random |g| from the 1d distribution
    sample2d(nrand=None, maxguess=0.1)
        Get random g1,g2 values by first drawing from the 1-d distribution
    sample2d_brute(nrand)
        Get random g1,g2 values using 2-d brute force method
    set_maxval1d_scipy()
        Use a simple minimizer to find the max value of the 1d distribution
    set_maxval1d(maxguess=0.1)
        Use a simple minimizer to find the max value of the 1d distribution
    get_prob_scalar1d_neg(g, *args)
        Helper function for the minimizer.
    dofit(xdata, ydata, guess=None, show=False)
        Fit the prior to data.
    """
    def __init__(self, pars, rng=None):
        PriorBase.__init__(self, rng=rng)

        self.pars = array(pars, dtype="f8")

        # sub-class may want to over-ride this, see GPriorM
        self.gmax = 1.0

    def fill_prob_array1d(self, g, output):
        """
        Fill the `output` array with the prob values at each `g`.
        """
        raise RuntimeError("over-ride me")

    def fill_lnprob_array2d(self, g1arr, g2arr, output):
        """
        Fill the `output` array with the lnprob values at each `g1`, `g2` pair.
        """
        raise RuntimeError("over-ride me")

    def fill_prob_array2d(self, g1arr, g2arr, output):
        """
        Fill the `output` array with the prob values at each `g1`, `g2` pair.
        """
        raise RuntimeError("over-ride me")

    def get_lnprob_scalar2d(self, g1, g2):
        """
        Get the 2d log prob
        """
        raise RuntimeError("over-ride me")

    def get_lnprob_array2d(self, g1arr, g2arr):
        """
        Get the 2d prior for the array inputs
        """

        g1arr = array(g1arr, dtype="f8", copy=False)
        g2arr = array(g2arr, dtype="f8", copy=False)

        output = numpy.zeros(g1arr.size) + LOWVAL
        self.fill_lnprob_array2d(g1arr, g2arr, output)
        return output

    def get_prob_scalar2d(self, g1, g2):
        """
        Get the 2d prob
        """
        raise RuntimeError("over-ride me")

    def get_prob_array2d(self, g1arr, g2arr):
        """
        Get the 2d prior for the array inputs
        """

        g1arr = array(g1arr, dtype="f8", copy=False)
        g2arr = array(g2arr, dtype="f8", copy=False)

        output = numpy.zeros(g1arr.size)
        self.fill_prob_array2d(g1arr, g2arr, output)
        return output

    def get_prob_scalar1d(self, g):
        """
        Get the 1d prob
        """
        raise RuntimeError("over-ride me")

    def get_prob_array1d(self, garr):
        """
        Get the 1d prior for the array inputs
        """

        garr = array(garr, dtype="f8", copy=False)

        output = numpy.zeros(garr.size)
        self.fill_prob_array1d(garr, output)
        return output

    def sample1d(self, nrand, maxguess=0.1):
        """
        Get random |g| from the 1d distribution

        Set self.gmax appropriately

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
        rng = self.rng

        if not hasattr(self, "maxval1d"):
            self.set_maxval1d(maxguess=maxguess)

        maxval1d = self.maxval1d * 1.1

        # don't go right up to the end
        # gmax=self.gmax
        gmax = self.gmax - 1.0e-4

        g = numpy.zeros(nrand)

        ngood = 0
        nleft = nrand
        while ngood < nrand:

            # generate total g in [0,gmax)
            grand = gmax * rng.uniform(size=nleft)

            # now the height from [0,maxval)
            h = maxval1d * rng.uniform(size=nleft)

            pvals = self.get_prob_array1d(grand)

            (w,) = numpy.where(h < pvals)
            if w.size > 0:
                g[ngood:ngood + w.size] = grand[w]
                ngood += w.size
                nleft -= w.size

        return g

    def sample2d(self, nrand=None, maxguess=0.1):
        """
        Get random g1,g2 values by first drawing
        from the 1-d distribution and assuming rotational symmetry.

        parameters
        ----------
        nrand: int
            Number to generate
        maxguess: float
            The guess for finding the maximum g value if it is needed.

        returns
        -------
        g1: array-like
            The generated g1 values.
        g2: array-like
            The generated g2 values.
        """
        rng = self.rng

        if nrand is None:
            nrand = 1
            is_scalar = True
        else:
            is_scalar = False

        grand = self.sample1d(nrand, maxguess=maxguess)
        theta = rng.uniform(size=nrand) * 2 * numpy.pi
        twotheta = 2 * theta
        g1rand = grand * numpy.cos(twotheta)
        g2rand = grand * numpy.sin(twotheta)

        if is_scalar:
            g1rand = g1rand[0]
            g2rand = g2rand[0]

        return g1rand, g2rand

    def sample2d_brute(self, nrand):
        """
        Get random g1,g2 values using 2-d brute
        force method

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
        rng = self.rng

        maxval2d = self.get_prob_scalar2d(0.0, 0.0)
        g1, g2 = numpy.zeros(nrand), numpy.zeros(nrand)

        ngood = 0
        nleft = nrand
        while ngood < nrand:

            # generate on cube [-1,1,h]
            g1rand = srandu(nleft, rng=rng)
            g2rand = srandu(nleft, rng=rng)

            # a bit of padding since we are modifying the distribution
            h = maxval2d * rng.uniform(size=nleft)

            vals = self.get_prob_array2d(g1rand, g2rand)

            (w,) = numpy.where(h < vals)
            if w.size > 0:
                g1[ngood:ngood + w.size] = g1rand[w]
                g2[ngood:ngood + w.size] = g2rand[w]
                ngood += w.size
                nleft -= w.size

        return g1, g2

    def set_maxval1d_scipy(self):
        """
        Use a simple minimizer to find the max value of the 1d distribution
        """
        import scipy.optimize

        (minvalx, fval, iterations, fcalls, warnflag) = scipy.optimize.fmin(
            self.get_prob_scalar1d_neg, 0.1, full_output=True, disp=False
        )
        if warnflag != 0:
            raise RuntimeError("failed to find min: warnflag %d" % warnflag)
        self.maxval1d = -fval
        self.maxval1d_loc = minvalx

    def set_maxval1d(self, maxguess=0.1):
        """
        Use a simple minimizer to find the max value of the 1d distribution

        parameters
        ----------
        maxguess: float
            The guess for finding the maximum g value if it is needed.
        """
        from .simplex import minimize_neldermead

        res = minimize_neldermead(
            self.get_prob_scalar1d_neg, maxguess, maxiter=4000, maxfev=4000
        )

        if res["status"] != 0:
            raise RuntimeError("failed to find min, flags: %d" % res["status"])

        self.maxval1d = -res["fun"]
        self.maxval1d_loc = res["x"]

        # print("maxguess:",maxguess)
        # print("maxloc:",self.maxval1d_loc)
        # print(res)

    def get_prob_scalar1d_neg(self, g, *args):
        """
        Helper function so we can use the minimizer
        """
        return -self.get_prob_scalar1d(g)

    def dofit(self, xdata, ydata, guess=None, show=False):
        """
        Fit the prior to data.

        parameters
        ----------
        xdata: array-like
            The x-values for the fit. Usually values of |g|.
        ydata: array-like
            The y-values for the fit. Usually values of p(|g|).
        guess: array-like or None
            The guess for the fitter. If you pass None, you will need to
            implement `_get_guess`.
        show: bool, optional
            If True, show a plot of the fit and data.
        """
        from .fitting import run_leastsq
        from .fitting import print_pars

        (w,) = where(ydata > 0)
        self.xdata = xdata[w]
        self.ydata = ydata[w]
        self.ierr = 1.0 / sqrt(self.ydata)

        if guess is None:
            guess = self._get_guess(self.ydata.sum())

        res = run_leastsq(self._calc_fdiff, guess, 0, maxfev=4000)

        self.fit_pars = res["pars"]
        self.fit_pars_cov = res["pars_cov"]
        self.fit_perr = res["pars_err"]

        print("flags:", res["flags"], "nfev:", res["nfev"])
        print_pars(res["pars"], front="pars: ")
        print_pars(res["pars_err"], front="perr: ")

        c = ["%g" % p for p in res["pars"]]
        c = "[" + ", ".join(c) + "]"
        print(c)

        if show:
            self._compare_fit(res["pars"])

    def _calc_fdiff(self, pars):
        # helper function for the fitter
        self.set_pars(pars)
        p = self.get_prob_array1d(self.xdata)
        fdiff = (p - self.ydata) * self.ierr
        return fdiff

    def _compare_fit(self, pars):
        import biggles

        self.set_pars(pars)

        plt = biggles.plot(self.xdata, self.ydata, visible=False)

        xvals = numpy.linspace(0.0, 1.0, 1000)
        p = self.get_prob_array1d(xvals)

        plt.add(biggles.Curve(xvals, p, color="red"))
        plt.show()


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


class FlatPrior(PriorBase):
    """
    A flat prior between `minval` and `maxval`.

    parameters
    ----------
    minval: float
        The minimum value of the allowed range.
    maxval: float
        The maximum value of the allowed range.
    rng: np.random.RandomState or None
        An RNG to use. If None, a new RNG is made using the numpy global RNG
        to generate a seed.
    """
    def __init__(self, minval, maxval, rng=None):
        PriorBase.__init__(self, rng=rng)

        self.minval = minval
        self.maxval = maxval

    def get_prob_scalar(self, val):
        """
        Returns 1 if the value is in [minval, maxval] or raises a GMixRangeError
        """
        retval = 1.0
        if val < self.minval or val > self.maxval:
            raise GMixRangeError(
                "value %s out of range: "
                "[%s,%s]" % (val, self.minval, self.maxval)
            )
        return retval

    def get_lnprob_scalar(self, val):
        """
        Returns 0.0 if the value is in [minval, maxval] or raises a GMixRangeError
        """
        retval = 0.0
        if val < self.minval or val > self.maxval:
            raise GMixRangeError(
                "value %s out of range: "
                "[%s,%s]" % (val, self.minval, self.maxval)
            )
        return retval

    def get_fdiff(self, val):
        """
        Compute sqrt(-2ln(p)) ~ (data - mode)/err for using with LM fitters.
        """
        retval = 0.0
        if val < self.minval or val > self.maxval:
            raise GMixRangeError(
                "value %s out of range: "
                "[%s,%s]" % (val, self.minval, self.maxval)
            )
        return retval

    def sample(self, nrand=None, n=None):
        """
        Returns samples uniformly on the interval.

        parameters
        ----------
        nrand: int or None
            The number of samples. If None, a single scalar sample is drawn.
            Default is None.

        returns
        -------
        samples: scalar or array-like
            The samples with shape (`nrand`,). If `nrand` is None, then a
            scalar is returned.
        """
        if n is None and nrand is not None:
            # if they have given nrand and not n, use that
            # this keeps the API the same but allows ppl to use the new API of nrand
            n = nrand

        if n is None:
            is_scalar = True
            n = 1
        else:
            is_scalar = False

        rvals = self.rng.uniform(size=n)
        rvals = self.minval + (self.maxval - self.minval) * rvals

        if is_scalar:
            rvals = rvals[0]

        return rvals


class TwoSidedErf(PriorBase):
    """
    A two-sided error function that evaluates to 1 in the middle, zero at
    extremes.

    A limitation seems to be the accuracy of the erf implementation.

    parameters
    ----------
    minval: float
        The minimum value. This is where p(x) = 0.5 at the lower end.
    width_at_min: float
        The width of the transition region from 0 to 1 at the lower end.
    maxval: float
        The maximum value. This is where p(x) = 0.5 at the upper end.
    width_at_max: float
        The width of the transition region from 1 to 0 at the upper end.
    rng: np.random.RandomState or None
        An RNG to use. If None, a new RNG is made using the numpy global RNG
        to generate a seed.
    """
    def __init__(self, minval, width_at_min, maxval, width_at_max, rng=None):
        PriorBase.__init__(self, rng=rng)

        self.minval = minval
        self.width_at_min = width_at_min

        self.maxval = maxval
        self.width_at_max = width_at_max

    def get_prob_scalar(self, val):
        """
        get the probability of the point
        """
        from math import erf

        p1 = 0.5 * erf((self.maxval - val) / self.width_at_max)
        p2 = 0.5 * erf((val - self.minval) / self.width_at_min)

        return p1 + p2

    def get_lnprob_scalar(self, val):
        """
        get the log probability of the point
        """

        p = self.get_prob_scalar(val)

        if p <= 0.0:
            lnp = LOWVAL
        else:
            lnp = log(p)
        return lnp

    def get_prob_array(self, vals):
        """
        get the probability of a set of points
        """

        vals = array(vals, ndmin=1, dtype="f8", copy=False)
        pvals = zeros(vals.size)

        for i in range(vals.size):
            pvals[i] = self.get_prob_scalar(vals[i])

        return pvals

    def get_lnprob_array(self, vals):
        """
        get the log probability of a set of points
        """

        p = self.get_prob_array(vals)

        lnp = numpy.zeros(p.size) + LOWVAL
        (w,) = numpy.where(p > 0.0)
        if w.size > 0:
            lnp[w] = numpy.log(p[w])
        return lnp

    def get_fdiff(self, x):
        """
        Compute sqrt(-2ln(p)) ~ (data - mode)/err for using with LM fitters.
        """
        if isinstance(x, numpy.ndarray):
            return self._get_fdiff_array(x)
        else:
            return self._get_fdiff_scalar(x)

    def _get_fdiff_array(self, vals):
        vals = array(vals, ndmin=1, dtype="f8", copy=False)
        fdiff = zeros(vals.size)

        for i in range(vals.size):
            fdiff[i] = self.get_fdiff_scalar(vals[i])

        return fdiff

    def _get_fdiff_scalar(self, val):
        # get something similar to a (model-data)/err.  Note however that with
        # the current implementation, the *sign* of the difference is lost in
        # this case.

        p = self.get_lnprob_scalar(val)

        p = -2 * p
        if p < 0.0:
            p = 0.0
        return sqrt(p)

    def sample(self, nrand=None):
        """
        Draw random samples of the prior.

        Note this function is not perfect in that it only goes from
        -5,5 sigma past each side.

        parameters
        ----------
        nrand: int or None
            The number of samples. If None, a single scalar sample is drawn.
            Default is None.

        returns
        -------
        samples: scalar or array-like
            The samples with shape (`nrand`,). If `nrand` is None, then a
            scalar is returned.
        """
        rng = self.rng

        if nrand is None:
            nrand = 1
            is_scalar = True
        else:
            is_scalar = False

        xmin = self.minval - 5.0 * self.width_at_min
        xmax = self.maxval + 5.0 * self.width_at_max

        rvals = zeros(nrand)

        ngood = 0
        nleft = nrand
        while ngood < nrand:
            randx = rng.uniform(low=xmin, high=xmax, size=nleft)

            pvals = self.get_prob_array(randx)
            randy = rng.uniform(size=nleft)

            (w,) = where(randy < pvals)
            if w.size > 0:
                rvals[ngood:ngood + w.size] = randx[w]
                ngood += w.size
                nleft -= w.size

        if is_scalar:
            rvals = rvals[0]

        return rvals


############################################
# MRB: I've gone through the first 1k lines.
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


class Normal(PriorBase):
    """
    A Normal distribution.

    This class provides an interface consistent with LogNormal
    """

    def __init__(self, mean, sigma, bounds=None, rng=None):
        super(Normal, self).__init__(rng=rng, bounds=bounds)

        self.mean = mean
        self.sigma = sigma
        self.sinv = 1.0 / sigma
        self.s2inv = 1.0 / sigma ** 2
        self.ndim = 1

    def get_lnprob(self, x):
        """
        -0.5 * ( (x-mean)/sigma )**2
        """
        diff = self.mean - x
        return -0.5 * diff * diff * self.s2inv

    get_lnprob_scalar = get_lnprob
    get_lnprob_array = get_lnprob

    def get_prob(self, x):
        """
        -0.5 * ( (x-mean)/sigma )**2
        """
        diff = self.mean - x
        lnp = -0.5 * diff * diff * self.s2inv
        return numpy.exp(lnp)

    get_prob_array = get_prob

    def get_prob_scalar(self, x):
        """
        -0.5 * ( (x-mean)/sigma )**2
        """
        from math import exp

        diff = self.mean - x
        lnp = -0.5 * diff * diff * self.s2inv
        return exp(lnp)

    def get_fdiff(self, x):
        """
        For use with LM fitter
        (model-data)/width for both coordinates
        """
        return (x - self.mean) * self.sinv

    def sample(self, size=None):
        """
        Get samples.  Send no args to get a scalar.
        """
        return self.rng.normal(loc=self.mean, scale=self.sigma, size=size,)


class LMBounds(PriorBase):
    """
    class to hold simple bounds for the leastsqbound version
    of LM.

    The fdiff is always zero, but the bounds  will be sent
    to the minimizer
    """

    def __init__(self, minval, maxval, rng=None):

        super(LMBounds, self).__init__(rng=rng)

        self.bounds = (minval, maxval)
        self.mean = (minval + maxval) / 2.0
        self.sigma = (maxval - minval) * 0.28

    def get_fdiff(self, val):
        """
        fdiff for the input val, always zero
        """
        return 0.0 * val

    def sample(self, n=None):
        """
        returns samples uniformly on the interval
        """

        return self.rng.uniform(
            low=self.bounds[0], high=self.bounds[1], size=n,
        )


class Bounded1D(PriorBase):
    """
    wrap a pdf and limit samples to the input bounds
    """

    def __init__(self, pdf, bounds):
        self.pdf = pdf
        self.bounds = bounds
        assert len(bounds) == 2, "bounds must be length 2"

    def sample(self, size=None):

        bounds = self.bounds

        if size is None:
            nval = 1
        else:
            nval = size

        values = numpy.zeros(nval)
        ngood = 0
        nleft = nval

        while nleft > 0:
            tmp = self.pdf.sample(nleft)

            (w,) = numpy.where((tmp > bounds[0]) & (tmp < bounds[1]))

            if w.size > 0:
                values[ngood:ngood + w.size] = tmp[w]

                ngood += w.size
                nleft -= w.size

        if size is None:
            values = values[0]
        return values


class LogNormal(PriorBase):
    """
    Lognormal distribution

    parameters
    ----------
    mean:
        such that <x> in linear space is mean.  This implies the mean in log(x)
        is
            <log(x)> = log(mean) - 0.5*log( 1 + sigma**2/mean**2 )
    sigma:
        such than the variace in linear space is sigma**2.  This implies
        the variance in log space is
            var(log(x)) = log( 1 + sigma**2/mean**2 )
    norm: optional
        When calling eval() the return value will be norm*prob(x)


    methods
    -------
    sample(nrand):
        Get nrand random deviates from the distribution
    lnprob(x):
        Get the natural logarithm of the probability of x.  x can
        be an array
    prob(x):
        Get the probability of x.  x can be an array
    """

    def __init__(self, mean, sigma, shift=None, rng=None):
        PriorBase.__init__(self, rng=rng)

        if mean <= 0:
            raise ValueError("mean %s is < 0" % mean)

        self.shift = shift
        self.mean = mean
        self.sigma = sigma

        logmean = numpy.log(self.mean) - 0.5 * numpy.log(
            1 + self.sigma ** 2 / self.mean ** 2
        )
        logvar = numpy.log(1 + self.sigma ** 2 / self.mean ** 2)
        logsigma = numpy.sqrt(logvar)
        logivar = 1.0 / logvar

        self.logmean = logmean
        self.logvar = logvar
        self.logsigma = logsigma
        self.logivar = logivar

        log_mode = self.logmean - self.logvar
        self.mode = numpy.exp(log_mode)
        chi2 = self.logivar * (log_mode - self.logmean) ** 2

        # subtract mode to make max 0.0
        self.lnprob_max = -0.5 * chi2 - log_mode

        self.log_mode = log_mode

    def get_lnprob_scalar(self, x):
        """
        This one has error checking
        """

        if self.shift is not None:
            x = x - self.shift

        if x <= 0:
            raise GMixRangeError("values of x must be > 0")

        logx = numpy.log(x)
        chi2 = self.logivar * (logx - self.logmean) ** 2

        # subtract mode to make max 0.0
        lnprob = -0.5 * chi2 - logx - self.lnprob_max

        return lnprob

    def get_lnprob_array(self, x):
        """
        This one no error checking
        """

        x = numpy.array(x, dtype="f8", copy=False)
        if self.shift is not None:
            x = x - self.shift

        (w,) = where(x <= 0)
        if w.size > 0:
            raise GMixRangeError("values of x must be > 0")

        logx = numpy.log(x)
        chi2 = self.logivar * (logx - self.logmean) ** 2

        # subtract mode to make max 0.0
        lnprob = -0.5 * chi2 - logx - self.lnprob_max

        return lnprob

    def get_prob_scalar(self, x):
        """
        Get the probability of x.
        """

        lnprob = self.get_lnprob_scalar(x)
        return numpy.exp(lnprob)

    def get_prob_array(self, x):
        """
        Get the probability of x.  x can be an array
        """

        lnp = self.get_lnprob_array(x)
        return numpy.exp(lnp)

    def get_fdiff(self, x):
        """
        this is kind of hokey
        """
        lnp = self.get_lnprob_scalar(x)
        chi2 = -2 * lnp
        if chi2 < 0.0:
            chi2 = 0.0
        fdiff = sqrt(chi2)
        return fdiff

    def sample(self, nrand=None):
        """
        Get nrand random deviates from the distribution

        If z is drawn from a normal random distribution, then
        exp(logmean+logsigma*z) is drawn from lognormal
        """
        z = self.rng.normal(size=nrand)
        r = numpy.exp(self.logmean + self.logsigma * z)

        if self.shift is not None:
            r += self.shift

        return r

    def sample_brute(self, nrand=None, maxval=None):
        """
        Get nrand random deviates from the distribution using brute force

        This is really to check that our probabilities are being calculated
        correctly, by comparing to the regular sampler
        """

        rng = self.rng

        if maxval is None:
            maxval = self.mean + 10 * self.sigma

        if nrand is None:
            is_scalar = True
            nrand = 1
        else:
            is_scalar = False

        samples = numpy.zeros(nrand)

        ngood = 0
        nleft = nrand
        while ngood < nrand:

            rvals = maxval * rng.rand(nleft)

            # between 0,1 which is max for prob
            h = rng.uniform(size=nleft)

            pvals = self.get_prob_array(rvals)

            (w,) = numpy.where(h < pvals)
            if w.size > 0:
                samples[ngood:ngood + w.size] = rvals[w]
                ngood += w.size
                nleft -= w.size

        if is_scalar:
            samples = samples[0]

        if self.shift is not None:
            samples += self.shift

        return samples

    def _calc_fdiff(self, pars):
        try:
            ln = LogNormal(pars[0], pars[1])
            model = ln.get_prob_array(self._fitx) * pars[2]
        except (GMixRangeError, ValueError):
            return self._fity * 0 - numpy.inf

        fdiff = model - self._fity
        return fdiff

    def fit(self, x, y):
        """
        fit to the input x and y
        """
        from .fitting import run_leastsq

        rng = self.rng

        self._fitx = x
        self._fity = y

        for i in range(4):
            f1, f2, f3 = 1.0 + rng.uniform(low=0.1, high=0.1, size=3)
            guess = numpy.array([x.mean() * f1, x.std() * f2, y.mean() * f3])

            res = run_leastsq(self._calc_fdiff, guess, 0,)
            if res["flags"] == 0:
                break

        return res


class MultivariateLogNormal(object):
    """
    multi-variate log-normal distribution

    parameters
    ----------
    mean: array-like
        [Ndim] of means for each dim. of the distribution
    cov: array-like
        [Ndim,Ndim] covariance matrix
    """

    def __init__(self, mean, cov):
        print("warning: can only use global rng")
        self.mean = array(mean, ndmin=1, dtype="f8", copy=True)
        self.cov = array(cov, ndmin=1, dtype="f8", copy=True)

        self._set_log_mean_cov()

    def get_lnprob_scalar(self, x):
        """
        get the log(prob) for the input x

        parameters
        ----------
        x: array
            Array with length equal to number of dimensions
        """
        from numpy import dot

        if x.size != self.ndim:
            raise ValueError(
                "x must have length %d, " "got %d" % (self.ndim, x.size)
            )

        (w,) = where(x <= 0.0)
        if w.size > 0:
            raise ValueError("some values <= zero")

        logx = log(x)

        xdiff = logx - self.lmean

        chi2 = dot(xdiff, dot(self.lcov_inv, xdiff))

        # subtract mode to make max 0.0
        lnprob = -0.5 * chi2 - logx

        return lnprob

    def get_prob_scalar(self, x):
        """
        get the prob for the input x

        parameters
        ----------
        x: array-like
            Array with length equal to number of dimensions
        """

        lnprob = self.get_lnprob_scalar(x)
        return exp(lnprob)

    def sample(self, n=None):
        """
        sample from the distribution

        parameters
        ----------
        n: int, optional
            Number of points to generate from the distribution

        returns
        -------
        random points:
            If n is None or 1, a 1-d array is returned, else a [N, ndim]
            array is returned
        """
        rl = self.log_dist.rvs(n)
        numpy.exp(rl, rl)

        return rl

    def _set_log_mean_cov(self):
        """
        Genrate the mean and covariance in log space,
        as well as the multivariate normal for the log space
        """
        import scipy.stats

        mean = self.mean
        cov = self.cov

        ndim = mean.size
        self.ndim = ndim

        if ndim != cov.shape[0] or ndim != cov.shape[1]:
            raise ValueError(
                "mean has size %d but "
                "cov has shape %s" % (ndim, str(cov.shape))
            )
        (w,) = where(mean <= 0)
        if w.size > 0:
            raise ValueError("some means <= 0: %s" % str(mean))

        mean_sq = mean ** 2

        lmean = zeros(ndim)
        lcov = zeros((ndim, ndim))

        # first fill in diagonal terms
        for i in range(ndim):
            lcov[i, i] = log(1.0 + cov[i, i] / mean_sq[i])
            lmean[i] = log(mean[i]) - 0.5 * lcov[i, i]
            # print("lcov[%d,%d]: %g" % (i,i,lcov[i,i]))

        # now fill in off-diagonals
        for i in range(ndim):
            for j in range(i, ndim):
                if i == j:
                    continue

                # this reduces to mean^2 for the diagonal
                earg = lmean[i] + lmean[j] + 0.5 * (lcov[i, i] + lcov[j, j])
                m2 = exp(earg)

                tcovar = log(1.0 + cov[i, j] / m2)
                # print("tcovar:",tcovar)
                lcov[i, j] = tcovar
                lcov[j, i] = tcovar

        self.lmean = lmean
        self.lcov = lcov

        # will raise numpy.linalg.linalg.LinAlgError
        self.lcov_inv = numpy.linalg.inv(lcov)

        self.log_dist = scipy.stats.multivariate_normal(mean=lmean, cov=lcov)


class MVNMom(object):
    """
    Wrapper for MultivariateLogNormal that properly scales and offsets
    the parameters

    parameters
    -----------
    mean: array
        [row, col, M1, M2, T, I]
    cov: array
        covariance matrix
    psf_means: array
        [Irr,Irc,Icc]
    """

    def __init__(self, mean, cov, psf_mean):
        print("warning: can only use global rng")
        self.mean = array(mean, ndmin=1, dtype="f8", copy=True)
        self.cov = array(cov, ndmin=1, dtype="f8", copy=True)
        self.sigmas = sqrt(diag(self.cov))
        self.psf_mean = array(psf_mean, ndmin=1, dtype="f8", copy=True)

        self.ndim = mean.size
        self._set_offsets()

    def sample(self, n=None):
        if n is None:
            is_scalar = True
            n = 1
        else:
            is_scalar = False

        r = self.mvn.sample(n=n)

        cen_offsets = self.cen_offsets
        M1 = self.mean[2]
        M2 = self.mean[3]
        M1_midval = self.M1_midval
        M2_midval = self.M2_midval

        r[:, 0] -= cen_offsets[0]
        r[:, 1] -= cen_offsets[1]

        if self.M1_is_neg:
            r[:, 2] = M1 - (r[:, 2] - M1_midval)
        else:
            r[:, 2] = M1 + (r[:, 2] - M1_midval)

        if self.M2_is_neg:
            r[:, 3] = M2 - (r[:, 3] - M2_midval)
        else:
            r[:, 3] = M2 + (r[:, 3] - M2_midval)

        r[:, 4] -= self.psf_T

        if is_scalar:
            r = r[0, :]

        return r

    def _set_offsets(self):
        """
        We determine offsets that should be added to the input
        mean to make them positive

        also for Irc we may multiply by -1
        """
        mean = self.mean
        psf_mean = self.psf_mean

        ndim = mean.size
        nband = ndim - 6 + 1
        if nband > 1:
            raise RuntimeError("nband ==1 for now")

        M1 = mean[2]
        M2 = mean[3]
        T = mean[4]

        # psf_mean is [Irr,Irc,Icc]
        psf_T = psf_mean[0] + psf_mean[2]
        psf_M1 = psf_mean[2] - psf_mean[0]
        psf_M2 = 2 * psf_mean[1]

        Ttot = T + psf_T

        M1_midval = Ttot - numpy.abs(M1 + psf_M1)
        M2_midval = Ttot - numpy.abs(M2 + psf_M2)

        if M1 < 0:
            self.M1_is_neg = True
        else:
            self.M1_is_neg = False

        if M2 < 0:
            self.M2_is_neg = True
        else:
            self.M2_is_neg = False

        cen_offsets = self._get_cen_offsets()

        lmean = mean.copy()
        lmean[0] += cen_offsets[0]
        lmean[1] += cen_offsets[1]
        lmean[2] = M1_midval
        lmean[3] = M2_midval
        lmean[4] += psf_T

        self.psf_T = psf_T

        self.cen_offsets = cen_offsets
        self.M1_midval = M1_midval
        self.M2_midval = M2_midval

        self.mvn = MultivariateLogNormal(lmean, self.cov)

    def _get_cen_offsets(self):
        # offset so that the lognormal is close to gaussian

        cen_offsets = zeros(2)

        cen = self.mean[0:0 + 2]
        cen_sigma = self.sigmas[0:0 + 2]
        nsig = 5
        rng = 5.0 + nsig * cen_sigma

        for i in range(2):
            if cen[i] < 0:
                cen_offsets[i] = -cen[i] + rng[i]
            elif cen[i] - rng[i] < 0:
                cen_offsets[i] = rng[i] - cen[i]

        return cen_offsets


def lognorm_convert_old(mean, sigma):
    logmean = log(mean) - 0.5 * log(1 + sigma ** 2 / mean ** 2)
    logvar = log(1 + sigma ** 2 / mean ** 2)
    logsigma = sqrt(logvar)

    return logmean, logsigma


def lognorm_convert(mean, sigma, base=math.e):
    from math import log

    lbase = log(base)

    logmean = log(mean, base) - 0.5 * lbase * log(
        1 + sigma ** 2 / mean ** 2, base
    )
    logvar = log(1 + sigma ** 2 / mean ** 2, base)
    logsigma = sqrt(logvar)

    return logmean, logsigma


class BFracBase(PriorBase):
    """
    Base class for bulge fraction distribution
    """

    def sample(self, nrand=None):
        if nrand is None:
            return self.sample_one()
        else:
            return self.sample_many(nrand)

    def sample_one(self):
        rng = self.rng
        while True:
            t = rng.uniform()
            if t < self.bd_frac:
                r = self.bd_sigma * rng.normal()
            else:
                r = 1.0 + self.dev_sigma * rng.normal()
            if r >= 0.0 and r <= 1.0:
                break
        return r

    def sample_many(self, nrand):
        rng = self.rng

        r = numpy.zeros(nrand) - 9999
        ngood = 0
        nleft = nrand
        while ngood < nrand:
            tr = numpy.zeros(nleft)

            tmpu = rng.uniform(size=nleft)
            (w,) = numpy.where(tmpu < self.bd_frac)
            if w.size > 0:
                tr[0:w.size] = self.bd_loc + self.bd_sigma * rng.normal(
                    size=w.size
                )
            if w.size < nleft:
                tr[w.size:] = self.dev_loc + self.dev_sigma * rng.normal(
                    size=nleft - w.size
                )

            (wg,) = numpy.where((tr >= 0.0) & (tr <= 1.0))

            nkeep = wg.size
            if nkeep > 0:
                r[ngood:ngood + nkeep] = tr[wg]
                ngood += nkeep
                nleft -= nkeep
        return r

    def get_lnprob_array(self, bfrac):
        """
        Get the ln(prob) for the input b frac value
        """
        bfrac = numpy.array(bfrac, dtype="f8", copy=False)
        n = bfrac.size

        lnp = numpy.zeros(n)

        for i in range(n):
            lnp[i] = self.get_lnprob_scalar(bfrac[i])

        return lnp


class BFrac(BFracBase):
    """
    Bulge fraction

    half gaussian at zero width 0.1 for bulge+disk galaxies.

    smaller half gaussian at 1.0 with width 0.01 for bulge-only
    galaxies
    """

    def __init__(self, rng=None):
        PriorBase.__init__(self, rng=rng)

        sq2pi = numpy.sqrt(2 * numpy.pi)

        # bd is half-gaussian centered at 0.0
        self.bd_loc = 0.0
        # fraction that are bulge+disk
        self.bd_frac = 0.9
        # width of bd prior
        self.bd_sigma = 0.1
        self.bd_ivar = 1.0 / self.bd_sigma ** 2
        self.bd_norm = self.bd_frac / (sq2pi * self.bd_sigma)

        # dev is half-gaussian centered at 1.0
        self.dev_loc = 1.0
        # fraction of objects that are pure dev
        self.dev_frac = 1.0 - self.bd_frac
        # width of prior for dev-only region
        self.dev_sigma = 0.01
        self.dev_ivar = 1.0 / self.dev_sigma ** 2
        self.dev_norm = self.dev_frac / (sq2pi * self.dev_sigma)

    def get_lnprob_scalar(self, bfrac):
        """
        Get the ln(prob) for the input b frac value
        """
        if bfrac < 0.0 or bfrac > 1.0:
            raise GMixRangeError("bfrac out of range")

        bd_diff = bfrac - self.bd_loc
        dev_diff = bfrac - self.dev_loc

        p_bd = self.bd_norm * numpy.exp(
            -0.5 * bd_diff * bd_diff * self.bd_ivar
        )
        p_dev = self.dev_norm * numpy.exp(
            -0.5 * dev_diff * dev_diff * self.dev_ivar
        )

        lnp = numpy.log(p_bd + p_dev)
        return lnp


class Sinh(PriorBase):
    """
    a sinh distribution with mean and scale.

    Currently only supports "fdiff" style usage as a prior,
    e.g. for LM.
    """

    def __init__(self, mean, scale, rng=None):
        PriorBase.__init__(self, rng=rng)

        self.mean = mean
        self.scale = scale

    def get_fdiff(self, x):
        """
        For use with LM fitter
        (model-data)/width for both coordinates
        """
        return numpy.sinh((x - self.mean) / self.scale)

    def sample(self, nrand=1):
        """
        sample around the mean, +/- a scale length
        """
        if nrand is None:
            is_scalar = True
            nrand = 1
        else:
            is_scalar = False

        vals = self.rng.uniform(
            low=self.mean - self.scale, high=self.mean + self.scale, size=nrand
        )

        if is_scalar:
            vals = vals[0]

        return vals


class TruncatedGaussian(PriorBase):
    """
    Truncated gaussian
    """

    def __init__(self, mean, sigma, minval, maxval, rng=None):
        PriorBase.__init__(self, rng=rng)

        self.mean = mean
        self.sigma = sigma
        self.ivar = 1.0 / sigma ** 2
        self.sinv = 1.0 / sigma
        self.minval = minval
        self.maxval = maxval

    def get_lnprob_scalar(self, x):
        """
        just raise error if out of rang
        """
        if x < self.minval or x > self.maxval:
            raise GMixRangeError("value out of range")
        diff = x - self.mean
        return -0.5 * diff * diff * self.ivar

    def get_lnprob_array(self, x):
        """
        just raise error if out of rang
        """
        lnp = zeros(x.size)
        lnp -= numpy.inf
        (w,) = where((x > self.minval) & (x < self.maxval))
        if w.size > 0:
            diff = x[w] - self.mean
            lnp[w] = -0.5 * diff * diff * self.ivar

        return lnp

    def get_fdiff(self, x):
        """
        For use with LM fitter
        (model-data)/width for both coordinates
        """
        if x < self.minval or x > self.maxval:
            raise GMixRangeError("value out of range")
        return (x - self.mean) * self.sinv

    def sample(self, nrand=1):

        rng = self.rng

        if nrand is None:
            is_scalar = True
            nrand = 1
        else:
            is_scalar = False

        vals = numpy.zeros(nrand)

        ngood = 0
        nleft = nrand
        while ngood < nrand:

            tvals = rng.normal(loc=self.mean, scale=self.sigma, size=nleft)

            (w,) = numpy.where((tvals > self.minval) & (tvals < self.maxval))
            if w.size > 0:
                vals[ngood:ngood + w.size] = tvals[w]
                ngood += w.size
                nleft -= w.size

        if is_scalar:
            vals = vals[0]

        return vals


class TruncatedGaussianPolar(PriorBase):
    """
    Truncated gaussian on a circle
    """

    def __init__(self, mean1, mean2, sigma1, sigma2, maxval, rng=None):
        PriorBase.__init__(self, rng=rng)

        self.mean1 = mean1
        self.mean2 = mean2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.ivar1 = 1.0 / sigma1 ** 2
        self.ivar2 = 1.0 / sigma2 ** 2

        self.maxval = maxval
        self.maxval2 = maxval ** 2

    def sample(self, nrand):
        """
        Sample from truncated gaussian
        """

        rng = self.rng

        x1 = numpy.zeros(nrand)
        x2 = numpy.zeros(nrand)

        nleft = nrand
        ngood = 0
        while nleft > 0:
            r1 = rng.normal(loc=self.mean1, scale=self.sigma1, size=nleft)
            r2 = rng.normal(loc=self.mean2, scale=self.sigma2, size=nleft)

            rsq = r1 ** 2 + r2 ** 2

            (w,) = numpy.where(rsq < self.maxval2)
            nkeep = w.size
            if nkeep > 0:
                x1[ngood:ngood + nkeep] = r1[w]
                x2[ngood:ngood + nkeep] = r2[w]
                nleft -= nkeep
                ngood += nkeep

        return x1, x2

    def get_lnprob_scalar(self, x1, x2):
        """
        ln(p) for scalar inputs
        """
        xsq = x1 ** 2 + x2 ** 2
        if xsq > self.maxval2:
            raise GMixRangeError("square value out of range: %s" % xsq)
        diff1 = x1 - self.mean1
        diff2 = x2 - self.mean2
        return (
            -0.5 * diff1 * diff1 * self.ivar1
            - 0.5 * diff2 * diff2 * self.ivar2
        )

    def get_lnprob_array(self, x1, x2):
        """
        ln(p) for a array inputs
        """
        x1 = numpy.array(x1, dtype="f8", copy=False)
        x2 = numpy.array(x2, dtype="f8", copy=False)

        x2 = x1 ** 2 + x2 ** 2
        (w,) = numpy.where(x2 > self.maxval2)
        if w.size > 0:
            raise GMixRangeError("values out of range")
        diff1 = x1 - self.mean1
        diff2 = x2 - self.mean2
        return (
            -0.5 * diff1 * diff1 * self.ivar1
            - 0.5 * diff2 * diff2 * self.ivar2
        )


class Student(object):
    def __init__(self, mean, sigma):
        """
        sigma is not std(x)
        """
        print("warning: cannot use local rng")
        self.reset(mean, sigma)

    def reset(self, mean, sigma):
        """
        complete reset
        """
        import scipy.stats

        self.mean = mean
        self.sigma = sigma

        self.tdist = scipy.stats.t(1.0, loc=mean, scale=sigma)

    def sample(self, nrand):
        """
        Draw samples from the distribution
        """
        return self.tdist.rvs(nrand)

    def get_lnprob_array(self, x):
        """
        get ln(prob)
        """
        return self.tdist.logpdf(x)

    get_lnprob_scalar = get_lnprob_array


class StudentPositive(Student):
    def sample(self, nrand):
        """
        Draw samples from the distribution
        """
        vals = numpy.zeros(nrand)

        nleft = nrand
        ngood = 0
        while nleft > 0:
            r = super(StudentPositive, self).sample(nleft)

            (w,) = numpy.where(r > 0.0)
            nkeep = w.size
            if nkeep > 0:
                vals[ngood:ngood + nkeep] = r[w]
                nleft -= nkeep
                ngood += nkeep

        return vals

    def get_lnprob_scalar(self, x):
        """
        get ln(prob)
        """

        if x <= 0:
            raise GMixRangeError("value less than zero")
        return super(StudentPositive, self).get_lnprob_scalar(x)

    def get_lnprob_array(self, x):
        """
        get ln(prob)
        """
        x = numpy.array(x, dtype="f8", copy=False)

        (w,) = numpy.where(x <= 0)
        if w.size != 0:
            raise GMixRangeError("values less than zero")
        return super(StudentPositive, self).get_lnprob_array(x)


class Student2D(object):
    def __init__(self, mean1, mean2, sigma1, sigma2):
        """
        circular student
        sigma is not std(x)
        """
        self.reset(mean1, mean2, sigma1, sigma2)

    def reset(self, mean1, mean2, sigma1, sigma2):
        """
        complete reset
        """
        self.mean1 = mean1
        self.sigma1 = sigma1
        self.mean2 = mean2
        self.sigma2 = sigma2

        self.tdist1 = Student(mean1, sigma1)
        self.tdist2 = Student(mean2, sigma2)

    def sample(self, nrand):
        """
        Draw samples from the distribution
        """
        r1 = self.tdist1.sample(nrand)
        r2 = self.tdist2.sample(nrand)

        return r1, r2

    def get_lnprob_array(self, x1, x2):
        """
        get ln(prob)
        """
        lnp1 = self.tdist1.get_lnprob_array(x1)
        lnp2 = self.tdist2.get_lnprob_array(x2)

        return lnp1 + lnp2

    get_lnprob_scalar = get_lnprob_array


class TruncatedStudentPolar(Student2D):
    """
    Truncated gaussian on a circle
    """

    def __init__(self, mean1, mean2, sigma1, sigma2, maxval):
        """
        sigma is not std(x)
        """
        self.reset(mean1, mean2, sigma1, sigma2, maxval)

    def reset(self, mean1, mean2, sigma1, sigma2, maxval):
        super(TruncatedStudentPolar, self).reset(mean1, mean2, sigma1, sigma2)
        self.maxval = maxval
        self.maxval2 = maxval ** 2

    def sample(self, nrand):
        """
        Sample from truncated gaussian
        """
        x1 = numpy.zeros(nrand)
        x2 = numpy.zeros(nrand)

        nleft = nrand
        ngood = 0
        while nleft > 0:
            r1, r2 = super(TruncatedStudentPolar, self).sample(nleft)

            rsq = r1 ** 2 + r2 ** 2

            (w,) = numpy.where(rsq < self.maxval2)
            nkeep = w.size
            if nkeep > 0:
                x1[ngood:ngood + nkeep] = r1[w]
                x2[ngood:ngood + nkeep] = r2[w]
                nleft -= nkeep
                ngood += nkeep

        return x1, x2

    def get_lnprob_scalar(self, x1, x2):
        """
        ln(p) for scalar inputs
        """

        x2 = x1 ** 2 + x2 ** 2
        if x2 > self.maxval2:
            raise GMixRangeError("square value out of range: %s" % x2)
        lnp = super(TruncatedStudentPolar, self).get_lnprob_scalar(x1, x2)
        return lnp

    def get_lnprob_array(self, x1, x2):
        """
        ln(p) for a array inputs
        """

        x1 = numpy.array(x1, dtype="f8", copy=False)
        x2 = numpy.array(x2, dtype="f8", copy=False)

        x2 = x1 ** 2 + x2 ** 2
        (w,) = numpy.where(x2 > self.maxval2)
        if w.size > 0:
            raise GMixRangeError("values out of range")

        lnp = super(TruncatedStudentPolar, self).get_lnprob_array(x1, x2)
        return lnp


def scipy_to_lognorm(shape, scale):
    """
    Wrong?
    """
    srat2 = numpy.exp(shape ** 2) - 1.0
    # srat2 = numpy.exp( shape ) - 1.0

    meanx = scale * numpy.exp(0.5 * numpy.log(1.0 + srat2))
    sigmax = numpy.sqrt(srat2 * meanx ** 2)

    return meanx, sigmax


class CenPrior(PriorBase):
    """
    Independent gaussians in each dimension
    """

    def __init__(self, cen1, cen2, sigma1, sigma2, rng=None):
        super(CenPrior, self).__init__(rng=rng)

        self.cen1 = float(cen1)
        self.cen2 = float(cen2)
        self.sigma1 = float(sigma1)
        self.sigma2 = float(sigma2)
        self.sinv1 = 1.0 / self.sigma1
        self.sinv2 = 1.0 / self.sigma2
        self.s2inv1 = 1.0 / self.sigma1 ** 2
        self.s2inv2 = 1.0 / self.sigma2 ** 2

    def get_fdiff(self, x1, x2):
        """
        For use with LM fitter
        (model-data)/width for both coordinates
        """
        d1 = (x1 - self.cen1) * self.sinv1
        d2 = (x2 - self.cen2) * self.sinv2
        return d1, d2

    def get_lnprob_scalar(self, x1, x2):
        """
        log probability at the specified point
        """
        d1 = self.cen1 - x1
        d2 = self.cen2 - x2
        return -0.5 * d1 * d1 * self.s2inv1 - 0.5 * d2 * d2 * self.s2inv2

    def get_lnprob_scalar_sep(self, x1, x2):
        """
        log probability at the specified point, but separately
        in the two dimensions
        """
        d1 = self.cen1 - x1
        d2 = self.cen2 - x2
        return -0.5 * d1 * d1 * self.s2inv1, -0.5 * d2 * d2 * self.s2inv2

    def get_prob_scalar(self, x1, x2):
        """
        log probability at the specified point
        """
        from math import exp

        d1 = self.cen1 - x1
        d2 = self.cen2 - x2
        lnp = -0.5 * d1 * d1 * self.s2inv1 - 0.5 * d2 * d2 * self.s2inv2
        return exp(lnp)

    def sample(self, n=None):
        """
        Get a single sample or arrays
        """

        rng = self.rng

        rand1 = rng.normal(loc=self.cen1, scale=self.sigma1, size=n)
        rand2 = rng.normal(loc=self.cen2, scale=self.sigma2, size=n)

        return rand1, rand2

    sample2d = sample


SimpleGauss2D = CenPrior


class TruncatedSimpleGauss2D(PriorBase):
    """
    Independent gaussians in each dimension, with a specified
    maximum length
    """

    def __init__(self, cen1, cen2, sigma1, sigma2, maxval, rng=None):
        PriorBase.__init__(self, rng=rng)

        self.cen1 = cen1
        self.cen2 = cen2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.s2inv1 = 1.0 / self.sigma1 ** 2
        self.s2inv2 = 1.0 / self.sigma2 ** 2

        self.maxval = maxval
        self.maxval_sq = maxval ** 2

    def get_lnprob_nothrow(self, p1, p2):
        """
        log probability
        """

        psq = p1 ** 2 + p2 ** 2
        if psq >= self.maxval_sq:
            lnp = LOWVAL
        else:
            d1 = self.cen1 - p1
            d2 = self.cen2 - p2
            lnp = -0.5 * d1 * d1 * self.s2inv1 - 0.5 * d2 * d2 * self.s2inv2

        return lnp

    def get_lnprob_scalar(self, p1, p2):
        """
        log probability
        """

        psq = p1 ** 2 + p2 ** 2
        if psq >= self.maxval_sq:
            raise GMixRangeError("value too big")

        d1 = self.cen1 - p1
        d2 = self.cen2 - p2
        lnp = -0.5 * d1 * d1 * self.s2inv1 - 0.5 * d2 * d2 * self.s2inv2

        return lnp

    def get_lnprob_array(self, p1, p2):
        """
        log probability
        """

        lnp = zeros(p1.size) - numpy.inf

        psq = p1 ** 2 + p2 ** 2
        (w,) = where(psq < self.maxval_sq)
        if w.size > 0:

            d1 = self.cen1 - p1[w]
            d2 = self.cen2 - p2[w]
            lnp[w] = -0.5 * d1 * d1 * self.s2inv1 - 0.5 * d2 * d2 * self.s2inv2

        return lnp

    def get_prob_scalar(self, p1, p2):
        """
        linear probability
        """
        lnp = self.get_lnprob(p1, p2)
        return numpy.exp(lnp)

    def sample(self, nrand=None):
        """
        Get nrand random deviates from the distribution
        """

        rng = self.rng

        if nrand is None:
            is_scalar = True
            nrand = 1
        else:
            is_scalar = False

        r1 = numpy.zeros(nrand)
        r2 = numpy.zeros(nrand)

        ngood = 0
        nleft = nrand
        while ngood < nrand:

            rvals1 = rng.normal(loc=self.cen1, scale=self.sigma1, size=nleft)
            rvals2 = rng.normal(loc=self.cen2, scale=self.sigma2, size=nleft)

            rsq = (rvals1 - self.cen1) ** 2 + (rvals2 - self.cen2) ** 2

            (w,) = numpy.where(rsq < self.maxval_sq)
            if w.size > 0:
                r1[ngood:ngood + w.size] = rvals1[w]
                r2[ngood:ngood + w.size] = rvals2[w]
                ngood += w.size
                nleft -= w.size

        if is_scalar:
            r1 = r1[0]
            r2 = r2[0]

        return r1, r2


JOINT_N_ITER = 1000
JOINT_MIN_COVAR = 1.0e-6
JOINT_COVARIANCE_TYPE = "full"


class TFluxPriorCosmosBase(PriorBase):
    """
    T is in arcsec**2
    """

    def __init__(self, T_bounds=[1.0e-6, 100.0], flux_bounds=[1.0e-6, 100.0]):

        print("warning: cannot use local rng")

        self.set_bounds(T_bounds=T_bounds, flux_bounds=flux_bounds)

        self.make_gmm()
        self.make_gmix()

    def set_bounds(self, T_bounds=None, flux_bounds=None):

        if len(T_bounds) != 2:
            raise ValueError("T_bounds must be len==2")
        if len(flux_bounds) != 2:
            raise ValueError("flux_bounds must be len==2")

        self.T_bounds = array(T_bounds, dtype="f8")
        self.flux_bounds = array(flux_bounds, dtype="f8")

    def get_flux_mode(self):
        return self.flux_mode

    def get_T_near(self):
        return self.T_near

    def get_lnprob_slow(self, T_and_flux_input):
        """
        ln prob for linear variables
        """

        nd = len(T_and_flux_input.shape)

        if nd == 1:
            T_and_flux = T_and_flux_input.reshape(1, 2)
        else:
            T_and_flux = T_and_flux_input

        T_bounds = self.T_bounds
        T = T_and_flux[0, 0]
        if T < T_bounds[0] or T > T_bounds[1]:
            return LOWVAL

        flux_bounds = self.flux_bounds
        flux = T_and_flux[0, 1]
        if flux < flux_bounds[0] or flux > flux_bounds[1]:
            return LOWVAL

        logvals = numpy.log10(T_and_flux)

        return self.gmm.score(logvals)

    def get_lnprob_many(self, T_and_flux):
        """
        The fast array one using a GMix
        """

        n = T_and_flux.shape[0]
        lnp = numpy.zeros(n)

        for i in range(n):
            lnp[i] = self.get_lnprob_one(T_and_flux[i, :])

        return lnp

    def get_lnprob_one(self, T_and_flux):
        """
        The fast scalar one using a GMix
        """

        # bounds cuts happen in pixel space (if scale!=1)
        T_bounds = self.T_bounds
        flux_bounds = self.flux_bounds

        T = T_and_flux[0]
        if T < T_bounds[0] or T > T_bounds[1]:
            return LOWVAL

        flux = T_and_flux[1]
        if flux < flux_bounds[0] or flux > flux_bounds[1]:
            return LOWVAL

        logpars = numpy.log10(T_and_flux)

        return self.gmix.get_loglike_rowcol(logpars[0], logpars[1])

    def sample(self, n):
        """
        Sample the linear variables
        """

        T_bounds = self.T_bounds
        flux_bounds = self.flux_bounds

        lin_samples = numpy.zeros((n, 2))
        nleft = n
        ngood = 0
        while nleft > 0:
            tsamples = self.gmm.sample(nleft)
            tlin_samples = 10.0 ** tsamples

            Tvals = tlin_samples[:, 0]
            flux_vals = tlin_samples[:, 1]
            (w,) = numpy.where(
                (Tvals > T_bounds[0])
                & (Tvals < T_bounds[1])
                & (flux_vals > flux_bounds[0])
                & (flux_vals < flux_bounds[1])
            )

            if w.size > 0:
                lin_samples[ngood:ngood + w.size, :] = tlin_samples[w, :]
                ngood += w.size
                nleft -= w.size

        return lin_samples

    def make_gmm(self):
        """
        Make a GMM object from the inputs
        """
        from sklearn.mixture import GMM

        # we will over-ride values, pars here shouldn't matter except
        # for consistency

        ngauss = self.weights.size
        gmm = GMM(
            n_components=ngauss,
            n_iter=JOINT_N_ITER,
            min_covar=JOINT_MIN_COVAR,
            covariance_type=JOINT_COVARIANCE_TYPE,
        )
        gmm.means_ = self.means
        gmm.covars_ = self.covars
        gmm.weights_ = self.weights

        self.gmm = gmm

    def make_gmix(self):
        """
        Make a GMix object for speed
        """
        from . import gmix

        ngauss = self.weights.size
        pars = numpy.zeros(6 * ngauss)
        for i in range(ngauss):
            index = i * 6
            pars[index + 0] = self.weights[i]
            pars[index + 1] = self.means[i, 0]
            pars[index + 2] = self.means[i, 1]

            pars[index + 3] = self.covars[i, 0, 0]
            pars[index + 4] = self.covars[i, 0, 1]
            pars[index + 5] = self.covars[i, 1, 1]

        self.gmix = gmix.GMix(ngauss=ngauss, pars=pars)


class TFluxPriorCosmosExp(TFluxPriorCosmosBase):
    """
    joint T-flux distribution based on fits to cosmos data
    using an Exp model
    """

    flux_mode = 0.121873372203

    # T_near is mean T near the flux mode
    T_near = 0.182461907543  # arcsec**2

    weights = array(
        [
            0.24725964,
            0.12439838,
            0.10301993,
            0.07903986,
            0.16064439,
            0.01162365,
            0.15587558,
            0.11813856,
        ]
    )
    means = array(
        [
            [-0.63771027, -0.55703495],
            [-1.09910453, -0.79649937],
            [-0.89605713, -0.9432781],
            [-1.01262357, -0.05649636],
            [-0.26622288, -0.16742824],
            [-0.50259072, -0.10134068],
            [-0.11414387, -0.49449105],
            [-0.39625801, -0.83781264],
        ]
    )

    covars = array(
        [
            [[0.17219003, -0.00650028], [-0.00650028, 0.05053097]],
            [[0.09800749, 0.00211059], [0.00211059, 0.01099591]],
            [[0.16110231, 0.00194853], [0.00194853, 0.00166609]],
            [[0.07933064, 0.08535596], [0.08535596, 0.20289928]],
            [[0.17805248, 0.1356711], [0.1356711, 0.24728999]],
            [[1.07112875, -0.08777646], [-0.08777646, 0.26131025]],
            [[0.06570913, 0.04912536], [0.04912536, 0.094739]],
            [[0.0512232, 0.00519115], [0.00519115, 0.00999365]],
        ]
    )


class TFluxPriorCosmosDev(TFluxPriorCosmosBase):
    """
    joint T-flux distribution based on fits to cosmos data
    using an Dev model

    pars from
    ~/lensing/galsim-cosmos-data/gmix-fits/001/dist/gmix-cosmos-001-dev-joint-dist.fits
    """

    flux_mode = 0.241579121406777
    # T_near is mean T near the flux mode
    T_near = 2.24560320009

    weights = array(
        [
            0.13271808,
            0.10622821,
            0.09982766,
            0.29088236,
            0.1031488,
            0.10655095,
            0.01631938,
            0.14432457,
        ]
    )
    means = array(
        [
            [0.22180692, 0.05099562],
            [0.92589409, -0.61785194],
            [-0.10599071, -0.82198516],
            [1.06182382, -0.29997484],
            [1.57670969, 0.25244698],
            [-0.1807241, -0.35644187],
            [0.73288177, 0.07028779],
            [-0.14674281, -0.65843109],
        ]
    )
    covars = array(
        [
            [[0.35760805, 0.17652397], [0.17652397, 0.28479473]],
            [[0.14969479, 0.03008289], [0.03008289, 0.01369154]],
            [[0.37153864, 0.03293217], [0.03293217, 0.00476308]],
            [[0.27138008, 0.08628813], [0.08628813, 0.0883718]],
            [[0.20242911, 0.12374718], [0.12374718, 0.22626528]],
            [[0.30836137, 0.07141098], [0.07141098, 0.05870399]],
            [[2.13662397, 0.04301596], [0.04301596, 0.17006638]],
            [[0.40151762, 0.05215542], [0.05215542, 0.01733399]],
        ]
    )


def srandu(num=None, rng=None):
    """
    Generate random numbers in the symmetric distribution [-1,1]
    """

    if rng is None:
        randu = numpy.random.uniform
    else:
        randu = rng.uniform

    return randu(low=-1.0, high=1.0, size=num)


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


_g_cosmos_t = array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.001,
        0.15,
        0.2,
        0.4,
        0.5,
        0.7,
        0.8,
        0.85,
        0.86,
        0.865,
        0.89,
        0.9,
        0.91,
        0.92,
        0.925,
        0.93,
        0.94,
        0.95,
        0.96,
        0.97,
        0.98,
        0.999,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
)
_g_cosmos_c = array(
    [
        0.00000000e00,
        4.16863836e01,
        1.08898496e03,
        1.11759878e03,
        1.08252322e03,
        8.37179646e02,
        5.01288706e02,
        2.85169581e02,
        1.82341644e01,
        9.86159479e00,
        2.35167895e00,
        -1.16749781e-02,
        2.18333603e-03,
        6.23967745e-03,
        4.25839705e-03,
        5.30413464e-03,
        4.89754389e-03,
        5.11359296e-03,
        4.87944452e-03,
        5.13580057e-03,
        4.78950152e-03,
        5.37471368e-03,
        4.00911261e-03,
        7.30958026e-03,
        -4.03798531e-20,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
    ]
)
_g_cosmos_k = 3


class ZDisk2D(object):
    """
    uniform over a disk centered at zero [0,0] with radius r
    """

    def __init__(self, radius, rng=None):

        self.rng = make_rng(rng=rng)

        self.radius = radius
        self.radius_sq = radius ** 2

    def get_lnprob_scalar1d(self, r):
        """
        get ln(prob) at radius r=sqrt(x^2 + y^2)
        """
        if r >= self.radius:
            raise GMixRangeError("position out of bounds")
        return 0.0

    def get_prob_scalar1d(self, r):
        """
        get prob at radius r=sqrt(x^2 + y^2)
        """
        if r >= self.radius:
            return 0.0
        else:
            return 1.0

    def get_lnprob_scalar2d(self, x, y):
        """
        get ln(prob) at the input position
        """
        r2 = x ** 2 + y ** 2
        if r2 >= self.radius_sq:
            raise GMixRangeError("position out of bounds")

        return 0.0

    def get_prob_scalar2d(self, x, y):
        """
        get ln(prob) at the input position
        """

        r2 = x ** 2 + y ** 2
        if r2 >= self.radius_sq:
            return 0.0
        else:
            return 1.0

    def get_prob_array2d(self, x, y):
        """
        probability, 1.0 inside disk, outside raises exception

        does not raise an exception
        """
        x = numpy.array(x, dtype="f8", ndmin=1, copy=False)
        y = numpy.array(y, dtype="f8", ndmin=1, copy=False)
        out = numpy.zeros(x.size, dtype="f8")

        r2 = x ** 2 + y ** 2
        (w,) = numpy.where(r2 < self.radius_sq)
        if w.size > 0:
            out[w] = 1.0

        return out

    def sample1d(self, n=None):
        """
        Get samples in 1-d radius
        """

        if n is None:
            n = 1
            is_scalar = True
        else:
            is_scalar = False

        r2 = self.radius_sq * self.rng.uniform(size=n)

        r = sqrt(r2)

        if is_scalar:
            r = r[0]

        return r

    def sample2d(self, n=None):
        """
        Get samples.  Send no args to get a scalar.
        """
        if n is None:
            n = 1
            is_scalar = True
        else:
            is_scalar = False

        radius = self.sample1d(n)

        theta = 2.0 * numpy.pi * self.rng.uniform(size=n)

        x = radius * cos(theta)
        y = radius * sin(theta)

        if is_scalar:
            x = x[0]
            y = y[0]

        return x, y


class ZAnnulus(ZDisk2D):
    """
    uniform over an annulus

    Note get_lnprob_scalar1d and get_prob_scalar1d and 2d are already
    part of base class
    """

    def __init__(self, rmin, rmax, rng=None):

        assert rmin < rmax

        self.rmin = rmin
        super(ZAnnulus, self).__init__(
            rmax, rng=rng,
        )

    def sample1d(self, n=None):
        if n is None:
            n = 1
            is_scalar = True
        else:
            is_scalar = False

        r = numpy.zeros(n)
        ngood = 0
        nleft = n

        while True:
            rtmp = super(ZAnnulus, self).sample1d(nleft)

            (w,) = numpy.where(rtmp > self.rmin)
            if w.size > 0:
                r[ngood:ngood + w.size] = rtmp[w]
                ngood += w.size
                nleft -= w.size

            if ngood == n:
                break

        if is_scalar:
            r = r[0]

        return r


class ZDisk2DErf(object):
    """
    uniform over a disk centered at zero [0,0] with radius r

    the cutoff begins at radius-6*width so that the prob is
    completely zero by the point of the radius
    """

    def __init__(self, max_radius=1.0, rolloff_point=0.98, width=0.005):
        self.rolloff_point = rolloff_point
        self.radius = max_radius
        self.radius_sq = max_radius ** 2

        self.width = width

    def get_prob_scalar1d(self, val):
        """
        scalar only
        """
        from ._gmix import erf

        if val > self.radius:
            return 0.0

        erf_val = erf((val - self.rolloff_point) / self.width)
        retval = 0.5 * (1.0 + erf_val)

        return retval

    def get_lnprob_scalar1d(self, val):
        """
        scalar only
        """
        from ._gmix import erf

        if val > self.radius:
            return LOWVAL

        erf_val = erf((val - self.rolloff_point) / self.width)
        linval = 0.5 * (1.0 + erf_val)

        if linval <= 0.0:
            retval = LOWVAL
        else:
            retval = log(linval)

        return retval

    def get_lnprob_scalar2d(self, g1, g2):
        g = sqrt(g1 ** 2 + g2 ** 2)
        return self.get_lnprob_scalar1d(g)

    def get_prob_array1d(self, vals):
        """
        works for both array and scalar
        """
        from ._gmix import erf_array

        vals = numpy.array(vals, ndmin=1, dtype="f8", copy=False)

        arg = (self.rolloff_point - vals) / self.width
        erf_vals = numpy.zeros(vals.size)

        erf_array(arg, erf_vals)
        prob = 0.5 * (1.0 + erf_vals)

        (w,) = numpy.where(vals >= self.radius)
        if w.size > 0:
            prob[w] = 0.0

        return prob

    def get_lnprob_array1d(self, vals):
        """
        works for both array and scalar
        """

        prob = self.get_prob_array1d(vals)

        lnprob = LOWVAL + zeros(prob.size)
        (w,) = numpy.where(prob > 0.0)
        if w.size > 0:
            lnprob[w] = log(prob[w])

        return lnprob


class UDisk2DCut(object):
    """
    uniform over a disk centered at zero [0,0] with radius r
    """

    def __init__(self, cutval=0.97):

        self.cutval = cutval
        self.fac = 1.0 / (1.0 - cutval)

    def get_lnprob_scalar1d(self, val):
        """
        works for both array and scalar
        """

        cutval = self.cutval
        if val > cutval:
            vdiff = (val - cutval) * self.fac
            retval = -numpy.arctanh(vdiff) ** 2
        else:
            retval = 0.0

        return retval

    def get_lnprob_scalar2d(self, x1, x2):
        """
        works for both array and scalar
        """

        x = sqrt(x1 ** 2 + x2 ** 2)
        return self.get_lnprob_scalar1d(x)

    def get_lnprob_array1d(self, vals):
        """
        works for both array and scalar
        """

        vals = numpy.array(vals, ndmin=1, copy=False)
        retvals = zeros(vals.size)

        cutval = self.cutval
        (w,) = numpy.where(vals > cutval)
        if w.size > 0:
            vdiff = (vals[w] - cutval) * self.fac
            retvals[w] = -numpy.arctanh(vdiff) ** 2

        return retvals

    def get_lnprob_array2d(self, x1, x2):
        """
        works for both array and scalar
        """

        x = sqrt(x1 ** 2 + x2 ** 2)
        return self.get_lnprob_array1d(x)


class KDE(object):
    """
    create a kde from the input data

    just a wrapper around scipy.stats.gaussian_kde to
    provide a uniform interface
    """

    def __init__(self, data, kde_factor):
        import scipy.stats

        if len(data.shape) == 1:
            self.is_1d = True
        else:
            self.is_1d = False

        self.kde = scipy.stats.gaussian_kde(
            data.transpose(), bw_method=kde_factor,
        )

    def sample(self, n=None):
        """
        draw random samples from the kde
        """
        if n is None:
            n = 1
            is_scalar = True
        else:
            is_scalar = False

        r = self.kde.resample(size=n).transpose()

        if self.is_1d:
            r = r[:, 0]

        if is_scalar:
            r = r[0]

        return r


class LimitPDF(object):
    """
    wrapper class to limit the sampled range of a PDF

    parameters
    ----------
    pdf: a pdf
        A PDF with the sample(nrand=) method
    limits: sequence
        2-element sequence [min, max]
    """

    def __init__(self, pdf, limits):
        self.pdf = pdf

        self.set_limits(limits)

    def sample(self, nrand=None):
        """
        sample from the distribution, limiting to the specified range
        """

        if nrand is None:
            return self._sample_one()
        else:
            return self._sample_many(nrand)

    def _sample_one(self):
        """
        sample a single value
        """

        limits = self.limits
        pdf = self.pdf

        while True:
            val = pdf.sample()

            if limits[0] < val < limits[1]:
                break

        return val

    def _sample_many(self, nrand):
        """
        sample an array of values
        """

        limits = self.limits
        pdf = self.pdf

        samples = numpy.zeros(nrand)

        ngood = 0
        nleft = nrand

        while ngood < nrand:

            rvals = pdf.sample(nleft)

            (w,) = numpy.where((rvals > limits[0]) & (rvals < limits[1]))
            if w.size > 0:
                samples[ngood:ngood + w.size] = rvals[w]
                ngood += w.size
                nleft -= w.size

        return samples

    def set_limits(self, limits):
        """
        set the limits
        """

        ok = False
        try:
            n = len(limits)
            if n == 2:
                ok = True
        except TypeError:
            pass

        if ok is False:
            raise ValueError(
                "expected limits to be 2-element sequence, " "got %s" % limits
            )

        if limits[0] >= limits[1]:
            raise ValueError(
                "limits[0] must be less than " "limits[1], got: %s" % limits
            )
        self.limits = limits
