"""
Convention is that all priors should have peak ln(prob)==0. This
helps use in priors for LM fitting
"""
import numpy
from numpy import where, array, exp, log, sqrt, zeros, sin, cos

from ..gexceptions import GMixRangeError
from .random import srandu
from .priors import PriorBase
import logging
from ..util import print_pars
from ..defaults import LOWVAL, copy_if_needed

LOGGER = logging.getLogger(__name__)


class GPriorBase(PriorBase):
    """
    Base object for priors on shear.

    Note that depending on your purpose, you may need to override the following
    abstract methods:

        fill_prob_array1d
        fill_lnprob_array2d
        fill_prob_array2d
        get_prob_scalar2d
        get_lnprob_scalar2d
        get_prob_scalar1d

    parameters
    ----------
    pars: float or array-like
        Parameters for the prior.
    rng: np.random.RandomState
        An random number generator (RNG) to use.

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
    def __init__(self, pars, rng):
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

        g1arr = array(g1arr, dtype="f8", copy=copy_if_needed)
        g2arr = array(g2arr, dtype="f8", copy=copy_if_needed)

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

        g1arr = array(g1arr, dtype="f8", copy=copy_if_needed)
        g2arr = array(g2arr, dtype="f8", copy=copy_if_needed)

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

        garr = array(garr, dtype="f8", copy=copy_if_needed)

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

    def set_maxval1d(self, maxguess=0.1):
        """
        Use a simple minimizer to find the max value of the 1d distribution

        parameters
        ----------
        maxguess: float
            The guess for finding the maximum g value if it is needed.
        """
        import scipy.optimize

        res = scipy.optimize.minimize(self.get_prob_scalar1d_neg, maxguess)

        if res["status"] != 0:
            raise RuntimeError("failed to find min, flags: %d" % res["status"])

        self.maxval1d = -res["fun"]
        self.maxval1d_loc = res["x"]

    def get_prob_scalar1d_neg(self, g, *args):
        """
        Helper function so we can use the minimizer
        """
        return -self.get_prob_scalar1d(g)

    def fit(self, xdata, ydata, guess=None, show=False):
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
        from ..fitting.leastsqbound import run_leastsq

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

        print("flags:", res["flags"], "\nnfev:", res["nfev"])
        print_pars(res["pars"], front="pars: ", logger=LOGGER)
        print_pars(res["pars_err"], front="perr: ", logger=LOGGER)

        c = ["%g" % p for p in res["pars"]]
        c = "[" + ", ".join(c) + "]"
        print("pars list:", c)

    def _calc_fdiff(self, pars):
        # helper function for the fitter
        self.set_pars(pars)
        p = self.get_prob_array1d(self.xdata)
        fdiff = (p - self.ydata) * self.ierr
        return fdiff

    # here for old api - can be removed.
    dofit = fit


class GPriorGauss(GPriorBase):
    """
    Gaussian shear prior.

    See `GPriorBase` for more documentation.

    parameters
    ----------
    pars: float
        The width of the Gaussian prior for g1, g2.
    rng: np.random.RandomState
        An random number generator (RNG) to use.
    """
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.sigma = float(self.pars)

    def sample1d(self, nrand=None, **kw):
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
    rng: np.random.RandomState
        An random number generator (RNG) to use.
    A: float, optional
        The overall amplitude of the prior. This is used for fitting, but not
        when evaluating lnprob. Default is 1.0.
    """
    def __init__(self, sigma, rng, A=1.0):
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

        return super().sample1d(nrand, maxguess=maxguess)

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
        lnp = self.get_lnprob_array2d(g1, g2)
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


class ZDisk2D(PriorBase):
    """
    Uniform prior over a disk centered at zero [0,0] with radius r.

    parameters
    ----------
    radius: float
        The maximum raidus of the disk.
    rng: np.random.RandomState
        An random number generator (RNG) to use.

    attributes
    ----------
    radius: float
        The maximum raidus of the disk.
    """
    def __init__(self, radius, rng):
        super().__init__(rng=rng)

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
        get prob at the input position
        """

        r2 = x ** 2 + y ** 2
        if r2 >= self.radius_sq:
            return 0.0
        else:
            return 1.0

    def get_prob_array2d(self, x, y):
        """
        get prob at the input positions
        """
        x = numpy.array(x, dtype="f8", ndmin=1, copy=copy_if_needed)
        y = numpy.array(y, dtype="f8", ndmin=1, copy=copy_if_needed)
        out = numpy.zeros(x.size, dtype="f8")

        r2 = x ** 2 + y ** 2
        (w,) = numpy.where(r2 < self.radius_sq)
        if w.size > 0:
            out[w] = 1.0

        return out

    def sample1d(self, nrand=None):
        """
        Get random |g| from the 1d distribution.

        parameters
        ----------
        nrand: int
            Number to generate

        returns
        -------
        g: array-like
            The generated |g| values.
        """
        if nrand is None:
            nrand = 1
            is_scalar = True
        else:
            is_scalar = False

        r2 = self.radius_sq * self.rng.uniform(size=nrand)

        r = sqrt(r2)

        if is_scalar:
            r = r[0]

        return r

    def sample2d(self, nrand=None):
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

        radius = self.sample1d(nrand=nrand)

        theta = 2.0 * numpy.pi * self.rng.uniform(size=nrand)

        x = radius * cos(theta)
        y = radius * sin(theta)

        if is_scalar:
            x = x[0]
            y = y[0]

        return x, y
