import numpy
from numpy import array, where, sqrt

from .random import make_rng
from .constants import LOWVAL
from .random import srandu


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
