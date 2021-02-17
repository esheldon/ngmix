"""
Convention is that all priors should have peak ln(prob)==0. This
helps use in priors for LM fitting.
"""
import numpy
from numpy import where, array, log, sqrt, zeros

from ..gexceptions import GMixRangeError
from .random import make_rng

LOWVAL = -numpy.inf
BIGVAL = 9999.0e47


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
            fdiff[i] = self._get_fdiff_scalar(vals[i])

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


class Normal(PriorBase):
    """
    A Normal distribution.

    This class provides an interface consistent with LogNormal.

    parameters
    ----------
    mean: float
        The mean of the Gaussian.
    sigma: float
        The standard deviation of the Gaussian.
    bounds: 2-tuple of floats or None
        The bounds of the parameter. Default of None means no bounds.
    rng: np.random.RandomState or None
        An RNG to use. If None, a new RNG is made using the numpy global RNG
        to generate a seed.

    attributes
    ----------
    mean: float
        The mean of the Gaussian.
    sigma: float
        The standard deviation of the Gaussian.
    """
    def __init__(self, mean, sigma, bounds=None, rng=None):
        super().__init__(rng=rng, bounds=bounds)

        self.mean = mean
        self.sigma = sigma
        self.sinv = 1.0 / sigma
        self.s2inv = 1.0 / sigma ** 2
        self.ndim = 1

    def get_lnprob(self, x):
        """
        Compute -0.5 * ( (x-mean)/sigma )**2.
        """
        diff = self.mean - x
        return -0.5 * diff * diff * self.s2inv

    get_lnprob_scalar = get_lnprob
    get_lnprob_array = get_lnprob

    def get_prob(self, x):
        """
        Compute exp(-0.5 * ( (x-mean)/sigma )**2)

        Note that this function is missing the normalization factor.
        """
        diff = self.mean - x
        lnp = -0.5 * diff * diff * self.s2inv
        return numpy.exp(lnp)

    get_prob_array = get_prob

    def get_prob_scalar(self, x):
        """
        Compute exp(-0.5 * ( (x-mean)/sigma )**2).

        Note that this function is missing the normalization factor.
        """
        from math import exp

        diff = self.mean - x
        lnp = -0.5 * diff * diff * self.s2inv
        return exp(lnp)

    def get_fdiff(self, x):
        """
        Compute sqrt(-2ln(p)) ~ (data - mode)/err for use with LM fitter.
        """
        return (x - self.mean) * self.sinv

    def sample(self, nrand=None, size=None):
        """
        Draw random samples of the prior.

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
        if size is None and nrand is not None:
            # if they have given nrand and not n, use that
            # this keeps the API the same but allows ppl to use the new API of nrand
            size = nrand

        return self.rng.normal(loc=self.mean, scale=self.sigma, size=size,)


class LMBounds(PriorBase):
    """
    Class to hold simple bounds for the leastsqbound version
    of LM.

    The fdiff is always zero, but the bounds will be sent
    to the minimizer.

    parameters
    ----------
    minval: float
        The minimum bound.
    maxval: float
        The maximum bound.
    rng: np.random.RandomState or None
        An RNG to use. If None, a new RNG is made using the numpy global RNG
        to generate a seed.

    attributes
    ----------
    mean: float
        The mean of the uniform distribution.
    sigma: float
        The standard deviation of the uniform distribution.
    """
    def __init__(self, minval, maxval, rng=None):

        super().__init__(rng=rng)

        self.bounds = (minval, maxval)
        self.mean = (minval + maxval) / 2.0
        self.sigma = (maxval - minval) * 0.28  # exact is 1/sqrt(12) ~ 0.28867513459

    def get_fdiff(self, val):
        """
        Compute sqrt(-2ln(p)) ~ (data - mode)/err for use with LM fitter. Always zero.
        """
        return 0.0 * val

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

        return self.rng.uniform(
            low=self.bounds[0], high=self.bounds[1], size=n,
        )


class Bounded1D(PriorBase):
    """
    Wrap a pdf and limit samples to the input bounds.

    parameters
    ----------
    pdf: object
        A PDF object with a `sample` method.
    bounds: 2-tuple of floats
        A 2-tuple of floats.

    attributes
    ----------
    pdf: object
        A PDF object with a `sample` method.
    """
    def __init__(self, pdf, bounds):
        self.pdf = pdf
        self.set_limits(bounds)

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
                "expected bounds to be 2-element sequence, got %s" % (limits,)
            )

        if limits[0] >= limits[1]:
            raise ValueError(
                "bounds[0] must be less than bounds[1], got: %s" % (limits,)
            )
        self.limits = limits
        self.bounds = limits

    def sample(self, nrand=None, size=None):
        """
        Draw random samples of the PDF with the bounds.

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
        if size is None and nrand is not None:
            # if they have given nrand and not n, use that
            # this keeps the API the same but allows ppl to use the new API of nrand
            size = nrand

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


# keep this so that the API stays the same
LimitPDF = Bounded1D


class LogNormal(PriorBase):
    """
    Lognormal distribution

    parameters
    ----------
    mean: float
        such that <x> in linear space is mean.  This implies the mean in log(x)
        is
            <log(x)> = log(mean) - 0.5*log( 1 + sigma**2/mean**2 )
    sigma: float
        such than the variace in linear space is sigma**2.  This implies
        the variance in log space is
            var(log(x)) = log( 1 + sigma**2/mean**2 )
    shift: float
        An optional shift to apply to the samples and the locations for
        evaluating the PDF. The shift is added to samples from the underlying
        log-normal.
    rng: np.random.RandomState or None
        An RNG to use. If None, a new RNG is made using the numpy global RNG
        to generate a seed.

    attributes
    ----------
    shift: float
        An optional shift to apply to the samples and the locations for
        evaluating the PDF. The shift is added to samples from the underlying
        log-normal.
    mean: float
        The linear-space mean <x>.
    sigma: float
        The linear-space standard deviation.
    logmean: float
        The log-space mean.
    logvar: float
        The log-space variance.
    logsigma: float
        The log-space standard deviation.
    logivar: float
        The inverse of the log-space variace.
    mode: float
        The linear-space mode.
    log_mode: float
        The log-space mode.
    lnprob_max: float
        The log of the maximum value of the distribution.
    """
    def __init__(self, mean, sigma, shift=None, rng=None):
        super().__init__(rng=rng)

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
        Get the log-probability of x.
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
        Get the log-probability of x.
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
        Get the probability of x.
        """

        lnp = self.get_lnprob_array(x)
        return numpy.exp(lnp)

    def get_fdiff(self, x):
        """
        Compute sqrt(-2ln(p)) ~ (data - mode)/err for use with LM fitter.
        """
        lnp = self.get_lnprob_scalar(x)
        chi2 = -2 * lnp
        if chi2 < 0.0:
            chi2 = 0.0
        fdiff = sqrt(chi2)
        return fdiff

    def sample(self, nrand=None):
        """
        Draw random samples from the LogNormal.

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
        z = self.rng.normal(size=nrand)
        r = numpy.exp(self.logmean + self.logsigma * z)

        if self.shift is not None:
            r += self.shift

        return r

    def sample_brute(self, nrand=None, maxval=None):
        """
        Draw random samples from the LogNormal using a brute force algorithm.

        This method is used to help check other methods.

        parameters
        ----------
        nrand: int or None
            The number of samples. If None, a single scalar sample is drawn.
            Default is None.
        maxval: float or None
            The maximum value to allow for draws.

        returns
        -------
        samples: scalar or array-like
            The samples with shape (`nrand`,). If `nrand` is None, then a
            scalar is returned.
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
            if self.shift is not None:
                rvals += self.shift

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
        Fit to the input x and y.

        parameters
        ----------
        x: array-like
            The x-values for the fit.
        y: array-like
            The y-values for the fit. Usually p(x).

        returns
        -------
        res: dict
            A dictionary with the best-fit parameters and other information
            from the fit.
        """
        from ..fitting import run_leastsq

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


class Sinh(PriorBase):
    """
    a sinh distribution with mean and scale.

    Currently only supports "fdiff" style usage as a prior,
    e.g. for LM.

    parameters
    ----------
    mean: float
        The mean value where the value of fdiff is zero.
    scale: float
        The value such that fdiff  of `mean` +/- `scale` is +/-1.
    rng: np.random.RandomState or None
        An RNG to use. If None, a new RNG is made using the numpy global RNG
        to generate a seed.

    attributes
    ----------
    mean: float
        The mean value where the value of fdiff is zero.
    scale: float
        The value such that fdiff  of `mean` +/- `scale` is +/-1.
    """
    def __init__(self, mean, scale, rng=None):
        super().__init__(rng=rng)
        self.mean = mean
        self.scale = scale

    def get_fdiff(self, x):
        """
        For use with LM fitter - computes sinh((model-data)/width)
        """
        return numpy.sinh((x - self.mean) / self.scale)

    def sample(self, nrand=None):
        """
        sample around the mean, +/- a scale length

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
    Truncated gaussian between [minval, maxval].

    parameters
    ----------
    mean: float
        The mean of the Gaussian.
    sigma: float
        The standard deviation of the Gaussian.
    minval: float
        The minimum of the distribution.
    maxval: float
        The maximum of the distribution.
    rng: np.random.RandomState or None
        An RNG to use. If None, a new RNG is made using the numpy global RNG
        to generate a seed.

    attributes
    ----------
    mean: float
        The mean of the Gaussian.
    sigma: float
        The standard deviation of the Gaussian.
    minval: float
        The minimum of the distribution.
    maxval: float
        The maximum of the distribution.
    """
    def __init__(self, mean, sigma, minval, maxval, rng=None):
        super().__init__(rng=rng)
        self.mean = mean
        self.sigma = sigma
        self.ivar = 1.0 / sigma ** 2
        self.sinv = 1.0 / sigma
        self.minval = minval
        self.maxval = maxval

    def get_lnprob_scalar(self, x):
        """
        get the log probability of the point - raises if not in [minval, maxval]
        """
        if x < self.minval or x > self.maxval:
            raise GMixRangeError("value out of range")
        diff = x - self.mean
        return -0.5 * diff * diff * self.ivar

    def get_lnprob_array(self, x):
        """
        get the log probability of an array - raises if not in [minval, maxval]
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
        Compute sqrt(-2ln(p)) ~ (data - mode)/err for use with LM fitter.
        """
        if x < self.minval or x > self.maxval:
            raise GMixRangeError("value out of range")
        return (x - self.mean) * self.sinv

    def sample(self, nrand=None):
        """
        Sample from the truncated Gaussian.

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


class Student(PriorBase):
    """
    A Student's t-distribution.

    parameters
    ----------
    mean: float
        The mean of the distribution.
    sigma: float
        The scale of the distribution. Not Std(x).
    rng: np.random.RandomState or None
        An RNG to use. If None, a new RNG is made using the numpy global RNG
        to generate a seed.

    attributes
    ----------
    mean: float
        The mean of the distribution.
    sigma: float
        The scale of the distribution. Not Std(x).
    tdist: scipy.stats.t
        The underlying scipy distribution.
    """
    def __init__(self, mean, sigma, rng=None):
        super().__init__(rng=rng)
        self.reset(mean, sigma, rng=self.rng)

    def reset(self, mean, sigma, rng=None):
        """
        complete reset of mean sigma and the RNG

        parameters
        ----------
        mean: float
            The mean of the distribution.
        sigma: float
            The scale of the distribution. Not Std(x).
        rng: np.random.RandomState or None
            An RNG to use. If None, a new RNG is made using the numpy global RNG
            to generate a seed.
        """
        import scipy.stats

        assert rng is not None
        self.rng = rng

        self.mean = mean
        self.sigma = sigma

        self.tdist = scipy.stats.t(1.0, loc=mean, scale=sigma)

    def sample(self, nrand=None):
        """
        sample from the distribution

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
        return self.tdist.rvs(size=nrand, random_state=self.rng)

    def get_lnprob_array(self, x):
        """
        get ln(prob) for an array
        """
        return self.tdist.logpdf(x)

    get_lnprob_scalar = get_lnprob_array


class StudentPositive(Student):
    """
    A Student's t-distribution truncated to only have positive values.

    parameters
    ----------
    mean: float
        The mean of the distribution.
    sigma: float
        The scale of the distribution. Not Std(x).
    rng: np.random.RandomState or None
        An RNG to use. If None, a new RNG is made using the numpy global RNG
        to generate a seed.

    attributes
    ----------
    mean: float
        The mean of the distribution.
    sigma: float
        The scale of the distribution. Not Std(x).
    tdist: scipy.stats.t
        The underlying scipy distribution.
    """

    def sample(self, nrand=None):
        """
        sample from the distribution

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
        if nrand is None:
            nrand = 1
            is_scalar = True
        else:
            is_scalar = False
        vals = numpy.zeros(nrand)

        nleft = nrand
        ngood = 0
        while nleft > 0:
            r = super().sample(nleft)

            (w,) = numpy.where(r > 0.0)
            nkeep = w.size
            if nkeep > 0:
                vals[ngood:ngood + nkeep] = r[w]
                nleft -= nkeep
                ngood += nkeep

        if is_scalar:
            vals = vals[0]

        return vals

    def get_lnprob_scalar(self, x):
        """
        get ln(prob) at a point.
        """
        if x <= 0:
            raise GMixRangeError("value less than zero")
        return super().get_lnprob_scalar(x)

    def get_lnprob_array(self, x):
        """
        get ln(prob) for an array
        """
        x = numpy.array(x, dtype="f8", copy=False)

        (w,) = numpy.where(x <= 0)
        if w.size != 0:
            raise GMixRangeError("values less than zero")
        return super().get_lnprob_array(x)
