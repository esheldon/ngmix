"""
Convention is that all priors should have peak ln(prob)==0. This
helps use in priors for LM fitting.
"""
import numpy as np

from ..gexceptions import GMixRangeError
from .random import make_rng
from ..defaults import LOWVAL, copy_if_needed


class PriorBase(object):
    """
    Base object for priors.

    Parameters
    ----------
    bounds: 2-tuple of floats or None
        The bounds of the parameter. Default of None means no bounds.
    rng: np.random.RandomState
        An random number generator (RNG) to use.

    attributes
    ----------
    bounds: 2-tuple of floats or None
        The bounds of the parameter. Default of None means no bounds.
    rng: np.random.RandomState
        The RNG.

    methods
    -------
    has_bounds()
        Returns True if the object has bounds defined and they are non-None, False
        otherwise.
    """
    def __init__(self, rng, bounds=None):
        assert rng is not None, 'rng is a required argument'

        self.bounds = bounds
        self.rng = make_rng(rng=rng)

    def has_bounds(self):
        """
        Returns True if the object has a bounds defined, False otherwise.
        """
        return hasattr(self, "bounds") and self.bounds is not None


class FlatPrior(PriorBase):
    """
    A flat prior between `minval` and `maxval`.

    Parameters
    ----------
    minval: float
        The minimum value of the allowed range.
    maxval: float
        The maximum value of the allowed range.
    rng: np.random.RandomState
        An random number generator (RNG) to use.
    """
    def __init__(self, minval, maxval, rng):
        super().__init__(rng=rng)

        self.minval = minval
        self.maxval = maxval

    def get_prob_scalar(self, val):
        """
        Returns 1 if the value is in [minval, maxval] or raises a GMixRangeError

        Parameters
        ----------
        val: number
            The location at which to evaluate
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

        Parameters
        ----------
        val: number
            The location at which to evaluate
        """
        retval = 0.0
        if val < self.minval or val > self.maxval:
            raise GMixRangeError(
                "value %s out of range: "
                "[%s,%s]" % (val, self.minval, self.maxval)
            )
        return retval

    def get_prob_array(self, vals):
        """
        Returns 1 if the value is in [minval, maxval] or raises a GMixRangeError

        Parameters
        ----------
        vals: array
            The locations at which to evaluate
        """
        retval = 1.0

        w, = np.where((vals < self.minval) | (vals > self.maxval))
        if w.size > 0:
            raise GMixRangeError(
                "values were out of range: "
                "[%s,%s]" % (self.minval, self.maxval)
            )

        return vals*0 + retval

    def get_lnprob_array(self, vals):
        """
        Returns 0.0 if the value is in [minval, maxval] or raises a GMixRangeError

        Parameters
        ----------
        vals: array
            The location at which to evaluate
        """
        retval = 0.0

        w, = np.where((vals < self.minval) | (vals > self.maxval))
        if w.size > 0:
            raise GMixRangeError(
                "values were out of range: "
                "[%s,%s]" % (self.minval, self.maxval)
            )

        return retval

    def get_fdiff(self, val):
        """
        Compute sqrt(-2ln(p)) ~ (data - mode)/err for using with LM fitters.

        Parameters
        ----------
        val: number
            The location at which to evaluate
        """
        retval = 0.0
        if val < self.minval or val > self.maxval:
            raise GMixRangeError(
                "value %s out of range: "
                "[%s,%s]" % (val, self.minval, self.maxval)
            )
        return retval

    def sample(self, nrand=None):
        """
        Returns samples uniformly on the interval.

        Parameters
        ----------
        nrand: int or None
            The number of samples. If None, a single scalar sample is drawn.
            Default is None.

        Returns
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

        rvals = self.rng.uniform(size=nrand)
        rvals = self.minval + (self.maxval - self.minval) * rvals

        if is_scalar:
            rvals = rvals[0]

        return rvals


class TwoSidedErf(PriorBase):
    """
    A two-sided error function that evaluates to 1 in the middle, zero at
    extremes.

    A limitation seems to be the accuracy of the erf implementation.

    Parameters
    ----------
    minval: float
        The minimum value. This is where p(x) = 0.5 at the lower end.
    width_at_min: float
        The width of the transition region from 0 to 1 at the lower end.
    maxval: float
        The maximum value. This is where p(x) = 0.5 at the upper end.
    width_at_max: float
        The width of the transition region from 1 to 0 at the upper end.
    rng: np.random.RandomState
        An random number generator (RNG) to use.
    """
    def __init__(self, minval, width_at_min, maxval, width_at_max, rng):
        super().__init__(rng=rng)

        self.minval = minval
        self.width_at_min = width_at_min

        self.maxval = maxval
        self.width_at_max = width_at_max

    def get_prob_scalar(self, val):
        """
        get the probability of the point

        Parameters
        ----------
        val: number
            The location at which to evaluate
        """
        from math import erf

        p1 = 0.5 * erf((self.maxval - val) / self.width_at_max)
        p2 = 0.5 * erf((val - self.minval) / self.width_at_min)

        return p1 + p2

    def get_lnprob_scalar(self, val):
        """
        get the log probability of the point

        Parameters
        ----------
        val: number
            The location at which to evaluate
        """

        p = self.get_prob_scalar(val)

        if p <= 0.0:
            lnp = LOWVAL
        else:
            lnp = np.log(p)
        return lnp

    def get_prob_array(self, vals):
        """
        get the probability of a set of points

        Parameters
        ----------
        vals: number
            The locations at which to evaluate
        """

        vals = np.array(vals, ndmin=1, dtype="f8", copy=copy_if_needed)
        pvals = np.zeros(vals.size)

        for i in range(vals.size):
            pvals[i] = self.get_prob_scalar(vals[i])

        return pvals

    def get_lnprob_array(self, vals):
        """
        get the log probability of a set of points

        Parameters
        ----------
        vals: array
            The locations at which to evaluate
        """

        p = self.get_prob_array(vals)

        lnp = np.zeros(p.size) + LOWVAL
        (w,) = np.where(p > 0.0)
        if w.size > 0:
            lnp[w] = np.log(p[w])
        return lnp

    def get_fdiff(self, val):
        """
        Compute sqrt(-2ln(p)) ~ (data - mode)/err for using with LM fitters.

        Parameters
        ----------
        val: number
            The location at which to evaluate
        """
        if isinstance(val, np.ndarray):
            return self._get_fdiff_array(val)
        else:
            return self._get_fdiff_scalar(val)

    def _get_fdiff_array(self, vals):
        """
        get diff array by evaluating one at a time

        Parameters
        ----------
        vals: number
            The locations at which to evaluate
        """
        vals = np.array(vals, ndmin=1, dtype="f8", copy=copy_if_needed)
        fdiff = np.zeros(vals.size)

        for i in range(vals.size):
            fdiff[i] = self._get_fdiff_scalar(vals[i])

        return fdiff

    def _get_fdiff_scalar(self, val):
        """
        get something similar to a (model-data)/err.  Note however that with
        the current implementation, the *sign* of the difference is lost in
        this case.

        Parameters
        ----------
        val: number
            The location at which to evaluate
        """

        p = self.get_lnprob_scalar(val)

        p = -2 * p
        if p < 0.0:
            p = 0.0
        return np.sqrt(p)

    def sample(self, nrand=None):
        """
        Draw random samples of the prior.

        Note this function is not perfect in that it only goes from
        -5,5 sigma past each side.

        Parameters
        ----------
        nrand: int or None
            The number of samples. If None, a single scalar sample is drawn.
            Default is None.

        Returns
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

        rvals = np.zeros(nrand)

        ngood = 0
        nleft = nrand
        while ngood < nrand:
            randx = rng.uniform(low=xmin, high=xmax, size=nleft)

            pvals = self.get_prob_array(randx)
            randy = rng.uniform(size=nleft)

            (w,) = np.where(randy < pvals)
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

    Parameters
    ----------
    mean: float
        The mean of the Gaussian.
    sigma: float
        The standard deviation of the Gaussian.
    bounds: 2-tuple of floats or None
        The bounds of the parameter. Default of None means no bounds.
    rng: np.random.RandomState
        An random number generator (RNG) to use.

    attributes
    ----------
    mean: float
        The mean of the Gaussian.
    sigma: float
        The standard deviation of the Gaussian.
    """
    def __init__(self, mean, sigma, rng, bounds=None):
        super().__init__(rng=rng, bounds=bounds)

        self.mean = mean
        self.sigma = sigma
        self.sinv = 1.0 / sigma
        self.s2inv = 1.0 / sigma ** 2
        self.ndim = 1

    def get_lnprob(self, val):
        """
        Compute -0.5 * ( (val-mean)/sigma )**2.

        Parameters
        ----------
        val: number
            Location at which to evaluate
        """
        diff = self.mean - val
        return -0.5 * diff * diff * self.s2inv

    get_lnprob_scalar = get_lnprob
    get_lnprob_array = get_lnprob

    def get_prob(self, val):
        """
        Compute exp(-0.5 * ( (val-mean)/sigma )**2)

        Note that this function is missing the normalization factor.

        Parameters
        ----------
        val: number
            Location at which to evaluate
        """
        diff = self.mean - val
        lnp = -0.5 * diff * diff * self.s2inv
        return np.exp(lnp)

    get_prob_array = get_prob

    def get_prob_scalar(self, val):
        """
        Compute exp(-0.5 * ( (x-mean)/sigma )**2).

        Note that this function is missing the normalization factor.

        Parameters
        ----------
        val: number
            The location at which to evaluate
        """
        from math import exp

        diff = self.mean - val
        lnp = -0.5 * diff * diff * self.s2inv
        return exp(lnp)

    def get_fdiff(self, val):
        """
        Compute sqrt(-2ln(p)) ~ (data - mode)/err for use with LM fitter.

        Parameters
        ----------
        val: number
            The location at which to evaluate
        """
        return (val - self.mean) * self.sinv

    def sample(self, nrand=None, size=None):
        """
        Draw random samples of the prior.

        Parameters
        ----------
        nrand: int or None
            The number of samples. If None, a single scalar sample is drawn.
            Default is None.

        Returns
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

    Parameters
    ----------
    minval: float
        The minimum bound.
    maxval: float
        The maximum bound.
    rng: np.random.RandomState
        An random number generator (RNG) to use.

    attributes
    ----------
    mean: float
        The mean of the uniform distribution.
    sigma: float
        The standard deviation of the uniform distribution.
    """
    def __init__(self, minval, maxval, rng):

        super().__init__(rng)

        self.bounds = (minval, maxval)
        self.mean = (minval + maxval) / 2.0
        self.sigma = (maxval - minval) * 0.28  # exact is 1/sqrt(12) ~ 0.28867513459

    def get_fdiff(self, val):
        """
        Compute sqrt(-2ln(p)) ~ (data - mode)/err for use with LM fitter. Always zero.

        Parameters
        ----------
        val: number
            The location at which to evaluate
        """
        return 0.0 * val

    def sample(self, nrand=None):
        """
        Returns samples uniformly on the interval.

        Parameters
        ----------
        nrand: int or None
            The number of samples. If None, a single scalar sample is drawn.
            Default is None.

        Returns
        -------
        samples: scalar or array-like
            The samples with shape (`nrand`,). If `nrand` is None, then a
            scalar is returned.
        """
        return self.rng.uniform(
            low=self.bounds[0], high=self.bounds[1], size=nrand,
        )


class Bounded1D(PriorBase):
    """
    Wrap a pdf and limit samples to the input bounds.

    Parameters
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

        Parameters
        ----------
        limits: sequence
            Limits to set
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

        Parameters
        ----------
        nrand: int or None
            The number of samples. If None, a single scalar sample is drawn.
            Default is None.

        Returns
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

        values = np.zeros(nval)
        ngood = 0
        nleft = nval

        while nleft > 0:
            tmp = self.pdf.sample(nleft)

            (w,) = np.where((tmp > bounds[0]) & (tmp < bounds[1]))

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

    Parameters
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
    rng: np.random.RandomState
        An random number generator (RNG) to use.

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
    def __init__(self, mean, sigma, rng, shift=None):
        super().__init__(rng=rng)

        if mean <= 0:
            raise ValueError("mean %s is < 0" % mean)

        self.shift = shift
        self.mean = mean
        self.sigma = sigma

        logmean = np.log(self.mean) - 0.5 * np.log(
            1 + self.sigma ** 2 / self.mean ** 2
        )
        logvar = np.log(1 + self.sigma ** 2 / self.mean ** 2)
        logsigma = np.sqrt(logvar)
        logivar = 1.0 / logvar

        self.logmean = logmean
        self.logvar = logvar
        self.logsigma = logsigma
        self.logivar = logivar

        log_mode = self.logmean - self.logvar
        self.mode = np.exp(log_mode)
        chi2 = self.logivar * (log_mode - self.logmean) ** 2

        # subtract mode to make max 0.0
        self.lnprob_max = -0.5 * chi2 - log_mode

        self.log_mode = log_mode

    def get_lnprob_scalar(self, val):
        """
        Get the log-probability of val

        Parameters
        ----------
        val: number
            The location at which to evaluate
        """
        if self.shift is not None:
            val = val - self.shift

        if val <= 0:
            raise GMixRangeError("values of val must be > 0")

        logval = np.log(val)
        chi2 = self.logivar * (logval - self.logmean) ** 2

        # subtract mode to make max 0.0
        lnprob = -0.5 * chi2 - logval - self.lnprob_max

        return lnprob

    def get_lnprob_array(self, vals):
        """
        Get the log-probability of vals.

        Parameters
        ----------
        vals: array
            The locations at which to evaluate
        """
        vals = np.array(vals, dtype="f8", copy=copy_if_needed)
        if self.shift is not None:
            vals = vals - self.shift

        (w,) = np.where(vals <= 0)
        if w.size > 0:
            raise GMixRangeError("values must be > 0")

        logvals = np.log(vals)
        chi2 = self.logivar * (logvals - self.logmean) ** 2

        # subtract mode to make max 0.0
        lnprob = -0.5 * chi2 - logvals - self.lnprob_max

        return lnprob

    def get_prob_scalar(self, val):
        """
        Get the probability of x.

        Parameters
        ----------
        val: number
            The location at which to evaluate
        """

        lnprob = self.get_lnprob_scalar(val)
        return np.exp(lnprob)

    def get_prob_array(self, vals):
        """
        Get the probability of val.

        Parameters
        ----------
        vals: number
            The locations at which to evaluate
        """

        lnp = self.get_lnprob_array(vals)
        return np.exp(lnp)

    def get_fdiff(self, val):
        """
        Compute sqrt(-2ln(p)) ~ (data - mode)/err for use with LM fitter.

        Parameters
        ----------
        val: number
            The location at which to evaluate
        """
        lnp = self.get_lnprob_scalar(val)
        chi2 = -2 * lnp
        if chi2 < 0.0:
            chi2 = 0.0
        fdiff = np.sqrt(chi2)
        return fdiff

    def sample(self, nrand=None):
        """
        Draw random samples from the LogNormal.

        Parameters
        ----------
        nrand: int or None
            The number of samples. If None, a single scalar sample is drawn.
            Default is None.

        Returns
        -------
        samples: scalar or array-like
            The samples with shape (`nrand`,). If `nrand` is None, then a
            scalar is returned.
        """
        z = self.rng.normal(size=nrand)
        r = np.exp(self.logmean + self.logsigma * z)

        if self.shift is not None:
            r += self.shift

        return r

    def sample_brute(self, nrand=None, maxval=None):
        """
        Draw random samples from the LogNormal using a brute force algorithm.

        This method is used to help check other methods.

        Parameters
        ----------
        nrand: int or None
            The number of samples. If None, a single scalar sample is drawn.
            Default is None.
        maxval: float or None
            The maximum value to allow for draws.

        Returns
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

        samples = np.zeros(nrand)

        ngood = 0
        nleft = nrand
        while ngood < nrand:

            rvals = maxval * rng.rand(nleft)
            if self.shift is not None:
                rvals += self.shift

            # between 0,1 which is max for prob
            h = rng.uniform(size=nleft)

            pvals = self.get_prob_array(rvals)

            (w,) = np.where(h < pvals)
            if w.size > 0:
                samples[ngood:ngood + w.size] = rvals[w]
                ngood += w.size
                nleft -= w.size

        if is_scalar:
            samples = samples[0]

        return samples

    def _calc_fdiff(self, pars):
        """
        calculate fdiff for fitting the parameters of the distribution

        Parameters
        ----------
        pars: array
            parameters at which to evaluate
        """
        try:
            ln = LogNormal(pars[0], pars[1], rng=self.rng)
            model = ln.get_prob_array(self._fitx) * pars[2]
        except (GMixRangeError, ValueError):
            return self._fity * 0 - np.inf

        fdiff = model - self._fity
        return fdiff

    def fit(self, x, y):
        """
        Fit to the input x and y.

        Parameters
        ----------
        x: array-like
            The x-values for the fit.
        y: array-like
            The y-values for the fit. Usually p(x).

        Returns
        -------
        res: dict
            A dictionary with the best-fit parameters and other information
            from the fit.
        """
        from ..fitting.leastsqbound import run_leastsq

        rng = self.rng

        self._fitx = x
        self._fity = y

        for i in range(4):
            f1, f2, f3 = 1.0 + rng.uniform(low=0.1, high=0.1, size=3)
            guess = np.array([x.mean() * f1, x.std() * f2, y.mean() * f3])

            res = run_leastsq(self._calc_fdiff, guess, 0,)
            if res["flags"] == 0:
                break

        return res


class Sinh(PriorBase):
    """
    a sinh distribution with mean and scale.

    Currently only supports "fdiff" style usage as a prior,
    e.g. for LM.

    Parameters
    ----------
    mean: float
        The mean value where the value of fdiff is zero.
    scale: float
        The value such that fdiff  of `mean` +/- `scale` is +/-1.
    rng: np.random.RandomState
        An random number generator (RNG) to use.

    attributes
    ----------
    mean: float
        The mean value where the value of fdiff is zero.
    scale: float
        The value such that fdiff  of `mean` +/- `scale` is +/-1.
    """
    def __init__(self, mean, scale, rng):
        super().__init__(rng=rng)
        self.mean = mean
        self.scale = scale

    def get_fdiff(self, val):
        """
        For use with LM fitter - computes sinh((model-data)/width)

        Parameters
        ----------
        val: number
            The location at which to evaluate
        """
        return np.sinh((val - self.mean) / self.scale)

    def sample(self, nrand=None):
        """
        sample around the mean, +/- a scale length

        Parameters
        ----------
        nrand: int or None
            The number of samples. If None, a single scalar sample is drawn.
            Default is None.

        Returns
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

    Parameters
    ----------
    mean: float
        The mean of the Gaussian.
    sigma: float
        The standard deviation of the Gaussian.
    minval: float
        The minimum of the distribution.
    maxval: float
        The maximum of the distribution.
    rng: np.random.RandomState
        An random number generator (RNG) to use.

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
    def __init__(self, mean, sigma, minval, maxval, rng):
        super().__init__(rng=rng)
        self.mean = mean
        self.sigma = sigma
        self.ivar = 1.0 / sigma ** 2
        self.sinv = 1.0 / sigma
        self.minval = minval
        self.maxval = maxval

    def get_lnprob_scalar(self, val):
        """
        get the log probability of the point - raises if not in [minval, maxval]

        Parameters
        ----------
        val: number
            The location at which to evaluate
        """
        if val < self.minval or val > self.maxval:
            raise GMixRangeError("value out of range")
        diff = val - self.mean
        return -0.5 * diff * diff * self.ivar

    def get_lnprob_array(self, val):
        """
        get the log probability of an array - raises if not in [minval, maval]

        Parameters
        ----------
        val: array
            The locations at which to evaluate
        """
        lnp = np.zeros(val.size)
        lnp -= np.inf
        (w,) = np.where((val > self.minval) & (val < self.maxval))
        if w.size > 0:
            diff = val[w] - self.mean
            lnp[w] = -0.5 * diff * diff * self.ivar

        return lnp

    def get_fdiff(self, val):
        """
        Compute sqrt(-2ln(p)) ~ (data - mode)/err for use with LM fitter.

        Parameters
        ----------
        val: number
            The location at which to evaluate
        """
        if val < self.minval or val > self.maxval:
            raise GMixRangeError("value out of range")
        return (val - self.mean) * self.sinv

    def sample(self, nrand=None):
        """
        Sample from the truncated Gaussian.

        Parameters
        ----------
        nrand: int or None
            The number of samples. If None, a single scalar sample is drawn.
            Default is None.

        Returns
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

        vals = np.zeros(nrand)

        ngood = 0
        nleft = nrand
        while ngood < nrand:

            tvals = rng.normal(loc=self.mean, scale=self.sigma, size=nleft)

            (w,) = np.where((tvals > self.minval) & (tvals < self.maxval))
            if w.size > 0:
                vals[ngood:ngood + w.size] = tvals[w]
                ngood += w.size
                nleft -= w.size

        if is_scalar:
            vals = vals[0]

        return vals
