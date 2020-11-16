

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
# MRB: I've gone through this file to here
# I am going to stop and write tests, then split this module into pieces
# and then do docs for the rest.


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
