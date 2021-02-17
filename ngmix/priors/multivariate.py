"""
Convention is that all priors should have peak ln(prob)==0. This
helps use in priors for LM fitting
"""
import numpy
from numpy import where, array, exp, log, sqrt, zeros, diag, cos, sin

from ..gexceptions import GMixRangeError
from .random import make_rng
from .priors import PriorBase, LOWVAL, Student


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


class TruncatedGaussianPolar(PriorBase):
    """
    Truncated gaussian on a circle
    """

    def __init__(self, mean1, mean2, sigma1, sigma2, maxval, rng=None):
        raise NotImplementedError('this is wrong')
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


class Student2D(object):
    def __init__(self, mean1, mean2, sigma1, sigma2, rng=None):
        """
        circular student
        sigma is not std(x)
        """
        self.reset(mean1, mean2, sigma1, sigma2, rng=rng)

    def reset(self, mean1, mean2, sigma1, sigma2, rng=None):
        """
        complete reset
        """
        assert rng is not None

        self.rng = rng
        self.mean1 = mean1
        self.sigma1 = sigma1
        self.mean2 = mean2
        self.sigma2 = sigma2

        self.tdist1 = Student(mean1, sigma1, rng=rng)
        self.tdist2 = Student(mean2, sigma2, rng=rng)

    def sample(self, nrand=None):
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
        raise NotImplementedError('this is wrong')
        self.reset(mean1, mean2, sigma1, sigma2, maxval)

    def reset(self, mean1, mean2, sigma1, sigma2, maxval):
        super().reset(mean1, mean2, sigma1, sigma2)
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
            r1, r2 = super().sample(nleft)

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
        lnp = super().get_lnprob_scalar(x1, x2)
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

        lnp = super().get_lnprob_array(x1, x2)
        return lnp


class CenPrior(PriorBase):
    """
    Independent gaussians in each dimension
    """

    def __init__(self, cen1, cen2, sigma1, sigma2, rng=None):
        super().__init__(rng=rng)

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

    def sample(self, nrand=None):
        """
        Get a single sample or arrays
        """

        rng = self.rng

        rand1 = rng.normal(loc=self.cen1, scale=self.sigma1, size=nrand)
        rand2 = rng.normal(loc=self.cen2, scale=self.sigma2, size=nrand)

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

    def sample1d(self, nrand=None):
        """
        Get samples in 1-d radius
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
        Get samples.  Send no args to get a scalar.
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


class ZAnnulus(ZDisk2D):
    """
    uniform over an annulus

    Note get_lnprob_scalar1d and get_prob_scalar1d and 2d are already
    part of base class
    """

    def __init__(self, rmin, rmax, rng=None):

        assert rmin < rmax

        self.rmin = rmin
        super().__init__(
            rmax, rng=rng,
        )

    def sample1d(self, nrand=None):
        if nrand is None:
            nrand = 1
            is_scalar = True
        else:
            is_scalar = False

        r = numpy.zeros(nrand)
        ngood = 0
        nleft = nrand

        while True:
            rtmp = super().sample1d(nleft)

            (w,) = numpy.where(rtmp > self.rmin)
            if w.size > 0:
                r[ngood:ngood + w.size] = rtmp[w]
                ngood += w.size
                nleft -= w.size

            if ngood == nrand:
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

    def __init__(self, radius=1.0, rolloff_point=0.98, width=0.005):
        self.rolloff_point = rolloff_point
        self.radius = radius
        self.radius_sq = radius ** 2

        self.width = width

    def get_prob_scalar1d(self, val):
        """
        scalar only
        """
        from .._gmix import erf

        if val > self.radius:
            return 0.0

        erf_val = erf((val - self.rolloff_point) / self.width)
        retval = 0.5 * (1.0 + erf_val)

        return retval

    def get_lnprob_scalar1d(self, val):
        """
        scalar only
        """
        from .._gmix import erf

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
        from .._gmix import erf_array

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
    cuts off like arctanh, not sure this is ever used in the wild, should
    remove
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
