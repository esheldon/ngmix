

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
