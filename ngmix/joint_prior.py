from numpy import zeros, exp, sqrt


class PriorSimpleSep(object):
    """
    Separate priors on each parameter

    Parameters
    ----------
    cen_prior:
        The center prior
    g_prior:
        The prior on g (g1,g2).
    T_prior:
        Prior on T or some size parameter
    F_prior:
        Prior on Flux.  Can be a list for a multi-band prior.
    """

    def __init__(self, cen_prior, g_prior, T_prior, F_prior):

        # print("JointPriorSimpleSep")

        self.cen_prior = cen_prior
        self.g_prior = g_prior
        self.T_prior = T_prior

        if isinstance(F_prior, list):
            self.nband = len(F_prior)
        else:
            self.nband = 1
            F_prior = [F_prior]

        self.F_priors = F_prior

        self.set_bounds()

    def set_bounds(self):
        """
        set possibe bounds
        """
        bounds = [
            (None, None),  # c1
            (None, None),  # c2
            (None, None),  # g1
            (None, None),  # g2
            # (-1.0,1.0),  # g1
            # (-1.0,1.0),  # g2
        ]

        allp = [self.T_prior] + self.F_priors

        some_have_bounds = False
        for i, p in enumerate(allp):
            if p.has_bounds():
                some_have_bounds = True
                bounds.append((p.bounds[0], p.bounds[1]))
            else:
                bounds.append((None, None))

        if not some_have_bounds:
            bounds = None

        self.bounds = bounds

    def get_widths(self, n=10000):
        """
        estimate the width in each dimension

        Parameters
        ----------
        n: int, optional
            Number of samples to draw
        """
        if not hasattr(self, "_sigma_estimates"):
            samples = self.sample(n)
            sigmas = samples.std(axis=0)

            # for e1,e2 we want to allow this a bit bigger
            # for very small objects.  Steps for MH could be
            # as large as half this
            sigmas[2] = 2.0
            sigmas[3] = 2.0

            self._sigma_estimates = sigmas

        return self._sigma_estimates

    def get_prob_scalar(self, pars):
        """
        probability for scalar input (meaning one point)
        """

        lnp = self.get_lnprob_scalar(pars)
        p = exp(lnp)
        return p

    def get_lnprob_scalar(self, pars):
        """
        log probability for scalar input (meaning one point)
        """

        lnp = self.cen_prior.get_lnprob_scalar(pars[0], pars[1])
        lnp += self.g_prior.get_lnprob_scalar2d(pars[2], pars[3])
        lnp += self.T_prior.get_lnprob_scalar(pars[4])

        for i, F_prior in enumerate(self.F_priors):
            lnp += F_prior.get_lnprob_scalar(pars[5 + i])

        return lnp

    def fill_fdiff(self, pars, fdiff):
        """
        set sqrt(-2ln(p)) ~ (model-data)/err
        """
        index = 0

        # fdiff[index] = self.cen_prior.get_lnprob_scalar(pars[0],pars[1])

        lnp1, lnp2 = self.cen_prior.get_lnprob_scalar_sep(pars[0], pars[1])

        fdiff[index] = lnp1
        index += 1
        fdiff[index] = lnp2
        index += 1

        fdiff[index] = self.g_prior.get_lnprob_scalar2d(pars[2], pars[3])
        index += 1
        fdiff[index] = self.T_prior.get_lnprob_scalar(pars[4])
        index += 1

        for i in range(self.nband):
            F_prior = self.F_priors[i]
            fdiff[index] = F_prior.get_lnprob_scalar(pars[5 + i])
            index += 1

        chi2 = -2 * fdiff[0:index]
        chi2.clip(min=0.0, max=None, out=chi2)
        fdiff[0:index] = sqrt(chi2)

        return index

    def get_prob_array(self, pars):
        """
        probability for array input [N,ndims]
        """

        lnp = self.get_lnprob_array(pars)
        p = exp(lnp)

        return p

    def get_lnprob_array(self, pars):
        """
        log probability for array input [N,ndims]
        """

        lnp = self.cen_prior.get_lnprob_array(pars[:, 0], pars[:, 1])
        lnp += self.g_prior.get_lnprob_array2d(pars[:, 2], pars[:, 3])
        lnp += self.T_prior.get_lnprob_array(pars[:, 4])

        for i in range(self.nband):
            F_prior = self.F_priors[i]
            lnp += F_prior.get_lnprob_array(pars[:, 5 + i])

        return lnp

    def sample(self, n=None):
        """
        Get random samples
        """

        if n is None:
            is_scalar = True
            n = 1
        else:
            is_scalar = False

        samples = zeros((n, 5 + self.nband))

        cen1, cen2 = self.cen_prior.sample(n)
        g1, g2 = self.g_prior.sample2d(n)
        T = self.T_prior.sample(n)

        samples[:, 0] = cen1
        samples[:, 1] = cen2
        samples[:, 2] = g1
        samples[:, 3] = g2
        samples[:, 4] = T

        for i in range(self.nband):
            F_prior = self.F_priors[i]
            F = F_prior.sample(n)
            samples[:, 5 + i] = F

        if is_scalar:
            samples = samples[0, :]
        return samples

    def __repr__(self):
        reps = []
        reps += [str(self.cen_prior), str(self.g_prior), str(self.T_prior)]

        for p in self.F_priors:
            reps.append(str(p))

        rep = "\n".join(reps)
        return rep


class PriorBDSep(PriorSimpleSep):
    """
    Separate priors on each parameter

    Parameters
    ----------
    cen_prior:
        The center prior
    g_prior:
        The prior on g (g1,g2).
    T_prior:
        Prior on T or some size parameter
    logTratio:
        Prior on Td/Te
    fracdev_prior:
        Prior on fracdev for bulge+disk
    F_prior:
        Prior on Flux.  Can be a list for a multi-band prior.
    """

    def __init__(
        self,
        cen_prior,
        g_prior,
        T_prior,
        logTratio_prior,
        fracdev_prior,
        F_prior,
    ):

        # print("JointPriorSimpleSep")

        self.cen_prior = cen_prior
        self.g_prior = g_prior
        self.T_prior = T_prior
        self.logTratio_prior = logTratio_prior
        self.fracdev_prior = fracdev_prior

        if isinstance(F_prior, (list, tuple)):
            self.nband = len(F_prior)
        else:
            self.nband = 1
            F_prior = [F_prior]

        self.F_priors = F_prior

        self.set_bounds()

    def set_bounds(self):
        """
        set possibe bounds
        """
        bounds = [
            (None, None),  # c1
            (None, None),  # c2
            (None, None),  # g1
            (None, None),  # g2
        ]

        allp = [
            self.T_prior,
            self.logTratio_prior,
            self.fracdev_prior,
        ] + self.F_priors

        some_have_bounds = False
        for i, p in enumerate(allp):
            if p.has_bounds():
                some_have_bounds = True
                bounds.append((p.bounds[0], p.bounds[1]))
            else:
                bounds.append((None, None))

        if not some_have_bounds:
            bounds = None

        self.bounds = bounds

    def get_lnprob_scalar(self, pars):
        """
        log probability for scalar input (meaning one point)
        """

        lnp = self.cen_prior.get_lnprob_scalar(pars[0], pars[1])
        lnp += self.g_prior.get_lnprob_scalar2d(pars[2], pars[3])
        lnp += self.T_prior.get_lnprob_scalar(pars[4])
        lnp += self.logTratio_prior.get_lnprob_scalar(pars[5])
        lnp += self.fracdev_prior.get_lnprob_scalar(pars[6])

        for i, F_prior in enumerate(self.F_priors):
            lnp += F_prior.get_lnprob_scalar(pars[7 + i])

        return lnp

    def fill_fdiff(self, pars, fdiff):
        """
        (model-data)/err
        but "data" here is the central value of a prior.
        """
        index = 0

        # fdiff[index] = self.cen_prior.get_lnprob_scalar(pars[0],pars[1])

        fdiff1, fdiff2 = self.cen_prior.get_fdiff(pars[0], pars[1])

        fdiff[index] = fdiff1
        index += 1
        fdiff[index] = fdiff2
        index += 1

        fdiff[index] = self.g_prior.get_fdiff(pars[2], pars[3])
        index += 1
        fdiff[index] = self.T_prior.get_fdiff(pars[4])
        index += 1

        fdiff[index] = self.logTratio_prior.get_fdiff(pars[5])
        index += 1

        fdiff[index] = self.fracdev_prior.get_fdiff(pars[6])
        index += 1

        for i in range(self.nband):
            F_prior = self.F_priors[i]
            fdiff[index] = F_prior.get_fdiff(pars[7 + i])
            index += 1

        return index

    def get_lnprob_array(self, pars):
        """
        log probability for array input [N,ndims]
        """

        lnp = self.cen_prior.get_lnprob_array(pars[:, 0], pars[:, 1])
        lnp += self.g_prior.get_lnprob_array2d(pars[:, 2], pars[:, 3])
        lnp += self.T_prior.get_lnprob_array(pars[:, 4])
        lnp += self.logTratio_prior.get_lnprob_array(pars[:, 5])
        lnp += self.fracdev_prior.get_lnprob_array(pars[:, 6])

        for i in range(self.nband):
            F_prior = self.F_priors[i]
            lnp += F_prior.get_lnprob_array(pars[:, 7 + i])

        return lnp

    def sample(self, n=None):
        """
        Get random samples
        """

        if n is None:
            is_scalar = True
            n = 1
        else:
            is_scalar = False

        samples = zeros((n, 7 + self.nband))

        cen1, cen2 = self.cen_prior.sample(n)
        g1, g2 = self.g_prior.sample2d(n)
        T = self.T_prior.sample(n)
        logTratio = self.logTratio_prior.sample(n)
        fracdev = self.fracdev_prior.sample(n)

        samples[:, 0] = cen1
        samples[:, 1] = cen2
        samples[:, 2] = g1
        samples[:, 3] = g2
        samples[:, 4] = T
        samples[:, 5] = logTratio
        samples[:, 6] = fracdev

        for i in range(self.nband):
            F_prior = self.F_priors[i]
            F = F_prior.sample(n)
            samples[:, 7 + i] = F

        if is_scalar:
            samples = samples[0, :]
        return samples

    def __repr__(self):
        reps = []
        reps += [
            str(self.cen_prior),
            str(self.g_prior),
            str(self.T_prior),
            str(self.logTratio_prior),
            str(self.fracdev_prior),
        ]

        for p in self.F_priors:
            reps.append(str(p))

        rep = "\n".join(reps)
        return rep


class PriorBDFSep(PriorSimpleSep):
    """
    Separate priors on each parameter

    Parameters
    ----------
    cen_prior:
        The center prior
    g_prior:
        The prior on g (g1,g2).
    T_prior:
        Prior on T or some size parameter
    fracdev_prior:
        Prior on fracdev for bulge+disk
    F_prior:
        Prior on Flux.  Can be a list for a multi-band prior.
    """

    def __init__(self, cen_prior, g_prior, T_prior, fracdev_prior, F_prior):

        # print("JointPriorSimpleSep")

        self.cen_prior = cen_prior
        self.g_prior = g_prior
        self.T_prior = T_prior
        self.fracdev_prior = fracdev_prior

        if isinstance(F_prior, (list, tuple)):
            self.nband = len(F_prior)
        else:
            self.nband = 1
            F_prior = [F_prior]

        self.F_priors = F_prior

        self.set_bounds()

    def set_bounds(self):
        """
        set possibe bounds
        """
        bounds = [
            (None, None),  # c1
            (None, None),  # c2
            (None, None),  # g1
            (None, None),  # g2
            # (-1.0,1.0),  # g1
            # (-1.0,1.0),  # g2
        ]

        allp = [self.T_prior, self.fracdev_prior] + self.F_priors

        some_have_bounds = False
        for i, p in enumerate(allp):
            if p.has_bounds():
                # print('bounds for:',i,p)
                some_have_bounds = True
                bounds.append((p.bounds[0], p.bounds[1]))
            else:
                bounds.append((None, None))

        if not some_have_bounds:
            bounds = None

        self.bounds = bounds

    def get_lnprob_scalar(self, pars):
        """
        log probability for scalar input (meaning one point)
        """

        lnp = self.cen_prior.get_lnprob_scalar(pars[0], pars[1])
        lnp += self.g_prior.get_lnprob_scalar2d(pars[2], pars[3])
        lnp += self.T_prior.get_lnprob_scalar(pars[4])
        lnp += self.fracdev_prior.get_lnprob_scalar(pars[5])

        for i, F_prior in enumerate(self.F_priors):
            lnp += F_prior.get_lnprob_scalar(pars[6 + i])

        return lnp

    def fill_fdiff(self, pars, fdiff):
        """
        (model-data)/err
        but "data" here is the central value of a prior.
        """
        index = 0

        # fdiff[index] = self.cen_prior.get_lnprob_scalar(pars[0],pars[1])

        fdiff1, fdiff2 = self.cen_prior.get_fdiff(pars[0], pars[1])

        fdiff[index] = fdiff1
        index += 1
        fdiff[index] = fdiff2
        index += 1

        fdiff[index] = self.g_prior.get_fdiff(pars[2], pars[3])
        index += 1
        fdiff[index] = self.T_prior.get_fdiff(pars[4])
        index += 1

        fdiff[index] = self.fracdev_prior.get_fdiff(pars[5])
        index += 1

        for i in range(self.nband):
            F_prior = self.F_priors[i]
            fdiff[index] = F_prior.get_fdiff(pars[6 + i])
            index += 1

        return index

    def get_lnprob_array(self, pars):
        """
        log probability for array input [N,ndims]
        """

        lnp = self.cen_prior.get_lnprob_array(pars[:, 0], pars[:, 1])
        lnp += self.g_prior.get_lnprob_array2d(pars[:, 2], pars[:, 3])
        lnp += self.T_prior.get_lnprob_array(pars[:, 4])
        lnp += self.fracdev_prior.get_lnprob_array(pars[:, 5])

        for i in range(self.nband):
            F_prior = self.F_priors[i]
            lnp += F_prior.get_lnprob_array(pars[:, 6 + i])

        return lnp

    def sample(self, n=None):
        """
        Get random samples
        """

        if n is None:
            is_scalar = True
            n = 1
        else:
            is_scalar = False

        samples = zeros((n, 6 + self.nband))

        cen1, cen2 = self.cen_prior.sample(n)
        g1, g2 = self.g_prior.sample2d(n)
        T = self.T_prior.sample(n)
        fracdev = self.fracdev_prior.sample(n)

        samples[:, 0] = cen1
        samples[:, 1] = cen2
        samples[:, 2] = g1
        samples[:, 3] = g2
        samples[:, 4] = T
        samples[:, 5] = fracdev

        for i in range(self.nband):
            F_prior = self.F_priors[i]
            F = F_prior.sample(n)
            samples[:, 6 + i] = F

        if is_scalar:
            samples = samples[0, :]
        return samples

    def __repr__(self):
        reps = []
        reps += [
            str(self.cen_prior),
            str(self.g_prior),
            str(self.T_prior),
            str(self.fracdev_prior),
        ]

        for p in self.F_priors:
            reps.append(str(p))

        rep = "\n".join(reps)
        return rep


class PriorSpergelSep(PriorSimpleSep):
    """
    Separate priors on each parameter of a Spergel profile

    Parameters
    ----------
    cen_prior:
        The center prior
    g_prior:
        The prior on g (g1,g2).
    r50_prior:
        Prior on r50
    nu_prior:
        Prior on the index nu
    F_prior:
        Prior on Flux.  Can be a list for a multi-band prior.
    """

    def __init__(self, cen_prior, g_prior, r50_prior, nu_prior, F_prior):

        self.cen_prior = cen_prior
        self.g_prior = g_prior
        self.r50_prior = r50_prior
        self.nu_prior = nu_prior

        if isinstance(F_prior, list):
            self.nband = len(F_prior)
        else:
            self.nband = 1
            F_prior = [F_prior]

        self.F_priors = F_prior

        self.set_bounds()

    def get_lnprob_scalar(self, pars):
        """
        log probability for scalar input (meaning one point)
        """

        lnp = self.cen_prior.get_lnprob_scalar(pars[0], pars[1])
        lnp += self.g_prior.get_lnprob_scalar2d(pars[2], pars[3])
        lnp += self.r50_prior.get_lnprob_scalar(pars[4])
        lnp += self.nu_prior.get_lnprob_scalar(pars[5])

        for i, F_prior in enumerate(self.F_priors):
            lnp += F_prior.get_lnprob_scalar(pars[6 + i])

        return lnp

    def fill_fdiff(self, pars, fdiff):
        """
        set sqrt(-2ln(p)) ~ (model-data)/err
        """
        index = 0

        # fdiff[index] = self.cen_prior.get_lnprob_scalar(pars[0],pars[1])

        lnp1, lnp2 = self.cen_prior.get_lnprob_scalar_sep(pars[0], pars[1])

        fdiff[index] = lnp1
        index += 1
        fdiff[index] = lnp2
        index += 1

        fdiff[index] = self.g_prior.get_lnprob_scalar2d(pars[2], pars[3])
        index += 1
        fdiff[index] = self.r50_prior.get_lnprob_scalar(pars[4])
        index += 1

        fdiff[index] = self.nu_prior.get_lnprob_scalar(pars[5])
        index += 1

        for i in range(self.nband):
            F_prior = self.F_priors[i]
            fdiff[index] = F_prior.get_lnprob_scalar(pars[6 + i])
            index += 1

        chi2 = -2 * fdiff[0:index]
        chi2.clip(min=0.0, max=None, out=chi2)
        fdiff[0:index] = sqrt(chi2)

        return index

    def get_lnprob_array(self, pars):
        """
        log probability for array input [N,ndims]
        """

        lnp = self.cen_prior.get_lnprob_array(pars[:, 0], pars[:, 1])
        lnp += self.g_prior.get_lnprob_array2d(pars[:, 2], pars[:, 3])
        lnp += self.r50_prior.get_lnprob_array(pars[:, 4])
        lnp += self.nu_prior.get_lnprob_array(pars[:, 5])

        for i in range(self.nband):
            F_prior = self.F_priors[i]
            lnp += F_prior.get_lnprob_array(pars[:, 6 + i])

        return lnp

    def sample(self, n=None):
        """
        Get random samples
        """

        if n is None:
            is_scalar = True
            n = 1
        else:
            is_scalar = False

        samples = zeros((n, 6 + self.nband))

        cen1, cen2 = self.cen_prior.sample(n)
        g1, g2 = self.g_prior.sample2d(n)
        r50 = self.r50_prior.sample(n)
        nu = self.nu_prior.sample(n)

        samples[:, 0] = cen1
        samples[:, 1] = cen2
        samples[:, 2] = g1
        samples[:, 3] = g2
        samples[:, 4] = r50
        samples[:, 5] = nu

        for i in range(self.nband):
            F_prior = self.F_priors[i]
            F = F_prior.sample(n)
            samples[:, 6 + i] = F

        if is_scalar:
            samples = samples[0, :]
        return samples

    def __repr__(self):
        reps = []
        reps += [
            str(self.cen_prior),
            str(self.g_prior),
            str(self.r50_prior),
            str(self.nu_prior),
        ]

        for p in self.F_priors:
            reps.append(str(p))

        rep = "\n".join(reps)
        return rep


class PriorCoellipSame(PriorSimpleSep):
    def __init__(self, ngauss, cen_prior, g_prior, T_prior, F_prior):

        super(PriorCoellipSame, self).__init__(
            cen_prior, g_prior, T_prior, F_prior
        )

        self.ngauss = ngauss

    def set_bounds(self):
        """
        set possibe bounds
        """

        self.bounds = None

        return

        bounds = [
            (None, None),  # c1
            (None, None),  # c2
            (None, None),  # g1
            (None, None),  # g2
        ]

        some_have_bounds = False

        for p in [self.T_prior, self.F_prior]:
            if p.has_bounds():
                some_have_bounds = True
                pbounds = [(p.bounds[0], p.bounds[1])]
            else:
                pbounds = [(None, None)]

            bounds += [pbounds] * self.ngauss

        if not some_have_bounds:
            bounds = None

        self.bounds = bounds

    def get_lnprob_scalar(self, pars):
        """
        log probability for scalar input (meaning one point)
        """

        ngauss = self.ngauss

        lnp = self.cen_prior.get_lnprob_scalar(pars[0], pars[1])
        lnp += self.g_prior.get_lnprob_scalar2d(pars[2], pars[3])

        for i in range(ngauss):
            lnp += self.T_prior.get_lnprob_scalar(pars[4 + i])

        F_prior = self.F_priors[0]
        for i in range(ngauss):
            lnp += F_prior.get_lnprob_scalar(pars[4 + ngauss + i])

        return lnp

    def fill_fdiff(self, pars, fdiff):
        """
        set sqrt(-2ln(p)) ~ (model-data)/err
        """

        ngauss = self.ngauss

        index = 0

        # fdiff[index] = self.cen_prior.get_lnprob_scalar(pars[0],pars[1])

        lnp1, lnp2 = self.cen_prior.get_lnprob_scalar_sep(pars[0], pars[1])

        fdiff[index] = lnp1
        index += 1
        fdiff[index] = lnp2
        index += 1

        fdiff[index] = self.g_prior.get_lnprob_scalar2d(pars[2], pars[3])
        index += 1

        for i in range(ngauss):
            fdiff[index] = self.T_prior.get_lnprob_scalar(pars[4 + i])
            index += 1

        F_prior = self.F_priors[0]
        for i in range(ngauss):
            fdiff[index] = F_prior.get_lnprob_scalar(
                pars[4 + ngauss + i]
            )
            index += 1

        chi2 = -2 * fdiff[0:index]
        chi2.clip(min=0.0, max=None, out=chi2)
        fdiff[0:index] = sqrt(chi2)

        return index
