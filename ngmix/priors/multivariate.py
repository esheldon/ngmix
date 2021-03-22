"""
Convention is that all priors should have peak ln(prob)==0. This
helps use in priors for LM fitting
"""
from .priors import PriorBase


class CenPrior(PriorBase):
    """
    Prior for independent gaussians in each dimension.

    parameters
    ----------
    cen1: float
        The mean in the first dimension.
    cen2: float
        The mean in the second dimension.
    sigma1: float
        The width in the first dimension.
    sigma2: float
        The width in the second dimension.
    rng: np.random.RandomState
        An random number generator (RNG) to use.

    attributes
    ----------
    cen1: float
        The mean in the first dimension.
    cen2: float
        The mean in the second dimension.
    sigma1: float
        The width in the first dimension.
    sigma2: float
        The width in the second dimension.
    """
    def __init__(self, cen1, cen2, sigma1, sigma2, rng):
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
        Compute sqrt(-2ln(p)) ~ (data - mode)/err for using with LM fitters.
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
        probability at the specified point
        """
        from math import exp

        d1 = self.cen1 - x1
        d2 = self.cen2 - x2
        lnp = -0.5 * d1 * d1 * self.s2inv1 - 0.5 * d2 * d2 * self.s2inv2
        return exp(lnp)

    get_prob_array = get_prob_scalar
    get_lnprob_array = get_lnprob_scalar

    def sample(self, nrand=None):
        """
        Get a single sample or arrays.

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

        rand1 = rng.normal(loc=self.cen1, scale=self.sigma1, size=nrand)
        rand2 = rng.normal(loc=self.cen2, scale=self.sigma2, size=nrand)

        return rand1, rand2

    sample2d = sample


SimpleGauss2D = CenPrior
