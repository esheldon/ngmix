

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
