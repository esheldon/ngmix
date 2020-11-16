JOINT_N_ITER = 1000
JOINT_MIN_COVAR = 1.0e-6
JOINT_COVARIANCE_TYPE = "full"


class TFluxPriorCosmosBase(PriorBase):
    """
    T is in arcsec**2
    """

    def __init__(self, T_bounds=[1.0e-6, 100.0], flux_bounds=[1.0e-6, 100.0]):

        print("warning: cannot use local rng")

        self.set_bounds(T_bounds=T_bounds, flux_bounds=flux_bounds)

        self.make_gmm()
        self.make_gmix()

    def set_bounds(self, T_bounds=None, flux_bounds=None):

        if len(T_bounds) != 2:
            raise ValueError("T_bounds must be len==2")
        if len(flux_bounds) != 2:
            raise ValueError("flux_bounds must be len==2")

        self.T_bounds = array(T_bounds, dtype="f8")
        self.flux_bounds = array(flux_bounds, dtype="f8")

    def get_flux_mode(self):
        return self.flux_mode

    def get_T_near(self):
        return self.T_near

    def get_lnprob_slow(self, T_and_flux_input):
        """
        ln prob for linear variables
        """

        nd = len(T_and_flux_input.shape)

        if nd == 1:
            T_and_flux = T_and_flux_input.reshape(1, 2)
        else:
            T_and_flux = T_and_flux_input

        T_bounds = self.T_bounds
        T = T_and_flux[0, 0]
        if T < T_bounds[0] or T > T_bounds[1]:
            return LOWVAL

        flux_bounds = self.flux_bounds
        flux = T_and_flux[0, 1]
        if flux < flux_bounds[0] or flux > flux_bounds[1]:
            return LOWVAL

        logvals = numpy.log10(T_and_flux)

        return self.gmm.score(logvals)

    def get_lnprob_many(self, T_and_flux):
        """
        The fast array one using a GMix
        """

        n = T_and_flux.shape[0]
        lnp = numpy.zeros(n)

        for i in range(n):
            lnp[i] = self.get_lnprob_one(T_and_flux[i, :])

        return lnp

    def get_lnprob_one(self, T_and_flux):
        """
        The fast scalar one using a GMix
        """

        # bounds cuts happen in pixel space (if scale!=1)
        T_bounds = self.T_bounds
        flux_bounds = self.flux_bounds

        T = T_and_flux[0]
        if T < T_bounds[0] or T > T_bounds[1]:
            return LOWVAL

        flux = T_and_flux[1]
        if flux < flux_bounds[0] or flux > flux_bounds[1]:
            return LOWVAL

        logpars = numpy.log10(T_and_flux)

        return self.gmix.get_loglike_rowcol(logpars[0], logpars[1])

    def sample(self, n):
        """
        Sample the linear variables
        """

        T_bounds = self.T_bounds
        flux_bounds = self.flux_bounds

        lin_samples = numpy.zeros((n, 2))
        nleft = n
        ngood = 0
        while nleft > 0:
            tsamples = self.gmm.sample(nleft)
            tlin_samples = 10.0 ** tsamples

            Tvals = tlin_samples[:, 0]
            flux_vals = tlin_samples[:, 1]
            (w,) = numpy.where(
                (Tvals > T_bounds[0])
                & (Tvals < T_bounds[1])
                & (flux_vals > flux_bounds[0])
                & (flux_vals < flux_bounds[1])
            )

            if w.size > 0:
                lin_samples[ngood:ngood + w.size, :] = tlin_samples[w, :]
                ngood += w.size
                nleft -= w.size

        return lin_samples

    def make_gmm(self):
        """
        Make a GMM object from the inputs
        """
        from sklearn.mixture import GMM

        # we will over-ride values, pars here shouldn't matter except
        # for consistency

        ngauss = self.weights.size
        gmm = GMM(
            n_components=ngauss,
            n_iter=JOINT_N_ITER,
            min_covar=JOINT_MIN_COVAR,
            covariance_type=JOINT_COVARIANCE_TYPE,
        )
        gmm.means_ = self.means
        gmm.covars_ = self.covars
        gmm.weights_ = self.weights

        self.gmm = gmm

    def make_gmix(self):
        """
        Make a GMix object for speed
        """
        from . import gmix

        ngauss = self.weights.size
        pars = numpy.zeros(6 * ngauss)
        for i in range(ngauss):
            index = i * 6
            pars[index + 0] = self.weights[i]
            pars[index + 1] = self.means[i, 0]
            pars[index + 2] = self.means[i, 1]

            pars[index + 3] = self.covars[i, 0, 0]
            pars[index + 4] = self.covars[i, 0, 1]
            pars[index + 5] = self.covars[i, 1, 1]

        self.gmix = gmix.GMix(ngauss=ngauss, pars=pars)


class TFluxPriorCosmosExp(TFluxPriorCosmosBase):
    """
    joint T-flux distribution based on fits to cosmos data
    using an Exp model
    """

    flux_mode = 0.121873372203

    # T_near is mean T near the flux mode
    T_near = 0.182461907543  # arcsec**2

    weights = array(
        [
            0.24725964,
            0.12439838,
            0.10301993,
            0.07903986,
            0.16064439,
            0.01162365,
            0.15587558,
            0.11813856,
        ]
    )
    means = array(
        [
            [-0.63771027, -0.55703495],
            [-1.09910453, -0.79649937],
            [-0.89605713, -0.9432781],
            [-1.01262357, -0.05649636],
            [-0.26622288, -0.16742824],
            [-0.50259072, -0.10134068],
            [-0.11414387, -0.49449105],
            [-0.39625801, -0.83781264],
        ]
    )

    covars = array(
        [
            [[0.17219003, -0.00650028], [-0.00650028, 0.05053097]],
            [[0.09800749, 0.00211059], [0.00211059, 0.01099591]],
            [[0.16110231, 0.00194853], [0.00194853, 0.00166609]],
            [[0.07933064, 0.08535596], [0.08535596, 0.20289928]],
            [[0.17805248, 0.1356711], [0.1356711, 0.24728999]],
            [[1.07112875, -0.08777646], [-0.08777646, 0.26131025]],
            [[0.06570913, 0.04912536], [0.04912536, 0.094739]],
            [[0.0512232, 0.00519115], [0.00519115, 0.00999365]],
        ]
    )


class TFluxPriorCosmosDev(TFluxPriorCosmosBase):
    """
    joint T-flux distribution based on fits to cosmos data
    using an Dev model

    pars from
    ~/lensing/galsim-cosmos-data/gmix-fits/001/dist/gmix-cosmos-001-dev-joint-dist.fits
    """

    flux_mode = 0.241579121406777
    # T_near is mean T near the flux mode
    T_near = 2.24560320009

    weights = array(
        [
            0.13271808,
            0.10622821,
            0.09982766,
            0.29088236,
            0.1031488,
            0.10655095,
            0.01631938,
            0.14432457,
        ]
    )
    means = array(
        [
            [0.22180692, 0.05099562],
            [0.92589409, -0.61785194],
            [-0.10599071, -0.82198516],
            [1.06182382, -0.29997484],
            [1.57670969, 0.25244698],
            [-0.1807241, -0.35644187],
            [0.73288177, 0.07028779],
            [-0.14674281, -0.65843109],
        ]
    )
    covars = array(
        [
            [[0.35760805, 0.17652397], [0.17652397, 0.28479473]],
            [[0.14969479, 0.03008289], [0.03008289, 0.01369154]],
            [[0.37153864, 0.03293217], [0.03293217, 0.00476308]],
            [[0.27138008, 0.08628813], [0.08628813, 0.0883718]],
            [[0.20242911, 0.12374718], [0.12374718, 0.22626528]],
            [[0.30836137, 0.07141098], [0.07141098, 0.05870399]],
            [[2.13662397, 0.04301596], [0.04301596, 0.17006638]],
            [[0.40151762, 0.05215542], [0.05215542, 0.01733399]],
        ]
    )
