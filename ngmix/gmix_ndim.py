"""
n-dimensional gaussian mixture

this is a wrapper for the sklearn mixture, providing convenient fitting and
loading, as well as very fast likelihood evaluation

"""
import numpy
from .gmix_ndim_nb import gmixnd_get_prob, gmixnd_get_prob_component


class GMixND(object):
    """
    Gaussian mixture in arbitrary dimensions.  A bit awkward
    in dim=1 e.g. becuase assumes means are [ndim,npars]
    """

    def __init__(
        self, weights=None, means=None, covars=None, file=None, rng=None
    ):

        if rng is None:
            rng = numpy.random.RandomState()
        self.rng = rng

        if file is not None:
            self.load_mixture(file)
        else:
            if (
                weights is not None
                and means is not None
                and covars is not None
            ):
                self.set_mixture(weights, means, covars)
            elif (
                weights is not None or means is not None or covars is not None
            ):
                raise RuntimeError(
                    "send all or none of weights, means, covars"
                )

    def set_mixture(self, weights, means, covars):
        """
        set the mixture elements
        """

        # copy all to avoid it getting changed under us and to
        # make sure native byte order

        weights = numpy.array(weights, dtype="f8", copy=True)
        means = numpy.array(means, dtype="f8", copy=True)
        covars = numpy.array(covars, dtype="f8", copy=True)

        if len(means.shape) == 1:
            means = means.reshape((means.size, 1))
        if len(covars.shape) == 1:
            covars = covars.reshape((covars.size, 1, 1))

        self.weights = weights
        self.means = means
        self.covars = covars

        self.ngauss = self.weights.size

        sh = means.shape
        if len(sh) == 1:
            raise ValueError("means must be 2-d even for ndim=1")

        self.ndim = sh[1]

        self._calc_icovars_and_norms()

        self.tmp_lnprob = numpy.zeros(self.ngauss)
        self.xdiff = numpy.zeros(self.ndim)

    def fit(
        self, data, ngauss, n_iter=5000, min_covar=1.0e-6, doplot=False, **keys
    ):
        """
        data is shape
            [npoints, ndim]
        """
        from sklearn.mixture import GaussianMixture

        if len(data.shape) == 1:
            data = data[:, numpy.newaxis]

        print("ngauss:   ", ngauss)
        print("n_iter:   ", n_iter)
        print("min_covar:", min_covar)

        gmm = GaussianMixture(
            n_components=ngauss,
            max_iter=n_iter,
            reg_covar=min_covar,
            covariance_type="full",
            random_state=self.rng,
        )

        gmm.fit(data)

        if not gmm.converged_:
            print("DID NOT CONVERGE")

        self._gmm = gmm
        self.set_mixture(gmm.weights_, gmm.means_, gmm.covariances_)

        if doplot:
            plt = self.plot(data=data, **keys)
            return plt

    def plot(
        self,
        min=None,
        max=None,
        npts=None,
        data=None,
        nbin=None,
        binsize=None,
        file=None,
        dpi=100,
        show=False,
        **plot_kws
    ):  # pragma: no cover
        """
        plot the model and each component.  Optionally plot a set of
        data as well.  Currently only works for 1d

        Parameters
        ----------
        min: float
            Min value to plot, if data is sent then this can be left
            out and the min value will be gotten from that data.
        max: float
            Max value to plot, if data is sent then this can be left
            out and the max value will be gotten from that data.
        npts: int, optional
            Number of points to use for the plot.  If data are sent you
            can leave this off and a suitable value will be chosen based
            on the data binsize
        data: array, optional
            Optional data to plot as a histogram
        nbin: int, optional
            Optional number of bins for histogramming data
        binsize: float, optional
            Optional binsize for histogramming data
        file: str, optional
            Optionally write out a plot file
        dpi: int, optional
            Optional dpi for graphics like png, default 100
        show: bool, optional
            If True, show the plot on the screen

        Returns
        -------
        plot object
        """
        import esutil as eu
        import hickory

        plt = hickory.Plot(**plot_kws)

        if data is not None:

            if min is None:
                min = data.min()
            if max is None:
                max = data.max()

            hd = eu.stat.histogram(
                data, min=min, max=max, nbin=nbin, binsize=binsize, more=True,
            )
            dsum = hd['hist'].sum()
            xvals = hd['center']
            dx_data = xvals[1] - xvals[0]

            if npts is None:
                dx_model = dx_data/10
                npts = int((max - min)/dx_model)

            xvals = numpy.linspace(
                min,
                max,
                npts,
            )
            dx_model = xvals[1] - xvals[0]

            plt.bar(hd['center'], hd['hist'], label='data', width=dx_data,
                    alpha=0.5, color='#a6a6a6')

        else:
            if npts is None:
                raise ValueError('send npts if not sending data')
            if min is None:
                raise ValueError('send min if not sending data')
            if max is None:
                raise ValueError('send max if not sending data')

            xvals = numpy.linspace(min, max, npts)

        predicted = self.get_prob_array(xvals)

        if data is not None:
            psum = predicted.sum()
            fac = dsum/psum * dx_data/dx_model
            predicted *= fac
        else:
            fac = 1

        plt.curve(xvals, predicted, label='model')
        for i in range(self.ngauss):
            predicted = fac*self.get_prob_array(xvals, component=i)

            label = 'component %d' % i
            plt.curve(xvals, predicted, label=label)

        if show:
            plt.show()

        if file is not None:
            print('writing:', file)
            plt.savefig(file, dpi=dpi)

        return plt

    def save_mixture(self, fname):
        """
        save the mixture to a file
        """
        import fitsio

        print("writing gaussian mixture to :", fname)
        with fitsio.FITS(fname, "rw", clobber=True) as fits:
            fits.write(self.weights, extname="weights")
            fits.write(self.means, extname="means")
            fits.write(self.covars, extname="covars")

    def load_mixture(self, fname):
        """
        load the mixture from a file
        """
        import fitsio

        print("loading gaussian mixture from:", fname)
        with fitsio.FITS(fname) as fits:
            weights = fits["weights"].read()
            means = fits["means"].read()
            covars = fits["covars"].read()
        self.set_mixture(weights, means, covars)

    def _get_prob(self, pars, dolog, component=None):
        """
        version with no checking
        """

        if component is None:
            return gmixnd_get_prob(
                self.log_pnorms,
                self.means,
                self.icovars,
                pars,
                self.xdiff,
                self.tmp_lnprob,
                dolog,
            )
        else:
            return gmixnd_get_prob_component(
                self.log_pnorms,
                self.means,
                self.icovars,
                pars,
                self.xdiff,
                dolog,
                component,
            )

    def _get_prob_array(self, pars, dolog, component=None):
        """
        version with no checking
        """

        n = pars.shape[0]
        retvals = numpy.zeros(n)

        for i in range(n):
            retvals[i] = self._get_prob(pars[i, :], dolog, component=component)

        return retvals

    def get_lnprob_scalar(self, pars_in, component=None):
        """
        (x-xmean) icovar (x-xmean)
        """
        dolog = 1
        pars = numpy.array(pars_in, dtype="f8", ndmin=1, order="C")
        return self._get_prob(pars, dolog, component=component)

    def get_prob_scalar(self, pars_in, component=None):
        """
        (x-xmean) icovar (x-xmean)
        """
        dolog = 0
        pars = numpy.array(pars_in, dtype="f8", ndmin=1, order="C")
        return self._get_prob(pars, dolog, component=component)

    def get_lnprob_array(self, pars, component=None):
        """
        array input
        """

        dolog = 1
        pars = numpy.array(pars, dtype="f8", ndmin=1, order="C")
        if len(pars.shape) == 1:
            pars = pars[:, numpy.newaxis]

        return self._get_prob_array(pars, dolog, component=component)

    def get_prob_array(self, pars, component=None):
        """
        array input
        """

        dolog = 0
        pars = numpy.array(pars, dtype="f8", ndmin=1, order="C")
        if len(pars.shape) == 1:
            pars = pars[:, numpy.newaxis]

        return self._get_prob_array(pars, dolog, component=component)

    def sample(self, n=None):
        """
        sample from the gaussian mixture
        """
        if not hasattr(self, "_gmm"):
            self._set_gmm()

        if n is None:
            is_one = True
            n = 1
        else:
            is_one = False

        samples, labels = self._gmm.sample(n)

        if self.ndim == 1:
            samples = samples[:, 0]

        if is_one:
            samples = samples[0]

        return samples

    def _make_gmm(self, ngauss):
        """
        Make a GMM object for sampling
        """
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(
            n_components=ngauss,
            max_iter=10000,
            reg_covar=1.0e-12,
            covariance_type="full",
            random_state=self.rng,
        )

        return gmm

    def _set_gmm(self):
        """
        Make a GMM object for sampling
        """
        from sklearn.mixture._gaussian_mixture import (
            _compute_precision_cholesky
        )

        # these numbers are not used because we set the means, etc by hand
        ngauss = self.weights.size

        gmm = self._make_gmm(ngauss)
        gmm.means_ = self.means.copy()
        gmm.covariances_ = self.covars.copy()
        gmm.weights_ = self.weights.copy()

        gmm.precisions_cholesky_ = _compute_precision_cholesky(
            self.covars, "full",
        )

        self._gmm = gmm

    def _calc_icovars_and_norms(self):
        """
        Calculate the normalizations and inverse covariance matrices
        """
        from numpy import pi

        twopi = 2.0 * pi

        # if self.ndim==1:
        if False:
            norms = 1.0 / numpy.sqrt(twopi * self.covars)
            icovars = 1.0 / self.covars
        else:
            norms = numpy.zeros(self.ngauss)
            icovars = numpy.zeros((self.ngauss, self.ndim, self.ndim))
            for i in range(self.ngauss):
                cov = self.covars[i, :, :]
                icov = numpy.linalg.inv(cov)

                det = numpy.linalg.det(cov)
                n = 1.0 / numpy.sqrt(twopi ** self.ndim * det)

                norms[i] = n
                icovars[i, :, :] = icov

        self.norms = norms
        self.pnorms = norms * self.weights
        self.log_pnorms = numpy.log(self.pnorms)
        self.icovars = icovars
