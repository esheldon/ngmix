"""
n-dimensional gaussian mixture

this is a wrapper for the sklearn mixture, providing convenient fitting and
loading, as well as very fast likelihood evaluation

"""
from __future__ import print_function, division, absolute_import
try:
    xrange
except:
    xrange=range

import numpy
from . import _gmix

class GMixND(object):
    """
    Gaussian mixture in arbitrary dimensions.  A bit awkward
    in dim=1 e.g. becuase assumes means are [ndim,npars]
    """
    def __init__(self, weights=None, means=None, covars=None, file=None, rng=None):

        if rng is None:
            rng=numpy.random.RandomState()
        self.rng=rng

        if file is not None:
            self.load_mixture(file)
        else:
            if (weights is not None
                    and means is not None
                    and covars is not None):
                self.set_mixture(weights, means, covars)
            elif (weights is not None
                    or means is not None
                    or covars is not None):
                raise RuntimeError("send all or none of weights, means, covars")

    def set_mixture(self, weights, means, covars):
        """
        set the mixture elements
        """

        # copy all to avoid it getting changed under us and to
        # make sure native byte order

        weights = numpy.array(weights, dtype='f8', copy=True)
        means   = numpy.array(means, dtype='f8', copy=True)
        covars  = numpy.array(covars, dtype='f8', copy=True)

        if len(means.shape) == 1:
            means = means.reshape( (means.size, 1) )
        if len(covars.shape) == 1:
            covars = covars.reshape( (covars.size, 1, 1) )

        self.weights = weights
        self.means=means
        self.covars=covars



        self.ngauss = self.weights.size

        sh=means.shape
        if len(sh) == 1:
            raise ValueError("means must be 2-d even for ndim=1")

        self.ndim = sh[1]

        self._calc_icovars_and_norms()

        self.tmp_lnprob = numpy.zeros(self.ngauss)

    def fit(self, data, ngauss, n_iter=5000, min_covar=1.0e-6,
            doplot=False, **keys):
        """
        data is shape
            [npoints, ndim]
        """
        from sklearn.mixture import GaussianMixture

        if len(data.shape) == 1:
            data = data[:,numpy.newaxis]

        print("ngauss:   ",ngauss)
        print("n_iter:   ",n_iter)
        print("min_covar:",min_covar)

        gmm=GaussianMixture(
            n_components=ngauss,
            max_iter=n_iter,
            reg_covar=min_covar,
            covariance_type='full',
        )

        gmm.fit(data)

        if not gmm.converged_:
            print("DID NOT CONVERGE")

        self._gmm=gmm
        self.set_mixture(gmm.weights_, gmm.means_, gmm.covariances_)

        if doplot:
            plt=self.plot_components(data=data,**keys)
            return plt


    def save_mixture(self, fname):
        """
        save the mixture to a file
        """
        import fitsio

        print("writing gaussian mixture to :",fname)
        with fitsio.FITS(fname,'rw',clobber=True) as fits:
            fits.write(self.weights, extname='weights')
            fits.write(self.means, extname='means')
            fits.write(self.covars, extname='covars')
        
    def load_mixture(self, fname):
        """
        load the mixture from a file
        """
        import fitsio

        print("loading gaussian mixture from:",fname)
        with fitsio.FITS(fname) as fits:
            weights = fits['weights'].read()
            means = fits['means'].read()
            covars = fits['covars'].read()
        self.set_mixture(weights, means, covars)

    def get_lnprob_scalar(self, pars_in):
        """
        (x-xmean) icovar (x-xmean)
        """
        dolog=1
        #pars=numpy.asanyarray(pars_in, dtype='f8')
        pars=numpy.array(pars_in, dtype='f8', ndmin=1, order='C')
        lnp=_gmix.gmixnd_get_prob_scalar(self.log_pnorms,
                                         self.means,
                                         self.icovars,
                                         self.tmp_lnprob,
                                         pars,
                                         None,
                                         dolog)
        return lnp

    def get_prob_scalar(self, pars_in):
        """
        (x-xmean) icovar (x-xmean)
        """
        dolog=0
        pars=numpy.array(pars_in, dtype='f8', ndmin=1, order='C')
        p=_gmix.gmixnd_get_prob_scalar(self.log_pnorms,
                                       self.means,
                                       self.icovars,
                                       self.tmp_lnprob,
                                       pars,
                                       None,
                                       dolog)
        return p


    def get_lnprob_array(self, pars):
        """
        array input
        """

        if len(pars.shape) == 1:
            pars = pars[:,numpy.newaxis]

        n=pars.shape[0]
        lnp=numpy.zeros(n)

        for i in xrange(n):
            lnp[i] = self.get_lnprob_scalar(pars[i,:])

        return lnp

    def get_prob_array(self, pars):
        """
        array input
        """

        if len(pars.shape) == 1:
            pars = pars[:,numpy.newaxis]

        n=pars.shape[0]
        p=numpy.zeros(n)

        for i in xrange(n):
            p[i] = self.get_prob_scalar(pars[i,:])

        return p

    def get_prob_scalar_sub(self, pars_in, use=None):
        """
        Only include certain components
        """

        if use is not None:
            use=numpy.array(use,dtype='i4',copy=False)
            assert use.size==self.ngauss

        dolog=0
        pars=numpy.array(pars_in, dtype='f8', ndmin=1, order='C')
        p=_gmix.gmixnd_get_prob_scalar(self.log_pnorms,
                                       self.means,
                                       self.icovars,
                                       self.tmp_lnprob,
                                       pars,
                                       use,
                                       dolog)
        return p

    def get_prob_array_sub(self, pars, use=None):
        """
        array input
        """

        if len(pars.shape) == 1:
            pars = pars[:,numpy.newaxis]

        n=pars.shape[0]
        p=numpy.zeros(n)

        for i in xrange(n):
            p[i] = self.get_prob_scalar_sub(pars[i,:], use=use)

        return p


    def sample(self, n=None):
        """
        sample from the gaussian mixture
        """
        if not hasattr(self, '_gmm'):
            self._set_gmm()

        if n is None:
            is_one=True
            n=1
        else:
            is_one=False

        samples,labels = self._gmm.sample(n)

        if self.ndim==1:
            samples = samples[:,0]

        if is_one:
            samples = samples[0]

        return samples

    def _make_gmm(self, ngauss):
        """
        Make a GMM object for sampling
        """
        from sklearn.mixture import GaussianMixture

        gmm=GaussianMixture(
            n_components=ngauss,
            max_iter=10000,
            reg_covar=1.0e-12,
            covariance_type='full',
            random_state=self.rng,
        )

        return gmm


    def _set_gmm(self):
        """
        Make a GMM object for sampling
        """
        import sklearn.mixture

        # these numbers are not used because we set the means, etc by hand
        ngauss=self.weights.size

        gmm = self._make_gmm(ngauss)
        gmm.means_ = self.means.copy()
        #gmm.covars_ = self.covars.copy()
        gmm.covariances_ = self.covars.copy()
        gmm.weights_ = self.weights.copy()

        gmm.precisions_cholesky_ = sklearn.mixture.gaussian_mixture._compute_precision_cholesky(
            self.covars, 'full',
        )

        self._gmm=gmm 

    def _calc_icovars_and_norms(self):
        """
        Calculate the normalizations and inverse covariance matrices
        """
        from numpy import pi

        twopi = 2.0*pi

        #if self.ndim==1:
        if False:
            norms = 1.0/numpy.sqrt(twopi*self.covars)
            icovars = 1.0/self.covars
        else:
            norms = numpy.zeros(self.ngauss)
            icovars = numpy.zeros( (self.ngauss, self.ndim, self.ndim) )
            for i in xrange(self.ngauss):
                cov = self.covars[i,:,:]
                icov = numpy.linalg.inv( cov )

                det = numpy.linalg.det(cov)
                n=1.0/numpy.sqrt( twopi**self.ndim * det )

                norms[i] = n
                icovars[i,:,:] = icov

        self.norms = norms
        self.pnorms = norms*self.weights
        self.log_pnorms = numpy.log(self.pnorms)
        self.icovars = icovars

    def plot_components(self, data=None, **keys):
        """
        """
        import biggles 
        import pcolors

        # first for 1d,then generalize
        assert self.ndim==1

        # draw a random sample to get a feel for the range

        xmin=keys.pop('min',None)
        xmax=keys.pop('max',None)
        if data is not None:
            if xmin is None:
                xmin=data.min()
            if xmax is None:
                xmax=data.max()
        else:
            r = self.sample(100000)
            if xmin is None:
                xmin=r.min()
            if xmax is None:
                xmax=r.max()

        x = numpy.linspace(xmin, xmax, num=10000)

        ytot = self.get_prob_array(x)
        ymax=ytot.max()
        ymin=1.0e-6*ymax
        ytot=ytot.clip(min=ymin)

        xrng=keys.pop('xrange',None) 
        yrng=keys.pop('yrange',None) 
        if xrng is None:
            xrng=[xmin,xmax]
        if yrng is None:
            yrng=[ymin,1.1*ymax]



        if data is not None:
            binsize=keys.pop('binsize',None)
            if binsize is None:
                binsize=0.1*data.std()

            histc = biggles.make_histc(
                data,
                min=xmin,
                max=xmax,
                yrange=yrng,
                binsize=binsize,
                color='orange',
                width=3,
                norm=1,
            )
            loghistc = biggles.make_histc(
                data,
                min=xmin,
                max=xmax,
                ylog=True,
                yrange=yrng,
                binsize=binsize,
                color='orange',
                width=3,
                norm=1,
            )
        else:
            histc=None
            loghistc=None




        plt=biggles.FramedPlot(
            xlabel='x',
            ylabel='P(x)',
            aspect_ratio=1.0/1.618,
            xrange=xrng,
            yrange=yrng,
        )
        logplt=biggles.FramedPlot(
            xlabel='x',
            ylabel='P(x)',
            ylog=True,
            xrange=xrng,
            yrange=yrng,
            aspect_ratio=1.0/1.618,
        )

        curves = []
        logcurves = []
        if histc is not None:        
            curves.append(histc)
            logcurves.append(loghistc)

        ctot = biggles.Curve(x, ytot, type='solid', color='black')
        curves.append(ctot)
        logcurves.append(ctot)



        colors = pcolors.rainbow(self.ngauss)
        use = numpy.zeros(self.ngauss, dtype='i4')
        for i in xrange(self.ngauss):

            use[i] = 1

            y = self.get_prob_array_sub(x, use=use)
            y = y.clip(min=ymin)


            c = biggles.Curve(x, y, type='solid', color=colors[i])
            curves.append(c)
            logcurves.append(c)

            use[:]=0

        ymax=ytot.max()

        plt.add(*curves)
        logplt.add(*logcurves)

        tab = biggles.Table(2,1)
        tab[0,0] = plt
        tab[1,0] = logplt
        show=keys.pop('show',False)
        if show:
            width=keys.pop('width',1000)
            height=keys.pop('height',1000)
            tab.show(width=width, height=height, **keys)
        return tab

    def plot_components_old(self, **keys):
        """
        """
        import biggles 
        import pcolors

        # first for 1d,then generalize
        assert self.ndim==1

        # draw a random sample to get a feel for the range
        r = self.sample(100000)

        xmin,xmax=r.min(),r.max()

        x = numpy.linspace(xmin, xmax, num=1000)

        ytot = self.get_prob_array(x)
        ymax=ytot.max()
        ymin=1.0e-6*ymax
        ytot=ytot.clip(min=ymin)

        plt=biggles.FramedPlot(
            xlabel='x',
            ylabel='P(x)',
            aspect_ratio=1.0/1.618,
        )
        logplt=biggles.FramedPlot(
            xlabel='x',
            ylabel='P(x)',
            ylog=True,
            aspect_ratio=1.0/1.618,
        )


        ctot = biggles.Curve(x, ytot, type='solid', color='black')

        curves = [ctot]

        ytot2 = ytot*0

        colors = pcolors.rainbow(self.ngauss)
        use = numpy.zeros(self.ngauss, dtype='i4')
        for i in xrange(self.ngauss):

            use[i] = 1

            y = self.get_prob_array_sub(x, use=use)
            y = y.clip(min=ymin)

            ytot2 += y

            c = biggles.Curve(x, y, type='solid', color=colors[i])
            curves.append(c)

            use[:]=0

        ymax=ytot.max()

        ctot2 = biggles.Curve(x, ytot2, type='dashed', color='red')
        curves.append(ctot2)
        plt.add(*curves)
        logplt.add(*curves)

        tab = biggles.Table(2,1)
        tab[0,0] = plt
        tab[1,0] = logplt
        show=keys.pop('show',False)
        if show:
            tab.show(**keys)
        return tab

