from __future__ import print_function

try:
    xrange = xrange
    # We have Python 2
except:
    xrange = range
    # We have Python 3

import copy
import numpy
from numpy import array, zeros, exp, log10, log, dot, sqrt, diag
from . import fastmath
from .jacobian import Jacobian
from .shape import Shape, g1g2_to_e1e2, e1e2_to_g1g2

from . import moments

from .gexceptions import GMixRangeError, GMixFatalError

from . import _gmix

def make_gmix_model(pars, model):
    """
    get a gaussian mixture model for the given model
    """
    model_num=get_model_num(model)
    if model==GMIX_COELLIP:
        return GMixCoellip(pars)
    elif model==GMIX_SERSIC:
        return GMixSersic(pars)
    elif model==GMIX_FULL:
        return GMix(pars=pars)
    else:
        return GMixModel(pars, model)

class GMix(object):
    """
    A general two-dimensional gaussian mixture.

    parameters
    ----------
    Send either ngauss= or pars=

    ngauss: number, optional
        number of gaussians.  data will be zeroed
    pars: array-like, optional
        6*ngauss elements to fill the gaussian mixture.

    methods
    -------
    copy(self):
        make a new copy of this GMix
    convolve(psf):
        Get a new GMix that is the convolution of the GMix with the input psf
    get_T():
        get T=sum(p*T_i)/sum(p)
    get_sigma():
        get sigma=sqrt(T/2)
    get_psum():
        get sum(p)
    set_psum(psum):
        set new overall sum(p)
    get_cen():
        get cen=sum(p*cen_i)/sum(p)
    set_cen(row,col):
        set the overall center to the input.
    """
    def __init__(self, ngauss=None, pars=None):

        self._model      = GMIX_FULL
        self._model_name = 'full'

        if ngauss is None and pars is None:
            raise GMixFatalError("send ngauss= or pars=")

        if pars is not None:
            npars = len(pars)
            if (npars % 6) != 0:
                raise GMixFatalError("len(pars) must be mutiple of 6 "
                                     "got %s" % npars)
            self._ngauss=npars/6
            self.reset()
            self.fill(pars)
        else:
            self._ngauss=ngauss
            self.reset()
        
        self._set_f8_type()

    def _set_f8_type(self):
        tmp=numpy.zeros(1)
        self._f8_type=tmp.dtype.descr[0][1]

    def get_data(self):
        """
        Get the underlying array
        """
        return self._data

    def get_full_pars(self):
        """
        Get a full parameter description.
           [p1,row1,col1,irr1,irc1,icc1,
            p2,row2,col2,irr2,irc2,icc2,
            ...
           ]

        """

        gm=self._get_gmix_data()

        n=self._ngauss
        pars=numpy.zeros(n*6)
        beg=0
        for i in xrange(n):
            pars[beg+0] = gm['p'][i]
            pars[beg+1] = gm['row'][i]
            pars[beg+2] = gm['col'][i]
            pars[beg+3] = gm['irr'][i]
            pars[beg+4] = gm['irc'][i]
            pars[beg+5] = gm['icc'][i]
            
            beg += 6
        return pars

    def get_cen(self):
        """
        get the center position (row,col)
        """

        gm=self._get_gmix_data()
        psum=gm['p'].sum()
        rowsum=(gm['row']*gm['p']).sum()
        colsum=(gm['col']*gm['p']).sum()

        row=rowsum/psum
        col=colsum/psum

        return row,col
    
    def set_cen(self, row, col):
        """
        Move the mixture to a new center
        """
        gm=self._get_gmix_data()

        row0,col0 = self.get_cen()
        row_shift = row - row0
        col_shift = col - col0

        gm['row'] += row_shift
        gm['col'] += col_shift

    def get_T(self):
        """
        get weighted average T sum(p*T)/sum(p)
        """

        gm=self._get_gmix_data()

        row,col=self.get_cen()

        rowdiff=gm['row']-row
        coldiff=gm['col']-col

        p=gm['p']
        ipsum=1.0/p.sum()

        irr= ((gm['irr'] + rowdiff**2)      * p).sum()*ipsum
        icc= ((gm['icc'] + coldiff**2)      * p).sum()*ipsum

        T = irr + icc

        return T


    def get_sigma(self):
        """
        get sigma=sqrt(T/2)
        """
        T=self.get_T()
        return sqrt(T/2.)

    def get_e1e2T(self):
        """
        Get e1,e2 and T for the total gaussian mixture.
        """

        gm=self._get_gmix_data()

        row,col=self.get_cen()

        rowdiff=gm['row']-row
        coldiff=gm['col']-col

        p=gm['p']
        ipsum=1.0/p.sum()

        irr= ((gm['irr'] + rowdiff**2)      * p).sum()*ipsum
        irc= ((gm['irc'] + rowdiff*coldiff) * p).sum()*ipsum
        icc= ((gm['icc'] + coldiff**2)      * p).sum()*ipsum

        T = irr + icc

        e1=(icc-irr)/T
        e2=2.0*irc/T
        return e1,e2,T

    def get_g1g2T(self):
        """
        Get g1,g2 and T for the total gaussian mixture.
        """
        e1,e2,T=self.get_e1e2T()
        g1,g2=e1e2_to_g1g2(e1,e2)
        return g1,g2,T

    def get_e1e2sigma(self):
        """
        Get e1,e2 and sigma for the total gmix.

        Warning: only really works if the centers are the same
        """

        e1,e2,T=self.get_e1e2T()
        sigma=sqrt(T/2)
        return e1,e2,sigma

    def get_g1g2sigma(self):
        """
        Get g1,g2 and sigma for the total gmix.

        Warning: only really works if the centers are the same
        """
        e1,e2,T=self.get_e1e2T()
        g1,g2=e1e2_to_g1g2(e1,e2)

        sigma=sqrt(T/2)
        return g1,g2,sigma

    def get_flux(self):
        """
        get sum(p)
        """
        gm=self._get_gmix_data()
        return gm['p'].sum()
    # alias
    get_psum=get_flux

    def set_flux(self, psum):
        """
        set a new value for sum(p)
        """
        gm=self._get_gmix_data()

        psum0 = gm['p'].sum()
        rat = psum/psum0
        gm['p'] *= rat

        # we will need to reset the pnorm values
        gm['norm_set']=0

    # alias
    set_psum=set_flux


    def set_norms(self):
        """
        Needed to actually evaluate the gaussian.  This is done internally
        by the c code so if all goes well you don't need to call this
        """
        gm=self._get_gmix_data()
        _gmix.set_norms(gm)

    def fill(self, pars):
        """
        fill the gaussian mixture from a 'full' parameter array.

        The length must match the internal size

        parameters
        ----------
        pars: array-like
            [p1,row1,col1,irr1,irc1,icc1,
             p2,row2,col2,irr2,irc2,icc2,
             ...]

             Should have length 6*ngauss
        """

        gm=self._get_gmix_data()
        pars=array(pars, dtype='f8', copy=False) 
        _gmix.gmix_fill(gm, pars, self._model)

    def copy(self):
        """
        Get a new GMix with the same parameters
        """
        gmix = GMix(self._ngauss)
        gmix._data[:] = self._data[:]
        return gmix

    def get_sheared(self, s1, s2=None):
        """
        Get a sheared version of the gaussian mixture

        call as either 
            gmnew = gm.get_sheared(shape)
        or
            gmnew = gm.get_sheared(g1,g2)
        """
        if isinstance(s1, Shape):
            shear1=s1.g1
            shear2=s1.g2
        elif s2 is not None:
            shear1=s1
            shear2=s2
        else:
            raise RuntimeError("send a Shape or s1,s2")

        new_gmix = self.copy()


        ndata = new_gmix._get_gmix_data()
        ndata['norm_set']=0

        for i in xrange(len(self)):
            irr=ndata['irr'][i]
            irc=ndata['irc'][i]
            icc=ndata['icc'][i]

            irr_s,irc_s,icc_s=moments.get_sheared_moments(
                irr,irc,icc,
                shear1,shear2
            )

            det = irr_s*icc_s - irc_s*irc_s
            ndata['irr'][i] = irr_s
            ndata['irc'][i] = irc_s
            ndata['icc'][i] = icc_s
            ndata['det'][i] = det

        return new_gmix


    def convolve(self, psf):
        """
        Get a new GMix that is the convolution of the GMix with the input psf

        parameters
        ----------
        psf: GMix object
        """
        if not isinstance(psf, GMix):
            raise TypeError("Can only convolve with another GMix "
                            " got type %s" % type(psf))

        ng=len(self)*len(psf)
        output = GMix(ngauss=ng)

        gm=self._get_gmix_data()
        _gmix.convolve_fill(output._data, gm, psf._data)
        return output

    def make_image(self, dims, nsub=1, npoints=None, jacobian=None, fast_exp=False):
        """
        Render the mixture into a new image

        parameters
        ----------
        dims: 2-element sequence
            dimensions [nrows, ncols]
        nsub: integer, optional
            Defines a grid for sub-pixel integration
        fast_exp: bool, optional
            use fast, approximate exp function
        """

        image=numpy.zeros(dims, dtype='f8')
        self._fill_image(image, nsub=nsub, npoints=npoints, jacobian=jacobian, fast_exp=fast_exp)
        return image

    def make_round(self, preserve_size=False):
        """
        make a round version of the mixture

        The transformation is performed as if a shear were applied,
        so

            Tround = T * (1-g^2) / (1+g^2)

        The center of all gaussians is set to the common mean

        returns
        -------
        New round gmix
        """
        from . import shape

        gm = self.copy()

        g1,g2,T=gm.get_g1g2T()

        if preserve_size:
            factor=1.0
        else:
            factor = shape.get_round_factor(g1,g2)

        gdata=gm._get_gmix_data()

        ngauss=len(gm)
        for i in xrange(ngauss):
            Ti = gdata['irr'][i] + gdata['icc'][i]
            gdata['irc'][i] = 0.0
            gdata['irr'][i] = 0.5*Ti*factor
            gdata['icc'][i] = 0.5*Ti*factor

        row,col=gm.get_cen()
        gm.set_cen(row, col)

        return gm


    def _fill_image(self, image, npoints=None, nsub=1, jacobian=None, fast_exp=False):
        """
        Internal routine.  Render the mixture into a new image.  No error
        checking on the image!

        parameters
        ----------
        image: 2-d double array
            image to render into
        nsub: integer, optional
            Defines a grid for sub-pixel integration
        fast_exp: bool, optional
            use fast, approximate exp function
        """

        if fast_exp:
            fexp = 1
        else:
            fexp = 0

        gm=self._get_gmix_data()
        if jacobian is not None:
            assert isinstance(jacobian,Jacobian)
            if npoints is not None:
                _gmix.render_jacob_gauleg(gm,
                                          image,
                                          npoints,
                                          jacobian._data,
                                          fexp)
            else:
                _gmix.render_jacob(gm,
                                   image,
                                   nsub,
                                   jacobian._data,
                                   fexp)
        else:
            if npoints is not None:
                _gmix.render_gauleg(gm, image, npoints, fexp)
            else:
                _gmix.render(gm, image, nsub, fexp)


    def fill_fdiff(self, obs, fdiff, start=0, nsub=1, npoints=None, nocheck=False):
        """
        Fill fdiff=(model-data)/err given the input Observation

        parameters
        ----------
        obs: Observation
            The Observation to compare with. See ngmix.observation.Observation
            The Observation must have a weight map set
        fdiff: 1-d array
            The fdiff to fill
        start: int, optional
            Where to start in the array, default 0
        """

        if obs.jacobian is not None:
            assert isinstance(obs.jacobian,Jacobian)

        if not nocheck:
            fdiff = numpy.ascontiguousarray(fdiff, dtype='f8')

        nuse=fdiff.size-start

        image=obs.image
        if nuse < image.size:
            raise ValueError("fdiff from start must have "
                             "len >= %d, got %d" % (image.size,nuse))
        assert nsub >= 1,"nsub must be >= 1"

        gm=self._get_gmix_data()
        if npoints is not None:
            s2n_numer,s2n_denom,npix=_gmix.fill_fdiff_gauleg(gm,
                                                             image,
                                                             obs.weight,
                                                             obs.jacobian._data,
                                                             fdiff,
                                                             start,
                                                             npoints)
        elif nsub > 1:
            s2n_numer,s2n_denom,npix=_gmix.fill_fdiff_sub(gm,
                                                          image,
                                                          obs.weight,
                                                          obs.jacobian._data,
                                                          fdiff,
                                                          start,
                                                          nsub)
        else:
            s2n_numer,s2n_denom,npix=_gmix.fill_fdiff(gm,
                                                      image,
                                                      obs.weight,
                                                      obs.jacobian._data,
                                                      fdiff,
                                                      start)

        return {'s2n_numer':s2n_numer,
                's2n_denom':s2n_denom,
                'npix':npix}

    def __call__(self, row, col, jacobian=None):
        """
        evaluate the mixture at the specified location

        no need to send jacobian unless row,col are actually image
        coords
        """

        gm=self._get_gmix_data()

        if jacobian is not None:
            assert isinstance(jacobian,Jacobian)
            return _gmix.eval_jacob(gm, jacobian._data, row, col)
        else:
            return _gmix.eval(gm, row, col)

    def get_model_s2n_sum(self, obs):
        """
        Get the s/n sum for the model, using only the weight
        map

            s2n_sum = sum(model_i^2 * ivar_i)

        The s/n would be sqrt(s2n_sum).  This sum can be
        added up over multiple images

        parameters
        ----------
        obs: Observation
            The Observation to compare with. See ngmix.observation.Observation
            The Observation must have a weight map set
        """

        if obs.jacobian is not None:
            assert isinstance(obs.jacobian,Jacobian)

        gm=self._get_gmix_data()

        s2n_sum =_gmix.get_model_s2n_sum(gm,
                                         obs.weight,
                                         obs.jacobian._data)
        return s2n_sum

    def get_model_s2n(self, obs):
        """
        Get the s/n for the model, using only the weight
        map

            s2n_sum = sum(model_i^2 * ivar_i)
            s2n = sqrt(s2n_sum)

        parameters
        ----------
        obs: Observation
            The Observation to compare with. See ngmix.observation.Observation
            The Observation must have a weight map set
        """
        
        s2n_sum = self.get_model_s2n_sum(obs)
        s2n = sqrt(s2n_sum)
        return s2n


    def get_model_s2n_Tvar_sums(self, obs, altweight=None):
        """

        Get the s/n sum and weighted var(T) related sums for the model, using
        only the weight map

            s2n_sum = sum(model_i^2 * ivar_i)
            r2sum = sum(model_i^2 * ivar_i * r^2 )
            r4sum = sum(model_i^2 * ivar_i * r^4 )

        parameters
        ----------
        obs: Observation
            The Observation to compare with. See ngmix.observation.Observation
            The Observation must have a weight map set
        """

        if obs.jacobian is not None:
            assert isinstance(obs.jacobian,Jacobian)

        gm=self._get_gmix_data()

        if altweight is not None:
            if isinstance(altweight, GMix):
                print("using altweight")
                wdata=altweight._get_gmix_data()
                res =_gmix.get_model_s2n_Tvar_sums_altweight(gm,
                                                             wdata,
                                                             obs.weight,
                                                             obs.jacobian._data)
            else:
                raise ValueError("altweight must be a GMix")

        else:

            res =_gmix.get_model_s2n_Tvar_sums(gm,
                                               obs.weight,
                                               obs.jacobian._data)

        return res

    def get_model_s2n_Tvar(self, obs, altweight=None):
        """

        Get the s/n for the model, and weighted error on T using only the
        weight map

        parameters
        ----------
        obs: Observation
            The Observation to compare with. See ngmix.observation.Observation
            The Observation must have a weight map set

        returns
        -------
        s2n, r2_mean, Tvar

        """

        s2n_sum, r2sum, r4sum = \
            self.get_model_s2n_Tvar_sums(obs, altweight=altweight)
        s2n = sqrt(s2n_sum)

        # weighted means
        r2_mean = r2sum/s2n_sum
        r4_mean = r4sum/s2n_sum

        # assume gaussian: T = 2<r^2>
        # var(T) = T^4 / nu^2 ( <r^4> )

        T = 2*r2_mean
        Tvar = T**4/( s2n**2 * r4_mean)

        #T=self.get_T()
        #Tvar = T**4 / ( s2n**2 * ( T**2 - 2*T*r2_mean + r4_mean ) )

        return s2n, r2_mean, r4_mean, Tvar


    def get_weighted_moments(self, obs, **kw):
        """
        Get the raw weighted moments of the image, using the input
        gaussian mixture as the weight function.  The moments are *not*
        normalized

        The weight map in the observation must be accurate for accurate
        error estimates

        parameters
        ----------
        obs: Observation
            The Observation to compare with. See ngmix.observation.Observation
            The Observation must have a weight map set

            These are moments, so there cannot be masked portions of the image,
            and the weight map of the observation is ignored.

        returns
        --------

        In the following, W is the weight function, I is the image

           Returns the folling in the 'pars' field, in this order
               sum(W * I * F[i])
           where
               F = {
                  v,
                  u,
                  u^2-v^2,
                  2*v*u,
                  u^2+v^2,
                  1.0
               }

        where v,u are in sky coordinates relative to the jacobian center.

        Also returned are the covariance sums in a 6x6 matrix

            sum( W^2 * V * F[i]*F[j] )

        where V is the variance from the weight map
        """

        if obs.jacobian is not None:
            assert isinstance(obs.jacobian,Jacobian)

        gm=self._get_gmix_data()
        pars=zeros(6)
        pcov=zeros( (6,6) )
        flags,wsum,s2n_numer,s2n_denom=_gmix.get_weighted_moments(
            obs.image,
            obs.weight,
            obs.jacobian._data,
            gm,

            pars, # these get modified internally
            pcov,
        )

        flagstr=_moms_flagmap[flags]
        return {
            'flags':flags,
            'flagstr':flagstr,

            'pars':pars,
            'pars_cov':pcov,

            'wsum':wsum,

            's2n_numer_sum':s2n_numer,
            's2n_denom_sum':s2n_denom,
        }


    def get_loglike(self, obs, nsub=1, npoints=None, more=False):
        """
        Calculate the log likelihood given the input Observation


        parameters
        ----------
        obs: Observation
            The Observation to compare with. See ngmix.observation.Observation
            The Observation must have a weight map set
        nsub: int, optional
            Integrate the model over each pixel using a nsubxnsub grid
        more:
            if True, return a dict with more informatioin
        """

        if obs.jacobian is not None:
            assert isinstance(obs.jacobian,Jacobian)

        gm=self._get_gmix_data()
        if npoints is not None:
            loglike,s2n_numer,s2n_denom,npix=_gmix.get_loglike_gauleg(gm,
                                                                      obs.image,
                                                                      obs.weight,
                                                                      obs.jacobian._data,
                                                                      npoints)

        elif nsub > 1:
            #print("doing nsub")
            loglike,s2n_numer,s2n_denom,npix=_gmix.get_loglike_sub(gm,
                                                                   obs.image,
                                                                   obs.weight,
                                                                   obs.jacobian._data,
                                                                   nsub)

        else:
            if obs.has_aperture():
                aperture=obs.get_aperture()
                #print("using aper:",aperture)
                loglike,s2n_numer,s2n_denom,npix=_gmix.get_loglike_aper(gm,
                                                                        obs.image,
                                                                        obs.weight,
                                                                        obs.jacobian._data,
                                                                        aperture)


            else:
                loglike,s2n_numer,s2n_denom,npix=_gmix.get_loglike(gm,
                                                                   obs.image,
                                                                   obs.weight,
                                                                   obs.jacobian._data)

        if more:
            return {'loglike':loglike,
                    's2n_numer':s2n_numer,
                    's2n_denom':s2n_denom,
                    'npix':npix}
        else:
            return loglike

    def get_loglike_robust(self, obs, nu, nsub=1, more=False):
        """
        Calculate the log likelihood given the input Observation
        using robust likelihood

        parameters
        ----------
        obs: Observation
            The Observation to compare with. See ngmix.observation.Observation
            The Observation must have a weight map set
        nu: parameter for robust likelihood - nu > 2, nu -> \infty is a Gaussian (or chi^2)
        """
        #print("using robust")
        assert nsub==1,"nsub must be 1 for robust"


        if obs.jacobian is not None:
            assert isinstance(obs.jacobian,Jacobian)

        gm=self._get_gmix_data()
        loglike,s2n_numer,s2n_denom,npix=_gmix.get_loglike_robust(gm,
                                                                  obs.image,
                                                                  obs.weight,
                                                                  obs.jacobian._data,
                                                                  nu)

        if more:
            return {'loglike':loglike,
                    's2n_numer':s2n_numer,
                    's2n_denom':s2n_denom,
                    'npix':npix}
        else:
            return loglike

    def get_loglike_margsky(self, obs, model_image, nsub=1, more=False):
        """
        Calculate the log likelihood given the input Observation, subtracting
        the mean of the image and model.  The model is first rendered into the
        input image so that rendering does not happen twice


        parameters
        ----------
        obs: Observation
            The Observation to compare with. See ngmix.observation.Observation
            The Observation must have a weight map set
            The Observation must have image_mean set
        model_image: 2-d double array
            image to render model into
        nsub: integer, optional
            Defines a grid for sub-pixel integration 
        """

        #print("using margsky")
        image=obs.image

        dt=model_image.dtype.descr[0][1]

        mess="image must be '%s', got '%s'"
        assert dt == self._f8_type,mess % (self._f8_type,dt)

        assert len(model_image.shape)==2,"image must be 2-d"
        assert model_image.shape==image.shape,"image and model must be same shape"

        model_image[:,:]=0
        self._fill_image(model_image, nsub=nsub, jacobian=obs.jacobian)

        model_mean=_gmix.get_image_mean(model_image, obs.weight)

        loglike,s2n_numer,s2n_denom,npix=\
                _gmix.get_loglike_images_margsky(image,
                                                 obs.image_mean,
                                                 obs.weight,
                                                 model_image,
                                                 model_mean)

        if more:
            return {'loglike':loglike,
                    's2n_numer':s2n_numer,
                    's2n_denom':s2n_denom,
                    'npix':npix}
        else:
            return loglike

    def _get_gmix_data(self):
        """
        same as get_data for normal models, but not all
        """
        return self._data

    def reset(self):
        """
        Replace the data array with a zeroed one.
        """
        self._data = zeros(self._ngauss, dtype=_gauss2d_dtype)

    def make_galsim_object(self):
        """
        make a galsim representation for the gaussian mixture
        """
        import galsim

        data = self._get_gmix_data()

        row,col = self.get_cen()

        gsobjects=[]
        for i in xrange(len(self)):
            flux = data['p'][i]
            T = data['irr'][i] + data['icc'][i]
            e1 = (data['icc'][i] - data['irr'][i])/T
            e2 = 2.0*data['irc'][i]/T

            rowshift = data['row'][i]-row
            colshift = data['col'][i]-col

            g1,g2=e1e2_to_g1g2(e1,e2)

            Tround = moments.get_Tround(T, g1, g2)
            sigma_round = sqrt(Tround/2.0)

            gsobj = galsim.Gaussian(flux=flux, sigma=sigma_round)

            gsobj = gsobj.shear(g1=g1, g2=g2)
            gsobj = gsobj.shift(colshift, rowshift)

            gsobjects.append( gsobj )

        gs_obj = galsim.Add(gsobjects)

        #rowshift = row-int(row)-0.5
        #colshift = col-int(col)-0.5
        #gs_obj = gs_obj.shift(colshift, rowshift)

        return gs_obj

    def __len__(self):
        return self._ngauss

    def __repr__(self):
        rep=[]
        #fmt="p: %-10.5g row: %-10.5g col: %-10.5g irr: %-10.5g irc: %-10.5g icc: %-10.5g"
        fmt="p: %.4g row: %.4g col: %.4g irr: %.4g irc: %.4g icc: %.4g"
        for i in xrange(self._ngauss):
            t=self._data[i]
            s=fmt % (t['p'],t['row'],t['col'],t['irr'],t['irc'],t['icc'])
            rep.append(s)

        rep='\n'.join(rep)
        return rep

class GMixList(list):
    """
    Hold a list of GMix objects

    This class provides a bit of type safety and ease of type checking
    """
    def append(self, gmix):
        """
        Add a new mixture

        over-riding this for type safety
        """
        assert isinstance(gmix,GMix),"gmix should be of type GMix"
        super(GMixList,self).append(gmix)

    def __setitem__(self, index, gmix):
        """
        over-riding this for type safety
        """
        assert isinstance(gmix,GMix),"gmix should be of type GMix"
        super(GMixList,self).__setitem__(index, gmix)

class MultiBandGMixList(list):
    """
    Hold a list of lists of GMixList objects, each representing a filter
    band

    This class provides a bit of type safety and ease of type checking
    """

    def append(self, gmix_list):
        """
        add a new GMixList

        over-riding this for type safety
        """
        assert isinstance(gmix_list,GMixList),"gmix_list should be of type GMixList"
        super(MultiBandGMixList,self).append(gmix_list)

    def __setitem__(self, index, gmix_list):
        """
        over-riding this for type safety
        """
        assert isinstance(gmix_list,GMixList),"gmix_list should be of type GMixList"
        super(MultiBandGMixList,self).__setitem__(index, gmix_list)



class GMixModel(GMix):
    """
    A two-dimensional gaussian mixture created from a set of model parameters

    Inherits from the more general GMix class, and all its methods.

    parameters
    ----------
    pars: array-like
        Parameter array. The number of elements will depend
        on the model type.
    model: string or gmix type
        e.g. 'exp' or GMIX_EXP
    """
    def __init__(self, pars, model):

        self._set_f8_type()
        self._model      = _gmix_model_dict[model]
        self._model_name = _gmix_string_dict[self._model]

        self._ngauss = _gmix_ngauss_dict[self._model]
        self._npars  = _gmix_npars_dict[self._model]

        self.reset()
        self.fill(pars)

    def copy(self):
        """
        Get a new GMix with the same parameters
        """
        gmix = GMixModel(self._pars, self._model_name)
        return gmix

    def set_cen(self, row, col):
        """
        Move the mixture to a new center

        set pars as well
        """
        gm=self._get_gmix_data()

        pars=self._pars
        row0,col0=self.get_cen()

        row_shift = row - row0
        col_shift = col - col0

        gm['row'] += row_shift
        gm['col'] += col_shift

        pars[0] = row
        pars[1] = col

    def fill(self, pars):
        """
        Fill in the gaussian mixture with new parameters

        parameters
        ----------
        pars: ndarray or sequence
            The parameters
        """


        pars = array(pars, dtype='f8', copy=True) 

        if pars.size != self._npars:
            err="model '%s' requires %s pars, got %s"
            err =err % (self._model_name,self._npars, pars.size)
            raise GMixFatalError(err)

        self._pars = pars

        gm=self._get_gmix_data()
        _gmix.gmix_fill(gm, pars, self._model)


class GMixCM(GMix):
    """
    Composite Model exp and dev using just fracdev

    Inherits from the more general GMix class, and all its methods.

    parameters
    ----------
    pars: array-like
        Parameter array. The number of elements will depend
        on the model type.
    model: string or gmix type
        e.g. 'exp' or GMIX_EXP
    """
    def __init__(self, fracdev, TdByTe, pars):

        self._fracdev = fracdev
        self._TdByTe = TdByTe
        self._Tfactor = _gmix.get_cm_Tfactor(fracdev, TdByTe)

        self._model      = _gmix_model_dict['fracdev']
        self._model_name = _gmix_string_dict[self._model]

        self._ngauss = _gmix_ngauss_dict[self._model]
        self._npars  = _gmix_npars_dict[self._model]

        self.reset()

        self.fill(pars)


    def copy(self):
        """
        Get a new GMix with the same parameters
        """
        #gmix = GMixCM(self._exp_pars, self._dev_pars, self._fracdev)
        gmix = GMixCM(self._fracdev,
                      self._TdByTe,
                      self._pars)
        return gmix

    def reset(self):
        """
        Replace the data array with a zeroed one.
        """
        self._data = zeros(self._ngauss, dtype=_cm_dtype)
        self._data['fracdev'][0] = self._fracdev
        self._data['TdByTe'][0] = self._TdByTe
        self._data['Tfactor'][0] = self._Tfactor


    def fill(self, pars):
        """
        Fill in the gaussian mixture with new parameters

        parameters
        ----------
        pars: ndarray or sequence
            The parameters
        """

        pars = array(pars, dtype='f8', copy=True) 

        if pars.size != 6:
            raise GMixFatalError("must have 6 pars")

        self._pars = pars

        data=self.get_data()
        _gmix.gmix_fill_cm(data, pars)

    def _get_gmix_data(self):
        """
        same as get_data for normal models, but not all
        """
        return self._data['gmix'][0]


    def __repr__(self):
        rep=[]
        #fmt="p: %-10.5g row: %-10.5g col: %-10.5g irr: %-10.5g irc: %-10.5g icc: %-10.5g"
        fmt="p: %.4g row: %.4g col: %.4g irr: %.4g irc: %.4g icc: %.4g"

        gm=self._get_gmix_data()
        for i in xrange(self._ngauss):
            t=gm[i]
            s=fmt % (t['p'],t['row'],t['col'],t['irr'],t['irc'],t['icc'])
            rep.append(s)

        rep='\n'.join(rep)
        return rep



def get_coellip_npars(ngauss):
    return 4 + 2*ngauss

def get_coellip_ngauss(npars):
    return (npars-4)/2

class GMixCoellip(GMixModel):
    """
    A two-dimensional gaussian mixture, each co-centeric and co-elliptical

    Inherits from the more general GMix class, and all its methods.

    parameters
    ----------
    pars: array-like
        Parameter array. The number of elements will depend
        on the model type.
    """


    def __init__(self, pars):

        self._model      = GMIX_COELLIP
        self._model_name = 'coellip'
        pars = array(pars, dtype='f8', copy=True) 

        npars=pars.size

        ncheck=npars-4
        if ( ncheck % 2 ) != 0:
            raise ValueError("coellip must have len(pars)==4+2*ngauss, got %s" % npars)

        self._pars=pars
        self._ngauss = ncheck/2
        self._npars = npars

        self.reset()
        gm=self._get_gmix_data()
        _gmix.gmix_fill(gm, pars, self._model)

    def fill(self, pars):
        """
        Fill in the gaussian mixture with new parameters
        """

        pars = array(pars, dtype='f8', copy=True) 

        if pars.size != self._npars:
            raise ValueError("input pars have size %d, "
                             "expected %d" % (pars.size, self._npars))

        self._pars[:]=pars[:]

        gm=self._get_gmix_data()
        _gmix.gmix_fill(self._data, pars, self._model)

    def copy(self):
        """
        Get a new GMix with the same parameters
        """
        gmix = GMixCoellip(self._pars)
        return gmix


def cbinary_search(a, x):
    """
    use weave inline for speed
    """
    import scipy.weave
    from scipy.weave import inline
    from scipy.weave.converters import blitz

    size=a.size

    code="""
    long up=size;
    long down=-1;

    for (;;) {
        if ( x < a(0) ) {
            return_val =  0;
            break;
        }
        if (x > a(up-1)) {
            return_val =  up-1;
            break;
        }

        long mid=0;
        double val=0;
        while ( (up-down) > 1 ) {
            mid = down + (up-down)/2;
            val=a(mid);

            if (x >= val) {
                down=mid;
            } else {
                up=mid;
            }
     
        }
        return_val = down;
    }
    """
    down=inline(code, ['a','x','size'],
                type_converters=blitz,
                compiler='gcc')

    return down



def cinterp_multi_scalar(xref, yref, xinterp, output):
    import scipy.weave
    from scipy.weave import inline
    from scipy.weave.converters import blitz

    npoints=xref.size
    ndim=yref.shape[1]

    ilo = cbinary_search(xref, xinterp)

    code="""
    double x=xinterp;

    if (ilo < 0) {
        ilo=0;
    }
    if (ilo >= (npoints-1)) {
        ilo=npoints-2;
    }

    int ihi = ilo+1;

    double xlo=xref(ilo);
    double xhi=xref(ihi);
    double xdiff = xhi-xlo;
    double xmxlo = x-xlo;

    for (int i=0; i<ndim; i++) {

        double ylo = yref(ilo, i);
        double yhi = yref(ihi, i);
        double ydiff = yhi - ylo;

        double slope = ydiff/xdiff;

        output(i) = xmxlo*slope + ylo;

    }

    return_val=1;
    """

    inline(code, ['xref','yref','xinterp','ilo','npoints','ndim','output'],
           type_converters=blitz)#, compiler='gcc')






MIN_SERSIC_N=0.751
MAX_SERSIC_N=5.999


_sersic_nvals_10gauss=array([0.75, 1.0, 1.05, 1.25, 1.5, 2.00, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00, 5.50, 6.00])
_sersic_data_10gauss=array([
    [0.000249964,  0.00160301,  0.00626292,  0.0192913,  0.0514659,  0.124931,  0.28234,  0.593018,  1.13983,  2.01192,  1.69267e-06,  2.49274e-05,  0.000196065,  0.00113602,  0.00553923,  0.0243836,  0.0958122,  0.287113,  0.434863,  0.15093], 

    [0.000141183,  0.00103725,  0.00453358,  0.0154615,  0.0452213,  0.118812,  0.288153,  0.657267,  1.43447,  3.09618,  8.69196e-06,  0.000114263,  0.00083198,  0.00442365,  0.0188262,  0.0653268,  0.177007,  0.32906,  0.317786,  0.0866153], 
    [0.000120697,  0.000909617,  0.0040539,  0.0140571,  0.0417659,  0.111547,  0.275608,  0.64312,  1.44609,  3.25247,  1.0192e-05,  0.000132091,  0.000947265,  0.00494159,  0.0205266,  0.0691038,  0.180863,  0.325843,  0.310749,  0.086884], 
    [6.39097e-05,  0.000527167,  0.00252544,  0.00935232,  0.0296281,  0.0845536,  0.224712,  0.571277,  1.4312,  3.73464,  1.58929e-05,  0.000195935,  0.00133795,  0.00658759,  0.0254544,  0.0785048,  0.18699,  0.31222,  0.29498,  0.0937132], 
    [2.9818e-05,  0.00027051,  0.00140013,  0.0055675,  0.0189225,  0.0581109,  0.167323,  0.466364,  1.3084,  3.997,  2.20616e-05,  0.000258835,  0.00168076,  0.00781816,  0.0283343,  0.0816486,  0.182828,  0.295325,  0.290403,  0.111681], 
    [7.18653e-06,  7.68446e-05,  0.00045508,  0.0020511,  0.00789117,  0.0275417,  0.0909392,  0.295217,  0.992545,  3.89542,  2.8716e-05,  0.000317049,  0.00193356,  0.00841468,  0.0284897,  0.0771123,  0.165373,  0.268048,  0.293013,  0.15727], 
    [1.98179e-06,  2.47646e-05,  0.000165708,  0.000835235,  0.00358485,  0.0139879,  0.0519524,  0.191828,  0.748381,  3.58525,  2.93838e-05,  0.000317696,  0.00188226,  0.00794749,  0.0261771,  0.0694859,  0.148591,  0.247823,  0.296409,  0.201336], 
    [6.50627e-07,  9.20752e-06,  6.82515e-05,  0.00037868,  0.0017871,  0.00768484,  0.0316306,  0.130712,  0.580699,  3.29779,  2.84846e-05,  0.000300823,  0.00174627,  0.00724537,  0.023567,  0.062296,  0.134507,  0.231803,  0.298465,  0.240041], 
    [3.70285e-07,  4.94355e-06,  3.70333e-05,  0.000213576,  0.0010627,  0.00486769,  0.0215516,  0.096939,  0.476936,  3.10563,  3.92413e-05,  0.000343401,  0.00182922,  0.00718711,  0.0225433,  0.0583249,  0.125256,  0.219312,  0.296521,  0.268644], 
    [2.95933e-07,  3.46867e-06,  2.48365e-05,  0.000143079,  0.000727616,  0.00345813,  0.0160859,  0.0770072,  0.410127,  2.98125,  6.50736e-05,  0.000441826,  0.00208608,  0.00759116,  0.0226025,  0.0565324,  0.11939,  0.209696,  0.292542,  0.289053], 
    [2.63657e-07,  2.75208e-06,  1.8623e-05,  0.000105636,  0.000541346,  0.00263354,  0.0126999,  0.0638365,  0.362754,  2.89231,  0.000107377,  0.000578318,  0.00242603,  0.00817028,  0.023058,  0.0556427,  0.11522,  0.202006,  0.288213,  0.304577], 
    [2.47725e-07,  2.35418e-06,  1.5056e-05,  8.35572e-05,  0.000427966,  0.0021121,  0.0104602,  0.0546547,  0.327721,  2.82692,  0.000170701,  0.00075224,  0.00283117,  0.00886048,  0.023743,  0.0552981,  0.112159,  0.195717,  0.283881,  0.316588], 
    [2.40362e-07,  2.1193e-06,  1.28581e-05,  6.96343e-05,  0.000354659,  0.00176456,  0.00891162,  0.0480232,  0.301129,  2.77845,  0.00026027,  0.000964948,  0.0032944,  0.00963429,  0.0245826,  0.0553232,  0.109883,  0.190484,  0.279651,  0.325921], 
    [2.3827e-07,  1.97827e-06,  1.14419e-05,  6.04297e-05,  0.000305109,  0.0015236,  0.00780465,  0.0431059,  0.280569,  2.74262,  0.000381366,  0.00121753,  0.00381056,  0.0104738,  0.0255305,  0.0556082,  0.108184,  0.186068,  0.275568,  0.333157]
])



GMIX_FULL=0
GMIX_GAUSS=1
GMIX_TURB=2
GMIX_EXP=3
GMIX_DEV=4
GMIX_BDC=5
GMIX_BDF=6
GMIX_COELLIP=7
GMIX_SERSIC=8
GMIX_FRACDEV=9

# Composite Model
GMIX_CM=10

# moments
GMIX_GAUSSMOM=11

_gmix_model_dict={'full':       GMIX_FULL,
                  GMIX_FULL:    GMIX_FULL,
                  'gauss':      GMIX_GAUSS,
                  GMIX_GAUSS:   GMIX_GAUSS,
                  'turb':       GMIX_TURB,
                  GMIX_TURB:    GMIX_TURB,
                  'exp':        GMIX_EXP,
                  GMIX_EXP:     GMIX_EXP,
                  'dev':        GMIX_DEV,
                  GMIX_DEV:     GMIX_DEV,
                  'bdc':        GMIX_BDC,
                  GMIX_BDC:     GMIX_BDC,
                  'bdf':        GMIX_BDF,
                  GMIX_BDF:     GMIX_BDF,

                  GMIX_FRACDEV: GMIX_FRACDEV,
                  'fracdev': GMIX_FRACDEV,

                  GMIX_CM: GMIX_CM,
                  'cm': GMIX_CM,

                  'coellip':    GMIX_COELLIP,
                  GMIX_COELLIP: GMIX_COELLIP,

                  'sersic':    GMIX_SERSIC,
                  GMIX_SERSIC: GMIX_SERSIC,
                
                  'gaussmom': GMIX_GAUSSMOM,
                  GMIX_GAUSSMOM: GMIX_GAUSSMOM}

_gmix_string_dict={GMIX_FULL:'full',
                   'full':'full',
                   GMIX_GAUSS:'gauss',
                   'gauss':'gauss',
                   GMIX_TURB:'turb',
                   'turb':'turb',
                   GMIX_EXP:'exp',
                   'exp':'exp',
                   GMIX_DEV:'dev',
                   'dev':'dev',
                   GMIX_BDC:'bdc',
                   'bdc':'bdc',
                   GMIX_BDF:'bdf',
                   'bdf':'bdf',

                   GMIX_FRACDEV:'fracdev',
                   'fracdev':'fracdev',

                   GMIX_CM:'cm',
                   'cm':'cm',

                   GMIX_COELLIP:'coellip',
                   'coellip':'coellip',

                   GMIX_SERSIC:'sersic',
                   'sersic':'sersic',
                   
                   GMIX_GAUSSMOM:'gaussmom',
                   'gaussmom':'gaussmom',
                  }


_gmix_npars_dict={GMIX_GAUSS:6,
                  GMIX_TURB:6,
                  GMIX_EXP:6,
                  GMIX_DEV:6,

                  GMIX_FRACDEV:1,
                  GMIX_CM:6,

                  GMIX_BDC:8,
                  GMIX_BDF:7,
                  GMIX_SERSIC:7,
                  GMIX_GAUSSMOM: 6}

_gmix_ngauss_dict={GMIX_GAUSS:1,
                   'gauss':1,
                   GMIX_TURB:3,
                   'turb':3,
                   GMIX_EXP:6,
                   'exp':6,
                   GMIX_DEV:10,
                   'dev':10,

                   GMIX_FRACDEV:16,

                   GMIX_CM:16,

                   GMIX_BDC:16,
                   GMIX_BDF:16,
                   GMIX_SERSIC:4,
                   GMIX_GAUSSMOM: 1,

                   'em1':1,
                   'em2':2,
                   'em3':3,
                   'coellip1':1,
                   'coellip2':2,
                   'coellip3':3}


_gauss2d_dtype=[('p','f8'),
                ('row','f8'),
                ('col','f8'),
                ('irr','f8'),
                ('irc','f8'),
                ('icc','f8'),
                ('det','f8'),
                ('norm_set','i4'),
                ('drr','f8'),
                ('drc','f8'),
                ('dcc','f8'),
                ('norm','f8'),
                ('pnorm','f8')]

_cm_dtype=[('fracdev','f8'),
                  ('TdByTe','f8'), # ratio Tdev/Texp
                  ('Tfactor','f8'),
                  ('gmix',_gauss2d_dtype,16)]

def get_model_num(model):
    """
    Get the numerical identifier for the input model,
    which could be string or number
    """
    return _gmix_model_dict[model]

def get_model_name(model):
    """
    Get the string identifier for the input model,
    which could be string or number
    """
    return _gmix_string_dict[model]

def get_model_npars(model):
    """
    Get the number of parameters for the input model,
    which could be string or number
    """
    mi=_gmix_model_dict[model]
    return _gmix_npars_dict[mi]


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

        self.weights = numpy.array(weights, dtype='f8', copy=True)
        self.means=numpy.array(means, dtype='f8', copy=True)
        self.covars=numpy.array(covars, dtype='f8', copy=True)

        self.ngauss = self.weights.size

        sh=means.shape
        if len(sh) == 1:
            raise ValueError("means must be 2-d even for ndim=1")

        self.ndim = sh[1]

        self._calc_icovars_and_norms()

        self.tmp_lnprob = zeros(self.ngauss)

    def fit(self, data, ngauss, n_iter=5000, min_covar=1.0e-6):
        """
        data is shape
            [npoints, ndim]
        """
        from sklearn.mixture import GMM

        if len(data.shape) == 1:
            data = data[:,numpy.newaxis]

        print("ngauss:   ",ngauss)
        print("n_iter:   ",n_iter)
        print("min_covar:",min_covar)

        gmm=GMM(n_components=ngauss,
                n_iter=n_iter,
                min_covar=min_covar,
                covariance_type='full')

        gmm.fit(data)

        if not gmm.converged_:
            print("DID NOT CONVERGE")

        self._gmm=gmm
        self.set_mixture(gmm.weights_, gmm.means_, gmm.covars_)

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
                                       dolog)
        return p


    def get_lnprob_array(self, pars):
        """
        array input
        """
        dolog=1
        n=pars.shape[0]
        lnp=zeros(n)

        for i in xrange(n):
            lnp[i] = self.get_lnprob_scalar(pars[i,:])

        return lnp

    def get_prob_array(self, pars):
        """
        array input
        """
        dolog=0
        n=pars.shape[0]
        p=zeros(n)

        for i in xrange(n):
            p[i] = self.get_prob_scalar(pars[i,:])

        return p

    def sample(self, n=None):
        """
        sample from the gaussian mixture
        """
        if not hasattr(self, '_gmm'):
            self._make_gmm()

        if n is None:
            is_one=True
            n=1
            if self.ndim==1:
                is_scalar=1
        else:
            is_one=False

        samples=self._gmm.sample(n)

        if is_one:
            samples = samples[0,:]
            if is_one:
                samples = samples[0]
        return samples



    def _make_gmm(self):
        """
        Make a GMM object for sampling
        """
        from sklearn.mixture import GMM

        # these numbers are not used because we set the means, etc by hand
        ngauss=self.weights.size
        gmm=GMM(n_components=self.ngauss,
                n_iter=10000,
                min_covar=1.0e-12,
                covariance_type='full',
                random_state=self.rng)
        gmm.means_ = self.means.copy()
        gmm.covars_ = self.covars.copy()
        gmm.weights_ = self.weights.copy()

        self._gmm=gmm 

    def _calc_icovars_and_norms(self):
        """
        Calculate the normalizations and inverse covariance matrices
        """
        from numpy import pi

        twopi = 2.0*pi

        #if self.ndim==1:
        if False:
            norms = 1.0/sqrt(twopi*self.covars)
            icovars = 1.0/self.covars
        else:
            norms = zeros(self.ngauss)
            icovars = zeros( (self.ngauss, self.ndim, self.ndim) )
            for i in xrange(self.ngauss):
                cov = self.covars[i,:,:]
                icov = numpy.linalg.inv( cov )

                det = numpy.linalg.det(cov)
                n=1.0/sqrt( twopi**self.ndim * det )

                norms[i] = n
                icovars[i,:,:] = icov

        self.norms = norms
        self.pnorms = norms*self.weights
        self.log_pnorms = log(self.pnorms)
        self.icovars = icovars



_moms_flagmap={
    0:'ok',
    1:'zero weight encountered',
}


