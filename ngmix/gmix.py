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
from .shape import g1g2_to_e1e2, e1e2_to_g1g2

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

    def make_image(self, dims, nsub=1, jacobian=None):
        """
        Render the mixture into a new image

        parameters
        ----------
        dims: 2-element sequence
            dimensions [nrows, ncols]
        nsub: integer, optional
            Defines a grid for sub-pixel integration 
        """

        image=numpy.zeros(dims, dtype='f8')
        self._fill_image(image, nsub=nsub, jacobian=jacobian)
        return image

    def make_round(self):
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


    def _fill_image(self, image, nsub=1, jacobian=None):
        """
        Internal routine.  Render the mixture into a new image.  No error
        checking on the image!

        parameters
        ----------
        image: 2-d double array
            image to render into
        nsub: integer, optional
            Defines a grid for sub-pixel integration 
        """

        gm=self._get_gmix_data()
        if jacobian is not None:
            _gmix.render_jacob(gm,
                               image,
                               nsub,
                               jacobian._data)
        else:
            _gmix.render(gm,
                         image,
                         nsub)


    def fill_fdiff(self, obs, fdiff, start=0, nsub=1):
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

        nuse=fdiff.size-start

        image=obs.image
        if nuse < image.size:
            raise ValueError("fdiff from start must have "
                             "len >= %d, got %d" % (image.size,nuse))
        assert nsub >= 1,"nsub must be >= 1"

        gm=self._get_gmix_data()
        if nsub > 1:
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


    def get_weighted_mom_sums(self,
                              obs,
                              maxiter=100,
                              centol=1.0e-4,
                              max_shift=5.0,
                              **kw):
        """
        Get the raw weighted moment sums of the image, using the input
        gaussian mixture as the weight function.  The moments are *not*
        normalized

        Just iterating for the centroid, with the first location taken as the
        jacobian center, so you should have a good guess

        parameters
        ----------
        obs: Observation
            The Observation to compare with. See ngmix.observation.Observation
            The Observation must have a weight map set

            These are moments, so there cannot be masked portions of the image,
            and the weight map of the observation is ignored.
        maxiter: int, optional
            Maximum number of iterations to find the center
        centol: float, optional
            Tolerance to find the center in either direction
        max_shift: float, optional
            Max allowed shift in centroid

        returns
        --------

        In the following, W is the weight function, I is the image, and
        w is the weight map

           Returns
               ucen  = sum(W*I*u)/sum(W*I)
               vcen  = sum(W*I*v)/sum(W*I)
               Isum  = sum(W*I)
               Tsum  = sum(W * I * {u^2 + v^2} )
               M1sum = sum(W * I * {u^2 - v^2} )
               M2sum = sum(W * I * 2*u*v)

        where u,v are relative to the jacobian center

        Also returned are sums used to calculate variances in these quantities, but
        note the covariance can be significant

               VIsum  = sum(W^2)
               VTsum  = sum(W^2 * {u^2 + v^2}^2 )
               VM1sum = sum(W^2 * {u^2 - v^2}^2 )
               VM2sum = sum(W^2 * {2*u*v}^2 )

        These should be multiplied by the noise^2 to turn them into proper variances

        """
        gm=self._get_gmix_data()
        cen=zeros(2)
        pars=zeros(6)
        pvar=zeros(6)
        niter,flags=_gmix.get_weighted_mom_sums(obs.image,
                                                gm,
                                                obs.jacobian._data,
                                                maxiter, centol,max_shift,
                                                cen,pars,pvar)
        flagstr=_moms_flagmap[flags]
        cov=diag(pvar)
        return {'cen':cen,
                'pars':pars,
                'pars_cov':cov,
                'pars_var':pvar,
                'maxiter':maxiter,
                'centol':centol,
                'niter':niter,
                'flags':flags,
                'flagstr':flagstr}
                

    def get_loglike(self, obs, nsub=1, more=False):
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

        gm=self._get_gmix_data()
        if nsub > 1:
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
        _gmix.gmix_fill(data, pars, self._model)

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





'''
@autojit
def binary_search(a, x):
    """
    Index of closest value from a smaller than x

    however, defaults to edges when out of bounds
    """

    up=a.size
    down=-1

    if x < a[0]:
        return 0
    if x > a[up-1]:
        return up-1

    while up-down > 1:
        mid = down + (up-down)//2
        val=a[mid]

        if x >= val:
            down=mid
        else:
            up=mid
        
    return down

@jit(argtypes=[float64[:], float64[:,:], float64, float64[:]])
def interp_multi_scalar(xref, yref, x, output):
    """
    parameters
    ----------
    xref: array
        shape (n,)
    yref: array
        shape (n,ndim)
    x: scalar
        point at which to interpolate
    output: array
        shape (ndim,)
    """

    np=xref.size
    ndim=output.size

    ilo = binary_search(xref, x)
    if (ilo >= (np-1)):
        ilo = np-2
    ihi = ilo + 1

    xlo=xref[ilo]
    xhi=xref[ihi]

    xdiff = xhi-xlo
    xmxlo = x-xlo

    for i in xrange(ndim):
        ylo = yref[ilo, i]
        yhi = yref[ihi, i]
        ydiff = yhi - ylo

        slope = ydiff/xdiff

        output[i] = xmxlo*slope + ylo


def interp_multi_array(xref, yref, x):
    """
    parameters
    ----------
    xref: array
        shape (n,)
    yref: array
        shape (n,ndim)
    x: array
        points at which to interpolate
    """

    ndim=yref.shape[1]
    npoints=x.size
    output=zeros( (npoints, ndim) )
    res=zeros(ndim)
    for i in xrange(npoints):
        interp_multi_scalar(xref, yref, x[i], res)
        output[i,:] = res

    return output
'''

MIN_SERSIC_N=0.751
MAX_SERSIC_N=5.999

'''
_sersic_nvals_10gauss_old=array([0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 1.05, 1.25, 1.5, 2.00, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00, 5.50, 6.00])
_sersic_data_10gauss_old=array([
    [0.000249964,  0.00160301,  0.00626292,  0.0192913,  0.0514659,  0.124931,  0.28234,  0.593018,  1.13983,  2.01192,  1.69267e-06,  2.49274e-05,  0.000196065,  0.00113602,  0.00553923,  0.0243836,  0.0958122,  0.287113,  0.434863,  0.15093], 

    [1.18859e-08,  0.00083824,  0.00546464,  0.020253,  0.0401572,  0.0865292,  0.22175,  0.520127,  1.09655,  2.11372,  6.69953e-15,  2.22988e-05,  0.000313528,  0.00184333,  0.00289131,  0.0192785,  0.0883558,  0.28068,  0.442032,  0.164584], 
    [0.000845563,  0.0057094,  0.0104267,  0.0234296,  0.0752091,  0.19667,  0.20659,  0.503588,  1.10842,  2.2614,  3.72074e-05,  0.000518945,  5.33233e-06,  0.00402581,  0.0225919,  2.11715e-06,  0.0971305,  0.288725,  0.428993,  0.15797], 
    [0.000362725,  0.00252473,  0.0104891,  0.0171537,  0.035252,  0.0990005,  0.250001,  0.578908,  1.24676,  2.56448,  1.55889e-05,  0.000212249,  0.00153732,  0.000149174,  0.00889449,  0.0390206,  0.133883,  0.317244,  0.380284,  0.118759], 
    [0.000165536,  0.00118477,  0.00508011,  0.0170428,  0.0490626,  0.12676,  0.301526,  0.671276,  1.41873,  2.93076,  7.25969e-06,  9.70786e-05,  0.000719604,  0.00390688,  0.0170668,  0.0612083,  0.172352,  0.332055,  0.325871,  0.0867157], 

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
'''

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


'''
def fit_sersic_splines(type, ngauss, order):
    """
    Fit interpolated splines to the ngauss-sersic fits as a function
    of n
    """
    from scipy.interpolate import InterpolatedUnivariateSpline

    #print("fitting",type,"sersic splines, ngauss",ngauss,"order",order)

    if type=='T':
        start=0
    else:
        start=ngauss

    if ngauss==10:
        sersic_data=_sersic_data_10gauss
        sersic_nvals=_sersic_nvals_10gauss
    else:
        raise ValueError("support other ngauss")

    splines=[]

    for i in xrange(ngauss):
        vals=sersic_data[:,start+i]

        interpolator=InterpolatedUnivariateSpline(sersic_nvals,vals,k=order)

        splines.append(interpolator)

    return splines


class GMixSersic(GMix):
    """
    A two-dimensional gaussian mixture approximating a Sersic profile

    Inherits from the more general GMix class, and all its methods.

    parameters
    ----------
    pars: array-like
        Parameter array with elements
            [cen1,cen2,g1,g2,T,flux,n] 
    """
    T_splines = fit_sersic_splines("T", 10, 3)
    F_splines = fit_sersic_splines("F", 10, 3)

    _n_vals=_sersic_nvals_10gauss
    _tf_vals=_sersic_data_10gauss
    def __init__(self, pars):
        self._model      = GMIX_SERSIC
        self._model_name = 'sersic'
        self._npars = 7

        self._ngauss = 10

        self._interp_res=zeros(2*self._ngauss)
        self._fvals=zeros(self._ngauss)
        self._pvals=zeros(self._ngauss)

        self.reset()
        self.fill(pars)

    def fill(self, parsin):
        """
        Fill in the gaussian mixture with new parameters
        """

        pars = array(parsin, dtype='f8', copy=False) 

        npars=pars.size

        if npars != 7:
            raise ValueError("sersic models require 7 pars, got %s" % npars)

        #self._set_fvals_pvals_own(pars[6])
        self._set_fvals_pvals_spline(pars[6])
        _fill_simple(self._data, pars, self._fvals, self._pvals)

    def _set_fvals_pvals_own(self, n):
        if n < MIN_SERSIC_N or n > MAX_SERSIC_N:
            raise GMixRangeError("n out of bounds")

        _interp_vals = cinterp_multi_scalar(GMixSersic._n_vals,
                                           GMixSersic._tf_vals,
                                           n,
                                           self._interp_res)
        ngauss=self._ngauss
        self._fvals[:] = self._interp_res[0:self._ngauss]
        self._pvals[:] = self._interp_res[self._ngauss:]


    def _set_fvals_pvals_spline(self, n):
        import scipy.interpolate

        fvals=self._fvals
        pvals=self._pvals

        narr=numpy.atleast_1d(n)

        T_splines=self.T_splines
        F_splines=self.F_splines

        ngauss=self._ngauss
        for i in xrange(ngauss):
            Tinterp=T_splines[i]
            Finterp=F_splines[i]

            t,c,k=Tinterp._eval_args
            fvals[i],err=scipy.interpolate.fitpack._fitpack._spl_(narr,
                                                                  0,
                                                                  t,
                                                                  c,
                                                                  k,
                                                                  0)
            if err:
                print("error occurred in T interp")

            t,c,k=Finterp._eval_args
            pvals[i],err=scipy.interpolate.fitpack._fitpack._spl_(narr,
                                                                  0,
                                                                  t,
                                                                  c,
                                                                  k,
                                                                  0)

            if err:
                print("error occurred in F interp")

'''

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
                   GMIX_TURB:3,
                   GMIX_EXP:6,
                   GMIX_DEV:10,

                   GMIX_FRACDEV:16,

                   GMIX_CM:16,

                   GMIX_BDC:16,
                   GMIX_BDF:16,
                   GMIX_SERSIC:4,
                   GMIX_GAUSSMOM: 1}

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


'''
@jit(argtypes=[ _gauss2d[:], float64[:] ] )
def _fill_coellip(self, pars):

    npars=pars.size
    ngauss = (npars-4)/2;

    row=pars[0]
    col=pars[1]
    g1=pars[2]
    g2=pars[3]

    e1,e2 = g1g2_to_e1e2(g1, g2)

    ngauss=self.size
    for i in xrange(ngauss):

        T = pars[4+i]
        counts=pars[4+ngauss+i]

        _gauss2d_set(self,
                     i,
                     counts,
                     row,
                     col, 
                     (T/2.)*(1-e1), 
                     (T/2.)*e2,
                     (T/2.)*(1+e1))

@jit(argtypes=[ _gauss2d[:], float64[:], float64[:], float64[:] ] )
def _fill_simple(self, pars, fvals, pvals):
    row=pars[0]
    col=pars[1]
    g1=pars[2]
    g2=pars[3]
    T=pars[4]
    counts=pars[5]

    e1,e2 = g1g2_to_e1e2(g1, g2)

    ngauss=self.size
    for i in xrange(ngauss):

        T_i = T*fvals[i]
        counts_i=counts*pvals[i]

        _gauss2d_set(self,
                     i,
                     counts_i,
                     row,
                     col, 
                     (T_i/2.)*(1-e1), 
                     (T_i/2.)*e2,
                     (T_i/2.)*(1+e1))

_gauss_fvals = array([1.0],dtype='f8')
_gauss_pvals = array([1.0],dtype='f8')

@jit(argtypes=[ _gauss2d[:], float64[:] ] )
def _fill_gauss(self, pars):
    _fill_simple(self, pars, _gauss_fvals, _gauss_pvals)


_exp_pvals = array([0.00061601229677880041, 
                    0.0079461395724623237, 
                    0.053280454055540001, 
                    0.21797364640726541, 
                    0.45496740582554868, 
                    0.26521634184240478],dtype='f8')
_exp_fvals = array([0.002467115141477932, 
                    0.018147435573256168, 
                    0.07944063151366336, 
                    0.27137669897479122, 
                    0.79782256866993773, 
                    2.1623306025075739],dtype='f8')


@jit(argtypes=[ _gauss2d[:], float64[:] ] )
def _fill_exp(self, pars):
    _fill_simple(self, pars, _exp_fvals, _exp_pvals)


_dev_pvals = array([6.5288960012625658e-05, 
                    0.00044199216814302695, 
                    0.0020859587871659754, 
                    0.0075913681418996841, 
                    0.02260266219257237, 
                    0.056532254390212859, 
                    0.11939049233042602, 
                    0.20969545753234975, 
                    0.29254151133139222, 
                    0.28905301416582552],dtype='f8')
_dev_fvals = 1.025*array([2.9934935706271918e-07, 
                          3.4651596338231207e-06, 
                          2.4807910570562753e-05, 
                          0.00014307404300535354, 
                          0.000727531692982395, 
                          0.003458246439442726, 
                          0.0160866454407191, 
                          0.077006776775654429, 
                          0.41012562102501476, 
                          2.9812509778548648],dtype='f8')

@jit(argtypes=[ _gauss2d[:], float64[:] ] )
def _fill_dev(self, pars):
    _fill_simple(self, pars, _dev_fvals, _dev_pvals)


_tmp_bulge_pars=zeros(_gmix_npars_dict[GMIX_DEV])
_tmp_disk_pars=zeros(_gmix_npars_dict[GMIX_EXP])
_tmp_bulge_gmix=zeros(_gmix_ngauss_dict[GMIX_DEV],dtype=_gauss2d_dtype)
_tmp_disk_gmix=zeros(_gmix_ngauss_dict[GMIX_EXP],dtype=_gauss2d_dtype)

#@jit(argtypes=[ _gauss2d[:], float64[:] ] )
def _fill_bdc(self, pars):
    """
    Fill a bulge+disk model, co-centric and co-elliptical

    pars are [c1,c2,g1,g2,Tbulge,Tdisk,Fbulge,Fdisk]
    """
    _tmp_bulge_pars[0]=pars[0]
    _tmp_bulge_pars[1]=pars[1]
    _tmp_bulge_pars[2]=pars[2]
    _tmp_bulge_pars[3]=pars[3]
    _tmp_bulge_pars[4]=pars[4]
    _tmp_bulge_pars[5]=pars[6]

    _tmp_disk_pars[0]=pars[0]
    _tmp_disk_pars[1]=pars[1]
    _tmp_disk_pars[2]=pars[2]
    _tmp_disk_pars[3]=pars[3]
    _tmp_disk_pars[4]=pars[5]
    _tmp_disk_pars[5]=pars[7]

    _fill_dev(_tmp_bulge_gmix, _tmp_bulge_pars)
    _fill_exp(_tmp_disk_gmix,  _tmp_disk_pars)

    ng_bulge=_tmp_bulge_gmix.size
    #ng_disk=_tmp_disk_gmix.size

    #for i in xrange(ng_bulge):
    #    self[i] = _tmp_bulge_gmix[i]
    #for i in xrange(ng_disk):
    #    self[ng_bulge+i] = _tmp_disk_gmix[i]
    self[0:ng_bulge] = _tmp_bulge_gmix[:]
    self[ng_bulge:]  = _tmp_disk_gmix[:]

def _fill_bdf(self, pars):
    """
    Fill a bulge+disk model, co-centric and co-elliptical
    and with Tbulge=Tdisk

    pars are [c1,c2,g1,g2,T,Fbulge,Fdisk]
    """
    _tmp_bulge_pars[:]=pars[0:6]

    _tmp_disk_pars[:]=_tmp_bulge_pars[:]
    _tmp_disk_pars[5]=pars[6]

    _fill_dev(_tmp_bulge_gmix, _tmp_bulge_pars)
    _fill_exp(_tmp_disk_gmix,  _tmp_disk_pars)

    ng_bulge=_tmp_bulge_gmix.size

    self[0:ng_bulge] = _tmp_bulge_gmix[:]
    self[ng_bulge:]  = _tmp_disk_gmix[:]



_turb_fvals = array([0.5793612389470884,1.621860687127999,7.019347162356363],dtype='f8')
_turb_pvals = array([0.596510042804182,0.4034898268889178,1.303069003078001e-07],dtype='f8')

@jit(argtypes=[ _gauss2d[:], float64[:] ] )
def _fill_turb(self, pars):
    _fill_simple(self, pars, _turb_fvals, _turb_pvals)
'''


'''
@jit(argtypes=[ _gauss2d[:] ])
def _get_cen(self):
    row=0.0
    col=0.0
    psum=0.0

    ngauss=self.size
    for i in xrange(ngauss):
        p=self[i].p
        row += p*self[i].row
        col += p*self[i].col
        psum += p

    row /= psum
    col /= psum

    return row, col, psum
@jit(argtypes=[ _gauss2d[:], float64, float64 ])
def _set_cen(self, row, col):

    row_cur, col_cur, _ =_get_cen(self)
    row_shift = row - row_cur
    col_shift = col - col_cur

    ngauss=self.size
    for i in xrange(ngauss):
        self[i].row += row_shift
        self[i].col += col_shift

@jit(argtypes=[ _gauss2d[:] ])
def _get_T(self):
    T=0.0
    psum=0.0

    ngauss=self.size
    for i in xrange(ngauss):
        p=self[i].p
        T += (self[i].irr + self[i].icc)*p
        psum += p

    T /= psum

    return T, psum

@jit(argtypes=[ _gauss2d[:] ])
def _get_e1e2T(self):
    e1=-9999.
    e2=-9999.
    irr=0.0
    irc=0.0
    icc=0.0

    psum=0.0

    ngauss=self.size
    for i in xrange(ngauss):
        p=self[i].p

        irr += p*self[i].irr
        irc += p*self[i].irc
        icc += p*self[i].icc

        psum += p

    ipsum = 1.0/psum
    irr *= ipsum
    irc *= ipsum
    icc *= ipsum

    T = irr + icc
    
    if T > 0:
        e1 = (icc-irr)/T
        e2 = 2*irc/T

    return e1, e2, T

@jit(argtypes=[_gauss2d[:], float64[:]] )
def _fill_full(self, pars):

    ngauss=self.size

    for i in xrange(ngauss): 

        beg=i*6
        _gauss2d_set(self,
                     int64(i),
                     pars[beg+0],
                     pars[beg+1],
                     pars[beg+2],
                     pars[beg+3],
                     pars[beg+4],
                     pars[beg+5])
def convolve_fill(self, gmix, psf):
    """
    Fill "self" with gmix convolved with psf
    """
    ng=len(gmix)*len(psf)
    if ng != len(self):
        raise GMixFatalError("target gmix is wrong size, %d "
                             "instead of %d" % (len(gmix),ng))

    _convolve_fill(self._data, gmix._data, psf._data)

@jit(argtypes=[ _gauss2d[:], _gauss2d[:], _gauss2d[:] ])
def _convolve_fill(self, obj_gmix, psf_gmix):
    
    nobj=obj_gmix.size
    npsf=psf_gmix.size

    psf_rowcen,psf_colcen,psf_psum = _get_cen(psf_gmix)
    psf_ipsum=1.0/psf_psum

    iself=0
    for iobj in xrange(nobj):
        for ipsf in xrange(npsf):
            p = obj_gmix[iobj].p*psf_gmix[ipsf].p*psf_ipsum

            row = obj_gmix[iobj].row + (psf_gmix[ipsf].row-psf_rowcen)
            col = obj_gmix[iobj].col + (psf_gmix[ipsf].col-psf_colcen)

            irr = obj_gmix[iobj].irr + psf_gmix[ipsf].irr
            irc = obj_gmix[iobj].irc + psf_gmix[ipsf].irc
            icc = obj_gmix[iobj].icc + psf_gmix[ipsf].icc

            _gauss2d_set(self, iself, p, row, col, irr, irc, icc)

            iself += 1
@jit(argtypes=[ _gauss2d[:], float64[:,:], int64 ])
def _render_slow(self, image, nsub):
    """
    Adds to image; make sure to zero the iamge first if that is what you want
    """
    ngauss=self.size
    nrows=image.shape[0]
    ncols=image.shape[1]

    stepsize = 1./nsub
    offset = (nsub-1)*stepsize/2.
    areafac = 1./(nsub*nsub)

    for row in xrange(nrows):
        for col in xrange(ncols):

            # we add to existing value
            model_val=image[row,col]

            tval = 0.0
            trow = row-offset
            for irowsub in xrange(nsub):
                tcol = col-offset
                for icolsub in xrange(nsub):

                    for i in xrange(ngauss):
                        u = trow - self[i].row
                        u2 = u*u
                        v = tcol - self[i].col
                        v2 = v*v

                        uv=u*v

                        chi2=self[i].dcc*u2 + self[i].drr*v2 - 2.0*self[i].drc*uv

                        if chi2 < 25.0 and chi2 >= 0.0:
                            pnorm = self[i].pnorm
                            tval += pnorm*numpy.exp( -0.5*chi2 )
                    tcol += stepsize
                trow += stepsize

            tval *= areafac
            model_val += tval
            image[row,col] = model_val


#
# create the fast lookup table for exponentials

# for this demand chi2 < 25
_exp3_ivals, _exp3_lookup = fastmath.make_exp_lookup(-26, 0)
# for this demand chi2 < 99
#_exp3_ivals, _exp3_lookup = fastmath.make_exp_lookup(-100, 0)

@jit(argtypes=[ _gauss2d[:], float64[:,:], int64, int64, float64[:] ])
def _render_fast3(self, image, nsub, i0, expvals):
    """
    Adds to image; make sure to zero the iamge first if that is what you want

    Uses 3rd order approximation to exponential function, only for negative
    arguments or zero

    This code is a mess because we can't do inlining in numba
    """
    ngauss=self.size
    nrows=image.shape[0]
    ncols=image.shape[1]

    stepsize = 1./nsub
    offset = (nsub-1)*stepsize/2.
    areafac = 1./(nsub*nsub)

    for row in xrange(nrows):
        for col in xrange(ncols):

            # we add to existing value
            model_val=image[row,col]

            tval = 0.0
            trow = row-offset
            for irowsub in xrange(nsub):
                tcol = col-offset
                for icolsub in xrange(nsub):

                    for i in xrange(ngauss):
                        u = trow - self[i].row
                        u2 = u*u
                        v = tcol - self[i].col
                        v2 = v*v

                        uv=u*v

                        chi2=self[i].dcc*u2 + self[i].drr*v2 - 2.0*self[i].drc*uv

                        if chi2 < 25.0 and chi2 >= 0.0:
                            pnorm = self[i].pnorm
                            x = -0.5*chi2

                            # 3rd order approximation to exp
                            #if x < 0.0:
                            #    ival = int64(x-0.5)
                            #else:
                            #    ival = int64(x+0.5)
                            ival = int64(x-0.5)
                            f = x - ival
                            index = ival-i0

                            expval = expvals[index]
                            fexp = (6+f*(6+f*(3+f)))*0.16666666
                            expval *= fexp

                            tval += pnorm*expval

                    tcol += stepsize
                trow += stepsize

            tval *= areafac
            model_val += tval
            image[row,col] = model_val


# evaluate a single point
#@jit(argtypes=[ _gauss2d[:], int64, float64[:], float64, float64 ])
#def _gauss2d_like(self, i0, expvals, row, col):
@jit(argtypes=[ _gauss2d[:], float64, float64 ])
def _gauss2d_like(self, row, col):

    like = 0.0

    ngauss=self.size
    for i in xrange(ngauss):

        u = row - self[i].row
        u2 = u*u
        v = col - self[i].col
        v2 = v*v

        uv=u*v

        chi2=self[i].dcc*u2 + self[i].drr*v2 - 2.0*self[i].drc*uv

        pnorm = self[i].pnorm
        tlike = pnorm*numpy.exp(-0.5*chi2)

        like += tlike

    return like

# evaluate a single point
#@jit(argtypes=[ _gauss2d[:], int64, float64[:], float64, float64 ])
#def _gauss2d_loglike(self, i0, expvals, row, col):
@jit(argtypes=[ _gauss2d[:], float64, float64 ])
def _gauss2d_loglike(self, row, col):
    #like = _gauss2d_like(self, i0, expvals, row, col)
    like = _gauss2d_like(self, row, col)
    loglike = numpy.log( like )
    return loglike


@jit(argtypes=[ _gauss2d[:], float64[:,:], int64, _jacobian[:], int64, float64[:] ])
def _render_jacob_fast3(self, image, nsub, j, i0, expvals):
    """
    Adds to image; make sure to zero the iamge first if that is what you want

    Uses 3rd order approximation to exponential function, only for negative
    arguments or zero

    This code is a mess because we can't do inlining in numba
    """
    ngauss=self.size
    nrows=image.shape[0]
    ncols=image.shape[1]

    col0=j[0].col0
    row0=j[0].row0
    dudrow=j[0].dudrow
    dudcol=j[0].dudcol
    dvdrow=j[0].dvdrow
    dvdcol=j[0].dvdcol

    stepsize = 1./nsub
    offset = (nsub-1)*stepsize/2.
    areafac = 1./(nsub*nsub)

    ustepsize = stepsize*dudcol
    vstepsize = stepsize*dvdcol

    for row in xrange(nrows):
        for col in xrange(ncols):

            # we add to existing value
            model_val=image[row,col]

            tval = 0.0
            trow = row-offset
            lowcol = col-offset

            for irowsub in xrange(nsub):
                # always start from lowcol position, then step u,v later
                u=dudrow*(trow - row0) + dudcol*(lowcol - col0)
                v=dvdrow*(trow - row0) + dvdcol*(lowcol - col0)
                for icolsub in xrange(nsub):

                    for i in xrange(ngauss):
                        udiff=u-self[i].row
                        vdiff=v-self[i].col

                        u2 = udiff*udiff
                        v2 = vdiff*vdiff
                        uv=udiff*vdiff

                        chi2=self[i].dcc*u2 + self[i].drr*v2 - 2.0*self[i].drc*uv

                        if chi2 < 25.0 and chi2 >= 0.0:
                            pnorm = self[i].pnorm
                            x = -0.5*chi2

                            # 3rd order approximation to exp
                            #if x < 0.0:
                            #    ival = int64(x-0.5)
                            #else:
                            #    ival = int64(x+0.5)
                            ival = int64(x-0.5)
                            f = x - ival
                            index = ival-i0

                            expval = expvals[index]
                            fexp = (6+f*(6+f*(3+f)))*0.16666666
                            expval *= fexp

                            tval += pnorm*expval

                    # move u and v for each "column" step
                    u += ustepsize
                    v += vstepsize

                # step to next sub-row
                trow += stepsize

            tval *= areafac
            model_val += tval
            image[row,col] = model_val

@jit(argtypes=[ _gauss2d[:], float64[:,:], float64[:,:], int64, float64[:] ])
def _loglike_fast3(self, image, weight, i0, expvals):
    """
    using 3rd order approximation to the exponential function

    This code is a mess because we can't do inlining in numba
    """
    ngauss=self.size
    nrows=image.shape[0]
    ncols=image.shape[1]

    s2n_numer=0.0
    s2n_denom=0.0
    loglike = 0.0
    for row in xrange(nrows):
        for col in xrange(ncols):

            ivar = weight[row,col]
            if ivar < 0.0:
                ivar=0.0

            model_val=0.0
            for i in xrange(ngauss):
                u = row - self[i].row
                u2 = u*u
                v = col - self[i].col
                v2 = v*v

                uv=u*v

                chi2=self[i].dcc*u2 + self[i].drr*v2 - 2.0*self[i].drc*uv

                if chi2 < 25.0 and chi2 >= 0.0:
                    pnorm = self[i].pnorm
                    x = -0.5*chi2

                    # 3rd order approximation to exp
                    #if x < 0.0:
                    #    ival = int64(x-0.5)
                    #else:
                    #    ival = int64(x+0.5)
                    ival = int64(x-0.5)
                    f = x - ival
                    index = ival-i0

                    expval = expvals[index]
                    fexp = (6+f*(6+f*(3+f)))*0.16666666
                    expval *= fexp

                    model_val += pnorm*expval

            pixval = image[row,col]
            diff = model_val-pixval
            loglike += diff*diff*ivar
            s2n_numer += pixval*model_val*ivar
            s2n_denom += model_val*model_val*ivar

    loglike *= (-0.5)

    return loglike, s2n_numer, s2n_denom


@jit(argtypes=[ _gauss2d[:], float64[:,:], float64[:,:], _jacobian[:], int64, float64[:] ])
def _loglike_jacob_fast3(self, image, weight, j, i0, expvals):
    """
    using 3rd order approximation to the exponential function

    This code is a mess because we can't do inlining in numba
    """
    ngauss=self.size
    nrows=image.shape[0]
    ncols=image.shape[1]

    s2n_numer=0.0
    s2n_denom=0.0
    loglike = 0.0
    for row in xrange(nrows):
        u=j[0].dudrow*(row - j[0].row0) + j[0].dudcol*(0 - j[0].col0)
        v=j[0].dvdrow*(row - j[0].row0) + j[0].dvdcol*(0 - j[0].col0)

        for col in xrange(ncols):

            ivar = weight[row,col]
            #if ivar < 0.0:
            #    ivar=0.0
            
            if ivar > 0.0:
                model_val=0.0
                for i in xrange(ngauss):
                    udiff=u-self[i].row
                    vdiff=v-self[i].col

                    u2 = udiff*udiff
                    v2 = vdiff*vdiff
                    uv=udiff*vdiff

                    chi2=self[i].dcc*u2 + self[i].drr*v2 - 2.0*self[i].drc*uv

                    if chi2 < 25.0 and chi2 >= 0.0:
                        pnorm = self[i].pnorm
                        x = -0.5*chi2

                        # 3rd order approximation to exp
                        #if x < 0.0:
                        #    ival = int64(x-0.5)
                        #else:
                        #    ival = int64(x+0.5)
                        ival = int64(x-0.5)
                        f = x - ival
                        index = ival-i0
                        
                        expval = expvals[index]
                        fexp = (6+f*(6+f*(3+f)))*0.16666666
                        expval *= fexp

                        model_val += pnorm*expval
                
                pixval = image[row,col]
                diff = model_val-pixval
                loglike += diff*diff*ivar
                s2n_numer += pixval*model_val*ivar
                s2n_denom += model_val*model_val*ivar

            u += j[0].dudcol
            v += j[0].dvdcol

    loglike *= (-0.5)

    return loglike, s2n_numer, s2n_denom

@jit(argtypes=[ _gauss2d[:], float64[:,:], float64[:,:], _jacobian[:], float64[:], int64, int64, float64[:] ])
def _fdiff_jacob_fast3(self, image, weight, j, fdiff, start, i0, expvals):
    """
    using 3rd order approximation to the exponential function

    This code is a mess because we can't do inlining in numba
    """
    ngauss=self.size
    nrows=image.shape[0]
    ncols=image.shape[1]

    s2n_numer=0.0
    s2n_denom=0.0

    fdiff_i=start

    for row in xrange(nrows):
        u=j[0].dudrow*(row - j[0].row0) + j[0].dudcol*(0 - j[0].col0)
        v=j[0].dvdrow*(row - j[0].row0) + j[0].dvdcol*(0 - j[0].col0)

        for col in xrange(ncols):

            ivar = weight[row,col]
            #if ivar < 0.0:
            #    ivar=0.0
            if ivar > 0.0:
                ierr=numpy.sqrt(ivar)

                model_val=0.0
                for i in xrange(ngauss):
                    udiff=u-self[i].row
                    vdiff=v-self[i].col

                    u2 = udiff*udiff
                    v2 = vdiff*vdiff
                    uv=udiff*vdiff

                    chi2=self[i].dcc*u2 + self[i].drr*v2 - 2.0*self[i].drc*uv

                    if chi2 < 25.0 and chi2 >= 0.0:
                        pnorm = self[i].pnorm
                        x = -0.5*chi2

                        # 3rd order approximation to exp
                        #if x < 0.0:
                        #    ival = int64(x-0.5)
                        #else:
                        #    ival = int64(x+0.5)
                        ival = int64(x-0.5)
                        f = x - ival
                        index = ival-i0
                        
                        expval = expvals[index]
                        fexp = (6+f*(6+f*(3+f)))*0.16666666
                        expval *= fexp

                        model_val += pnorm*expval
                
                pixval = image[row,col]
                fdiff[fdiff_i] = (model_val-pixval)*ierr
                s2n_numer += pixval*model_val*ivar
                s2n_denom += model_val*model_val*ivar

            fdiff_i += 1
            u += j[0].dudcol
            v += j[0].dvdcol


    return s2n_numer, s2n_denom
'''

class GMixND(object):
    """
    Gaussian mixture in arbitrary dimensions.  A bit awkward
    in dim=1 e.g. becuase assumes means are [ndim,npars]
    """
    def __init__(self, weights=None, means=None, covars=None):

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

    def get_lnprob_scalar(self, pars_in):
        """
        (x-xmean) icovar (x-xmean)
        """
        dolog=1
        pars=numpy.asanyarray(pars_in, dtype='f8')
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
        pars=numpy.asanyarray(pars_in, dtype='f8')
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
            is_scalar=True
            n=1
        else:
            is_scalar=False

        samples=self._gmm.sample(n)

        if is_scalar:
            samples = samples[0,:]
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
                covariance_type='full')
        gmm.means_ = self.means.copy()
        gmm.covars_ = self.covars.copy()
        gmm.weights_ = self.weights.copy()

        self._gmm=gmm 



    '''
    def get_prob_scalar_old(self, pars):
        """
        Use a compile function
        """
        dolog=0
        if self.ndim==3:
            return _get_gmixnd_3d(self.log_pnorms,
                                  self.means,
                                  self.icovars,
                                  self.tmp_lnprob,
                                  pars,
                                  dolog)
        elif self.ndim==2:
            return _get_gmixnd_2d(self.log_pnorms,
                                  self.means,
                                  self.icovars,
                                  self.tmp_lnprob,
                                  pars,
                                  dolog)
        elif self.ndim==1:
            return _get_gmixnd_1d(self.log_pnorms,
                                  self.means,
                                  self.icovars,
                                  self.tmp_lnprob,
                                  pars,
                                  dolog)

        else:
            raise RuntimeError("only have fast formula for 1,2,3 dims")

    def get_lnprob_scalar_old(self, pars):
        """
        (x-xmean) icovar (x-xmean)
        """
        dolog=1
        if self.ndim==3:
            return _get_gmixnd_3d(self.log_pnorms,
                                  self.means,
                                  self.icovars,
                                  self.tmp_lnprob,
                                  pars,
                                  dolog)
        elif self.ndim==2:
            return _get_gmixnd_2d(self.log_pnorms,
                                  self.means,
                                  self.icovars,
                                  self.tmp_lnprob,
                                  pars,
                                  dolog)
        elif self.ndim==1:
            return _get_gmixnd_1d(self.log_pnorms,
                                  self.means,
                                  self.icovars,
                                  self.tmp_lnprob,
                                  pars,
                                  dolog)

        else:
            raise RuntimeError("only have fast formula for 1,2,3 dims")
        return lnp

    def get_prob_array_old(self, pars):
        """
        array input
        """
        dolog=0
        n=pars.shape[0]
        p=zeros(n)
        if self.ndim==3:
            _get_gmixnd_array_3d(self.log_pnorms,
                                 self.means,
                                 self.icovars,
                                 self.tmp_lnprob,
                                 pars,
                                 dolog,
                                 p)
        elif self.ndim==2:
            _get_gmixnd_array_2d(self.log_pnorms,
                                 self.means,
                                 self.icovars,
                                 self.tmp_lnprob,
                                 pars,
                                 dolog,
                                 p)

        elif self.ndim==1:
            _get_gmixnd_array_1d(self.log_pnorms,
                                 self.means,
                                 self.icovars,
                                 self.tmp_lnprob,
                                 pars,
                                 dolog,
                                 p)

        else:
            raise RuntimeError("only have fast formula for 1,2,3 dims")
        return p

    def get_lnprob_array_old(self, pars):
        """
        array input
        """
        dolog=1
        n=pars.shape[0]
        lnp=zeros(n)
        if self.ndim==3:
            _get_gmixnd_array_3d(self.log_pnorms,
                                 self.means,
                                 self.icovars,
                                 self.tmp_lnprob,
                                 pars,
                                 dolog,
                                 lnp)
        if self.ndim==2:
            _get_gmixnd_array_2d(self.log_pnorms,
                                 self.means,
                                 self.icovars,
                                 self.tmp_lnprob,
                                 pars,
                                 dolog,
                                 lnp)
        if self.ndim==1:
            _get_gmixnd_array_1d(self.log_pnorms,
                                 self.means,
                                 self.icovars,
                                 self.tmp_lnprob,
                                 pars,
                                 dolog,
                                 lnp)

        else:
            raise RuntimeError("only have fast formula for 2,3 dims")

        return lnp

    '''


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



_moms_flagmap={0:'ok',
               1:'maxit',
               2:'low s2n',
               4:'max shift'}

'''

@autojit
def _get_gmixnd_array_3d(log_pnorms, means, icovars, tmp_lnprob, x, dolog, output):
    """
    Fill the output array
    """
    n=output.size
    for i in xrange(n):
        output[i] = _get_gmixnd_3d(log_pnorms, means, icovars, tmp_lnprob, x[i,:], dolog)

@autojit
def _get_gmixnd_3d(log_pnorms, means, icovars, tmp_lnprob, x, dolog):
    """
    Trying to avoid underflow
    vmax = arr.max(axis=0)
    out = np.log(np.sum(np.exp(arr - vmax), axis=0))
    out += vmax
    return out
    """
    ngauss=log_pnorms.size

    p=0.0
    lnp=0.0
    lnpmax=-9.99e9

    for i in xrange(ngauss):
        logpnorm = log_pnorms[i]

        x1diff=x[0]-means[i,0]
        x2diff=x[1]-means[i,1]
        x3diff=x[2]-means[i,2]

        chi2  = x1diff*(icovars[i,0,0]*x1diff + icovars[i,0,1]*x2diff + icovars[i,0,2]*x3diff)
        chi2 += x2diff*(icovars[i,1,0]*x1diff + icovars[i,1,1]*x2diff + icovars[i,1,2]*x3diff)
        chi2 += x3diff*(icovars[i,2,0]*x1diff + icovars[i,2,1]*x2diff + icovars[i,2,2]*x3diff)

        lnp = -0.5*chi2 + logpnorm
        if lnp > lnpmax:
            lnpmax=lnp
        tmp_lnprob[i] = lnp
        
    for i in xrange(ngauss):
        p += exp(tmp_lnprob[i] - lnpmax)

    out=0.0
    if dolog==1:
        out = log(p) + lnpmax
    else:
        out=p*exp(lnpmax)

    return out

@autojit
def _get_gmixnd_array_2d(log_pnorms, means, icovars, tmp_lnprob, x, dolog, output):
    """
    Fill the output array
    """
    n=output.size
    for i in xrange(n):
        output[i] = _get_gmixnd_2d(log_pnorms, means, icovars, tmp_lnprob, x[i,:], dolog)

@autojit
def _get_gmixnd_2d(log_pnorms, means, icovars, tmp_lnprob, x, dolog):
    """
    Trying to avoid underflow
    vmax = arr.max(axis=0)
    out = np.log(np.sum(np.exp(arr - vmax), axis=0))
    out += vmax
    return out
    """
    ngauss=log_pnorms.size

    p=0.0
    lnp=0.0
    lnpmax=-9.99e9

    for i in xrange(ngauss):
        logpnorm = log_pnorms[i]

        x1diff=x[0]-means[i,0]
        x2diff=x[1]-means[i,1]

        chi2  = x1diff*(icovars[i,0,0]*x1diff + icovars[i,0,1]*x2diff)
        chi2 += x2diff*(icovars[i,1,0]*x1diff + icovars[i,1,1]*x2diff)

        lnp = -0.5*chi2 + logpnorm
        if lnp > lnpmax:
            lnpmax=lnp
        tmp_lnprob[i] = lnp
        
    for i in xrange(ngauss):
        p += exp(tmp_lnprob[i] - lnpmax)

    out=0.0
    if dolog==1:
        out = log(p) + lnpmax
    else:
        out=p*exp(lnpmax)

    return out


@autojit
def _get_gmixnd_array_1d(log_pnorms, means, icovars, tmp_lnprob, x, dolog, output):
    """
    Fill the output array
    """
    n=output.size
    for i in xrange(n):
        output[i] = _get_gmixnd_1d(log_pnorms, means, icovars, tmp_lnprob, x[i], dolog)

@autojit
def _get_gmixnd_1d(log_pnorms, means, icovars, tmp_lnprob, x, dolog):
    """
    Trying to avoid underflow
    vmax = arr.max(axis=0)
    out = np.log(np.sum(np.exp(arr - vmax), axis=0))
    out += vmax
    return out
    """
    ngauss=log_pnorms.size

    p=0.0
    lnp=0.0
    lnpmax=-9.99e9

    for i in xrange(ngauss):
        logpnorm = log_pnorms[i]

        xdiff=x-means[i]

        chi2  = xdiff*xdiff*icovars[i]

        lnp = -0.5*chi2 + logpnorm
        if lnp > lnpmax:
            lnpmax=lnp
        tmp_lnprob[i] = lnp
        
    for i in xrange(ngauss):
        p += exp(tmp_lnprob[i] - lnpmax)

    out=0.0
    if dolog==1:
        out = log(p) + lnpmax
    else:
        out=p*exp(lnpmax)

    return out
'''
