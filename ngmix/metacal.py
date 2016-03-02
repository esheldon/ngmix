"""
class to create manipulated images for use in metacalibration

based off reading through Eric Huffs code, but it has departed
significantly

"""
from __future__ import print_function
import numpy
from numpy import zeros, ones, newaxis, sqrt, diag, dot, linalg, array
from numpy import median, where
from .jacobian import Jacobian, UnitJacobian
from .observation import Observation, ObsList, MultiBandObsList
from .shape import Shape

try:
    import galsim
except ImportError:
    pass

LANCZOS_PARS_DEFAULT={'order':5, 'conserve_dc':True, 'tol':1.0e-4}

METACAL_TYPES = [
    '1p','1m','2p','2m',
    '1p_psf','1m_psf','2p_psf','2m_psf',
]

def get_all_metacal(obs, step=0.01, **kw):
    """
    Get all combinations of metacal images in a dict

    parameters
    ----------
    obs: Observation, ObsList, or MultiBandObsList
        The values in the dict correspond to these
    step: float
        The shear step value to use for metacal

    returns
    -------
    A dictionary with all the relevant metacaled images
        dict keys:
            1p -> ( shear, 0)
            1m -> (-shear, 0)
            2p -> ( 0, shear)
            2m -> ( 0, -shear)
        simular for 1p_psf etc.
    """

    if isinstance(obs, Observation):

        use_psf_model = kw.get('use_psf_model',False)
        if use_psf_model:
            assert 'shape' in kw,"shape keyword missing"

            shape = kw['shape']
            print("    Using psf model with shape",shape)
            m=MetacalAnalyticPSF(obs, shape, **kw)

        else:
            m=Metacal(obs, **kw)

        odict=m.get_all(step, **kw)


    elif isinstance(obs, MultiBandObsList):
        odict=_make_metacal_mb_obs_list_dict(obs, step, **kw)
    elif isinstance(obs, ObsList):
        odict=_make_metacal_obs_list_dict(obs, step, **kw)
    else:
        raise ValueError("obs must be Observation, ObsList, "
                         "or MultiBandObsList")

    return odict

class Metacal(object):
    """
    Create manipulated images for use in metacalibration

    parameters
    ----------
    image: numpy array
        2d array representing the image
    psf_image: numpy array
        2d array representing the psf image
    jacobian: Jacobian, optional
        An ngmix.Jacobian or None.  If None, an ngmix.UnitJacobian is
        constructed

    examples
    --------

    psf_obs=Observation(psf_image)
    obs=Observation(image, psf=psf_obs)

    mc=Metacal(obs)

    # observations used to calculate R

    sh1m=ngmix.Shape(-0.01,  0.00 )
    sh1p=ngmix.Shape( 0.01,  0.00 )
    sh2m=ngmix.Shape( 0.00, -0.01 )
    sh2p=ngmix.Shape( 0.00,  0.01 )

    R_obs1m = mc.get_obs_galshear(sh1m)
    R_obs1p = mc.get_obs_galshear(sh1p)
    R_obs2m = mc.get_obs_galshear(sh2m)
    R_obs2p = mc.get_obs_galshear(sh2p)

    # you can also get an unsheared, just convolved obs
    R_obs1m, R_obs1m_unsheared = mc.get_obs_galshear(sh1p, get_unsheared=True)

    # observations used to calculate Rpsf
    Rpsf_obs1m = mc.get_obs_psfshear(sh1m)
    Rpsf_obs1p = mc.get_obs_psfshear(sh1p)
    Rpsf_obs2m = mc.get_obs_psfshear(sh2m)
    Rpsf_obs2p = mc.get_obs_psfshear(sh2p)
    """

    def __init__(self, obs, **kw):

        self.obs=obs

        self._setup()
        self._set_data()

    def get_all(self, step, **kw):
        """
        Get all combinations of metacal images in a dict

        parameters
        ----------
        step: float
            The shear step value to use for metacal
        types: list
            Types to get.  Default is given in METACAL_TYPES

        returns
        -------
        A dictionary with all the relevant metacaled images
            dict keys:
                1p -> ( shear, 0)
                1m -> (-shear, 0)
                2p -> ( 0, shear)
                2m -> ( 0, -shear)
            simular for 1p_psf etc.
        """
        types=kw.get('types',METACAL_TYPES)

        shdict={}

        # galshear keys
        shdict['1m']=Shape(-step,  0.0)
        shdict['1p']=Shape( step,  0.0)

        shdict['2m']=Shape(0.0, -step)
        shdict['2p']=Shape(0.0,  step)

        # psfshear keys
        keys=list(shdict.keys())
        for key in keys:
            pkey = '%s_psf' % key
            shdict[pkey] = shdict[key].copy()

        odict={}

        for type in types:
            sh=shdict[type]

            if 'psf' in type:
                obs = self.get_obs_psfshear(sh)
            else:
                obs = self.get_obs_galshear(sh)

            odict[type] = obs

        return odict


    def get_obs_galshear(self, shear, get_unsheared=False):
        """
        This is the case where we shear the image, for calculating R

        parameters
        ----------
        shear: ngmix.Shape
            The shear to apply

        get_unsheared: bool
            Get an observation only convolved by the target psf, not
            sheared
        """

        type='gal_shear'

        newpsf_image, newpsf_obj = self.get_target_psf(shear, type)
        sheared_image = self.get_target_image(newpsf_obj, shear=shear)

        newobs = self._make_obs(sheared_image, newpsf_image)

        if get_unsheared:
            unsheared_image = self.get_target_image(newpsf_obj, shear=None)

            uobs = self._make_obs(unsheared_image, newpsf_image)
            return newobs, uobs
        else:
            return newobs

    def get_obs_dilated_only(self, shear):
        """
        Unsheared image, just with psf dilated

        parameters
        ----------
        shear: ngmix.Shape
            The shear to apply
        """

        newpsf_image, newpsf_obj = self.get_target_psf(shear, 'gal_shear')
        unsheared_image = self.get_target_image(newpsf_obj, shear=None)

        uobs = self._make_obs(unsheared_image, newpsf_image)

        return uobs

    def get_obs_psfshear(self, shear):
        """
        This is the case where we shear the psf image, for calculating Rpsf

        parameters
        ----------
        shear: ngmix.Shape
            The shear to apply
        """
        newpsf_image, newpsf_obj = self.get_target_psf(shear, 'psf_shear')
        conv_image = self.get_target_image(newpsf_obj, shear=None)

        newobs = self._make_obs(conv_image, newpsf_image)
        return newobs


    def get_target_psf(self, shear, type):
        """
        get galsim object for the dilated, possibly sheared, psf

        parameters
        ----------
        shear: ngmix.Shape
            The applied shear
        type: string
            Type of psf target.  For type='gal_shear', the psf is just dilated to
            deal with noise amplification.  For type='psf_shear' the psf is also
            sheared for calculating Rpsf

        returns
        -------
        galsim object
        """

        _check_shape(shear)

        psf_grown = self._get_dilated_psf(shear)

        if type=='psf_shear':
            # eric remarked that he thought we should shear the pixelized version
            psf_grown = psf_grown.shear(g1=shear.g1, g2=shear.g2)

        psf_grown_image = galsim.ImageD(self.psf_dims[1], self.psf_dims[0])

        # TODO not general, using just pixel scale
        psf_grown.drawImage(image=psf_grown_image,
                            scale=self.pixel_scale,
                            method='no_pixel')

        return psf_grown_image, psf_grown

    def _get_dilated_psf(self, shear):
        """
        this case for psf being an interpolated image
        """
        psf_grown_nopix = _do_dilate(self.psf_int_nopix, shear)
        psf_grown_interp = galsim.Convolve(psf_grown_nopix,self.pixel)

        return psf_grown_interp

    def get_target_image(self, psf_obj, shear=None):
        """
        get the target image, convolved with the specified psf
        and possibly sheared

        parameters
        ----------
        psf_obj: A galsim object
            psf object by which to convolve.  An interpolated image,
            or surface brightness profile
        shear: ngmix.Shape, optional
            The shear to apply

        returns
        -------
        galsim image object
        """
        if shear is not None:
            shim_nopsf = self.get_sheared_image_nopsf(shear)
        else:
            shim_nopsf = self.image_int_nopsf

        imconv = galsim.Convolve([shim_nopsf, psf_obj])

        # Draw reconvolved, sheared image to an ImageD object, and return.
        # pixel is already in the psf
        newim = galsim.ImageD(self.im_dims[1], self.im_dims[0])
        imconv.drawImage(image=newim,
                         method='no_pixel',
                         scale=self.pixel_scale)

        return newim

    def get_sheared_image_nopsf(self, shear):
        """
        get the image sheared by the reqested amount, pre-psf and pre-pixel

        parameters
        ----------
        shear: ngmix.Shape
            The shear to apply

        returns
        -------
        galsim image object
        """
        _check_shape(shear)
        # this is the interpolated, devonvolved image
        sheared_image = self.image_int_nopsf.shear(g1=shear.g1, g2=shear.g2)
        return sheared_image

    def _setup(self):

        obs=self.obs
        if not obs.has_psf():
            raise ValueError("observation must have a psf observation set")

        self._set_wcs(obs.jacobian)
        self._set_pixel()
        self._set_interp()

    def _set_data(self):
        """
        create galsim objects based on the input observation
        """

        obs=self.obs

        # these would share data with the original numpy arrays, make copies
        # to be sure they don't get modified
        #
        self.image = galsim.Image(obs.image.copy(),
                                  wcs=self.gs_wcs)
        self.psf_image = galsim.Image(obs.psf.image.copy(),
                                      wcs=self.gs_wcs)

        self.psf_dims=obs.psf.image.shape
        self.im_dims=obs.image.shape

        # interpolated psf image
        self.psf_int = galsim.InterpolatedImage(self.psf_image,
                                                x_interpolant = self.interp)
        # interpolated psf deconvolved from pixel
        self.psf_int_nopix = galsim.Convolve([self.psf_int, self.pixel_inv])

        # this can be used to deconvolve the psf from the galaxy image
        psf_int_inv = galsim.Deconvolve(self.psf_int)

        image_int = galsim.InterpolatedImage(self.image,
                                             x_interpolant=self.interp)



        # this will get passed on through
        self._set_uncorrelated_noise(image_int, obs.weight)

        # deconvolved galaxy image, psf+pixel removed
        self.image_int_nopsf = galsim.Convolve(image_int,
                                               psf_int_inv)



    def _set_uncorrelated_noise(self, im, wt):
        w=numpy.where(wt > 0)
        variance = numpy.median(1/wt[w])

        im.noise = galsim.UncorrelatedNoise(variance=variance)

    def _set_wcs(self, jacobian):
        """
        create a galsim JacobianWCS from the input ngmix.Jacobian, as
        well as pixel objects
        """

        self.jacobian=jacobian

        # TODO get conventions right and use full jacobian
        '''
        self.gs_wcs = galsim.JacobianWCS(jacobian.dudrow,
                                         jacobian.dudcol,
                                         jacobian.dvdrow, 
                                         jacobian.dvdcol)

        # TODO how this gets used does not seem general, why not use full wcs
        self.pixel_scale=self.gs_wcs.maxLinearScale()
        '''
        self.pixel_scale=self.jacobian.get_scale()
        self.gs_wcs = galsim.JacobianWCS(self.pixel_scale,
                                         0.0,
                                         0.0, 
                                         self.pixel_scale)


    def _set_pixel(self):
        """
        set the pixel based on the pixel scale, for convolutions
        """

        self.pixel = galsim.Pixel(self.pixel_scale)
        self.pixel_inv = galsim.Deconvolve(self.pixel)

    def _set_interp(self):
        """
        set the laczos interpolation configuration
        """
        self.interp = galsim.Lanczos(LANCZOS_PARS_DEFAULT['order'],
                                     LANCZOS_PARS_DEFAULT['conserve_dc'],
                                     LANCZOS_PARS_DEFAULT['tol'])

    def _make_obs(self, im, psf_im):
        """
        inputs are galsim objects
        """

        obs=self.obs

        psf_obs = Observation(psf_im.array,
                              weight=obs.psf.weight.copy(),
                              jacobian=obs.psf.jacobian.copy())

        newobs=Observation(im.array,
                           jacobian=obs.jacobian.copy(),
                           weight=obs.weight.copy(),
                           psf=psf_obs)
        return newobs

class MetacalAnalyticPSF(Metacal):
    """
    The user inputs a galsim object (e.g. galsim.Gaussian) 
    and the size of the requested postage stamp

    The psf galsim object should have the pixelization in it
    """
    def __init__(self, obs, psf_dims, **kw):
        psf_gmix = obs.get_psf_gmix()

        psf_obj = psf_gmix.make_galsim_object()
        self.psf_obj = psf_obj
        self.psf_dims = psf_dims


        super(MetacalAnalyticPSF,self).__init__(obs, **kw)

    def _get_dilated_psf(self, shear):
        """
        this case for psf being an interpolated image
        """
        return _do_dilate(self.psf_obj, shear)

    def _set_data(self):
        """
        create galsim objects based on the input observation
        """

        obs=self.obs

        # these would share data with the original numpy arrays, make copies
        # to be sure they don't get modified
        #
        self.image = galsim.Image(obs.image.copy(), wcs=self.gs_wcs)
        self.im_dims=obs.image.shape

        # interpolated galaxy image, still pixelized
        image_int = galsim.InterpolatedImage(self.image,
                                             x_interpolant=self.interp)

        # this will get passed on through
        self._set_uncorrelated_noise(image_int, obs.weight)

        # deconvolved galaxy image, psf+pixel removed
        psf_inv = galsim.Deconvolve(self.psf_obj)
        self.image_int_nopsf = galsim.Convolve(image_int, psf_inv)




def _do_dilate(obj, shear):
    """
    obj could be an interpolated image or a galsim object
    """
    g = sqrt(shear.g1**2 + shear.g2**2)
    dilation = 1.0 + 2.0*g
    return obj.dilate(dilation)


def _check_shape(shape):
    if not isinstance(shape, Shape):
        raise TypeError("shape must be of type ngmix.Shape")

def jackknife_shear(g, gpsf, R, Rpsf, chunksize=1):
    """
    get the shear metacalibration style

    parameters
    ----------
    g: array
        [N,2] shape measurements
    R: array
        [N,2,2] shape response measurements
    Rpsf: array, optional
        [N,2] psf response
    chunksize: int, optional
        chunksize for jackknifing
    """


    ntot = g.shape[0]

    nchunks = ntot/chunksize

    g_sum = g.sum(axis=0)
    R_sum = R.sum(axis=0)

    psf_corr = (Rpsf*gpsf).sum(axis=0)
    g_sum -= psf_corr

    R_sum_inv = numpy.linalg.inv(R_sum)
    shear = numpy.dot(R_sum_inv, g_sum)


    shears = zeros( (nchunks, 2) )
    for i in xrange(nchunks):

        beg = i*chunksize
        end = (i+1)*chunksize

        tgsum = g[beg:end,:].sum(axis=0)
        tR_sum = R[beg:end,:,:].sum(axis=0)

        tpsfcorr_sum = (Rpsf[beg:end,:]*gpsf[beg:end,:]).sum(axis=0)
        tgsum -= tRpsf_sum

        j_g_sum = g_sum - tgsum
        j_R_sum = R_sum - tR_sum

        j_R_inv = numpy.linalg.inv(j_R_sum)

        shears[i, :] = numpy.dot(j_R_inv, j_g_sum)

    shear_cov = zeros( (2,2) )
    fac = (nchunks-1)/float(nchunks)

    shear_cov[0,0] = fac*( ((shear[0]-shears[:,0])**2).sum() )
    shear_cov[0,1] = fac*( ((shear[0]-shears[:,0]) * (shear[1]-shears[:,1])).sum() )
    shear_cov[1,0] = shear_cov[0,1]
    shear_cov[1,1] = fac*( ((shear[1]-shears[:,1])**2).sum() )

    out={'shear':shear,
         'shear_cov':shear_cov,
         'g_sum':g_sum,
         'R_sum':R_sum,
         'gsens_sum':R_sum, # another name
         'R_sum_inv':R_sum_inv,
         'nuse':g.shape[0],
         'shears':shears}
    if Rpsf is not None:
        out['Rpsf_sum'] = Rpsf_sum
    return out



def jackknife_shear_weighted(g, gsens, weights, chunksize=1):
    """
    get the shear metacal style

    parameters
    ----------
    g: array
        [N,2] shape measurements
    gsens: array
        [N,2,2] shape sensitivity measurements
    weights: array, optional
        Weights to apply
    chunksize: int, optional
        chunksize for jackknifing
    """

    if weights is None:
        weights=ones(g.shape[0])

    ntot = g.shape[0]

    nchunks = ntot/chunksize

    wsum = weights.sum()
    wa=weights[:,newaxis]
    waa=weights[:,newaxis,newaxis]

    g_sum = (g*wa).sum(axis=0)
    gsens_sum = (gsens*waa).sum(axis=0)

    gsens_sum_inv = numpy.linalg.inv(gsens_sum)
    shear = numpy.dot(gsens_sum_inv, g_sum)

    shears = zeros( (nchunks, 2) )
    for i in xrange(nchunks):

        beg = i*chunksize
        end = (i+1)*chunksize

        wtsa = (weights[beg:end])[:,newaxis]
        wtsaa = (weights[beg:end])[:,newaxis,newaxis]

        tgsum = (g[beg:end,:]*wtsa).sum(axis=0)
        tgsens_sum = (gsens[beg:end,:,:]*wtsaa).sum(axis=0)


        j_g_sum     = g_sum     - tgsum
        j_gsens_sum = gsens_sum - tgsens_sum

        j_gsens_inv = numpy.linalg.inv(j_gsens_sum)

        shears[i, :] = numpy.dot(j_gsens_inv, j_g_sum)

    shear_cov = zeros( (2,2) )
    fac = (nchunks-1)/float(nchunks)

    shear_cov[0,0] = fac*( ((shear[0]-shears[:,0])**2).sum() )
    shear_cov[0,1] = fac*( ((shear[0]-shears[:,0]) * (shear[1]-shears[:,1])).sum() )
    shear_cov[1,0] = shear_cov[0,1]
    shear_cov[1,1] = fac*( ((shear[1]-shears[:,1])**2).sum() )

    return {'shear':shear,
            'shear_cov':shear_cov,
            'g_sum':g_sum,
            'gsens_sum':gsens_sum,
            'gsens_sum_inv':gsens_sum_inv,
            'shears':shears,
            'weights':weights,
            'wsum':wsum,
            'nuse':g.shape[0]}


def bootstrap_shear(g, gpsf, R, Rpsf, nboot, verbose=False):
    """
    get the shear metacalstyle

    The responses are bootstrapped independently of the
    shear estimators

    parameters
    ----------
    g: array
        [N,2] shape measurements
    gpsf: array
        [N,2] shape measurements
    R: array
        [NR,2,2] shape response measurements
    Rpsf: array
        [NR,2] psf response
    nboot: int
        number of bootstraps to do
    """

    ng = g.shape[0]
    nR = R.shape[0]

    # overall mean
    if verbose:
        print("    getting overall mean and naive error")
    res = get_mean_shear(g, gpsf, R, Rpsf)
    if verbose:
        print("    shear:         ",res['shear'])
        print("    shear_err:     ",res['shear_err'])

    # need workspace for ng from both data and
    # deep response data

    g_scratch    = zeros( (ng, 2) )
    gpsf_scratch = zeros( (ng, 2) )
    R_scratch    = zeros( (nR, 2, 2) )
    Rpsf_scratch = zeros( (nR, 2) )

    shears = zeros( (nboot, 2) )

    for i in xrange(nboot):
        if verbose:
            print("    boot %d/%d" % (i+1,nboot))

        g_rind = numpy.random.randint(0, ng, ng)
        R_rind = numpy.random.randint(0, nR, nR)

        g_scratch[:, :]    = g[g_rind, :]
        gpsf_scratch[:, :] = gpsf[g_rind, :]
        R_scratch[:, :, :] = R[R_rind, :, :]
        Rpsf_scratch[:, :] = Rpsf[R_rind, :]

        tres = get_mean_shear(g_scratch,
                              gpsf_scratch,
                              R_scratch,
                              Rpsf_scratch)
        shears[i,:] = tres['shear']

    shear_cov = zeros( (2,2) )

    shear = res['shear']
    shear_mean = shears.mean(axis=0)

    fac = 1.0/(nboot-1.0)
    shear_cov[0,0] = fac*( ((shear[0]-shears[:,0])**2).sum() )
    shear_cov[0,1] = fac*( ((shear[0]-shears[:,0]) * (shear[1]-shears[:,1])).sum() )
    shear_cov[1,0] = shear_cov[0,1]
    shear_cov[1,1] = fac*( ((shear[1]-shears[:,1])**2).sum() )

    res['shear_mean'] = shear_mean
    res['shear_err'] = sqrt(diag(shear_cov))
    res['shear_cov'] = shear_cov
    res['shears'] = shears
    return res

def get_mean_shear(g, gpsf, R, Rpsf):

    g_sum = g.sum(axis=0)
    g_err = g.std(axis=0)/sqrt(g.shape[0])

    g_mean = g_sum/g.shape[0]

    R_sum = R.sum(axis=0)
    R_mean = R_sum/R.shape[0]

    Rpsf_sum = Rpsf.sum(axis=0)
    Rpsf_mean = Rpsf_sum/Rpsf.shape[0]

    psf_corr_arr = gpsf.copy()
    psf_corr_arr[:,0] *= Rpsf_mean[0]
    psf_corr_arr[:,1] *= Rpsf_mean[1]

    psf_corr_sum = psf_corr_arr.sum(axis=0)
    psf_corr = psf_corr_sum/g.shape[0]

    Rinv = linalg.inv(R_mean)

    shear = dot(Rinv, g_mean - psf_corr)
    # naive error
    shear_err = dot(Rinv, g_err)

    return {
            'shear':shear,
            'shear_err':shear_err,
            'g_mean':g_mean,

            'R':R_mean,
            'Rpsf':Rpsf_mean,
            'psf_corr':psf_corr,

            'g_sum':g_sum,
            'R_sum':R_sum,
            'Rpsf_sum':Rpsf_sum,
            'psf_corr_sum':psf_corr_sum,
            'ng': g.shape[0],
            'nR': R.shape[0]
           }

def _make_metacal_mb_obs_list_dict(mb_obs_list, step, **kw):

    new_dict=None
    for obs_list in mb_obs_list:
        odict = _make_metacal_obs_list_dict(obs_list, step, **kw)

        if new_dict is None:
            new_dict=_init_mb_obs_list_dict(odict.keys())

        for key in odict:
            new_dict[key].append(odict[key])

    return new_dict

def _make_metacal_obs_list_dict(obs_list, step, **kw):
    odict = None
    first=True
    for obs in obs_list:

        #mc=Metacal(obs)
        #todict=mc.get_all(step)

        todict=get_all_metacal(obs, step=step, **kw)

        if odict is None:
            odict=_init_obs_list_dict(todict.keys())

        for key in odict:
            odict[key].append( todict[key] )

    return odict

def _init_obs_list_dict(keys):
    odict={}
    for key in keys:
        odict[key] = ObsList()
    return odict

def _init_mb_obs_list_dict(keys):
    odict={}
    for key in keys:
        odict[key] = MultiBandObsList()
    return odict


def test():
    import images
    import fitsio
    import os
    #import mchuff

    dir='./mcal-tests'
    if not os.path.exists(dir):
        os.makedirs(dir)

    step=0.01
    shears=[Shape(step,0.0), Shape(0.0,step)]

    for i,shear in enumerate(shears):
        for type in ['gal','psf']:

            obs, obs_sheared_dilated = _get_sim_obs(shear.g1,shear.g2,
                                                    r50=2.0, r50_psf=1.5)

            m=Metacal(obs)

           

            if type=='gal':
                obs_mcal = m.get_obs_galshear(shear)
            else:
                obs_mcal = m.get_obs_psfshear(shear)

            '''
            s, us, tpsf = mchuff.metaCalibrate(galsim.Image(obs.image, scale=1.0),
                                               galsim.Image(obs.psf.image,scale=1.0),
                                               g1=shear.g1, g2=shear.g2,
                                               gal_shear=type=='gal')
 
            print("psf:",numpy.abs(tpsf.array - obs_mcal.psf.image).max()/obs_mcal.psf.image.max())
            print("im:",numpy.abs(s.array - obs_mcal.image).max()/obs_mcal.image.max())
            '''

            images.compare_images(obs_sheared_dilated.image,
                                  obs_mcal.image,
                                  label1='shear/dilate',
                                  label2='metacal',
                                  width=1000,
                                  height=1000)

            if i==0 and type=='gal':
                imfile='test-image.fits'
                imfile=os.path.join(dir, imfile)
                print("writing image:",imfile)
                fitsio.write(imfile, obs.image, clobber=True)

                psffile='test-psf.fits'
                psffile=os.path.join(dir, psffile)
                print("writing psf:",psffile)
                fitsio.write(psffile, obs.psf.image, clobber=True)

            mcalfile='test-image-mcal-%sshear-%.2f-%.2f.fits' % (type,shear.g1,shear.g2)
            mcalfile=os.path.join(dir,mcalfile)
            print("writing metacaled imag:",mcalfile)
            fitsio.write(mcalfile, obs_mcal.image, clobber=True)

            mcal_psf_file='test-psf-mcal-%sshear-%.2f-%.2f.fits' % (type,shear.g1,shear.g2)
            mcal_psf_file=os.path.join(dir,mcal_psf_file)
            print("writing metacaled psf image:",mcal_psf_file)
            fitsio.write(mcal_psf_file, obs_mcal.psf.image, clobber=True)


    readme=os.path.join(dir, 'README')
    with open(readme,'w') as fobj:
        fobj.write("""metacal test files

original images:
----------------
test-image.fits
test-psf.fits

metacaled images:
-----------------

# galaxy sheared
test-image-mcal-galshear-0.01-0.00.fits
test-psf-mcal-galshear-0.01-0.00.fits

test-image-mcal-galshear-0.00-0.01.fits
test-psf-mcal-galshear-0.00-0.01.fits

# psf sheared
test-image-mcal-psfshear-0.01-0.00.fits
test-psf-mcal-psfshear-0.01-0.00.fits

test-image-mcal-psfshear-0.00-0.01.fits
test-psf-mcal-psfshear-0.00-0.01.fits
"""     )

def _get_sim_obs(s1, s2, g1=0.2, g2=0.1, r50=3.0, r50_psf=1.8):

    dims=32,32

    flux=100.0

    g1psf=0.05
    g2psf=-0.07
    fluxpsf=1.0

    s1abs = abs(s1)
    s2abs = abs(s2)
    dilate = 1. + 2.*max([s1abs,s2abs])

    r50_psf_dilated = r50_psf * dilate

    gal0 = galsim.Gaussian(flux=flux, half_light_radius=r50)
    gal0 = gal0.shear(g1=g1, g2=g2)
    gal0_sheared = gal0.shear(g1=s1, g2=s2)

    psf = galsim.Gaussian(flux=fluxpsf, half_light_radius=r50_psf)
    psf = psf.shear(g1=g1psf,g2=g2psf)

    psf_dilated = galsim.Gaussian(flux=fluxpsf, half_light_radius=r50_psf_dilated)
    psf_dilated = psf_dilated.shear(g1=g1psf,g2=g2psf)

    gal = galsim.Convolve([psf, gal0])
    gal_sheared_dilated = galsim.Convolve([psf_dilated, gal0_sheared])

    psf_image = psf.drawImage(nx=dims[1],
                              ny=dims[0],
                              scale=1.0,
                              dtype=numpy.float64)
    psf_image_dilated = psf_dilated.drawImage(nx=dims[1],
                                              ny=dims[0],
                                              scale=1.0,
                                              dtype=numpy.float64)


    image = gal.drawImage(nx=dims[1],
                          ny=dims[0],
                          scale=1.0,
                          dtype=numpy.float64)

    image_sheared_dilated = gal_sheared_dilated.drawImage(nx=dims[1],
                                                          ny=dims[0],
                                                          scale=1.0,
                                                          dtype=numpy.float64)


    psf_obs = Observation(psf_image.array)
    psf_obs_dilated = Observation(psf_image_dilated.array)

    obs = Observation(image.array, psf=psf_obs)
    obs_sheared_dilated = Observation(image_sheared_dilated.array,
                                      psf=psf_obs_dilated)

    return obs, obs_sheared_dilated
