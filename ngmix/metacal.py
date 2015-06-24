import numpy
from ngmix import Jacobian, UnitJacobian

LANCZOS_PARS_DEFAULT={'order':5, 'conserve_dc':True, 'tol':1.0e-4}

class Metacal(object):
    """
    parameters
    ----------
    image: numpy array
        2d array representing the image
    psf_image: numpy array
        2d array representing the psf image
    jacobian: Jacobian, optional
        An ngmix.Jacobian or None.  If None, an ngmix.UnitJacobian is
        constructed
    lanczos_pars: dict, optional
        The lanczos pars.  Default is 
        {'order':5, 'conserve_dc':True, 'tol':1.0e-4}

    examples
    --------

    # observations used to calculate R
    mc=Metacal(im, psf_im)
    sh1m=ngmix.Shape(-0.01,  0.00 )
    sh1p=ngmix.Shape( 0.01,  0.00 )
    sh2m=ngmix.Shape( 0.00, -0.01 )
    sh2p=ngmix.Shape( 0.00,  0.01 )

    R_obs1m = mc.get_obs_galshear(sh11)
    R_obs1p = mc.get_obs_galshear(sh12)
    R_obs2m = mc.get_obs_galshear(sh21)
    R_obs2p = mc.get_obs_galshear(sh22)

    # you can also get an unsheared, just convolved obs
    R_obs1m, R_obs1m_unsheared = mc.get_obs_galshear(sh11, get_unsheared=True)

    # observations used to calculate Rpsf
    Rpsf_obs1m = mc.get_obs_psfshear(sh11)
    Rpsf_obs1p = mc.get_obs_psfshear(sh12)
    Rpsf_obs2m = mc.get_obs_psfshear(sh21)
    Rpsf_obs2p = mc.get_obs_psfshear(sh22)

    """
    def __init__(self, obs, lanczos_pars=None):

        self.set_data(self, obs, lanczos_pars=lanczos_pars)

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

        newpsf = self.get_target_psf(shear, 'gal_shear')
        sheared_image = self.get_target_image(newpsf, shear=shear)

        newobs = self._make_obs(sheared_image, newpsf)

        if get_unsheared:
            unsheared_image = self.get_target_image(newpsf, shear=None)
            uobs = self._make_obs(unsheared_image, newpsf)
            return newobs, uobs
        else:
            return newobs

    def get_obs_psfshear(self, shear):
        """
        This is the case where we shear the psf image, for calculating Rpsf

        parameters
        ----------
        shear: ngmix.Shape
            The shear to apply
        """
        newpsf = self.get_target_psf(shear, 'psf_shear')
        conv_image = self.get_target_image(newpsf, shear=None)

        newobs = self._make_obs(conv_image, newpsf)
        return newobs


    def get_target_psf(self, shear, type):
        """
        parameters
        ----------
        shear: ngmix.Shape
            The applied shear
        type: string
            Type of psf target.  For type='gal_shear', the psf is just dilated to
            deal with noise amplification.  For type='psf_shear' the psf is also
            sheared for calculating Rpsf
        """
        import galsim

        psf_grown_nopix = self.gs_psf_nopix.dilate(1 + 2*max([shear.g1,shear.g2]))
        psf_grown = galsim.Convolve(psf_grown_nopix,self.pixel)

        if type=='psf_shear':
            psf_grown = psf_grown.shear(g1=shear.g1, g2=shear.g2)

        im = galsim.ImageD(selg.gs_psf_image.bounds)

        newpsf = psf_grown.drawImage(image=im,
                                     scale=pixelscale,
                                     method='no_pixel')
        return newpsf

    def get_target_image(self, psf, shear=None):
        """
        get the target image, convolved with the specified psf
        and possible sheared

        parameters
        ----------
        psf: A galsim image
            psf by which to convolve
        shear: ngmix.Shape, optional
            The shear to apply
        """
        import galsim
        if shear is not None:
            shim_nopsf = self.get_sheared_image_nopsf(shear)
        else:
            shim_nopsf = self.gs_image_nopsf

        psfint = galsim.InterpolatedImage(psf, x_interpolant = self.l5int)
        im = galsim.Convolve([shim_nopsf, psfint])

        return im

    def get_sheared_image_nopsf(self, shear):
        """
        get the image sheared by the reqested amount, pre-psf and pre-pixel

        parameters
        ----------
        shear: ngmix.Shape
            The shear to apply
        """
        sheared_image = self.gs_image_nopsf.shear(g1=shear.g1, g2=shear.g2)
        return sheared_image

    def set_data(self, obs, lanczos_pars=None):

        import galsim

        if not obs.has_psf():
            raise ValueError("observation must have a psf observation set")


        self.obs=obs
        self.set_jacobian(obs.jacobian)
        self.set_lanczos(lanczos_pars=lanczos_pars)

        # these share data with the original numpy arrays
        self.gs_image = galsim.Image(obs.image,wcs=self.gs_wcs)
        self.gs_psf_image = galsim.Image(obs.psf_image,wcs=self.gs_wcs)

        # interpolated psf image
        self.gs_psf_int = galsim.InterpolatedImage(self.gs_psf_image,
                                                   x_interpolant = self.l5int)
        # psf deconvolved from pixel
        self.gs_psf_nopix = galsim.Convolve([self.gs_psf_int, self.pixel_inv])

        # like the inverse of the psf.  this can be used to deconvolve the
        # galaxy image
        self.gs_psf_inv = galsim.Deconvolve(self.gs_psf_int)

        # deconvolved galaxy image
        self.gs_image_nopsf = galsim.Convolve(self.gs_image, self.gs_psf_inv)


    def set_jacobian(self, jacobian):
        import galsim
        self.jacobian=jacobian

        self.gs_wcs = galsim.JacobianWCS(jacobian.dudrow,
                                         jacobian.dudcol,
                                         jacobian.dvdrow, 
                                         jacobian.dvdcol)

        self.pixel_scale=j.maxLinearScale()
        self.pixel = galsim.Pixel(self.pixel_scale)
        self.pixel_inv = galsim.Deconvolve(self.pixel)

    def set_lanczos(self, lanczos_pars=None):
        import galsim
        if lanczos_pars is None:
            lanczos_pars=LANCZOS_PARS_DEFAULT
        else:
            for n in ['order','conserve_dc','tol']:
                lanczos_pars[n]=lanczos_pars.get(n,LANCZOS_PARS_DEFAULT[n])

        self.lanczos_pars=lanczos_pars

        self.l5 = galsim.Lanczos(lanczos_pars['order'],
                                 lanczos_pars['conserve_dc'],
                                 lanczos_pars['tol'])
        self.l5int = galsim.InterpolantXY(l5)

    def _make_obs(self, im, psf_im):
        obs=self.obs

        psf_obs = Observation(psf_im,
                              jacobian=obs.jacobian)
        newobs=Observation(im,
                           jacobian=obs.jacobian,
                           weight=obs.weight,
                           psf=psf)
        return obs


