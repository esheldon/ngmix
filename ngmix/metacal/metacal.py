"""
code to create sheared images for use in metacalibration

Originally based off reading through Eric Huffs code; it has departed
significantly.
"""
import copy
import logging
from functools import lru_cache

import numpy as np
from ..gexceptions import GMixRangeError, BootPSFFailure
from ..shape import Shape
from .. import moments
from .defaults import DEFAULT_STEP, METACAL_TYPES, METACAL_MINIMAL_TYPES


__all__ = [
    'MetacalDilatePSF', 'MetacalGaussPSF', 'MetacalFitGaussPSF', 'MetacalAnalyticPSF',
]

logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def _cached_galsim_stuff(img, wcs_repr, xinterp):
    import galsim
    image = galsim.Image(np.array(img), wcs=eval(wcs_repr))
    image_int = galsim.InterpolatedImage(image, x_interpolant=xinterp)
    return image, image_int


class MetacalDilatePSF(object):
    """
    Create manipulated images for use in metacalibration

    Parameters
    ----------
    obs: ngmix.Observation
        The observation must have a psf observation set, holding
        the psf image

    examples
    --------

    mc = MetacalDilatePSF(obs)

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

    def __init__(self, obs):

        self.obs = obs

        if not obs.has_psf():
            raise ValueError("observation must have a psf observation set")

        self._set_pixel()
        self._set_interp()
        self._set_data()
        self._psf_cache = {}

    def get_all(self, step=DEFAULT_STEP, types=None):
        """
        Get metacal images in a dict for the requested image types

        parameters
        ----------
        step: float
            The shear step value to use for metacal. Default 0.01
        types: list
            Types to get.  Default is the full possible set listed in
            METACAL_TYPES = ['noshear','1p','1m','2p','2m',
                             '1p_psf','1m_psf','2p_psf','2m_psf']

            If you are not using a round PSF, you should also request the
            sheared psf terms to make psf leakage corrections
            ['1p_psf','1m_psf','2p_psf','2m_psf'].  You can get this
            full set in METACAL_TYPES

        returns
        -------
        A dictionary with all the relevant metacaled images, e.g.
            with dict keys:
                noshear -> (0, 0)
                1p -> ( shear, 0)
                1m -> (-shear, 0)
                2p -> ( 0,  shear)
                2m -> ( 0, -shear)
        """

        if types is None:
            types = copy.deepcopy(METACAL_TYPES)
        else:
            for t in types:
                assert t in METACAL_TYPES, 'bad metacal type: %s' % t

        # we add 1p here if we want noshear since we get both of those
        # at once below

        if 'noshear' in types and '1p' not in types:
            types.append('1p')

        shdict = {}

        # galshear keys
        shdict['1m'] = Shape(-step, 0.0)
        shdict['1p'] = Shape(+step, 0.0)

        shdict['2m'] = Shape(0.0, -step)
        shdict['2p'] = Shape(0.0, +step)

        # psfshear keys
        keys = list(shdict.keys())
        for key in keys:
            pkey = '%s_psf' % key
            shdict[pkey] = shdict[key].copy()

        odict = {}

        for type in types:
            if type == 'noshear':
                # we get noshear with 1p
                continue

            sh = shdict[type]

            if 'psf' in type:
                obs = self.get_obs_psfshear(sh)
            else:
                if type == '1p':
                    # add in noshear from this one
                    obs, obs_noshear = self.get_obs_galshear(
                        sh,
                        get_unsheared=True
                    )
                    odict['noshear'] = obs_noshear
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

        type = 'gal_shear'

        newpsf_image, newpsf_obj = self.get_target_psf(shear, type)

        sheared_image = self.get_target_image(newpsf_obj, shear=shear)

        newobs = self._make_obs(sheared_image, newpsf_image)

        # this is the pixel-convolved psf object, used to draw the
        # psf image
        newobs.psf.galsim_obj = newpsf_obj

        if get_unsheared:
            unsheared_image = self.get_target_image(newpsf_obj, shear=None)

            uobs = self._make_obs(unsheared_image, newpsf_image)
            uobs.psf.galsim_obj = newpsf_obj

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
        newpsf_image, newpsf_obj = self.get_target_psf(shear, 'psf_shear')
        conv_image = self.get_target_image(newpsf_obj, shear=None)

        newobs = self._make_obs(conv_image, newpsf_image)
        return newobs

    def get_target_psf(self, shear, type):
        """
        get image and galsim object for the dilated, possibly sheared, psf

        parameters
        ----------
        shear: ngmix.Shape
            The applied shear
        type: string
            Type of psf target.  For type='gal_shear', the psf is just dilated
            to deal with noise amplification.  For type='psf_shear' the psf is
            also sheared for calculating Rpsf

        returns
        -------
        image, galsim object
        """

        _check_shape(shear)

        if type == 'psf_shear':
            doshear = True
        else:
            doshear = False

        key = self._get_psf_key(shear, doshear)
        if key not in self._psf_cache:
            psf_grown = self._get_dilated_psf(shear, doshear=doshear)

            # this should carry over the wcs
            psf_grown_image = self.psf_image.copy()

            try:
                psf_grown.drawImage(
                    image=psf_grown_image,
                    method='no_pixel',  # pixel is already in psf
                )
            except RuntimeError as err:
                # argh, galsim uses generic exceptions
                raise GMixRangeError("galsim error: '%s'" % str(err))

            self._psf_cache[key] = (psf_grown_image, psf_grown)

        psf_grown_image, psf_grown = self._psf_cache[key]
        return psf_grown_image.copy(), psf_grown

    def _get_dilated_psf(self, shear, doshear=False):
        """
        dilate the psf by the input shear and reconvolve by the pixel.  See
        _do_dilate for the algorithm

        If doshear, also shear it
        """
        import galsim

        psf_grown_nopix = self._do_dilate(self.psf_int_nopix, shear, doshear=doshear)

        if doshear:
            psf_grown_nopix = psf_grown_nopix.shear(g1=shear.g1, g2=shear.g2)

        psf_grown = galsim.Convolve(psf_grown_nopix, self.pixel)
        return psf_grown

    def _do_dilate(self, psf, shear, doshear=False):
        key = self._get_psf_key(shear, doshear)
        if key not in self._psf_cache:
            self._psf_cache[key] = _do_dilate(psf, shear)

        return self._psf_cache[key]

    def _get_psf_key(self, shear, doshear):
        """
        need full g1 and g2 in key to support psf shearing
        """
        return '%s-%s-%s' % (doshear, shear.g1, shear.g2)

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

        imconv = self._get_target_gal_obj(psf_obj, shear=shear)

        ny, nx = self.image.array.shape

        try:
            newim = imconv.drawImage(
                nx=nx,
                ny=ny,
                wcs=self.image.wcs,
                dtype=np.float64,
                method='no_pixel',
            )
        except RuntimeError as err:
            # argh, galsim uses generic exceptions
            raise GMixRangeError("galsim error: '%s'" % str(err))

        return newim

    def _get_target_gal_obj(self, psf_obj, shear=None):
        import galsim

        if shear is not None:
            shim_nopsf = self.get_sheared_image_nopsf(shear)
        else:
            shim_nopsf = self.image_int_nopsf

        imconv = galsim.Convolve([shim_nopsf, psf_obj])

        return imconv

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

    def _set_data(self):
        """
        create galsim objects based on the input observation
        """
        import galsim

        obs = self.obs

        # these would share data with the original numpy arrays, make copies
        # to be sure they don't get modified
        #
        image, image_int = _cached_galsim_stuff(
            tuple(tuple(ii) for ii in obs.image.copy()),
            repr(self.get_wcs()),
            self.interp,
        )
        self.image = image
        self.image_int = image_int

        psf_image, psf_int = _cached_galsim_stuff(
            tuple(tuple(ii) for ii in obs.psf.image.copy()),
            repr(self.get_psf_wcs()),
            self.interp,
        )
        self.psf_image = psf_image

        # this can be used to deconvolve the psf from the galaxy image
        psf_int_inv = galsim.Deconvolve(psf_int)

        # deconvolved galaxy image, psf+pixel removed
        self.image_int_nopsf = galsim.Convolve(self.image_int,
                                               psf_int_inv)

        # interpolated psf deconvolved from pixel.  This is what
        # we dilate, shear, etc and reconvolve the image by
        self.psf_int_nopix = galsim.Convolve([psf_int, self.pixel_inv])

    def get_wcs(self):
        """
        get a galsim wcs from the input jacobian
        """
        return self.obs.jacobian.get_galsim_wcs()

    def get_psf_wcs(self):
        """
        get a galsim wcs from the input jacobian
        """
        return self.obs.psf.jacobian.get_galsim_wcs()

    def _set_pixel(self):
        """
        set the pixel based on the pixel scale, for convolutions

        Thanks to M. Jarvis for the suggestion to use toWorld
        to get the proper pixel
        """
        import galsim

        wcs = self.get_wcs()
        self.pixel = wcs.toWorld(galsim.Pixel(scale=1))
        self.pixel_inv = galsim.Deconvolve(self.pixel)

    def _set_interp(self):
        """
        set the laczos interpolation configuration
        """
        self.interp = 'lanczos15'

    def _make_psf_obs(self, psf_im):

        new_psf_obs = self.obs.psf.copy()
        new_psf_obs.image = psf_im.array
        return new_psf_obs

    def _make_obs(self, im, psf_im):
        """
        b
        Make new Observation objects with the new image and psf.

        parameters
        ----------
        im: Galsim Image
        psf_im: Galsim Image

        returns
        -------
        A new Observation
        """

        newobs = self.obs.copy()
        newobs.image = im.array
        newobs.psf = self._make_psf_obs(psf_im)
        return newobs


class MetacalGaussPSF(MetacalDilatePSF):
    """
    Create manipulated images for use in metacalibration.  The reconvolution
    kernel is a gaussian generated based on the input psf

    Parameters
    ----------
    obs: ngmix.Observation
        The observation must have a psf observation set, holding
        the psf image
    rng: numpy.random.RandomState, optional
        Optional random number generator for adding a small amount of noise to
        the gaussian psf image

    examples
    --------

    mc = MetacalGaussPFF(obs)

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
    """

    def __init__(self, obs, rng=None):

        super().__init__(obs=obs)
        self.rng = rng
        self._setup_psf_noise()

    def get_all(self, step=DEFAULT_STEP, types=None):
        """
        Get metacal images in a dict for the requested image types

        parameters
        ----------
        step: float
            The shear step value to use for metacal. Default 0.01
        types: list
            Types to get.  Default is the full possible set listed in
            METACAL_MINIMAL_TYPES = ['noshear','1p','1m','2p','2m']

        returns
        -------
        A dictionary with all the relevant metacaled images, e.g.
            with dict keys:
                noshear -> (0, 0)
                1p -> ( shear, 0)
                1m -> (-shear, 0)
                2p -> ( 0,  shear)
                2m -> ( 0, -shear)
        """

        if types is None:
            types = copy.deepcopy(METACAL_MINIMAL_TYPES)
        else:
            for t in types:
                assert t in METACAL_MINIMAL_TYPES, 'bad metacal type: %s' % t

        return super().get_all(step=step, types=types)

    def _setup_psf_noise(self):
        pim = self.obs.psf.image
        self.psf_flux = pim.sum()

        self.psf_noise = pim.max()/50000.0

        if self.rng is not None:
            self.psf_noise_image = self.rng.normal(
                size=pim.shape,
                scale=self.psf_noise,
            )
        else:
            self.psf_noise_image = None

        self.psf_weight = pim * 0 + 1.0/self.psf_noise**2

    def _get_dilated_psf(self, shear, doshear=False):
        """
        dilate the psf by the input shear and reconvolve by the pixel.  See
        _do_dilate for the algorithm

        """
        import galsim

        assert doshear is False, 'no shearing gauss psf'

        gauss_psf = _get_gauss_target_psf(
            self.psf_int_nopix, flux=self.psf_flux,
        )
        psf_grown_nopix = _do_dilate(gauss_psf, shear)
        psf_grown = galsim.Convolve(psf_grown_nopix, self.pixel)
        return psf_grown

    def _make_psf_obs(self, gsim):

        psf_im = gsim.array.copy()

        if self.psf_noise_image is not None:
            psf_im += self.psf_noise_image

        new_psf_obs = self.obs.psf.copy()
        with new_psf_obs.writeable():
            new_psf_obs.image[:, :] = psf_im
            new_psf_obs.weight[:, :] = self.psf_weight

            # Reset the center on the jacobian.
            # We drew the model psf as the exact center
            cen = (np.array(psf_im.shape) - 1.0)/2.0
            new_psf_obs.jacobian.set_cen(row=cen[0], col=cen[1])

        return new_psf_obs


class MetacalFitGaussPSF(MetacalGaussPSF):
    """
    Create manipulated images for use in metacalibration.  The reconvolution
    kernel is a gaussian generated based a fit to the input psf

    Parameters
    ----------
    obs: ngmix.Observation
        Observation on which to run metacal
    rng: numpy.random.RandomState
        Random number generator.  Used to generate guesses for the fit, and for
        adding a small amount of noise to the psf image.

    examples
    --------

    rng = np.random.RandomState(seed)
    mc = MetacalFitGaussPSF(obs, rng)

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
    """
    def __init__(self, obs, rng=None):
        super().__init__(obs=obs, rng=rng)
        if rng is None:
            raise ValueError('send an rng to MetacalFitGaussPSF')
        self._do_psf_fit()

    def _get_dilated_psf(self, shear, doshear=False):
        """
        dilate the psf by the input shear and reconvolve by the pixel.  See
        _do_dilate for the algorithm
        """

        assert doshear is False, 'no shearing fitgauss psf'

        psf_grown = _do_dilate(self.gauss_psf, shear)

        # we don't convolve by the pixel, its already in there
        return psf_grown

    def _do_psf_fit(self):
        """
        do the gaussian fit.

        try the following in order
            - adaptive moments
            - maximim likelihood
            - see if there is already a gmix object


        if the above all fail, rase BootPSFFailure
        """
        import galsim

        from ..admom import AdmomFitter
        from ..guessers import GMixPSFGuesser, SimplePSFGuesser
        from ..runners import run_psf_fitter
        from ..fitting import Fitter

        psfobs = self.obs.psf

        ntry = 4
        guesser = GMixPSFGuesser(rng=self.rng, ngauss=1)

        # try adaptive moments first
        fitter = AdmomFitter(rng=self.rng)

        res = run_psf_fitter(obs=psfobs, fitter=fitter, guesser=guesser, ntry=ntry)

        if res['flags'] == 0:
            e1, e2 = res['e']
            T = res['T']
        else:
            # try maximum likelihood

            lm_pars = {
                'maxfev': 2000,
                'ftol': 1.0e-05,
                'xtol': 1.0e-05,
            }

            fitter = Fitter(model='gauss', fit_pars=lm_pars)
            guesser = SimplePSFGuesser(rng=self.rng)

            res = run_psf_fitter(
                obs=psfobs, fitter=fitter, guesser=guesser, ntry=ntry,
                set_result=False,
            )

            if res['flags'] == 0:
                psf_gmix = res.get_gmix()
            else:

                # see if there was already a gmix that we might use instead
                if psfobs.has_gmix() and len(psfobs.gmix) == 1:
                    psf_gmix = psfobs.gmix.copy()
                else:
                    # ok, just raise and exception
                    raise BootPSFFailure('failed to fit psf '
                                         'for MetacalFitGaussPSF')
            try:
                e1, e2, T = psf_gmix.get_e1e2T()
            except GMixRangeError as err:
                logger.info('%s', err)
                raise BootPSFFailure(
                    'could not get e1,e2 from psf fit for MetacalFitGaussPSF'
                )

        dilation = _get_ellip_dilation(e1, e2, T)
        T_dilated = T*dilation
        sigma = np.sqrt(T_dilated/2.0)

        self.gauss_psf = galsim.Gaussian(
            sigma=sigma,
            flux=self.psf_flux,
        )


class MetacalAnalyticPSF(MetacalGaussPSF):
    """
    Create manipulated images for use in metacalibration.  The reconvolution
    kernel is set to the input galsim object

    Parameters
    ----------
    obs: ngmix.Observation
        The observation must have a psf observation set, holding
        the psf image
    psf: galsim GSObjec
        The psf used for reconvolution
    rng: numpy.random.RandomState
        Optional random number generator for adding a small amount of noise to
        the gaussian psf image

    examples
    --------

    mc = MetacalGaussPFF(obs)

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
    """
    def __init__(self, obs, psf, rng=None):
        import galsim
        super().__init__(obs=obs, rng=rng)

        assert isinstance(psf, galsim.GSObject)
        self.psf_obj = psf

    def _get_dilated_psf(self, shear, doshear=False):
        """
        For this version we never pixelize the input
        analytic model
        """

        assert doshear is False, 'no shearing analytic psf'

        psf_grown = _do_dilate(self.psf_obj, shear)
        return psf_grown


def _get_ellip_dilation(e1, e2, T):
    """
    when making a new image after shearing, we need to dilate the PSF to hide
    modes that get exposed
    """
    irr, irc, icc = moments.e2mom(e1, e2, T)

    mat = np.zeros((2, 2))
    mat[0, 0] = irr
    mat[0, 1] = irc
    mat[1, 0] = irc
    mat[1, 1] = icc

    eigs = np.linalg.eigvals(mat)

    dilation = eigs.max()/(T/2.)
    dilation = np.sqrt(dilation)

    dilation = 1.0 + 2*(dilation-1.0)

    if dilation > 1.1:
        dilation = 1.1

    return dilation


def _do_dilate(obj, shear):
    """
    Dilate the input Galsim image object according to
    the input shear

    dilation = 1.0 + 2.0*|g|

    parameters
    ----------
    obj: Galsim Image or object
        The object to dilate
    shear: ngmix.Shape
        The shape to use for dilation
    """
    g = np.sqrt(shear.g1**2 + shear.g2**2)
    dilation = 1.0 + 2.0*g
    return obj.dilate(dilation)


def _check_shape(shape):
    """
    ensure the input is an instantiation of ngmix.Shape
    """
    if not isinstance(shape, Shape):
        raise TypeError("shape must be of type ngmix.Shape")


def _get_gauss_target_psf(psf, flux):
    """
    taken from galsim/tests/test_metacal.py


    assumes the psf is centered
    """
    import galsim

    dk = psf.stepk/4.0

    small_kval = 1.e-2    # Find the k where the given psf hits this kvalue
    smaller_kval = 3.e-3  # Target PSF will have this kvalue at the same k

    kim = psf.drawKImage(scale=dk)
    karr_r = kim.real.array
    # Find the smallest r where the kval < small_kval
    nk = karr_r.shape[0]
    kx, ky = np.meshgrid(np.arange(-nk/2, nk/2), np.arange(-nk/2, nk/2))
    ksq = (kx**2 + ky**2) * dk**2
    ksq_max = np.min(ksq[karr_r < small_kval * psf.flux])

    # We take our target PSF to be the (round) Gaussian that is even smaller at
    # this ksq
    # exp(-0.5 * ksq_max * sigma_sq) = smaller_kval
    sigma_sq = -2. * np.log(smaller_kval) / ksq_max

    return galsim.Gaussian(sigma=np.sqrt(sigma_sq), flux=flux)
