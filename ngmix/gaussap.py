"""
synthesize gaussian aperture fluxes
"""

from __future__ import print_function
import numpy as np
from . import moments
from .gmix import GMixModel, GMixCM, GMixBDF
from .gexceptions import GMixRangeError
from .jacobian import DiagonalJacobian

DEFAULT_FLUX = -9999.0
NO_ATTEMPT = 2**0
RANGE_ERROR = 2**1


def get_gaussap_flux(pars,
                     model,
                     pixel_scale,
                     weight_fwhm,
                     dim=None,
                     psf_fwhm=None,
                     fracdev=None,
                     TdByTe=None,
                     mask=None,
                     verbose=True):
    """
    Measure synthesized gaussian weighted apertures for a ngmix
    models

    parameters
    ----------
    pars: array
        Shape [nobj, 6]
    model: string
        e.g. exp,dev,gauss,cm
    pixel_scale: float
        Pixel scale for the images.
    weight_fwhm: float
        FWHM of the weight function in the same units as the
        pixel scale.
    dims: size of the image to draw
        If not set, it is taken to be 2*5*sigma of the weight function
    psf_fwhm: float, optional
        Size of the small psf with which to convolve the profile. Default
        is the pixel scale.  This psf helps avoid issues with resolution
    fracdev: array
        Send for model 'cm'
    TdByTe: array
        Send for model 'cm'
    verbose: bool
        If True, print otu progress
    """

    fracdev, TdByTe, pars, mask, gapmeas = _prepare(
        pars,
        model,
        pixel_scale,
        weight_fwhm,
        dim=dim,
        psf_fwhm=psf_fwhm,
        fracdev=fracdev,
        TdByTe=TdByTe,
        mask=mask,
    )

    nband = _get_nband(pars, model)

    flags = np.zeros((pars.shape[0], nband), dtype='i4')
    gap_flux = np.zeros( (pars.shape[0], nband) )
    gap_flux[:, :] = DEFAULT_FLUX

    nobj = pars.shape[0]
    for i in range(nobj):

        if verbose and ((i+1) % 1000) == 0:
            print("%d/%d" % (i+1, nobj))

        if not mask[i]:
            flags[i] = NO_ATTEMPT
            continue

        for band in range(nband):
            tflux, tflags = _do_gap(fracdev, TdByTe, pars, gapmeas, i, band)
            gap_flux[i, band] = tflux
            flags[i, band] = tflags

    return gap_flux, flags


def _do_gap(fracdev, TdByTe, pars, gapmeas, i, band):

    flux = DEFAULT_FLUX
    flags = RANGE_ERROR

    try:

        tpars = _get_band_pars(pars, gapmeas.model, i, band)
        """
        tpars = np.zeros(6)
        tpars[0:5] = pars[i, 0:5]
        tpars[-1] = pars[i, 5+band]

        tpars[4] = tpars[4].clip(min=0.0001)
        """

        if gapmeas.model == 'cm':
            flux = gapmeas.get_aper_flux(
                fracdev[i],
                TdByTe[i],
                tpars,
            )
        else:
            flux = gapmeas.get_aper_flux(tpars)

        flags = 0
    except GMixRangeError as err:
        print(str(err))

    return flux, flags


def _prepare(pars,
             model,
             pixel_scale,
             weight_fwhm,
             dim=None,
             psf_fwhm=None,
             fracdev=None,
             TdByTe=None,
             mask=None):

    pars = np.array(pars, dtype='f8', ndmin=2, copy=False)

    if mask is not None:
        mask = np.array(mask, dtype=np.bool_, ndmin=1, copy=False)
        assert mask.shape[0] == pars.shape[0], \
            'mask and pars must be same length'
    else:
        mask = np.ones(pars.shape[0], dtype=np.bool_)

    if len(pars.shape) == 1:
        oldpars = pars
        pars = np.zeros((1, pars.shape[0]), dtype='f8')
        pars[0, :] = oldpars

    if model == 'cm':

        fracdev = np.array(fracdev, dtype='f8', ndmin=1, copy=False)
        TdByTe = np.array(TdByTe, dtype='f8', ndmin=1, copy=False)
        assert fracdev.size == pars.shape[0], 'fracdev/pars must be same size'
        assert TdByTe.size == pars.shape[0], 'TdByTe/pars must be same length'

        gapmeas = GaussAperCM(
            pixel_scale=pixel_scale,
            weight_fwhm=weight_fwhm,
            psf_fwhm=psf_fwhm,
            dim=dim,
        )

    elif model == 'bdf':

        gapmeas = GaussAperBDF(
            pixel_scale=pixel_scale,
            weight_fwhm=weight_fwhm,
            psf_fwhm=psf_fwhm,
            dim=dim,
        )

    else:
        gapmeas = GaussAper(
            pixel_scale=pixel_scale,
            weight_fwhm=weight_fwhm,
            model=model,
            psf_fwhm=psf_fwhm,
            dim=dim,
        )

    return fracdev, TdByTe, pars, mask, gapmeas


class GaussAper(object):
    def __init__(self,
                 pixel_scale,
                 weight_fwhm,
                 model,
                 psf_fwhm=None,
                 dim=None):
        """
        measure synthesized gaussian weighted apertures for simple ngmix
        models and bdf

        Parameters
        ----------
        pixel_scale: float
            Pixel scale for images.
        weight_fwhm: float
            FWHM of the weight function in the same units as the
            pixel scale.
        model: string
            The ngmix simple model or bdf
        psf_fwhm: float, optional
            Size of the small psf with which to convolve the profile. Default
            is the sqrt92*pixel_scale.  This psf helps avoid issues with
            resolution
        dim: int, optional
            Dimension of the images to simulate. If not sent, 2*5*sigma of the
            weight is used
        """

        self.pixel_scale = pixel_scale
        self.model = model
        if psf_fwhm is None:
            psf_fwhm = np.sqrt(2)*pixel_scale

        if dim is None:
            sigma = moments.fwhm_to_sigma(weight_fwhm)
            dim = 2*5*sigma

        self.dims = [dim]*2

        self._set_jacobian()

        self._set_weight(weight_fwhm)
        self._set_weight_image()
        self._set_psf(psf_fwhm)
        self._set_expected_npars()

    def get_aper_flux(self, pars):
        """
        get the aperture flux for the specified parameters

        Parameters
        ----------
        pars: 6-element sequence
            The parameters [cen1,cen2,g1,g2,T,flux]
            The center offset is not used; the model will always be centered in
            the image.

        Returns
        -------
        fluxe: float
            gaussian weighted fluxe in the aperture
        """

        gm = self._get_object_gmix(pars)
        im = gm.make_image(self.dims, jacobian=self.jacobian)

        return self._do_weighted_flux(im)

    def _do_weighted_flux(self, im):
        """
        do the actual flux calcuation over the images
        """
        weighted_im = self.weight_image*im
        flux = weighted_im.sum()*self.pixel_scale**2
        return flux

    def _get_object_gmix(self, pars):
        """
        get a gmix for the model
        """
        gm0 = GMixModel(pars, self.model)
        gm = gm0.convolve(self.psf)
        return gm

    def _set_jacobian(self):
        """
        set the jacobian based on the pixel scale and dims
        """

        cen = (np.array(self.dims)-1.0)/2.0

        self.jacobian = DiagonalJacobian(
            row=cen[0],
            col=cen[1],
            scale=self.pixel_scale,
        )

    def _set_weight_image(self):
        """
        set the weight image
        """
        wtim = self.weight.make_image(
            self.dims,
            jacobian=self.jacobian,
        )
        wtim *= 1.0/wtim.max()
        self.weight_image = wtim

    def _set_psf(self, psf_fwhm):
        """
        set the psf gmix
        """
        self.psf_fwhm = psf_fwhm
        self.psf = self._get_gauss_model(psf_fwhm)

    def _set_weight(self, weight_fwhm):
        """
        set the weight gmix
        """
        self.weight_fwhm = weight_fwhm
        weight = self._get_gauss_model(weight_fwhm)
        self.weight = weight

    def _get_gauss_model(self, fwhm):
        """
        get a gaussian model for the input fwhm
        """
        sigma = moments.fwhm_to_sigma(fwhm)
        T = 2*sigma**2

        pars = [
            0.0,
            0.0,
            0.0,
            0.0,
            T,
            1.0,
        ]
        return GMixModel(pars, "gauss")

    def _set_expected_npars(self):
        """
        set the expected number of pars
        """
        self._expected_npars = 6

    def _check_pars(self, pars):
        """
        check the parameters have the right size
        """
        if len(pars) != self._expected_npars:
            m = 'expected pars to be a %d-element sequence for model %s'
            m = m % (self._expected_npars, self.model)
            raise ValueError(m)


class GaussAperBDF(GaussAper):
    def __init__(self, pixel_scale, weight_fwhm, dim=None, psf_fwhm=None):
        """
        measure synthesized gaussian weighted apertures for a the bdf model

        Parameters
        ----------
        pixel_scale: float
            Pixel scale for images.
        weight_fwhm: float
            FWHM of the weight function in the same units as the
            pixel scale.
        psf_fwhm: float, optional
            Size of the small psf with which to convolve the profile. Default
            is the sqrt92*pixel_scale.  This psf helps avoid issues with
            resolution
        dim: int, optional
            Dimension of the images to simulate. If not sent, 2*5*sigma of the
            weight is used
        """

        super(GaussAperBDF, self).__init__(
            pixel_scale,
            weight_fwhm,
            'bdf',
            dim=dim,
            psf_fwhm=psf_fwhm,
        )

    def _do_weighted_flux(self, im):
        """
        do the actual flux calcuation over the images
        """
        weighted_im = self.weight_image*im
        flux = weighted_im.sum()*self.pixel_scale**2
        return flux

    def _get_object_gmix(self, pars):
        """
        get a gmix for the model
        """
        gm0 = GMixBDF(pars)
        gm = gm0.convolve(self.psf)
        return gm

    def _set_expected_npars(self):
        """
        set the expected number of pars
        """
        self._expected_npars = 7


class GaussAperCM(GaussAper):
    def __init__(self, pixel_scale, weight_fwhm, dim=None, psf_fwhm=None):
        """
        measure synthesized gaussian weighted apertures for the CM model

        Parameters
        ----------
        pixel_scale: float
            Pixel scale for images.
        weight_fwhm: float
            FWHM of the weight function in the same units as the
            pixel scale.
        psf_fwhm: float, optional
            Size of the small psf with which to convolve the profile. Default
            is the sqrt92*pixel_scale.  This psf helps avoid issues with
            resolution
        dim: int, optional
            Dimension of the images to simulate. If not sent, 2*5*sigma of the
            weight is used
        """

        super(GaussAperCM, self).__init__(
            pixel_scale,
            weight_fwhm,
            'cm',
            dim=dim,
            psf_fwhm=psf_fwhm,
        )

    def get_aper_flux(self, fracdev, TdByTe, pars):
        """
        get the aperture flux for the specified parameters

        parameters
        ----------
        fracdev: float
            Fraction of light in the dev component
        TdByTe: float
            Ratio of bulge T to disk T Tdev/Texp
        pars: 6-element sequence
            The parameters [cen1,cen2,g1,g2,T,flux].  The center is not used,
            the model will always be centered in the image.

        returns
        -------
        fluxes: array
            gaussian weighted fluxes in the aperture with length
            nband
        """

        self._check_pars(pars)

        gm = self._get_object_gmix(fracdev, TdByTe, pars)
        im = gm.make_image(self.dims, jacobian=self.jacobian)
        return self._do_weighted_flux(im)

    def _get_object_gmix(self, fracdev, TdByTe, pars):
        """
        get the cm gmix for this object
        """

        gm0 = GMixCM(
            fracdev,
            TdByTe,
            pars,
        )

        gm = gm0.convolve(self.psf)
        return gm


def _get_band_pars(pars, model, index, band):

    npars = _get_band_npars(model)

    flux_start = npars-1

    tpars = np.zeros(npars)
    tpars[0:npars-1] = pars[index, 0:npars-1]
    tpars[-1] = pars[index, flux_start+band]

    tpars[4] = tpars[4].clip(min=0.0001)

    return tpars

def _get_nband(pars, model):
    if model=='bdf':
        nband = len(pars[0])-7+1
    else:
        nband = len(pars[0])-6+1

    return nband


def _get_band_npars(model):
    if model=='bdf':
        nband = 7
    else:
        nband = 6

    return nband
