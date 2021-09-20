from ngmix.ksigmamom import _PrePSFMom


class PrePSFGaussMom(_PrePSFMom):
    """Measure pre-PSF weighted real-space moments w/ a Gaussian kernel.

    This fitter differs from `GaussMom` in that it deconvolves the PSF first.

    If the fwhm of the weight/kernel function is of similar size to the PSF or
    smaller, then the object properties returned by this fitter will be very noisy.

    Parameters
    ----------
    fwhm : float
        This parameter is the real-space FWHM of the kernel. The units are
        whatever the Jacobian on the obs converts pixels units to. This is typically
        arcseconds.
    pad_factor : int, optional
        The factor by which to pad the FFTs used for the image. Default is 4.
    """
    def __init__(self, fwhm, pad_factor=4):
        super().__init__(fwhm, kernel='gauss', pad_factor=pad_factor)
