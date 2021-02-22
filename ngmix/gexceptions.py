class NGmixBaseException(Exception):
    """Base exception class for ngmix"""
    def __init__(self, value):
        super(NGmixBaseException, self).__init__(value)
        self.value = value

    def __str__(self):
        return repr(self.value)


class GMixRangeError(NGmixBaseException):
    """
    Some number was out of range.
    """
    pass


class GMixFatalError(NGmixBaseException):
    """
    A fatal error in the Gaussian mixtures.
    """
    pass


class GMixMaxIterEM(NGmixBaseException):
    """
    EM algorithm hit max iter.
    """
    pass


class PSFFluxFailure(NGmixBaseException):
    """
    Failure to fit PSF fluxes.  This usually only fails for all zero weight
    images, so this means we can't proceed with other types of fits or
    measurements either
    """
    pass


class BootPSFFailure(NGmixBaseException):
    """
    Failure to bootstrap PSF
    """
    pass


class BootGalFailure(NGmixBaseException):
    """
    Failure to bootstrap galaxy
    """
    pass
