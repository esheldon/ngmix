from ._gmix import GMixRangeError
from ._gmix import GMixFatalError
#from ._gmix import GMixMaxIterEM

class GMixMaxIterEM(Exception):
    """
    EM algorithm hit max iter
    """
    def __init__(self, value):
         self.value = value
    def __str__(self):
        return repr(self.value)

class BootPSFFailure(Exception):
    """
    failure to bootstrap PSF
    """
    def __init__(self, value):
         self.value = value
    def __str__(self):
        return repr(self.value)

class BootGalFailure(Exception):
    """
    failure to bootstrap galaxy
    """
    def __init__(self, value):
         self.value = value
    def __str__(self):
        return repr(self.value)


