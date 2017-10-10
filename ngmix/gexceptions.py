class GMixRangeError(Exception):
    """
    Some number was out of range
    """
    def __init__(self, value):
         self.value = value
    def __str__(self):
        return repr(self.value)

class GMixFatalError(Exception):
    """
    Some number was out of range
    """
    def __init__(self, value):
         self.value = value
    def __str__(self):
        return repr(self.value)



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

class BootPSampleFailure(Exception):
    """
    failure to bootstrap galaxy
    """
    def __init__(self, value):
         self.value = value
    def __str__(self):
        return repr(self.value)


