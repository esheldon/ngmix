from ._gmix import GMixRangeError
from ._gmix import GMixFatalError
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

'''
class GMixRangeError(Exception):
    """
    Error for ranges, e.g. ellipticity out of range
    
    We usually want to recover gracefully from this error
    """
    def __init__(self, value):
         self.value = value
    def __str__(self):
        return repr(self.value)

class GMixFatalError(Exception):
    """
    Represents an irrecoverable error
    """
    def __init__(self, value):
         self.value = value
    def __str__(self):
        return repr(self.value)
'''

