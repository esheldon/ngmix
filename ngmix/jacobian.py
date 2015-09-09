import numpy
from numpy import zeros, sqrt

import copy

'''
_jacobian=struct([('row0',float64),
                  ('col0',float64),
                  ('dudrow',float64),
                  ('dudcol',float64),
                  ('dvdrow',float64),
                  ('dvdcol',float64),
                  ('det',float64),
                  ('sdet',float64)],packed=True)
_jacobian_dtype=_jacobian.get_dtype()
'''
_jacobian_dtype=[('row0','f8'),
                  ('col0','f8'),
                  ('dudrow','f8'),
                  ('dudcol','f8'),
                  ('dvdrow','f8'),
                  ('dvdcol','f8'),
                  ('det','f8'),
                  ('sdet','f8')]


class Jacobian(object):
    def __init__(self, row0, col0, dudrow, dudcol, dvdrow, dvdcol):
        self._data = zeros(1, dtype=_jacobian_dtype)
        self._data['row0']=row0
        self._data['col0']=col0

        self._data['dudrow']=dudrow
        self._data['dudcol']=dudcol

        self._data['dvdrow']=dvdrow
        self._data['dvdcol']=dvdcol

        self._data['det'] = numpy.abs( dudrow*dvdcol-dudcol*dvdrow )
        self._data['sdet'] = sqrt(self._data['det'])

    def get_cen(self):
        """
        Get the center of the coordinate system
        """
        return copy.deepcopy(self._data['row0']), copy.deepcopy(self._data['col0'])

    def get_dudrow(self):
        """
        get the dudrow value
        """
        return copy.deepcopy(self._data['dudrow'][0])

    dudrow=property(fget=get_dudrow)

    def get_dudcol(self):
        """
        get the dudcol value
        """
        return copy.deepcopy(self._data['dudcol'][0])
    dudcol=property(fget=get_dudcol)

    def get_dvdrow(self):
        """
        get the dvdrow value
        """
        return copy.deepcopy(self._data['dvdrow'][0])
    dvdrow=property(fget=get_dvdrow)

    def get_dvdcol(self):
        """
        get the dvdcol value
        """
        return copy.deepcopy(self._data['dvdcol'][0])
    dvdcol=property(fget=get_dvdcol)


    def set_cen(self, row0, col0):
        """
        reset the center
        """
        self._data['row0'] = row0
        self._data['col0'] = col0

    def get_det(self):
        """
        Get the determinant of the jacobian matrix
        """
        return copy.deepcopy(self._data['det'][0])

    def get_sdet(self):
        """
        Get the sqrt(determinant) of the jacobian matrix
        """
        return copy.deepcopy(self._data['sdet'][0])

    def get_scale(self):
        """
        Get the scale, defined as sqrt(det)
        """
        return copy.deepcopy(self._data['sdet'][0])

    def copy(self):
        return Jacobian(self._data['row0'][0],
                        self._data['col0'][0],
                        self._data['dudrow'][0],
                        self._data['dudcol'][0],
                        self._data['dvdrow'][0],
                        self._data['dvdcol'][0])
    def __repr__(self):
        fmt="row0: %-10.5g col0: %-10.5g dudrow: %-10.5g dudcol: %-10.5g dvdrow: %-10.5g dvdcol: %-10.5g"
        return fmt % (self._data['row0'][0],
                      self._data['col0'][0],
                      self._data['dudrow'][0],
                      self._data['dudcol'][0],
                      self._data['dvdrow'][0],
                      self._data['dvdcol'][0])

class DiagonalJacobian(Jacobian):
    def __init__(self, cen1, cen2, scale=1.0):
        super(DiagonalJacobian,self).__init__(cen1, cen2, scale, 0., 0., scale)

class UnitJacobian(Jacobian):
    def __init__(self, cen1, cen2):
        super(UnitJacobian,self).__init__(cen1, cen2, 1., 0., 0., 1.)


