from numpy import zeros
from numba import float64, struct, jit, autojit

_jacobian=struct([('row0',float64),
                  ('col0',float64),
                  ('dudrow',float64),
                  ('dudcol',float64),
                  ('dvdrow',float64),
                  ('dvdcol',float64)])
_jacobian_dtype=_jacobian.get_dtype()


class Jacobian(object):
    def __init__(self, row0, col0, dudrow, dudcol, dvdrow, dvdcol):
        self._data = zeros(1, dtype=_jacobian_dtype)
        self._data['row0']=row0
        self._data['col0']=col0

        self._data['dudrow']=dudrow
        self._data['dudcol']=dudcol

        self._data['dvdrow']=dvdrow
        self._data['dvdcol']=dvdcol

    def __repr__(self):
        fmt="row0: %-10.5g col0: %-10.5g dudrow: %-10.5g dudcol: %-10.5g dvdrow: %-10.5g dvdcol: %-10.5g"
        return fmt % (self._data['row0'][0],
                      self._data['col0'][0],
                      self._data['dudrow'][0],
                      self._data['dudcol'][0],
                      self._data['dvdrow'][0],
                      self._data['dvdcol'][0])

class UnitJacobian(Jacobian):
    def __init__(self, cen1, cen2):
        super(UnitJacobian,self).__init__(cen1, cen2, 1., 0., 0., 1.)
