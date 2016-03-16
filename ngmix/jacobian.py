import numpy
from numpy import zeros, sqrt

import copy

_jacobian_dtype=[('row0','f8'),
                 ('col0','f8'),
                 ('dvdrow','f8'),
                 ('dvdcol','f8'),
                 ('dudrow','f8'),
                 ('dudcol','f8'),
                 ('det','f8'),
                 ('sdet','f8')]

_ROWCOL_REQ=['row','col',
             'dvdrow',
             'dvdcol',
             'dudrow',
             'dudcol']

_XY_REQ=['x','y',
         'dudx',
         'dudy',
         'dvdx',
         'dvdy']

class Jacobian(object):
    """
    A class representing a jacobian matrix of a transformation.  The
    jacobian is defined relative to the input center

    You can create the jacobian using either row,col or x,y naming convention,
    but internally row,col is used to make correspondence to C row-major arrays
    clear

    parameters for row,col mode
    ---------------------------
    row: keyword
        The row of the jacobian center
    col: keyword
        The column of the jacobian center
    dvdrow: keyword
        How v varies with row
    dvdcol: keyword
        How v varies with column
    dudrow: keyword
        How u varies with row
    dudcol: keyword
        How u varies with column

    parameters for x,y mode
    ---------------------------
    x: keyword
        The x (column) of the jacobian center
    y: keyword
        The y (row) of the jacobian center
    dudx: keyword
        How u varies with x
    dudy: keyword
        How u varies with y
    dvdx: keyword
        How v varies with x
    dvdy: keyword
        How v varies with y
    """
    def __init__(self, **kw):
        self._data = zeros(1, dtype=_jacobian_dtype)

        if 'x' in kw:
            self._init_xy(**kw)
        elif 'row' in kw:
            self._init_rowcol(**kw)
        else:
            raise ValueError("send by row,col or x,y")

    def _init_rowcol(self, **kw):
        for k in _ROWCOL_REQ:
            if k not in kw:
                raise ValueError("missing keyword: '%s'" % k)

        dvdrow=kw['dvdrow']
        dvdcol=kw['dvdcol']

        dudrow=kw['dudrow']
        dudcol=kw['dudcol']


        self._data['row0']=kw['row']
        self._data['col0']=kw['col']

        self._data['dvdrow']=dvdrow
        self._data['dvdcol']=dvdcol

        self._data['dudrow']=dudrow
        self._data['dudcol']=dudcol

        self._data['det'] = numpy.abs( dudrow*dvdcol-dudcol*dvdrow )
        self._data['sdet'] = sqrt(self._data['det'])

    def _init_xy(self, **kw):
        for k in _XY_REQ:
            if k not in kw:
                raise ValueError("missing keyword: '%s'" % k)

        dvdrow=kw['dvdy']
        dvdcol=kw['dvdx']

        dudrow=kw['dudy']
        dudcol=kw['dudx']

        self._data['row0']=kw['y']
        self._data['col0']=kw['x']

        self._data['dvdrow']=dvdrow
        self._data['dvdcol']=dvdcol

        self._data['dudrow']=dudrow
        self._data['dudcol']=dudcol

        self._data['det'] = numpy.abs( dudrow*dvdcol-dudcol*dvdrow )
        self._data['sdet'] = sqrt(self._data['det'])


    def get_cen(self):
        """
        Get the center of the coordinate system

        returns
            (row,col)
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


    def set_cen(self, **kw):
        """
        reset the center
        """

        if 'row' in kw:
            self._data['row0'] = kw['row']
            self._data['col0'] = kw['col']
        elif 'x' in kw:
            self._data['row0'] = kw['y']
            self._data['col0'] = kw['x']
        else:
            raise ValueError("expected row=,col= or x=,y=")

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
        return Jacobian(row=self._data['row0'][0],
                        col=self._data['col0'][0],
                        dudrow=self._data['dudrow'][0],
                        dudcol=self._data['dudcol'][0],
                        dvdrow=self._data['dvdrow'][0],
                        dvdcol=self._data['dvdcol'][0])
    def __repr__(self):
        fmt="row0: %-10.5g col0: %-10.5g dvdrow: %-10.5g dvdcol: %-10.5g dudrow: %-10.5g dudcol: %-10.5g"
        return fmt % (self._data['row0'][0],
                      self._data['col0'][0],
                      self._data['dvdrow'][0],
                      self._data['dvdcol'][0],
                      self._data['dudrow'][0],
                      self._data['dudcol'][0])

class DiagonalJacobian(Jacobian):
    """
    Define a diagonal jacobian based on the input scale.  Diagonal
    means that u varies directly with column/x and v varies directly
    with row/y

    parameters
    ----------
    scale: keyword
        The scale of the jacobian, the diagonal elements of
        the matrix.  Default 1.0

    parameters for row,col mode
    ---------------------------
    row: keyword
        The row of the jacobian center
    col: keyword
        The column of the jacobian center

    parameters for x,y mode
    ---------------------------
    x: keyword
        The x (column) of the jacobian center
    y: keyword
        The y (row) of the jacobian center
    """
    def __init__(self, scale=1.0, **kw):

        if 'x' in kw:
            assert 'y' in kw,"send both x= and y="
            super(DiagonalJacobian,self).__init__(x=kw['x'],
                                                  y=kw['y'],
                                                  dudx=scale,
                                                  dudy=0.0,
                                                  dvdx=0.0,
                                                  dvdy=scale)
        elif 'row' in kw:
            assert 'col' in kw,"send both row= and col="
            super(DiagonalJacobian,self).__init__(row=kw['row'],
                                                  col=kw['col'],
                                                  dvdrow=scale,
                                                  dvdcol=0.0,
                                                  dudrow=0.0,
                                                  dudcol=scale)
        else:
            raise ValueError("expected row=,col= or x=,y=")
        super(DiagonalJacobian,self).__init__(cen1, cen2, scale, 0., 0., scale)

class UnitJacobian(DiagonalJacobian):
    """
    Define a diagonal jacobian with scale=1.0

    Diagonal means that u varies directly with column/x and v varies directly
    with row/y


    parameters for row,col mode
    ---------------------------
    row: keyword
        The row of the jacobian center
    col: keyword
        The column of the jacobian center

    parameters for x,y mode
    ---------------------------
    x: keyword
        The x (column) of the jacobian center
    y: keyword
        The y (row) of the jacobian center
    """

    def __init__(self, **kw):
        super(UnitJacobian,self).__init__(scale=1.0, **kw)


