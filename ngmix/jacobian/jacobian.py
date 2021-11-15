from numpy import zeros, sqrt, abs

__all__ = ['Jacobian', 'DiagonalJacobian', 'UnitJacobian']

_jacobian_dtype = [
    ('row0', 'f8'),
    ('col0', 'f8'),
    ('dvdrow', 'f8'),
    ('dvdcol', 'f8'),
    ('dudrow', 'f8'),
    ('dudcol', 'f8'),
    ('det', 'f8'),
    ('scale', 'f8')]

_ROWCOL_REQ = [
    'row', 'col',
    'dvdrow',
    'dvdcol',
    'dudrow',
    'dudcol']

_XY_REQ = [
    'x', 'y',
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

    parameters
    -----------
    Note: You can always send wcs= instead of the individual derivatives, which
    must have the attributes following the galsim convention dudx, dudy, etc.

    parameters for row,col mode
    ---------------------------

    row: keyword
        The row of the jacobian center
    col: keyword
        The column of the jacobian center
    Either of the following
            wcs: keyword
                object with attributes .dudx,.dudy,etc.
        OR
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
    Either of the following
            wcs: keyword
                object with attributes .dudx,.dudy,etc.
        OR
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

    def get_data(self):
        """
        Get a reference to the underlying numpy array that represents
        the jacobian

        Returns
        -------
        structured array reference
        """
        return self._data

    def get_cen(self):
        """
        Get the center of the coordinate system

        returns
        -------
        row,col: float
            The row and column of the jacobian center.
        """
        return self._data['row0'][0], self._data['col0'][0]

    def get_row0(self):
        """
        get the dvdrow value
        """
        return self._data['row0'][0]

    def get_col0(self):
        """
        get the dvdrow value
        """
        return self._data['col0'][0]

    def get_dvdrow(self):
        """
        get the dvdrow value
        """
        return self._data['dvdrow'][0]

    def get_dvdcol(self):
        """
        get the dvdcol value
        """
        return self._data['dvdcol'][0]

    def get_dudrow(self):
        """
        get the dudrow value
        """
        return self._data['dudrow'][0]

    def get_dudcol(self):
        """
        get the dudcol value
        """
        return self._data['dudcol'][0]

    def get_det(self):
        """
        Get the determinant of the jacobian matrix
        """
        return self._data['det'][0]

    def get_scale(self):
        """
        Get the scale, defined as sqrt(det)
        """
        return self._data['scale'][0]

    def get_area(self):
        """
        Get the area of a pixel
        """
        return self.scale**2

    def get_vu(self, row, col):
        """
        get v,u given row,col

        parameters
        ----------
        row,col: float or array
            The row and column of the input points.

        returns
        -------
        v,u: float or array
            The output (v,u) locations in the tangent plane.
        """
        from .jacobian_nb import jacobian_get_vu
        return jacobian_get_vu(self._data, row, col)

    def get_rowcol(self, v, u):
        """
        get row,col given v,u

        parameters
        ----------
        v,u: float or array
            The v,u location of the input points in the tangent plane.

        returns
        -------
        row,col: float or array
            The row,col image location of the input points.
        """
        from .jacobian_nb import jacobian_get_rowcol
        return jacobian_get_rowcol(self._data, v, u)

    def __call__(self, row, col):
        from .jacobian_nb import jacobian_get_vu
        return jacobian_get_vu(self._data, row, col)

    cen = property(fget=get_cen)

    row0 = property(fget=get_row0)
    col0 = property(fget=get_col0)

    dvdrow = property(fget=get_dvdrow)
    dvdcol = property(fget=get_dvdcol)
    dudrow = property(fget=get_dudrow)
    dudcol = property(fget=get_dudcol)

    det = property(fget=get_det)
    scale = property(fget=get_scale)
    area = property(fget=get_area)

    def set_cen(self, **kw):
        """
        reset the center

        parameters
        ----------
        row,col: float
            The row,col location of the center sent as keywords.

        or

        x,y: float
            The x,y location of the center sent as keywords.
        """

        if 'row' in kw:
            self._data['row0'] = kw['row']
            self._data['col0'] = kw['col']
        elif 'x' in kw:
            self._data['row0'] = kw['y']
            self._data['col0'] = kw['x']
        else:
            raise ValueError("expected row=,col= or x=,y=")

    def copy(self):
        """
        get a new Jacobian with the same values as self
        """
        return Jacobian(row=self.row0,
                        col=self.col0,
                        dudrow=self.dudrow,
                        dudcol=self.dudcol,
                        dvdrow=self.dvdrow,
                        dvdcol=self.dvdcol)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self):
        return self.copy()

    def __eq__(self, jacobian):
        self_data = self.get_data()
        data = jacobian.get_data()
        return self_data == data

    def get_galsim_wcs(self):
        """
        get a galsim.JacobianWCS object with the same contents as self
        """
        import galsim

        dudx = self.dudcol
        dudy = self.dudrow
        dvdx = self.dvdcol
        dvdy = self.dvdrow

        return galsim.JacobianWCS(dudx, dudy, dvdx, dvdy)

    def _init_rowcol(self, **kw):

        if 'wcs' in kw:
            dvdrow, dvdcol, dudrow, dudcol = self._extract_wcs(kw['wcs'])
        else:
            for k in _ROWCOL_REQ:
                if k not in kw:
                    raise ValueError("missing keyword: '%s'" % k)

            dvdrow = kw['dvdrow']
            dvdcol = kw['dvdcol']

            dudrow = kw['dudrow']
            dudcol = kw['dudcol']

        self._finish_init(kw['row'], kw['col'],
                          dvdrow, dvdcol, dudrow, dudcol)

    def _init_xy(self, **kw):
        if 'wcs' in kw:
            dvdrow, dvdcol, dudrow, dudcol = self._extract_wcs(kw['wcs'])
        else:
            for k in _XY_REQ:
                if k not in kw:
                    raise ValueError("missing keyword: '%s'" % k)

            dvdrow = kw['dvdy']
            dvdcol = kw['dvdx']

            dudrow = kw['dudy']
            dudcol = kw['dudx']

        self._finish_init(kw['y'], kw['x'],
                          dvdrow, dvdcol, dudrow, dudcol)

    def _extract_wcs(self, wcs):
        dvdrow = wcs.dvdy
        dvdcol = wcs.dvdx

        dudrow = wcs.dudy
        dudcol = wcs.dudx

        return dvdrow, dvdcol, dudrow, dudcol

    def _finish_init(self, row0, col0, dvdrow, dvdcol, dudrow, dudcol):
        self._data['row0'] = row0
        self._data['col0'] = col0

        self._data['dvdrow'] = dvdrow
        self._data['dvdcol'] = dvdcol

        self._data['dudrow'] = dudrow
        self._data['dudcol'] = dudcol

        self._data['det'] = dvdrow*dudcol - dvdcol*dudrow
        self._data['scale'] = sqrt(abs(self._data['det']))

    def __repr__(self):
        fmt = (
            'row0: %-10.5g col0: %-10.5g dvdrow: %-10.5g '
            'dvdcol: %-10.5g dudrow: %-10.5g dudcol: %-10.5g'
        )
        return fmt % (self.row0,
                      self.col0,
                      self.dvdrow,
                      self.dvdcol,
                      self.dudrow,
                      self.dudcol)


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
    ------------------------
    x: keyword
        The x (column) of the jacobian center
    y: keyword
        The y (row) of the jacobian center
    """
    def __init__(self, scale=1.0, **kw):

        if 'x' in kw:
            assert 'y' in kw, "send both x= and y="
            super(DiagonalJacobian, self).__init__(x=kw['x'],
                                                   y=kw['y'],
                                                   dudx=scale,
                                                   dudy=0.0,
                                                   dvdx=0.0,
                                                   dvdy=scale)
        elif 'row' in kw:
            assert 'col' in kw, "send both row= and col="
            super(DiagonalJacobian, self).__init__(row=kw['row'],
                                                   col=kw['col'],
                                                   dvdrow=scale,
                                                   dvdcol=0.0,
                                                   dudrow=0.0,
                                                   dudcol=scale)
        else:
            raise ValueError("expected row=,col= or x=,y=")


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
        super(UnitJacobian, self).__init__(scale=1.0, **kw)
