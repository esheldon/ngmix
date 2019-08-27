import numpy as np
import galsim
import pytest
from ngmix.jacobian import Jacobian, UnitJacobian, DiagonalJacobian


@pytest.mark.parametrize('kind', ['row-col', 'x-y', 'galsim'])
def test_jacobian_smoke(kind):
    dudcol = 0.25
    dudrow = 0.1
    dvdcol = -0.4
    dvdrow = 0.34
    col = 5.6
    row = -10.4

    if kind == 'row-col':
        jac = Jacobian(
            col=col,
            row=row,
            dudcol=dudcol,
            dudrow=dudrow,
            dvdcol=dvdcol,
            dvdrow=dvdrow)
    elif kind == 'x-y':
        jac = Jacobian(
            x=col,
            y=row,
            dudx=dudcol,
            dudy=dudrow,
            dvdx=dvdcol,
            dvdy=dvdrow)
    else:
        wcs = galsim.JacobianWCS(
            dudx=dudcol,
            dudy=dudrow,
            dvdx=dvdcol,
            dvdy=dvdrow)
        jac = Jacobian(x=col, y=row, wcs=wcs)

    assert np.allclose(jac.row0, row)
    assert np.allclose(jac.col0, col)
    assert np.allclose(jac.dudcol, dudcol)
    assert np.allclose(jac.dudrow, dudrow)
    assert np.allclose(jac.dvdcol, dvdcol)
    assert np.allclose(jac.dvdrow, dvdrow)

    assert np.allclose(jac.det, dudcol * dvdrow - dudrow * dvdcol)
    assert np.allclose(
        jac.sdet, np.sqrt(np.abs(dudcol * dvdrow - dudrow * dvdcol)))
    assert np.allclose(
        jac.scale, np.sqrt(np.abs(dudcol * dvdrow - dudrow * dvdcol)))

    r, c = 20.0, -44.5
    v, u = jac.get_vu(r, c)
    assert np.allclose(v, dvdrow * (r - row) + dvdcol * (c - col))
    assert np.allclose(u, dudrow * (r - row) + dudcol * (c - col))
    v, u = jac(r, c)
    assert np.allclose(v, dvdrow * (r - row) + dvdcol * (c - col))
    assert np.allclose(u, dudrow * (r - row) + dudcol * (c - col))

    v, u = 20.0, -44.5
    r, c = jac.get_rowcol(v, u)
    assert np.allclose(r, (dudcol * v - dvdcol * u)/jac.det + row)
    assert np.allclose(c, (-dudrow * v + dvdrow * u)/jac.det + col)

    gs_wcs = jac.get_galsim_wcs()
    assert np.allclose(gs_wcs.dudx, jac.dudcol)
    assert np.allclose(gs_wcs.dudy, jac.dudrow)
    assert np.allclose(gs_wcs.dvdx, jac.dvdcol)
    assert np.allclose(gs_wcs.dvdy, jac.dvdrow)

    cpy_jac = jac.copy()
    cpy_jac.set_cen(row=-11, col=-12)
    assert np.allclose(jac.row0, row)
    assert np.allclose(jac.col0, col)
    assert np.allclose(cpy_jac.row0, -11)
    assert np.allclose(cpy_jac.col0, -12)


@pytest.mark.parametrize('kind', ['row-col', 'x-y'])
def test_diagonal_jacobian_smoke(kind):
    scale = 0.25
    dudcol = scale
    dudrow = 0
    dvdcol = 0
    dvdrow = scale
    col = 5.6
    row = -10.4

    if kind == 'row-col':
        jac = DiagonalJacobian(
            scale=scale,
            col=col,
            row=row)
    elif kind == 'x-y':
        jac = DiagonalJacobian(
            scale=scale,
            x=col,
            y=row)

    assert np.allclose(jac.row0, row)
    assert np.allclose(jac.col0, col)
    assert np.allclose(jac.dudcol, dudcol)
    assert np.allclose(jac.dudrow, dudrow)
    assert np.allclose(jac.dvdcol, dvdcol)
    assert np.allclose(jac.dvdrow, dvdrow)

    assert np.allclose(jac.det, dudcol * dvdrow - dudrow * dvdcol)
    assert np.allclose(
        jac.sdet, np.sqrt(np.abs(dudcol * dvdrow - dudrow * dvdcol)))
    assert np.allclose(
        jac.scale, np.sqrt(np.abs(dudcol * dvdrow - dudrow * dvdcol)))

    r, c = 20.0, -44.5
    v, u = jac.get_vu(r, c)
    assert np.allclose(v, dvdrow * (r - row) + dvdcol * (c - col))
    assert np.allclose(u, dudrow * (r - row) + dudcol * (c - col))
    v, u = jac(r, c)
    assert np.allclose(v, dvdrow * (r - row) + dvdcol * (c - col))
    assert np.allclose(u, dudrow * (r - row) + dudcol * (c - col))

    v, u = 20.0, -44.5
    r, c = jac.get_rowcol(v, u)
    assert np.allclose(r, (dudcol * v - dvdcol * u)/jac.det + row)
    assert np.allclose(c, (-dudrow * v + dvdrow * u)/jac.det + col)

    gs_wcs = jac.get_galsim_wcs()
    assert np.allclose(gs_wcs.dudx, jac.dudcol)
    assert np.allclose(gs_wcs.dudy, jac.dudrow)
    assert np.allclose(gs_wcs.dvdx, jac.dvdcol)
    assert np.allclose(gs_wcs.dvdy, jac.dvdrow)

    cpy_jac = jac.copy()
    cpy_jac.set_cen(row=-11, col=-12)
    assert np.allclose(jac.row0, row)
    assert np.allclose(jac.col0, col)
    assert np.allclose(cpy_jac.row0, -11)
    assert np.allclose(cpy_jac.col0, -12)


@pytest.mark.parametrize('kind', ['row-col', 'x-y'])
def test_unit_jacobian_smoke(kind):
    scale = 1
    dudcol = scale
    dudrow = 0
    dvdcol = 0
    dvdrow = scale
    col = 5.6
    row = -10.4

    if kind == 'row-col':
        jac = UnitJacobian(
            col=col,
            row=row)
    elif kind == 'x-y':
        jac = UnitJacobian(
            x=col,
            y=row)

    assert np.allclose(jac.row0, row)
    assert np.allclose(jac.col0, col)
    assert np.allclose(jac.dudcol, dudcol)
    assert np.allclose(jac.dudrow, dudrow)
    assert np.allclose(jac.dvdcol, dvdcol)
    assert np.allclose(jac.dvdrow, dvdrow)

    assert np.allclose(jac.det, dudcol * dvdrow - dudrow * dvdcol)
    assert np.allclose(
        jac.sdet, np.sqrt(np.abs(dudcol * dvdrow - dudrow * dvdcol)))
    assert np.allclose(
        jac.scale, np.sqrt(np.abs(dudcol * dvdrow - dudrow * dvdcol)))

    r, c = 20.0, -44.5
    v, u = jac.get_vu(r, c)
    assert np.allclose(v, dvdrow * (r - row) + dvdcol * (c - col))
    assert np.allclose(u, dudrow * (r - row) + dudcol * (c - col))
    v, u = jac(r, c)
    assert np.allclose(v, dvdrow * (r - row) + dvdcol * (c - col))
    assert np.allclose(u, dudrow * (r - row) + dudcol * (c - col))

    v, u = 20.0, -44.5
    r, c = jac.get_rowcol(v, u)
    assert np.allclose(r, (dudcol * v - dvdcol * u)/jac.det + row)
    assert np.allclose(c, (-dudrow * v + dvdrow * u)/jac.det + col)

    gs_wcs = jac.get_galsim_wcs()
    assert np.allclose(gs_wcs.dudx, jac.dudcol)
    assert np.allclose(gs_wcs.dudy, jac.dudrow)
    assert np.allclose(gs_wcs.dvdx, jac.dvdcol)
    assert np.allclose(gs_wcs.dvdy, jac.dvdrow)

    cpy_jac = jac.copy()
    cpy_jac.set_cen(row=-11, col=-12)
    assert np.allclose(jac.row0, row)
    assert np.allclose(jac.col0, col)
    assert np.allclose(cpy_jac.row0, -11)
    assert np.allclose(cpy_jac.col0, -12)


@pytest.mark.parametrize('kind', ['row-col', 'x-y'])
def test_jacobian_missing_kwargs(kind):
    dudcol = 0.25
    dudrow = 0.1
    dvdcol = -0.4
    dvdrow = 0.34
    col = 5.6
    row = -10.4

    if kind == 'row-col':
        kwargs = dict(
            col=col,
            row=row,
            dudcol=dudcol,
            dudrow=dudrow,
            dvdcol=dvdcol,
            dvdrow=dvdrow)
    elif kind == 'x-y':
        kwargs = dict(
            x=col,
            y=row,
            dudx=dudcol,
            dudy=dudrow,
            dvdx=dvdcol,
            dvdy=dvdrow)

    for key in kwargs:
        tst_kwargs = {}
        tst_kwargs.update(kwargs)
        del tst_kwargs[key]

        with pytest.raises(ValueError):
            Jacobian(**tst_kwargs)


@pytest.mark.parametrize('kind', ['row-col', 'x-y'])
def test_diagonal_jacobian_missing_kwargs(kind):
    scale = 0.25
    col = 5.6
    row = -10.4

    if kind == 'row-col':
        kwargs = dict(
            col=col,
            row=row,
            scale=scale)
    elif kind == 'x-y':
        kwargs = dict(
            x=col,
            y=row,
            scale=scale)

    for key in kwargs:
        tst_kwargs = {}
        tst_kwargs.update(kwargs)
        del tst_kwargs[key]

        with pytest.raises(Exception):
            Jacobian(**tst_kwargs)


@pytest.mark.parametrize('kind', ['row-col', 'x-y'])
def test_unit_jacobian_missing_kwargs(kind):
    col = 5.6
    row = -10.4

    if kind == 'row-col':
        kwargs = dict(
            col=col,
            row=row)
    elif kind == 'x-y':
        kwargs = dict(
            x=col,
            y=row)

    for key in kwargs:
        tst_kwargs = {}
        tst_kwargs.update(kwargs)
        del tst_kwargs[key]

        with pytest.raises(Exception):
            Jacobian(**tst_kwargs)
