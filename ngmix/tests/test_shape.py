import numpy as np
import galsim
import pytest

from ..shape import (
    Shape,
    shear_reduced,
    g1g2_to_e1e2,
    e1e2_to_g1g2,
    eta1eta2_to_g1g2,
    g1g2_to_eta1eta2,
    e1e2_to_eta1eta2,
    dgs_by_dgo_jacob,
    get_round_factor,
    rotate_shape,
)
from ..gexceptions import GMixRangeError


@pytest.mark.parametrize("g1,g2,ang,rg1,rg2", [
    (0.1, 0.0, np.pi/2, -0.1, 0.0),
    (0.1, 0.0, -np.pi/2, -0.1, 0.0),
    (0.1, 0.0, np.pi, 0.1, 0.0),
    (0.0, 0.2, np.pi/2, 0.0, -0.2),
    (0.0, 0.2, -np.pi/2, 0.0, -0.2),
    (0.0, 0.2, np.pi, 0.0, 0.2),
    (0.1, 0.0, np.pi/4, 0.0, -0.1),
    (0.1, 0.0, -np.pi/4, 0.0, 0.1),
])
def test_rotate_shape(g1, g2, ang, rg1, rg2):
    trg1_trg2 = rotate_shape(g1, g2, ang)
    assert np.allclose(trg1_trg2, [rg1, rg2])


@pytest.mark.parametrize("g1,g2,s1,s2,g1o,g2o", [
    (0.1, -0.2, 0.0, 0.0, 0.1, -0.2),
    (0.0, 0.0, 0.1, -0.2, 0.1, -0.2),
])
def test_shape_reduced_zero(g1, g2, s1, s2, g1o, g2o):
    g1o_g2o = shear_reduced(g1, g2, s1, s2)
    assert np.allclose(g1o_g2o, [g1o, g2o])


def test_shape_reduced_range():
    g1o, g2o = shear_reduced(-0.5, 0.7, 0.999, 0.0)
    assert g1o >= -1.0
    assert g1o <= 1.0
    assert g2o >= -1.0
    assert g2o <= 1.0
    assert np.sqrt(g1o**2 + g2o**2) <= 1.0


def test_shape_reduced_value_galsim():
    g1o, g2o = shear_reduced(0.1, 0.2, -0.2, 0.03)
    ggs = galsim.Shear(g1=0.1, g2=0.2)
    sgs = galsim.Shear(g1=-0.2, g2=0.03)
    gtot = sgs + ggs  # note this order matters given how galsim does things
    assert np.allclose(g1o, gtot.g1)
    assert np.allclose(g2o, gtot.g2)


@pytest.mark.parametrize("e1,e2", [
    (0.0, -1.0),
    (-1.0, 0.0),
    (1.0, 0.0),
    (0.0, 1.0),
    (np.array([1.0, 0.0]), np.array([0.0, 0.0])),
])
def test_shape_e2g_raises(e1, e2):
    with pytest.raises(GMixRangeError):
        e1e2_to_g1g2(e1, e2)


@pytest.mark.parametrize("e1,e2", [
    (0.0, 0.0),
    (1.0 - 1e-15, 0.0),
    (0.0, 0.9999),
    (0.2, -0.1),
    (-0.2, 0.0),
    (0.1, 0.0),
    (-0.1, 0.2),
    (0.0, 0.2),
    (0.0, -0.1),
    (
        np.array([0.0, 1.0 - 1e-15, 0.0, 0.2, -0.2, 0.1, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.9999, -0.1, 0.0, 0.0, 0.2, 0.2, -0.1]),
    )
])
def test_shape_e2g_roundtrip(e1, e2):
    g1, g2 = e1e2_to_g1g2(e1, e2)
    e1_e2 = g1g2_to_e1e2(g1, g2)
    assert np.allclose(e1_e2, [e1, e2])


@pytest.mark.parametrize("e1,e2", [
    (0.0, 0.0),
    (1.0 - 5.5555e-17, 0.0),
    (np.sqrt(0.5) - 5e-16, np.sqrt(0.5)),
    (0.0, 0.9999),
    (0.2, -0.1),
    (-0.2, 0.0),
    (0.1, 0.0),
    (-0.1, 0.2),
    (0.0, 0.2),
    (0.0, -0.1),
    (
        np.array([0.0, 1.0 - 6e-17, 0.0, 0.2, -0.2, 0.1, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.9999, -0.1, 0.0, 0.0, 0.2, 0.2, -0.1]),
    )
])
def test_shape_e2g_galsim(e1, e2):
    g1, g2 = e1e2_to_g1g2(e1, e2)
    e1 = np.atleast_1d(e1)
    e2 = np.atleast_1d(e2)
    g1 = np.atleast_1d(g1)
    g2 = np.atleast_1d(g2)
    for _e1, _e2, _g1, _g2 in zip(e1, e2, g1, g2):
        s = galsim.Shear(e1=_e1, e2=_e2)
        assert np.allclose([s.g1, s.g2], [_g1, _g2])


@pytest.mark.parametrize("g1,g2", [
    (0.0, -1.0),
    (-1.0, 0.0),
    (1.0, 0.0),
    (0.0, 1.0),
    (np.array([1.0, 0.0]), np.array([0.0, 0.0])),
])
def test_shape_g2e_raises(g1, g2):
    with pytest.raises(GMixRangeError):
        g1g2_to_e1e2(g1, g2)


@pytest.mark.parametrize("g1,g2", [
    (0.0, 0.0),
    (1.0 - 1e-15, 0.0),
    (0.0, 0.9999),
    (0.2, -0.1),
    (-0.2, 0.0),
    (0.1, 0.0),
    (-0.1, 0.2),
    (0.0, 0.2),
    (0.0, -0.1),
    (
        np.array([0.0, 1.0 - 1e-15, 0.0, 0.2, -0.2, 0.1, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.9999, -0.1, 0.0, 0.0, 0.2, 0.2, -0.1]),
    )
])
def test_shape_g2e_roundtrip(g1, g2):
    e1, e2 = g1g2_to_e1e2(g1, g2)
    g1_g2 = e1e2_to_g1g2(e1, e2)
    assert np.allclose(g1_g2, [g1, g2]), (
        g1 - g1_g2[0], g2-g1_g2[1]
    )


@pytest.mark.parametrize("g1,g2", [
    (0.0, 0.0),
    (1.0 - 1e-15, 0.0),
    (0.0, 0.9999),
    (0.2, -0.1),
    (-0.2, 0.0),
    (0.1, 0.0),
    (-0.1, 0.2),
    (0.0, 0.2),
    (0.0, -0.1),
    (
        np.array([0.0, 1.0 - 1e-15, 0.0, 0.2, -0.2, 0.1, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.9999, -0.1, 0.0, 0.0, 0.2, 0.2, -0.1]),
    )
])
def test_shape_g2e_galsim(g1, g2):
    e1, e2 = g1g2_to_e1e2(g1, g2)
    e1 = np.atleast_1d(e1)
    e2 = np.atleast_1d(e2)
    g1 = np.atleast_1d(g1)
    g2 = np.atleast_1d(g2)
    for _e1, _e2, _g1, _g2 in zip(e1, e2, g1, g2):
        s = galsim.Shear(g1=_g1, g2=_g2)
        assert np.allclose([s.e1, s.e2], [_e1, _e2])


@pytest.mark.parametrize("eta1,eta2", [
    (0.0, -1e32),
    (-1e32, 0.0),
    (1e32, 0.0),
    (0.0, 1e32),
    (np.array([1e32, 0.0]), np.array([0.0, 0.0])),
])
def test_shape_eta2g_raises(eta1, eta2):
    with pytest.raises(GMixRangeError):
        eta1eta2_to_g1g2(eta1, eta2)


@pytest.mark.parametrize("eta1,eta2", [
    (0.0, 0.0),
    (-0.1, 0.2),
    (-0.2, 0.0),
    (0.1, 0.0),
    (0.2, 0.2),
    (0.2, -1.0),
    (0.0, -1e1),
    (-1e1, 0.0),
    (1e1, 0.0),
    (0.0, 1e1),
    (
        np.array([1e1, 0.0, 0.1, 0.2, -0.2, -0.2]),
        np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.1])),
])
def test_shape_eta2g_roundtrip(eta1, eta2):
    g1, g2 = eta1eta2_to_g1g2(eta1, eta2)
    eta1_eta2 = g1g2_to_eta1eta2(g1, g2)
    assert np.allclose(eta1_eta2, [eta1, eta2]), (
        eta1 - eta1_eta2[0], eta2-eta1_eta2[1]
    )


@pytest.mark.parametrize("eta1,eta2", [
    (0.0, 0.0),
    (-0.1, 0.2),
    (-0.2, 0.0),
    (0.1, 0.0),
    (0.2, 0.2),
    (0.2, -1.0),
    (0.0, -1e1),
    (-1e1, 0.0),
    (1e1, 0.0),
    (0.0, 1e1),
    (
        np.array([1e1, 0.0, 0.1, 0.2, -0.2, -0.2]),
        np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.1])),
])
def test_shape_eta2g_galsim(eta1, eta2):
    g1, g2 = eta1eta2_to_g1g2(eta1, eta2)
    eta1 = np.atleast_1d(eta1)
    eta2 = np.atleast_1d(eta2)
    g1 = np.atleast_1d(g1)
    g2 = np.atleast_1d(g2)
    for _eta1, _eta2, _g1, _g2 in zip(eta1, eta2, g1, g2):
        s = galsim.Shear(eta1=_eta1, eta2=_eta2)
        assert np.allclose([s.g1, s.g2], [_g1, _g2])


@pytest.mark.parametrize("eta1,eta2", [
    (0.0, 0.0),
    (-0.1, 0.2),
    (-0.2, 0.0),
    (0.1, 0.0),
    (0.2, 0.2),
    (0.2, -1.0),
    (0.0, -1e1),
    (-1e1, 0.0),
    (1e1, 0.0),
    (0.0, 1e1),
    (
        np.array([1e1, 0.0, 0.1, 0.2, -0.2, -0.2]),
        np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.1])),
])
def test_shape_eta2e_roundtrip(eta1, eta2):
    g1, g2 = eta1eta2_to_g1g2(eta1, eta2)
    e1, e2 = g1g2_to_e1e2(g1, g2)
    eta1_eta2 = e1e2_to_eta1eta2(e1, e2)
    assert np.allclose(eta1_eta2, [eta1, eta2]), (
        eta1 - eta1_eta2[0], eta2-eta1_eta2[1]
    )


@pytest.mark.parametrize("g1,g2", [
    (0.0, -1.0),
    (-1.0, 0.0),
    (1.0, 0.0),
    (0.0, 1.0),
    (np.array([1.0, 0.0]), np.array([0.0, 0.0])),
])
def test_shape_g2eta_raises(g1, g2):
    with pytest.raises(GMixRangeError):
        g1g2_to_eta1eta2(g1, g2)


@pytest.mark.parametrize("g1,g2", [
    (0.0, 0.0),
    (1.0 - 1e-15, 0.0),
    (0.0, 0.9999),
    (0.2, -0.1),
    (-0.2, 0.0),
    (0.1, 0.0),
    (-0.1, 0.2),
    (0.0, 0.2),
    (0.0, -0.1),
    (
        np.array([0.0, 1.0 - 1e-15, 0.0, 0.2, -0.2, 0.1, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.9999, -0.1, 0.0, 0.0, 0.2, 0.2, -0.1]),
    )
])
def test_shape_g2eta_roundtrip(g1, g2):
    eta1, eta2 = g1g2_to_eta1eta2(g1, g2)
    g1_g2 = eta1eta2_to_g1g2(eta1, eta2)
    assert np.allclose(g1_g2, [g1, g2]), (
        g1 - g1_g2[0], g2-g1_g2[1]
    )


@pytest.mark.parametrize("g1,g2", [
    (0.0, 0.0),
    (1.0 - 1e-15, 0.0),
    (0.0, 0.9999),
    (0.2, -0.1),
    (-0.2, 0.0),
    (0.1, 0.0),
    (-0.1, 0.2),
    (0.0, 0.2),
    (0.0, -0.1),
    (
        np.array([0.0, 1.0 - 1e-15, 0.0, 0.2, -0.2, 0.1, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.9999, -0.1, 0.0, 0.0, 0.2, 0.2, -0.1]),
    )
])
def test_shape_g2eta_galsim(g1, g2):
    eta1, eta2 = g1g2_to_eta1eta2(g1, g2)
    eta1 = np.atleast_1d(eta1)
    eta2 = np.atleast_1d(eta2)
    g1 = np.atleast_1d(g1)
    g2 = np.atleast_1d(g2)
    for _eta1, _eta2, _g1, _g2 in zip(eta1, eta2, g1, g2):
        s = galsim.Shear(g1=_g1, g2=_g2)
        assert np.allclose([s.eta1, s.eta2], [_eta1, _eta2])


@pytest.mark.parametrize("e1,e2", [
    (0.0, -1.0),
    (-1.0, 0.0),
    (1.0, 0.0),
    (0.0, 1.0),
    (np.array([1.0, 0.0]), np.array([0.0, 0.0])),
])
def test_shape_e2eta_raises(e1, e2):
    with pytest.raises(GMixRangeError):
        e1e2_to_eta1eta2(e1, e2)


@pytest.mark.parametrize("e1,e2", [
    (0.0, 0.0),
    (1.0 - 1e-15, 0.0),
    (0.0, 0.9999),
    (0.2, -0.1),
    (-0.2, 0.0),
    (0.1, 0.0),
    (-0.1, 0.2),
    (0.0, 0.2),
    (0.0, -0.1),
    (
        np.array([0.0, 1.0 - 1e-15, 0.0, 0.2, -0.2, 0.1, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.9999, -0.1, 0.0, 0.0, 0.2, 0.2, -0.1]),
    )
])
def test_shape_e2eta_roundtrip(e1, e2):
    eta1, eta2 = e1e2_to_eta1eta2(e1, e2)
    g1_g2 = eta1eta2_to_g1g2(eta1, eta2)
    e1_e2 = g1g2_to_e1e2(*g1_g2)
    assert np.allclose(e1_e2, [e1, e2]), (
        e1 - e1_e2[0], e2-e1_e2[1]
    )


@pytest.mark.parametrize("e1,e2", [
    (0.0, 0.0),
    (1.0 - 1e-15, 0.0),
    (0.0, 0.9999),
    (0.2, -0.1),
    (-0.2, 0.0),
    (0.1, 0.0),
    (-0.1, 0.2),
    (0.0, 0.2),
    (0.0, -0.1),
    (
        np.array([0.0, 0.9999, 0.0, 0.2, -0.2, 0.1, -0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.9999, -0.1, 0.0, 0.0, 0.2, 0.2, -0.1]),
    )
])
def test_shape_e2eta_galsim(e1, e2):
    eta1, eta2 = e1e2_to_eta1eta2(e1, e2)
    eta1 = np.atleast_1d(eta1)
    eta2 = np.atleast_1d(eta2)
    e1 = np.atleast_1d(e1)
    e2 = np.atleast_1d(e2)
    for _eta1, _eta2, _e1, _e2 in zip(eta1, eta2, e1, e2):
        s = galsim.Shear(e1=_e1, e2=_e2)
        assert np.allclose([s.eta1, s.eta2], [_eta1, _eta2])


@pytest.mark.parametrize("s1,s2", [
    (0.0, -1.0),
    (-1.0, 0.0),
    (1.0, 0.0),
    (0.0, 1.0),
    (np.array([1.0, 0.0]), np.array([0.0, 1.0])),
])
@pytest.mark.parametrize("g1,g2", [
    (0.0, 0.2),
    (0.2, 0.0),
    (-0.2, 0.1),
    (0.1, -0.2),
])
def test_shape_dgs_by_dgo_jacob_zero(g1, g2, s1, s2):
    assert np.allclose(0, dgs_by_dgo_jacob(g1, g2, s1, s2))


@pytest.mark.parametrize("g1,g2", [
    (0.0, -1.0),
    (-1.0, 0.0),
    (1.0, 0.0),
    (0.0, 1.0),
    (np.array([1.0, 0.0]), np.array([0.0, 1.0])),
])
def test_shape_get_round_factor_zero(g1, g2):
    assert np.allclose(0, get_round_factor(g1, g2))


@pytest.mark.parametrize("g1,g2", [
    (0.0, 0.0),
    (np.array([0.0, 0.0]), np.array([0.0, 0.0])),
])
def test_shape_get_round_factor_one(g1, g2):
    assert np.allclose(1, get_round_factor(g1, g2))


@pytest.mark.parametrize("func", [
    g1g2_to_e1e2,
    e1e2_to_g1g2,
    g1g2_to_eta1eta2,
    eta1eta2_to_g1g2,
    e1e2_to_eta1eta2,
])
@pytest.mark.parametrize("val", [
    0,
    np.zeros(10),
    np.zeros(1),
])
def test_shape_zero2zero(func, val):
    out = func(val, val)
    assert np.allclose(0, out)
    assert np.ndim(val) == np.ndim(out[0])
    assert np.ndim(val) == np.ndim(out[1])
    if isinstance(val, np.ndarray):
        assert isinstance(out[0], np.ndarray)
        assert isinstance(out[1], np.ndarray)


@pytest.mark.parametrize("g1,g2", [
    (0.0, -1.0),
    (-1.0, 0.0),
    (1.0, 0.0),
    (0.0, 1.0),
])
def test_shape_class_raises(g1, g2):
    with pytest.raises(GMixRangeError):
        Shape(g1, g2)

    s = Shape(0.1, 0.2)
    with pytest.raises(GMixRangeError):
        s.set_g1g2(g1, g2)


def test_shape_class_set():
    s = Shape(0.1, 0.2)
    assert np.allclose(s.g1, 0.1)
    assert np.allclose(s.g2, 0.2)
    assert np.allclose(s.g, np.sqrt(0.1**2 + 0.2**2))
    s.set_g1g2(-0.1, 0.05)
    assert np.allclose(s.g1, -0.1)
    assert np.allclose(s.g2, 0.05)
    assert np.allclose(s.g, np.sqrt(0.1**2 + 0.05**2))


def test_shape_class_neg():
    s = Shape(0.1, 0.2)
    assert np.allclose(s.g1, 0.1)
    assert np.allclose(s.g2, 0.2)
    assert np.allclose(s.g, np.sqrt(0.1**2 + 0.2**2))
    sn = -s
    assert np.allclose(sn.g1, -0.1)
    assert np.allclose(sn.g2, -0.2)
    assert np.allclose(sn.g, np.sqrt(0.1**2 + 0.2**2))
    assert sn is not s


def test_shape_class_copy():
    s = Shape(0.1, 0.2)
    sc = s.copy()
    assert s is not sc
    assert np.allclose(s.g1, sc.g1)
    assert np.allclose(s.g2, sc.g2)
    assert np.allclose(s.g, sc.g)


def test_shape_class_repr():
    s = Shape(0.1, 0.2)
    se = eval("Shape" + repr(s))
    assert np.allclose(se.g1, s.g1)
    assert np.allclose(se.g2, s.g2)


@pytest.mark.parametrize("g1,g2,ang,rg1,rg2", [
    (0.1, 0.0, np.pi/2, -0.1, 0.0),
    (0.1, 0.0, -np.pi/2, -0.1, 0.0),
    (0.1, 0.0, np.pi, 0.1, 0.0),
    (0.0, 0.2, np.pi/2, 0.0, -0.2),
    (0.0, 0.2, -np.pi/2, 0.0, -0.2),
    (0.0, 0.2, np.pi, 0.0, 0.2),
    (0.1, 0.0, np.pi/4, 0.0, -0.1),
    (0.1, 0.0, -np.pi/4, 0.0, 0.1),
])
def test_shape_class_rotated(g1, g2, ang, rg1, rg2):
    s = Shape(g1, g2)
    sr = s.get_rotated(ang)
    assert np.allclose([sr.g1, sr.g2], [rg1, rg2])

    s.rotate(ang)
    assert np.allclose([s.g1, s.g2], [rg1, rg2])


def test_shape_class_sheared():
    g = Shape(0.1, 0.2)
    sg = g.get_sheared(0.05, -0.05)
    s = Shape(0.05, -0.05)
    sgs = g.get_sheared(s)
    assert sg is not g
    assert sg is not sgs
    assert sgs is not g
    assert np.allclose(sg.g1, sgs.g1)
    assert np.allclose(sg.g2, sgs.g2)
    assert np.allclose(sg.g, sgs.g)


def test_shape_class_sheared_raises():
    g = Shape(0.1, 0.2)
    with pytest.raises(ValueError):
        g.get_sheared(0.1)
