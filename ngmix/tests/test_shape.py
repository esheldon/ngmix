import numpy as np
import galsim
import pytest

from ..shape import (
    Shape,
    shear_reduced,
    g1g2_to_e1e2,
    e1e2_to_g1g2,
    g1g2_to_eta1eta2,
    eta1eta2_to_g1g2,
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
def test_shear_reduced_zero(g1, g2, s1, s2, g1o, g2o):
    g1o_g2o = shear_reduced(g1, g2, s1, s2)
    assert np.allclose(g1o_g2o, [g1o, g2o])


def test_shear_reduced_range():
    g1o, g2o = shear_reduced(-0.5, 0.7, 0.999, 0.0)
    assert g1o >= -1.0
    assert g1o <= 1.0
    assert g2o >= -1.0
    assert g2o <= 1.0
    assert np.sqrt(g1o**2 + g2o**2) <= 1.0


def test_shear_reduced_value_galsim():
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
def test_shear_e2g_raises(e1, e2):
    with pytest.raises(GMixRangeError):
        e1e2_to_g1g2(e1, e2)


@pytest.mark.parametrize("e1,e2", [
    (0.0, 0.0),
    (0.9999, 0.0),
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
def test_shear_e2g_roundtrip(e1, e2):
    g1, g2 = e1e2_to_g1g2(e1, e2)
    e1_e2 = g1g2_to_e1e2(g1, g2)
    assert np.allclose(e1_e2, [e1, e2])


@pytest.mark.parametrize("g1,g2", [
    (0.0, -1.0),
    (-1.0, 0.0),
    (1.0, 0.0),
    (0.0, 1.0),
    (np.array([1.0, 0.0]), np.array([0.0, 0.0])),
])
def test_shear_g2e_raises(g1, g2):
    with pytest.raises(GMixRangeError):
        g1g2_to_e1e2(g1, g2)


@pytest.mark.parametrize("g1,g2", [
    (0.0, 0.0),
    (0.9999, 0.0),
    (0.0, 0.9999),
    (0.2, -0.1),
    (-0.2, 0.0),
    (0.1, 0.0),
    (-0.1, 0.2),
    (0.0, 0.2),
    (0.0, -0.1),
    (
        np.array([0.0, 0.9999, 0.0, 0.2, -0.2, 0.1, -0.1, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 0.9999, -0.1, 0.0, 0.0, 0.2, 0.2, -0.1], dtype=np.float64),
    )
])
def test_shear_g2e_roundtrip(g1, g2):
    e1, e2 = g1g2_to_e1e2(g1, g2)
    g1_g2 = e1e2_to_g1g2(e1, e2)
    assert np.allclose(g1_g2, [g1, g2]), (
        g1 - g1_g2[0], g2-g1_g2[1]
    )
