import numpy as np
import pytest

from ..shape import (
    Shape,
    shear_reduced,
    g1g2_to_e1e2,
    e1e2_to_g1g2,
    g1g2_to_eta1eta2,
    e1e2_to_eta1eta2,
    eta1eta2_to_g1g2,
    dgs_by_dgo_jacob,
    get_round_factor,
    rotate_shape,
)


@pytest.mark.parametrize("g1,g2,ang,rg1,rg2", [
    (0.1, 0.0, np.pi/2, -0.1, 0.0),
])
def test_rotate_shape(g1, g2, ang, rg1, rg2):
    trg1_trg2 = rotate_shape(g1, g2, ang)
    assert np.allclose(trg1_trg2, [rg1, rg2])
