import numpy as np
import galsim
import pytest

from ngmix.pixels import make_coords, make_pixels
from ngmix.jacobian import Jacobian


@pytest.mark.parametrize('x', [-31, 0.0, 45.1])
@pytest.mark.parametrize('y', [-45.1, 0.0, 31.6])
@pytest.mark.parametrize('wcs_g1', [-0.2, 0.0, 0.5])
@pytest.mark.parametrize('wcs_g2', [-0.5, 0.0, 0.2])
def test_coords_smoke(x, y, wcs_g1, wcs_g2):
    gs_wcs = galsim.ShearWCS(
        0.25, galsim.Shear(g1=wcs_g1, g2=wcs_g2)).jacobian()
    jac = Jacobian(
        y=y, x=x,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)
    dims = (13, 15)

    coords = make_coords(dims, jac)
    loc = 0
    for y in range(dims[0]):
        for x in range(dims[1]):
            v, u = jac(y, x)
            assert u == coords['u'][loc]
            assert v == coords['v'][loc]
            loc += 1


@pytest.mark.parametrize('ignore_zero_weight', [False, True])
@pytest.mark.parametrize('x', [-31, 0.0, 45.1])
@pytest.mark.parametrize('y', [-45.1, 0.0, 31.6])
@pytest.mark.parametrize('wcs_g1', [-0.2, 0.0, 0.5])
@pytest.mark.parametrize('wcs_g2', [-0.5, 0.0, 0.2])
def test_pixels_smoke(x, y, wcs_g1, wcs_g2, ignore_zero_weight):
    gs_wcs = galsim.ShearWCS(
        0.25, galsim.Shear(g1=wcs_g1, g2=wcs_g2)).jacobian()
    jac = Jacobian(
        y=y, x=x,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)
    dims = (13, 15)

    rng = np.random.RandomState(seed=11)
    image = rng.normal(size=dims)
    weight = np.exp(rng.normal(size=dims))

    weight[10, 9] = 0
    weight[8, 7] = 0

    pixels = make_pixels(
        image, weight, jac, ignore_zero_weight=ignore_zero_weight)

    found_zero = 0
    for i in range(len(pixels)):
        y, x = jac.get_rowcol(pixels['v'][i], pixels['u'][i])
        assert np.allclose(x, int(x + 0.5))
        assert np.allclose(y, int(y + 0.5))
        x = int(x + 0.5)
        y = int(y + 0.5)

        assert pixels['val'][i] == image[y, x]
        assert np.allclose(pixels['ierr'][i], np.sqrt(weight[y, x]))

        if x == 9 and y == 10:
            found_zero += 1
        if y == 8 and x == 7:
            found_zero += 1

    if ignore_zero_weight:
        assert found_zero == 0
    else:
        assert found_zero == 2
