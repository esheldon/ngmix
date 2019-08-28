import galsim
import numpy as np
import pytest

from ngmix import Jacobian
from ngmix.admom import Admom
from ngmix import Observation
from ngmix.moments import fwhm_to_T
from ngmix.gexceptions import GMixRangeError

GTOL = 7e-5


@pytest.mark.parametrize('wcs_g1', [-0.05, 0, 0.02])
@pytest.mark.parametrize('wcs_g2', [-0.02, 0, 0.05])
@pytest.mark.parametrize('g1_true', [-0.1, 0, 0.2])
@pytest.mark.parametrize('g2_true', [-0.2, 0, 0.1])
def test_admom_smoke(g1_true, g2_true, wcs_g1, wcs_g2):
    rng = np.random.RandomState(seed=100)

    image_size = 53
    cen = (image_size - 1)/2
    gs_wcs = galsim.ShearWCS(
        0.25, galsim.Shear(g1=wcs_g1, g2=wcs_g2)).jacobian()

    obj = galsim.Gaussian(
        fwhm=0.9
    ).shear(
        g1=g1_true, g2=g2_true
    ).withFlux(
        400)
    im = obj.drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs,
        method='no_pixel').array
    noise = np.sqrt(np.sum(im**2)) / 1e18
    wgt = np.ones_like(im) / noise**2
    scale = np.sqrt(gs_wcs.pixelArea())

    g1arr = []
    g2arr = []
    Tarr = []
    for _ in range(50):
        shift = rng.uniform(low=-scale/2, high=scale/2, size=2)
        xy = gs_wcs.toImage(galsim.PositionD(shift))

        im = obj.shift(
            dx=shift[0], dy=shift[1]
        ).drawImage(
            nx=image_size,
            ny=image_size,
            wcs=gs_wcs,
            method='no_pixel').array

        jac = Jacobian(
            y=cen + xy.y, x=cen + xy.x,
            dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
            dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

        _im = im + (rng.normal(size=im.shape) * noise)
        obs = Observation(
            image=_im,
            weight=wgt,
            jacobian=jac)
        fitter = Admom(obs, rng=rng)
        try:
            fitter.go(1)
            gm = fitter.get_gmix()
            _g1, _g2, _T = gm.get_g1g2T()

            g1arr.append(_g1)
            g2arr.append(_g2)
            Tarr.append(_T)
        except GMixRangeError:
            pass

    g1 = np.mean(g1arr)
    g1_err = np.std(g1arr) / np.sqrt(len(g1arr))
    g2 = np.mean(g2arr)
    g2_err = np.std(g2arr) / np.sqrt(len(g2arr))
    assert np.abs(g1 - g1_true) < GTOL
    assert np.abs(g1 - g1_true) < g1_err * 5
    assert np.abs(g2 - g2_true) < GTOL
    assert np.abs(g2 - g2_true) < g2_err * 5

    if g1 == 0 and g2 == 0:
        T = np.mean(Tarr)
        T_err = np.std(Tarr) / np.sqrt(len(Tarr))
        assert np.abs(T - fwhm_to_T(0.9)) < T_err * 5
