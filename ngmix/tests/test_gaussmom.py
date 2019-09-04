import galsim
import numpy as np
import pytest

from ngmix import Jacobian
from ngmix.gaussmom import GaussMom
from ngmix import Observation
from ngmix.gexceptions import GMixRangeError
from ngmix.moments import fwhm_to_T
from ngmix.shape import e1e2_to_g1g2


@pytest.mark.parametrize('weight_fac', [1, 1e5])
@pytest.mark.parametrize('wcs_g1', [-0.5, 0, 0.2])
@pytest.mark.parametrize('wcs_g2', [-0.2, 0, 0.5])
@pytest.mark.parametrize('g1_true', [-0.1, 0, 0.2])
@pytest.mark.parametrize('g2_true', [-0.2, 0, 0.1])
def test_admom_smoke(g1_true, g2_true, wcs_g1, wcs_g2, weight_fac):
    rng = np.random.RandomState(seed=100)

    fwhm = 0.9
    image_size = 107
    cen = (image_size - 1)/2
    gs_wcs = galsim.ShearWCS(
        0.125, galsim.Shear(g1=wcs_g1, g2=wcs_g2)).jacobian()

    obj = galsim.Gaussian(
        fwhm=fwhm
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
            method='no_pixel',
            dtype=np.float64).array

        jac = Jacobian(
            y=cen + xy.y, x=cen + xy.x,
            dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
            dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

        _im = im + (rng.normal(size=im.shape) * noise)
        obs = Observation(
            image=_im,
            weight=wgt,
            jacobian=jac)
        # use a huge weight so that we get the raw moments back out
        fitter = GaussMom(obs, fwhm * weight_fac, rng=rng)
        try:
            fitter.go()
            res = fitter.get_result()
            if res['flags'] == 0:
                if weight_fac > 1:
                    _g1, _g2 = e1e2_to_g1g2(res['e'][0], res['e'][1])
                else:
                    _g1, _g2 = res['e'][0], res['e'][1]
                g1arr.append(_g1)
                g2arr.append(_g2)
                Tarr.append(res['pars'][4])
        except GMixRangeError:
            pass

    g1 = np.mean(g1arr)
    g2 = np.mean(g2arr)

    gtol = 1e-9
    assert np.abs(g1 - g1_true) < gtol, (g1, np.std(g1arr)/np.sqrt(len(g1arr)))
    assert np.abs(g2 - g2_true) < gtol, (g2, np.std(g2arr)/np.sqrt(len(g2arr)))

    # T test should only pass when the weight function is constant so
    # weight_fac needs to be rally big
    if g1_true == 0 and g2_true == 0 and weight_fac > 1:
        T = np.mean(Tarr)
        assert np.abs(T - fwhm_to_T(fwhm)) < 1e-6
