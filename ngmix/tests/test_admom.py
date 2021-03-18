import copy
import galsim
import numpy as np
import pytest

import ngmix
from ngmix import Jacobian
from ngmix.admom import run_admom
from ngmix import Observation
from ngmix.moments import fwhm_to_T
from ngmix.gexceptions import GMixRangeError


@pytest.mark.parametrize('wcs_g1', [-0.5, 0, 0.2])
@pytest.mark.parametrize('wcs_g2', [-0.2, 0, 0.5])
@pytest.mark.parametrize('g1_true', [-0.1, 0, 0.2])
@pytest.mark.parametrize('g2_true', [-0.2, 0, 0.1])
def test_admom_smoke(g1_true, g2_true, wcs_g1, wcs_g2):
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
            dtype=np.float64,
        ).array

        jac = Jacobian(
            y=cen + xy.y, x=cen + xy.x,
            dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
            dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

        _im = im + (rng.normal(size=im.shape) * noise)
        obs = Observation(
            image=_im,
            weight=wgt,
            jacobian=jac)
        try:
            Tguess = fwhm_to_T(fwhm) + rng.normal()*0.01
            res = run_admom(obs=obs, guess=Tguess)
            if res['flags'] == 0:
                gm = res.get_gmix()
                _g1, _g2, _T = gm.get_g1g2T()

                g1arr.append(_g1)
                g2arr.append(_g2)
                Tarr.append(_T)

                fim = res.make_image()
                assert fim.shape == im.shape

        except GMixRangeError:
            pass

        res['flags'] = 5
        with pytest.raises(RuntimeError):
            res.make_image()
        with pytest.raises(RuntimeError):
            res.get_gmix()

    g1 = np.mean(g1arr)
    g2 = np.mean(g2arr)
    gtol = 1e-6
    assert np.abs(g1 - g1_true) < gtol, (g1, np.std(g1arr)/np.sqrt(len(g1arr)))
    assert np.abs(g2 - g2_true) < gtol, (g2, np.std(g2arr)/np.sqrt(len(g2arr)))

    if g1_true == 0 and g2_true == 0:
        T = np.mean(Tarr)
        assert np.abs(T - fwhm_to_T(fwhm)) < 1e-6

    with pytest.raises(ValueError):
        _ = run_admom(None, None)

    # cover some branches
    tres = copy.deepcopy(res)

    tres['flags'] = 0
    tres['sums_cov'][:, :] = np.nan
    tres = ngmix.admom.get_result(tres)
    assert tres['e1err'] == 9999.0

    tres = copy.deepcopy(res)
    tres['flags'] = 0
    tres['pars'][4] = -1
    tres = ngmix.admom.get_result(tres)
    assert tres['flags'] == 0x8
