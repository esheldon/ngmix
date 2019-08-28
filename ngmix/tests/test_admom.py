import galsim
import numpy as np
import pytest

from ngmix import Jacobian
from ngmix.admom import Admom
from ngmix import Observation
from ngmix.moments import fwhm_to_T
from ngmix.gexceptions import GMixRangeError

GTOL = 1e-3


@pytest.mark.parametrize('s2n', [1e3, 1e16])
@pytest.mark.parametrize('wcs_g1', [-0.005, 0, 0.002])
@pytest.mark.parametrize('wcs_g2', [-0.002, 0, 0.005])
@pytest.mark.parametrize('g1_true', [-0.1, 0, 0.2])
@pytest.mark.parametrize('g2_true', [-0.2, 0, 0.1])
def test_admom_smoke(g1_true, g2_true, wcs_g1, wcs_g2, s2n):
    jc = galsim.ShearWCS(
        0.25, galsim.Shear(g1=wcs_g1, g2=wcs_g2)).jacobian()
    jac = Jacobian(
        y=16, x=16,
        dudx=jc.dudx, dudy=jc.dudy, dvdx=jc.dvdx, dvdy=jc.dvdy)

    rng = np.random.RandomState(seed=100)

    gs_wcs = jac.get_galsim_wcs()
    im = galsim.Gaussian(
        fwhm=0.9
    ).shear(
        g1=g1_true, g2=g2_true
    ).withFlux(
        400
    ).drawImage(
        nx=53,
        ny=53,
        wcs=gs_wcs,
        method='no_pixel')
    im = im.array

    noise = np.sqrt(np.sum(im**2))/s2n

    wgt = np.ones_like(im) / noise**2

    g1arr = []
    g2arr = []
    Tarr = []
    for _ in range(50):
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
    assert np.abs(g1 - g1_true) < max(GTOL, g1_err * 5)
    assert np.abs(g2 - g2_true) < max(GTOL, g2_err * 5)

    if g1 == 0 and g2 == 0:
        T = np.mean(Tarr)
        T_err = np.std(Tarr) / np.sqrt(len(Tarr))
        assert np.abs(T - fwhm_to_T(0.9)) < T_err * 5
