import galsim
import numpy as np
import pytest

from ngmix import Jacobian
from ngmix.admom import Admom
from ngmix import Observation
from ngmix.moments import fwhm_to_T


@pytest.mark.parametrize('jac', [
    Jacobian(y=26, x=26, dudx=0.25, dudy=0, dvdx=0, dvdy=0.25),
    Jacobian(y=26, x=26, dudx=0.25, dudy=0, dvdx=0, dvdy=0.3),
    Jacobian(y=26, x=26, dudx=0.25, dudy=0.01, dvdx=-0.02, dvdy=0.3)])
@pytest.mark.parametrize('g1_true,g2_true', [
    (0, 0),
    (0.1, -0.2),
    (-0.1, 0.2)])
def test_admom_smoke(g1_true, g2_true, jac):
    rng = np.random.RandomState(seed=10)

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
        wcs=gs_wcs)
    im = im.array

    noise = np.sqrt(np.sum(im**2)/5e2)

    wgt = np.ones_like(im) / noise**2

    g1arr = []
    g2arr = []
    Tarr = []
    for _ in range(100):
        _im = im + (rng.normal(size=im.shape) * noise)
        obs = Observation(
            image=_im,
            weight=wgt,
            jacobian=jac)
        fitter = Admom(obs, rng=rng)
        fitter.go(1)
        gm = fitter.get_gmix()
        _g1, _g2, _T = gm.get_g1g2T()

        g1arr.append(_g1)
        g2arr.append(_g2)
        Tarr.append(_T)

    g1 = np.mean(g1arr)
    g1_err = np.std(g1arr) / np.sqrt(len(g1arr))
    g2 = np.mean(g2arr)
    g2_err = np.std(g2arr) / np.sqrt(len(g2arr))
    assert np.abs(g1 - g1_true) < g1_err * 5
    assert np.abs(g2 - g2_true) < g2_err * 5

    if g1 == 0 and g2 == 0:
        T = np.mean(Tarr)
        T_err = np.std(Tarr) / np.sqrt(len(Tarr))
        assert np.abs(T - fwhm_to_T(0.9)) < T_err * 5
