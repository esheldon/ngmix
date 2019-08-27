import galsim
import numpy as np
import pytest

from ngmix.admom import Admom
from ngmix import Observation, DiagonalJacobian
from ngmix.moments import fwhm_to_T


@pytest.mark.parametrize('g1_true,g2_true', [
    (0, 0),
    (0.1, -0.2),
    (-0.1, 0.2)])
def test_admom_smoke(g1_true, g2_true):
    rng = np.random.RandomState(seed=10)

    scale = 0.25
    im = galsim.Gaussian(
        fwhm=0.9
    ).shear(
        g1=g1_true, g2=g2_true
    ).withFlux(
        400
    ).drawImage(
        nx=53,
        ny=53,
        scale=scale)
    im = im.array

    noise = np.sqrt(np.sum(im**2)/5e2)

    wgt = np.ones_like(im) / noise**2

    g1arr = []
    g2arr = []
    Tarr = []
    for _ in range(100):
        _im = im + (rng.normal(size=im.shape) * noise)
        obs = Observation(
            image=_im / (scale**2),
            weight=wgt * (scale**4),
            jacobian=DiagonalJacobian(row=26, col=26, scale=scale))
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
