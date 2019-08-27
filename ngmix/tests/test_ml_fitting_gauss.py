import galsim
import numpy as np
import pytest

from ngmix.fitting import LMSimple
from ngmix import Jacobian
from ngmix import Observation
from ngmix.moments import fwhm_to_T

GTOL = 3e-4


@pytest.mark.parametrize('s2n', [1e2, 1e3, 1e9])
@pytest.mark.parametrize('jac', [
    Jacobian(y=16, x=16, dudx=0.25, dudy=0, dvdx=0, dvdy=0.25),
    Jacobian(y=26, x=26, dudx=0.25, dudy=0, dvdx=0, dvdy=0.3),
    Jacobian(y=26, x=26, dudx=0.25, dudy=0.01, dvdx=-0.02, dvdy=0.3)
])
@pytest.mark.parametrize('g1_true,g2_true', [
    (0, 0),
    (0.1, -0.2),
    (-0.1, 0.2)
])
def test_ml_fitting_gauss_smoke(g1_true, g2_true, jac, s2n):
    rng = np.random.RandomState(seed=10)

    gs_wcs = jac.get_galsim_wcs()
    im = galsim.Gaussian(
        fwhm=0.9
    ).shear(
        g1=g1_true, g2=g2_true
    ).withFlux(
        400
    ).drawImage(
        nx=33,
        ny=33,
        wcs=gs_wcs,
        method='no_pixel')
    im = im.array

    noise = np.sqrt(np.sum(im**2))/s2n

    wgt = np.ones_like(im) / noise**2

    guess = np.ones(6) * 0.1
    guess[0] = 0
    guess[1] = 0
    guess[2] = g1_true
    guess[3] = g2_true
    guess[4] = 2
    guess[5] = 400

    g1arr = []
    g2arr = []
    Tarr = []
    for _ in range(100):
        _im = im + (rng.normal(size=im.shape) * noise)
        obs = Observation(
            image=_im,
            weight=wgt,
            jacobian=jac)
        fitter = LMSimple(obs, 'gauss')
        fitter.go(guess + rng.normal(size=6) * 0.01)
        res = fitter.get_result()
        if res['flags'] == 0:
            _g1, _g2, _T = res['g'][0], res['g'][1], res['pars'][4]
            g1arr.append(_g1)
            g2arr.append(_g2)
            Tarr.append(_T)

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
