import pytest
import numpy as np
import ngmix
import tempfile
import os

FRAC_TOL = 0.001


def _exercise_gd(gd, data):
    gd.get_prob_scalar(data[0])
    gd.get_lnprob_scalar(data[0])

    gd.get_prob_array(data)
    gd.get_lnprob_array(data)

    _ = gd.sample()
    _ = gd.sample(n=10)


@pytest.mark.parametrize('ndim', [1, 2, 3])
def test_gmix_ndim_smoke(ndim):
    rng = np.random.RandomState(888)

    covar = 1
    mean = 1
    num = 10000

    data = rng.multivariate_normal(
        mean=[mean]*ndim,
        cov=np.diag([covar]*ndim),
        size=num,
    )

    gd = ngmix.gmix_ndim.GMixND(rng=rng)
    gd.fit(data, ngauss=1, min_covar=0.1)

    _exercise_gd(gd, data)

    newgd = ngmix.gmix_ndim.GMixND(
        rng=rng,
        weights=gd.weights,
        means=gd.means,
        covars=gd.covars,
    )
    _exercise_gd(newgd, data)

    with tempfile.TemporaryDirectory() as dir:
        fname = os.path.join(dir, 'blah.fits')
        gd.save_mixture(fname)
        gd.load_mixture(fname)

        _exercise_gd(gd, data)


@pytest.mark.parametrize('seed', [5, 10, 100])
def test_gmix_ndim(seed):
    rng = np.random.RandomState(seed)

    frac1 = 0.4
    frac2 = 0.6
    sigma1 = 1
    sigma2 = 2
    cen1 = 0
    cen2 = 10
    num = 1000000
    num1 = int(frac1 * num)
    num2 = int(frac2 * num)
    data1 = rng.normal(scale=sigma1, loc=cen1, size=num1)
    data2 = rng.normal(scale=sigma2, loc=cen2, size=num2)

    data = np.hstack((data1, data2))
    # import hickory
    # hickory.plot_hist(data, bins=100)

    gd = ngmix.gmix_ndim.GMixND(rng=rng)
    gd.fit(data, ngauss=2, min_covar=0.1)

    s = gd.weights.argsort()
    assert np.allclose(gd.weights[s], [frac1, frac2], atol=0.001)

    fitmeans = gd.means.ravel()
    s = fitmeans.argsort()
    assert np.allclose(fitmeans[s], [cen1, cen2], atol=0.005)

    fitsigmas = np.sqrt(gd.covars.ravel())
    s = fitsigmas.argsort()
    assert np.allclose(fitsigmas[s], [sigma1, sigma2], atol=0.07)
