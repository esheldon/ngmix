import numpy as np
import ngmix


def test_kde():
    rng = np.random.RandomState(38234)

    data = rng.normal(size=100)
    kde = ngmix.priors.KDE(data=data, kde_factor='scott', rng=rng)

    assert np.isscalar(kde.sample())
    nrand = 5
    assert kde.sample(nrand).shape == (nrand, )

    ndim = 2
    data = rng.normal(size=(100, ndim))
    kde = ngmix.priors.KDE(data=data, kde_factor='scott', rng=rng)
    assert kde.sample().shape == (ndim, )
    nrand = 5
    assert kde.sample(nrand).shape == (nrand, ndim)
