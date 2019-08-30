import numpy as np
import pytest

import ngmix

# @pytest.mark.parametrize('model', ['gauss', 'exp', 'bdf', 'cm'])
@pytest.mark.parametrize('model', ['gauss', 'exp', 'dev'])
def test_gaussap_simple_smoke(model):
    rng = np.random.RandomState(seed=31415)

    weight_fwhm = 2.5
    nobj = 10
    nband = 5

    pars = np.zeros((nobj, 5+nband))

    for i in range(nobj):
        this_pars = [
            rng.uniform(low=-0.1, high=0.1),  # cen1
            rng.uniform(low=-0.1, high=0.1),  # cen2
            rng.uniform(low=-0.1, high=0.1),  # g1
            rng.uniform(low=-0.1, high=0.1),  # g2
            rng.uniform(low=0.2, high=1.0),   # T
        ]

        for band in range(nband):
            this_pars += [
                rng.uniform(low=100, high=200),
            ]

        pars[i, :] = this_pars

    gap_fluxes, flags = ngmix.gaussap.get_gaussap_flux(
        pars,
        model,
        weight_fwhm,
    )

    assert np.all(flags == 0)


def test_gaussap_cm_smoke():
    rng = np.random.RandomState(seed=314)

    weight_fwhm = 2.5
    nobj = 10
    nband = 5

    fracdev = rng.uniform(low=0.1, high=0.8, size=nobj)
    TdByTe = rng.uniform(low=0.5, high=1.5, size=nobj)
    pars = np.zeros((nobj, 5+nband))

    for i in range(nobj):
        this_pars = [
            rng.uniform(low=-0.1, high=0.1),  # cen1
            rng.uniform(low=-0.1, high=0.1),  # cen2
            rng.uniform(low=-0.1, high=0.1),  # g1
            rng.uniform(low=-0.1, high=0.1),  # g2
            rng.uniform(low=0.2, high=1.0),   # T
        ]

        for band in range(nband):
            this_pars += [
                rng.uniform(low=100, high=200),
            ]

        pars[i, :] = this_pars

    gap_fluxes, flags = ngmix.gaussap.get_gaussap_flux(
        pars,
        'cm',
        weight_fwhm,
        fracdev=fracdev,
        TdByTe=TdByTe,
    )

    assert np.all(flags == 0)


def test_gaussap_bdf_smoke():
    rng = np.random.RandomState(seed=314)

    weight_fwhm = 2.5
    nobj = 10
    nband = 5

    fracdev = rng.uniform(low=0.1, high=0.8, size=nobj)
    TdByTe = rng.uniform(low=0.5, high=1.5, size=nobj)
    pars = np.zeros((nobj, 6+nband))

    for i in range(nobj):
        this_pars = [
            rng.uniform(low=-0.1, high=0.1),  # cen1
            rng.uniform(low=-0.1, high=0.1),  # cen2
            rng.uniform(low=-0.1, high=0.1),  # g1
            rng.uniform(low=-0.1, high=0.1),  # g2
            rng.uniform(low=0.2, high=1.0),   # T
            rng.uniform(low=0.1, high=0.9),   # fracdev
        ]

        for band in range(nband):
            this_pars += [
                rng.uniform(low=100, high=200),
            ]

        pars[i, :] = this_pars

    gap_fluxes, flags = ngmix.gaussap.get_gaussap_flux(
        pars,
        'bdf',
        weight_fwhm,
    )

    assert np.all(flags == 0)
