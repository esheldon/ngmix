import numpy as np
import pytest

import ngmix


@pytest.mark.parametrize('model', ['gauss', 'exp', 'dev'])
@pytest.mark.parametrize('weight_fwhm', [2.5, 1000.0])
def test_gaussap_simple_smoke(model, weight_fwhm):
    rng = np.random.RandomState(seed=31415)

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

    mask = np.ones(nobj, dtype=bool)
    ibad = 5
    mask[ibad] = False
    gap_fluxes, flags = ngmix.gaussap.get_gaussap_flux(
        pars,
        model,
        weight_fwhm,
        mask=mask,
    )

    w, = np.where(flags[:, 0] == 0)
    assert flags[ibad, 0] != 0
    assert w.size == nobj-1

    bstart = 5
    assert gap_fluxes.shape == pars[:, bstart:bstart+nband].shape

    if weight_fwhm > 100:
        # for large weight functions we should recover the original flux
        assert np.all(np.abs(pars[w, bstart:bstart+nband] - gap_fluxes[w]) < 0.01)
    else:
        # for smaller weight functions we just verify the flux is smaller
        assert np.all(gap_fluxes[w] < pars[w, bstart:bstart+nband])


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

    bstart = 5
    assert gap_fluxes.shape == pars[:, bstart:bstart+nband].shape

    if weight_fwhm > 100:
        # for large weight functions we should recover the original flux
        assert np.all(np.abs(pars[:, bstart:bstart+nband] - gap_fluxes) < 0.01)
    else:
        # for smaller weight functions we just verify the flux is smaller
        assert np.all(gap_fluxes < pars[:, bstart:bstart+nband])


def test_gaussap_bdf_smoke():
    rng = np.random.RandomState(seed=314)

    weight_fwhm = 2.5
    nobj = 10
    nband = 5

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

    bstart = 6
    assert gap_fluxes.shape == pars[:, bstart:bstart+nband].shape

    if weight_fwhm > 100:
        # for large weight functions we should recover the original flux
        assert np.all(np.abs(pars[:, bstart:bstart+nband] - gap_fluxes) < 0.01)
    else:
        # for smaller weight functions we just verify the flux is smaller
        assert np.all(gap_fluxes < pars[:, bstart:bstart+nband])


@pytest.mark.parametrize('nobj', [1, 10])
def test_gaussap_corner_cases(nobj):
    rng = np.random.RandomState(seed=31415)

    model = 'exp'
    weight_fwhm = 2.0

    pars = np.zeros((nobj, 6))

    pars[:, 0] = rng.uniform(low=-0.1, high=0.1, size=nobj)  # cen1
    pars[:, 1] = rng.uniform(low=-0.1, high=0.1, size=nobj)  # cen2
    pars[:, 2] = rng.uniform(low=-0.1, high=0.1, size=nobj)  # g1
    pars[:, 3] = rng.uniform(low=-0.1, high=0.1, size=nobj)  # g2
    pars[:, 4] = rng.uniform(low=0.2, high=1.0, size=nobj)  # T
    pars[:, 5] = rng.uniform(low=100, high=200, size=nobj)  # flux

    ibad = 0
    pars[ibad, 2] = 999

    if nobj == 1:
        pars = pars[0, :]

    gap_fluxes, flags = ngmix.gaussap.get_gaussap_flux(pars, model, weight_fwhm)
    assert flags[ibad] != 0
