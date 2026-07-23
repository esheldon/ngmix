"""
tests for the composite exp+dev ('bdf') model in the pre-psf
adaptive moments fitter, and the fixcen option

The scenes are rendered from the same ngmix gaussian expansions the
model fits, so the model is exact and the fits must recover the
truth at high s2n
"""
import numpy as np
import pytest

import ngmix
from ngmix.moments import fwhm_to_T
from ngmix.prepsfadmom import run_prepsf_admom, PAdmomFitter

SCALE = 0.2
DIM = 48
PSF_FWHM = 0.8
FWHM_SMOOTH = 1.05 * PSF_FWHM


def g_to_e(g1, g2):
    gsq = g1 * g1 + g2 * g2
    fac = 2.0 / (1.0 + gsq)
    return fac * g1, fac * g2


def make_bdf_obs(
    fracdev, T, g1, g2, flux, noise_sigma, TdByTe=1.0, rng=None,
    voff=0.0, uoff=0.0, dim=DIM,
):
    """
    exp + dev with shared center and shape, dev T equal to TdByTe
    times the exp T, convolved with a round gaussian psf; the bdf
    model represents this data exactly
    """
    cen = (dim - 1) / 2
    jacobian = ngmix.DiagonalJacobian(scale=SCALE, row=cen, col=cen)

    psf_T = fwhm_to_T(PSF_FWHM)
    psf_gm = ngmix.GMixModel([0, 0, 0, 0, psf_T, 1.0], 'gauss')
    psf_im = psf_gm.make_image((dim, dim), jacobian=jacobian)
    psf_obs = ngmix.Observation(
        psf_im, weight=np.ones((dim, dim)) * 1.0e12,
        jacobian=jacobian,
    )

    im = np.zeros((dim, dim))
    for name, f, Tc in (
        ('exp', (1 - fracdev) * flux, T),
        ('dev', fracdev * flux, TdByTe * T),
    ):
        if f == 0:
            continue
        gm0 = ngmix.GMixModel([voff, uoff, g1, g2, Tc, f], name)
        gm = gm0.convolve(psf_gm)
        im += gm.make_image((dim, dim), jacobian=jacobian)

    if rng is not None:
        im = im + rng.normal(scale=noise_sigma, size=im.shape)

    return ngmix.Observation(
        im, weight=np.ones((dim, dim)) / noise_sigma ** 2,
        jacobian=jacobian, psf=psf_obs,
    )


def test_model_spec_api():
    """
    string and dict specs are equivalent; bad specs raise clearly
    """
    f1 = PAdmomFitter(model='exp')
    f2 = PAdmomFitter(model={'type': 'exp'})
    assert f1.model == f2.model == 'exp'
    assert f1.TdByTe is None

    f3 = PAdmomFitter(model={'type': 'bdf', 'TdByTe': 1.5})
    assert f3.model == 'bdf'
    assert f3.TdByTe == 1.5

    with pytest.raises(ValueError, match='TdByTe'):
        PAdmomFitter(model={'type': 'bdf'})
    with pytest.raises(ValueError, match='TdByTe'):
        PAdmomFitter(model='bdf')
    with pytest.raises(ValueError, match='positive'):
        PAdmomFitter(model={'type': 'bdf', 'TdByTe': -1.0})
    with pytest.raises(ValueError, match='unexpected'):
        PAdmomFitter(model={'type': 'exp', 'TdByTe': 1.0})
    with pytest.raises(ValueError, match='bad model'):
        PAdmomFitter(model={'type': 'bulgedisk', 'TdByTe': 1.0})
    with pytest.raises(ValueError, match="'type'"):
        PAdmomFitter(model={'TdByTe': 1.0})

    f4 = PAdmomFitter(model={
        'type': 'bdf', 'TdByTe': 1.0,
        'fracdev0': 0.0, 'fracdev_sigma0': 0.3,
    })
    assert f4.fracdev_shrink == (0.0, 0.3)
    assert f3.fracdev_shrink is None

    with pytest.raises(ValueError, match='shrinkage'):
        PAdmomFitter(model={
            'type': 'bdf', 'TdByTe': 1.0, 'fracdev0': 0.0,
        })
    with pytest.raises(ValueError, match='shrinkage'):
        PAdmomFitter(model={
            'type': 'bdf', 'TdByTe': 1.0, 'fracdev_sigma0': 0.3,
        })
    with pytest.raises(ValueError, match='non-negative'):
        PAdmomFitter(model={
            'type': 'bdf', 'TdByTe': 1.0,
            'fracdev0': 0.0, 'fracdev_sigma0': -0.1,
        })


def test_bdf_shrinkage():
    """
    shrinkage is negligible for bright objects, exact freeze at
    sigma0=0, and the raw split is reported alongside
    """
    fracdev = 0.3
    obs = make_bdf_obs(fracdev, 0.5, 0.08, -0.04, 100.0, 1.0e-6)

    # bright: the prior has negligible weight
    res = run_prepsf_admom(
        obs, model={
            'type': 'bdf', 'TdByTe': 1.0,
            'fracdev0': 0.0, 'fracdev_sigma0': 0.3,
        },
        fwhm_smooth=FWHM_SMOOTH, ap_rad=0.0,
        rng=np.random.RandomState(5),
    )
    assert res['flags'] == 0
    assert abs(res['fracdev'] - fracdev) < 1.0e-3
    assert abs(res['fracdev_gls'] - fracdev) < 1.0e-3
    assert res['fracdev_gls_err'] > 0

    # hard freeze: the model split is pinned.  The raw split is
    # conditional on the structure, which converged under the
    # frozen (wrong) mixture with T compensating, so it is pulled
    # toward the truth but does not fully recover it
    res = run_prepsf_admom(
        obs, model={
            'type': 'bdf', 'TdByTe': 1.0,
            'fracdev0': 0.1, 'fracdev_sigma0': 0.0,
        },
        fwhm_smooth=FWHM_SMOOTH, ap_rad=0.0,
        rng=np.random.RandomState(5),
    )
    assert res['flags'] == 0
    assert res['fracdev'] == 0.1
    assert res['fracdev_gls'] > 0.12


@pytest.mark.parametrize('TdByTe', [1.0, 2.0])
@pytest.mark.parametrize('offset', [(0.0, 0.0), (0.31, -0.22)])
def test_bdf_exactness(TdByTe, offset):
    """
    noiseless model-consistent recovery of structure, split and
    per-band fluxes, centered and offset
    """
    fracdev = 0.3
    T = 0.5
    g1, g2 = 0.08, -0.04
    flux = 100.0
    voff, uoff = offset

    obs = make_bdf_obs(
        fracdev, T, g1, g2, flux, 1.0e-6, TdByTe=TdByTe,
        voff=voff, uoff=uoff,
    )
    res = run_prepsf_admom(
        obs, model={'type': 'bdf', 'TdByTe': TdByTe},
        fwhm_smooth=FWHM_SMOOTH, ap_rad=0.0,
        rng=np.random.RandomState(5),
    )
    assert res['flags'] == 0

    e1t, e2t = g_to_e(g1, g2)
    assert abs(res['cen'][0] - voff) < 1.0e-3
    assert abs(res['cen'][1] - uoff) < 1.0e-3
    assert abs(res['T'] / T - 1) < 1.0e-3
    assert abs(res['e1'] - e1t) < 1.0e-3
    assert abs(res['e2'] - e2t) < 1.0e-3
    assert abs(res['fracdev'] - fracdev) < 1.0e-3
    assert abs(res['flux'] / flux - 1) < 1.0e-3
    assert abs(
        res['flux_exp'] / ((1 - fracdev) * flux) - 1
    ) < 2.0e-3
    assert abs(res['flux_dev'] / (fracdev * flux) - 1) < 5.0e-3
    assert res['TdByTe'] == TdByTe


@pytest.mark.parametrize('fracdev', [0.0, 0.3, 0.7])
def test_bdf_fracdev_scan(fracdev):
    """
    the split is recovered across the fracdev range
    """
    obs = make_bdf_obs(fracdev, 0.5, 0.08, -0.04, 100.0, 1.0e-6)
    res = run_prepsf_admom(
        obs, model={'type': 'bdf', 'TdByTe': 1.0},
        fwhm_smooth=FWHM_SMOOTH, ap_rad=0.0,
        rng=np.random.RandomState(5),
    )
    assert res['flags'] == 0
    assert abs(res['fracdev'] - fracdev) < 2.0e-3
    assert abs(res['T'] / 0.5 - 1) < 2.0e-3


def test_bdf_multiband_split():
    """
    the flux split is per band: bands rendered with different
    fracdev recover their own component fluxes (bulge and disk
    colors) while sharing the structure
    """
    T = 0.5
    g1, g2 = 0.08, -0.04
    fds = (0.2, 0.6)
    flux = 100.0

    mbobs = ngmix.MultiBandObsList()
    for fd in fds:
        obslist = ngmix.ObsList()
        obslist.append(
            make_bdf_obs(fd, T, g1, g2, flux, 1.0e-6)
        )
        mbobs.append(obslist)

    res = run_prepsf_admom(
        mbobs, model={'type': 'bdf', 'TdByTe': 1.0},
        fwhm_smooth=FWHM_SMOOTH, ap_rad=0.0,
        rng=np.random.RandomState(5),
    )
    assert res['flags'] == 0
    assert abs(res['T'] / T - 1) < 2.0e-3
    for band, fd in enumerate(fds):
        assert abs(
            res['flux_exp'][band] / ((1 - fd) * flux) - 1
        ) < 5.0e-3
        assert abs(
            res['flux_dev'][band] / (fd * flux) - 1
        ) < 1.0e-2
        assert abs(res['flux'][band] / flux - 1) < 2.0e-3


def test_fixcen():
    """
    with fixcen the center stays at the guess for every model, and
    a centered fit still recovers the truth
    """
    fracdev = 0.3
    obs = make_bdf_obs(fracdev, 0.5, 0.08, -0.04, 100.0, 1.0e-6)

    for model in ('gauss', 'exp', {'type': 'bdf', 'TdByTe': 1.0}):
        guess = ngmix.GMixModel([0, 0, 0, 0, 0.3, 1.0], 'gauss')
        res = run_prepsf_admom(
            obs, model=model, fwhm_smooth=FWHM_SMOOTH, ap_rad=0.0,
            fixcen=True, guess=guess,
            rng=np.random.RandomState(5),
        )
        assert res['flags'] == 0
        assert res['cen'][0] == 0.0
        assert res['cen'][1] == 0.0

    res = run_prepsf_admom(
        obs, model={'type': 'bdf', 'TdByTe': 1.0},
        fwhm_smooth=FWHM_SMOOTH, ap_rad=0.0, fixcen=True,
        guess=ngmix.GMixModel([0, 0, 0, 0, 0.3, 1.0], 'gauss'),
        rng=np.random.RandomState(5),
    )
    assert abs(res['fracdev'] - fracdev) < 1.0e-3
    assert abs(res['T'] / 0.5 - 1) < 1.0e-3


def test_bdf_noise():
    """
    noisy fits converge without failures, the split is unbiased,
    and the reported flux errors are within the documented
    conditional-underprediction bounds
    """
    rng = np.random.RandomState(31)
    noise_sigma = 0.15

    vals = []
    perr = []
    nfail = 0
    for i in range(50):
        obs = make_bdf_obs(
            0.3, 0.5, 0.08, -0.04, 100.0, noise_sigma, rng=rng,
            voff=rng.uniform(-0.1, 0.1), uoff=rng.uniform(-0.1, 0.1),
        )
        res = run_prepsf_admom(
            obs, model={'type': 'bdf', 'TdByTe': 1.0},
            fwhm_smooth=FWHM_SMOOTH, ap_rad=0.0,
            maxiter=500, rng=np.random.RandomState(i),
        )
        if res['flags'] != 0:
            nfail += 1
            continue
        vals.append((res['fracdev'], res['flux'], res['T']))
        perr.append(res['flux_err'])

    assert nfail <= 2
    vals = np.array(vals)

    # unbiased within a few sigma of the mean
    fd_err = vals[:, 0].std() / np.sqrt(vals.shape[0])
    assert abs(vals[:, 0].mean() - 0.3) < 4 * fd_err
    flux_err = vals[:, 1].std() / np.sqrt(vals.shape[0])
    assert abs(vals[:, 1].mean() - 100.0) < 4 * flux_err

    # the sandwich flux errors are conditional on the split, so
    # they underpredict the total scatter; they must still be the
    # right order
    ratio = vals[:, 1].std() / np.mean(perr)
    assert 0.8 < ratio < 4.0


def test_gauss_entries():
    """
    the gauss (converged-weight) shape entries are filled for
    every model and are identical across models on the same data,
    since the weight iteration is model independent; for the
    gauss model they match the primary shape outputs
    """
    obs = make_bdf_obs(0.3, 0.5, 0.08, -0.04, 100.0, 0.1,
                       rng=np.random.RandomState(3))

    results = {}
    for name, model in (
        ('gauss', 'gauss'),
        ('exp', 'exp'),
        ('bdf', {'type': 'bdf', 'TdByTe': 1.0,
                 'fracdev0': 0.0, 'fracdev_sigma0': 0.3}),
    ):
        results[name] = run_prepsf_admom(
            obs, model=model, fwhm_smooth=FWHM_SMOOTH, ap_rad=0.0,
            rng=np.random.RandomState(5),
        )
        res = results[name]
        assert res['flags'] == 0
        assert res['gauss_e_flags'] == 0
        assert np.isfinite(res['gauss_e1'])
        assert res['gauss_e1err'] > 0
        assert res['gauss_T'] > 0
        assert res['gauss_T_err'] > 0

    rg = results['gauss']
    assert abs(rg['gauss_e1'] - rg['e1']) < 1.0e-9
    assert abs(rg['gauss_e2'] - rg['e2']) < 1.0e-9
    assert abs(rg['gauss_T'] - rg['T']) < 1.0e-9

    for name in ('exp', 'bdf'):
        r = results[name]
        assert abs(r['gauss_e1'] - rg['gauss_e1']) < 1.0e-4
        assert abs(r['gauss_e2'] - rg['gauss_e2']) < 1.0e-4
        assert abs(r['gauss_T'] / rg['gauss_T'] - 1) < 1.0e-3


def test_mixture_regression():
    """
    the exp path through the refactored spec/validity plumbing
    still recovers a pure exp scene
    """
    obs = make_bdf_obs(0.0, 0.5, 0.08, -0.04, 100.0, 1.0e-6)
    res = run_prepsf_admom(
        obs, model='exp', fwhm_smooth=FWHM_SMOOTH, ap_rad=0.0,
        rng=np.random.RandomState(5),
    )
    e1t, e2t = g_to_e(0.08, -0.04)
    assert res['flags'] == 0
    assert abs(res['T'] / 0.5 - 1) < 1.0e-3
    assert abs(res['e1'] - e1t) < 1.0e-3
    assert abs(res['flux'] / 100.0 - 1) < 1.0e-3
