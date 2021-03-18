import pytest
import numpy as np

from ngmix.em import fit_em
from ._sims import get_ngauss_obs

FRAC_TOL = 0.001


def randomize_gmix(rng, gmix, pixel_scale):
    gm_data = gmix.get_data()
    for gauss in gm_data:
        gauss["p"] *= rng.uniform(low=0.9, high=1.1)
        gauss["row"] += rng.uniform(low=-pixel_scale, high=pixel_scale)
        gauss["col"] += rng.uniform(low=-pixel_scale, high=pixel_scale)
        gauss["irr"] += 0.1 * pixel_scale**2 * rng.uniform(low=-1, high=1)
        gauss["irc"] += 0.1 * pixel_scale**2 * rng.uniform(low=-1, high=1)
        gauss["icc"] += 0.1 * pixel_scale**2 * rng.uniform(low=-1, high=1)


@pytest.mark.parametrize('noise', [0.0, 0.05])
def test_em_1gauss(noise):
    """
    see if we can recover the input with and without noise to high precision
    even with a bad guess

    Use ngmix to make the image to make sure there are
    no pixelization effects
    """

    rng = np.random.RandomState(42587)
    ngauss = 1
    data = get_ngauss_obs(rng=rng, ngauss=ngauss, noise=noise)

    obs = data['obs']
    gm = data['gmix']

    pixel_scale = obs.jacobian.scale

    pars = gm.get_full_pars()

    gm_guess = gm.copy()
    randomize_gmix(rng=rng, gmix=gm_guess, pixel_scale=pixel_scale)

    res = fit_em(obs=obs, guess=gm_guess)

    assert res['flags'] == 0

    fit_gm = res.get_gmix()

    fitpars = fit_gm.get_full_pars()

    if noise == 0.0:
        assert abs(fitpars[0]/pars[0]-1) < FRAC_TOL
        assert abs(fitpars[1]-pars[1]) < pixel_scale/10
        assert abs(fitpars[2]-pars[2]) < pixel_scale/10
        assert abs(fitpars[3]/pars[3]-1) < FRAC_TOL
        assert abs(fitpars[4]/pars[4]-1) < FRAC_TOL
        assert abs(fitpars[5]/pars[5]-1) < FRAC_TOL

    # check reconstructed image allowing for noise
    imfit = res.make_image()
    imtol = 0.001 / pixel_scale**2 + noise*5
    assert np.all(np.abs(imfit - obs.image) < imtol)


@pytest.mark.parametrize('noise', [0.0, 0.05])
def test_em_2gauss(noise):
    """
    see if we can recover the input with and without noise to high precision
    even with a bad guess

    Use ngmix to make the image to make sure there are
    no pixelization effects
    """

    rng = np.random.RandomState(42587)
    ngauss = 2
    data = get_ngauss_obs(rng=rng, ngauss=ngauss, noise=noise)
    obs = data['obs']
    gm = data['gmix']

    pixel_scale = obs.jacobian.scale

    pars = gm.get_full_pars()

    gm_guess = gm.copy()
    randomize_gmix(rng=rng, gmix=gm_guess, pixel_scale=pixel_scale)

    res = fit_em(obs=obs, guess=gm_guess)
    assert res['flags'] == 0

    fit_gm = res.get_gmix()

    fitpars = fit_gm.get_full_pars()

    f1 = pars[0]
    f2 = pars[6]
    if f1 > f2:
        indices = [1, 0]
    else:
        indices = [0, 1]

    # only check pars for no noise
    if noise == 0.0:
        for i in range(ngauss):
            start = i*6
            end = (i+1)*6

            truepars = pars[start:end]

            fitstart = indices[i]*6
            fitend = (indices[i]+1)*6
            thispars = fitpars[fitstart:fitend]

            assert abs(thispars[0]/truepars[0]-1) < FRAC_TOL
            assert abs(thispars[1]-truepars[1]) < pixel_scale/10
            assert abs(thispars[2]-truepars[2]) < pixel_scale/10
            assert abs(thispars[3]/truepars[3]-1) < FRAC_TOL
            assert abs(thispars[4]/truepars[4]-1) < FRAC_TOL
            assert abs(thispars[5]/truepars[5]-1) < FRAC_TOL

    # check reconstructed image allowing for noise
    imfit = res.make_image()
    imtol = 0.001 / pixel_scale**2 + noise*5
    assert np.all(np.abs(imfit - obs.image) < imtol)


@pytest.mark.parametrize('noise', [0.0, 0.05])
def test_em_2gauss_withpsf(noise):
    """
    see if we can recover the input with and without noise to high precision
    even with a bad guess, with a psf convolved

    Use ngmix to make the image to make sure there are
    no pixelization effects
    """

    rng = np.random.RandomState(587)
    ngauss = 2
    data = get_ngauss_obs(
        rng=rng, ngauss=ngauss, noise=noise, with_psf=True,
    )
    obs = data['obs']
    gm = data['gmix']
    psf_gm = data['psf_gmix']

    # we won't fit for the psf, just use the truth
    obs.psf.gmix = psf_gm

    pixel_scale = obs.jacobian.scale

    pars = gm.get_full_pars()

    gm_guess = gm.copy()
    randomize_gmix(rng=rng, gmix=gm_guess, pixel_scale=pixel_scale)

    res = fit_em(obs=obs, guess=gm_guess, tol=1.0e-5)
    assert res['flags'] == 0

    fit_gm = res.get_gmix()

    fitpars = fit_gm.get_full_pars()

    f1 = pars[0]
    f2 = pars[6]
    if f1 > f2:
        indices = [1, 0]
    else:
        indices = [0, 1]

    # only check pars for no noise
    if noise == 0.0:
        for i in range(ngauss):
            start = i*6
            end = (i+1)*6

            truepars = pars[start:end]

            fitstart = indices[i]*6
            fitend = (indices[i]+1)*6
            thispars = fitpars[fitstart:fitend]

            # seems irc is harder to get right, boost tolerance
            assert abs(thispars[0]/truepars[0]-1) < FRAC_TOL
            assert abs(thispars[1]-truepars[1]) < pixel_scale/10
            assert abs(thispars[2]-truepars[2]) < pixel_scale/10
            assert abs(thispars[3]/truepars[3]-1) < FRAC_TOL
            assert abs(thispars[4]/truepars[4]-1) < FRAC_TOL * 3
            assert abs(thispars[5]/truepars[5]-1) < FRAC_TOL

    # check reconstructed image allowing for noise
    imfit = res.make_image()
    imtol = 0.001 / pixel_scale**2 + noise*5
    assert np.all(np.abs(imfit - obs.image) < imtol)


@pytest.mark.parametrize('noise', [0.0, 0.05])
@pytest.mark.parametrize('em_type', ['fixcen', 'fluxonly'])
def test_em_types(em_type, noise):
    """
    test fixcen and fluxonly fitters
    """

    rng = np.random.RandomState(42587)
    ngauss = 1
    data = get_ngauss_obs(rng=rng, ngauss=ngauss, noise=noise)

    obs = data['obs']
    gm = data['gmix']

    pixel_scale = obs.jacobian.scale

    pars = gm.get_full_pars()

    gm_guess = gm.copy()

    fixcen = False
    fluxonly = False
    if em_type == 'fixcen':
        fixcen = True
    if em_type == 'fluxonly':
        fluxonly = True

    res = fit_em(obs=obs, guess=gm_guess, fixcen=fixcen, fluxonly=fluxonly)

    assert res['flags'] == 0

    fit_gm = res.get_gmix()

    fitpars = fit_gm.get_full_pars()

    if noise == 0.0:
        assert abs(fitpars[0]/pars[0]-1) < FRAC_TOL
        assert abs(fitpars[1]-pars[1]) < pixel_scale/10
        assert abs(fitpars[2]-pars[2]) < pixel_scale/10
        assert abs(fitpars[3]/pars[3]-1) < FRAC_TOL
        assert abs(fitpars[4]/pars[4]-1) < FRAC_TOL
        assert abs(fitpars[5]/pars[5]-1) < FRAC_TOL
