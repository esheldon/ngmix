import pytest
import numpy as np

from ngmix.em import fit_em
from ngmix.priors import srandu
from ._sims import get_obs


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
    data = get_obs(rng=rng, ngauss=ngauss, noise=noise)

    obs = data['obs']
    gm = data['gmix']

    pixel_scale = obs.jacobian.scale

    pars = gm.get_full_pars()
    counts = pars[0]

    gm_guess = gm.copy()
    gm_guess._data["p"] += counts/10 * srandu(rng=rng)
    gm_guess._data["row"] += 4 * pixel_scale * srandu(rng=rng)
    gm_guess._data["col"] += 4 * pixel_scale * srandu(rng=rng)
    gm_guess._data["irr"] += 0.5 * pixel_scale**2 * srandu(rng=rng)
    gm_guess._data["irc"] += 0.5 * pixel_scale**2 * srandu(rng=rng)
    gm_guess._data["icc"] += 0.5 * pixel_scale**2 * srandu(rng=rng)

    fitter = fit_em(obs=obs, guess=gm_guess)

    fit_gm = fitter.get_gmix()
    res = fitter.get_result()
    assert res['flags'] == 0

    fitpars = fit_gm.get_full_pars()

    if noise == 0.0:
        tol = 1.0e-4
        assert (fitpars[0]/pars[0]-1) < tol
        assert (fitpars[1]-pars[1]) < pixel_scale/10
        assert (fitpars[2]-pars[2]) < pixel_scale/10
        assert (fitpars[3]/pars[3]-1) < tol
        assert (fitpars[4]/pars[4]-1) < tol
        assert (fitpars[5]/pars[5]-1) < tol

    # check reconstructed image allowing for noise
    imfit = fitter.make_image()
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
    data = get_obs(rng=rng, ngauss=ngauss, noise=noise)
    obs = data['obs']
    gm = data['gmix']

    pixel_scale = obs.jacobian.scale

    pars = gm.get_full_pars()
    counts_1 = pars[0]

    gm_guess = gm.copy()
    gm_guess._data["p"] += counts_1/10 * srandu(2, rng=rng)
    gm_guess._data["row"] += 4 * pixel_scale * srandu(2, rng=rng)
    gm_guess._data["col"] += 4 * pixel_scale * srandu(2, rng=rng)
    gm_guess._data["irr"] += 0.5 * pixel_scale**2 * srandu(2, rng=rng)
    gm_guess._data["irc"] += 0.5 * pixel_scale**2 * srandu(2, rng=rng)
    gm_guess._data["icc"] += 0.5 * pixel_scale**2 * srandu(2, rng=rng)

    fitter = fit_em(obs=obs, guess=gm_guess)

    fit_gm = fitter.get_gmix()

    res = fitter.get_result()
    assert res['flags'] == 0

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

            tol = 1.0e-4
            assert (thispars[0]/truepars[0]-1) < tol
            assert (thispars[1]-truepars[1]) < pixel_scale/10
            assert (thispars[2]-truepars[2]) < pixel_scale/10
            assert (thispars[3]/truepars[3]-1) < tol
            assert (thispars[4]/truepars[4]-1) < tol
            assert (thispars[5]/truepars[5]-1) < tol

    # check reconstructed image allowing for noise
    imfit = fitter.make_image()
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

    rng = np.random.RandomState(42587)
    ngauss = 2
    data = get_obs(
        rng=rng, ngauss=ngauss, noise=noise, with_psf=True,
    )
    obs = data['obs']
    gm = data['gmix']
    psf_gm = data['psf_gmix']

    # we won't fit for the psf, just use the truth
    obs.psf.gmix = psf_gm

    pixel_scale = obs.jacobian.scale

    pars = gm.get_full_pars()
    counts_1 = pars[0]

    gm_guess = gm.copy()
    gm_guess._data["p"] += counts_1/10 * srandu(2, rng=rng)
    gm_guess._data["row"] += pixel_scale * srandu(2, rng=rng)
    gm_guess._data["col"] += pixel_scale * srandu(2, rng=rng)
    gm_guess._data["irr"] += 0.1 * pixel_scale**2 * srandu(2, rng=rng)
    gm_guess._data["irc"] += 0.1 * pixel_scale**2 * srandu(2, rng=rng)
    gm_guess._data["icc"] += 0.1 * pixel_scale**2 * srandu(2, rng=rng)

    fitter = fit_em(obs=obs, guess=gm_guess)

    fit_gm = fitter.get_gmix()
    res = fitter.get_result()
    assert res['flags'] == 0

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

            tol = 1.0e-4
            assert (thispars[0]/truepars[0]-1) < tol
            assert (thispars[1]-truepars[1]) < pixel_scale/10
            assert (thispars[2]-truepars[2]) < pixel_scale/10
            assert (thispars[3]/truepars[3]-1) < tol
            assert (thispars[4]/truepars[4]-1) < tol
            assert (thispars[5]/truepars[5]-1) < tol

    # check reconstructed image allowing for noise
    imfit = fitter.make_image()
    imtol = 0.001 / pixel_scale**2 + noise*5
    assert np.all(np.abs(imfit - obs.image) < imtol)
