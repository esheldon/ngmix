import numpy as np

from ngmix import UnitJacobian, GMix
from ngmix.em import GMixEM, prep_image
from ngmix import Observation
from ngmix.priors import srandu


def test_2gauss():
    """
    see if we can recover the input with no noise to
    high precision even with a bad guess

    Use ngmix to make the image to make sure there are
    no pixelization effects
    """

    ngauss = 2
    counts = 100.0
    noise = 0.0
    rng = np.random.RandomState(42587)

    dims = [25, 25]

    cen = (np.array(dims) - 1.0) / 2.0
    jacob = UnitJacobian(row=cen[0], col=cen[1])

    cen1 = [-3.25, -3.25]
    cen2 = [3.0, 0.5]

    e1_1 = 0.1
    e2_1 = 0.05
    T_1 = 8.0
    counts_1 = 0.4 * counts
    irr_1 = T_1 / 2.0 * (1 - e1_1)
    irc_1 = T_1 / 2.0 * e2_1
    icc_1 = T_1 / 2.0 * (1 + e1_1)

    e1_2 = -0.2
    e2_2 = -0.1
    T_2 = 4.0
    counts_2 = 0.6 * counts
    irr_2 = T_2 / 2.0 * (1 - e1_2)
    irc_2 = T_2 / 2.0 * e2_2
    icc_2 = T_2 / 2.0 * (1 + e1_2)

    pars = [
        counts_1,
        cen1[0],
        cen1[1],
        irr_1,
        irc_1,
        icc_1,
        counts_2,
        cen2[0],
        cen2[1],
        irr_2,
        irc_2,
        icc_2,
    ]

    gm = GMix(pars=pars)

    im0 = gm.make_image(dims, jacobian=jacob)
    im = im0 + noise * np.random.randn(im0.size).reshape(dims)

    imsky, sky = prep_image(im)

    obs = Observation(imsky, jacobian=jacob)

    gm_guess = gm.copy()
    gm_guess._data["p"] += counts_1/10 * srandu(2, rng=rng)
    gm_guess._data["row"] += 4 * srandu(2, rng=rng)
    gm_guess._data["col"] += 4 * srandu(2, rng=rng)
    gm_guess._data["irr"] += 0.5 * srandu(2, rng=rng)
    gm_guess._data["irc"] += 0.5 * srandu(2, rng=rng)
    gm_guess._data["icc"] += 0.5 * srandu(2, rng=rng)

    em = GMixEM(obs)
    em.go(gm_guess, sky)

    fit_gm = em.get_gmix()
    res = em.get_result()
    assert res['flags'] == 0

    fitpars = fit_gm.get_full_pars()
    for i in range(ngauss):
        start = i*6
        end = (i+1)*6

        truepars = pars[start:end]
        thispars = fitpars[start:end]

        tol = 1.0e-4
        assert (thispars[0]/truepars[0]-1) < tol
        assert (thispars[3]/truepars[3]-1) < tol
        assert (thispars[4]/truepars[4]-1) < tol
        assert (thispars[5]/truepars[5]-1) < tol

    imfit = em.make_image()
    assert np.all((imfit - im) < 0.001)
