import numpy as np
import pytest
import ngmix


@pytest.mark.parametrize('use_bounds', [False, True])
@pytest.mark.parametrize('nband', [None, 2])
def test_joint_priors_simple_smoke(use_bounds, nband):

    rng = np.random.RandomState(932)
    npars = 6
    if nband is not None:
        npars += nband - 1

    cen_prior = ngmix.priors.CenPrior(cen1=1, cen2=1, sigma1=1, sigma2=1, rng=rng)
    g_prior = ngmix.priors.GPriorBA(sigma=0.3, rng=rng)
    if use_bounds:
        # we would never use a normal for T, but want to test bounds
        T_prior = ngmix.priors.Normal(mean=10, sigma=5, bounds=[-1, 1000], rng=rng)
    else:
        T_prior = ngmix.priors.FlatPrior(minval=-1, maxval=100, rng=rng)

    F_prior = ngmix.priors.FlatPrior(minval=-1, maxval=100, rng=rng)
    if nband is not None:
        F_prior = [F_prior]*nband

    jp = ngmix.joint_prior.PriorSimpleSep(
        cen_prior=cen_prior,
        g_prior=g_prior,
        T_prior=T_prior,
        F_prior=F_prior,
    )

    pars = jp.sample()
    assert len(pars.shape) == 1
    assert pars.size == npars

    num = 10
    pars = jp.sample(num)
    assert pars.shape == (num, npars)

    assert np.isscalar(jp.get_prob_scalar(pars[0]))
    assert np.isscalar(jp.get_lnprob_scalar(pars[0]))
    assert jp.get_prob_array(pars).shape == (num, )
    assert jp.get_lnprob_array(pars).shape == (num, )

    fdiff = np.zeros(10)
    jp.fill_fdiff(pars[0], fdiff)

    widths = jp.get_widths()
    assert widths.size == npars


@pytest.mark.parametrize('use_bounds', [False, True])
@pytest.mark.parametrize('nband', [None, 2])
def test_joint_priors_galsim_simple_smoke(use_bounds, nband):

    rng = np.random.RandomState(32)
    npars = 6
    if nband is not None:
        npars += nband - 1

    cen_prior = ngmix.priors.CenPrior(cen1=1, cen2=1, sigma1=1, sigma2=1, rng=rng)
    g_prior = ngmix.priors.GPriorBA(sigma=0.3, rng=rng)
    if use_bounds:
        # we would never use a normal for r50, but want to test bounds
        r50_prior = ngmix.priors.Normal(mean=2, sigma=1, bounds=[0.01, 1000], rng=rng)
    else:
        r50_prior = ngmix.priors.FlatPrior(minval=0.001, maxval=100, rng=rng)

    F_prior = ngmix.priors.FlatPrior(minval=-1, maxval=100, rng=rng)
    if nband is not None:
        F_prior = [F_prior]*nband

    jp = ngmix.joint_prior.PriorGalsimSimpleSep(
        cen_prior=cen_prior,
        g_prior=g_prior,
        r50_prior=r50_prior,
        F_prior=F_prior,
    )

    pars = jp.sample()
    assert len(pars.shape) == 1
    assert pars.size == npars

    num = 10
    pars = jp.sample(num)
    assert pars.shape == (num, npars)

    assert np.isscalar(jp.get_prob_scalar(pars[0]))
    assert np.isscalar(jp.get_lnprob_scalar(pars[0]))
    assert jp.get_prob_array(pars).shape == (num, )
    assert jp.get_lnprob_array(pars).shape == (num, )

    fdiff = np.zeros(10)
    jp.fill_fdiff(pars[0], fdiff)


@pytest.mark.parametrize('use_bounds', [False, True])
@pytest.mark.parametrize('nband', [None, 2])
def test_joint_priors_bd_smoke(use_bounds, nband):

    rng = np.random.RandomState(32)
    npars = 8
    if nband is not None:
        npars += nband - 1

    cen_prior = ngmix.priors.CenPrior(cen1=1, cen2=1, sigma1=1, sigma2=1, rng=rng)
    g_prior = ngmix.priors.GPriorBA(sigma=0.3, rng=rng)

    T_prior = ngmix.priors.FlatPrior(minval=0.001, maxval=100, rng=rng)

    if use_bounds:
        logTratio_prior = ngmix.priors.Normal(
            mean=0, sigma=0.1, bounds=[-1, 1], rng=rng,
        )
        fracdev_prior = ngmix.priors.Normal(
            mean=0.5, sigma=0.1, bounds=[0, 1], rng=rng,
        )
    else:
        logTratio_prior = ngmix.priors.Normal(mean=0, sigma=0.1, rng=rng)
        fracdev_prior = ngmix.priors.Normal(mean=0.5, sigma=0.1, rng=rng)

    F_prior = ngmix.priors.FlatPrior(minval=-1, maxval=100, rng=rng)

    if nband is not None:
        F_prior = [F_prior]*nband

    jp = ngmix.joint_prior.PriorBDSep(
        cen_prior=cen_prior,
        g_prior=g_prior,
        T_prior=T_prior,
        logTratio_prior=logTratio_prior,
        fracdev_prior=fracdev_prior,
        F_prior=F_prior,
    )

    pars = jp.sample()
    assert len(pars.shape) == 1
    assert pars.size == npars

    num = 10
    pars = jp.sample(num)
    assert pars.shape == (num, npars)

    assert np.isscalar(jp.get_prob_scalar(pars[0]))
    assert np.isscalar(jp.get_lnprob_scalar(pars[0]))
    assert jp.get_prob_array(pars).shape == (num, )
    assert jp.get_lnprob_array(pars).shape == (num, )

    fdiff = np.zeros(10)
    jp.fill_fdiff(pars[0], fdiff)


@pytest.mark.parametrize('use_bounds', [False, True])
@pytest.mark.parametrize('nband', [None, 2])
def test_joint_priors_bdf_smoke(use_bounds, nband):

    rng = np.random.RandomState(32)
    npars = 7
    if nband is not None:
        npars += nband - 1

    cen_prior = ngmix.priors.CenPrior(cen1=1, cen2=1, sigma1=1, sigma2=1, rng=rng)
    g_prior = ngmix.priors.GPriorBA(sigma=0.3, rng=rng)

    T_prior = ngmix.priors.FlatPrior(minval=-1, maxval=100, rng=rng)
    if use_bounds:
        fracdev_prior = ngmix.priors.Normal(mean=0.5, sigma=0.1, bounds=[0, 1], rng=rng)
    else:
        fracdev_prior = ngmix.priors.Normal(mean=0.5, sigma=0.1, rng=rng)
    F_prior = ngmix.priors.FlatPrior(minval=-1, maxval=100, rng=rng)

    if nband is not None:
        F_prior = [F_prior]*nband

    jp = ngmix.joint_prior.PriorBDFSep(
        cen_prior=cen_prior,
        g_prior=g_prior,
        T_prior=T_prior,
        fracdev_prior=fracdev_prior,
        F_prior=F_prior,
    )

    pars = jp.sample()
    assert len(pars.shape) == 1
    assert pars.size == npars

    num = 10
    pars = jp.sample(num)
    assert pars.shape == (num, npars)

    assert np.isscalar(jp.get_prob_scalar(pars[0]))
    assert np.isscalar(jp.get_lnprob_scalar(pars[0]))
    assert jp.get_prob_array(pars).shape == (num, )
    assert jp.get_lnprob_array(pars).shape == (num, )

    fdiff = np.zeros(10)
    jp.fill_fdiff(pars[0], fdiff)


@pytest.mark.parametrize('use_bounds', [False, True])
@pytest.mark.parametrize('nband', [None, 2])
def test_joint_priors_spergel_smoke(use_bounds, nband):

    rng = np.random.RandomState(32)
    npars = 7
    if nband is not None:
        npars += nband - 1

    cen_prior = ngmix.priors.CenPrior(cen1=1, cen2=1, sigma1=1, sigma2=1, rng=rng)
    g_prior = ngmix.priors.GPriorBA(sigma=0.3, rng=rng)

    r50_prior = ngmix.priors.FlatPrior(minval=0.001, maxval=100, rng=rng)

    if use_bounds:
        nu_prior = ngmix.priors.Normal(mean=2, sigma=2, bounds=[-0.5, 3], rng=rng)
    else:
        nu_prior = ngmix.priors.Normal(mean=2, sigma=2, rng=rng)
    F_prior = ngmix.priors.FlatPrior(minval=-1, maxval=100, rng=rng)

    if nband is not None:
        F_prior = [F_prior]*nband

    jp = ngmix.joint_prior.PriorSpergelSep(
        cen_prior=cen_prior,
        g_prior=g_prior,
        r50_prior=r50_prior,
        nu_prior=nu_prior,
        F_prior=F_prior,
    )

    pars = jp.sample()
    assert len(pars.shape) == 1
    assert pars.size == npars

    num = 10
    pars = jp.sample(num)
    assert pars.shape == (num, npars)

    assert np.isscalar(jp.get_prob_scalar(pars[0]))
    assert np.isscalar(jp.get_lnprob_scalar(pars[0]))
    assert jp.get_prob_array(pars).shape == (num, )
    assert jp.get_lnprob_array(pars).shape == (num, )

    fdiff = np.zeros(10)
    jp.fill_fdiff(pars[0], fdiff)


@pytest.mark.parametrize('use_bounds', [False, True])
def test_joint_priors_coellip_smoke(use_bounds):

    rng = np.random.RandomState(32)
    ngauss = 2
    npars = ngmix.gmix.get_coellip_npars(ngauss)

    cen_prior = ngmix.priors.CenPrior(cen1=1, cen2=1, sigma1=1, sigma2=1, rng=rng)
    g_prior = ngmix.priors.GPriorBA(sigma=0.3, rng=rng)

    if use_bounds:
        # we would never use a normal for T, but want to test bounds
        T_prior = ngmix.priors.Normal(mean=10, sigma=5, bounds=[-1, 1000], rng=rng)
    else:
        T_prior = ngmix.priors.FlatPrior(minval=-1, maxval=100, rng=rng)

    F_prior = ngmix.priors.FlatPrior(minval=-1, maxval=100, rng=rng)

    jp = ngmix.joint_prior.PriorCoellipSame(
        ngauss=ngauss,
        cen_prior=cen_prior,
        g_prior=g_prior,
        T_prior=T_prior,
        F_prior=F_prior,
    )

    pars = jp.sample()
    assert len(pars.shape) == 1
    assert pars.size == npars

    num = 10
    pars = jp.sample(num)
    assert pars.shape == (num, npars)

    assert np.isscalar(jp.get_lnprob_scalar(pars[0]))

    fdiff = np.zeros(10)
    jp.fill_fdiff(pars[0], fdiff)

    with pytest.raises(ValueError):
        jp.get_lnprob_scalar(np.zeros(100))

    with pytest.raises(ValueError):
        jp.fill_fdiff(np.zeros(100), fdiff)

    with pytest.raises(ValueError):
        jp = ngmix.joint_prior.PriorCoellipSame(
            ngauss=ngauss,
            cen_prior=cen_prior,
            g_prior=g_prior,
            T_prior=T_prior,
            F_prior=[F_prior]*3,
        )
