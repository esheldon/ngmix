import ngmix


def get_prior(*, fit_model, rng, scale, T_range=None, F_range=None, nband=None):

    if T_range is None:
        T_range = [-1.0, 1.e3]
    if F_range is None:
        F_range = [-100.0, 1.e9]

    g_prior = ngmix.priors.GPriorBA(sigma=0.1, rng=rng)
    cen_prior = ngmix.priors.CenPrior(
        cen1=0, cen2=0, sigma1=scale, sigma2=scale, rng=rng,
    )
    # T_prior = ngmix.priors.FlatPrior(minval=0.01, maxval=2, rng=rng)
    # F_prior = ngmix.priors.FlatPrior(minval=1e-4, maxval=1e9, rng=rng)
    T_prior = ngmix.priors.FlatPrior(minval=T_range[0], maxval=T_range[1], rng=rng)
    F_prior = ngmix.priors.FlatPrior(minval=F_range[0], maxval=F_range[1], rng=rng)

    if nband is not None:
        F_prior = [F_prior]*nband

    if fit_model == 'bd':
        fracdev_prior = ngmix.priors.Normal(mean=0.5, sigma=0.1, rng=rng)
        lrat_prior = ngmix.priors.Normal(mean=0.0, sigma=0.1, rng=rng)
        prior = ngmix.joint_prior.PriorBDSep(
            cen_prior=cen_prior,
            g_prior=g_prior,
            T_prior=T_prior,
            logTratio_prior=lrat_prior,
            fracdev_prior=fracdev_prior,
            F_prior=F_prior,
        )

    elif fit_model == 'bdf':
        fracdev_prior = ngmix.priors.Normal(mean=0.5, sigma=0.1, rng=rng)
        prior = ngmix.joint_prior.PriorBDFSep(
            cen_prior=cen_prior,
            g_prior=g_prior,
            T_prior=T_prior,
            fracdev_prior=fracdev_prior,
            F_prior=F_prior,
        )
    else:
        prior = ngmix.joint_prior.PriorSimpleSep(
            cen_prior=cen_prior,
            g_prior=g_prior,
            T_prior=T_prior,
            F_prior=F_prior,
        )

    return prior


def get_prior_galsimfit(*, model, rng, scale, r50_range=None, F_range=None):

    if r50_range is None:
        r50_range = [0.01, 10]
    if F_range is None:
        F_range = [1.0e-4, 1.e9]

    g_prior = ngmix.priors.GPriorBA(sigma=0.1, rng=rng)
    cen_prior = ngmix.priors.CenPrior(
        cen1=0, cen2=0, sigma1=scale, sigma2=scale, rng=rng,
    )
    r50_prior = ngmix.priors.FlatPrior(
        minval=r50_range[0], maxval=r50_range[1], rng=rng,
    )
    F_prior = ngmix.priors.FlatPrior(
        minval=F_range[0], maxval=F_range[1], rng=rng,
    )

    if model == 'spergel':
        nu_prior = ngmix.priors.Normal(mean=2, sigma=2, bounds=[-0.5, 3], rng=rng)
        prior = ngmix.joint_prior.PriorSpergelSep(
            cen_prior=cen_prior,
            g_prior=g_prior,
            r50_prior=r50_prior,
            nu_prior=nu_prior,
            F_prior=F_prior,
        )

    else:

        prior = ngmix.joint_prior.PriorGalsimSimpleSep(
            cen_prior=cen_prior,
            g_prior=g_prior,
            r50_prior=r50_prior,
            F_prior=F_prior,
        )

    return prior
