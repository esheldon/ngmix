import ngmix


def get_prior(*, fit_model, rng, scale):
    g_prior = ngmix.priors.GPriorBA(sigma=0.1, rng=rng)
    cen_prior = ngmix.priors.CenPrior(
        cen1=0, cen2=0, sigma1=scale, sigma2=scale, rng=rng,
    )
    T_prior = ngmix.priors.FlatPrior(minval=0.01, maxval=2, rng=rng)
    F_prior = ngmix.priors.FlatPrior(minval=1e-4, maxval=1e9, rng=rng)

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
