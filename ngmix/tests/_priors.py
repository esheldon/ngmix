import ngmix


def get_prior(*, fit_model, rng, scale):
    g_prior = ngmix.priors.GPriorBA(0.1, rng=rng)
    cen_prior = ngmix.priors.CenPrior(0, 0, scale, scale, rng=rng)
    T_prior = ngmix.priors.FlatPrior(0.01, 2, rng=rng)
    F_prior = ngmix.priors.FlatPrior(1e-4, 1e9, rng=rng)

    if fit_model == 'bd':
        fracdev_prior = ngmix.priors.Normal(0.5, 0.1, rng=rng)
        lrat_prior = ngmix.priors.Normal(0.0, 0.1, rng=rng)
        prior = ngmix.joint_prior.PriorBDSep(
            cen_prior=cen_prior,
            g_prior=g_prior,
            T_prior=T_prior,
            logTratio_prior=lrat_prior,
            fracdev_prior=fracdev_prior,
            F_prior=F_prior,
        )

    elif fit_model == 'bdf':
        fracdev_prior = ngmix.priors.Normal(0.5, 0.1, rng=rng)
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
