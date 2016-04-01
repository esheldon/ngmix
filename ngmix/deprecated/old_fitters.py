class LMGaussMom(LMSimple):
    """
    Fit gaussian in moment space
    """
    def __init__(self, obs, **keys):

        super(LMGaussMom,self).__init__(obs, 'gaussmom', **keys)

        #                 c1 c2  M1  M2   T   Ii
        self.n_prior_pars=1 + 1 + 1 + 1 + 1 + self.nband
        self.fdiff_size=self.totpix + self.n_prior_pars

    def go(self, guess):
        """
        Run leastsq and set the result
        """

        guess=array(guess,dtype='f8',copy=False)
        self._setup_data(guess)

        result = run_leastsq(self._calc_fdiff,
                             guess,
                             self.n_prior_pars,
                             **self.lm_pars)

        result['model'] = self.model_name
        if result['flags']==0:
            stat_dict=self.get_fit_stats(result['pars'])
            result.update(stat_dict)

        self._result=result

    def get_band_pars(self, pars_in, band):
        """
        Get linear pars for the specified band

        pars are [c1,c2,M1,M2,T,I1,I2...]

        Where M1 = Icc-Irr
              m2 = 2*Irc
        """

        pars=self._band_pars

        pars[0:5] = pars_in[0:5]
        pars[5] = pars_in[5+band]

        return pars

    def calc_cov(self, h, *args, **kw):
        """
        Run get_cov() to calculate the covariance matrix at the best-fit point.
        If all goes well, add 'pars_cov', 'pars_err', and 'g_cov' to the result
        array

        Note in get_cov, if the Hessian is singular, a diagonal cov matrix is
        attempted to be inverted. If that finally fails LinAlgError is raised.
        In that case we catch it and set a flag EIG_NOTFINITE and the cov is
        not added to the result dict

        Also if there are negative diagonal elements of the cov matrix, the
        EIG_NOTFINITE flag is set and the cov is not added to the result dict
        """

        diag_on_fail=kw.get('diag_on_fail',True)

        res=self.get_result()

        bad=True

        try:
            cov = self.get_cov(res['pars'], h=h, diag_on_fail=diag_on_fail)

            cdiag = diag(cov)

            w,=where(cdiag <= 0)
            if w.size == 0:

                err = sqrt(cdiag)
                w,=where(isfinite(err))
                if w.size != err.size:
                    print_pars(err, front="diagonals not finite:")
                else:
                    # everything looks OK
                    bad=False
            else:
                print_pars(cdiag,front='    diagonals negative:')

        except LinAlgError:
            print("caught LinAlgError")

        if bad:
            res['flags'] |= EIG_NOTFINITE
        else:
            res['pars_cov'] = cov
            res['pars_err']= err

            if len(err) >= 6:
                res['g_cov'] = cov[2:2+2, 2:2+2]

    def get_cov(self, pars, h, diag_on_fail=True):
        """
        calculate the covariance matrix at the specified point

        parameters
        ----------
        pars: array
            Array of parameters at which to evaluate the cov matrix
        h: step size, optional
            Step size for finite differences, default 1.0e-3

        Raises
        ------
        LinAlgError:
            If the hessian is singular a diagonal version is tried
            and if that fails finally a LinAlgError is raised.
        """
        import covmatrix

        # get a copy as an array
        pars=numpy.array(pars)

        # we could call covmatrix.get_cov directly but we want to fall back
        # to a diagonal hessian if it is singular

        hess=covmatrix.calc_hess(self.calc_lnprob, pars, h)

        try:
            cov = -linalg.inv(hess)
        except LinAlgError:
            # pull out a diagonal version of the hessian
            # this might still fail

            if diag_on_fail:
                hdiag=diag(diag(hess))
                cov = -linalg.inv(hess)
            else:
                raise
        return cov

class LMSimpleRound(LMSimple):
    """
    This version fits [cen1,cen2,T,F]
    """
    def __init__(self, *args, **keys):
        super(LMSimpleRound,self).__init__(*args, **keys)

        self.npars = self.npars-2
        self.n_prior_pars=self.n_prior_pars - 1
        self.fdiff_size=self.totpix + self.n_prior_pars

        if self.use_logpars:
            # allbands
            self._pars_allbands=zeros(self.npars+2)


    def run_lm(self, guess):
        """
        Run leastsq and set the result
        """

        guess=array(guess,dtype='f8',copy=False)
        self._setup_data(guess)

        result = run_leastsq(self._calc_fdiff,
                             guess,
                             self.n_prior_pars,
                             **self.lm_pars)

        result['model'] = self.model_name
        if result['flags']==0:
            stat_dict=self.get_fit_stats(result['pars'])
            result.update(stat_dict)

        self._result=result

    def get_band_pars(self, pars_in, band):
        """
        Get linear pars for the specified band
        """


        pars=self._band_pars
        if self.use_logpars:
            _get_simple_band_pars_round_logpars(pars_in,
                                                self._pars_allbands,
                                                pars,
                                                band)
        else:
            _get_simple_band_pars_round_linpars(pars_in,
                                                pars,
                                                band)
        return pars


class LMSimpleFixT(LMSimple):
    """
    This version fits [cen1,cen2,g1,g2,F]
    """
    def __init__(self, *args, **keys):
        super(LMSimpleFixT,self).__init__(*args, **keys)

        self.npars = self.npars-1
        self.n_prior_pars=self.n_prior_pars - 1
        self.fdiff_size=self.totpix + self.n_prior_pars

        self.T = keys['T']

    def run_lm(self, guess):
        """
        Run leastsq and set the result
        """

        guess=array(guess,dtype='f8',copy=False)
        self._setup_data(guess)

        result = run_leastsq(self._calc_fdiff, guess, self.n_prior_pars, **self.lm_pars)

        result['model'] = self.model_name
        if result['flags']==0:
            stat_dict=self.get_fit_stats(result['pars'])
            result.update(stat_dict)

        self._result=result

    def get_band_pars(self, pars_in, band):
        """
        Get linear pars for the specified band
        """


        pars=self._band_pars
        pars[0:4] = pars_in[0:4]
        pars[4]=self.T

        if self.use_logpars:
            pars[5]=exp( pars_in[4+band] )
        else:
            pars[5]=pars_in[4]

        return pars

class LMSimpleGOnly(LMSimple):
    """
    This version fits [cen1,cen2,g1,g2,F]
    """
    def __init__(self, *args, **keys):
        super(LMSimpleGOnly,self).__init__(*args, **keys)

        self.pars_in0 = array(keys['pars'], dtype='f8')

        pars_in=self.pars_in0.copy()
        if self.use_logpars:
            pars_in[4:] = exp(self.pars_in0[4:])
        self.pars_in=pars_in

        self.npars = 2
        self.n_prior_pars=1
        self.fdiff_size=self.totpix + self.n_prior_pars

    def run_lm(self, guess):
        """
        Run leastsq and set the result
        """

        guess=array(guess,dtype='f8',copy=False)
        self._setup_data(guess)

        result = run_leastsq(self._calc_fdiff, guess, self.n_prior_pars, **self.lm_pars)

        result['model'] = self.model_name
        if result['flags']==0:
            stat_dict=self.get_fit_stats(result['pars'])
            result.update(stat_dict)

        self._result=result

    def get_band_pars(self, pars_in, band):
        """
        pars in are [g1,g2]
        """

        pars=self.pars_in
        pars[2]=pars_in[0]
        pars[3]=pars_in[1]
        return pars


def _get_simple_band_pars_round_logpars(pars_in, pars_allband, band_pars, band):
    """
    pars in are [cen1,cen2,log(T),log(F)]
    """
    # all band
    pars_allband[0:2] = pars_in[0:2]
    # 2:2+2 remain zero for roundness
    pars_allband[4:] = pars_in[2:]

    _gmix.convert_simple_double_logpars_band(pars_allband, band_pars, band)

def _get_simple_band_pars_round_linpars(pars_in, band_pars, band):
    """
    pars in are [cen1,cen2,T,F]
    """
    band_pars[0:2] = pars_in[0:2]
    # 2:2+2 remain zero for roundness
    band_pars[4] = pars_in[2]
    band_pars[5] = pars_in[3+band]

def _get_simple_band_pars_fixT_linpars(T, pars_in, band_pars, band):
    """
    pars in are [cen1,cen2,T,F]
    """
    band_pars[0:4] = pars_in[0:4]
    band_pars[4]=T
    band_pars[5] = pars_in[3+band]

def _get_simple_band_pars_fixT_logpars(T, pars_in, band_pars, band):
    """
    pars in are [cen1,cen2,T,F]
    """
    band_pars[0:4] = pars_in[0:4]
    band_pars[4]=T
    band_pars[5] = pars_in[3+band]

    _gmix.convert_simple_double_logpars_band(pars_allband, band_pars, band)


# most of this is the same as LMSimple
class LMCompositeRound(LMComposite):
    """
    This version fits [cen1,cen2,T,F]
    """
    def __init__(self, *args, **keys):
        super(LMCompositeRound,self).__init__(*args, **keys)

        self.npars = self.npars-2
        self.n_prior_pars=self.n_prior_pars - 1
        self.fdiff_size=self.totpix + self.n_prior_pars

        if self.use_logpars:
            self._pars_allbands=zeros(self.npars+2)


    def run_lm(self, guess):
        """
        Run leastsq and set the result
        """

        guess=array(guess,dtype='f8',copy=False)
        self._setup_data(guess)

        result = run_leastsq(self._calc_fdiff, guess, self.n_prior_pars, **self.lm_pars)

        result['model'] = self.model_name
        if result['flags']==0:
            stat_dict=self.get_fit_stats(result['pars'])
            result.update(stat_dict)

        self._result=result

    def get_band_pars(self, pars_in, band):
        """
        Get linear pars for the specified band
        """

        pars=self._band_pars
        if self.use_logpars:
            _get_simple_band_pars_round_logpars(pars_in,
                                                self._pars_allbands,
                                                pars,
                                                band)
        else:
            _get_simple_band_pars_round_linpars(pars_in,
                                                pars,
                                                band)

        return pars

class LMSersic(LMSimple):
    def __init__(self, image, weight, jacobian, guess, **keys):
        super(LMSimple,self).__init__(image, weight, jacobian, "sersic", **keys)
        # this is a dict
        # can contain maxfev (maxiter), ftol (tol in sum of squares)
        # xtol (tol in solution), etc
        self.lm_pars=keys['lm_pars']

        self.guess=array( guess, dtype='f8' )

        self.n_prior=keys['n_prior']

        n_prior_pars=7
        self.fdiff_size=self.totpix + n_prior_pars

    def get_band_pars(self, pars, band):
        raise RuntimeError("adapt to new style")
        if band > 0:
            raise ValueError("support more than one band")
        return pars.copy()



######### MCMC stuff below ###########

class MCMCGaussMom(MCMCSimple):
    """
    Fit gaussian in moment space, no psf
    """
    def __init__(self, obs, **keys):

        super(MCMCGaussMom,self).__init__(obs, 'gaussmom', **keys)

    def calc_result(self, **kw):
        """
        Some extra stats for simple models
        """
        super(MCMCSimple,self).calc_result(**kw)

    def get_band_pars(self, pars_in, band):
        """
        Get linear pars for the specified band

        pars are [c1,c2,M1,M2,T,I1,I2...]

        Where M1 = Icc-Irr
              m2 = 2*Irc
        """

        pars=self._band_pars

        pars[0:4] = pars_in[0:4]
        pars[5] = pars_in[5+band]

        return pars

    def get_par_names(self, **kw):
        """
        parameter names for each dimension
        """

        names=['cen1','cen2', 'M1','M2','T']

        if self.nband == 1:
            names += ['F']
        else:
            for band in xrange(self.nband):
                names += ['F_%s' % band]

        return names

class MCMCGaussMomSum(MCMCSimple):
    """
    Fit gaussian in moment space, no psf
    """
    def __init__(self, obs, **keys):

        super(MCMCGaussMomSum,self).__init__(obs, 'gauss', **keys)

        self.model=gmix.GMIX_FULL
        self.model_name='full'

    def calc_result(self, **kw):
        """
        Some extra stats for simple models
        """
        super(MCMCSimple,self).calc_result(**kw)

    def get_band_pars(self, pars_in, band):
        """

        pars are [c1,c2,M1sum,M2sum,Tsum,Isum,...]
        """

        #c1    = pars_in[0]
        #c2    = pars_in[1]
        c1sum    = pars_in[0]
        c2sum    = pars_in[1]
        M1sum = pars_in[2]
        M2sum = pars_in[3]
        Tsum  = pars_in[4]
        Isum  = pars_in[5+band]

        c1=c1sum/Isum
        c2=c2sum/Isum
        M1 = M1sum/Isum
        M2 = M2sum/Isum
        T  = Tsum/Isum

        Irr = (T-M1)*0.5
        Irc = M2/2
        Icc = (T+M1)*0.5

        pars=self._band_pars

        pars[0] = Isum
        pars[1] = c1
        pars[2] = c2
        pars[3] = Irr
        pars[4] = Irc
        pars[5] = Icc

        return pars

    def get_par_names(self, dolog=False):
        """
        parameter names for each dimension
        """
        names=[]

        names=['cen1','cen2', 'M1sum','M2sum','Tsum']

        if self.nband == 1:
            names += ['Isum']
        else:
            for band in xrange(self.nband):
                names += ['Isum_%s' % band]

        return names



class MCMCSimpleEta(MCMCSimple):
    """
    search eta space
    """

    def get_band_pars(self, pars_in, band):
        """
        Get linear pars for the specified band
        """

        pars=self._band_pars

        status=_gmix.convert_simple_eta2g_band(pars_in, pars, band)
        if status != 1:
            raise GMixRangeError("shape out of bounds")
        #print("eta:",pars_in[2],pars_in[3])
        #print("g:  ",pars[2], pars[3])
        return pars


    def get_par_names(self, dolog=False):
        names=['cen1','cen2', 'eta1','eta2', 'T']
        if self.nband == 1:
            names += ['F']
        else:
            for band in xrange(self.nband):
                names += ['F_%s' % band]

        return names

class MCMCSersic(MCMCSimple):
    def __init__(self, obs, **keys):

        raise RuntimeError("adapt to new system")
        self.g1i=2
        self.g2i=3

        MCMCBase.__init__(self, obs, "sersic", **keys)


    def _setup_sampler_and_data(self, pos):
        """
        try very hard to initialize the mixtures
        """

        self.flags=0
        self._tau=0.0
        self.pos=pos
        self.npars=pos.shape[1]

        self.sampler = self._make_sampler()
        self._best_lnprob=None

        ok=False
        for i in xrange(self.nwalkers):
            try:
                self._init_gmix_all(self.pos[i,:])
                ok=True
                break
            except GMixRangeError as gerror:
                continue
            except ZeroDivisionError:
                continue

        if ok:
            return

        print('failed init gmix lol from input guess:',str(gerror))
        print('getting a new guess')
        for j in xrange(10):
            self.pos=self._get_random_guess()
            ok=False
            for i in xrange(self.nwalkers):
                try:
                    self._init_gmix_all(self.pos[i,:])
                    ok=True
                    break
                except GMixRangeError as gerror:
                    continue
                except ZeroDivisionError:
                    continue
            if ok:
                break

        if not ok:
            raise gerror

    def run_mcmc(self, pos, nstep):
        """
        user can run steps
        """

        if not hasattr(self,'sampler'):
            self._setup_sampler_and_data(pos)

        sampler=self.sampler
        sampler.reset()
        self.pos, prob, state = sampler.run_mcmc(self.pos, nstep)

        lnprobs = sampler.lnprobability.reshape(self.nwalkers*nstep)
        w=lnprobs.argmax()
        bp=lnprobs[w]
        if self._best_lnprob is None or bp > self._best_lnprob:
            self._best_lnprob=bp
            self._best_pars=sampler.flatchain[w,:]

        arates = sampler.acceptance_fraction
        self._arate = arates.mean()

        self._trials=trials

        return self.pos

    def _get_priors(self, pars):
        """
        # go in simple
        add any priors that were sent on construction
        """

        lnp=0.0

        if self.cen_prior is not None:
            lnp += self.cen_prior.get_lnprob(pars[0], pars[1])

        if self.g_prior is not None:
            if self.g_prior_during:
                lnp += self.g_prior.get_lnprob_scalar2d(pars[2],pars[3])
            else:
                # may have bounds
                g = sqrt(pars[2]**2 + pars[3]**2)
                if g > self.g_prior.gmax:
                    raise GMixRangeError("g too big")
        else:
            g = sqrt(pars[2]**2 + pars[3]**2)
            if g >= 0.99999:
                raise GMixRangeError("g too big")

        if self.T_prior is not None:
            lnp += self.T_prior.get_lnprob_scalar(pars[4])

        if self.counts_prior is not None:
            for i,cp in enumerate(self.counts_prior):
                counts=pars[5+i]
                lnp += cp.get_lnprob_scalar(counts)

        lnp += self.n_prior.get_lnprob_scalar(pars[6])

        return lnp

    def get_par_names(self):
        names=['cen1','cen2', 'g1','g2','T','F','n']
        return names

    def _set_npars(self):
        """
        this is actually set elsewhere
        """
        pass

    def get_band_pars(self, pars, band):
        if band > 0:
            raise ValueError("support multi-band for sersic")
        return pars.copy()

class MCMCSersicJointHybrid(MCMCSersic):
    def __init__(self, image, weight, jacobian, **keys):
        raise RuntimeError("adapt to new system")

        self.g1i=2
        self.g2i=3

        self.joint_prior=keys.get('joint_prior',None)

        if (self.joint_prior is None):
            raise ValueError("send joint_prior for sersic joint")

        self.prior_during=keys.get('prior_during',False)

        MCMCBase.__init__(self, image, weight, jacobian, "sersic", **keys)


    def _get_priors(self, pars):
        """
        Apply simple priors
        """
        lnp=0.0

        if self.cen_prior is not None:
            lnp += self.cen_prior.get_lnprob(pars[0], pars[1])

        jp=self.joint_prior

        # this is just the structural parameters
        lnp += jp.get_lnprob_scalar(pars[4:])

        if self.prior_during:
            lnp += jp.g_prior.get_lnprob_scalar2d(pars[2],pars[3])

        return lnp

    def get_band_pars(self, pars, band):
        """
        Extract pars for the specified band and convert to linear
        """
        raise RuntimeError("adapt to new style")
        if band != 0:
            raise ValueError("deal with more than one band")
        linpars=pars.copy()

        linpars[4] = 10.0**linpars[4]
        linpars[5] = 10.0**linpars[5]
        linpars[6] = 10.0**linpars[6]

        return linpars


    def _get_priors(self, pars):
        """
        Apply simple priors
        """
        lnp=0.0

        if self.cen_prior is not None:
            lnp += self.cen_prior.get_lnprob(pars[0], pars[1])

        jp=self.joint_prior

        # this is just the structural parameters
        lnp += jp.get_lnprob_scalar(pars[4:])

        if self.prior_during:
            lnp += jp.g_prior.get_lnprob_scalar2d(pars[2],pars[3])

        return lnp

    def _get_PQR(self):
        """
        get the marginalized P,Q,R from Bernstein & Armstrong
        """

        g_prior=self.joint_prior.g_prior
        trials=self._trials
        g1=trials[:,2]
        g2=trials[:,3]

        #print("get pqr joint simple hybrid")
        sh=self.shear_expand
        if sh is None:
            Pi,Qi,Ri = g_prior.get_pqr_num(g1,g2)
        else:
            print("        expanding about shear:",sh)
            Pi,Qi,Ri = g_prior.get_pqr_num(g1,g2, s1=sh[0], s2=sh[1])

        if self.prior_during:
            # We measured the posterior surface.  But the integrals are over
            # the likelihood.  So divide by the prior.
            #
            # Also note the p we divide by is in principle different from the
            # Pi above, which are evaluated at the shear expansion value

            print("undoing prior for pqr")

            prior_vals=self._get_g_prior_vals()

            w,=numpy.where(prior_vals > 0.0)

            Pinv = 1.0/prior_vals[w]
            Pinv_sum=Pinv.sum()

            Pi = Pi[w]
            Qi = Qi[w,:]
            Ri = Ri[w,:,:]

            # this is not unity if expanding about some shear
            Pi *= Pinv
            Qi[:,0] *= Pinv
            Qi[:,1] *= Pinv

            Ri[:,0,0] *= Pinv
            Ri[:,0,1] *= Pinv
            Ri[:,1,0] *= Pinv
            Ri[:,1,1] *= Pinv

            P = Pi.sum()/Pinv_sum
            Q = Qi.sum(axis=0)/Pinv_sum
            R = Ri.sum(axis=0)/Pinv_sum
        else:
            P = Pi.mean()
            Q = Qi.mean(axis=0)
            R = Ri.mean(axis=0)

        return P,Q,R


    def get_gmix(self):
        """
        Get a gaussian mixture at the "best" parameter set, which
        definition depends on the sub-class
        """
        raise RuntimeError("adapt to new style")
        logpars=self._result['pars']
        pars=logpars.copy()
        pars[4] = 10.0**logpars[4]
        pars[5] = 10.0**logpars[5]
        pars[6] = 10.0**logpars[6]

        gm=gmix.make_gmix_model(pars, self.model)
        return gm


    def _get_g_prior_vals(self):
        if not hasattr(self,'joint_prior_vals'):
            trials=self._trials
            g1,g2=trials[:,2],trials[:,3]
            self.joint_prior_vals = self.joint_prior.g_prior.get_prob_array2d(g1,g2)
        return self.joint_prior_vals

    def get_par_names(self):
        names=[r'$cen_1$',
               r'$cen_2$',
               r'$g_1$',
               r'$g_2$',
               r'$log_{10}(T)$',
               r'$log_{10}(F)$',
               r'$log_{10}(n)$']
        return names


class MCMCSersicDefault(MCMCSimple):
    def __init__(self, image, weight, jacobian, **keys):
        raise RuntimeError("adapt to new system")

        self.full_guess=keys.get('full_guess',None)
        self.g1i=2
        self.g2i=3

        self.n_prior=keys.get('n_prior',None)

        if (self.full_guess is None
                or self.n_prior is None):
            raise ValueError("send full guess n_prior for sersic")

        MCMCBase.__init__(self, image, weight, jacobian, "sersic", **keys)


    def _get_priors(self, pars):
        """
        # go in simple
        add any priors that were sent on construction
        """

        lnp=0.0

        if self.cen_prior is not None:
            lnp += self.cen_prior.get_lnprob(pars[0], pars[1])

        if self.g_prior is not None:
            if self.g_prior_during:
                lnp += self.g_prior.get_lnprob_scalar2d(pars[2],pars[3])
            else:
                # may have bounds
                g = sqrt(pars[2]**2 + pars[3]**2)
                if g > self.g_prior.gmax:
                    raise GMixRangeError("g too big")

        if self.T_prior is not None:
            lnp += self.T_prior.get_lnprob_scalar(pars[4])

        if self.counts_prior is not None:
            for i,cp in enumerate(self.counts_prior):
                counts=pars[5+i]
                lnp += cp.get_lnprob_scalar(counts)

        lnp += self.n_prior.get_lnprob_scalar(pars[6])

        return lnp

    def _get_guess(self):
        return self.full_guess

    def get_par_names(self):
        names=['cen1','cen2', 'g1','g2','T','F','n']
        return names

    def _set_npars(self):
        """
        nband should be set in set_lists, called before this
        """
        self.npars=self.full_guess.shape[1]

    def get_band_pars(self, pars, band):
        if band > 0:
            raise ValueError("support multi-band for sersic")
        return pars.copy()


class MCMCSimpleFixed(MCMCSimple):
    """
    Fix everything but shapes
    """
    def __init__(self, image, weight, jacobian, model, **keys):
        raise RuntimeError("adapt to new system")
        super(MCMCSimpleFixed,self).__init__(image, weight, jacobian, model, **keys)

        # value of elements 2,3 are not important as those are the ones to be
        # varied
        self.fixed_pars=keys['fixed_pars']

        self.npars=2
        self.g1i = 0
        self.g2i = 1

    def _get_priors(self, pars):
        """
        # go in simple
        add any priors that were sent on construction

        """
        lnp=0.0

        if self.g_prior is not None:
            # may have bounds
            g = sqrt(pars[2]**2 + pars[3]**2)
            if g > self.g_prior.gmax:
                raise GMixRangeError("g too big")

        return lnp

    def get_band_pars(self, pars, band):
        raise RuntimeError("adapt to new style")
        bpars= self.fixed_pars[ [0,1,2,3,4,5+band] ]
        bpars[2:2+2] = pars
        return bpars

    def get_par_names(self):
        return ['g1','g2']


class MCMCBDC(MCMCSimple):
    """
    Add additional features to the base class to support coelliptical bulge+disk
    """
    def __init__(self, image, weight, jacobian, **keys):
        raise RuntimeError("adapt to new system")
        super(MCMCBDC,self).__init__(image, weight, jacobian, "bdc", **keys)

        if self.full_guess is None:
            raise ValueError("For BDC you must currently send a full guess")
        self.T_b_prior = keys.get('T_b_prior',None)
        self.T_d_prior = keys.get('T_d_prior',None)
        self.counts_b_prior = keys.get('counts_b_prior',None)
        self.counts_d_prior = keys.get('counts_d_prior',None)

        # we cover this one case, but otherwise the user just have
        # to give this in the right shape
        if self.counts_b_prior is not None:
            self.counts_b_prior=[self.counts_b_prior]
        if self.counts_d_prior is not None:
            self.counts_d_prior=[self.counts_d_prior]

    def _get_priors(self, pars):
        """
        # go in simple
        add any priors that were sent on construction
        """
        lnp=0.0

        if self.cen_prior is not None:
            lnp += self.cen_prior.get_lnprob(pars[0], pars[1])

        if self.g_prior is not None:
            # may have bounds
            g = sqrt(pars[2]**2 + pars[3]**2)
            if g > self.g_prior.gmax:
                raise GMixRangeError("g too big")

        # bulge size
        if self.T_b_prior is not None:
            lnp += self.T_b_prior.get_lnprob_scalar(pars[4])
        # disk size
        if self.T_d_prior is not None:
            lnp += self.T_d_prior.get_lnprob_scalar(pars[5])

        raise ValueError("fix to put prior on total counts and bdfrac")
        # bulge flux in each band
        if self.counts_b_prior is not None:
            for i,cp in enumerate(self.counts_b_prior):
                counts=pars[6+i]
                lnp += cp.get_lnprob_scalar(counts)

        # disk flux in each band
        if self.counts_d_prior is not None:
            for i,cp in enumerate(self.counts_d_prior):
                counts=pars[6+self.nband+i]
                lnp += cp.get_lnprob_scalar(counts)

        return lnp

    def get_band_pars(self, pars, band):
        """
        pars are
            [c1,c2,g1,g2,Tb,Td, Fb1,Fb2,Fb3, ..., Fd1,Fd2,Fd3 ...]
        """
        raise RuntimeError("adapt to new style")
        Fbstart=6
        Fdstart=6+self.nband
        return pars[ [0,1,2,3,4,5, Fbstart+band, Fdstart+band] ]


    def get_par_names(self):
        names=['cen1','cen2', 'g1','g2','Tb','Td']
        if self.nband == 1:
            names += ['Fb','Fd']
        else:
            for band in xrange(self.nband):
                names += ['Fb_%s' % band]
            for band in xrange(self.nband):
                names += ['Fd_%s' % band]

        return names


class MCMCBDF(MCMCSimple):
    """
    Add additional features to the base class to support simple models
    """
    def __init__(self, image, weight, jacobian, **keys):
        raise RuntimeError("adapt to new system")
        super(MCMCBDF,self).__init__(image, weight, jacobian, "bdf", **keys)

        if self.full_guess is None:
            raise ValueError("For BDF you must currently send a full guess")

        # we alrady have T_prior and counts_prior from base class

        # fraction of flux in bulge
        self.bfrac_prior = keys.get('bfrac_prior',None)

        # demand flux for both components is > 0
        self.positive_components = keys.get('positive_components',True)

    def _get_priors(self, pars):
        """
        # go in simple
        add any priors that were sent on construction
        """
        lnp=0.0

        if self.cen_prior is not None:
            lnp += self.cen_prior.get_lnprob(pars[0], pars[1])

        if self.g_prior is not None:
            # may have bounds
            g = sqrt(pars[2]**2 + pars[3]**2)
            if g > self.g_prior.gmax:
                raise GMixRangeError("g too big")

        # prior on total size
        if self.T_prior is not None:
            lnp += self.T_prior.get_lnprob_scalar(pars[4])

        if self.positive_components:
            # both bulge and disk components positive
            if pars[5] <= 0.0 or pars[6] <= 0.0:
                raise GMixRangeError("out of bounds")

        # prior on total counts
        if self.counts_prior is not None:
            for i,cp in enumerate(self.counts_prior):
                counts=pars[5:].sum()
                lnp += cp.get_lnprob_scalar(counts)

        # prior on fraction of total flux in the bulge
        if self.bfrac_prior is not None:

            counts = pars[5:].sum()
            counts_b = pars[5]

            if counts == 0:
                raise GMixRangeError("total counts exactly zero")

            bfrac = counts_b/counts
            lnp += self.bfrac_prior.get_lnprob_scalar(bfrac)

        return lnp

    def get_band_pars(self, pars, band):
        """
        pars are
            [c1,c2,g1,g2,T, Fb1,Fb2,Fb3, ..., Fd1,Fd2,Fd3 ...]
        """
        raise RuntimeError("adapt to new style")
        Fbstart=5
        Fdstart=5+self.nband
        return pars[ [0,1,2,3,4, Fbstart+band, Fdstart+band] ].copy()


    def get_par_names(self):
        names=['cen1','cen2', 'g1','g2','T']
        if self.nband == 1:
            names += ['Fb','Fd']
        else:
            fbnames = []
            fdnames = []
            for band in xrange(self.nband):
                fbnames.append('Fb_%s' % band)
                fdnames.append('Fd_%s' % band)
            names += fbnames
            names += fdnames
        return names


class MCMCBDFJoint(MCMCBDF):
    """
    BDF with a joint prior on [g1,g2,T,Fb,Fd]
    """
    def __init__(self, image, weight, jacobian, **keys):
        raise RuntimeError("adapt to new system")
        super(MCMCBDF,self).__init__(image, weight, jacobian, "bdf", **keys)

        if self.full_guess is None:
            raise ValueError("For BDF you must currently send a full guess")

        # we alrady have T_prior and counts_prior from base class

        # fraction of flux in bulge
        if self.joint_prior is None:
            raise ValueError("send joint prior for MCMCBDFJoint")

        self.Tfracdiff_max = keys['Tfracdiff_max']

    def _get_priors(self, pars):
        """
        # go in simple
        add any priors that were sent on construction
        """
        lnp=0.0

        if self.cen_prior is not None:
            lnp += self.cen_prior.get_lnprob(pars[0], pars[1])

        jp = self.joint_prior
        if jp is not None:
            T_bounds = jp.T_bounds
            Flux_bounds = jp.Flux_bounds
            T=pars[4]
            Fb=pars[5]
            Fd=pars[6]
            if (T < T_bounds[0] or T > T_bounds[1]
                    or Fb < Flux_bounds[0] or Fb > Flux_bounds[1]
                    or Fd < Flux_bounds[0] or Fd > Flux_bounds[1]):
                raise GMixRangeError("T or flux out of range")
        else:
            # even without a prior, we want to enforce positive
            if pars[4] < 0.0 or pars[5] < 0.0 or pars[6] < 0.0:
                raise GMixRangeError("negative T or flux")


        #lnp = self.joint_prior.get_lnprob(pars[2:])

        return lnp

    def _get_PQR(self):
        """
        get the marginalized P,Q,R from Bernstein & Armstrong
        """

        sh=self.shear_expand
        if sh is None:
            Pi,Qi,Ri = self.joint_prior.get_pqr_num(self._trials[:, 2:])
        else:
            Pi,Qi,Ri = self.joint_prior.get_pqr_num(self._trials[:, 2:],
                                                    s1=sh[0],s2=sh[1])
        P,Q,R = self._get_mean_pqr(Pi,Qi,Ri)

        return P,Q,R


    def _do_trials(self):
        """
        run the sampler
        """
        import emcee

        if emcee.ensemble.acor is not None:
            have_acor=True
        else:
            have_acor=False

        # over-ridden
        guess=self._get_guess()
        for i in xrange(10):
            try:
                self._init_gmix_all(guess[0,:])
                break
            except GMixRangeError as gerror:
                # make sure we draw random guess if we got failure
                print('failed init gmix lol:',str(gerror) )
                print('getting a new guess')
                guess=self._get_random_guess()
        if i==9:
            raise gerror

        sampler = self._make_sampler()
        self.sampler=sampler

        self._tau=9999.0

        Tfracdiff_max=self.Tfracdiff_max


        burnin=self.burnin
        self.last_pos = guess

        print('        burnin runs:',burnin)

        ntry=10
        for i in xrange(ntry):

            if i == 3:
                burnin = burnin*2
                print('        burnin:',burnin)

            sampler.reset()
            self.last_pos, prob, state = sampler.run_mcmc(self.last_pos, burnin)

            trials  = sampler.flatchain
            wts = self.joint_prior.get_prob_array(trials[:,2:], throw=False)

            wsum=wts.sum()

            Tvals=trials[:,4]
            Tmean = (Tvals*wts).sum()/wsum
            Terr2 = ( wts**2 * (Tvals-Tmean)**2 ).sum()
            Terr = sqrt( Terr2 )/wsum

            if i > 0:
                Tfracdiff =abs(Tmean/Tmean_last-1.0)
                Tfracdiff_err = Terr/Tmean_last

                tfmess='Tmean: %.3g +/- %.3g Tfracdiff: %.3f +/- %.3f'
                tfmess=tfmess % (Tmean,Terr,Tfracdiff,Tfracdiff_err)

                if (Tfracdiff-1.5*Tfracdiff_err) < Tfracdiff_max:
                    print('        last burn',tfmess)
                    break

                print('        ',tfmess)

            Tmean_last=Tmean
            i += 1

        print('        final run:',self.nstep)
        sampler.reset()
        self.last_pos, prob, state = sampler.run_mcmc(self.last_pos, self.nstep)

        self._trials  = sampler.flatchain
        self.joint_prior_vals = self.joint_prior.get_prob_array(self._trials[:,2:], throw=False)

        arates = sampler.acceptance_fraction
        self._arate = arates.mean()

        lnprobs = sampler.lnprobability.reshape(self.nwalkers*self.nstep)
        w=lnprobs.argmax()
        bp=lnprobs[w]

        self._best_lnprob=bp
        self._best_pars=sampler.flatchain[w,:]

        self.flags=0


class MCMCSimpleJointHybrid(MCMCSimple):
    """
    Simple with a joint prior on [T,F],separate on g1,g2
    """
    def __init__(self, image, weight, jacobian, model, **keys):
        raise RuntimeError("adapt to new system")
        super(MCMCSimpleJointHybrid,self).__init__(image, weight, jacobian, model, **keys)

        if self.full_guess is None:
            raise ValueError("For joint simple you must currently send a full guess")

        if self.joint_prior is None:
            raise ValueError("send joint prior for MCMCSimpleJointHybrid")

        self.prior_during=keys.get('prior_during',False)

    def get_band_pars(self, pars, band):
        """
        Extract pars for the specified band and convert to linear
        """
        from .shape import eta1eta2_to_g1g2
        raise RuntimeError("adapt to new style")
        linpars=pars[ [0,1,2,3,4,5+band] ].copy()

        linpars[4] = 10.0**linpars[4]
        linpars[5] = 10.0**linpars[5]

        return linpars


    def _get_priors(self, pars):
        """
        Apply simple priors
        """
        lnp=0.0

        if self.cen_prior is not None:
            lnp += self.cen_prior.get_lnprob(pars[0], pars[1])

        jp=self.joint_prior

        # this is just the structural parameters
        lnp += jp.get_lnprob_scalar(pars[4:])

        if self.prior_during:
            lnp += jp.g_prior.get_lnprob_scalar2d(pars[2],pars[3])

        return lnp

    def _get_PQR(self):
        """
        get the marginalized P,Q,R from Bernstein & Armstrong
        """

        g_prior=self.joint_prior.g_prior
        trials=self._trials
        g1=trials[:,2]
        g2=trials[:,3]

        #print("get pqr joint simple hybrid")
        sh=self.shear_expand
        if sh is None:
            Pi,Qi,Ri = g_prior.get_pqr_num(g1,g2)
        else:
            print("        expanding about shear:",sh)
            Pi,Qi,Ri = g_prior.get_pqr_num(g1,g2, s1=sh[0], s2=sh[1])

        if self.prior_during:
            # We measured the posterior surface.  But the integrals are over
            # the likelihood.  So divide by the prior.
            #
            # Also note the p we divide by is in principle different from the
            # Pi above, which are evaluated at the shear expansion value

            print("undoing prior for pqr")

            prior_vals=self._get_g_prior_vals()

            w,=numpy.where(prior_vals > 0.0)

            Pinv = 1.0/prior_vals[w]
            Pinv_sum=Pinv.sum()

            Pi = Pi[w]
            Qi = Qi[w,:]
            Ri = Ri[w,:,:]

            # this is not unity if expanding about some shear
            Pi *= Pinv
            Qi[:,0] *= Pinv
            Qi[:,1] *= Pinv

            Ri[:,0,0] *= Pinv
            Ri[:,0,1] *= Pinv
            Ri[:,1,0] *= Pinv
            Ri[:,1,1] *= Pinv

            P = Pi.sum()/Pinv_sum
            Q = Qi.sum(axis=0)/Pinv_sum
            R = Ri.sum(axis=0)/Pinv_sum
        else:
            P = Pi.mean()
            Q = Qi.mean(axis=0)
            R = Ri.mean(axis=0)

        return P,Q,R


    def get_gmix(self):
        """
        Get a gaussian mixture at the "best" parameter set, which
        definition depends on the sub-class
        """
        raise RuntimeError("adapt to new style")
        logpars=self._result['pars']
        pars=logpars.copy()
        pars[4] = 10.0**logpars[4]
        pars[5] = 10.0**logpars[5]

        gm=gmix.make_gmix_model(pars, self.model)
        return gm


    def _get_g_prior_vals(self):
        if not hasattr(self,'joint_prior_vals'):
            trials=self._trials
            g1,g2=trials[:,2],trials[:,3]
            self.joint_prior_vals = self.joint_prior.g_prior.get_prob_array2d(g1,g2)
        return self.joint_prior_vals

    def get_par_names(self):
        names=[r'$cen_1$',
               r'$cen_2$',
               r'$g_1$',
               r'$g_2$',
               r'$log_{10}(T)$']
        if self.nband == 1:
            names += [r'$log_{10}(F)$']
        else:
            for band in xrange(self.nband):
                names += [r'$log_{10}(F_%s)$' % band]
        return names


class MCMCBDFJointHybrid(MCMCSimpleJointHybrid):
    """
    BDF with a joint prior on [T,Fb,Fd] separate on g1,g2
    """

    def __init__(self, image, weight, jacobian, **keys):
        raise RuntimeError("adapt to new system")
        super(MCMCBDFJointHybrid,self).__init__(image, weight, jacobian, "bdf", **keys)

    def get_band_pars(self, pars, band):
        """
        Extract pars for the specified band and convert to linear
        """
        raise RuntimeError("adapt to new style")
        Fbstart=5
        Fdstart=5+self.nband

        linpars = pars[ [0,1,2,3,4, Fbstart+band, Fdstart+band] ].copy()

        linpars[4] = 10.0**linpars[4]
        linpars[5] = 10.0**linpars[5]
        linpars[6] = 10.0**linpars[6]

        return linpars

    def get_gmix(self):
        """
        Get a gaussian mixture at the "best" parameter set, which
        definition depends on the sub-class
        """
        raise RuntimeError("adapt to new style")
        logpars=self._result['pars']
        pars=logpars.copy()
        pars[4] = 10.0**logpars[4]
        pars[5] = 10.0**logpars[5]
        pars[6] = 10.0**logpars[6]

        gm=gmix.make_gmix_model(pars, self.model)
        return gm

    def get_par_names(self):
        names=[r'$cen_1$',
               r'$cen_2$',
               r'$g_1$',
               r'$g_2$',
               r'$log_{10}(T)$']
        if self.nband == 1:
            names += [r'$log_{10}(F_b)$',r'$log_{10}(F_d)$']
        else:
            for ftype in ['b','d']:
                for band in xrange(self.nband):
                    names += [r'$log_{10}(F_%s^%s)$' % (ftype,band)]
        return names



class MCMCSimpleJointLinPars(MCMCSimple):
    """
    Simple with a joint prior on [g1,g2,T,Fb,Fd]
    """
    def __init__(self, image, weight, jacobian, model, **keys):
        raise RuntimeError("adapt to new system")
        super(MCMCSimpleJointLinPars,self).__init__(image, weight, jacobian, model, **keys)

        if self.full_guess is None:
            raise ValueError("For joint simple you must currently send a full guess")

        if self.joint_prior is None:
            raise ValueError("send joint prior for MCMCSimpleJointLinPars")

        self.prior_during=keys['prior_during']

    def _get_eabs_pars(self, pars):
        """
        don't include centroid, and only total ellipticity
        """
        if len(pars.shape) == 2:
            eabs_pars=zeros( (pars.shape[0], self.ndim-3) )
            eabs_pars[:,0] = sqrt(pars[:,2]**2 + pars[:,3]**2)
            eabs_pars[:,1:] = pars[:,4:]
        else:
            eabs_pars=zeros(self.ndim-3)

            eabs_pars[0] = sqrt(pars[2]**2 + pars[3]**2)
            eabs_pars[1:] = pars[4:]

        return eabs_pars

    def _get_priors(self, pars):
        """
        Apply simple priors
        """
        lnp=0.0

        if self.cen_prior is not None:
            lnp += self.cen_prior.get_lnprob(pars[0], pars[1])

        eabs_pars=self._get_eabs_pars(pars)

        jp=self.joint_prior
        if self.prior_during:
            lnp += jp.get_lnprob_scalar(eabs_pars)
        else:
            # this can raise a GMixRangeError exception
            jp.check_bounds_scalar(eabs_pars)

        return lnp

    def _get_PQR(self):
        """
        get the marginalized P,Q,R from Bernstein & Armstrong
        """

        print("get pqr joint simple")
        sh=self.shear_expand
        if sh is None:
            Pi,Qi,Ri = self.joint_prior.get_pqr_num(self._trials[:,2:])
        else:
            Pi,Qi,Ri = self.joint_prior.get_pqr_num(self._trials[:,2:],
                                                    s1=sh[0],
                                                    s2=sh[1])

        if self.prior_during:
            # We measured the posterior surface.  But the integrals are over
            # the likelihood.  So divide by the prior.
            #
            # Also note the p we divide by is in principle different from the
            # Pi above, which are evaluated at the shear expansion value

            prior_vals=self._get_joint_prior_vals()

            w,=numpy.where(prior_vals > 0.0)

            Pinv = 1.0/prior_vals[w]
            Pinv_sum=Pinv.sum()

            Pi = Pi[w]
            Qi = Qi[w,:]
            Ri = Ri[w,:,:]

            # this is not unity if expanding about some shear
            Pi *= Pinv
            Qi[:,0] *= Pinv
            Qi[:,1] *= Pinv

            Ri[:,0,0] *= Pinv
            Ri[:,0,1] *= Pinv
            Ri[:,1,0] *= Pinv
            Ri[:,1,1] *= Pinv

            P = Pi.sum()/Pinv_sum
            Q = Qi.sum(axis=0)/Pinv_sum
            R = Ri.sum(axis=0)/Pinv_sum
        else:
            P = Pi.mean()
            Q = Qi.mean(axis=0)
            R = Ri.mean(axis=0)

        return P,Q,R


    def _get_joint_prior_vals(self):
        if not hasattr(self,'joint_prior_vals'):
            eabs_pars=self._get_eabs_pars(self._trials)
            self.joint_prior_vals = self.joint_prior.get_prob_array(eabs_pars)
        return self.joint_prior_vals


class MCMCSimpleJointLogPars(MCMCSimple):
    """
    Simple with a joint prior on [g1,g2,T,Fb,Fd]
    """
    def __init__(self, image, weight, jacobian, model, **keys):
        raise RuntimeError("adapt to new style")
        super(MCMCSimpleJointLogPars,self).__init__(image, weight, jacobian, model, **keys)

        if self.full_guess is None:
            raise ValueError("For joint simple you must currently send a full guess")

        # we alrady have T_prior and counts_prior from base class

        # fraction of flux in bulge
        if self.joint_prior is None:
            raise ValueError("send joint prior for MCMCSimpleJointLogPars")

        self.prior_during=keys['prior_during']

    def get_band_pars(self, pars, band):
        """
        Extract pars for the specified band and convert to linear
        """
        raise RuntimeError("deal with non logpars")
        from .shape import eta1eta2_to_g1g2
        linpars=pars[ [0,1,2,3,4,5+band] ].copy()

        g1,g2=eta1eta2_to_g1g2(pars[2],pars[3])
        linpars[2] = g1
        linpars[3] = g2
        linpars[4] = 10.0**pars[4]
        linpars[5] = 10.0**pars[5]

        return linpars

    def _get_priors(self, pars):
        """
        Apply simple priors
        """
        lnp=0.0

        if self.cen_prior is not None:
            lnp += self.cen_prior.get_lnprob(pars[0], pars[1])

        jp=self.joint_prior
        if self.prior_during:
            lnp += jp.get_lnprob_scalar(pars[2:])
        else:
            # this can raise a GMixRangeError exception
            jp.check_bounds_scalar(pars[2:])

        return lnp

    def _get_PQR(self):
        """
        get the marginalized P,Q,R from Bernstein & Armstrong
        """

        print("get pqr joint simple")
        sh=self.shear_expand
        if sh is None:
            Pi,Qi,Ri = self.joint_prior.get_pqr_num(self._trials[:, 2:])
        else:
            Pi,Qi,Ri = self.joint_prior.get_pqr_num(self._trials[:, 2:],
                                                    s1=sh[0],
                                                    s2=sh[1])

        if self.prior_during:
            # We measured the posterior surface.  But the integrals are over
            # the likelihood.  So divide by the prior.
            #
            # Also note the p we divide by is in principle different from the
            # Pi above, which are evaluated at the shear expansion value

            prior_vals=self._get_joint_prior_vals()

            w,=numpy.where(prior_vals > 0.0)

            Pinv = 1.0/prior_vals[w]
            Pinv_sum=Pinv.sum()

            Pi = Pi[w]
            Qi = Qi[w,:]
            Ri = Ri[w,:,:]

            # this is not unity if expanding about some shear
            Pi *= Pinv
            Qi[:,0] *= Pinv
            Qi[:,1] *= Pinv

            Ri[:,0,0] *= Pinv
            Ri[:,0,1] *= Pinv
            Ri[:,1,0] *= Pinv
            Ri[:,1,1] *= Pinv

            P = Pi.sum()/Pinv_sum
            Q = Qi.sum(axis=0)/Pinv_sum
            R = Ri.sum(axis=0)/Pinv_sum
        else:
            P = Pi.mean()
            Q = Qi.mean(axis=0)
            R = Ri.mean(axis=0)

        return P,Q,R


    def _get_joint_prior_vals(self):
        if not hasattr(self,'joint_prior_vals'):
            self.joint_prior_vals = self.joint_prior.get_prob_array(self._trials[:,2:])
        return self.joint_prior_vals

    def get_par_names(self):
        names=[r'$cen_1$',
               r'$cen_2$',
               r'$\eta_1$',
               r'$\eta_2$',
               r'$log_{10}(T)$']
        if self.nband == 1:
            names += [r'$log_{10}(T)$']
        else:
            for band in xrange(self.nband):
                names += [r'$log_{10}(F_%s)$' % band]
        return names


class MCMCCoellip(MCMCSimple):
    """
    Add additional features to the base class to support simple models
    """
    def __init__(self, image, weight, jacobian, **keys):

        raise RuntimeError("adapt to new system")

        self.full_guess=keys.get('full_guess',None)
        self.ngauss=gmix.get_coellip_ngauss(self.full_guess.shape[1])
        self.g1i=2
        self.g2i=3

        if self.full_guess is None:
            raise ValueError("send full guess for coellip")

        MCMCBase.__init__(self, image, weight, jacobian, "coellip", **keys)

        self.priors_are_log=keys.get('priors_are_log',False)

        # should make this configurable
        self.first_T_prior=keys.get('first_T_prior',None)
        if self.first_T_prior is not None:
            print("will use first_T_prior")

        # halt tendency to wander off
        #self.sigma_max=keys.get('sigma_max',30.0)
        #self.T_max = 2*self.sigma_max**2

    def _get_guess(self):
        return self.full_guess

    def get_par_names(self):
        names=['cen1','cen2', 'g1','g2']

        for i in xrange(self.ngauss):
            names.append(r'$T_%s$' % i)
        for i in xrange(self.ngauss):
            names.append(r'$F_%s$' % i)

        return names


    def _set_npars(self):
        """
        nband should be set in set_lists, called before this
        """
        self.npars=self.full_guess.shape[1]

    def _get_priors(self, pars):
        """
        # go in simple
        add any priors that were sent on construction
        """

        lnp=0.0

        if self.cen_prior is not None:
            lnp += self.cen_prior.get_lnprob(pars[0], pars[1])

        if self.g_prior is not None:
            if self.g_prior_during:
                lnp += self.g_prior.get_lnprob_scalar2d(pars[2],pars[3])
            else:
                # may have bounds
                g = sqrt(pars[2]**2 + pars[3]**2)
                if g > self.g_prior.gmax:
                    raise GMixRangeError("g too big")

        # make sure the first one is constrained in size
        if self.first_T_prior is not None:
            lnp += self.first_T_prior.get_lnprob_scalar(pars[4])

        wbad,=where( pars[4:] <= 0.0 )
        if wbad.size != 0:
            raise GMixRangeError("gauss T or counts too small")


        if self.counts_prior is not None or self.T_prior is not None:
            ngauss=self.ngauss

            Tvals = pars[4:4+ngauss]

            #wbad,=where( (Tvals <= 0.0) | (Tvals > self.T_max) )
            wbad,=where( (Tvals <= 0.0) )
            if wbad.size != 0:
                raise GMixRangeError("out of bounds T values")

            counts_vals = pars[4+ngauss:]
            counts_total=counts_vals.sum()

            if self.counts_prior is not None:
                if len(self.counts_prior) > 1:
                    raise ValueError("make work with multiple bands")

                priors_are_log=self.priors_are_log
                cp=self.counts_prior[0]
                if priors_are_log:
                    if counts_total < 1.e-10:
                        raise GMixRangeError("counts too small")
                    logF = log10(counts_total)
                    lnp += cp.get_lnprob_scalar(logF)
                else:
                    lnp += cp.get_lnprob_scalar(counts_total)

            if self.T_prior is not None:
                T_total = (counts_vals*Tvals).sum()/counts_total

                if priors_are_log:
                    if T_total < 1.e-10:
                        raise GMixRangeError("T too small")
                    logT = log10(T_total)
                    lnp += self.T_prior.get_lnprob_scalar(logT)
                else:
                    lnp += self.T_prior.get_lnprob_scalar(T_total)

        return lnp


    def get_band_pars(self, pars, band):
        if band > 0:
            raise ValueError("support multi-band for coellip")
        return pars.copy()


######### Sampler below #############

class ISamplerMom(ISampler):
    def calc_result(self, weights=None):
        """
        Calculate the mcmc stats and the "best fit" stats
        """
        super(ISamplerMom,self).calc_result(weights=None)
        del self._result['g']
        del self._result['g_cov']

    def sample(self, nrand=None):
        """
        Get nrand random deviates from the distribution
        """

        if nrand is None:
            is_scalar=True
            nrand=1
        else:
            is_scalar=False

        vals = self._pdf.rvs(nrand)

        if is_scalar:
            vals = vals[0,:]

        return vals
