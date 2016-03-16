from __future__ import print_function
from .bootstrap import *

class BootstrapperTest(object):
    def __init__(self, obs,
                 use_logpars=False,
                 find_cen=False,
                 verbose=False,
                 **kw):
        """
        The data can be mutated: If a PSF fit is performed, the gmix will be
        set for the input PSF observation

        parameters
        ----------
        obs: observation(s)
            Either an Observation, ObsList, or MultiBandObsList The
            Observations must have a psf set.
            
            If the psf observations already have gmix objects set, there is no
            need to run fit_psfs()
        """

        self.use_logpars=use_logpars
        self.find_cen=find_cen
        self.verbose=verbose

        # this never gets modified in any way
        self.mb_obs_list_orig = get_mb_obs(obs)

        # this will get replaced if fit_psfs is run
        self.mb_obs_list=self.mb_obs_list_orig

        if self.find_cen:
            self._find_cen()

        self.model_fits={}

    def get_max_fitter(self):
        """
        get the maxlike fitter for the galaxy
        """
        if not hasattr(self,'max_fitter'):
            raise RuntimeError("you need to run fit_max successfully first")
        return self.max_fitter

    def get_psf_flux_result(self):
        """
        get the result fromrunning fit_gal_psf_flux
        """
        if not hasattr(self,'psf_flux_res'):
            self.fit_gal_psf_flux()

        return self.psf_flux_res

    def get_metacal_max_result(self):
        """
        get result of metacal with a max likelihood fitter
        """
        if not hasattr(self, 'metacal_max_res'):
            raise RuntimeError("you need to run fit_metacal_max first")
        return self.metacal_max_res


    def get_round_result(self):
        """
        get result of set_round_s2n()
        """
        if not hasattr(self, 'round_res'):
            raise RuntimeError("you need to run set_round_s2n")
        return self.round_res


    def set_round_s2n(self, fitter_type='max'):
        """
        set the s/n and (s/n)_T for the round model
        """

        if fitter_type=='isample':
            fitter = self.get_isampler()
        elif fitter_type=='max':
            fitter = self.get_max_fitter()
        else:
            raise ValueError("fitter_type should be isample or max")

        res=fitter.get_result()

        pars, pars_lin = self._get_round_pars(res['pars'])

        s2n, psf_T, flags = self._get_s2n_round(pars)

        T_r = pars_lin[4]
        self.round_res={'pars':pars,
                        'pars_lin':pars_lin,
                        'flags':flags,
                        's2n_r':s2n,
                        'T_r':T_r,
                        'psf_T_r':psf_T}

    def _get_s2n_round(self, pars_round):
        s2n=-9999.0
        psf_T=-9999.0
        flags=0

        s2n_sum = 0.0
        psf_T_sum=0.0
        wsum=0.0

        max_fitter=self.get_max_fitter()

        try:
            for band,obslist in enumerate(self.mb_obs_list):

                # pars_round could be in log and include all bands
                # this is unconvolved by a psf
                gm_round_band0 = self._get_gmix_round(max_fitter, pars_round, band)

                for obs in obslist:
                    psf_round = obs.psf.gmix.make_round()
                    gmix = gm_round_band0.convolve(psf_round)

                    # only the weight map is used
                    s2n_sum += gmix.get_model_s2n_sum(obs)

                    wt=obs.weight.sum()
                    psf_T=psf_round.get_T()
                    psf_T_sum += wt*psf_T
                    wsum += wt
            
            if s2n_sum <= 0.0:
                print("    failure: s2n_sum <= 0.0 :",s2n_sum)
                flags |= BOOT_S2N_LOW
            elif wsum <= 0.0:
                print("    failure: wsum <= 0.0 :",wsum)
                flags |= BOOT_WEIGHTS_LOW
            else:
                s2n=sqrt(s2n_sum)
                psf_T=psf_T_sum/wsum

        except GMixRangeError:
            print("    failure convolving round gmixes")
            flags |= BOOT_ROUND_CONVOLVE_FAIL 


        return s2n, psf_T, flags
 

    def _get_gmix_round(self, fitter, pars, band):
        res=fitter.get_result()
        band_pars = fitter.get_band_pars(pars,band)
        gm_round = GMixModel(band_pars, res['model'])
        return gm_round

    def _get_round_pars(self, pars_in):

        pars=pars_in.copy()
        pars_lin=pars.copy()

        if self.use_logpars:
            pars_lin[4:4+2] = exp(pars[4:4+2])

        g1,g2,T = pars_lin[2],pars_lin[3],pars_lin[4]

        f = get_round_factor(g1, g2)
        Tround = T*f

        pars[2]=0.0
        pars[3]=0.0
        pars_lin[2]=0.0
        pars_lin[3]=0.0

        pars_lin[4]=Tround

        if self.use_logpars:
            pars[4] = log(pars_lin[4])
        else:
            pars[4] = pars_lin[4]

        return pars, pars_lin


    def _find_cen(self, ntry=10):
        """
        run a single-gaussian em fit, just to find the center

        Modify the jacobian of our version of the observations accordingly.

        If it fails, don't modify anything
        """

        if len(self.mb_obs_list) > 1 or len(self.mb_obs_list[0]) > 1:
            raise RuntimeError("adapt to multiple observations")

        obs_orig = self.mb_obs_list[0][0]
        jacob=obs_orig.jacobian

        row0,col0=jacob.get_cen()

        T0=8.0

        for i in xrange(ntry):
            guess=[1.0*(1.0 + 0.01*srandu()),
                   row0*(1.0+0.5*srandu()),
                   col0*(1.0+0.5*srandu()),
                   T0/2.0*(1.0 + 0.1*srandu()),
                   0.1*srandu(),
                   T0/2.0*(1.0 + 0.1*srandu())]

            gm_guess = GMix(pars=guess)

            im,sky = prep_image(obs_orig.image)

            obs = Observation(im)

            fitter=GMixEM(obs)
            fitter.go(gm_guess, sky, maxiter=4000) 

            res=fitter.get_result()

            if res['flags']==0:
                break

        if res['flags']==0:
            gm=fitter.get_gmix()
            row,col=gm.get_cen()
            if self.verbose:
                print("        setting jacobian cen to:",row,col,
                      "numiter:",res['numiter'])
            jacob.set_cen(row=row,col=col)
        else:
            print("        failed to find cen")

    def fit_psfs(self, psf_model, Tguess,
                 skip_failed=True,
                 ntry=4,
                 fit_pars=None):
        """
        Fit all psfs.  If the psf observations already have a gmix
        then this step is not necessary

        parameters
        ----------
        psf_model: string
            The model to fit, e.g. 'em1','em2','em3','turb','gauss', etc.
        Tguess: float
            Guess for T
        skip_failed: bool
            If True, failures are just skipped when fitting the galaxy;
            in other words those observations will be ignored.  If False
            then an exception is raised
        ntry: integer
            Number of retries if the psf fit fails
        fit_pars: dict
            Fitting parameters for psf.
        """

        ntot=0
        new_mb_obslist=MultiBandObsList()

        mb_obs_list = self.mb_obs_list
        for band,obslist in enumerate(mb_obs_list):
            new_obslist=ObsList()

            for i,obs in enumerate(obslist):
                if not obs.has_psf():
                    raise RuntimeError("observation does not have a psf set")

                psf_obs = obs.get_psf()
                    
                try:                        
                    self._fit_one_psf(psf_obs, psf_model, Tguess, ntry, fit_pars)
                    new_obslist.append(obs)
                    ntot += 1
                except BootPSFFailure:
                    if not skip_failed:
                        raise
                    else:
                        mess=("    failed psf fit band %d obs %d, "
                              "skipping observation" % (band,i))
                        print(mess)
                        continue 
                

            new_mb_obslist.append(new_obslist)
        
        if ntot == 0:
            raise BootPSFFailure("no psf fits succeeded")

        self.mb_obs_list=new_mb_obslist

    def _fit_one_psf(self, psf_obs, psf_model, Tguess, ntry, fit_pars):
        """
        fit the psf using a PSFRunner or EMRunner

        TODO: add bootstrapping T guess as well, from unweighted moments
        """

        if 'em' in psf_model:
            runner=self._fit_one_psf_em(psf_obs, psf_model, Tguess, ntry, fit_pars)
        elif 'coellip' in psf_model:
            runner=self._fit_one_psf_coellip(psf_obs, psf_model, Tguess, ntry, fit_pars)
        else:
            runner=self._fit_one_psf_max(psf_obs, psf_model, Tguess, ntry, fit_pars)

        psf_fitter = runner.fitter
        res=psf_fitter.get_result()
        psf_obs.update_meta_data({'fitter':psf_fitter})
        
        if res['flags']==0:
            self.psf_fitter=psf_fitter
            gmix=self.psf_fitter.get_gmix()
            
            psf_obs.set_gmix(gmix)

        else:
            raise BootPSFFailure("failed to fit psfs")

    def _fit_one_psf_em(self, psf_obs, psf_model, Tguess, ntry, fit_pars):

        ngauss=get_em_ngauss(psf_model)
        em_pars={'tol': 1.0e-6, 'maxiter': 50000}
        if fit_pars is not None:
            em_pars.update(fit_pars)
        
        runner=EMRunner(psf_obs, Tguess, ngauss, em_pars)
        runner.go(ntry=ntry)

        return runner
    
    def _fit_one_psf_coellip(self, psf_obs, psf_model, Tguess, ntry, fit_pars):

        ngauss=get_coellip_ngauss(psf_model)
        lm_pars={'maxfev': 4000}

        if fit_pars is not None:
            lm_pars.update(fit_pars)
        
        runner=PSFRunnerCoellip(psf_obs, Tguess, ngauss, lm_pars)
        runner.go(ntry=ntry)

        return runner
 
    def _fit_one_psf_max(self, psf_obs, psf_model, Tguess, ntry, fit_pars):
        lm_pars={'maxfev': 4000}

        if fit_pars is not None:
            lm_pars.update(fit_pars)

        runner=PSFRunner(psf_obs, psf_model, Tguess, lm_pars)
        runner.go(ntry=ntry)

        return runner


    def fit_max(self, gal_model, pars,
                guess=None,
                guess_widths=None,
                prior=None,
                extra_priors=None,
                ntry=1):
        """
        fit the galaxy.  You must run fit_psf() successfully first

        extra_priors is ignored here but used in composite
        """

        self.max_fitter = self._fit_one_model_max(gal_model,
                                                  pars,
                                                  prior=prior,
                                                  ntry=ntry,
                                                  guess=guess,
                                                  guess_widths=guess_widths)
    def fit_metacal_max(self,
                        psf_model,
                        gal_model,
                        pars,
                        psf_Tguess,
                        psf_fit_pars=None,
                        metacal_pars=None,
                        prior=None,
                        psf_ntry=10,
                        ntry=1):
        """
        run metacalibration

        parameters
        ----------
        gal_model: string
            model to fit
        pars: dict
            parameters for the maximum likelihood fitter
        metacal_pars: dict, optional
            Parameters for metacal, default {'step':0.01}
        prior: prior on parameters, optional
            Optional prior to apply
        ntry: int, optional
            Number of times to retry fitting, default 1
        """

        if len(self.mb_obs_list) > 1 or len(self.mb_obs_list[0]) > 1:
            raise NotImplementedError("only a single obs for now")

        metacal_pars = self._get_metacal_pars(metacal_pars=metacal_pars)

        oobs = self.mb_obs_list[0][0]

        mcal_obs_dict = self.get_metacal_obsdict(oobs, metacal_pars)


        res = self._fit_metacal_max_one(metacal_pars,
                                        mcal_obs_dict,
                                        psf_model, gal_model, pars, psf_Tguess,
                                        prior, psf_ntry, ntry,
                                        psf_fit_pars)


        self.metacal_max_res = res

    def fit_metacal_max_addnoise(self,
                                 extra_noise,
                                 nrand,
                                 psf_model,
                                 gal_model,
                                 pars,
                                 psf_Tguess,
                                 psf_fit_pars=None,
                                 metacal_pars=None,
                                 prior=None,
                                 psf_ntry=10,
                                 ntry=1):
        """
        run metacalibration

        parameters
        ----------
        gal_model: string
            model to fit
        pars: dict
            parameters for the maximum likelihood fitter
        metacal_pars: dict, optional
            Parameters for metacal, default {'step':0.01}
        prior: prior on parameters, optional
            Optional prior to apply
        ntry: int, optional
            Number of times to retry fitting, default 1
        """

        if len(self.mb_obs_list) > 1 or len(self.mb_obs_list[0]) > 1:
            raise NotImplementedError("only a single obs for now")

        metacal_pars = self._get_metacal_pars(metacal_pars=metacal_pars)

        oobs = self.mb_obs_list[0][0]

        # these psfs do not have a gmix set
        obs_dict_orig = self.get_metacal_obsdict(oobs, metacal_pars)

        for i in xrange(nrand):

            obs_dict = self._add_noise_to_metacal_obsdict(obs_dict_orig, extra_noise)

            if self.verbose:
                print("    irand: %d/%d" % (i+1,nrand))

            tres = self._fit_metacal_max_one(metacal_pars,
                                             obs_dict,
                                             psf_model, gal_model, pars, psf_Tguess,
                                             prior, psf_ntry, ntry,
                                             psf_fit_pars)
            if i == 0:
                res=tres
            else:
                for key in res:
                    res[key] = res[key] + tres[key]
            
        if nrand > 1:
            for key in res:
                res[key] = res[key]/float(nrand)

        self.metacal_max_res = res

    def _get_metacal_pars(self, metacal_pars=None):
        mpars={'step':0.01}
        if metacal_pars is not None:
            mpars.update(metacal_pars)

        return mpars

    def _add_noise_to_metacal_obsdict(self, obs_dict, extra_noise):
        noise_image = self._get_noise_image(obs_dict['1p'].image.shape,
                                            extra_noise)

        nobs_dict={}
        for key in obs_dict:

            obs=obs_dict[key]

            new_weight = self._get_degraded_weight_image(obs, extra_noise)
            new_obs = self._get_degraded_obs(obs, noise_image, new_weight)

            nobs_dict[key] = new_obs

        return nobs_dict



    def _fit_metacal_max_one(self,
                             metacal_pars,
                             obs_dict,
                             psf_model, gal_model, max_pars, 
                             psf_Tguess, prior, psf_ntry, ntry, 
                             psf_fit_pars):

        step = metacal_pars['step']

        fits = self._do_metacal_fits(obs_dict,
                                     psf_model, gal_model, max_pars, psf_Tguess,
                                     prior, psf_ntry, ntry,
                                     psf_fit_pars)

        pars=fits['pars']
        pars_mean = (pars['1p']+
                     pars['1m']+
                     pars['2p']+
                     pars['2m'])/4.0

        pars_cov=fits['pars_cov']
        pars_cov_mean = (pars_cov['1p']+
                         pars_cov['1m']+
                         pars_cov['2p']+
                         pars_cov['2m'])/4.0

        if self.verbose:
            print_pars(pars_mean, front='    mcmean:   ')

        R=zeros( (2,2) ) 
        Rpsf=zeros(2)

        fac = 1.0/(2.0*step)

        R[0,0] = (pars['1p'][2]-pars['1m'][2])*fac
        R[0,1] = (pars['1p'][3]-pars['1m'][3])*fac
        R[1,0] = (pars['2p'][2]-pars['2m'][2])*fac
        R[1,1] = (pars['2p'][3]-pars['2m'][3])*fac

        Rpsf[0] = (pars['1p_psf'][2]-pars['1m_psf'][2])*fac
        Rpsf[1] = (pars['2p_psf'][3]-pars['2m_psf'][3])*fac

        pars_noshear = pars['noshear']

        c = pars_mean[2:2+2] - pars_noshear[2:2+2]

        res = {'mcal_pars':pars_mean,
               'mcal_pars_cov':pars_cov_mean,
               'mcal_g':pars_mean[2:2+2],
               'mcal_g_cov':pars_cov_mean[2:2+2, 2:2+2],
               'mcal_pars_noshear':pars_noshear,
               'mcal_c':c,
               'mcal_R':R,
               'mcal_Rpsf':Rpsf,
               'mcal_gpsf':fits['gpsf'],
               'mcal_s2n_r':fits['s2n_r'],
               'mcal_s2n_simple':fits['s2n_simple'],
               'mcal_T_r':fits['T_r'],
               'mcal_psf_T_r':fits['psf_T_r']}
        return res



    def _do_metacal_fits(self, obs_dict, psf_model, gal_model, pars, 
                         psf_Tguess, prior, psf_ntry, ntry, 
                         psf_fit_pars):

        bdict={}
        for key in obs_dict:
            boot = Bootstrapper(obs_dict[key],
                                use_logpars=self.use_logpars,
                                find_cen=self.find_cen,
                                verbose=self.verbose)

            boot.fit_psfs(psf_model, psf_Tguess, ntry=psf_ntry, fit_pars=psf_fit_pars)
            boot.fit_max(gal_model, pars, prior=prior, ntry=ntry)
            boot.set_round_s2n()
            
            # verbose can be bool or a number
            if self.verbose > 1:
                if 'psf' in key:
                    front='    psf mcpars:'
                else:
                    front='        mcpars:'
                print_pars(boot.get_max_fitter().get_result()['pars'],
                           front=front)

            bdict[key] = boot

        res={'pars':{}, 'pars_cov':{}}
        s2n_r_mean   = 0.0
        T_r_mean     = 0.0
        psf_T_r_mean = 0.0
        gpsf_mean = zeros(2)
        npsf=0
        navg=0

        for i,key in enumerate(bdict):
            boot = bdict[key]
            tres=boot.get_max_fitter().get_result()

            res['pars'][key] = tres['pars']
            res['pars_cov'][key] = tres['pars_cov']

            #
            # averaging
            #

            if key=='noshear' or 'psf' in key:
                # don't include noshear in the averages
                # don't average over psf sheared model
                continue

            for obslist in boot.mb_obs_list:
                for obs in obslist:
                    g1,g2,T=obs.psf.gmix.get_g1g2T()
                    gpsf_mean[0] += g1
                    gpsf_mean[1] += g2
                    npsf+=1

            rres=boot.get_round_result()

            s2n_r_mean   += rres['s2n_r']
            T_r_mean     += rres['T_r']
            psf_T_r_mean += rres['psf_T_r']
            navg += 1

        assert navg==4,"expected 4 to average"

        bnoshear=bdict['noshear']
        res['s2n_simple'] = bnoshear.mb_obs_list.get_s2n()

        res['s2n_r']   = s2n_r_mean/navg
        res['T_r']     = T_r_mean/navg
        res['psf_T_r'] = psf_T_r_mean/navg
        res['gpsf'] = gpsf_mean/npsf

        return res

    def get_metacal_obsdict(self, oobs, metacal_pars):
        """
        get Observations for the sheared images


        for same noise we add the noise *after* shearing/convolving etc.

        otherwise we degrade the original and metacal happens on that
        just as with the real data
        """
        from .metacal import Metacal
        from .shape import Shape

        step=metacal_pars['step']

        print("        step:",step)

        mc=Metacal(oobs)

        sh1m=Shape(-step,  0.00 )
        sh1p=Shape( step,  0.00 )

        sh2m=Shape(0.0, -step)
        sh2p=Shape(0.0,  step)

        obs1p, obs_noshear = mc.get_obs_galshear(sh1p, get_unsheared=True)
        obs1m = mc.get_obs_galshear(sh1m)
        obs2p = mc.get_obs_galshear(sh2p)
        obs2m = mc.get_obs_galshear(sh2m)

        obs1p_psf = mc.get_obs_psfshear(sh1p)
        obs1m_psf = mc.get_obs_psfshear(sh1m)
        obs2p_psf = mc.get_obs_psfshear(sh2p)
        obs2m_psf = mc.get_obs_psfshear(sh2m)
 
        obs_dict = {
                    '1p':obs1p,
                    '1m':obs1m,
                    '2p':obs2p,
                    '2m':obs2m,

                    '1p_psf':obs1p_psf,
                    '1m_psf':obs1m_psf,
                    '2p_psf':obs2p_psf,
                    '2m_psf':obs2m_psf,

                    'noshear': obs_noshear,
                   }

        return obs_dict

    def _get_degraded_obs(self, obs, noise_image, new_weight):
        """
        get a new obs with extra noise added to image and the weight
        map modified appropriately
        """


        new_im = obs.image + noise_image

        new_obs = Observation(new_im,
                              weight=new_weight,
                              jacobian=obs.jacobian)
        if obs.has_psf():
            new_obs.set_psf(obs.psf)

        return new_obs

    def _get_noise_image(self, dims, noise):
        """
        get a noise image for use in degrading a high s/n image
        """

        noise_image = numpy.random.normal(loc=0.0,
                                          scale=noise,
                                          size=dims)

        return noise_image

    def _get_degraded_weight_image(self, obs, noise):
        """
        get a new weight map reflecting additional noise
        """

        new_weight = obs.weight.copy()
        w=numpy.where(new_weight > 0)

        if w[0].size > 0:
            new_weight[w] = 1.0/(1.0/new_weight[w] + noise**2)

        return new_weight





    def _fit_one_model_max(self, gal_model, pars, guess=None, prior=None, ntry=1, obs=None, guess_widths=None):
        """
        fit the galaxy.  You must run fit_psf() successfully first
        """

        if obs is None:
            obs = self.mb_obs_list

        if not hasattr(self,'psf_flux_res'):
            self.fit_gal_psf_flux()

        guesser=self._get_max_guesser(guess=guess, prior=prior, widths=guess_widths)

        runner=MaxRunner(obs, gal_model, pars, guesser,
                         prior=prior,
                         use_logpars=self.use_logpars)

        runner.go(ntry=ntry)

        fitter=runner.fitter

        res=fitter.get_result()

        if res['flags'] != 0:
            raise BootGalFailure("failed to fit galaxy with maxlike")

        return fitter





    def fit_gal_psf_flux(self, normalize_psf=True):
        """
        use psf as a template, measure flux (linear)
        """


        mbo = self.mb_obs_list
        nband = len(mbo)

        if not mbo[0][0].psf.has_gmix():
            raise RuntimeError("you need to fit the psfs first")

        flags=[]
        psf_flux = zeros(nband) - 9999.0
        psf_flux_err = zeros(nband)

        for i in xrange(nband):
            obs_list = mbo[i]

            fitter=fitting.TemplateFluxFitter(obs_list, do_psf=True, normalize_psf=normalize_psf)
            fitter.go()

            res=fitter.get_result()
            tflags = res['flags']
            flags.append( tflags )

            if tflags == 0:

                psf_flux[i] = res['flux']
                psf_flux_err[i] = res['flux_err']

                #print("    psf flux %d: %.3f +/- %.3f" % (i,res['flux'],res['flux_err']))
            else:
                print("failed to fit psf flux for band",i)

        self.psf_flux_res={'flags':flags,
                           'psf_flux':psf_flux,
                           'psf_flux_err':psf_flux_err}

    def _get_max_guesser(self, guess=None, prior=None, widths=None):
        """
        get a guesser that uses the psf T and galaxy psf flux to
        generate a guess, drawing from priors on the other parameters
        """

        if self.use_logpars:
            scaling='log'
        else:
            scaling='linear'

        if guess is not None:
            guesser=ParsGuesser(guess, scaling=scaling, widths=widths)
        else:
            psf_T = self.mb_obs_list[0][0].psf.gmix.get_T()

            pres=self.get_psf_flux_result()


            if prior is None:
                guesser=TFluxGuesser(psf_T,
                                     pres['psf_flux'],
                                     scaling=scaling)
            else:
                guesser=TFluxAndPriorGuesser(psf_T,
                                             pres['psf_flux'],
                                             prior,
                                             scaling=scaling)
        return guesser



    def try_replace_cov(self, cov_pars, fitter=None):
        """
        the lm cov often mis-estimates the error on the ellipticity parameters,
        try to replace it
        """
        if not hasattr(self,'max_fitter'):
            raise RuntimeError("you need to fit with the max like first")

        if fitter is None:
            fitter=self.max_fitter

        # reference to res
        res=fitter.get_result()

        fitter.calc_cov(cov_pars['h'],cov_pars['m'])

        if res['flags'] != 0:
            print("        cov replacement failed")
            res['flags']=0


