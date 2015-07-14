"""
TODO

    - make a tester for it
    - test it in nsim
    - make it possible to specify the guess type (not just psf)

"""
from __future__ import print_function

from pprint import pprint
import numpy
from numpy import where, array, sqrt, exp, log, linspace, zeros
from numpy import isfinite, median
from numpy.linalg import LinAlgError

from . import fitting
from .gmix import GMix, GMixModel, GMixCM
from .em import GMixEM, prep_image
from .observation import Observation, ObsList, MultiBandObsList, get_mb_obs
from .priors import srandu
from .shape import get_round_factor
from .guessers import TFluxGuesser, TFluxAndPriorGuesser, ParsGuesser, RoundParsGuesser
from .gexceptions import GMixRangeError, BootPSFFailure, BootGalFailure

from . import roundify

BOOT_S2N_LOW = 2**0
BOOT_R2_LOW = 2**1
BOOT_R4_LOW = 2**2
BOOT_TS2N_ROUND_FAIL = 2**3
BOOT_ROUND_CONVOLVE_FAIL = 2**4
BOOT_WEIGHTS_LOW= 2**5



class Bootstrapper(object):
    def __init__(self, obs, use_logpars=False):
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
        self.mb_obs_list_orig = get_mb_obs(obs)

        # this will get replaced if fit_psfs is run
        self.mb_obs_list=self.mb_obs_list_orig


        self.model_fits={}

    def get_isampler(self):
        """
        get the importance sampler
        """
        if not hasattr(self,'isampler'):
            raise RuntimeError("you need to run isample() successfully first")
        return self.isampler

    def get_psampler(self):
        """
        get the prior samples sampler
        """
        if not hasattr(self,'psampler'):
            raise RuntimeError("you need to run psample() successfully first")
        return self.psampler

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

    def get_round_result(self):
        """
        get result of set_round_s2n()
        """
        if not hasattr(self, 'round_res'):
            raise RuntimeError("you need to run set_round_s2n")
        return self.round_res

    def set_round_s2n(self, max_pars,
                      ntry=4,
                      fitter_type='max',
                      method='simple',
                      round_prior=None):
        """
        set the s/n and (s/n)_T for the round model

        round_prior used for sim fit
        """

        assert method=="simple","method for round s/n must be simple"

        if fitter_type=='isample':
            fitter = self.get_isampler()
        elif fitter_type=='max':
            fitter = self.get_max_fitter()
        else:
            raise ValueError("fitter_type should be isample or max")

        res=fitter.get_result()

        pars, pars_lin = self._get_round_pars(res['pars'])

        '''
        if method=='sim':
            s2n, Ts2n, psf_T, flags = self._get_s2n_Ts2n_r_sim(fitter,
                                                               pars, ntry,
                                                               max_pars,
                                                               round_prior=round_prior)
        elif method=='alg':
            s2n, Ts2n, flags = self._get_s2n_Ts2n_r_alg(gm0_round)
        '''

        s2n, psf_T, flags = self._get_s2n_round(pars)

        self.round_res={'pars':pars,
                        'flags':flags,
                        's2n_r':s2n,
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
 
    def _get_s2n_Ts2n_r_sim(self, fitter, pars_round, ntry, max_pars,
                            round_prior=None):

        s2n=-9999.0
        Ts2n=-9999.0
        psf_T=-9999.0
        flags=0

        # first set the gmix on all observations
        max_fitter=self.get_max_fitter()
        print("    setting gmix in each obs")
        for band,obslist in enumerate(self.mb_obs_list):
            # pars_round could be in log and include all bands
            gm_round_band = self._get_gmix_round(max_fitter, pars_round, band)
            for obs in obslist:
                obs.gmix = gm_round_band.copy()

        # now get roundified observations
        print("    getting round obs")
        mb_obs_list = roundify.get_round_mb_obs_list(self.mb_obs_list,
                                                     sim_image=True)

        print("    getting s2n")
        s2n_sum=0.0
        psf_T_sum=0.0
        wsum=0.0
        try:
            for obslist in mb_obs_list:
                for obs in obslist:
                    gm = obs.gmix.convolve( obs.psf.gmix )

                    s2n_sum += obs.gmix.get_model_s2n_sum(obs)

                    wt=obs.weight.sum()
                    psf_T=obs.psf.gmix.get_T()
                    psf_T_sum += wt*psf_T
                    wsum += wt

        except GMixRangeError:
            print("    failure convolving round gmixes")
            s2n_sum=-9999.0

        if s2n_sum <= 0.0:
            print("    failure: s2n_sum <= 0.0 :",s2n_sum)
            flags |= BOOT_S2N_LOW
        else:
            s2n=sqrt(s2n_sum)

            psf_T=psf_T_sum/wsum

            # and finally, do the fit 
            try:
                round_fitter=self._fit_sim_round(fitter,
                                                 pars_round,
                                                 mb_obs_list,
                                                 ntry,
                                                 max_pars,
                                                 round_prior=round_prior)
                res=round_fitter.get_result()
                if res['flags'] != 0:
                    print("        round fit fail")
                    flags |= BOOT_TS2N_ROUND_FAIL 
                else:
                    import covmatrix
                    try:
                        tcov=covmatrix.calc_cov(round_fitter.calc_lnprob, res['pars'], 1.0e-3)
                    except LinAlgError:
                        tcov=None

                    if tcov is not None and tcov[2,2] > 0.0:
                        cov=tcov
                    else:
                        print("    replace cov failed, using LM cov")
                        cov=res['pars_cov']

                    if self.use_logpars:
                        Ts2n = sqrt(1.0/cov[2,2])
                    else:
                        Ts2n = res['pars'][2]/sqrt(cov[2,2])
            except GMixRangeError as err:
                print(str(err))
                flags |= BOOT_TS2N_ROUND_FAIL 

        return s2n, Ts2n, psf_T, flags

    def _fit_sim_round(self, fitter, pars_round, mb_obs_list,
                       ntry, max_pars, round_prior=None):

        res=fitter.get_result()
        runner=MaxRunnerRound(pars_round,
                              mb_obs_list,
                              res['model'],
                              max_pars,
                              prior=round_prior,
                              use_logpars=self.use_logpars)

        runner.go(ntry=ntry)

        return runner.fitter
 
    def _get_s2n_Ts2n_r_alg(self, gm0_round):
        """
        get the round s2n and Ts2n
        """
        raise RuntimeError("this isn't multi-band")

        Tround = gm0_round.get_T()

        flags=0

        s2n_sum=0.0
        r4sum=0.0
        r2sum=0.0

        for obslist in self.mb_obs_list:
            for obs in obslist:
                psf_gm=obs.psf.gmix

                # only makes sense for symmetric psf
                psf_gm_round = psf_gm.make_round()

                gm = gm0_round.convolve(psf_gm_round)

                # these use only the weight maps. Use the same weight map
                # for gal and psf
                t_s2n_sum, t_r2sum, t_r4sum = \
                    gm.get_model_s2n_Tvar_sums(obs)

                s2n_sum += t_s2n_sum
                r2sum += t_r2sum
                r4sum += t_r4sum

        if s2n_sum <= 0.0:
            print("    failure: s2n_sum <= 0.0 :",s2n_sum)
            flags |= BOOT_S2N_LOW
            s2n=-9999.0
            Ts2n=-9999.0
        else:
            s2n=sqrt(s2n_sum)

            # weighted means
            r2_mean = r2sum/s2n_sum
            r4_mean = r4sum/s2n_sum

            if r2_mean <= 0.0:
                print("    failure: round r2 <= 0.0 :",r2_mean)
                flags |= BOOT_R2_LOW
                Ts2n=-9999.0
            elif r4_mean <= 0.0:
                print("    failure: round r2 == 0.0 :",r2_mean)
                flags |= BOOT_R4_LOW
                Ts2n=-9999.0
            else:

                # this one partially accounts for T-F covariance
                r2sq = r2_mean**2
                Ts2n = Tround * s2n * sqrt(r4_mean-r2sq) / (4. * r2sq)

        return s2n, Ts2n, flags

    def set_round_s2n_old(self, prior=None, ntry=4):
        """
        set the s/n and (s/n)_T for the round model

        the s/n measure is stable, the size will require a prior and may
        need retries
        """

        raise RuntimeError("adapt to multiple observations")

        obs=self.gal_obs

        max_fitter=self.get_max_fitter()
        res=max_fitter.get_result()

        pars, pars_lin = self._get_round_pars(res['pars'])

        gm0_round = self._get_gmix_round(res, pars_lin)

        gmpsf_round = obs.psf.gmix.make_round()

        gm_round = gm0_round.convolve(gmpsf_round)

        # first the overall s/n, this is stable
        res['round_pars'] = pars
        res['s2n_r'] = gm_round.get_model_s2n(obs)

        # now the covariance matrix, which can be more unstable
        cov=self._sim_cov_round(obs,
                                gm_round, gmpsf_round,
                                res, pars,
                                prior=prior,
                                ntry=ntry)

        if cov is None:
            print("    failed to fit round (S/N)_T")
            res['T_s2n_r'] = -9999.0
        else:
            if self.use_logpars:
                Ts2n_round = sqrt(1.0/cov[4,4])
            else:
                Ts2n_round = pars_lin[4]/sqrt(cov[4,4])
            res['T_s2n_r'] = Ts2n_round

    def _get_gmix_round(self, fitter, pars, band):
        res=fitter.get_result()
        band_pars = fitter.get_band_pars(pars,band)
        gm_round = GMixModel(band_pars, res['model'])
        return gm_round

    def _get_gmix_round_old(self, res, pars):
        gm_round = GMixModel(pars, res['model'])
        return gm_round

    def _sim_cov_round_old(self,
                       obs,
                       gm_round, gmpsf_round,
                       res, pars_round,
                       prior=None,
                       ntry=4):
        """
        deprecated
        gm_round is convolved
        """
        
        im0=gm_round.make_image(obs.image.shape,
                                jacobian=obs.jacobian)
        
        
        # image here is not used
        psf_obs = Observation(im0, gmix=gmpsf_round)

        noise=1.0/sqrt( median(obs.weight) )

        cov=None
        for i in xrange(ntry):
            nim = numpy.random.normal(scale=noise,
                                      size=im0.shape)

            im = im0 + nim
            

            newobs = Observation(im,
                                 weight=obs.weight,
                                 jacobian=obs.get_jacobian(),
                                 psf=psf_obs)
            
            fitter=self._get_round_fitter_old(newobs, res, prior=prior)

            # we can't recover from this error
            try:
                fitter._setup_data(pars_round)
            except GMixRangeError:
                break

            try:
                tcov=fitter.get_cov(pars_round, 1.0e-3, 5.0)
                if tcov[4,4] > 0:
                    cov=tcov
                    break
            except LinAlgError:
                pass

        return cov

    def _get_round_fitter_old(self, obs, res, prior=None):
        fitter=fitting.LMSimple(obs, res['model'],
                                prior=prior,
                                use_logpars=self.use_logpars)
        return fitter


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


    def find_cen(self):
        """
        run a single-gaussian em fit, just to find the center

        Modify the jacobian center accordingly.

        If it fails, don't modify anything
        """

        raise RuntimeError("adapt to multiple observations")

        jacob=self.gal_obs.jacobian

        row,col=jacob.get_cen()

        guess=array([1.0, # p
                     row, # row in current jacobian coords
                     col, # col in current jacobian coords
                     4.0,
                     0.0,
                     4.0])

        gm_guess = GMix(pars=guess)

        im,sky = prep_image(self.gal_obs.image)
        obs = Observation(im)
        fitter=GMixEM(obs)
        fitter.go(gm_guess, sky, maxiter=4000) 
        res=fitter.get_result()
        if res['flags']==0:
            gm=fitter.get_gmix()
            row,col=gm.get_cen()
            print("        setting jacobian cen to:",row,col,
                  "numiter:",res['numiter'])
            jacob.set_cen(row,col)
        else:
            print("        failed to find cen")

    def fit_psfs(self, psf_model, Tguess, skip_failed=True, ntry=4):
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
        """

        ntot=0
        new_mb_obslist=MultiBandObsList()
        for band,obslist in enumerate(self.mb_obs_list_orig):
            new_obslist=ObsList()

            for i,obs in enumerate(obslist):
                if not obs.has_psf():
                    raise RuntimeError("observation does not have a psf set")

                psf_obs = obs.get_psf()
                try:
                    self._fit_one_psf(psf_obs, psf_model, Tguess, ntry)
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

    def _fit_one_psf(self, psf_obs, psf_model, Tguess, ntry):
        """
        fit the psf using a PSFRunner or EMRunner

        TODO: add bootstrapping T guess as well, from unweighted moments
        """

        if 'em' in psf_model:
            runner=self._fit_one_psf_em(psf_obs, psf_model, Tguess, ntry)
        else:
            runner=self._fit_one_psf_max(psf_obs, psf_model, Tguess, ntry)

        psf_fitter = runner.fitter
        res=psf_fitter.get_result()

        if res['flags']==0:
            self.psf_fitter=psf_fitter
            gmix=self.psf_fitter.get_gmix()

            psf_obs.set_gmix(gmix)

        else:
            raise BootPSFFailure("failed to fit psfs")

    def _fit_one_psf_em(self, psf_obs, psf_model, Tguess, ntry):

        ngauss=get_em_ngauss(psf_model)
        em_pars={'tol': 1.0e-6, 'maxiter': 50000}

        runner=EMRunner(psf_obs, Tguess, ngauss, em_pars)
        runner.go(ntry=ntry)

        return runner


    def _fit_one_psf_max(self, psf_obs, psf_model, Tguess, ntry):
        lm_pars={'maxfev': 4000}
        runner=PSFRunner(psf_obs, psf_model, Tguess, lm_pars)
        runner.go(ntry=ntry)

        return runner


    def fit_max(self, gal_model, pars, guess=None, prior=None, extra_priors=None, ntry=1):
        """
        fit the galaxy.  You must run fit_psf() successfully first

        extra_priors is ignored here but used in composite
        """

        self.max_fitter = self._fit_one_model_max(gal_model,
                                                  pars,
                                                  guess=guess,
                                                  prior=prior,
                                                  ntry=ntry)
        res=self.max_fitter.get_result()

    def fit_max_fixT(self, gal_model, pars, T,
                     guess=None, prior=None, extra_priors=None, ntry=1):
        """
        fit the galaxy.  You must run fit_psf() successfully first

        extra_priors is ignored here but used in composite
        """

        if not hasattr(self,'psf_flux_res'):
            self.fit_gal_psf_flux()

        guesser=self._get_max_guesser(guess=guess, prior=prior)

        runner=MaxRunnerFixT(self.mb_obs_list, gal_model, pars, guesser, T,
                             prior=prior,
                             use_logpars=self.use_logpars)

        runner.go(ntry=ntry)

        fitter=runner.fitter

        res=fitter.get_result()

        if res['flags'] != 0:
            raise BootGalFailure("failed to fit galaxy with maxlike")



        self.max_fitter = fitter

    def fit_max_gonly(self, gal_model, max_pars, pars_in,
                      guess=None, prior=None, extra_priors=None, ntry=1):
        """
        fit the galaxy.  You must run fit_psf() successfully first

        extra_priors is ignored here but used in composite
        """

        if not hasattr(self,'psf_flux_res'):
            self.fit_gal_psf_flux()

        if prior is not None:
            guesser=prior.sample
        else:
            guesser=self._get_max_guesser(guess=guess, prior=prior)

        runner=MaxRunnerGOnly(self.mb_obs_list, gal_model, max_pars, guesser, pars_in,
                              prior=prior,
                              use_logpars=self.use_logpars)

        runner.go(ntry=ntry)

        fitter=runner.fitter

        res=fitter.get_result()

        if res['flags'] != 0:
            raise BootGalFailure("failed to fit galaxy with maxlike")

        self.max_fitter = fitter



    def _fit_one_model_max(self, gal_model, pars, guess=None, prior=None, ntry=1):
        """
        fit the galaxy.  You must run fit_psf() successfully first
        """

        if not hasattr(self,'psf_flux_res'):
            self.fit_gal_psf_flux()

        guesser=self._get_max_guesser(guess=guess, prior=prior)

        runner=MaxRunner(self.mb_obs_list, gal_model, pars, guesser,
                         prior=prior,
                         use_logpars=self.use_logpars)

        runner.go(ntry=ntry)

        fitter=runner.fitter

        res=fitter.get_result()

        if res['flags'] != 0:
            raise BootGalFailure("failed to fit galaxy with maxlike")

        return fitter



    def isample(self, ipars, prior=None, verbose=True):
        """
        bootstrap off the maxlike run
        """

        max_fitter=self.max_fitter
        use_fitter=max_fitter

        niter=len(ipars['nsample'])
        for i,nsample in enumerate(ipars['nsample']):
            sampler=self._make_isampler(use_fitter, ipars, verbose=verbose)
            if sampler is None:
                raise BootGalFailure("isampling failed")

            sampler.make_samples(nsample)

            sampler.set_iweights(max_fitter.calc_lnprob)
            sampler.calc_result()

            tres=sampler.get_result()

            if verbose:
                print("    eff iter %d: %.2f" % (i,tres['efficiency']))
            use_fitter = sampler

        maxres=max_fitter.get_result()
        tres['model'] = maxres['model']

        self.isampler=sampler

    def _make_isampler(self, fitter, ipars, verbose=True):
        from .fitting import ISampler
        from numpy.linalg import LinAlgError

        res=fitter.get_result()
        icov = res['pars_cov']

        try:
            sampler=ISampler(res['pars'],
                             icov,
                             ipars['df'],
                             min_err=ipars['min_err'],
                             max_err=ipars['max_err'],
                             ifactor=ipars.get('ifactor',1.0),
                             asinh_pars=ipars.get('asinh_pars',[]),
                             verbose=verbose)
        except LinAlgError:
            print("        bad cov")
            sampler=None

        return sampler

    def psample(self, psample_pars, samples, verbose=True):
        """
        bootstrap off the maxlike run
        """
        from .fitting import PSampler, MaxSimple

        max_fitter=self.get_max_fitter()
        res=max_fitter.get_result()

        model=res['model']
        tfitter=MaxSimple(self.mb_obs_list,
                          model,
                          use_logpars=self.use_logpars)
        tfitter._setup_data(res['pars'])
 
        sampler=PSampler(res['pars'],
                         res['pars_err'],
                         samples,
                         verbose=verbose,
                         **psample_pars)

        sampler.calc_loglikes(tfitter.calc_lnprob)

        self.psampler=sampler

        res=sampler.get_result()
        if res['flags'] != 0:
            raise BootGalFailure("psampling failed")



    def fit_gal_psf_flux(self):
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

            fitter=fitting.TemplateFluxFitter(obs_list, do_psf=True)
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

    def _get_max_guesser(self, guess=None, prior=None):
        """
        get a guesser that uses the psf T and galaxy psf flux to
        generate a guess, drawing from priors on the other parameters
        """

        if self.use_logpars:
            scaling='log'
        else:
            scaling='linear'

        if guess is not None:
            #print("    using ParsGuesser")
            guesser=ParsGuesser(guess, scaling=scaling)
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

class BootstrapperGaussMom(Bootstrapper):
    def __init__(self, obs):
        super(BootstrapperGaussMom,self).__init__(obs)

    def fit_max(self, pars, guess=None, prior=None, extra_priors=None, ntry=1):
        """
        fit the galaxy.  You must run fit_psf() successfully first

        extra_priors is ignored here but used in composite
        """

        self.max_fitter = self._fit_one_model_max(pars,
                                                  guess=guess,
                                                  prior=prior,
                                                  ntry=ntry)
        res=self.max_fitter.get_result()

    def _fit_one_model_max(self, pars, guess=None, prior=None, ntry=1):
        """
        fit the galaxy.  You must run fit_psf() successfully first
        """

        if not hasattr(self,'psf_flux_res'):
            self.fit_gal_psf_flux()

        guesser=self._get_max_guesser(guess=guess, prior=prior)

        runner=MaxRunnerGaussMom(self.mb_obs_list, pars, guesser,
                                 prior=prior,
                                 use_logpars=self.use_logpars)

        runner.go(ntry=ntry)

        fitter=runner.fitter

        res=fitter.get_result()

        if res['flags'] != 0:
            raise BootGalFailure("failed to fit galaxy with maxlike")

        return fitter

    def _make_isampler(self, fitter, ipars, verbose=True):
        from .fitting import ISamplerMom
        from numpy.linalg import LinAlgError

        res=fitter.get_result()
        icov = res['pars_cov']

        try:
            sampler=ISamplerMom(res['pars'],
                                icov,
                                ipars['df'],
                                min_err=ipars['min_err'],
                                max_err=ipars['max_err'],
                                ifactor=ipars.get('ifactor',1.0),
                                asinh_pars=ipars.get('asinh_pars',[]),
                                verbose=verbose)
        except LinAlgError:
            print("        bad cov")
            sampler=None

        return sampler


    def _get_max_guesser(self, guess=None, prior=None):
        """
        get a guesser that uses the psf T and galaxy psf flux to
        generate a guess, drawing from priors on the other parameters
        """
        from .guessers import MomGuesser

        if guess is None:
            psf_T = self.mb_obs_list[0][0].psf.gmix.get_T()

            pres=self.get_psf_flux_result()

            guess=array( [0.0, 0.0, 0.0, 0.0, psf_T, pres['psf_flux']] )

        guesser=MomGuesser(guess, prior=prior)
        return guesser



class CompositeBootstrapper(Bootstrapper):
    def __init__(self, obs,
                 use_logpars=False,
                 fracdev_prior=None,
                 fracdev_grid=None):

        super(CompositeBootstrapper,self).__init__(obs,
                                                   use_logpars=use_logpars)

        self.fracdev_prior=fracdev_prior
        if fracdev_grid is not None:
            self.fracdev_tests=linspace(fracdev_grid['min'],
                                        fracdev_grid['max'],
                                        fracdev_grid['num'])
        else:
            self.fracdev_tests=linspace(-1.0,1.5,26)

    def fit_max(self,
                model,
                pars,
                guess=None,
                prior=None,
                extra_priors=None,
                ntry=1):
        """
        fit the galaxy.  You must run fit_psf() successfully first
        """

        assert model=='cm','model must be cm'
        if extra_priors is None:
            exp_prior=prior
            dev_prior=prior
        else:
            exp_prior=extra_priors['exp']
            dev_prior=extra_priors['dev']

        if not hasattr(self,'psf_flux_res'):
            self.fit_gal_psf_flux()

        print("    fitting exp")
        exp_fitter=self._fit_one_model_max('exp',pars,
                                           prior=exp_prior,ntry=ntry)
        fitting.print_pars(exp_fitter.get_result()['pars'], front='        gal_pars:')
        fitting.print_pars(exp_fitter.get_result()['pars_err'], front='        gal_perr:')
        print('        lnprob: %e' % exp_fitter.get_result()['lnprob'])
        
        print("    fitting dev")
        dev_fitter=self._fit_one_model_max('dev',pars,
                                           prior=dev_prior,ntry=ntry)
        fitting.print_pars(dev_fitter.get_result()['pars'], front='        gal_pars:')
        fitting.print_pars(dev_fitter.get_result()['pars_err'], front='        gal_perr:')
        print('        lnprob: %e' % dev_fitter.get_result()['lnprob'])           
            
        print("    fitting fracdev")
        use_grid=pars.get('use_fracdev_grid',False)
        fres=self._fit_fracdev(exp_fitter, dev_fitter, use_grid=use_grid)

        fracdev = fres['fracdev']
        fracdev_clipped = self._clip_fracdev(fracdev,pars)

        mess='        nfev: %d fracdev: %.3f +/- %.3f clipped: %.3f'
        print(mess % (fres['nfev'],fracdev,fres['fracdev_err'],fracdev_clipped))


        TdByTe = self._get_TdByTe(exp_fitter, dev_fitter)

        guesser=self._get_max_guesser(guess=guess, prior=prior)

        print("    fitting composite")
        ok=False
        for i in range(1,5):
            try:
                runner=CompositeMaxRunner(self.mb_obs_list,
                                          pars,
                                          guesser,
                                          fracdev_clipped,
                                          TdByTe,
                                          prior=prior,
                                          use_logpars=self.use_logpars)
                runner.go(ntry=ntry)
                ok=True
                break
            except GMixRangeError:
                #if i==1:
                #    print("caught GMixRange, clipping [-1.0,1.5]")
                #    fracdev_clipped = fracdev_clipped.clip(min=-1.0, max=1.5)
                #elif i==2:
                #    print("caught GMixRange, clipping [ 0.0,1.0]")
                #    fracdev_clipped = fracdev_clipped.clip(min=0.0, max=1.0)
                print("caught GMixRange, clipping [ 0.0,1.0]")
                fracdev_clipped = fracdev_clipped.clip(min=0.0, max=1.0)


        if not ok:
            raise BootGalFailure("failed to fit galaxy with maxlike: GMixRange "
                                 "indicating model problems")

        self.max_fitter=runner.fitter

        res=self.max_fitter.get_result()
        if res['flags'] != 0:
            raise BootGalFailure("failed to fit galaxy with maxlike")

        if res['flags']==0 and 'lnprob' not in res:
            print("lnprob missing:")
            pprint(res)
            raise BootGalFailure("weird error with lnprob missing")

        fitting.print_pars(res['pars'], front='        gal_pars:')
        fitting.print_pars(res['pars_err'], front='        gal_perr:')
        print('        lnprob: %e' % res['lnprob'])
        

        res['TdByTe'] = TdByTe
        res['fracdev_nfev'] = fres['nfev']
        res['fracdev'] = fracdev_clipped
        res['fracdev_noclip'] = fracdev
        res['fracdev_err'] = fres['fracdev_err']

    def _maybe_clip(self, efitter, dfitter, pars, fracdev):
        """
        allow the user to send a s/n above which the clip
        is applied.  Default is effectively no clipping,
        since fracdev_s2n_clip_min defaults to 1.0e9
        """
        eres=efitter.get_result()
        dres=dfitter.get_result()

        s2n_max=max( eres['s2n_w'], dres['s2n_w'] )

        clip_min = pars.get('fracdev_s2n_clip_min',1.e9)
        if s2n_max > clip_min:
            print("        clipping")
            frange=pars.get('fracdev_range',[-2.0, 2.0])
            fracdev_clipped = fracdev.clip(min=frange[0],max=frange[1])
        else:
            fracdev_clipped = 0.0 + fracdev

        return fracdev_clipped

    def _clip_fracdev(self, fracdev, pars):
        """
        clip according to parameters
        """
        frange=pars.get('fracdev_range',[-2.0, 2.0])
        fracdev_clipped = fracdev.clip(min=frange[0],max=frange[1])
        return fracdev_clipped


    def _get_gmix_round(self, fitter, pars, band):
        res=fitter.get_result()
        band_pars = fitter.get_band_pars(pars,band)
        gm_round = GMixCM(res['fracdev'],
                          res['TdByTe'],
                          band_pars)
        return gm_round

    def _get_gmix_round_old(self, res, pars):
        gm_round = GMixCM(res['fracdev'],
                          res['TdByTe'],
                          pars)
        return gm_round


    def _get_round_fitter_old(self, obs, res, prior=None):
        fitter=fitting.LMComposite(obs,
                                   res['fracdev'],
                                   res['TdByTe'],
                                   prior=prior,
                                   use_logpars=self.use_logpars)

        return fitter

    def _fit_sim_round(self, fitter, pars_round, mb_obs_list,
                       ntry, max_pars, round_prior=None):

        res=fitter.get_result()
        runner=CompositeMaxRunnerRound(pars_round,
                                       mb_obs_list,
                                       max_pars,
                                       res['fracdev'],
                                       res['TdByTe'],
                                       prior=round_prior,
                                       use_logpars=self.use_logpars)

        runner.go(ntry=ntry)

        return runner.fitter
 

    def isample(self, ipars, prior=None):
        super(CompositeBootstrapper,self).isample(ipars,prior=prior)
        maxres=self.max_fitter.get_result()
        ires=self.isampler.get_result()

        ires['TdByTe']=maxres['TdByTe']
        ires['fracdev']=maxres['fracdev']
        ires['fracdev_noclip']=maxres['fracdev_noclip']
        ires['fracdev_err']=maxres['fracdev_err']

    def _fit_fracdev(self, exp_fitter, dev_fitter, use_grid=False):
        from .fitting import FracdevFitter, FracdevFitterMax

        eres=exp_fitter.get_result()
        dres=dev_fitter.get_result()
        epars=eres['pars']
        dpars=dres['pars']

        fprior=self.fracdev_prior
        if fprior is None:
            ffitter = FracdevFitter(self.mb_obs_list, epars, dpars,
                                    use_logpars=self.use_logpars)
            res=ffitter.get_result()
        else:

            ffitter = FracdevFitterMax(self.mb_obs_list, epars, dpars,
                                       use_logpars=self.use_logpars,
                                       prior=fprior)
            if use_grid:
                res=self._fit_fracdev_grid(ffitter)
            else:

                guess=self._get_fracdev_guess(ffitter)

                print("        fracdev guess:",guess)
                if guess is None:
                    raise BootGalFailure("failed to fit fracdev")

                ffitter.go(guess)

                res=ffitter.get_result()

        if res['flags'] != 0:
            raise BootGalFailure("failed to fit fracdev")


        self.fracdev_fitter=ffitter
        return res

    def _fit_fracdev_grid(self, ffitter):
        """
        just use the grid
        """

        fracdev=self._get_fracdev_guess(ffitter)

        if fracdev is None:
            raise BootGalFailure("failed to fit fracdev")

        res={'flags':0,
             'fracdev':fracdev,
             'fracdev_err':1.0,
             'nfev':self.fracdev_tests.size}

        return res



    def _get_fracdev_guess(self, fitter):
        tests=self.fracdev_tests
        lnps=zeros(tests.size)
        for i in xrange(tests.size):
            lnps[i] = fitter.calc_lnprob(tests[i:i+1])

        w,=where(isfinite(lnps))
        if w.size == 0:
            return None

        if False:
            from biggles import plot
            plot(tests[w], lnps[w])
            key=raw_input('hit a key: ')

        ibest=lnps[w].argmax()
        guess=tests[w[ibest]]
        return guess

    def _get_TdByTe(self, exp_fitter, dev_fitter):
        epars=exp_fitter.get_result()['pars']
        dpars=dev_fitter.get_result()['pars']

        if self.use_logpars:
            Te = exp(epars[4])
            Td = exp(dpars[4])
        else:
            Te = epars[4]
            Td = dpars[4]
        TdByTe = Td/Te

        print('        Td/Te: %.3f' % TdByTe)
        return TdByTe


class BestBootstrapper(Bootstrapper):
    def __init__(self, obs,
                 use_logpars=False,
                 fracdev_prior=None):
        super(BestBootstrapper,self).__init__(obs,
                                              use_logpars=use_logpars)

    def fit_max(self, exp_prior, dev_prior, exp_rate, pars, ntry=1):
        """
        fit the galaxy.  You must run fit_psf() successfully first
        """

        if not hasattr(self,'psf_flux_res'):
            self.fit_gal_psf_flux()

        print("    fitting exp")
        exp_fitter=self._fit_one_model_max('exp',pars,prior=exp_prior,ntry=ntry)
        print("    fitting dev")
        dev_fitter=self._fit_one_model_max('dev',pars,prior=dev_prior,ntry=ntry)

        exp_res=exp_fitter.get_result()
        dev_res=dev_fitter.get_result()

        log_exp_rate = log(exp_rate)
        log_dev_rate = log(1.0-exp_rate)
        exp_lnprob = exp_res['lnprob'] + log_exp_rate
        dev_lnprob = dev_res['lnprob'] + log_dev_rate

        if exp_lnprob > dev_lnprob:
            self.max_fitter = exp_fitter
            self.prior=exp_prior
            res=exp_res
        else:
            self.max_fitter = dev_fitter
            self.prior=dev_prior
            res=dev_res

        mess="    exp_lnp: %.6g dev_lnp: %.6g best: %s"
        #mess=mess % (exp(exp_lnprob),exp(dev_lnprob),res['model'])
        mess=mess % (exp_lnprob,dev_lnprob,res['model'])
        print(mess)


    def isample(self, ipars):
        super(BestBootstrapper,self).isample(ipars,prior=self.prior)


class PSFRunner(object):
    """
    wrapper to generate guesses and run the psf fitter a few times
    """
    def __init__(self, obs, model, Tguess, lm_pars):
        self.obs=obs

        mess="psf model should be turb or gauss,got '%s'" % model
        assert model in ['turb','gauss'],mess

        self.model=model
        self.lm_pars=lm_pars
        self.set_guess0(Tguess)

    def go(self, ntry=1):
        from .fitting import LMSimple

        for i in xrange(ntry):
            guess=self.get_guess()
            fitter=LMSimple(self.obs,self.model,lm_pars=self.lm_pars)
            fitter.go(guess)

            res=fitter.get_result()
            if res['flags']==0:
                break

        self.fitter=fitter

    def get_guess(self):
        guess=self.guess0.copy()

        guess[0:0+2] + 0.01*srandu(2)
        guess[2:2+2] + 0.1*srandu(2)
        guess[4] = guess[4]*(1.0 + 0.1*srandu())
        guess[5] = guess[5]*(1.0 + 0.1*srandu())

        return guess

    def set_guess0(self, Tguess):
        Fguess = self.obs.image.sum()
        Fguess *= self.obs.jacobian.get_scale()**2
        self.guess0=array( [0.0, 0.0, 0.0, 0.0, Tguess, Fguess] )

class EMRunner(object):
    """
    wrapper to generate guesses and run the psf fitter a few times
    """
    def __init__(self, obs, Tguess, ngauss, em_pars):

        self.ngauss = ngauss
        self.Tguess = Tguess
        self.sigma_guess = sqrt(Tguess/2)
        self.set_obs(obs)

        self.em_pars=em_pars

    def set_obs(self, obsin):
        """
        set a new observation with sky
        """
        im_with_sky, sky = prep_image(obsin.image)

        self.obs   = Observation(im_with_sky, jacobian=obsin.jacobian)
        self.sky   = sky

    def get_fitter(self):
        return self.fitter

    def go(self, ntry=1):

        fitter=GMixEM(self.obs)
        for i in xrange(ntry):
            guess=self.get_guess()

            fitter.go(guess, self.sky, **self.em_pars)

            res=fitter.get_result()
            if res['flags']==0:
                break

        res['ntry'] = i+1
        self.fitter=fitter

    def get_guess(self):
        """
        Guess for the EM algorithm
        """

        if self.ngauss==1:
            return self._get_em_guess_1gauss()
        elif self.ngauss==2:
            return self._get_em_guess_2gauss()
        elif self.ngauss==3:
            return self._get_em_guess_3gauss()
        else:
            raise ValueError("bad ngauss: %d" % self.ngauss)

    def _get_em_guess_1gauss(self):

        sigma2 = self.sigma_guess**2
        pars=array( [1.0 + 0.1*srandu(),
                     0.1*srandu(),
                     0.1*srandu(), 
                     sigma2*(1.0 + 0.1*srandu()),
                     0.2*sigma2*srandu(),
                     sigma2*(1.0 + 0.1*srandu())] )

        return GMix(pars=pars)

    def _get_em_guess_2gauss(self):

        sigma2 = self.sigma_guess**2

        pars=array( [_em2_pguess[0],
                     0.1*srandu(),
                     0.1*srandu(),
                     _em2_fguess[0]*sigma2*(1.0 + 0.1*srandu()),
                     0.0,
                     _em2_fguess[0]*sigma2*(1.0 + 0.1*srandu()),

                     _em2_pguess[1],
                     0.1*srandu(),
                     0.1*srandu(),
                     _em2_fguess[1]*sigma2*(1.0 + 0.1*srandu()),
                     0.0,
                     _em2_fguess[1]*sigma2*(1.0 + 0.1*srandu())] )


        return GMix(pars=pars)

    def _get_em_guess_3gauss(self):

        sigma2 = self.sigma_guess**2

        pars=array( [_em3_pguess[0]*(1.0+0.1*srandu()),
                     0.1*srandu(),
                     0.1*srandu(),
                     _em3_fguess[0]*sigma2*(1.0 + 0.1*srandu()),
                     0.01*srandu(),
                     _em3_fguess[0]*sigma2*(1.0 + 0.1*srandu()),

                     _em3_pguess[1]*(1.0+0.1*srandu()),
                     0.1*srandu(),
                     0.1*srandu(),
                     _em3_fguess[1]*sigma2*(1.0 + 0.1*srandu()),
                     0.01*srandu(),
                     _em3_fguess[1]*sigma2*(1.0 + 0.1*srandu()),

                     _em3_pguess[2]*(1.0+0.1*srandu()),
                     0.1*srandu(),
                     0.1*srandu(),
                     _em3_fguess[2]*sigma2*(1.0 + 0.1*srandu()),
                     0.01*srandu(),
                     _em3_fguess[2]*sigma2*(1.0 + 0.1*srandu())]

                  )


        return GMix(pars=pars)



class MaxRunner(object):
    """
    wrapper to generate guesses and run the fitter a few times
    """
    def __init__(self, obs, model, max_pars, guesser, prior=None, use_logpars=False):
        self.obs=obs

        self.max_pars=max_pars
        self.method=max_pars['method']
        if self.method == 'lm':
            self.send_pars=max_pars['lm_pars']
        else:
            self.send_pars=max_pars

        mess="model should be exp,dev,gauss, got '%s'" % model
        assert model in ['exp','dev','gauss'],mess

        self.model=model
        self.prior=prior
        self.use_logpars=use_logpars

        self.guesser=guesser

    def go(self, ntry=1):
        if self.method=='lm':
            method=self._go_lm
        else:
            raise ValueError("bad method '%s'" % self.method)

        lnprob_max=-numpy.inf
        method(ntry=ntry)

    def _go_lm(self, ntry=1):
        
        fitclass=self._get_lm_fitter_class()

        for i in xrange(ntry):
            guess=self.guesser()
            fitter=fitclass(self.obs,
                            self.model,
                            lm_pars=self.send_pars,
                            use_logpars=self.use_logpars,
                            prior=self.prior)

            fitter.go(guess)

            res=fitter.get_result()
            if res['flags']==0:
                break

        res['ntry'] = i+1
        self.fitter=fitter

    def _get_lm_fitter_class(self):
        from .fitting import LMSimple
        return LMSimple


class MaxRunnerGaussMom(object):
    """
    wrapper to generate guesses and run the fitter a few times
    """
    def __init__(self, obs, max_pars, guesser, prior=None, use_logpars=False):
        self.obs=obs

        self.max_pars=max_pars
        self.method=max_pars['method']
        if self.method == 'lm':
            self.send_pars=max_pars['lm_pars']
        else:
            self.send_pars=max_pars

        self.prior=prior
        self.use_logpars=use_logpars

        self.guesser=guesser

    def go(self, ntry=1):
        if self.method=='lm':
            method=self._go_lm
        else:
            raise ValueError("bad method '%s'" % self.method)

        lnprob_max=-numpy.inf
        method(ntry=ntry)

    def _go_lm(self, ntry=1):
        
        fitclass=self._get_lm_fitter_class()

        for i in xrange(ntry):
            guess=self.guesser()
            fitter=fitclass(self.obs,
                            lm_pars=self.send_pars,
                            use_logpars=self.use_logpars,
                            prior=self.prior)

            fitter.go(guess)

            res=fitter.get_result()
            if res['flags']==0:
                break

        res['ntry'] = i+1
        self.fitter=fitter

    def _get_lm_fitter_class(self):
        from .fitting import LMGaussMom
        return LMGaussMom


class MaxRunnerFixT(MaxRunner):
    def __init__(self, obs, model, max_pars, guesser, T, prior=None, use_logpars=False):
        self.obs=obs

        self.max_pars=max_pars
        self.method=max_pars['method']
        assert self.method=='lm-fixT',"method must be lm-fixT"
        self.send_pars=max_pars['lm_pars']

        mess="model should be exp/dev/gauss, got '%s'" % model
        assert model in ['exp','dev','gauss'],mess

        self.model=model
        self.prior=prior
        self.use_logpars=use_logpars
        self.T=T

        self.guesser=guesser

    def go(self, ntry=1):
        from .fitting import LMSimpleFixT
        

        for i in xrange(ntry):
            fitter=LMSimpleFixT(self.obs,
                                self.model,
                                T=self.T,
                                lm_pars=self.send_pars,
                                use_logpars=self.use_logpars,
                                prior=self.prior)

            guess0=self.guesser()
            guess=zeros(guess0.size-1)
            guess[0:4]=guess0[0:4]
            guess[4:]=guess0[5:]
            fitter.go(guess)

            res=fitter.get_result()
            if res['flags']==0:
                break

        res['ntry'] = i+1
        self.fitter=fitter

class MaxRunnerGOnly(MaxRunner):
    def __init__(self, obs, model, max_pars, guesser, pars_in, prior=None, use_logpars=False):
        self.obs=obs

        self.max_pars=max_pars
        self.method=max_pars['method']
        assert self.method=='lm-gonly',"method must be lm-gonly"
        self.send_pars=max_pars['lm_pars']

        mess="model should be exp/dev/gauss, got '%s'" % model
        assert model in ['exp','dev','gauss'],mess

        self.model=model
        self.prior=prior
        self.use_logpars=use_logpars
        self.pars_in=pars_in

        self.guesser=guesser

    def go(self, ntry=1):
        from .fitting import LMSimpleGOnly
        

        for i in xrange(ntry):
            fitter=LMSimpleGOnly(self.obs,
                                 self.model,
                                 pars=self.pars_in,
                                 lm_pars=self.send_pars,
                                 use_logpars=self.use_logpars,
                                 prior=self.prior)

            guess0=self.guesser()
            if guess0.size==2:
                guess=guess0
            else:
                guess=zeros(2)
                guess[0]=guess0[2]
                guess[1]=guess0[3]
            fitter.go(guess)

            res=fitter.get_result()
            if res['flags']==0:
                break

        res['ntry'] = i+1
        self.fitter=fitter


class RoundRunnerBase(object):
    def _get_round_guesser(self, **kw):

        use_logpars=kw.get('use_logpars',False)
        if use_logpars:
            scaling='log'
        else:
            scaling='linear'

        prior=kw.get('prior',None)
        guesser=RoundParsGuesser(self.pars_sub,
                                 prior=prior,
                                 scaling=scaling)
        return guesser

class MaxRunnerRound(MaxRunner,RoundRunnerBase):
    """
    pars_full should be roundified

    make sure prior (and guesser) are also for round
    """
    def __init__(self, pars_full, obs, model, max_pars, *args, **kw):

        self.pars_full=pars_full
        self.pars_sub=roundify.get_simple_sub_pars(pars_full)
        guesser=self._get_round_guesser(**kw)

        super(MaxRunnerRound,self).__init__(obs, model, max_pars, guesser,
                                            **kw)

    def _get_lm_fitter_class(self):
        from .fitting import LMSimpleRound
        return LMSimpleRound

class CompositeMaxRunner(MaxRunner):
    """
    wrapper to generate guesses and run the psf fitter a few times
    """
    def __init__(self, obs, max_pars, guesser, fracdev, TdByTe,
                 prior=None, use_logpars=False):
        self.obs=obs

        self.max_pars=max_pars
        self.fracdev=fracdev
        self.TdByTe=TdByTe

        self.method=max_pars['method']
        if self.method == 'lm':
            self.send_pars=max_pars['lm_pars']
        else:
            self.send_pars=max_pars

        self.prior=prior
        self.use_logpars=use_logpars

        self.guesser=guesser

    def _go_lm(self, ntry=1):
        from .fitting import LMComposite

        fitclass=self._get_lm_fitter_class()

        for i in xrange(ntry):
            guess=self.guesser()
            fitter=fitclass(self.obs,
                            self.fracdev,
                            self.TdByTe,
                            lm_pars=self.send_pars,
                            use_logpars=self.use_logpars,
                            prior=self.prior)

            fitter.go(guess)

            res=fitter.get_result()
            if res['flags']==0:
                break

        self.fitter=fitter

    def _get_lm_fitter_class(self):
        from .fitting import LMComposite
        return LMComposite

class CompositeMaxRunnerRound(CompositeMaxRunner,RoundRunnerBase):
    """
    make sure prior (and guesser) are also for round
    """
    def __init__(self, pars_full, obs, max_pars, fracdev, TdByTe, **kw):

        self.pars_full=pars_full
        self.pars_sub=roundify.get_simple_sub_pars(pars_full)
        guesser=self._get_round_guesser(**kw)

        super(CompositeMaxRunnerRound,self).__init__(obs,
                                                     max_pars,
                                                     guesser,
                                                     fracdev,
                                                     TdByTe,
                                                     **kw)


    def _get_lm_fitter_class(self):
        from .fitting import LMCompositeRound
        return LMCompositeRound

def get_em_ngauss(name):
    ngauss=int( name[2:] )
    return ngauss

_em2_fguess =array([0.5793612389470884,1.621860687127999])
_em2_pguess =array([0.596510042804182,0.4034898268889178])

_em3_pguess = array([0.596510042804182,0.4034898268889178,1.303069003078001e-07])
_em3_fguess = array([0.5793612389470884,1.621860687127999,7.019347162356363],dtype='f8')


def test_boot(model,**keys):
    from .test import make_test_observations

    psf_obs, obs = make_test_observations(model, **keys)

    obs.set_psf(psf_obs)

    boot=Bootstrapper(obs)

    psf_model=keys.get('psf_model','gauss')
    Tguess=4.0
    boot.fit_psfs(psf_model, Tguess)

    pars={'method':'lm',
          'lm_pars':{'maxfev':4000}}
    boot.fit_max(model, pars)
