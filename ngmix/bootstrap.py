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
from numpy import isfinite, median, diag
from numpy.linalg import LinAlgError

from . import fitting
from .fitting import print_pars
from .gmix import GMix, GMixModel, GMixCM, get_coellip_npars
from .em import GMixEM, prep_image
from .observation import Observation, ObsList, MultiBandObsList, get_mb_obs
from .priors import srandu
from .shape import get_round_factor
from .guessers import TFluxGuesser, TFluxAndPriorGuesser, ParsGuesser, RoundParsGuesser
from .gexceptions import GMixRangeError, BootPSFFailure, BootGalFailure

from . import roundify
from . import metacal

from copy import deepcopy

BOOT_S2N_LOW = 2**0
BOOT_R2_LOW = 2**1
BOOT_R4_LOW = 2**2
BOOT_TS2N_ROUND_FAIL = 2**3
BOOT_ROUND_CONVOLVE_FAIL = 2**4
BOOT_WEIGHTS_LOW= 2**5



class Bootstrapper(object):
    def __init__(self, obs,
                 use_logpars=False, intpars=None, find_cen=False,
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
        self.intpars=intpars
        self.find_cen=find_cen
        self.verbose=verbose

        self.use_round_T = kw.get('use_round_T',False)

        # this never gets modified in any way
        self.mb_obs_list_orig = get_mb_obs(obs)

        # this will get replaced if fit_psfs is run
        self.mb_obs_list=self.mb_obs_list_orig

        if self.find_cen:
            self._find_cen()

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

    def get_metacal_max_result(self):
        """
        get result of metacal with a max likelihood fitter
        """
        if not hasattr(self, 'metacal_max_res'):
            raise RuntimeError("you need to run fit_metacal_max first")
        return self.metacal_max_res

    def get_metacal_regauss_result(self):
        """
        get result of metacal with a max likelihood fitter
        """
        if not hasattr(self, 'metacal_regauss_res'):
            raise RuntimeError("you need to run fit_metacal_regauss first")
        return self.metacal_regauss_res



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
                gm_round_band0 = self._get_gmix_round(max_fitter,
                                                      pars_round,
                                                      band)

                for obs in obslist:
                    # note psf does not use round T
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


    def _get_gmix_round(self, fitter, pars, band):
        """
        input pars are round
        """
        res=fitter.get_result()
        band_pars = fitter.get_band_pars(pars,band)
        gm_round = GMixModel(band_pars, res['model'])
        return gm_round

    def _get_gmix_round_old(self, res, pars):
        gm_round = GMixModel(pars, res['model'])
        return gm_round

    def _get_round_pars(self, pars_in):

        pars=pars_in.copy()
        pars_lin=pars.copy()

        if self.use_logpars:
            pars_lin[4:4+2] = exp(pars[4:4+2])

        g1,g2,T = pars_lin[2],pars_lin[3],pars_lin[4]

        if self.use_round_T:
            Tround=T
        else:
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
            jacob.set_cen(row,col)
        else:
            print("        failed to find cen")

    def fit_psfs(self, psf_model, Tguess,
                 Tguess_key=None,
                 skip_failed=True,
                 ntry=4,
                 fit_pars=None,
                 skip_already_done=True,
                 norm_key=None):
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
        skip_already_done: bool
            Skip psfs with a gmix already set
        norm_key: will use this key in the PSF meta data to fudge the normalization 
            of the PSF model via amplitude -> amplitude*norm where amplitude is the PSF normalization
            (usually 1)
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
                if skip_already_done:
                    # if have a gmix, skip it
                    if psf_obs.has_gmix():
                        new_obslist.append(obs)
                        ntot += 1
                        continue

                    # if have a fitter and flags != 0, skip it
                    if 'fitter' in psf_obs.meta:
                        tres=psf_obs.meta['fitter'].get_result()
                        if tres['flags'] != 0:
                            mess=("    failed psf fit band %d obs %d, "
                                  "skipping observation" % (band,i))
                            print(mess)
                            continue

                try:
                    if Tguess_key is not None:
                        Tguess_i = psf_obs.meta[Tguess_key]
                    else:
                        Tguess_i = Tguess
                    self._fit_one_psf(psf_obs, psf_model, Tguess_i,
                                      ntry, fit_pars, norm_key=norm_key)
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

    def _fit_one_psf(self, psf_obs, psf_model, Tguess, ntry,
                     fit_pars, norm_key=None):
        """
        fit the psf using a PSFRunner or EMRunner

        TODO: add bootstrapping T guess as well, from unweighted moments
        """

        if 'em' in psf_model:
            assert self.intpars is None,"pixel integration only for max like fitting"
            assert self.use_round_T==False,"round_T only in simple fitters for now"

            runner=self._fit_one_psf_em(psf_obs, psf_model,
                                        Tguess, ntry, fit_pars)
        elif 'coellip' in psf_model:
            assert self.use_round_T==False,"round_T only in simple fitters for now"
            runner=self._fit_one_psf_coellip(psf_obs, psf_model,
                                             Tguess, ntry, fit_pars)
        else:
            runner=self._fit_one_psf_max(psf_obs, psf_model,
                                         Tguess, ntry, fit_pars)

        psf_fitter = runner.fitter
        res=psf_fitter.get_result()
        psf_obs.update_meta_data({'fitter':psf_fitter})

        if res['flags']==0:
            self.psf_fitter=psf_fitter
            gmix=self.psf_fitter.get_gmix()

            if norm_key is not None:
                gmix.set_psum(psf_obs.meta[norm_key])

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

        runner=PSFRunnerCoellip(psf_obs, Tguess, ngauss, lm_pars, intpars=self.intpars)
        runner.go(ntry=ntry)

        return runner


    def _fit_one_psf_max(self, psf_obs, psf_model, Tguess, ntry, fit_pars):
        lm_pars={'maxfev': 4000}

        if fit_pars is not None:
            lm_pars.update(fit_pars)

        runner=PSFRunner(psf_obs, psf_model, Tguess, lm_pars,
                         intpars=self.intpars)
        runner.go(ntry=ntry)

        return runner

    def replace_masked_pixels(self,
                              inplace=False,
                              method='best-fit',
                              fitter=None):
        """
        replaced masked pixels

        If a modification is made, the original image is stored for each
        Observation as .image_orig

        The original mb_obs_list is always in self.mb_obs_list_old,
        which is just a ref if inplace=True

        parameters
        ----------
        inplace: bool
            If True, modify the data in place.  Default False; a full
            copy is made.
        method: string, optional
            Method for replacement.  Supported methods are 'best-fit'.
            Default is 'best-fit'
        fitter: a fitter from fitters.py
            If not sent, the max fitter from self is used.
        """

        self.mb_obs_list_old = self.mb_obs_list

        if fitter is None:
            fitter=self.get_max_fitter()

        self.mb_obs_list=replace_masked_pixels(self.mb_obs_list,
                                               inplace=inplace,
                                               method=method,
                                               fitter=fitter)
        '''
        assert method=='best-fit',"only best-fit replacement is supported"

        mbo = self.mb_obs_list
        nband = len(mbo)

        for band in xrange(nband):
            olist = mbo[band]
            nobs = len(olist)
            for iobs,obs in enumerate(olist):

                bmask = obs.bmask
                if bmask is not None:
                    w=where(bmask != 0)

                    if w[0].size > 0:
                        print("    replacing %d/%d masked pixels" % (w[0].size,bmask.size))
                        obs.image_orig = obs.image.copy()
                        gm = fitter.get_convolved_gmix(band=band, obsnum=iobs)

                        im = obs.image
                        model_image = gm.make_image(im.shape, jacobian=obs.jacobian)

                        im[w] = model_image[w]

                        if False:
                            import images
                            imdiff=im-obs.image_orig
                            images.view_mosaic([bmask,obs.image_orig,im,imdiff],
                                               titles=['mask','orig','mod image','mod-orig'])
                            maxdiff=numpy.abs(imdiff).max()
                            print("    Max abs diff:",maxdiff)
                            images.multiview(imdiff,title='mod-orig max diff %g' % maxdiff)
                            if raw_input('hit a key: ') == 'q':
                                stop
        '''
    def fit_max(self,
                gal_model,
                pars,
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
                                                  guess=guess,
                                                  prior=prior,
                                                  ntry=ntry,
                                                  guess_widths=guess_widths)
    def fit_metacal_max(self,
                        psf_model,
                        gal_model,
                        pars,
                        psf_Tguess,
                        psf_fit_pars=None,
                        target_noise=None,
                        extra_noise=None,
                        metacal_obs=None,
                        nrand=1,
                        metacal_pars=None,
                        prior=None,
                        psf_ntry=10,
                        ntry=1,
                        guess_from_max=False,
                        **kw):
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

        if extra_noise is not None or target_noise is not None:
            raise NotImplementedError("fix noise adding")

        if guess_from_max:
            guess = self.get_max_fitter().get_result()['pars']
        else:
            guess=None

        mpars={'step':0.01}
        if metacal_pars is not None:
            mpars.update(metacal_pars)

        metacal_pars=mpars

        #oobs = self.mb_obs_list[0][0]
        #extra_noise = self._get_extra_noise(oobs, target_noise, extra_noise)

        if extra_noise is None:
            nrand=1

        if metacal_obs is not None:
            if self.verbose:
                print("        using input metacal obs dict")
            obs_dict_orig=metacal_obs
        else:
            # metacal pars can contain extra keywords such as
            # use_psf_model and psf_shape
            # this can contain things like type= for metacal types
            metacal_pars.update(kw)
            obs_dict_orig = metacal.get_all_metacal(self.mb_obs_list,
                                                    **metacal_pars)

        if False:
            import images
            images.multiview(obs_dict_orig['1p'][0][0].image,
                            width=1000,height=1000)
            images.compare_images(self.mb_obs_list[0][0].image,
                                  obs_dict_orig['1p'][0][0].image,
                                  width=1000,height=1000)
            if raw_input('hit a key: ')=='q':
                stop

        reslist=[]
        for i in xrange(nrand):

            #if extra_noise is not None:
            #    obs_dict = self._add_noise_to_metacal_obsdict(obs_dict_orig,
            #                                                  extra_noise)
            #else:
            #    obs_dict=obs_dict_orig
            obs_dict=obs_dict_orig

            if nrand > 1 and self.verbose:
                print("    irand: %d/%d" % (i+1,nrand))

            fits = self._do_metacal_max_fits(obs_dict,
                                             psf_model, gal_model,
                                             pars, psf_Tguess,
                                             prior, psf_ntry, ntry,
                                             psf_fit_pars,
                                             extra_noise,
                                             guess=guess)
            tres=self._extract_metacal_responses(fits, metacal_pars)


            reslist.append(tres)

        if nrand==1:
            res=reslist[0]
        else:
            res=self._do_mean_dictlist(reslist)
        res['obs_dict'] = obs_dict_orig
        self.metacal_max_res = res

        return obs_dict_orig

    def _do_mean_dictlist(self, reslist):
        n=len(reslist)

        res={}

        keys=list(reslist[0].keys())

        for i,tres in enumerate(reslist):
            for key in keys:
                if i==0:
                    res[key] = tres[key]
                else:
                    res[key] = res[key] + tres[key]

        for key in keys:
            res[key] = res[key]/n

        for i,tres in enumerate(reslist):
            for key in keys:
                stdkey = '%s_std' % key

                diff = tres[key] - res[key]
                diff2 = diff**2

                if i==0:
                    res[stdkey] = diff2
                else:
                    res[stdkey] = res[stdkey] + diff2

        if n > 1:
            for key in keys:
                stdkey = '%s_std' % key
                errkey = '%s_err' % key
                res[stdkey] = sqrt(res[stdkey]/(n-1))
                res[errkey] = res[stdkey]/sqrt(n)

        return res

    def _get_extra_noise(self, obs, target_noise, extra_noise):
        if target_noise is not None:
            return self._get_extra_noise_from_target(obs, target_noise)
        else:
            if extra_noise is not None:
                print("    extra_noise: %g" % extra_noise)
            return extra_noise

    def _get_extra_noise_from_target(self, obs, target_noise):
        weight = obs.weight
        w=where(weight > 0.0)
        if w[0].size == 0:
            print("    no weight > 0")
            extra_noise = target_noise
        else:
            noise_mean = sqrt( (1.0/weight[w]).mean() )

            if target_noise < noise_mean:
                tup = (target_noise, noise_mean)
                raise ValueError("target noise %g is "
                                 "less than mean noise "
                                 "in weight map: %g" % tup)

            extra_noise = sqrt(target_noise**2 - noise_mean**2)

        print("    target_noise: %g extra_noise: %g" % (target_noise, extra_noise))
        return extra_noise

    def _add_noise_to_metacal_obsdict(self, obs_dict, extra_noise):
        """
        add noise to all images, adjusting weight maps accordingly
        """
        noise_image = self._get_noise_image(obs_dict['1p'].image.shape,
                                            extra_noise)

        nobs_dict={}
        for key in obs_dict:

            obs=obs_dict[key]

            new_weight = self._get_degraded_weight_image(obs, extra_noise)
            new_obs = self._get_degraded_obs(obs, noise_image, new_weight)

            nobs_dict[key] = new_obs

        return nobs_dict

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

    def _extract_metacal_responses(self, fits,metacal_pars, shape_type='g'):
        """
        pars pars_cov gpsf, s2n_r, s2n_simple, T_r, psf_T_r required

        expect the shape to be in pars[2] and pars[3]
        """
        step = metacal_pars['step']

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

        pars_mean[2] = 0.5*(pars['1p'][2] + pars['1m'][2])
        pars_mean[3] = 0.5*(pars['2p'][3] + pars['2m'][3])

        if self.verbose:
            print_pars(pars_mean, front='    mcmean:   ')

        R=zeros( (2,2) ) 
        Rpsf=zeros(2)

        fac = 1.0/(2.0*step)

        R[0,0] = (pars['1p'][2]-pars['1m'][2])*fac
        R[0,1] = (pars['1p'][3]-pars['1m'][3])*fac
        R[1,0] = (pars['2p'][2]-pars['2m'][2])*fac
        R[1,1] = (pars['2p'][3]-pars['2m'][3])*fac

        if '1p_psf' in pars:
            Rpsf[0] = (pars['1p_psf'][2]-pars['1m_psf'][2])*fac
            Rpsf[1] = (pars['2p_psf'][3]-pars['2m_psf'][3])*fac

        gname = 'mcal_%s' % shape_type
        gcovname = 'mcal_%s_cov' % shape_type
        gpsf_name = 'mcal_%spsf' % shape_type
        raw_gpsf_name = '%spsf' % shape_type
        res = {
            'mcal_pars':pars_mean,
            'mcal_pars_cov':pars_cov_mean,
            gname:pars_mean[2:2+2],
            gcovname:pars_cov_mean[2:2+2, 2:2+2],
            'mcal_R':R,
            'mcal_Rpsf':Rpsf,
            gpsf_name:fits[raw_gpsf_name],
            'mcal_s2n_r':fits['s2n_r'],
            'mcal_T_r':fits['T_r'],
            'mcal_psf_T':fits['psf_T'],
            'mcal_psf_T_r':fits['psf_T_r'],
        }

        if 'noshear' in pars:
            res['mcal_pars_noshear'] = pars['noshear']
            res['c'] = pars_mean[2:2+2] - pars['noshear'][2:2+2]

        if 's2n_simple' in fits:
            res['s2n_simple'] = fits['s2n_simple']
        return res


    def _do_metacal_max_fits(self, obs_dict, psf_model, gal_model, pars, 
                             psf_Tguess, prior, psf_ntry, ntry, 
                             psf_fit_pars,
                             extra_noise,
                             guess=None):

        if guess is not None:
            guess_widths = guess*0.0 + 1.0e-6
            #print("    using guess: ",guess)
            #print("    using widths:",guess_widths)
        else:
            guess_widths=None

        bdict={}
        for key in sorted(obs_dict):
            boot = Bootstrapper(obs_dict[key],
                                use_logpars=self.use_logpars,
                                intpars=self.intpars,
                                use_round_T=self.use_round_T,
                                find_cen=self.find_cen,
                                verbose=self.verbose)

            boot.fit_psfs(psf_model, psf_Tguess, ntry=psf_ntry, fit_pars=psf_fit_pars)
            boot.fit_max(gal_model, pars, prior=prior, ntry=ntry,
                         guess=guess,
                         guess_widths=guess_widths)
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
        psf_T_mean = 0.0
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
                    psf_T_mean += T
                    npsf+=1

            rres=boot.get_round_result()

            s2n_r_mean   += rres['s2n_r']
            T_r_mean     += rres['T_r']
            psf_T_r_mean += rres['psf_T_r']
            navg += 1

        assert navg==4,"expected 4 to average"

        if 'noshear' in bdict:
            bnoshear=bdict['noshear']
            res['s2n_simple'] = bnoshear.mb_obs_list.get_s2n()

        res['s2n_r']   = s2n_r_mean/navg
        res['T_r']     = T_r_mean/navg
        res['psf_T_r'] = psf_T_r_mean/navg
        res['psf_T'] = psf_T_mean/navg
        res['gpsf'] = gpsf_mean/npsf

        return res

    def fit_metacal_regauss(self,
                            psf_Tguess,
                            Tguess,
                            psf_ntry=1,
                            ntry=1,
                            metacal_pars=None,
                            target_noise=None,
                            extra_noise=None):

        """
        run metacalibration using regauss from the galsim
        HSM module

        parameters
        ----------
        psf_Tguess: float
            guess for T
        metacal_pars: dict, optional
            Parameters for metacal, default {'step':0.01}
        extra_noise: float
            extra noise to add, default None
        target_noise: float
            Add extra noise to reach target, over-rides extra_noise
        psf_ntry: int
            number of tries at psf fitting with different size guess
        ntry: int
            number of tries at shear fitting with different size guess
        """

        if len(self.mb_obs_list) > 1 or len(self.mb_obs_list[0]) > 1:
            raise NotImplementedError("only a single obs for now")

        mpars={'step':0.01}
        if metacal_pars is not None:
            mpars.update(metacal_pars)

        metacal_pars=mpars

        oobs = self.mb_obs_list[0][0]

        if target_noise is not None:
            extra_noise = self._get_extra_noise_from_target(oobs,
                                                            target_noise)

        obs_dict_orig = ngmix.metacal.get_all_metacal(oobs,
                                                      **metacal_pars)



        if extra_noise is not None:
            obs_dict = self._add_noise_to_metacal_obsdict(obs_dict_orig,
                                                          extra_noise)
        else:
            obs_dict=obs_dict_orig

        fits = self._do_metacal_regauss(obs_dict,
                                        psf_Tguess,
                                        Tguess,
                                        psf_ntry,
                                        ntry)

        res=self._extract_metacal_responses(fits, metacal_pars, shape_type='e')

        res['flags']=0

        self.metacal_regauss_res = res

    def _do_metacal_regauss(self,
                            obs_dict,
                            psf_Tguess,
                            Tguess,
                            psf_ntry,
                            ntry):

        import galsim

        res_dict={}
        for key in obs_dict:

            obs = obs_dict[key]

            im = galsim.Image( obs.image )
            psf_im = galsim.Image( obs.psf.image )
            wt = galsim.Image( obs.weight )

            res = self._do_one_regauss(
                im,
                psf_im,
                wt,
                psf_Tguess,
                Tguess,
                psf_ntry,
                ntry
            )
            res_dict[key] = res

        res={'pars':{}, 'pars_cov':{}}
        epsf_mean = zeros(2)
        npsf=0

        for i,key in enumerate(res_dict):
            tres = res_dict[key]

            res['pars'][key] = tres['pars']
            res['pars_cov'][key] = tres['pars_cov']

            #
            # averaging
            #

            if key=='noshear' or 'psf' in key:
                # don't include noshear in the averages
                # don't average over psf sheared model
                continue

            psf_pars=tres['psf_pars']

            epsf_mean[:] += psf_pars[2:2+2]
            npsf+=1

        res['epsf'] = epsf_mean/npsf

        return res

    def _get_regauss_sig_guess(self, psf_im, Tguess, ntry):
        import galsim

        # Eric recommends trying a few different psf sigma sizes
        ok=False
        for i in xrange(ntry):
            psf_guess_sig = sqrt(Tguess/2.0)*(1.0 + 0.1*srandu())
            try:
                galsim.hsm.FindAdaptiveMom(psf_im, guess_sig=psf_guess_sig)
                ok=True
                break
            except RuntimeError as err:
                print("error in FindAdaptiveMoments: '%s'" % str(err))
                pass
        
        if not ok:
            raise BootPSFFailure("failed to fit regauss psf: '%s'" % str(err))

        return psf_guess_sig

    def _do_one_regauss(self,
                        im,
                        psf_im,
                        wt,
                        psf_Tguess,
                        Tguess,
                        psf_ntry,
                        ntry):
        import galsim

        psf_guess_sig = self._get_regauss_sig_guess(psf_im,
                                                    psf_Tguess,
                                                    psf_ntry)


        # hsm doesn't use the weight map except for "non-zero"
        # still need to get the sky var

        w=where(wt.array > 0)
        if w[0].size == 0:
            raise BootGalFailure("no weight > 0")

        var = 1.0/wt.array[w]
        sky_var = median(var)

        ok=False
        for i in xrange(ntry):
            guess_sig = sqrt(Tguess/2.0)*(1.0 + 0.1*srandu())
            try:
                hres = galsim.hsm.EstimateShear(im,
                                                psf_im,
                                                sky_var=sky_var,
                                                weight=wt,
                                                guess_sig_PSF=psf_guess_sig,
                                                guess_sig_gal=guess_sig,
                                                strict=True)
                ok=True
                break
            except RuntimeError as err:
                print("        error in regauss: '%s'" % str(err))
                pass
        
        if not ok:
            raise BootGalFailure("failed to do regauss: '%s'" % str(err))

        # the centroid is not recorded!  Use 0.0
        row,col=0.0,0.0
        # pre-psf size not recorded, so use the observed
        T = 2.0*hres.moments_sigma**2
        e1,e2=hres.corrected_e1, hres.corrected_e2
        eerr = hres.corrected_shape_err
        evar = eerr**2

        pars=array([0.0, 0.0,
                    e1, e2,
                    T,
                    hres.moments_amp])
        perr = array([9999.0, 9999.0,
                      eerr,eerr,
                      9999.0,
                      9999.0])
        pvar = array([9999.0, 9999.0,
                      evar,evar,
                      9999.0,
                      9999.0])

        psf_pars = array([0.0, 0.0,
                          hres.psf_shape.e1, hres.psf_shape.e2,
                          2*hres.psf_sigma**2,
                          1.0])
        pcov = diag(pvar)

        res=dict(pars=pars,
                 perr=perr,
                 pars_cov=pcov,
                 psf_pars=psf_pars)
        return res


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



    def _fit_one_model_max(self,
                           gal_model,
                           pars,
                           guess=None,
                           guess_widths=None,
                           prior=None,
                           ntry=1,
                           obs=None):
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
                         intpars=self.intpars,
                         use_logpars=self.use_logpars)

        runner.go(ntry=ntry)

        fitter=runner.fitter

        res=fitter.get_result()

        if res['flags'] != 0:
            raise BootGalFailure("failed to fit galaxy with maxlike")

        return fitter



    def isample(self, ipars, prior=None):
        """
        bootstrap off the maxlike run
        """

        max_fitter=self.max_fitter
        use_fitter=max_fitter

        niter=len(ipars['nsample'])
        for i,nsample in enumerate(ipars['nsample']):
            sampler=self._make_isampler(use_fitter, ipars)
            if sampler is None:
                raise BootGalFailure("isampling failed")

            sampler.make_samples(nsample)

            sampler.set_iweights(max_fitter.calc_lnprob)
            sampler.calc_result()

            tres=sampler.get_result()

            if self.verbose:
                print("    eff iter %d: %.2f" % (i,tres['efficiency']))
            use_fitter = sampler

        maxres=max_fitter.get_result()
        tres['model'] = maxres['model']

        self.isampler=sampler

    def _make_isampler(self, fitter, ipars):
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
                             verbose=self.verbose)
        except LinAlgError:
            print("        bad cov")
            sampler=None

        return sampler

    def psample(self, psample_pars, samples):
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
                         verbose=self.verbose,
                         **psample_pars)

        sampler.calc_loglikes(tfitter.calc_lnprob)

        self.psampler=sampler

        res=sampler.get_result()
        if res['flags'] != 0:
            raise BootGalFailure("psampling failed")



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
        fitters = []
        results = []
        for i in xrange(nband):
            obs_list = mbo[i]

            fitter=fitting.TemplateFluxFitter(obs_list, do_psf=True, normalize_psf=normalize_psf)
            fitter.go()

            res=fitter.get_result()
            tflags = res['flags']
            flags.append( tflags )
            fitters.append(fitter)
            results.append(res)
            
            if tflags == 0:

                psf_flux[i] = res['flux']
                psf_flux_err[i] = res['flux_err']

                #print("    psf flux %d: %.3f +/- %.3f" % (i,res['flux'],res['flux_err']))
            else:
                print("failed to fit psf flux for band",i)

        self.psf_flux_res={'flags':flags,
                           'psf_flux':psf_flux,
                           'psf_flux_err':psf_flux_err,
                           'fitters':fitters,
                           'results':results}

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

class BootstrapperGaussMom(Bootstrapper):
    def __init__(self, obs, **kw):
        super(BootstrapperGaussMom,self).__init__(obs, **kw)

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

    def _make_isampler(self, fitter, ipars):
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
                                verbose=self.verbose)
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
                 fracdev_grid=None, **kw):

        super(CompositeBootstrapper,self).__init__(obs,
                                                   use_logpars=use_logpars, **kw)

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
                guess_TdbyTe=1.0,
                prior=None,
                extra_priors=None,
                guess_widths=None,
                ntry=1):
        """
        fit the galaxy.  You must run fit_psf() successfully first
        """

        def get_bic(chi2per,dof,npars):
            chi2 = chi2per*dof
            ndata = dof + npars
            lnprob = -0.5*chi2
            bic = -2.0*lnprob + npars*(numpy.log(ndata) - numpy.log(2.0*numpy.pi))
            return bic

        assert model=='cm','model must be cm'
        if extra_priors is None:
            exp_prior=prior
            dev_prior=prior
        else:
            exp_prior=extra_priors['exp']
            dev_prior=extra_priors['dev']

        # do template flux fit
        from .fitting import MultiBandTemplateFluxFitter,LMComposite
        flux_fit = MultiBandTemplateFluxFitter(self.mb_obs_list,do_psf=True,normalize_psf=True)
        try:
            flux_fit.go()
            flux_res = flux_fit.get_result()        
        except:
            raise BootGalFailure("failed to fit galaxy with PSF flux fitter")
        
        if flux_res['flags'] != 0:
            raise BootGalFailure("failed to fit galaxy with PSF flux fitter")
        
        # get PSF BIC
        psf_bic = get_bic(flux_res['chi2per'],flux_res['dof'],len(flux_res['pars']))
        
        print('        lnprob: %e' % (flux_res['lnprob']))
        print('        BIC: %e' % (psf_bic))
        
        if guess is not None and not numpy.array_equal(guess[0:5],numpy.zeros(5)):
            exp_guess = guess.copy()
            dev_guess = guess.copy()
            dev_guess[4] *= guess_TdbyTe
        else:
            exp_guess = None
            dev_guess = None
            
        print("    fitting exp")
        try:
            exp_fitter=self._fit_one_model_max('exp',pars,guess=exp_guess,
                                               prior=exp_prior,ntry=ntry,guess_widths=guess_widths)
            fitting.print_pars(exp_fitter.get_result()['pars'], front='        gal_pars:')
            fitting.print_pars(exp_fitter.get_result()['pars_err'], front='        gal_perr:')
            print('        lnprob: %e' % exp_fitter.get_result()['lnprob'])
        
            # get exp bic
            exp_bic = get_bic(exp_fitter.get_result()['chi2per'],
                              exp_fitter.get_result()['dof'],
                              len(exp_fitter.get_result()['pars']))
            print('        BIC: %e' % (exp_bic))

            psf_to_exp_odds = numpy.exp(numpy.clip(-0.5*psf_bic + 0.5*exp_bic,-300.0,300.0))
            prob_psf = psf_to_exp_odds/(1.0 + psf_to_exp_odds)
            print('    psf vs exp prob: %.6e' % prob_psf)
            do_psf = False
        except:
            if pars.get('resort_to_psf_on_exp_failure',False):
                print('    exp fit failed - resorting to PSF model')
                do_psf = True
                pass
            else:
                raise
            
        if do_psf or (pars.get('use_bic_test_for_simp_models',False) and prob_psf > pars.get('psf_prob_thresh',1.1)):
            print("    fitting psf only")
            
            fracdev = numpy.array(1.0)
            fracdev_clipped = numpy.array(1.0)
            TdByTe = numpy.array(0.0)
                            
            res = {}
            res['TdByTe'] = TdByTe
            res['fracdev_nfev'] = -9999.0
            res['fracdev'] = fracdev_clipped
            res['fracdev_noclip'] = fracdev
            res['fracdev_err'] = -9999.0
            
            npars = 5+len(flux_res['pars'])
            err = 1.0e-3
            res['pars'] = numpy.zeros(npars)
            res['pars_err'] = numpy.zeros(npars)
            res['pars_cov'] = numpy.zeros((npars,npars))                
            
            res['pars'][5:] = flux_res['pars']
            res['pars_err'][0:5] = err
            res['pars_err'][5:] = flux_res['pars_err']
            res['chi2per'] = flux_res['chi2per']
            res['dof'] = flux_res['dof']
            res['lnprob'] = flux_res['lnprob']
            res['flags'] = flux_res['flags']
            res['npix'] = flux_res['npix']
            res['model'] = 'cm'
            res['g'] = numpy.zeros(2)
            res['g_cov'] = numpy.zeros((2,2))                
            res['s2n_w'] = flux_res['s2n_w']
            
            for i in xrange(5):
                res['pars_cov'][i,i] = err*err
            for i in xrange(npars-5):
                res['pars_cov'][5+i,5+i] = flux_res['pars_err'][i]**2
                
            self.max_fitter = LMComposite(self.mb_obs_list,fracdev_clipped,TdByTe)
            self.max_fitter._setup_data(res['pars'])
            self.max_fitter._result = res
            
        elif pars.get('use_bic_test_for_simp_models',False) and prob_psf > pars.get('exp_prob_thresh',1.1):
            print('    fitting exp only')
            fracdev = numpy.array(0.0)
            fracdev_clipped = fracdev
            TdByTe = numpy.array(0.0)
            
            guesser=self._get_max_guesser(guess=guess, prior=prior, widths=guess_widths)
            
            mess='        fracpsf: %.3f clipped: %.3f'
            print(mess % (fracdev,fracdev_clipped))
            print('        Td/Te: %.3f' % (TdByTe))
            
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
            res['TdByTe'] = TdByTe
            res['fracdev_nfev'] = -9999.0
            res['fracdev'] = fracdev_clipped
            res['fracdev_noclip'] = fracdev
            res['fracdev_err'] = -9999.0
            
        else:
            print("    fitting dev")
            dev_fitter=self._fit_one_model_max('dev',pars,guess=dev_guess,
                                               prior=dev_prior,ntry=ntry,guess_widths=guess_widths)
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
            
            TdByTe_raw = self._get_TdByTe(exp_fitter, dev_fitter)
            TdByTe_range = pars.get('TdByTe_range',[-1.0e9,1.0e-9])
            TdByTe = numpy.clip(TdByTe_raw,TdByTe_range[0],TdByTe_range[1])
            print('        Td/Te: %.3f clipped: %.3f' % (TdByTe_raw,TdByTe))
            
            guesser=self._get_max_guesser(guess=guess, prior=prior, widths=guess_widths)

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
            res['TdByTe'] = TdByTe
            res['fracdev_nfev'] = fres['nfev']
            res['fracdev'] = fracdev_clipped
            res['fracdev_noclip'] = fracdev
            res['fracdev_err'] = fres['fracdev_err']
            
        if res['flags'] != 0:
            raise BootGalFailure("failed to fit galaxy with maxlike")

        if res['flags']==0 and 'lnprob' not in res:
            print("lnprob missing:")
            pprint(res)
            raise BootGalFailure("weird error with lnprob missing")

        fitting.print_pars(res['pars'], front='        gal_pars:')
        fitting.print_pars(res['pars_err'], front='        gal_perr:')
        print('        lnprob: %e' % res['lnprob'])
        



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
        """
        input pars are round
        """
        res=fitter.get_result()
        band_pars = fitter.get_band_pars(pars,band)
        gm_round = GMixCM(res['fracdev'],
                          res['TdByTe'],
                          band_pars)        
        return gm_round

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
        
        return TdByTe


class BestBootstrapper(Bootstrapper):
    def __init__(self, obs,
                 use_logpars=False,
                 fracdev_prior=None, **kw):
        super(BestBootstrapper,self).__init__(obs,
                                              use_logpars=use_logpars, **kw)

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

    I never use "round_T"
    """
    def __init__(self, obs, model, Tguess, lm_pars,
                 intpars=None, use_round_T=False):

        self.obs=obs
        self.intpars=intpars
        self.use_round_T=use_round_T

        mess="psf model should be turb or gauss,got '%s'" % model
        assert model in ['turb','gauss'],mess

        self.model=model
        self.lm_pars=lm_pars
        self.set_guess0(Tguess)

    def go(self, ntry=1):
        from .fitting import LMSimple

        if self.intpars is not None:
            npoints=self.intpars['npoints']
            #print("psffit using npoints:",npoints)
        else:
            npoints=None


        for i in xrange(ntry):
            guess=self.get_guess()
            fitter=LMSimple(self.obs,self.model,lm_pars=self.lm_pars,
                            use_round_T=self.use_round_T,
                            npoints=npoints)
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

class PSFRunnerCoellip(object):
    """
    wrapper to generate guesses and run the psf fitter a few times
    """
    def __init__(self, obs, Tguess, ngauss, lm_pars, intpars=None):

        self.obs=obs
        self.intpars=intpars

        self.ngauss=ngauss
        self.npars = get_coellip_npars(ngauss)
        self.model='coellip'
        self.lm_pars=lm_pars
        self.set_guess0(Tguess)
        self._set_prior()

    def _set_prior(self):
        from .joint_prior import PriorCoellipSame
        from .priors import CenPrior, ZDisk2D, TwoSidedErf

        Tguess=self.Tguess
        Fguess=self.Fguess

        cen_width=2*self.pixel_scale
        cen_prior = CenPrior(0.0, 0.0, cen_width, cen_width)
        g_prior=ZDisk2D(1.0)
        T_prior = TwoSidedErf(0.01*Tguess, 0.001*Tguess, 100*Tguess, Tguess)
        F_prior = TwoSidedErf(0.01*Fguess, 0.001*Fguess, 100*Fguess, Fguess)

        self.prior=PriorCoellipSame(self.ngauss,
                                    cen_prior,
                                    g_prior,
                                    T_prior,
                                    F_prior)

    def go(self, ntry=1):
        from .fitting import LMCoellip

        if self.intpars is not None:
            npoints=self.intpars['npoints']
            #print("psf coellip fit using npoints:",npoints)
        else:
            npoints=None

        for i in xrange(ntry):
            guess=self.get_guess()
            fitter=LMCoellip(self.obs,self.ngauss,lm_pars=self.lm_pars, prior=self.prior,
                             npoints=npoints)
            fitter.go(guess)

            res=fitter.get_result()
            if res['flags']==0:
                break

        self.fitter=fitter

    def get_guess(self):

        guess=numpy.zeros(self.npars)

        guess[0:0+2] = 0.01*srandu(2)
        guess[2:2+2] = 0.05*srandu(2)

        fac=0.01
        if self.ngauss==1:
            guess[4] = self.Tguess*(1.0 + 0.1*srandu())
            guess[5] = self.Fguess*(1.0 + 0.1*srandu())
        elif self.ngauss==2:
            guess[4] = self.Tguess*_moffat2_fguess[0]*(1.0 + fac*srandu())
            guess[5] = self.Tguess*_moffat2_fguess[1]*(1.0 + fac*srandu())

            guess[6] = self.Fguess*_moffat2_pguess[0]*(1.0 + fac*srandu())
            guess[7] = self.Fguess*_moffat2_pguess[1]*(1.0 + fac*srandu())

        elif self.ngauss==3:
            guess[4] = self.Tguess*_moffat3_fguess[0]*(1.0 + fac*srandu())
            guess[5] = self.Tguess*_moffat3_fguess[1]*(1.0 + fac*srandu())
            guess[6] = self.Tguess*_moffat3_fguess[2]*(1.0 + fac*srandu())

            guess[7] = self.Fguess*_moffat3_pguess[0]*(1.0 + fac*srandu())
            guess[8] = self.Fguess*_moffat3_pguess[1]*(1.0 + fac*srandu())
            guess[9] = self.Fguess*_moffat3_pguess[2]*(1.0 + fac*srandu())

        elif self.ngauss==4:
            guess[4] = self.Tguess*_moffat4_fguess[0]*(1.0 + fac*srandu())
            guess[5] = self.Tguess*_moffat4_fguess[1]*(1.0 + fac*srandu())
            guess[6] = self.Tguess*_moffat4_fguess[2]*(1.0 + fac*srandu())
            guess[7] = self.Tguess*_moffat4_fguess[3]*(1.0 + fac*srandu())

            guess[8]  = self.Fguess*_moffat4_pguess[0]*(1.0 + fac*srandu())
            guess[9]  = self.Fguess*_moffat4_pguess[1]*(1.0 + fac*srandu())
            guess[10] = self.Fguess*_moffat4_pguess[2]*(1.0 + fac*srandu())
            guess[11] = self.Fguess*_moffat4_pguess[3]*(1.0 + fac*srandu())


        else:
            raise RuntimeError("ngauss should be 1,2,3,4")

        return guess

    def set_guess0(self, Tguess):

        self.pixel_scale = self.obs.jacobian.get_scale()
        self.Tguess=Tguess
        Fguess = self.obs.image.sum()
        Fguess *= self.pixel_scale**2

        self.Fguess=Fguess

_moffat2_pguess=array([0.5, 0.5])
_moffat2_fguess=array([0.48955064,  1.50658978])

_moffat3_pguess=array([ 0.27559669,  0.55817131,  0.166232  ])
_moffat3_fguess=array([ 0.36123609,  0.8426139,   2.58747785])

_moffat4_pguess=array([0.44534,  0.366951,  0.10506,  0.0826497])
_moffat4_fguess=array([0.541019,  1.19701,  0.282176,  3.51086])
#_moffat3_pguess=array([0.45, 0.45, 0.1])
#_moffat3_fguess=array([0.48955064,  1.50658978, 3.0])

class MaxRunner(object):
    """
    wrapper to generate guesses and run the fitter a few times
    """
    def __init__(self, obs, model, max_pars, guesser, prior=None,
                 intpars=None,
                 use_logpars=False, use_round_T=False):

        self.obs=obs
        self.intpars=intpars
        self.use_round_T=use_round_T

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
        
        if self.intpars is not None:
            npoints=self.intpars['npoints']
            #print("max gal fit using npoints:",npoints)
        else:
            npoints=None

        fitclass=self._get_lm_fitter_class()

        for i in xrange(ntry):
            guess=self.guesser()
            fitter=fitclass(self.obs,
                            self.model,
                            lm_pars=self.send_pars,
                            use_logpars=self.use_logpars,
                            use_round_T=self.use_round_T,
                            npoints=npoints,
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

def get_coellip_ngauss(name):
    ngauss=int( name[7:] )
    return ngauss


def replace_masked_pixels(mb_obs_list,
                          inplace=False,
                          method='best-fit',
                          fitter=None):
    """
    replaced masked pixels

    The original image is stored for each Observation as .image_orig

    parameters
    ----------
    mb_obs_list: MultiBandObsList
        The original observations
    inplace: bool
        If True, modify the data in place.  Default False; a full
        copy is made.
    method: string, optional
        Method for replacement.  Supported methods are 'best-fit'.
        Default is 'best-fit'
    fitter:
        when method=='best-fit', a fitter from fitting.py
    """

    assert method=='best-fit',"only best-fit replacement is supported"
    assert fitter is not None,"fitter required"

    if inplace:
        mbo = mb_obs_list
    else:
        mbo = deepcopy( mb_obs_list )

    nband = len(mbo)

    for band in xrange(nband):
        olist = mbo[band]
        nobs = len(olist)
        for iobs,obs in enumerate(olist):

            bmask = obs.bmask
            if bmask is not None:
                w=where(bmask != 0)

                if w[0].size > 0:
                    print("    replacing %d/%d masked pixels" % (w[0].size,bmask.size))
                    obs.image_orig = obs.image.copy()
                    gm = fitter.get_convolved_gmix(band=band, obsnum=iobs)

                    im = obs.image
                    model_image = gm.make_image(im.shape, jacobian=obs.jacobian)

                    im[w] = model_image[w]
                else:
                    obs.image_orig=None

    return mbo


_em2_fguess =array([0.5793612389470884,1.621860687127999])
_em2_pguess =array([0.596510042804182,0.4034898268889178])

_em3_pguess = array([0.596510042804182,0.4034898268889178,1.303069003078001e-07])
_em3_fguess = array([0.5793612389470884,1.621860687127999,7.019347162356363],dtype='f8')
#_em3_pguess = array([0.7189864,0.2347828,0.04623086])
#_em3_fguess = array([0.4431912,1.354587,8.274546])

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
