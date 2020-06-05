"""
- todo
    - remove old unused fitters
"""
from sys import stdout
import numpy
from numpy import array, zeros, diag
from numpy import exp, sqrt, where, isfinite
from numpy import linalg
from numpy.linalg import LinAlgError
from pprint import pformat

from . import gmix
from .gmix import GMixList, MultiBandGMixList

from .priors import LOWVAL, BIGVAL

from .gexceptions import GMixRangeError

from .observation import Observation,ObsList,get_mb_obs

from . import stats


from .gmix_nb import gmix_convolve_fill
from .fitting_nb import fill_fdiff

MAX_TAU=0.1
MIN_ARATE=0.2
MCMC_NTRY=1

# default values
PDEF=-9.999e9    # parameter defaults
CDEF=9.999e9     # covariance or error defaults

# flags

BAD_VAR=2**0     # variance not positive definite
LOW_ARATE=2**1   # info flag about arate

# error codes in LM start at 2**0 and go to 2**3
# this is because we set 2**(ier-5)
LM_SINGULAR_MATRIX = 2**4
LM_NEG_COV_EIG = 2**5
LM_NEG_COV_DIAG = 2**6
EIG_NOTFINITE = 2**7
LM_FUNC_NOTFINITE = 2**8

DIV_ZERO = 2**9  # division by zero

ZERO_DOF = 2**10 # dof zero so can't do chi^2/dof


class FitterBase(object):
    """
    Base for other fitters

    The basic input is the Observation (or ObsList or MultiBandObsList)

    Designed to fit many images at once.  For this reason, a jacobian
    transformation is used to put all on the same system; this is part of each
    Observation object. For the same reason, the center of the model is
    relative to "zero", which points to the common center used by all
    transformation objects; the row0,col0 in pixels for each should correspond
    to that center in the common coordinates (e.g. sky coords)

    Fluxes and sizes will also be in the transformed system.

    """
    def __init__(self, obs, model, **keys):
        self.keys=keys

        self.use_round_T=keys.get('use_round_T',False)
        assert self.use_round_T is False,"no longer support round T"

        self.set_obs(obs)

        self.prior = keys.get('prior',None)

        # in this case, image, weight, jacobian, psf are going to
        # be lists of lists.

        self.model=gmix.get_model_num(model)
        self.model_name=gmix.get_model_name(self.model)
        self._set_npars()

        self._set_totpix()

        self._gmix_all=None

    def __repr__(self):
        rep="""
    %(model)s
    %(extra)s
        """
        if hasattr(self,'_result'):
            extra=pformat(self._result)
        else:
            extra=''

        rep=rep % {
            'model':self.model_name,
            'extra':extra,
        }
        return rep

    def get_result(self):
        """
        Result will not be non-None until sampler is run
        """

        if not hasattr(self,'_result'):
            raise ValueError('No result, you must run_mcmc '
                             'and calc_result first')
        return self._result

    def get_gmix(self, band=0):
        """
        Get a gaussian mixture at the fit parameter set, which
        definition depends on the sub-class

        parameters
        ----------
        band: int, optional
            Band index, default 0
        """
        res=self.get_result()
        pars=self.get_band_pars(res['pars'], band)
        return gmix.make_gmix_model(pars, self.model)

    def get_convolved_gmix(self, band=0, obsnum=0):
        """
        get a gaussian mixture at the fit parameters, convolved by the psf if
        fitting a pre-convolved model

        parameters
        ----------
        band: int, optional
            Band index, default 0
        obsnum: int, optional
            Number of observation for the given band,
            default 0
        """

        gm = self.get_gmix(band)

        obs = self.obs[band][obsnum]
        if obs.has_psf_gmix():
            gm = gm.convolve(obs.psf.gmix)

        return gm

    def set_obs(self, obs_in):
        """
        Input should be an Observation, ObsList, or MultiBandObsList
        """

        self.obs = get_mb_obs(obs_in)


        self.nband=len(self.obs)
        nimage = 0
        for obslist in self.obs:
            for obs in obslist:
                nimage += 1
        self.nimage=nimage

    def _set_totpix(self):
        """
        Make sure the data are consistent.
        """

        totpix=0
        for obs_list in self.obs:
            for obs in obs_list:
                totpix += obs.pixels.size

        self.totpix=totpix

    def _set_npars(self):
        """
        nband should be set in set_lists, called before this
        """
        self.npars=gmix.get_model_npars(self.model) + self.nband-1


    def get_effective_npix_old(self):
        """
        Because of the weight map, each pixel gets a different weight in the
        chi^2.  This changes the effective degrees of freedom.  The extreme
        case is when the weight is zero; these pixels are essentially not used.

        We replace the number of pixels with

            eff_npix = sum(weights)maxweight
        """
        raise RuntimeError("this is bogus")
        if not hasattr(self, 'eff_npix'):
            wtmax = 0.0
            wtsum = 0.0

            for obs_list in self.obs:
                for obs in obs_list:
                    wt=obs.weight

                    this_wtmax = wt.max()
                    if this_wtmax > wtmax:
                        wtmax = this_wtmax

                    wtsum += wt.sum()

            self.eff_npix=wtsum/wtmax

        if self.eff_npix <= 0:
            self.eff_npix=1.e-6

        return self.eff_npix


    def calc_lnprob(self, pars, more=False):
        """
        This is all we use for mcmc approaches, but also used generally for the
        "get_fit_stats" method.  For the max likelihood fitter we also have a
        _get_ydiff method
        """

        try:

            # these are the log pars (if working in log space)
            ln_priors = self._get_priors(pars)

            lnprob = 0.0
            s2n_numer=0.0
            s2n_denom=0.0
            npix = 0


            self._fill_gmix_all(pars)
            for band in range(self.nband):

                obs_list=self.obs[band]
                gmix_list=self._gmix_all[band]

                for obs,gm in zip(obs_list, gmix_list):

                    res = gm.get_loglike(
                        obs,
                        more=more,
                    )

                    if more:
                        lnprob    += res['loglike']
                        s2n_numer += res['s2n_numer']
                        s2n_denom += res['s2n_denom']
                        npix      += res['npix']
                    else:
                        lnprob += res

            # total over all bands
            lnprob += ln_priors

        except GMixRangeError:
            lnprob = LOWVAL
            s2n_numer=0.0
            s2n_denom=BIGVAL
            npix = 0

        if more:
            return {
                'lnprob':lnprob,
                's2n_numer':s2n_numer,
                's2n_denom':s2n_denom,
                'npix':npix,
            }
        else:
            return lnprob

    def get_fit_stats(self, pars):
        """
        Get some fit statistics for the input pars.

        pars must be in the log scaling!
        """

        res=self.calc_lnprob(pars, more=True)

        if res['s2n_denom'] > 0:
            s2n=res['s2n_numer']/sqrt(res['s2n_denom'])
        else:
            s2n=0.0

        chi2    = res['lnprob']/(-0.5)
        dof     = res['npix'] - self.npars
        chi2per = chi2/dof

        res['chi2per'] = chi2per
        res['dof']     = dof
        res['s2n_w']   = s2n
        res['s2n']     = s2n

        return res

    def _make_model(self, band_pars):
        gm0=gmix.make_gmix_model(band_pars, self.model)
        return gm0

    def _init_gmix_all(self, pars):
        """
        input pars are in linear space

        initialize the list of lists of gaussian mixtures
        """

        if self.obs[0][0].has_psf_gmix():
            self.dopsf=True
        else:
            self.dopsf=False

        gmix_all0 = MultiBandGMixList()
        gmix_all  = MultiBandGMixList()

        for band,obs_list in enumerate(self.obs):
            gmix_list0=GMixList()
            gmix_list=GMixList()

            # pars for this band, in linear space
            band_pars=self.get_band_pars(pars, band)

            for obs in obs_list:
                gm0 = self._make_model(band_pars)
                if self.dopsf:
                    psf_gmix=obs.psf.gmix
                    gm=gm0.convolve(psf_gmix)
                else:
                    gm=gm0.copy()

                gmix_list0.append(gm0)
                gmix_list.append(gm)

            gmix_all0.append(gmix_list0)
            gmix_all.append(gmix_list)

        self._gmix_all0 = gmix_all0
        self._gmix_all  = gmix_all

    def _convolve_gmix(self, gm, gm0, psf_gmix):
        """
        norms get set
        """
        gmix_convolve_fill(gm._data, gm0._data, psf_gmix._data)

    def _fill_gmix_all(self, pars):
        """
        input pars are in linear space

        Fill the list of lists of gmix objects for the given parameters
        """

        if not self.dopsf:
            self._fill_gmix_all_nopsf(pars)
            return

        for band,obs_list in enumerate(self.obs):
            gmix_list0=self._gmix_all0[band]
            gmix_list=self._gmix_all[band]

            band_pars=self.get_band_pars(pars, band)

            for i,obs in enumerate(obs_list):

                psf_gmix=obs.psf.gmix

                gm0=gmix_list0[i]
                gm=gmix_list[i]

                gm0._fill(band_pars)
                self._convolve_gmix(gm, gm0, psf_gmix)


    def _fill_gmix_all_nopsf(self, pars):
        """
        Fill the list of lists of gmix objects for the given parameters
        """

        for band,obs_list in enumerate(self.obs):
            gmix_list=self._gmix_all[band]

            band_pars=self.get_band_pars(pars, band)

            for i,obs in enumerate(obs_list):

                gm=gmix_list[i]

                gm._fill(band_pars)

    def _get_priors(self, pars):
        """
        get the sum of ln(prob) from the priors or 0.0 if
        no priors were sent
        """
        if self.prior is None:
            return 0.0
        else:
            return self.prior.get_lnprob_scalar(pars)

    def plot_residuals(self, title=None, show=False,
                       width=1920, height=1200,**keys):
        import images
        import biggles

        biggles.configure('screen','width', width)
        biggles.configure('screen','height', height)

        res=self.get_result()
        try:
            self._fill_gmix_all(res['pars'])
        except GMixRangeError as gerror:
            print(str(gerror))
            return None

        plist=[]
        for band in range(self.nband):

            band_list=[]

            obs_list=self.obs[band]
            gmix_list=self._gmix_all[band]

            nim=len(gmix_list)

            ttitle='band: %s' % band
            if title is not None:
                ttitle='%s %s' % (title, ttitle)

            for i in range(nim):

                this_title = '%s cutout: %d' % (ttitle, i+1)

                obs=obs_list[i]
                gm=gmix_list[i]

                im=obs.image
                wt=obs.weight
                j=obs.jacobian

                model=gm.make_image(im.shape,jacobian=j)

                showim = im*wt
                showmod = model*wt

                sub_tab=images.compare_images(showim, showmod,show=False,
                                              label1='galaxy',
                                              label2='model',
                                              **keys)
                sub_tab.title=this_title

                band_list.append(sub_tab)

                if show:
                    sub_tab.show()

            plist.append(band_list)
        return plist

    def calc_cov(self, h, m, diag_on_fail=True):
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

        res=self.get_result()

        bad=True

        try:
            cov = self.get_cov(
                res['pars'],
                h=h,
                m=m,
                diag_on_fail=diag_on_fail,
            )

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

    def get_cov(self, pars, h, m, diag_on_fail=True):
        """
        calculate the covariance matrix at the specified point

        This method understands the natural bounds on ellipticity.
        If the ellipticity is larger than 1-m*h then it is scaled
        back, perserving the angle.

        If the Hessian is singular, an attempt is made to invert
        a diagonal version. If that fails, LinAlgError is raised.

        parameters
        ----------
        pars: array
            Array of parameters at which to evaluate the cov matrix
        h: step size, optional
            Step size for finite differences, default 1.0e-3
        m: scalar
            The max allowed ellipticity is 1-m*h.
            Note the derivatives require evaluations at +/- h,
            so m should be greater than 1.

        Raises
        ------
        LinAlgError:
            If the hessian is singular a diagonal version is tried
            and if that fails finally a LinAlgError is raised.
        """
        import covmatrix

        # get a copy as an array
        pars=numpy.array(pars)

        g1=pars[2]
        g2=pars[3]

        g=sqrt(g1**2 + g2**2)

        maxg=1.0-m*h

        if g > maxg:
            fac = maxg/g
            g1 *= fac
            g2 *= fac
            pars[2] = g1
            pars[3] = g2

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
                cov = -linalg.inv(hdiag)
            else:
                raise
        return cov



class TemplateFluxFitter(FitterBase):
    """
    We fix the center, so this is linear.  Just cross-correlations
    between model and data.

    The center of the jacobian(s) must point to a common place on the sky, and
    if the center is input (to reset the gmix centers),) it is relative to that
    position

    parameters
    -----------
    obs: Observation or ObsList
        See ngmix.observation.Observation.  The observation should
        have a gmix set.
    cen: 2-element sequence, optional
        The center in sky coordinates, relative to the jacobian center(s).  If
        not sent, the gmix (or psf gmix) object(s) in the observation(s) should
        be set to the wanted center.
    normalize_psf: True or False
        if True, then normalize PSF gmix to flux of unity, otherwise use input
        normalization.

    """
    def __init__(self, obs, **keys):

        self.keys=keys
        self.do_psf=keys.get('do_psf',False)
        self.cen=keys.get('cen',None)

        self.normalize_psf = keys.get('normalize_psf',True)
        self.simulate_err=keys.get('simulate_err',False)

        if self.simulate_err:
            rng=keys.get("rng",None)
            if rng is None:
                rng = numpy.random.RandomState()
            self.rng=rng


        if self.cen is None:
            self.cen_was_sent=False
        else:
            self.cen_was_sent=True

        self._set_obs(obs)

        self.model_name='template'
        self.npars=1

        self._set_totpix()

    def go(self):
        """
        calculate the flux using zero-lag cross-correlation
        """

        flags=0

        xcorr_sum=0.0
        msq_sum=0.0

        chi2=0.0

        nobs=len(self.obs)

        flux=PDEF
        flux_err=CDEF

        for ipass in [1,2]:
            for iobs in range(nobs):
                obs  = self.obs[iobs]

                im=obs.image
                wt=obs.weight

                if ipass==1:
                    model = self._get_model(iobs)
                    xcorr_sum += (model*im*wt).sum()
                    msq_sum += (model*model*wt).sum()
                else:
                    model = self._get_model(iobs, flux=flux)

                    chi2 += self._get_chi2(model, im, wt)

            if ipass==1:
                if msq_sum==0:
                    break
                flux = xcorr_sum/msq_sum

        # chi^2 per dof and error checking
        dof=self.get_dof()
        chi2per=9999.0
        if dof > 0:
            chi2per=chi2/dof
        else:
            flags |= ZERO_DOF

        # final flux calculation with error checking
        if msq_sum==0 or self.totpix==1:
            flags |= DIV_ZERO
        else:

            arg=chi2/msq_sum/(self.totpix-1)
            if arg >= 0.0:
                flux_err = sqrt(arg)
            else:
                flags |= BAD_VAR

        self._result={'model':self.model_name,
                      'flags':flags,
                      'chi2per':chi2per,
                      'dof':dof,
                      'flux':flux,
                      'flux_err':flux_err}


    def _get_chi2(self, model, im, wt):
        """
        get the chi^2 for this image/model

        we can simulate when needed
        """
        if self.simulate_err:
            err = numpy.zeros(model.shape)
            w=numpy.where(wt > 0)
            err[w] = numpy.sqrt(1.0/wt[w])
            noisy_model = model.copy()
            noisy_model += self.rng.normal(size=model.shape)*err
            chi2 =( (model-noisy_model)**2 *wt ).sum()
        else:
            chi2 =( (model-im)**2 *wt ).sum()

        return chi2

    def _get_model(self, iobs, flux=None):
        """
        get the model image
        """
        if self.use_template:
            if flux is not None:
                model = self.template_list[iobs].copy()
                norm = self.norm_list[iobs]
                model *= (norm*flux)/model.sum()
            else:
                model = self.template_list[iobs]

        else:

            if flux is None:
                gm = self.gmix_list[iobs]
            else:
                gm = self.gmix_list[iobs].copy()
                norm = self.norm_list[iobs]
                gm.set_flux(flux*norm)

            obs = self.obs[iobs]
            dims = obs.image.shape
            jac = obs.jacobian
            model = gm.make_image(dims, jacobian=jac)

        return model

    def get_dof(self):
        """
        Effective def based on effective number of pixels
        """
        npix=self.get_effective_npix()
        dof = npix-self.npars
        if dof <= 0:
            dof = 1.e-6
        return dof



    def _set_obs(self, obs_in):
        """
        Input should be an Observation, ObsList
        """

        if isinstance(obs_in,Observation):
            obs_list=ObsList()
            obs_list.append(obs_in)
        elif isinstance(obs_in,ObsList):
            obs_list=obs_in
        else:
            raise ValueError("obs should be Observation or ObsList")

        tobs = obs_list[0]
        if self.do_psf:
            tobs = tobs.psf

        if not tobs.has_gmix():
            if not hasattr(tobs, 'template'):
                raise ValueError("neither gmix or template image are set")

        self.obs = obs_list
        if tobs.has_gmix():
            self._set_gmix_and_norms()
        else:
            self._set_templates_and_norms()

    def _set_gmix_and_norms(self):
        self.use_template=False
        cen=self.cen
        gmix_list = []
        norm_list = []
        for obs in self.obs:
            # these return copies, ok to modify
            if self.do_psf:
                gmix=obs.get_psf_gmix()
                if self.normalize_psf:
                    gmix.set_flux(1.0)
            else:
                gmix=obs.get_gmix()
                gmix.set_flux(1.0)

            if self.cen_was_sent:
                gmix.set_cen(cen[0], cen[1])

            gmix_list.append(gmix)
            norm_list.append(gmix.get_flux())

        self.gmix_list = gmix_list
        self.norm_list = norm_list

    def _set_templates_and_norms(self):
        self.use_template=True

        assert self.cen_was_sent is False

        template_list = []
        norm_list = []
        for obs in self.obs:
            if self.do_psf:
                template=obs.psf.template.copy()
                norm = template.sum()
                if self.normalize_psf:
                    template *= 1.0/norm
                    norm = 1.0
            else:
                template=obs.template.copy()
                template *= 1.0/template.sum()
                norm = 1.0

            template_list.append(template)
            norm_list.append(norm)

        self.template_list = template_list
        self.norm_list = norm_list


    def _set_totpix(self):
        """
        Make sure the data are consistent.
        """

        totpix=0
        for obs in self.obs:
            totpix += obs.pixels.size

        self.totpix=totpix

    def get_effective_npix(self):
        """
        We don't use all pixels, only those with weight > 0
        """
        if not hasattr(self, 'eff_npix'):

            npix=0
            for obs in self.obs:
                wt=obs.weight

                w=where(wt > 0)
                npix += w[0].size

            self.eff_npix=npix

        return self.eff_npix

    def get_npix(self):
        """
        just get the total number of pixels in all images
        """
        if not hasattr(self, '_npix'):
            npix=0
            for obs in self.obs:
                npix += obs.image.size

            self._npix=npix

        return self._npix

class FracdevFitterMax(FitterBase):
    def __init__(self, obs, exp_pars, dev_pars, **keys):
        """
        obs must have psf (with gmix) set
        """
        self.prior=keys.get('prior',None)
        method=keys.get('method','nm')
        if self.prior is not None:
            assert method=='nm','if prior sent method must be nm'
            if self.prior.ndim > 1:
                if len(exp_pars) > 6:
                    raise RuntimeError("currently only joint "
                                       "fracdev prior single band")

        self.method=method

        self.set_obs(obs)
        self._set_images(exp_pars, dev_pars)

        self.model_name='fracdev'

        self._set_totpix()

        pars=keys.get('pars',None)
        if pars is None:
            pars={'maxfev':4000}

        self.pars=pars
        self.npars=1

        self.fdiff_size=self.totpix

    def go(self, fracdev_guess):
        """
        run max like fit
        """
        fracdev_guess=array(fracdev_guess,dtype='f8',ndmin=1,copy=False)
        if fracdev_guess.size != 1:
            raise ValueError("fracdev is a scalar or len 1 array")

        if self.method=='lm':
            return self._go_lm(fracdev_guess)
        else:
            return self._go_nm(fracdev_guess)

    def _go_lm(self, fracdev_guess):
        """
        Run leastsq and set the result
        """

        n_prior_pars=0
        result = run_leastsq(self._calc_fdiff,
                             fracdev_guess,
                             n_prior_pars,
                             **self.pars)

        result['model'] = self.model_name
        if result['flags']==0:
            result['fracdev'] = result['pars'][0].copy()
            result['fracdev_err'] = sqrt(result['pars_cov'][0,0])

        self._result=result

    def _go_nm(self, fracdev_guess):
        """
        Run leastsq and set the result
        """
        from .simplex import minimize_neldermead

        result = minimize_neldermead(self._calc_neg_lnprob,
                                     fracdev_guess,
                                     **self.pars)

        self._result = result

        result['model'] = self.model_name
        if result['success']:
            result['flags'] = 0
        else:
            result['flags'] = 1

        if 'x' in result:
            pars=result['x']
            result['pars'] = pars

            h=1.0e-3
            m=5.0
            self.calc_cov(h, m)

            result['fracdev'] = result['pars'][0].copy()

            if 'pars_err' in result:
                result['fracdev_err'] = result['pars_err'][0]

    def get_cov(self, pars, h, m, diag_on_fail=True):
        """
        calculate the covariance matrix at the specified point

        This method understands the natural bounds on ellipticity.
        If the ellipticity is larger than 1-m*h then it is scaled
        back, perserving the angle.

        If the Hessian is singular, an attempt is made to invert
        a diagonal version. If that fails, LinAlgError is raised.

        parameters
        ----------
        pars: array
            Array of parameters at which to evaluate the cov matrix
        h: step size, optional
            Step size for finite differences, default 1.0e-3
        m: scalar
            The max allowed ellipticity is 1-m*h.
            Note the derivatives require evaluations at +/- h,
            so m should be greater than 1.

        Raises
        ------
        LinAlgError:
            If the hessian is singular a diagonal version is tried
            and if that fails finally a LinAlgError is raised.
        """
        import covmatrix

        hess=covmatrix.calc_hess(self.calc_lnprob, pars, h)

        try:
            cov = -linalg.inv(hess)
        except LinAlgError:
            # pull out a diagonal version of the hessian
            # this might still fail
            if diag_on_fail:
                hdiag=diag(diag(hess))
                cov = -linalg.inv(hdiag)
            else:
                raise
        return cov



    def _calc_fdiff(self, pars, **keys):
        """

        vector with (model-data)/error.

        The npars elements contain -ln(prior)
        """

        fracdev = pars[0]

        # we cannot keep sending existing array into leastsq, don't know why
        fdiff=zeros(self.fdiff_size)

        if fracdev <= -1 or fracdev >= 2.0:
            fdiff[:] = LOWVAL
            return fdiff

        start=0

        for band in range(self.nband):

            obs_list=self.obs[band]

            images   = self._images[band]
            sweights = self._sweights[band]
            eimages  = self._eimages[band]
            dimages  = self._dimages[band]

            for epoch,obs in enumerate(obs_list):

                image   = images[epoch]
                sweight = sweights[epoch]

                npix = image.size

                tfdiff    = eimages[epoch].copy()
                dev_image = dimages[epoch].copy()

                tfdiff *= (1.0-fracdev)

                dev_image *= fracdev

                tfdiff += dev_image

                tfdiff -= image
                tfdiff *= sweight

                fdiff[start:start+npix] = tfdiff

                start += npix


        return fdiff

    def _calc_neg_lnprob(self, pars):
        return -self.calc_lnprob(pars)

    def calc_lnprob(self, pars):
        """
        return log(prob)
        """

        fracdev = pars[0]
        ifracdev = 1.0-pars[0]

        if fracdev <= -1 or fracdev >= 2.0:
            return LOWVAL


        lnprob = 0.0

        for band in range(self.nband):

            obs_list=self.obs[band]

            images   = self._images[band]
            weights  = self._weights[band]
            eimages  = self._eimages[band]
            dimages  = self._dimages[band]

            for epoch,obs in enumerate(obs_list):

                image   = images[epoch]
                weight = weights[epoch]

                exp_image = eimages[epoch].copy()
                dev_image = dimages[epoch].copy()

                exp_image *= ifracdev
                dev_image *= fracdev

                tfdiff = exp_image

                tfdiff += dev_image

                tfdiff -= image
                tfdiff *= tfdiff
                tfdiff *= weight

                lnprob += tfdiff.sum()


        lnprob *= (-0.5)

        if self.prior is not None:
            if self.prior.ndim==1:
                #print("using prior")
                lnprob += self.prior.get_lnprob_scalar(pars)
            elif self.prior.ndim==2:
                #print("using joint F prior")
                # it is a joint prior on F and fracdev
                Fe = ifracdev* self.lin_exp_F[0]
                Fd = fracdev * self.lin_dev_F[0]
                F = Fe + Fd

                bad=False
                allpars=numpy.array( [F, fracdev] )

                if bad:
                    lnprob = -numpy.inf
                else:
                    lnp = self.prior.get_lnprob_scalar(allpars)
                    lnprob += lnp

            else:
                #print("using joint prior")
                # it is a joint prior on TF and fracdev
                Fe = ifracdev* self.lin_exp_F[0]
                Fd = fracdev * self.lin_dev_F[0]
                F = Fe + Fd
                T = (Fe*self.lin_exp_T + Fd*self.lin_dev_T)/F

                bad=False
                allpars=numpy.array( [T, F, fracdev] )

                if bad:
                    lnprob = -numpy.inf
                else:
                    lnp = self.prior.get_lnprob_scalar(allpars)
                    lnprob += lnp

        return lnprob



    def _set_images(self, exp_pars, dev_pars):
        exp_pars=array(exp_pars,dtype='f8',ndmin=1,copy=True)
        dev_pars=array(dev_pars,dtype='f8',ndmin=1,copy=True)

        self.lin_exp_T = exp_pars[4]
        self.lin_dev_T = dev_pars[4]
        self.lin_exp_F = exp_pars[5:].copy()
        self.lin_dev_F = dev_pars[5:].copy()

        self.TdByTe = dev_pars[4]/exp_pars[4]

        nb=self.nband

        enb=exp_pars.size-5
        dnb=dev_pars.size-5
        mess="expected %d bands, got %d"
        if enb != nb:
            raise ValueError(mess % (nb,enb))
        if dnb != nb:
            raise ValueError(mess % (nb,dnb))


        tepars = zeros(6)
        tdpars = zeros(6)

        tepars[0:5] = exp_pars[0:5]
        tdpars[0:5] = dev_pars[0:5]


        all_images=[]
        all_weights=[]
        all_sweights=[]
        all_eimages=[]
        all_dimages=[]

        for band in range(nb):
            # these will get reset later based on fracdev
            tepars[5] = exp_pars[5+band]
            tdpars[5] = dev_pars[5+band]

            band_obs = self.obs[band]
            nepoch = len(band_obs)

            images=[]
            weights=[]
            sweights=[]
            eimages=[]
            dimages=[]

            for iepoch in range(nepoch):
                epoch_obs = band_obs[iepoch]

                image=epoch_obs.image.copy()

                psf_gmix=epoch_obs.psf.gmix

                cweight = epoch_obs.weight.clip(min=0.0)
                sweight = sqrt(cweight)

                tegm = gmix.GMixModel(tepars,'exp')
                tdgm = gmix.GMixModel(tdpars,'dev')

                egm = tegm.convolve( psf_gmix )
                dgm = tdgm.convolve( psf_gmix )

                eimage = egm.make_image(image.shape,
                                        jacobian=epoch_obs.jacobian)
                dimage = dgm.make_image(image.shape,
                                        jacobian=epoch_obs.jacobian)


                images.append(image.ravel())
                weights.append(cweight.ravel())
                sweights.append(sweight.ravel())
                eimages.append(eimage.ravel())
                dimages.append(dimage.ravel())

            all_images.append(images)
            all_weights.append(weights)
            all_sweights.append(sweights)
            all_eimages.append(eimages)
            all_dimages.append(dimages)

        self._images   = all_images
        self._weights = all_weights
        self._sweights = all_sweights
        self._eimages  = all_eimages
        self._dimages  = all_dimages


class FracdevFitter(FitterBase):
    def __init__(self, obs, exp_pars, dev_pars, **keys):
        """
        obs must have psf (with gmix) set
        """

        self.npars=1

        self.set_obs(obs)

        self.model_name='fracdev'

        self._set_totpix()

        self._set_arrays(exp_pars, dev_pars)

        self._do_lstsq()

    def _do_lstsq(self):
        """
        (data - expmod) = (devmod-expmod) * fracdev

        or

        Y = X * fracdev
        """

        X = self.X[:,numpy.newaxis]
        Y = self.Y

        self._result={'model': 'fracdev', 'flags':0,'nfev':1}


        result=self._result

        try:
            pars, resid, _, _ = numpy.linalg.lstsq(X, Y)

            result['pars'] = pars
            result['fracdev'] = pars[0]

            self.calc_cov()

            result['fracdev_err'] = result['pars_err'][0]

        except LinAlgError:
            result['flags'] |= EIG_NOTFINITE


    def calc_cov(self):
        """
        calculate the error
        """
        import covmatrix

        res=self.get_result()

        h=1.0e-3
        hess=covmatrix.calc_hess(self.calc_lnprob, res['pars'], h)
        try:
            cov = -linalg.inv(hess)
        except LinAlgError:
            # pull out a diagonal version of the hessian
            # this might still fail
            hdiag=diag(diag(hess))
            cov = -linalg.inv(hdiag)

        res['pars_cov'] = cov
        res['pars_err'] = sqrt(diag(cov))

    def calc_lnprob(self, pars):
        fracdev = pars[0]

        lnp_array = (self.Y - fracdev*self.X)
        lnp_array *= lnp_array

        lnp = lnp_array.sum()
        lnp *= (-0.5)

        return lnp

    def _set_arrays(self, exp_pars, dev_pars):
        """
        (data - expmod) = (devmod-expmod) * fracdev

        or

        Y = X * fracdev
        """
        X = zeros(self.totpix)
        Y = zeros(self.totpix)

        exp_pars=array(exp_pars,dtype='f8',ndmin=1,copy=True)
        dev_pars=array(dev_pars,dtype='f8',ndmin=1,copy=True)

        nb=self.nband

        enb=exp_pars.size-5
        dnb=dev_pars.size-5
        mess="expected %d bands, got %d"
        if enb != nb:
            raise ValueError(mess % (nb,enb))
        if dnb != nb:
            raise ValueError(mess % (nb,dnb))


        tepars = zeros(6)
        tdpars = zeros(6)

        tepars[0:5] = exp_pars[0:5]
        tdpars[0:5] = dev_pars[0:5]

        start=0
        for band in range(nb):
            # these will get reset later based on fracdev
            tepars[5] = exp_pars[5+band]
            tdpars[5] = dev_pars[5+band]

            band_obs = self.obs[band]
            nepoch = len(band_obs)

            for iepoch in range(nepoch):
                epoch_obs = band_obs[iepoch]

                image=epoch_obs.image.copy()

                psf_gmix=epoch_obs.psf.gmix

                cweight = epoch_obs.weight.clip(min=0.0)
                sweight = sqrt(cweight)

                tegm = gmix.GMixModel(tepars,'exp')
                tdgm = gmix.GMixModel(tdpars,'dev')

                egm = tegm.convolve( psf_gmix )
                dgm = tdgm.convolve( psf_gmix )

                eimage = egm.make_image(image.shape,
                                        jacobian=epoch_obs.jacobian)
                dimage = dgm.make_image(image.shape,
                                        jacobian=epoch_obs.jacobian)


                end = start + image.size
                X[start:end] = ( (dimage-eimage)*sweight ).ravel()
                Y[start:end] = ( (image-eimage)*sweight  ).ravel()

                start += image.size


        self.X = X
        self.Y = Y



class MaxSimple(FitterBase):
    """
    A class for direct maximization of the likelihood.
    Useful for seeding model parameters.

    some keywords to control fitting
    -----------------------------------------------------
    xtol: float, optional
        Tolerance in the vertices, relative to the vertex with
        the lowest function value.  Default 1.0e-4
    ftol: float, optional
        Tolerance in the function value, relative to the
        lowest function value for all vertices.  Default 1.0e-4
    maxiter: int, optional
        Default is npars*200
    maxfev:
        Default is npars*200

    """
    def __init__(self,
                 obs,
                 model,
                 method='Nelder-Mead',
                 options=None,
                 **keys):
        super(MaxSimple,self).__init__(obs, model, **keys)
        self._obs = obs

        self._model = model

        self._band_pars = numpy.zeros(6)

        self._method = method

        if method=='Nelder-Mead':
            if options is not None:
                self._options = options
            else:
                self._options = {}
        else:
            self._options=options

            self._max_keys={}
            tocheck=['jac','hess', 'hessp', 'bounds',
                     'constraints', 'tol', 'callback']
            for key in tocheck:
                if key in keys:
                    self._max_keys[key] = keys[key]

    def _setup_data(self, guess):
        """
        initialize the gaussian mixtures
        """

        if hasattr(self,'_result'):
            del self._result

        self.flags=0

        npars=guess.size
        mess="guess has npars=%d, expected %d" % (npars,self.npars)
        assert (npars==self.npars),mess

        try:
            # this can raise GMixRangeError
            self._init_gmix_all(guess)
        except ZeroDivisionError:
            raise GMixRangeError("got zero division")

    def get_band_pars(self, pars_in, band):
        """
        Get linear pars for the specified band
        """

        pars=self._band_pars

        pars[:] = pars_in[:]
        return pars



    def neglnprob(self, pars):
        return -1.0*self.calc_lnprob(pars)

    def run_max(self, guess):
        """
        Run maximizer and set the result.

        parameters
        ----------
        guess: array/sequence
            Starting guess for the fitter
        """
        if self._method in ['nm','Nelder-Mead']:
            self.run_max_nm(guess, **self._options)
        else:
            import scipy.optimize

            guess=numpy.array(guess,dtype='f8',copy=False)
            self._setup_data(guess)

            result = scipy.optimize.minimize(
                self.neglnprob,
                guess,
                method=self._method,
                options=self._options,
                **self._max_keys
            )
            self._result = result

            result['model'] = self.model_name
            if result['success']:
                result['flags'] = 0
            else:
                result['flags'] = result['status']

            if 'x' in result:
                pars=result['x']
                result['pars'] = pars
                result['g'] = pars[2:2+2]

                # based on last entry
                fit_stats = self.get_fit_stats(pars)
                result.update(fit_stats)
    go=run_max

    def run_max_nm(self, guess, **keys):
        """
        Run maximizer and set the result.

        extra keywords to override those sent on construction
        -----------------------------------------------------
        xtol: float, optional
            Tolerance in the vertices, relative to the vertex with
            the lowest function value.  Default 1.0e-4
        ftol: float, optional
            Tolerance in the function value, relative to the
            lowest function value for all vertices.  Default 1.0e-4
        maxiter: int, optional
            Default is npars*200
        maxfev:
            Default is npars*200
        """
        from .simplex import minimize_neldermead

        options=self._options
        options.update(keys)

        guess=numpy.array(guess,dtype='f8',copy=False)
        self._setup_data(guess)

        result = minimize_neldermead(self.neglnprob,
                                     guess,
                                     **keys)
        self._result = result

        result['model'] = self.model_name
        if result['success']:
            result['flags'] = 0
        else:
            result['flags'] = 1

        if 'x' in result:
            pars=result['x']
            result['pars'] = pars
            result['g'] = pars[2:2+2]

            # based on last entry
            fit_stats = self.get_fit_stats(pars)
            result.update(fit_stats)

            h=1.0e-3
            m=5.0
            self.calc_cov(h, m)

class MaxCoellip(MaxSimple):
    """
    A class for direct maximization of the likelihood.
    Useful for seeding model parameters.
    """
    def __init__(self, obs, ngauss, method='Nelder-Mead', **keys):

        self._ngauss=ngauss

        super(MaxCoellip,self).__init__(obs, 'coellip', method=method, **keys)

        if self.nband != 1:
            raise ValueError("MaxCoellip only supports one band")

        # over-write the band pars created by MaxSimple
        self._band_pars=zeros(self.npars)

    def _set_npars(self):
        """
        single band, npars determined from ngauss
        """
        self.npars=4 + 2*self._ngauss

    def get_band_pars(self, pars_in, band):
        """
        Get linear pars for the specified band
        """

        pars=self._band_pars

        pars[:] = pars_in[:]
        return pars




_default_lm_pars={'maxfev':4000,
                  'ftol': 1.0e-5,
                  'xtol': 1.0e-5}


class LMSimple(FitterBase):
    """
    A class for doing a fit using levenberg marquardt

    """
    def __init__(self, obs, model, **keys):
        super(LMSimple,self).__init__(obs, model, **keys)

        # this is a dict
        # can contain maxfev, ftol (tol in sum of squares)
        # xtol (tol in solution), etc

        lm_pars=keys.get('lm_pars',None)
        if lm_pars is None:
            lm_pars=_default_lm_pars
        self.lm_pars=lm_pars

        self._set_n_prior_pars()

        self._set_fdiff_size()

        self._band_pars=zeros(6)

    def _set_n_prior_pars(self):
        # center1 + center2 + shape + T + fluxes
        if self.prior is None:
            self.n_prior_pars=0
        else:
            self.n_prior_pars=1 + 1 + 1 + 1 + self.nband


    def _set_fdiff_size(self):
        self.fdiff_size=self.totpix + self.n_prior_pars

    def go(self, guess):
        """
        Run leastsq and set the result
        """

        guess=array(guess,dtype='f8',copy=False)
        self._setup_data(guess)

        self._make_lists()

        bounds = self._get_bounds()

        result = run_leastsq(
            self._calc_fdiff,
            guess,
            self.n_prior_pars,
            bounds=bounds,
            **self.lm_pars
        )


        result['model'] = self.model_name
        if result['flags']==0:
            result['g'] = result['pars'][2:2+2].copy()
            result['g_cov'] = result['pars_cov'][2:2+2, 2:2+2].copy()
            self._set_flux(result)
            self._set_T(result)
            stat_dict=self.get_fit_stats(result['pars'])
            result.update(stat_dict)

        self._result=result

    def _get_bounds(self):
        """
        get bounds on parameters
        """
        bounds=None
        if self.prior is not None:
            if hasattr(self.prior,'bounds'):
                bounds=self.prior.bounds
        return bounds

    run_max=go
    run_lm=go

    def _set_T(self, res):
        res['T'] = res['pars'][4]
        res['T_err'] = sqrt(res['pars_cov'][4,4])

    def _set_flux(self, res):
        if self.nband==1:
            res['flux'] = res['pars'][5]
            res['flux_err'] = sqrt(res['pars_cov'][5,5])
        else:
            res['flux'] = res['pars'][5:]
            res['flux_cov'] = res['pars_cov'][5:, 5:]
            res['flux_err'] = sqrt(diag(res['flux_cov']))

    def _setup_data(self, guess):
        """
        try very hard to initialize the mixtures
        """

        if hasattr(self,'_result'):
            del self._result

        self.flags=0

        npars=guess.size
        mess="guess has npars=%d, expected %d" % (npars,self.npars)
        assert (npars==self.npars),mess

        try:
            # this can raise GMixRangeError
            self._init_gmix_all(guess)
        except ZeroDivisionError:
            raise GMixRangeError("got zero division")
        
    def get_band_pars(self, pars_in, band):
        """
        Get linear pars for the specified band
        """

        pars=self._band_pars

        pars[0:5] = pars_in[0:5]
        pars[5] = pars_in[5+band]

        return pars

    def get_T_s2n(self):
        """
        Get the s/n of T, dealing properly
        with logarithmic variables
        """
        res=self.get_result()
        T=res['pars'][4]
        Terr=res['pars_err'][4]

        if T==0.0 or Terr==0.0:
            T_s2n=0.0
        else:
            T_s2n = T/Terr

        return T_s2n

    def _make_lists(self):
        """
        lists of references.
        """
        pixels_list      = []
        gmix_data_list   = []

        for band in range(self.nband):

            obs_list=self.obs[band]
            gmix_list=self._gmix_all[band]

            for obs,gm in zip(obs_list, gmix_list):

                gmdata=gm.get_data()

                pixels_list.append(obs._pixels)
                gmix_data_list.append(gmdata)


        self._pixels_list=pixels_list
        self._gmix_data_list=gmix_data_list

    def _calc_fdiff(self, pars):
        """
        vector with (model-data)/error.

        The npars elements contain -ln(prior)
        """

        # we cannot keep sending existing array into leastsq, don't know why
        fdiff=zeros(self.fdiff_size)

        try:

            # all norms are set after fill
            self._fill_gmix_all(pars)

            start=self._fill_priors(pars, fdiff)

            for pixels,gm in zip(self._pixels_list,self._gmix_data_list):
                fill_fdiff(
                    gm,
                    pixels,
                    fdiff,
                    start,
                )

                start += pixels.size

        except GMixRangeError as err:
            fdiff[:] = LOWVAL

        return fdiff

    def _fill_priors(self, pars, fdiff):
        """
        Fill priors at the beginning of the array.

        ret the position after last par

        We require all the lnprobs are < 0, equivalent to
        the peak probability always being 1.0

        I have verified all our priors have this property.
        """

        if self.prior is None:
            nprior=0
        else:
            nprior=self.prior.fill_fdiff(pars, fdiff)

        return nprior


class LMCoellip(LMSimple):
    """
    class to run the LM leastsq code for the coelliptical model

    TODO make special set_flux and set_T methods
    """
    def __init__(self, obs, ngauss, **keys):
        self._ngauss=ngauss
        super(LMCoellip,self).__init__(obs, 'coellip', **keys)

        if self.nband != 1:
            raise ValueError("MaxCoellip only supports one band")

        #                 c1 + c2 + g + Ts     + Fs
        self.n_prior_pars=1  + 1  + 1 + ngauss + ngauss

        self.fdiff_size=self.totpix + self.n_prior_pars

        # over-write the band pars created by MaxSimple
        self._band_pars=zeros(self.npars)

    def _set_npars(self):
        """
        single band, npars determined from ngauss
        """
        self.npars=4 + 2*self._ngauss

    def get_band_pars(self, pars_in, band):
        """
        Get linear pars for the specified band
        """

        pars=self._band_pars

        pars[:] = pars_in[:]
        return pars


class LMComposite(LMSimple):
    """
    exp+dev model with pre-determined fracdev and ratio Tdev/Texp
    """
    def __init__(self, obs, fracdev, TdByTe, **keys):
        super(LMComposite,self).__init__(obs, 'cm', **keys)

        self.fracdev=fracdev
        self.TdByTe=TdByTe

    def get_gmix(self, band=0):
        """
        Get a gaussian mixture at the "best" parameter set, which
        definition depends on the sub-class
        """
        res=self.get_result()
        pars=self.get_band_pars(res['pars'], band)
        return gmix.GMixCM(self.fracdev,
                           self.TdByTe,
                           pars)


    def _make_model(self, band_pars):
        gm0=gmix.GMixCM(self.fracdev, self.TdByTe, band_pars)
        return gm0

class LMBD(LMSimple):
    """
    exp+dev model with Td/Te and fracdev free
    """
    def __init__(self, obs, **keys):
        super(LMBD,self).__init__(obs, 'bd', **keys)

        self._band_pars=zeros(8)

    def _set_n_prior_pars(self):
        # center1 + center2 + shape + T + TdByTe + fracdev + fluxes
        if self.prior is None:
            self.n_prior_pars=0
        else:
            self.n_prior_pars=1 + 1 + 1 + 1 + 1 + 1 + self.nband

    def get_gmix(self, band=0):
        """
        Get a gaussian mixture at the "best" parameter set, which
        definition depends on the sub-class
        """
        res=self.get_result()
        pars=self.get_band_pars(res['pars'], band)
        return self._make_model(pars)

    def get_band_pars(self, pars_in, band):
        """
        Get linear pars for the specified band
        """

        pars=self._band_pars

        pars[0:7] = pars_in[0:7]
        pars[7] = pars_in[7+band]

        return pars


    def _make_model(self, band_pars):
        """
        incoming parameters are
            [c1,c2,g1,g2,T,TdByTe,fracdev,F]
        """
        return gmix.GMixModel(band_pars,'bd')

    def _set_flux(self, res):
        if self.nband==1:
            res['flux'] = res['pars'][7]
            res['flux_err'] = sqrt(res['pars_cov'][7,7])
        else:
            res['flux'] = res['pars'][7:]
            res['flux_cov'] = res['pars_cov'][7:, 7:]
            res['flux_err'] = sqrt(diag(res['flux_cov']))


class LMBDF(LMSimple):
    """
    exp+dev model with Tdev/Texp=1 but fracdev free
    """
    def __init__(self, obs, **keys):
        super(LMBDF,self).__init__(obs, 'bdf', **keys)

        self._band_pars=zeros(7)

    def _set_n_prior_pars(self):
        # center1 + center2 + shape + T + fracdev + fluxes
        if self.prior is None:
            self.n_prior_pars=0
        else:
            self.n_prior_pars=1 + 1 + 1 + 1 + 1 + self.nband

    def get_gmix(self, band=0):
        """
        Get a gaussian mixture at the "best" parameter set, which
        definition depends on the sub-class
        """
        res=self.get_result()
        pars=self.get_band_pars(res['pars'], band)
        return self._make_model(pars)

    def get_band_pars(self, pars_in, band):
        """
        Get linear pars for the specified band
        """

        pars=self._band_pars

        pars[0:6] = pars_in[0:6]
        pars[6] = pars_in[6+band]

        return pars


    def _make_model(self, band_pars):
        """
        incoming parameters are
            [c1,c2,g1,g2,T,fracdev,F]
        """
        return gmix.GMixBDF(band_pars)

    def _set_flux(self, res):
        if self.nband==1:
            res['flux'] = res['pars'][6]
            res['flux_err'] = sqrt(res['pars_cov'][6,6])
        else:
            res['flux'] = res['pars'][6:]
            res['flux_cov'] = res['pars_cov'][6:, 6:]
            res['flux_err'] = sqrt(diag(res['flux_cov']))


NOTFINITE_BIT=11
def run_leastsq(func, guess, n_prior_pars, **keys):
    """
    run leastsq from scipy.optimize.  Deal with certain
    types of errors

    TODO make this do all the checking and fill in cov etc.  return
    a dict

    parameters
    ----------
    func:
        the function to minimize
    guess:
        guess at pars
    n_prior_pars:
        number of slots in fdiff for priors

    some useful keywords
    maxfev:
        maximum number of function evaluations. e.g. 1000
    epsfcn:
        Step for jacobian estimation (derivatives). 1.0e-6
    ftol:
        Relative error desired in sum of squares, 1.0e06
    xtol:
        Relative error desired in solution. 1.0e-6
    """
    #from scipy.optimize import leastsq
    from .leastsqbound import leastsqbound as leastsq

    npars=guess.size
    k_space=keys.pop('k_space',False)

    res={}
    try:
        lm_tup = leastsq(func, guess, full_output=1, **keys)

        pars, pcov0, infodict, errmsg, ier = lm_tup

        if ier == 0:
            # wrong args, this is a bug
            raise ValueError(errmsg)

        flags = 0
        if ier > 4:
            flags = 2**(ier-5)
            pars,pcov,perr=_get_def_stuff(npars)
            print('    ',errmsg)

        elif pcov0 is None:
            # why on earth is this not in the flags?
            flags += LM_SINGULAR_MATRIX
            errmsg = "singular covariance"
            print('    ',errmsg)
            print_pars(pars,front='    pars at singular:')
            junk,pcov,perr=_get_def_stuff(npars)
        else:
            # Scale the covariance matrix returned from leastsq; this will
            # recover the covariance of the parameters in the right units.
            fdiff=func(pars)

            # npars: to remove priors

            if k_space:
                dof = (fdiff.size - n_prior_pars)//2 - npars
            else:
                dof = fdiff.size - n_prior_pars - npars

            if dof==0:
                junk,pcov,perr=_get_def_stuff(npars)
                flags |= ZERO_DOF
            else:
                s_sq = (fdiff[n_prior_pars:]**2).sum()/dof
                pcov = pcov0 * s_sq

                cflags = _test_cov(pcov)
                if cflags != 0:
                    flags += cflags
                    errmsg = "bad covariance matrix"
                    print('    ',errmsg)
                    junk1,junk2,perr=_get_def_stuff(npars)
                else:
                    # only if we reach here did everything go well
                    perr=sqrt( diag(pcov) )

        res['flags']=flags
        res['nfev'] = infodict['nfev']
        res['ier'] = ier
        res['errmsg'] = errmsg

        res['pars'] = pars
        res['pars_err']=perr
        res['pars_cov0'] = pcov0
        res['pars_cov']=pcov

    except ValueError as e:
        serr=str(e)
        if 'NaNs' in serr or 'infs' in serr:
            pars,pcov,perr=_get_def_stuff(npars)

            res['pars']=pars
            res['pars_cov0']=pcov
            res['pars_cov']=pcov
            res['nfev']=-1
            res['flags']=LM_FUNC_NOTFINITE
            res['errmsg']="not finite"
            print('    not finite')
        else:
            raise e

    except ZeroDivisionError:
        pars,pcov,perr=_get_def_stuff(npars)

        res['pars']=pars
        res['pars_cov0']=pcov
        res['pars_cov']=pcov
        res['nfev']=-1

        res['flags']=DIV_ZERO
        res['errmsg']="zero division"
        print('    zero division')

    return res

def _get_def_stuff(npars):
    pars=zeros(npars) + PDEF
    cov=zeros( (npars,npars) ) + CDEF
    err=zeros(npars) + CDEF
    return pars,cov,err

def _test_cov(pcov):
    flags=0
    try:
        e,v = numpy.linalg.eig(pcov)
        weig,=numpy.where(e < 0)
        if weig.size > 0:
            flags += LM_NEG_COV_EIG

        wneg,=numpy.where(numpy.diag(pcov) < 0)
        if wneg.size > 0:
            flags += LM_NEG_COV_DIAG

    except numpy.linalg.linalg.LinAlgError:
        flags |= EIG_NOTFINITE

    return flags


class MCMCBase(FitterBase):
    """
    A base class for MCMC runs using emcee.

    Extra user-facing methods are
    run_mcmc(), calc_result(), get_trials(), get_sampler(), make_plots()
    """
    def __init__(self, obs, model, **keys):
        super(MCMCBase,self).__init__(obs, model, **keys)

        # this should be a numpy.random.RandomState object, unlike emcee which
        # through the random_state parameter takes the tuple state
        self.random_state = keys.get('random_state',None)

        # emcee specific
        self.nwalkers=keys['nwalkers']
        self.mca_a=keys.get('mca_a',2.0)

    def get_trials(self):
        """
        Get the set of trials
        """

        if not hasattr(self,'_trials'):
            raise RuntimeError("you need to run the mcmc chain first")

        return self._trials

    def get_lnprobs(self):
        """
        Get the set of ln(prob) values
        """

        if not hasattr(self,'_lnprobs'):
            raise RuntimeError("you need to run the mcmc chain first")

        return self._lnprobs

    def get_best_pars(self):
        """
        get the parameters with the highest probability
        """
        if not hasattr(self,'_lnprobs'):
            raise RuntimeError("you need to run the mcmc chain first")

        return self._best_pars.copy()

    def get_best_lnprob(self):
        """
        get the highest probability
        """
        if not hasattr(self,'_lnprobs'):
            raise RuntimeError("you need to run the mcmc chain first")

        return self._best_lnprob


    def get_sampler(self):
        """
        get the emcee sampler
        """
        return self.sampler

    def get_arate(self):
        """
        get the acceptance rate
        """
        return self._arate

    def get_tau(self):
        """
        2*tau/nstep
        """
        return self._tau

    def run_mcmc(self, pos0, nstep, thin=1, **kw):
        """
        run steps, starting at the input position(s)

        input and output pos are in linear space

        keywords to run_mcmc/sample are passed along, such as thin
        """

        pos0=array(pos0, dtype='f8')

        if not hasattr(self,'sampler'):
            self._setup_sampler_and_data(pos0)

        sampler=self.sampler
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(pos0, nstep, thin=thin, **kw)

        trials  = sampler.flatchain
        lnprobs = sampler.lnprobability.reshape(self.nwalkers*nstep//thin)

        self._trials=trials
        self._lnprobs=lnprobs

        w=lnprobs.argmax()
        bp=lnprobs[w]
        if self._best_lnprob is None or bp > self._best_lnprob:
            self._best_lnprob=bp
            self._best_pars=trials[w,:]

        arates = sampler.acceptance_fraction
        self._arate = arates.mean()
        self._set_tau()

        self._last_pos=pos
        return pos
    go=run_mcmc

    def get_last_pos(self):
        return self._last_pos

    def get_weights(self):
        """
        default weights are none
        """
        return None

    def get_stats(self, sigma_clip=False, weights=None, **kw):
        """
        get mean and covariance.

        parameters
        ----------
        weights: array
            Extra weights to apply.
        """
        this_weights = self.get_weights()

        if this_weights is not None and weights is not None:
            weights = this_weights * weights
        elif this_weights is not None:
            weights=this_weights
        else:
            # input weights are used, None or no
            pass

        trials=self.get_trials()

        pars,pars_cov = stats.calc_mcmc_stats(
            trials,
            sigma_clip=sigma_clip,
            weights=weights,
            **kw
        )

        return pars, pars_cov

    def calc_result(self, sigma_clip=False, weights=None, **kw):
        """
        Calculate the mcmc stats and the "best fit" stats
        """

        pars,pars_cov = self.get_stats(
            sigma_clip=sigma_clip,
            weights=weights,
            **kw
        )
        pars_err=sqrt(diag(pars_cov))
        res={'model':self.model_name,
             'flags':self.flags,
             'pars':pars,
             'pars_cov':pars_cov,
             'pars_err':pars_err,
             'tau':self._tau,
             'arate':self._arate}

        # note get_fits_stats expects pars in log space
        fit_stats = self.get_fit_stats(pars)
        res.update(fit_stats)

        self._result=res


    def _setup_sampler_and_data(self, pos):
        """
        try very hard to initialize the mixtures

        we work in T,F as log(1+x) so watch for low values
        """

        self.flags=0
        self._tau=0.0

        npars=pos.shape[1]
        mess="pos has npars=%d, expected %d" % (npars,self.npars)
        assert (npars==self.npars),mess

        self.sampler = self._make_sampler()
        self._best_lnprob=None

        ok=False
        gerror=None
        for i in range(self.nwalkers):
            try:
                self._init_gmix_all(pos[i,:])
                ok=True
                break
            except GMixRangeError as gerror:
                continue
            except ZeroDivisionError:
                continue

        if not ok:
            print('failed init gmix from input guess: %s' % str(gerror))
            raise gerror

    def _set_tau(self):
        """
        auto-correlation for emcee
        """
        self._tau = 1.0e9

    def _make_sampler(self):
        """
        Instantiate the sampler
        """
        import emcee
        sampler = emcee.EnsembleSampler(self.nwalkers,
                                        self.npars,
                                        self.calc_lnprob,
                                        a=self.mca_a)

        if self.random_state is not None:

            # this is a property, runs set_state internally. sadly this will
            # fail silently which is the stupidest thing I have ever seen in my
            # entire life.  If I want to set the state it is important to me!

            #print('            replacing random state')
            #sampler.random_state=self.random_state.get_state()

            # OK, we will just hope that _random doesn't change names in the
            # future.  but at least we get control back
            sampler._random = self.random_state

        return sampler


    def make_plots(self,
                   show=False,
                   prompt=False,
                   do_residual=False,
                   do_triangle=False,
                   width=1200,
                   height=1200,
                   separate=False,
                   title=None,
                   weights=None,
                   **keys):
        """
        Plot the mcmc chain and some residual plots
        """
        import mcmc
        import biggles

        biggles.configure('screen','width', width)
        biggles.configure('screen','height', height)

        names=self.get_par_names()

        if separate:
            # returns a tuple burn_plt, hist_plt
            plotfunc =mcmc.plot_results_separate
        else:
            plotfunc =mcmc.plot_results

        trials=self.get_trials()
        pdict={}
        pdict['trials']=plotfunc(trials,
                                 names=names,
                                 title=title,
                                 show=show,
                                 **keys)


        if weights is not None:
            pdict['wtrials']=plotfunc(trials,
                                      weights=weights,
                                      names=names,
                                      title='%s weighted' % title,
                                      show=show)

        if do_residual:
            pdict['resid']=self.plot_residuals(title=title,show=show,
                                               width=width,
                                               height=height,
                                               **keys)

        if do_triangle:
            try:
                # we will crash on a batch job if we don't do this.
                # also if pyplot has already been imported, it will
                # crash (god I hate matplotlib)
                import matplotlib as mpl
                mpl.use('Agg')
                import triangle
                figure = triangle.corner(trials,
                                         labels=names,
                                         quantiles=[0.16, 0.5, 0.84],
                                         show_titles=True,
                                         title_args={"fontsize": 12},
                                         bins=25)
                pdict['triangle'] = figure
            except:
                print("could not do triangle")

        if show and prompt:
            key=raw_input('hit a key: ')
            if key=='q':
                stop

        return pdict


    def get_par_names(self):
        raise RuntimeError("over-ride me")


class MCMCSimple(MCMCBase):
    """
    Add additional features to the base class to support simple models
    """
    def __init__(self, obs, model,  **keys):
        super(MCMCSimple,self).__init__(obs, model, **keys)

        # where g1,g2 are located in a pars array
        self.g1i = 2
        self.g2i = 3

        self._band_pars=zeros(6)

    def calc_result(self, **kw):
        """
        Some extra stats for simple models
        """

        super(MCMCSimple,self).calc_result(**kw)

        g1i=self.g1i
        g2i=self.g2i

        self._result['g'] = self._result['pars'][g1i:g1i+2].copy()
        self._result['g_cov'] = \
            self._result['pars_cov'][g1i:g1i+2, g2i:g2i+2].copy()

    def get_band_pars(self, pars_in, band):
        """
        Get linear pars for the specified band
        """

        pars=self._band_pars

        pars[:] = pars_in[:]
        return pars

    def get_par_names(self, dolog=False):
        names=['cen1','cen2', 'g1','g2', 'T']
        if self.nband == 1:
            names += ['F']
        else:
            for band in range(self.nband):
                names += ['F_%s' % band]

        return names



class MH(object):
    """
    Run a Monte Carlo Markov Chain (MCMC) using metropolis hastings.

    parameters
    ----------
    lnprob_func: function or method
        A function to calculate the log proability given the input
        parameters.  Can be a method of a class.
            ln_prob = lnprob_func(pars)

    stepper: function or method
        A function to take a step given the input parameters.
        Can be a method of a class.
            newpars = stepper(pars)

    seed: floating point, optional
        An optional seed for the random number generator.
    random_state: optional
        A random number generator with method .uniform()
        e.g. numpy.random.RandomState.  Takes precedence over
        seed

    examples
    ---------
    m=mcmc.MH(lnprob_func, stepper, seed=34231)
    m.run(pars_start, nstep)

    means, cov = m.get_stats()

    trials = m.get_trials()
    loglike = m.get_loglike()
    arate = m.get_acceptance_rate()

    """
    def __init__(self, lnprob_func, stepper,
                 seed=None, random_state=None):
        self._lnprob_func=lnprob_func
        self._stepper=stepper

        self.set_random_state(seed=seed, state=random_state)

    def get_trials(self):
        """
        Get the trials array
        """
        return self._trials

    def get_loglike(self):
        """
        Get the log like array
        """
        return self._loglike
    get_lnprob=get_loglike

    def get_acceptance_rate(self):
        """
        Get the acceptance rate
        """
        return self._arate
    get_arate=get_acceptance_rate

    def get_accepted(self):
        """
        Get the accepted array
        """
        return self._accepted


    def get_stats(self, sigma_clip=False, weights=None, **kw):
        """
        get mean and covariance.

        parameters
        ----------
        weights: array
            Extra weights to apply.
        """
        from .stats import calc_mcmc_stats
        stats = calc_mcmc_stats(
            self._trials,
            sigma_clip=sigma_clip,
            weights=weights,
            **kw
        )
        return stats

    def set_random_state(self, seed=None, state=None):
        """
        set the random state

        parameters
        ----------
        seed: integer, optional
            If state= is not set, the random state is set to
            numpy.random.RandomState(seed=seed)
        state: optional
            A random number generator with method .uniform()
            e.g. numpy.random.RandomState.  Takes precedence over
            seed
        """
        if state is not None:
            self._random_state=state
        else:
            self._random_state=numpy.random.RandomState(seed=seed)

    def run_mcmc(self, pars_start, nstep):
        """
        Run the MCMC chain.  Append new steps if trials already
        exist in the chain.

        parameters
        ----------
        pars_start: sequence
            Starting point for the chain in the n-d parameter space.
        nstep: integer
            Number of steps in the chain.
        """

        self._init_data(pars_start, nstep)

        for i in range(1,nstep):
            self._step()

        self._arate=self._accepted.sum()/float(self._accepted.size)
        return self._trials[-1,:]

    def _step(self):
        """
        Take the next step in the MCMC chain.

        Calls the stepper lnprob_func methods sent during construction.  If the
        new loglike is not greater than the previous, or a uniformly generated
        random number is greater than the the ratio of new to old likelihoods,
        the new step is not used, and the new parameters are the same as the
        old.  Otherwise the new step is kept.

        This is an internal function that is called by the .run method.
        It is not intended for call by the user.
        """

        index=self._current

        oldpars=self._oldpars
        oldlike=self._oldlike

        # Take a step and evaluate the likelihood
        newpars = self._stepper(oldpars)
        newlike = self._lnprob_func(newpars)

        log_likeratio = newlike-oldlike

        randnum = self._random_state.uniform()
        log_randnum = numpy.log(randnum)

        # we allow use of -infinity as a sign we are out of bounds
        if (isfinite(newlike)
                and ( (newlike > oldlike) | (log_randnum < log_likeratio)) ):

            self._accepted[index]  = 1
            self._loglike[index]   = newlike
            self._trials[index, :] = newpars

            self._oldpars = newpars
            self._oldlike = newlike

        else:
            self._accepted[index] = 0
            self._loglike[index]  = oldlike
            self._trials[index,:] = oldpars

        self._current += 1

    def _init_data(self, pars_start, nstep):
        """
        Set the trials and accept array.
        """

        pars_start=array(pars_start,dtype='f8',copy=False)
        npars = pars_start.size

        self._trials   = numpy.zeros( (nstep, npars) )
        self._loglike  = numpy.zeros(nstep)
        self._accepted = numpy.zeros(nstep, dtype='i1')
        self._current  = 1

        self._oldpars = pars_start.copy()
        self._oldlike = self._lnprob_func(pars_start)

        self._trials[0,:] = pars_start
        self._loglike[0]  = self._oldlike
        self._accepted[0] = 1

class MHTemp(MH):
    """
    Run a Monte Carlo Markov Chain (MCMC) using metropolis hastings
    with the specified temperature.

    parameters
    ----------
    lnprob_func: function or method
        A function to calculate the log proability given the input
        parameters.  Can be a method of a class.
            ln_prob = lnprob_func(pars)
    stepper: function or method
        A function to take a step given the input parameters.
        Can be a method of a class.
            newpars = stepper(pars)
    T: float
        Temperature.

    seed: floating point, optional
        An optional seed for the random number generator.
    state: optional
        A random number generator with method .uniform()
        e.g. numpy.random.RandomState.  Takes precedence over
        seed

    examples
    ---------
    T=1.5
    m=mcmc.MHTemp(lnprob_func, stepper, T, seed=34231)
    m.run(pars_start, nstep)
    trials = m.get_trials()

    means,cov = m.get_stats()

    # the above uses the weights, so is equivalent to
    # the following

    weights = m.get_weights()

    wsum=weights.sum()
    mean0 = (weights*trials[:,0]).sum()/wsum

    fdiff0 = trials[:,0]-mean0
    var00 = (weights*fdiff0*fdiff0).sum()/wsum

    fdiff1 = trials[:,1]-mean1

    var01 = (weights*fdiff0*fdiff1).sum()/wsum

    etc. for the other parameters and covariances
    """

    def __init__(self, lnprob_func, stepper, T,
                 seed=None, random_state=None):

        super(MHTemp,self).__init__(lnprob_func, stepper,
                                    seed=seed,
                                    random_state=random_state)
        self.T=T
        self.Tinv=1.0/self.T

    def get_stats(self, weights=None):
        """
        get mean and covariance.

        parameters
        ----------
        weights: array
            Extra weights to apply.
        """
        this_weights = self.get_weights()

        if weights is not None:
            weights = this_weights * weights
        else:
            weights = this_weights

        return super(MHTemp,self).get_stats(weights=weights)

    def get_loglike_T(self):
        """
        Get the log like array ln(like)/T
        """
        return self._loglike_T

    def get_weights(self):
        """
        get weights that put the loglike back at temp=1
        """
        if not hasattr(self,'_weights'):
            self._max_loglike = self._loglike.max()
            logdiff = self._loglike-self._max_loglike
            self._weights = exp(logdiff*(1.0 - self.Tinv))
        return self._weights

    def _step(self):
        """
        Take the next step in the MCMC chain.

        Calls the stepper lnprob_func methods sent during construction.  If the
        new loglike is not greater than the previous, or a uniformly generated
        random number is greater than the the ratio of new to old likelihoods,
        the new step is not used, and the new parameters are the same as the
        old.  Otherwise the new step is kept.

        This is an internal function that is called by the .run method.
        It is not intended for call by the user.
        """

        index=self._current

        oldpars=self._oldpars
        oldlike=self._oldlike
        oldlike_T=self._oldlike_T

        # Take a step and evaluate the likelihood
        newpars = self._stepper(oldpars)
        newlike = self._lnprob_func(newpars)
        newlike_T = newlike*self.Tinv

        log_likeratio = newlike_T-oldlike_T

        randnum = self._random_state.uniform()
        log_randnum = numpy.log(randnum)

        # we allow use of -infinity as a sign we are out of bounds
        if (isfinite(newlike_T)
                and ( (newlike_T > oldlike_T) | (log_randnum < log_likeratio)) ):

            self._accepted[index]  = 1
            self._loglike[index]   = newlike
            self._loglike_T[index]   = newlike_T
            self._trials[index, :] = newpars

            self._oldpars = newpars
            self._oldlike = newlike
            self._oldlike_T = newlike_T

        else:
            self._accepted[index] = 0
            self._loglike[index]  = oldlike
            self._loglike_T[index]  = oldlike_T
            self._trials[index,:] = oldpars

        self._current += 1

    def _init_data(self, pars_start, nstep):
        """
        Set the trials and accept array.
        """
        super(MHTemp,self)._init_data(pars_start, nstep)

        oldlike_T = self._oldlike*self.Tinv

        loglike_T = self._loglike.copy()
        loglike_T[0] = oldlike_T

        self._oldlike_T=oldlike_T
        self._loglike_T = loglike_T


class MHSimple(MCMCSimple):
    def __init__(self, obs, model, step_sizes, **keys):
        """
        not inheriting init from MCMCSsimple or MCMCbase

        step sizes in linear space
        """
        FitterBase.__init__(self, obs, model, **keys)

        # where g1,g2 are located in a pars array
        self.g1i = 2
        self.g2i = 3

        self._band_pars=zeros(6)

        self.set_step_sizes(step_sizes)

        seed=keys.get('seed',None)
        state=keys.get('random_state',None)
        self.set_random_state(seed=seed, state=state)

    def set_step_sizes(self, step_sizes):
        """
        set the step sizes to the input
        """
        step_sizes=numpy.asanyarray(step_sizes, dtype='f8')
        sdim = step_sizes.shape
        if len(sdim) == 1:
            ns=step_sizes.size
            mess="step_sizes has size=%d, expected %d" % (ns,self.npars)
            assert (ns == self.npars),mess

            mess="step sizes must all be > 0"
            assert numpy.all(step_sizes > 0),mess

        elif len(sdim) == 2:
            mess=(
                'step_sizes needs to be a square '
                'matrix, has dims %dx%d.' % sdim
            )
            assert (sdim[0] == sdim[1]),mess
            ns=sdim[0]
            mess="step_sizes has size=%d, expected %d" % (ns,self.npars)
            assert (ns == self.npars),mess
            assert numpy.all(numpy.linalg.eigvals(step_sizes) > 0),\
                "step_sizes must be positive definite."
        else:
            assert len(sdim) <= 2, (
                'step_sizes cannot have dimension greater than '
                '2, has %d dims.' % len(sdim)
            )
        self._step_sizes=step_sizes
        self._ndim_step_sizes = len(sdim)

    def set_random_state(self, seed=None, state=None):
        """
        set the random state

        parameters
        ----------
        state: optional
            A random number generator with method .uniform()
            e.g. an instance of numpy.random.RandomState
        seed: integer, optional
            If state= is not set, the random state is set to
            numpy.random.RandomState(seed=seed)
        """
        if state is not None:
            self.random_state=state
        else:
            self.random_state=numpy.random.RandomState(seed=seed)

    def run_mcmc(self, pos0, nstep):
        """
        run steps, starting at the input position
        """

        pos0=array(pos0,dtype='f8',copy=False)

        if not hasattr(self,'sampler'):
            self._setup_sampler_and_data(pos0)

        sampler=self.sampler

        pos = sampler.run_mcmc(pos0, nstep)

        trials = sampler.get_trials()
        lnprobs = sampler.get_lnprob()

        self._trials=trials
        self._lnprobs=lnprobs

        w=lnprobs.argmax()
        bp=lnprobs[w]
        if self._best_lnprob is None or bp > self._best_lnprob:
            self._best_lnprob=bp
            self._best_pars=trials[w,:]

        self._arate = sampler.get_arate()
        self._set_tau()

        self._last_pos=pos
        return pos

    def take_step(self, pos):
        """
        Take gaussian steps
        """
        if self._ndim_step_sizes == 1:
            r=self.random_state.normal(size=self.npars)
            return pos+self._step_sizes*r
        else:
            return numpy.random.multivariate_normal(pos, self._step_sizes)

    def _setup_sampler_and_data(self, pos):
        """
        pos in linear space

        Try to initialize the gaussian mixtures. If failure, most
        probablly a GMixRangeError will be raised
        """

        self.flags=0

        npars=pos.size
        mess="pos has npars=%d, expected %d" % (npars,self.npars)
        assert (npars==self.npars),mess

        # initialize all the gmix objects; may raise an error
        self._init_gmix_all(pos)

        self.sampler = MH(self.calc_lnprob, self.take_step,
                          random_state=self.random_state)
        self._best_lnprob=None


    def _set_tau(self):
        """
        auto-correlation scale lenght*2 divided by the number of steps
        """

        # need to figure out how to make this work.
        # could not find good examples on line

        self._tau = 1.0e9


class MHTempSimple(MHSimple):
    """
    Run with a temperature != 1.  Use the weights when
    getting stats
    """
    def __init__(self, obs, model, step_sizes, **keys):
        super(MHTempSimple,self).__init__(obs, model, step_sizes, **keys)
        self.temp=keys.get('temp',1.0)
        print("MHTempSimple doing temperature:",self.temp)

    def get_weights(self):
        """
        Get the temperature weights
        """
        return self.sampler.get_weights()

    def _setup_sampler_and_data(self, pos):
        """
        Try to initialize the gaussian mixtures. If failure, most
        probablly a GMixRangeError will be raised
        """

        self.flags=0
        self.pos=pos

        npars=pos.size
        mess="pos has npars=%d, expected %d" % (npars,self.npars)
        assert (npars==self.npars),mess

        # initialize all the gmix objects; may raise an error
        self._init_gmix_all(pos)

        self.sampler = MHTemp(self.calc_lnprob, self.take_step, self.temp,
                              random_state=self.random_state)
        self._best_lnprob=None

_default_min_err=array([1.e-4,1.e-4,1.e-3,1.e-3,1.0e-4,1.0e-4])
_default_max_err=array([1.0,1.0,5.0,5.0,1.0,1.0])

class ISampler(object):
    """
    sampler using multivariate T distribution
    """
    def __init__(self, pars, cov, df,
                 min_err=_default_min_err,
                 max_err=_default_max_err,
                 ifactor=1.0,asinh_pars=[],verbose=True):
        """
        min_err=0.001 for s/n=1000 T=4*Tpsf
        max_err=0.5 for small T s/n ~5
        """
        self._pars_orig=array(pars)
        self._cov_orig=array(cov)
        self._ifac = ifactor
        self._asinh_pars = asinh_pars
        self._npars = self._pars_orig.size
        self._set_pars_and_cov()

        self.verbose=verbose
        self._set_minmax_err(min_err, max_err)

        self._clip_cov()

        self._df=df
        self._set_pdf()

    def _set_pars_and_cov(self):
        from math import asinh

        self._pars = self._pars_orig.copy()
        self._cov = self._cov_orig.copy()
        jac = numpy.ones(self._npars)

        for ind in self._asinh_pars:
            self._pars[ind] = asinh(self._pars_orig[ind])
            jac[ind] = 1.0/numpy.sqrt(
                1.0 + self._pars_orig[ind]*self._pars_orig[ind]
            )

        for i in range(self._npars):
            for j in range(self._npars):
                self._cov[i,j] = jac[i]*jac[j]*self._cov_orig[i,j]

        self._cov = self._cov*self._ifac*self._ifac

    def _lndetjac(self,pars_orig):
        npars = pars_orig.shape[0]
        logdetjac = zeros(npars)
        for ind in self._asinh_pars:
            logdetjac += numpy.log(
                1.0/numpy.sqrt(1.0 + pars_orig[:,ind]*pars_orig[:,ind])
            )
        return logdetjac

    def pars_to_pars_orig(self,pars):
        pars_orig = pars.copy()
        for ind in self._asinh_pars:
            pars_orig[:,ind] = numpy.sinh(pars[:,ind])
        return pars_orig

    def make_samples(self, n=None):
        """
        run sample() and set internal trials attribute
        """
        self._trials = self.sample(n)
        self._trials_orig = self.pars_to_pars_orig(self._trials)

    make_trials=make_samples

    def get_result(self):
        """
        get the result dict
        """
        return self._result

    def calc_result(self, weights=None):
        """
        Calculate the mcmc stats and the "best fit" stats
        """
        from numpy import diag

        pars,pars_cov,neff = self.get_stats(weights=weights)
        pars_err=sqrt(diag(pars_cov))

        trials=self.get_trials()
        nsample=trials.shape[0]
        efficiency = neff/nsample

        res={'flags':0,
             'pars':pars,
             'pars_cov':pars_cov,
             'pars_err':pars_err,
             'g':pars[2:2+2],
             'g_cov':pars_cov[2:2+2, 2:2+2],
             'nsample':nsample,
             'neff':neff,
             'efficiency':efficiency}

        self._result=res

    def get_stats(self, weights=None):
        """
        get expectation values and covariance for
        g from the trials
        """
        from ngmix import stats
        trials=self.get_trials()
        iweights = self.get_iweights()

        # should we modify this for extra input weights?
        # maybe not: we want to know how well we sample
        # what was input
        #neff = iweights.sum()

        if weights is not None:
            weights = weights * iweights
            neff = weights.sum()/weights.max()
        else:
            weights = iweights
            neff = weights.sum()

        pars, pars_cov = stats.calc_mcmc_stats(trials, weights=weights)

        return pars, pars_cov, neff

    def get_trials(self):
        """
        return a ref to the trials
        """
        return self._trials_orig
    get_samples=get_trials

    def get_prob(self, pars):
        """
        get probability for input points

        depends on scipy
        """
        if not hasattr(self, '_pdf'):
            self._set_pdf()

        return self._pdf.pdf(pars)

    def get_lnprob(self, pars):
        """
        get log probability for input points

        depends on scipy
        """
        if not hasattr(self, '_pdf'):
            self._set_pdf()

        return self._pdf.logpdf(pars)

    def get_iweights(self):
        """
        set self._iweights for self._trials given the
        lnprob_func.  You need to run make_trials first
        """
        return self._iweights

    def set_iweights(self, lnprob_func):
        """
        get importance sample weights for the input
        samples and lnprob function
        """

        proposed_lnprob = self.get_lnprob(self._trials)
        lndetjac = self._lndetjac(self._trials_orig)

        samples = self._trials_orig
        nsample = samples.shape[0]
        lnprob = zeros(nsample)
        for i in range(nsample):
            lnprob[i] = lnprob_func(samples[i,:])

        lnpdiff = lnprob - proposed_lnprob - lndetjac
        lnpdiff -= lnpdiff.max()
        self._iweights = exp(lnpdiff)

    def sample(self, nrand=None):
        """
        Get nrand random deviates from the distribution
        """

        if nrand is None:
            is_scalar=True
            nrand=1
        else:
            is_scalar=False

        vals = numpy.zeros( (nrand,self._npars) )

        ngood=0
        nleft=nrand
        while ngood < nrand:

            samples = self._sample_raw(nleft)

            gtot = samples[:,2]**2 + samples[:,3]**2

            w,=numpy.where(gtot < 1.0)
            if w.size > 0:
                vals[ngood:ngood+w.size, :] = samples[w,:]
                ngood += w.size
                nleft -= w.size

        if is_scalar:
            vals = vals[0,:]

        return vals

    def make_plots(self,
                   title=None,
                   separate=False,
                   width=1200,
                   height=1200,
                   show=False,
                   prompt=False,
                   **keys):
        import mcmc
        import biggles

        weights = self._iweights

        biggles.configure('screen','width', width)
        biggles.configure('screen','height', height)

        if separate:
            # returns a tuple burn_plt, hist_plt
            plotfunc =mcmc.plot_results_separate
        else:
            plotfunc =mcmc.plot_results

        trials=self.get_trials()
        pdict={}
        pdict['trials']=plotfunc(trials,
                                 title=title,
                                 show=show,
                                 **keys)


        if weights is not None:
            pdict['wtrials']=plotfunc(trials,
                                      weights=weights,
                                      title='%s weighted' % title,
                                      show=show,
                                      **keys)

        if show and prompt:
            key=raw_input('hit a key: ')
            if key=='q':
                stop

        return pdict

    def _sample_raw(self, n):
        """
        sample without truncation
        """
        return self._pdf.rvs(n)

    #def _sample_raw(self, n):
    #    """
    #    sample from the cov, no truncation
    #    """
    #    from numpy.random import multivariate_normal

    #    vals=multivariate_normal(self._pars, self._cov, n)
    #    return vals

    def _set_pdf(self):
        from statsmodels.sandbox.distributions.mv_normal import MVT
        self._pdf=MVT(self._pars, self._cov, self._df)


    #def _set_pdf(self):
    #    """
    #    don't do automatically, since depends on scipy
    #    """
    #    from scipy.stats import multivariate_normal
    #    self._pdf = multivariate_normal(mean=self._pars, cov=self._cov)

    def _clip_cov(self):
        """
        clip the steps to a desired range.  Can work with
        either diagonals or cov
        """
        from numpy import sqrt, diag

        cov=self._cov

        # correlation matrix
        dsigma = sqrt(diag(cov))
        corr = cov.copy()
        for i in range(cov.shape[0]):
            for j in range(cov.shape[1]):
                corr[i,j] /= dsigma[i]
                corr[i,j] /= dsigma[j]

        w,=numpy.where(dsigma < self._min_err)
        if w.size > 0:
            dsigma[w] = self._min_err[w]
        w,=numpy.where(dsigma > self._max_err)
        if w.size > 0:
            dsigma[w] = self._max_err[w]

        # remake the covariance matrix
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                corr[i,j] *= dsigma[i]
                corr[i,j] *= dsigma[j]

        cov = corr.copy()

        # make sure the matrix is well behavied
        if numpy.all(numpy.isfinite(cov)):
            eigvals=numpy.linalg.eigvals(cov)
            if numpy.any(eigvals <= 0):
                raise LinAlgError("bad cov")

        if self.verbose:
            print_pars(sqrt(diag(cov)), front="    using err:")

        self._cov = cov

    def _set_minmax_err(self, min_err, max_err):
        min_err=array(min_err,copy=False)
        max_err=array(max_err,copy=False)

        assert min_err.size==self._pars.size,\
            "min_err must be same size as pars"
        assert max_err.size==self._pars.size,\
            "max_err must be same size as pars"

        self._min_err=min_err
        self._max_err=max_err

# alias
GCovSamplerT=ISampler

def print_pars(pars, stream=stdout, fmt='%8.3g',front=None):
    """
    print the parameters with a uniform width
    """
    if front is not None:
        stream.write(front)
        stream.write(' ')
    if pars is None:
        stream.write('%s\n' % None)
    else:
        #fmt = ' '.join( [fmt+' ']*len(pars) )
        #stream.write(fmt % tuple(pars))
        s = format_pars(pars, fmt=fmt)
        stream.write(s)
        stream.write('\n')

def format_pars(pars, fmt='%8.3g'):
    """
    make a nice string of the pars with no line breaks
    """
    fmt = ' '.join( [fmt+' ']*len(pars) )
    return fmt % tuple(pars)
