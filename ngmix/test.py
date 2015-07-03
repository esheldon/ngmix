from __future__ import print_function
import numpy
from numpy import array, zeros, diag, exp, sqrt, where, log, log10, isfinite
from numpy.random import random as randu
from pprint import pprint

from .priors import srandu

from . import joint_prior
from .fitting import *
from .gexceptions import *

def test_mcmc_psf(model="gauss",
                  g1=0.0,
                  g2=0.0,
                  T=1.10, # about Tpix=4
                  flux=100.0,
                  noise=0.1,
                  jfac=1.0,
                  nsub_render=1,
                  nsub_fit=1):
    """
    timing tests
    """
    import pylab
    import time

    nwalkers=80
    burnin=400
    nstep=400

    print("making sim")
    sigma_pix=sqrt(T/2.)/jfac
    dim=2.0*5.0*sigma_pix
    dims=[dim]*2
    cen=[(dim-1)/2.]*2

    jacobian=Jacobian(cen[0], cen[1], jfac, 0.0, 0.0, jfac)

    pars = array( [0.0, 0.0, g1, g2, T, flux], dtype='f8' )
    gm=gmix.GMixModel(pars, model)

    im=gm.make_image(dims, jacobian=jacobian, nsub=nsub_render)

    im[:,:] += noise*numpy.random.randn(im.size).reshape(im.shape)

    wt=zeros(im.shape) + 1./noise**2

    obs=Observation(im, weight=wt, jacobian=jacobian)

    print("making guess")
    guess=zeros( (nwalkers, pars.size) )
    guess[:,0] = 0.1*srandu(nwalkers)
    guess[:,1] = 0.1*srandu(nwalkers)
    guess[:,2] = g1 + 0.1*srandu(nwalkers)
    guess[:,3] = g2 + 0.1*srandu(nwalkers)
    guess[:,4] = T*(1.0 + 0.1*srandu(nwalkers))
    guess[:,5] = flux*(1.0 + 0.1*srandu(nwalkers))

    # one run to warm up the jit compiler
    mc=MCMCSimple(obs, model, nwalkers=nwalkers, nsub=nsub_fit)
    print("burnin")
    pos=mc.run_mcmc(guess, burnin)
    print("steps")
    pos=mc.run_mcmc(pos, nstep)

    mc.calc_result()


    res=mc.get_result()

    print_pars(pars,            front='true:')
    print_pars(res['pars'],     front='pars:')
    print_pars(res['pars_err'], front='err: ')

    mc.make_plots(do_residual=True,show=True,prompt=False)

def make_test_observations(model,
                           g1_obj=0.1,
                           g2_obj=0.05,
                           T_obj=16.0,
                           counts_obj=100.0,
                           noise_obj=0.001,
                           psf_model="gauss",
                           g1_psf=0.0,
                           g2_psf=0.0,
                           T_psf=4.0,
                           counts_psf=100.0,
                           noise_psf=0.001,
                           more=True):

    from . import em

    sigma=sqrt( (T_obj + T_psf)/2. )
    dims=[2.*5.*sigma]*2
    cen=[dims[0]/2., dims[1]/2.]

    j=UnitJacobian(cen[0],cen[1])

    pars_psf = [0.0, 0.0, g1_psf, g2_psf, T_psf, counts_psf]
    gm_psf=gmix.GMixModel(pars_psf, psf_model)

    pars_obj = array([0.0, 0.0, g1_obj, g2_obj, T_obj, counts_obj])
    npars=pars_obj.size
    gm_obj0=gmix.GMixModel(pars_obj, model)

    gm=gm_obj0.convolve(gm_psf)

    im_psf=gm_psf.make_image(dims, jacobian=j, nsub=16)
    npsf=noise_psf*numpy.random.randn(im_psf.size).reshape(im_psf.shape)
    im_psf[:,:] += npsf
    wt_psf=zeros(im_psf.shape) + 1./noise_psf**2

    im_obj=gm.make_image(dims, jacobian=j, nsub=16)
    n=noise_obj*numpy.random.randn(im_obj.size).reshape(im_obj.shape)
    im_obj[:,:] += n
    wt_obj=zeros(im_obj.shape) + 1./noise_obj**2

    #
    # fitting
    #


    # psf using EM
    psf_obs = Observation(im_psf, jacobian=j)

    obs=Observation(im_obj, weight=wt_obj, jacobian=j)

    if more:
        return {'psf_obs':psf_obs,
                'obs':obs,
                'pars':pars_obj,
                'gm_obj0':gm_obj0,
                'gm_obj':gm,
                'gm_psf':gm_psf}
    else:
        return psf_obs, obs

def test_model(model,
               g1_obj=0.1,
               g2_obj=0.05,
               T=16.0,
               counts=100.0,
               g1_psf=0.0,
               g2_psf=0.0,
               T_psf=4.0,
               noise=0.001,
               nimages=1,
               nwalkers=80,
               burnin=800,
               nstep=800,
               thin=2,
               g_prior=None,
               do_triangle=False,
               bins=25,
               seed=None,
               show=False):
    """
    Test fitting the specified model.

    Send g_prior to do prior during exploration
    """
    from . import em
    from . import joint_prior
    import time

    numpy.random.seed(seed)

    #
    # simulation
    #

    # PSF pars
    counts_psf=100.0
    noise_psf=0.001

    sigma=sqrt( (T + T_psf)/2. )
    dims=[2.*5.*sigma]*2
    cen=[dims[0]/2., dims[1]/2.]
    j=UnitJacobian(cen[0],cen[1])

    pars_psf = [0.0, 0.0, g1_psf, g2_psf, T_psf, counts_psf]
    gm_psf=gmix.GMixModel(pars_psf, "gauss")

    pars_obj = array([0.0, 0.0, g1_obj, g2_obj, T, counts])
    npars=pars_obj.size
    gm_obj0=gmix.GMixModel(pars_obj, model)

    gm=gm_obj0.convolve(gm_psf)

    im_psf=gm_psf.make_image(dims, jacobian=j)
    im_psf[:,:] += noise_psf*numpy.random.randn(im_psf.size).reshape(im_psf.shape)
    wt_psf=zeros(im_psf.shape) + 1./noise_psf**2

    im_obj=gm.make_image(dims, jacobian=j)
    im_obj[:,:] += noise*numpy.random.randn(im_obj.size).reshape(im_obj.shape)
    wt_obj=zeros(im_obj.shape) + 1./noise**2

    #
    # fitting
    #


    # psf using EM
    im_psf_sky,sky=em.prep_image(im_psf)
    psf_obs = Observation(im_psf_sky, jacobian=j)
    mc_psf=em.GMixEM(psf_obs)

    emo_guess=gm_psf.copy()
    emo_guess._data['p'] = 1.0
    emo_guess._data['row'] += 0.1*srandu()
    emo_guess._data['col'] += 0.1*srandu()
    emo_guess._data['irr'] += 0.5*srandu()
    emo_guess._data['irc'] += 0.1*srandu()
    emo_guess._data['icc'] += 0.5*srandu()

    mc_psf.run_em(emo_guess, sky)
    res_psf=mc_psf.get_result()
    print('psf numiter:',res_psf['numiter'],'fdiff:',res_psf['fdiff'])

    psf_fit=mc_psf.get_gmix()

    psf_obs.set_gmix(psf_fit)

    if g_prior is None:
        prior=joint_prior.make_uniform_simple_sep([0.0,0.0],
                                                  [0.1,0.1],
                                                  [-10.0,3500.],
                                                  [-0.97,1.0e9])
    else:
        print("prior during")
        cen_prior=priors.CenPrior(0.0, 0.0, 0.1, 0.1)
        T_prior=priors.FlatPrior(-10.0, 3600.0)
        F_prior=priors.FlatPrior(-0.97, 1.0e9)

        prior=joint_prior.PriorSimpleSep(cen_prior, g_prior, T_prior, F_prior)

    #prior=None
    obs=Observation(im_obj, weight=wt_obj, jacobian=j, psf=psf_obs)
    mc_obj=MCMCSimple(obs, model, nwalkers=nwalkers, prior=prior)

    guess=zeros( (nwalkers, npars) )
    guess[:,0] = 0.1*srandu(nwalkers)
    guess[:,1] = 0.1*srandu(nwalkers)

    # intentionally bad guesses
    guess[:,2] = 0.1*srandu(nwalkers)
    guess[:,3] = 0.1*srandu(nwalkers)
    guess[:,4] = T*(1.0 + 0.1*srandu(nwalkers))
    guess[:,5] = counts*(1.0 + 0.1*srandu(nwalkers))

    t0=time.time()
    pos=mc_obj.run_mcmc(guess, burnin)
    pos=mc_obj.run_mcmc(pos, nstep, thin=thin)
    mc_obj.calc_result()
    tm=time.time()-t0

    trials=mc_obj.get_trials()
    print("T minmax:",trials[:,4].min(), trials[:,4].max())
    print("F minmax:",trials[:,5].min(), trials[:,5].max())

    res_obj=mc_obj.get_result()

    print_pars(pars_obj,            front='true pars:')
    print_pars(res_obj['pars'],     front='pars_obj: ')
    print_pars(res_obj['pars_err'], front='perr_obj: ')
    print('T: %.4g +/- %.4g' % (res_obj['pars'][4], res_obj['pars_err'][4]))
    print("s2n:",res_obj['s2n_w'],"arate:",res_obj['arate'],"tau:",res_obj['tau'],"chi2per:",res_obj['chi2per'])

    #gmfit0=mc_obj.get_gmix()
    #gmfit=gmfit0.convolve(psf_fit)

    if show:
        import images
        imfit_psf=mc_psf.make_image(counts=im_psf.sum())
        images.compare_images(im_psf, imfit_psf, label1='psf',label2='fit')

        mc_obj.make_plots(do_residual=True,show=True,prompt=False)
        #imfit_obj=gmfit.make_image(im_obj.shape, jacobian=j)
        #images.compare_images(im_obj, imfit_obj, label1=model,label2='fit')
        #mcmc.plot_results(mc_obj.get_trials())

    if do_triangle:
        import triangle
        labels=[r"$cen_1$", r"$cen_2$",
                r"$e_1$",r"$e_2$",
                r"$T$",r"$F$"]
        figure = triangle.corner(trials, 
                                 labels=labels,
                                 quantiles=[0.16, 0.5, 0.84],
                                 show_titles=True,
                                 title_args={"fontsize": 12},
                                 bins=bins,
                                 smooth=10)
        figure.show()
        figure.savefig('test.png')

    return tm

def test_model_margsky_many(Tfracs=None, T_psf=4.0, show=False, ntrial=10, skyfac=0.0, noise=0.001):
    """
    ntrial is number to average over for each Tfrac
    """
    import esutil as eu
    import biggles
    import time
    plt=biggles.FramedPlot()
    xlabel=r'$\sigma^2_{psf}/\sigma^2_{gal}$'
    plt.xlabel=xlabel
    plt.ylabel=r'$\Delta e$'

    splt=biggles.FramedPlot()
    splt.xlabel=xlabel
    splt.ylabel=r'$ellip error per measurement$'

    model='exp'
    # Tobj/Tpsf
    if Tfracs is None:
        Tfracs = numpy.array([0.5**2,0.75**2,1.0**2,1.5**2,2.0**2])
    else:
        Tfracs = numpy.array(Tfracs, dtype='f8')
    #Tfracs = numpy.array([2.0**2])
    #Tfracs = numpy.array([0.5**2])
    # Tpsf/Tobj
    Pfracs =1.0/Tfracs 
    Pfracs.sort()
    
    plt.add(biggles.Curve(Pfracs, Pfracs*0))

    e1colors=['blue','steelblue']
    e2colors=['red','orange']
    e1types=['filled circle','circle']
    e2types=['filled square','square']
    e1ctypes=['solid','dotted']
    e2ctypes=['dashed','dotdashed']

    e1labels=[r'$e_1$',r'$e_1 margsky$']
    e2labels=[r'$e_2$',r'$e_2 margsky$']

    plist=[]
    for imarg,margsky in enumerate([False,True]):
        print("-"*70)
        print("marg:",margsky)

        tm0=time.time()

        g1=numpy.zeros(len(Tfracs))
        g1err=numpy.zeros(len(Tfracs))
        g1std=numpy.zeros(len(Tfracs))
        g2=numpy.zeros(len(Tfracs))
        g2err=numpy.zeros(len(Tfracs))
        g2std=numpy.zeros(len(Tfracs))

        for i,Pfrac in enumerate(Pfracs):
            print("="*70)
            Tfrac = 1.0/Pfrac
            T=Tfrac*T_psf

            print("Pfrac:",Pfrac,"Tfrac:",Tfrac)

            e1s=numpy.zeros(ntrial)
            e1s_err=numpy.zeros(ntrial)
            e2s=numpy.zeros(ntrial)
            e2s_err=numpy.zeros(ntrial)
            for trial in xrange(ntrial):
                print("trial:",trial+1)
                for retry in xrange(100):
                    res= test_model_margsky(model, T=T, T_psf=T_psf,
                                            margsky=margsky,
                                            noise=noise,
                                            skyfac=skyfac)
                    if 0.49 < res['arate'] < 0.55:
                        break
                    print("        try:",retry+1,"arate:",res['arate'])

                e1s[trial] = res['g'][0]
                e1s_err[trial] = sqrt( res['g_cov'][0,0] )
                e2s[trial] = res['g'][1]
                e2s_err[trial] = sqrt( res['g_cov'][1,1] )

            print("av e1 with sigma clip")
            mn,sig,err=eu.stat.sigma_clip(e1s, weights=1.0/e1s_err**2,get_err=True,verbose=True)
            g1[i] = mn
            g1err[i] = err
            g1std[i] = sig

            print("av e2 with sigma clip")
            mn,sig,err=eu.stat.sigma_clip(e2s, weights=1.0/e2s_err**2,get_err=True,verbose=True)
            g2[i] = mn
            g2err[i] = err
            g2std[i] = sig

            #g1[i],g2[i] = e1s.mean(), e2s.mean()
            #g1std[i],g2std[i] = e1s.std(), e2s.std()
            #g1err[i],g2err[i] = e1s.std()/sqrt(ntrial), e2s.std()/sqrt(ntrial)
        
        e1pts=biggles.Points(Pfracs, g1, color=e1colors[imarg], type=e1types[imarg])
        e2pts=biggles.Points(Pfracs, g2, color=e2colors[imarg], type=e2types[imarg])
        e1c=biggles.Curve(Pfracs, g1, color=e1colors[imarg], type=e1ctypes[imarg])
        e2c=biggles.Curve(Pfracs, g2, color=e2colors[imarg], type=e2ctypes[imarg])
        e1errp=biggles.SymmetricErrorBarsY(Pfracs, g1, g1err,color=e1colors[imarg])
        e2errp=biggles.SymmetricErrorBarsY(Pfracs, g2, g2err,color=e2colors[imarg])

        e1pts.label=e1labels[imarg]
        e2pts.label=e2labels[imarg]

        plist += [e1pts, e2pts]

        plt.add(e1pts,e1c,e1errp,e2pts,e2c,e2errp)

        e1spts=biggles.Points(Pfracs, g1std, color=e1colors[imarg], type=e1types[imarg])
        e2spts=biggles.Points(Pfracs, g2std, color=e2colors[imarg], type=e2types[imarg])
        e1sc=biggles.Curve(Pfracs, g1std, color=e1colors[imarg], type=e1ctypes[imarg])
        e2sc=biggles.Curve(Pfracs, g2std, color=e2colors[imarg], type=e2ctypes[imarg])

        splt.add(e1spts,e1sc,e2spts,e2sc)

        print("time:",time.time()-tm0)


    key=biggles.PlotKey(0.9,0.9,plist,halign='right')
    plt.add(key)
    splt.add(key)

    epsfile='margsky-test-skyfac%.4f.eps' % skyfac
    print("writing:",epsfile)
    plt.write_eps(epsfile)

    sepsfile='margsky-test-skyfac%.4f-std.eps' % skyfac
    print("writing:",sepsfile)
    splt.write_eps(sepsfile)


    if show:
        plt.show()
        splt.show()

def test_model_margsky(model,
                       T=16.0,
                       counts=100.0,
                       T_psf=4.0,
                       g1_psf=0.0,
                       g2_psf=0.0,
                       noise=0.001,
                       nwalkers=80, burnin=800, nstep=800,
                       fitter_type='mcmc',
                       skyfac=0.0,
                       margsky=True,
                       show=False):
    """
    Test fitting the specified model.
    """
    from . import em
    from . import joint_prior
    import time

    #
    # simulation
    #

    # PSF pars
    counts_psf=100.0
    noise_psf=0.01

    # object pars
    g1_obj=0.0
    g2_obj=0.0

    sigma=sqrt( (T + T_psf)/2. )
    dim=int(2*5*sigma)
    if (dim % 2) == 0:
        dim+=1
    dims=[dim]*2
    cen=array([(dims[0]-1)/2.]*2)

    j=UnitJacobian(cen[0],cen[1])

    pars_psf = [0.0, 0.0, g1_psf, g2_psf, T_psf, counts_psf]
    gm_psf=gmix.GMixModel(pars_psf, "gauss")

    pars_obj = array([0.0, 0.0, g1_obj, g2_obj, T, counts])
    npars=pars_obj.size
    gm_obj0=gmix.GMixModel(pars_obj, model)

    gm=gm_obj0.convolve(gm_psf)
    
    pcen=(dim-1)/2.
    pj=UnitJacobian(pcen,pcen)
    #pj=j
    im_psf=gm_psf.make_image(dims, jacobian=pj)
    im_psf[:,:] += noise_psf*numpy.random.randn(im_psf.size).reshape(im_psf.shape)
    wt_psf=zeros(im_psf.shape) + 1./noise_psf**2

    im_obj=gm.make_image(dims, jacobian=j)
    im_obj[:,:] += noise*numpy.random.randn(im_obj.size).reshape(im_obj.shape)
    wt_obj=zeros(im_obj.shape) + 1./noise**2

    sky=skyfac*im_obj.max()
    im_obj += sky

    #
    # fitting
    #


    # psf using EM
    im_psf_sky,sky=em.prep_image(im_psf)
    psf_obs = Observation(im_psf_sky, jacobian=pj)
    mc_psf=em.GMixEM(psf_obs)

    emo_guess=gm_psf.copy()
    emo_guess._data['p'] = 1.0
    emo_guess._data['row'] += 0.1*srandu()
    emo_guess._data['col'] += 0.1*srandu()
    emo_guess._data['irr'] += 0.5*srandu()
    emo_guess._data['irc'] += 0.1*srandu()
    emo_guess._data['icc'] += 0.5*srandu()

    mc_psf.run_em(emo_guess, sky)
    res_psf=mc_psf.get_result()
    #print('psf numiter:',res_psf['numiter'],'fdiff:',res_psf['fdiff'])

    psf_gmix=mc_psf.get_gmix()

    psf_obs.set_gmix(psf_gmix)

    prior=joint_prior.make_uniform_simple_sep([0.0,0.0], # cen
                                              [10.0,10.0], # cen width
                                              [-0.97,1.0e9], # T
                                              [-0.97,1.0e9]) # counts
    #prior=None
    obs=Observation(im_obj, weight=wt_obj, jacobian=j, psf=psf_obs)
    t0=time.time()
    if fitter_type=='mcmc':
        fitter=MCMCSimple(obs, model, nwalkers=nwalkers, prior=prior, margsky=margsky)

        guess=zeros( (nwalkers, npars) )
        guess[:,0] = 0.1*srandu(nwalkers)
        guess[:,1] = 0.1*srandu(nwalkers)

        guess[:,2] = 0.1*srandu(nwalkers)
        guess[:,3] = 0.1*srandu(nwalkers)
        guess[:,4] = T*(1.0 + 0.1*srandu(nwalkers))
        guess[:,5] = counts*(1.0 + 0.1*srandu(nwalkers))

        pos=fitter.run_mcmc(guess, burnin)
        pos=fitter.run_mcmc(pos, nstep)
        fitter.calc_result()

    else:
        guess=zeros(npars)
        guess[0] = 0.1*srandu()
        guess[1] = 0.1*srandu()

        guess[2] = 0.1*srandu()
        guess[3] = 0.1*srandu()
        guess[4] = T*(1.0 + 0.05*srandu())
        guess[5] = counts*(1.0 + 0.05*srandu())


        fitter=MaxSimple(obs, model,
                         prior=prior, margsky=margsky,
                         method=fitter_type)
        fitter.run_max(guess, maxiter=4000, maxfev=4000)

    tm=time.time()-t0

    res_obj=fitter.get_result()

    print_pars(pars_obj,            front='    true pars:')
    print_pars(res_obj['pars'],     front='    pars_obj: ')
    print_pars(res_obj['pars_err'], front='    perr_obj: ')
    print('    T: %.4g +/- %.4g' % (res_obj['pars'][4], res_obj['pars_err'][4]))
    if 'arate' in res_obj:
        print("    s2n:",res_obj['s2n_w'],"arate:",res_obj['arate'],"tau:",res_obj['tau'])
    else:
        print("    s2n:",res_obj['s2n_w'])

    if show:
        import images
        imfit_psf=mc_psf.make_image(counts=im_psf.sum())
        images.compare_images(im_psf, imfit_psf, label1='psf',label2='fit')

        if fitter_type=='mcmc':
            fitter.make_plots(do_residual=True,show=True,prompt=False)
        else:
            gm0=fitter.get_gmix()
            gm=gm0.convolve(psf_gmix)
            imfit_obj=gm.make_image(im_obj.shape, jacobian=j)
            images.compare_images(im_obj, imfit_obj, label1=model,label2='fit')

    return res_obj



def get_mh_prior(T, F):
    from . import priors, joint_prior
    cen_prior=priors.CenPrior(0.0, 0.0, 0.5, 0.5)
    g_prior = priors.make_gprior_cosmos_sersic(type='erf')
    g_prior_flat = priors.ZDisk2D(1.0)

    Twidth=0.3*T
    T_prior = priors.LogNormal(T, Twidth)

    Fwidth=0.3*T
    F_prior = priors.LogNormal(F, Fwidth)

    prior = joint_prior.PriorSimpleSep(cen_prior, g_prior, T_prior, F_prior)

    prior_gflat = joint_prior.PriorSimpleSep(cen_prior, g_prior_flat,
                                             T_prior, F_prior)

    return prior, prior_gflat

def test_model_mh(model,
                  burnin=5000,
                  nstep=5000,
                  noise_obj=0.01,
                  show=False,
                  temp=None):
    """
    Test fitting the specified model.

    Send g_prior to do some lensfit/pqr calculations
    """
    import mcmc
    from . import em



    dims=[25,25]
    cen=[dims[0]/2., dims[1]/2.]

    jacob=UnitJacobian(cen[0],cen[1])

    #
    # simulation
    #

    # PSF pars
    counts_psf=100.0
    noise_psf=0.01
    g1_psf=0.05
    g2_psf=-0.01
    T_psf=4.0

    # object pars
    counts_obj=100.0
    T_obj=16.0

    pars_psf = [0.0, 0.0, g1_psf, g2_psf, T_psf, counts_psf]
    gm_psf=gmix.GMixModel(pars_psf, "gauss")

    prior,prior_gflat=get_mh_prior(T_obj, counts_obj)

    pars_obj = prior.sample()

    #g1_obj, g2_obj = prior.g_prior.sample2d(1)
    #g1_obj=g1_obj[0]
    #g2_obj=g2_obj[0]

    #pars_obj = array([0.0, 0.0, g1_obj, g2_obj, T_obj, counts_obj])

    npars=pars_obj.size
    gm_obj0=gmix.GMixModel(pars_obj, model)

    gm=gm_obj0.convolve(gm_psf)

    im_psf=gm_psf.make_image(dims, jacobian=jacob)
    im_psf[:,:] += noise_psf*numpy.random.randn(im_psf.size).reshape(im_psf.shape)
    wt_psf=zeros(im_psf.shape) + 1./noise_psf**2

    im_obj=gm.make_image(dims, jacobian=jacob)
    im_obj[:,:] += noise_obj*numpy.random.randn(im_obj.size).reshape(im_obj.shape)
    wt_obj=zeros(im_obj.shape) + 1./noise_obj**2

    #
    # fitting
    #


    # psf using EM
    im_psf_sky,sky=em.prep_image(im_psf)
    psf_obs = Observation(im_psf_sky, jacobian=jacob)
    mc_psf=em.GMixEM(psf_obs)

    emo_guess=gm_psf.copy()
    emo_guess._data['p'] = 1.0
    emo_guess._data['row'] += 0.01*srandu()
    emo_guess._data['col'] += 0.01*srandu()
    emo_guess._data['irr'] += 0.01*srandu()
    emo_guess._data['irc'] += 0.01*srandu()
    emo_guess._data['icc'] += 0.01*srandu()

    mc_psf.go(emo_guess, sky)
    res_psf=mc_psf.get_result()
    print('psf numiter:',res_psf['numiter'],'fdiff:',res_psf['fdiff'])

    psf_fit=mc_psf.get_gmix()
    print("psf gmix:")
    print(psf_fit)
    print()

    # first fit with LM
    psf_obs.set_gmix(psf_fit)
    obs=Observation(im_obj, jacobian=jacob, weight=wt_obj, psf=psf_obs)

    lm_pars={'maxfev': 300,
             'ftol':   1.0e-6,
             'xtol':   1.0e-6,
             'epsfcn': 1.0e-6}

    lm_fitter=LMSimple(obs, model, lm_pars=lm_pars, prior=prior)

    guess=prior.sample()
    print_pars(guess, front="lm guess:")

    lm_fitter.run_lm(guess)
    lm_res=lm_fitter.get_result()

    mh_guess=lm_res['pars'].copy()
    step_sizes = 0.5*lm_res['pars_err'].copy()

    print_pars(lm_res['pars'], front="lm result:")
    print_pars(lm_res['pars_err'], front="lm err:   ")
    print()

    print_pars(step_sizes, front="step sizes:")
    if temp is not None:
        print("doing temperature:",temp)
        step_sizes *= sqrt(temp)
        mh_fitter=MHTempSimple(obs, model, step_sizes,
                               temp=temp, prior=prior)
    else:
        mh_fitter=MHSimple(obs, model, step_sizes, prior=prior)

    pos=mh_fitter.run_mcmc(mh_guess, burnin)
    pos=mh_fitter.run_mcmc(pos, nstep)
    mh_fitter.calc_result()

    res_obj=mh_fitter.get_result()

    print_pars(pars_obj,            front='true pars:')
    print_pars(res_obj['pars'],     front='pars_obj: ')
    print_pars(res_obj['pars_err'], front='perr_obj: ')

    print('T: %.4g +/- %.4g' % (res_obj['pars'][4], res_obj['pars_err'][4]))
    print("arate:",res_obj['arate'],"s2n:",res_obj['s2n_w'],"tau:",res_obj['tau'])

    gmfit0=mh_fitter.get_gmix()
    gmfit=gmfit0.convolve(psf_fit)

    if show:
        import images
        imfit_psf=mc_psf.make_image(counts=im_psf.sum())
        images.compare_images(im_psf, imfit_psf, label1='psf',label2='fit')

        mh_fitter.make_plots(do_residual=True,show=True,prompt=False)

def test_many_model_coellip(ntrial,
                            model,
                            ngauss,
                            **keys):
    import time

    tm0=time.time()
    g1fit=zeros(ntrial)
    g2fit=zeros(ntrial)

    for i in xrange(ntrial):
        print("-"*40)
        print("%d/%d" % (i+1,ntrial))

        true_pars, fit_pars= test_model_coellip(model,
                                                ngauss,
                                                **keys)
        g1fit[i] = fit_pars[2]
        g2fit[i] = fit_pars[3]

        print(g1fit[i],g2fit[i])

    frac1_arr=g1fit/true_pars[2]-1
    frac2_arr=g2fit/true_pars[3]-1

    frac1 = frac1_arr.mean()
    frac1_err = frac1_arr.std()/sqrt(ntrial)
    frac2 = frac2_arr.mean()
    frac2_err = frac2_arr.std()/sqrt(ntrial)

    print("-"*40)
    print("%g +/- %g" % (frac1, frac1_err))
    print("%g +/- %g" % (frac2, frac2_err))

    tm=time.time()-tm0
    print("time per:",tm/ntrial)

def test_model_coellip(model, ngauss,
                       T=4.0,
                       counts=100.0, noise=0.00001,
                       nwalkers=320,
                       g1=0.1, g2=0.1,
                       burnin=800,
                       nstep=800,
                       doplots=False):
    """
    fit an n gauss coellip model to a different model

    parameters
    ----------
    model:
        the true model
    ngauss:
        number of gaussians to fit to the true model
    """
    import images
    import mcmc
    from . import em

    #
    # simulation
    #

    psf_obs, obs = make_test_observations(model,
                                           T_obj=T,
                                           counts_obj=counts,
                                           noise_obj=noise,
                                           g1_obj=g1,
                                           g2_obj=g2)


    #
    # fitting
    #

    # psf using EM
    im_psf_sky,sky=em.prep_image(psf_obs.image)
    obs_sky = Observation(im_psf_sky, jacobian=psf_obs.jacobian)
    pguess_pars=[1.0,0.1*srandu(),0.1*srandu(),
                 4.0 + 0.1*srandu(),
                 0.01*srandu(),
                 4.0 + 0.1*randu()]
    pguess=GMix(pars=pguess_pars)

    mc_psf=em.GMixEM(obs_sky)
    mc_psf.go(pguess, sky, maxiter=5000)
    res_psf=mc_psf.get_result()
    print('psf numiter:',res_psf['numiter'],'fdiff:',res_psf['fdiff'])

    psf_fit=mc_psf.get_gmix()
    psf_obs.set_gmix(psf_fit)

    obs.set_psf(psf_obs)

    g1_guess=0.0
    g2_guess=0.0
    full_guess=test_guess_coellip(ngauss,
                                  g1_guess, g2_guess,
                                  T, counts)


    mc_obj=MaxCoellip(im_obj, wt_obj, jacob,
                       psf=psf_fit,
                       nwalkers=nwalkers,
                       burnin=burnin,
                       nstep=nstep,
                       priors_are_log=priors_are_log,
                       counts_prior=counts_prior,
                       cen_prior=cen_prior,
                       T_prior=T_prior,
                       full_guess=full_guess)
    mc_obj.go()


    res=mc_obj.get_result()
    if doplots:
        mc_obj.make_plots(show=True, do_residual=True,
                          width=1100,height=750,
                          separate=True)

    res_obj=mc_obj.get_result()
    gm=mc_obj.get_gmix()

    pars=res_obj['pars']
    perr=res_obj['pars_err']

    trials=mc_obj.get_trials()

    Ttrials = trials[:,4:4+ngauss]
    Ftrials = trials[:,4+ngauss:]

    Ftot = Ftrials.sum(axis=1)
    Ttot = (Ttrials*Ftrials).sum(axis=1)/Ftot

    Fmeas = Ftot.mean()
    Ferr = Ftot.std()

    Tmeas = Ttot.mean()
    Terr = Ttot.std()

    print("true T:",T_obj,"F:",counts_obj)
    print("s2n_w:",res["s2n_w"])
    print("arate:",res['arate'])
    print('Tgmix: %g Fluxgmix: %g' % (gm.get_T(),gm.get_psum()) )
    print('Tmeas: %g +/- %g Fluxmeas: %g +/- %g' % (Tmeas,Terr,Fmeas,Ferr))
    print_pars(res_obj['pars'], front='pars_obj:')
    print_pars(res_obj['pars_err'], front='perr_obj:')

    return pars_obj, res_obj['pars']

def test_guess_coellip(ngauss,
                       g1_obj, g2_obj, T_obj, counts_obj,
                       num=1):

    if num==1:
        is_scalar=True
    else:
        is_scalar=False

    npars=gmix.get_coellip_npars(ngauss)
    full_guess=zeros( (num, npars) )
    full_guess[:,0] = 0.1*srandu(num)
    full_guess[:,1] = 0.1*srandu(num)
    full_guess[:,2] = g1_obj + 0.1*srandu(num)
    full_guess[:,3] = g2_obj + 0.1*srandu(num)

    if ngauss==3:
        for i in xrange(ngauss):
            if i==0:
                full_guess[:,4+i] = 0.1*T_obj*(1.0 + 0.01*srandu(num))
                full_guess[:,4+ngauss+i] = 0.1*counts_obj*(1.0 + 0.01*srandu(num))
            elif i==1:
                full_guess[:,4+i] = 1.0*T_obj*(1.0 + 0.01*srandu(num))
                full_guess[:,4+ngauss+i] = 0.5*counts_obj*(1.0 + 0.01*srandu(num))
            elif i==2:
                full_guess[:,4+i] = 2.0*T_obj*(1.0 + 0.01*srandu(num))
                full_guess[:,4+ngauss+i] = 0.4*counts_obj*(1.0 + 0.01*srandu(num))
    elif ngauss==4:
        # implement this
        # 0.710759     3.66662     22.9798     173.704
        # 19.6636     18.3341     31.3521     29.5486
        # nromalized
        pars0=array([0.01183116, 0.06115546,  0.3829298 ,  2.89446939,
                     0.19880675,  0.18535747, 0.31701891,  0.29881687])
        #pars0=array([1.0e-6, 0.06115546,  0.3829298 ,  2.89446939,
        #             0.19880675,  0.18535747, 0.31701891,  0.29881687])

        for i in xrange(ngauss):
            full_guess[:,4+i] = T_obj*pars0[i]*(1.0 + 0.01*srandu(num))
            full_guess[:,4+ngauss+i] = counts_obj*pars0[ngauss+i]*(1.0 + 0.01*srandu(num))

            """
            if i==0:
                full_guess[:,4+i] = 0.01*T_obj*(1.0 + 0.01*srandu(num))
                full_guess[:,4+ngauss+i] = 0.1*counts_obj*(1.0 + 0.01*srandu(num))
            elif i==1:
                full_guess[:,4+i] = 0.1*T_obj*(1.0 + 0.01*srandu(num))
                full_guess[:,4+ngauss+i] = 0.2*counts_obj*(1.0 + 0.01*srandu(num))
            elif i==2:
                full_guess[:,4+i] = 1.0*T_obj*(1.0 + 0.01*srandu(num))
                full_guess[:,4+ngauss+i] = 0.5*counts_obj*(1.0 + 0.01*srandu(num))
            elif i==3:
                full_guess[:,4+i] = 2.0*T_obj*(1.0 + 0.01*srandu(num))
                full_guess[:,4+ngauss+i] = 0.2*counts_obj*(1.0 + 0.01*srandu(num))
            """
    else:
        raise ValueError("try other ngauss")

    if is_scalar:
        full_guess=full_guess[0,:]

    return full_guess



def make_sersic_images(model, hlr, flux, n, noise, g1, g2):
    import galsim 

    psf_sigma=1.414
    pixel_scale=1.0

    gal = galsim.Sersic(n,
                        half_light_radius=hlr,
                        flux=flux)
    gal.applyShear(g1=g1, g2=g2)

    psf = galsim.Gaussian(sigma=psf_sigma, flux=1.0)
    pixel=galsim.Pixel(pixel_scale)

    gal_final = galsim.Convolve([gal, psf, pixel])
    psf_final = galsim.Convolve([psf, pixel])

    # deal with massive unannounced api changes
    try:
        image_obj = gal_final.draw(scale=pixel_scale)
        psf_obj   = psf_final.draw(scale=pixel_scale)
    except:
        image_obj = gal_final.draw(dx=pixel_scale)
        psf_obj   = psf_final.draw(dx=pixel_scale)

    image_obj.addNoise(galsim.GaussianNoise(sigma=noise))

    image = image_obj.array.astype('f8')

    psf_image = psf_obj.array.astype('f8')

    wt = image*0 + ( 1.0/noise**2 )

    print("image dims:",image.shape)
    print("image sum:",image.sum())
    return image, wt, psf_image

def profile_sersic(model, **keys):
    import cProfile
    import pstats

    cProfile.runctx('test_sersic(model, **keys)',
                    globals(),locals(),
                    'profile_stats')
    p = pstats.Stats('profile_stats')
    p.sort_stats('time').print_stats()


def test_sersic(model,
                fit_model,
                do_mcmc=False,
                n=None, # only needed if model is 'sersic'
                hlr=2.0,
                counts=100.0,
                noise=1.0e-4,
                nwalkers=20,
                g1=0.1,
                g2=0.1,
                burnin=400,
                nstep=800,
                ntry=1,
                show=False):
    """
    fit an n gauss coellip model to a different model

    parameters
    ----------
    model:
        the true model
    fitmodel:
        the model to fit
    n: optional
        if true model is sersic, send n
    """
    import images
    from . import em
    from . import gmix
    from ngmix.joint_prior import PriorSimpleSep

    if model != 'sersic':
        if model=='exp':
            n=1.0
        elif model=='dev':
            n=4.0
        else:
            raise ValueError("bad model: '%s'" % model)
    #
    # simulation
    #

    # PSF pars
    sigma_psf=sqrt(2)

    im, wt, im_psf=make_sersic_images(model, hlr, counts, n, noise, g1, g2)

    cen=(im.shape[0]-1)/2.
    jacob=UnitJacobian(cen,cen)

    psf_cen=(im_psf.shape[0]-1)/2.
    psf_jacob=UnitJacobian(psf_cen,psf_cen)

    obs=Observation(im, weight=wt, jacobian=jacob)

    #
    # fitting
    #

    # psf using EM
    im_psf_sky,sky=em.prep_image(im_psf)

    psf_obs=Observation(im_psf, jacobian=psf_jacob)
    mc_psf=em.GMixEM(psf_obs)

    psf_pars_guess=[1.0,
                    0.01*srandu(),
                    0.01*srandu(),
                    sigma_psf**2,
                    0.01*srandu(),
                    sigma_psf**2]
    emo_guess=gmix.GMix(pars=psf_pars_guess)

    mc_psf.go(emo_guess, sky, maxiter=5000)
    res_psf=mc_psf.get_result()
    print('psf numiter:',res_psf['numiter'],'fdiff:',res_psf['fdiff'])

    psf_fit=mc_psf.get_gmix()
    print("psf gmix:")
    print(psf_fit)

    psf_obs.set_gmix(psf_fit)
    obs.set_psf(psf_obs)

    # fitting with max and emcee

    # terrible guess
    T_guess=2*hlr**2

    cen_prior=priors.CenPrior(0.0, 0.0, 1.0, 1.0)
    g_prior=priors.ZDisk2D(1.0)
    counts_prior=priors.FlatPrior(0.01*counts, 100*counts)
    T_prior=priors.FlatPrior(0.01*T_guess, 100*T_guess)

    prior=PriorSimpleSep(cen_prior,
                         g_prior,
                         T_prior,
                         counts_prior)
 

    print("max fit")
    while True:
        max_guess=test_guess_simple(T_guess, counts)
        max_fitter=LMSimple(obs, fit_model, lm_pars={'maxfev':4000})
        max_fitter.run_max(max_guess)
        res=max_fitter.get_result()
        if res['flags'] == 0:
            break

    print("    nfev: %d s2n: %.1f chi2per: %.3f" % (res['nfev'],res['s2n_w'],res['chi2per']))
    print_pars(res['pars'],    front="    max pars:  ")
    print_pars(res['pars_err'],front="    max perr:  ")
    full_guess=test_guess_simple_full(res['pars'], n=nwalkers, scaling='linear')

    if do_mcmc:
        mc_obj=MCMCSimple(obs, fit_model,
                          nwalkers=nwalkers,
                          prior=prior)

        pars=mc_obj.run_mcmc(full_guess, burnin)
        pars=mc_obj.run_mcmc(pars, nstep)

        mc_obj.calc_result()

        res=mc_obj.get_result()
        gm=mc_obj.get_gmix()
        gmc=gm.convolve(psf_fit)

        if show:
            mc_obj.make_plots(show=True, prompt=False,
                              width=1900,height=1200,
                              separate=False)
            model_im=gmc.make_image(im.shape, jacobian=jacob)
            images.compare_images(im, model_im) 

        res=mc_obj.get_result()

        print('arate:',res['arate'])
        print_pars(res['pars'],     front='    mcmc pars:')
        print_pars(res['pars_err'], front='    mcmc perr:')

def test_guess_simple(T, counts, n=None):
    from numpy.random import random as randu
    from . import gmix

    if n is None:
        n=1
        is_scalar=True
    else:
        is_scalar=False
    guess=zeros( (n, 6) )
    guess[:,0] = 0.1*srandu(n)
    guess[:,1] = 0.1*srandu(n)
    guess[:,2] = 0.1*srandu(n)
    guess[:,3] = 0.1*srandu(n)

    guess[:,4] = T*(1.0 + 0.2*srandu(n))
    guess[:,5] = counts*(1.0 + 0.2*srandu(n))

    if is_scalar:
        guess=guess[0,:]
    return guess

def test_guess_simple_full(pars, n=None, scaling='linear'):
    from numpy.random import random as randu
    from . import gmix

    if n is None:
        n=1
        is_scalar=True
    else:
        is_scalar=False

    guess=zeros( (n, 6) )
    guess[:,0] = pars[0] + 0.01*srandu(n)
    guess[:,1] = pars[1] + 0.01*srandu(n)

    ngood=0
    nleft=n
    while nleft > 0:
        g1 = pars[2] + 0.1*srandu(nleft)
        g2 = pars[3] + 0.1*srandu(nleft)
        g=sqrt(g1**2 + g2**2)
        w,=where(g < 1.0)
        if w.size > 0:
            guess[ngood:ngood+w.size,2] = g1[w]
            guess[ngood:ngood+w.size,3] = g2[w]
            nleft -= w.size
            ngood += w.size

    if scaling=='linear':
        guess[:,4] = pars[4]*(1.0 + 0.2*srandu(n))
        guess[:,5] = pars[5]*(1.0 + 0.2*srandu(n))
    else:
        guess[:,4] = pars[4] + 0.1*srandu(n)
        guess[:,5] = pars[5] + 0.1*srandu(n)

    if is_scalar:
        guess=guess[0,:]
    return guess


def test_guess_sersic(nwalkers, T, counts):
    from numpy.random import random as randu
    from . import gmix

    full_guess=zeros( (nwalkers, 7) )
    full_guess[:,0] = 0.1*srandu(nwalkers)
    full_guess[:,1] = 0.1*srandu(nwalkers)
    full_guess[:,2] = 0.1*srandu(nwalkers)
    full_guess[:,3] = 0.1*srandu(nwalkers)

    full_guess[:,4] = T*(1.0 + 0.2*srandu(nwalkers))
    full_guess[:,5] = counts*(1.0 + 0.2*srandu(nwalkers))

    nmin = gmix.MIN_SERSIC_N
    nmax = gmix.MAX_SERSIC_N

    full_guess[:,6] = nmin + (nmax-nmin)*randu(nwalkers)

    return full_guess

def test_guess_sersic_from_pars(nwalkers, pars):
    from numpy.random import random as randu
    from . import gmix

    full_guess=zeros( (nwalkers, 7) )
    full_guess[:,0] = pars[0] + 0.01*srandu(nwalkers)
    full_guess[:,1] = pars[1] + 0.01*srandu(nwalkers)
    full_guess[:,2] = pars[2] + 0.01*srandu(nwalkers)
    full_guess[:,3] = pars[3] + 0.01*srandu(nwalkers)

    full_guess[:,4] = pars[4]*(1.0 + 0.01*srandu(nwalkers))
    full_guess[:,5] = pars[5]*(1.0 + 0.01*srandu(nwalkers))

    nmin = gmix.MIN_SERSIC_N
    nmax = gmix.MAX_SERSIC_N

    nleft=nwalkers
    ngood=0
    while nleft > 0:
        vals = pars[6]*(1.0 + 0.01*srandu(nleft))
        w,=numpy.where( (vals > nmin) & (vals < nmax) )
        nkeep=w.size
        if nkeep > 0:
            full_guess[ngood:ngood+nkeep,6] = vals[w]
            nleft -= w.size
            ngood += w.size

    return full_guess


def test_model_priors(model,
                      counts_sky=100.0,
                      noise_sky=0.01,
                      nimages=1,
                      jfac=0.27,
                      do_lensfit=False,
                      do_pqr=False):
    """
    testing jacobian stuff
    """
    import images
    import mcmc
    from . import em

    dims=[25,25]
    cen=[dims[0]/2., dims[1]/2.]

    jfac2=jfac**2
    j=Jacobian(cen[0],cen[1], jfac, 0.0, 0.0, jfac)

    #
    # simulation
    #

    # PSF pars
    counts_sky_psf=100.0
    counts_pix_psf=counts_sky_psf/jfac2
    g1_psf=0.05
    g2_psf=-0.01
    Tpix_psf=4.0
    Tsky_psf=Tpix_psf*jfac2

    # object pars
    g1_obj=0.1
    g2_obj=0.05
    Tpix_obj=16.0
    Tsky_obj=Tpix_obj*jfac2

    counts_sky_obj=counts_sky
    noise_sky_obj=noise_sky
    counts_pix_obj=counts_sky_obj/jfac2
    noise_pix_obj=noise_sky_obj/jfac2

    pars_psf = [0.0, 0.0, g1_psf, g2_psf, Tsky_psf, counts_sky_psf]
    gm_psf=gmix.GMixModel(pars_psf, "gauss")

    pars_obj = [0.0, 0.0, g1_obj, g2_obj, Tsky_obj, counts_sky_obj]
    gm_obj0=gmix.GMixModel(pars_obj, model)

    gm=gm_obj0.convolve(gm_psf)

    im_psf=gm_psf.make_image(dims, jacobian=j)
    im_obj=gm.make_image(dims, jacobian=j)

    im_obj[:,:] += noise_pix_obj*numpy.random.randn(im_obj.size).reshape(im_obj.shape)
    wt_obj=zeros(im_obj.shape) + 1./noise_pix_obj**2

    #
    # priors
    #

    cen_prior=priors.CenPrior(0.0, 0.0, 0.1, 0.1)
    T_prior=priors.LogNormal(Tsky_obj, 0.1*Tsky_obj)
    counts_prior=priors.LogNormal(counts_sky_obj, 0.1*counts_sky_obj)
    g_prior = priors.GPriorBA(0.3)

    #
    # fitting
    #

    # psf using EM
    im_psf_sky,sky=em.prep_image(im_psf)
    mc_psf=em.GMixEM(im_psf_sky, jacobian=j)
    emo_guess=gm_psf.copy()
    emo_guess._data['p'] = 1.0
    emo_guess._data['row'] += 0.1*srandu()
    emo_guess._data['col'] += 0.1*srandu()
    emo_guess._data['irr'] += 0.5*srandu()
    emo_guess._data['irc'] += 0.1*srandu()
    emo_guess._data['icc'] += 0.5*srandu()
    mc_psf.go(emo_guess, sky)
    res_psf=mc_psf.get_result()
    print('psf numiter:',res_psf['numiter'],'fdiff:',res_psf['fdiff'])

    psf_fit=mc_psf.get_gmix()
    imfit_psf=mc_psf.make_image(counts=im_psf.sum())
    images.compare_images(im_psf, imfit_psf, label1='psf',label2='fit')

    # obj
    jlist=[j]*nimages
    imlist_obj=[im_obj]*nimages
    wtlist_obj=[wt_obj]*nimages
    psf_fit_list=[psf_fit]*nimages

    mc_obj=MCMCSimple(imlist_obj, wtlist_obj, jlist, model,
                      psf=psf_fit_list,
                      T=Tsky_obj,
                      counts=counts_sky_obj,
                      cen_prior=cen_prior,
                      T_prior=T_prior,
                      counts_prior=counts_prior,
                      g_prior=g_prior,
                      do_lensfit=do_lensfit,
                      do_pqr=do_pqr)
    mc_obj.go()

    res_obj=mc_obj.get_result()

    pprint(res_obj)
    print_pars(res_obj['pars'], front='pars_obj:')
    print_pars(res_obj['pars_err'], front='perr_obj:')
    print('Tpix: %.4g +/- %.4g' % (res_obj['pars'][4]/jfac2, res_obj['pars_err'][4]/jfac2))
    if do_lensfit:
        print('gsens:',res_obj['g_sens'])
    if do_pqr:
        print('P:',res_obj['P'])
        print('Q:',res_obj['Q'])
        print('R:',res_obj['R'])

    gmfit0=mc_obj.get_gmix()
    gmfit=gmfit0.convolve(psf_fit)
    imfit_obj=gmfit.make_image(im_obj.shape, jacobian=j)

    images.compare_images(im_obj, imfit_obj, label1=model,label2='fit')
    mcmc.plot_results(mc_obj.get_trials())


def test_model_mb(model,
                  counts_sky=[100.0, 88., 77., 95.0], # determines nband
                  noise_sky=0.1,
                  nimages=10, # in each band
                  jfac=0.27,
                  do_lensfit=False,
                  do_pqr=False,

                  nwalkers=80,
                  burnin=400,
                  nstep=800,

                  rand_center=True,

                  show=False):
    """
    testing mb stuff
    """
    import images
    import mcmc
    from . import em
    import time

    from ngmix.joint_prior import PriorSimpleSep
 
    jfac2=jfac**2

    dims=[25,25]
    cen=array( [dims[0]/2., dims[1]/2.] )

    # object pars
    g1_obj=0.1
    g2_obj=0.05
    Tpix_obj=16.0
    Tsky_obj=Tpix_obj*jfac2

    true_pars=array([0.0,0.0,g1_obj,g2_obj,Tsky_obj]+counts_sky)

    counts_sky_psf=100.0
    counts_pix_psf=counts_sky_psf/jfac2

    nband=len(counts_sky)

    mb_obs_list=MultiBandObsList()

    tmpsf=0.0
    for band in xrange(nband):

        if rand_center:
            cen_i = cen + srandu(2)
        else:
            cen_i = cen.copy()

        # not always at same center
        jacob=Jacobian(cen_i[0],
                       cen_i[1],
                       jfac,
                       0.0,
                       0.0,
                       jfac)
        counts_sky_obj=counts_sky[band]
        counts_pix_obj=counts_sky_obj/jfac2
        noise_pix_obj=noise_sky/jfac2

        obs_list=ObsList()
        for i in xrange(nimages):
            # PSF pars
            psf_cen1=0.1*srandu()
            psf_cen2=0.1*srandu()
            g1_psf= 0.05 + 0.1*srandu()
            g2_psf=-0.01 + 0.1*srandu()
            Tpix_psf=4.0*(1.0 + 0.1*srandu())
            Tsky_psf=Tpix_psf*jfac2

            pars_psf = [psf_cen1,psf_cen2, g1_psf, g2_psf, Tsky_psf, counts_sky_psf]
            gm_psf=gmix.GMixModel(pars_psf, "gauss")

            # 0 means at jacobian row0,col0
            pars_obj = [0.0, 0.0, g1_obj, g2_obj, Tsky_obj, counts_sky_obj]
            gm_obj0=gmix.GMixModel(pars_obj, model)

            gm=gm_obj0.convolve(gm_psf)

            im_psf=gm_psf.make_image(dims, jacobian=jacob, nsub=16)
            im_obj=gm.make_image(dims, jacobian=jacob, nsub=16)

            im_obj[:,:] += noise_pix_obj*numpy.random.randn(im_obj.size).reshape(im_obj.shape)
            wt_obj=zeros(im_obj.shape) + 1./noise_pix_obj**2

            # psf using EM
            tmpsf0=time.time()

            obs_i = Observation(im_obj, weight=wt_obj, jacobian=jacob)

            im_psf_sky,sky=em.prep_image(im_psf)

            psf_obs_i = Observation(im_psf_sky, jacobian=jacob)

            mc_psf=em.GMixEM(psf_obs_i)

            emo_guess=gm_psf.copy()
            emo_guess._data['p'] = 1.0
            mc_psf.go(emo_guess, sky)
            res_psf=mc_psf.get_result()

            tmpsf+=time.time()-tmpsf0
            #print 'psf numiter:',res_psf['numiter'],'fdiff:',res_psf['fdiff']

            psf_fit=mc_psf.get_gmix()

            psf_obs_i.set_gmix(psf_fit)

            obs_i.set_psf(psf_obs_i)

            obs_list.append(obs_i)

        mb_obs_list.append(obs_list)


    tmrest=time.time()
    #
    # priors
    # not really accurate since we are not varying the input
    #

    cen_prior=priors.CenPrior(0.0, 0.0, 0.1, 0.1)

    log10_T = log10(Tsky_obj)

    T_prior=priors.FlatPrior(log10_T-2.0, log10_T+2.0)
    counts_prior=[]
    for band in xrange(nband):
        counts=counts_sky[band]
        log10_counts = log10(counts)
        cp = priors.FlatPrior(log10_counts-2.0, log10_counts+2.0)
        counts_prior.append(cp)

    g_prior = priors.GPriorBA(0.3)

    prior=PriorSimpleSep(cen_prior,
                         g_prior,
                         T_prior,
                         counts_prior)
    #
    # fitting
    #

    mc_obj=MCMCSimple(mb_obs_list,
                      model,
                      prior=prior,
                      nwalkers=nwalkers)

    print("making guess from priors")
    guess=prior.sample(nwalkers)

    print("burnin",burnin)
    pos=mc_obj.run_mcmc(guess, burnin)
    print("steps",nstep)
    pos=mc_obj.run_mcmc(pos, nstep)

    mc_obj.calc_result()

    res=mc_obj.get_result()

    tmrest = time.time()-tmrest

    tmtot=tmrest + tmpsf
    print('\ntime total:',tmtot)
    print('time psf:  ',tmpsf)
    print('time rest: ',tmrest)
    print()

    print('arate:',res['arate'])
    print_pars(true_pars, front='true:    ')
    print_pars(res['pars'], front='pars_obj:')
    print_pars(res['pars_err'], front='perr_obj:')

    if do_lensfit:
        print('gsens:',res['g_sens'])
    if do_pqr:
        print('P:',res['P'])
        print('Q:',res['Q'])
        print('R:',res['R'])
           
    if show:
        mc_obj.make_plots(show=True, do_residual=True)


def _get_test_psf_flux_pars(ngauss, cen, jfac, counts_sky):

    jfac2=jfac**2
    if ngauss==1:
        e1=0.1*srandu()
        e2=0.1*srandu()
        Tpix=4.0*(1.0 + 0.2*srandu())

        Tsky=Tpix*jfac2
        pars=array([counts_sky,
                    cen[0],
                    cen[1],
                    (Tsky/2.)*(1-e1),
                    (Tsky/2.)*e2,
                    (Tsky/2.)*(1+e1)],dtype='f8')

    elif ngauss==2:
        e1_1=0.1*srandu()
        e2_1=0.1*srandu()
        e1_2=0.1*srandu()
        e2_2=0.1*srandu()

        counts_frac1 = 0.6*(1.0 + 0.1*srandu())
        counts_frac2 = 1.0 - counts_frac1
        T1pix=4.0*(1.0 + 0.2*srandu())
        T2pix=8.0*(1.0 + 0.2*srandu())

        T1sky=T1pix*jfac2
        T2sky=T2pix*jfac2
        pars=array([counts_frac1*counts_sky,
                    cen[0],
                    cen[1],
                    (T1sky/2.)*(1-e1_1),
                    (T1sky/2.)*e2_1,
                    (T1sky/2.)*(1+e1_1),

                    counts_frac2*counts_sky,
                    cen[0],
                    cen[1],
                    (T2sky/2.)*(1-e1_2),
                    (T2sky/2.)*e2_2,
                    (T2sky/2.)*(1+e1_2)])


    else:
        raise ValueError("bad ngauss: %s" % ngauss)

    gm=gmix.GMix(pars=pars)
    return gm

def test_template_flux_errors(ngauss, ntrial, **keys):
    fluxes=numpy.zeros(ntrial)
    errors=numpy.zeros(ntrial)

    for i in xrange(ntrial):
        res=test_template_flux(ngauss, **keys)
        fluxes[i] = res['flux']
        errors[i] = res['flux_err']

    print("mean error:",errors.mean())
    print("scatter:   ",fluxes.std())


def test_template_flux(ngauss,
                       send_center_as_keyword=True, # let the template fitting code reset the centers
                       do_psf=True,
                       counts_sky=100.0,
                       noise_sky=0.01,
                       nimages=1,
                       jfac=0.27,
                       jcen_offset=None,
                       show=False):
    """

    For do_psf, the gmix are in the psf observations, otherwise in the
    observation

    If reset_centers, the cen= is sent, otherwise the gmix centers are
    set before calling
    """
    from .em import GMixMaxIterEM
    import images
    import mcmc
    from . import em

    # arcsec
    #cen_sky = array([0.8, -1.2])
    cen_sky = array([1.8, -2.1])

    dims=[40,40]
    jcen=array( [dims[0]/2., dims[1]/2.] )
    if jcen_offset is not None:
        jcen_offset = array(jcen_offset)
        jcen += jcen_offset

    jcenfac=2.0
    jfac2=jfac**2
    noise_pix=noise_sky/jfac2

    ntry=10

    tm_em=0.0

    obs_list=ObsList()
    for i in xrange(nimages):
        # gmix is in sky coords.  Note center is cen_sky not the jacobian center
        gm=_get_test_psf_flux_pars(ngauss, cen_sky, jfac, counts_sky)

        # put row0,col0 at a random place
        j=Jacobian(jcen[0]+jcenfac*srandu(),jcen[1]+jcenfac*srandu(), jfac, 0.0, 0.0, jfac)

        im0=gm.make_image(dims, jacobian=j)
        if show:
            import images
            images.view(im0,title='image %s' % (i+1))

        im = im0 + noise_pix*numpy.random.randn(im0.size).reshape(dims)

        im0_skyset,sky=em.prep_image(im0)

        tobs=Observation(im0_skyset, jacobian=j)
        mc=em.GMixEM(tobs)

        # gm is also guess
        gm_guess=gm.copy()
        gm_guess.set_psum(1.0)
        gm_guess.set_cen(0.0, 0.0)
        for k in xrange(ntry):
            try:
                mc.go(gm_guess, sky, tol=1.e-5)
                break
            except GMixMaxIterEM:
                if (k==ntry-1):
                    raise
                else:
                    res=mc.get_result()
                    print('try:',k,'fdiff:',res['fdiff'],'numiter:',res['numiter'])
                    print(mc.get_gmix())
                    gm_guess.set_cen(0.1*srandu(), 0.1*srandu())
                    gm_guess._data['irr'] = gm._data['irr']*(1.0 + 0.1*srandu(ngauss))
                    gm_guess._data['icc'] = gm._data['icc']*(1.0 + 0.1*srandu(ngauss))
        psf_fit=mc.get_gmix()

        wt=0.0*im.copy() + 1./noise_pix**2

        obs=Observation(im, weight=wt, jacobian=j)

        if do_psf:
            tobs.set_gmix(psf_fit)
            obs.set_psf(tobs)
        else:
            obs.set_gmix(psf_fit)

        obs_list.append(obs)

        res=mc.get_result()
        print(i+1,res['numiter'])


    if send_center_as_keyword:
        fitter=TemplateFluxFitter(obs_list, cen=cen_sky, do_psf=do_psf)
    else:
        fitter=TemplateFluxFitter(obs_list, do_psf=do_psf)

    fitter.go()

    res=fitter.get_result()

    print("flux(sky):",counts_sky)
    print("meas: %g +/- %g" % (res['flux'], res['flux_err']))

    return res

def _make_sheared_pars(pars, shear_g1, shear_g2):
    from .shape import Shape
    shpars=pars.copy()

    sh=Shape(shpars[2], shpars[3])
    sh.shear(shear_g1, shear_g2)

    shpars[2]=sh.g1
    shpars[3]=sh.g2

    return shpars

def _make_obs(pars, model, noise_image, jacob, weight, psf_obs, nsub):
    """
    note nsub is 1 here since we are using the fit to the observed data
    """
    raise ValueError("adapt to new style")
    gm0=gmix.GMixModel(pars, model)
    gm=gm0.convolve(psf_obs.gmix)
    im = gm.make_image(noise_image.shape, jacobian=jacob, nsub=nsub)

    im += noise_image

    obs=Observation(im, jacobian=jacob, weight=weight, psf=psf_obs)

    return obs

class RetryError(Exception):
    """
    EM algorithm hit max iter
    """
    def __init__(self, value):
         self.value = value
    def __str__(self):
        return repr(self.value)

def _do_lm_fit(obs, prior, sample_prior, model, prior_during=True):
    lm_pars={'maxfev': 300,
             'ftol':   1.0e-6,
             'xtol':   1.0e-6,
             'epsfcn': 1.0e-6}

    if prior_during:
        lm_fitter=LMSimple(obs, model, lm_pars=lm_pars, prior=prior)
    else:
        lm_fitter=LMSimple(obs, model, lm_pars=lm_pars)

    nmax=1000
    i=0
    while True:

        guess=sample_prior.sample()

        try:

            lm_fitter.run_lm(guess)
        
            res=lm_fitter.get_result()

            if res['flags']==0:
                break

        except GMixRangeError as err:
            print("caught range error: %s" % str(err))

        if i > nmax:
            raise RetryError("too many tries")
        i += 1

    return res

def test_lm_metacal(model,
                    shear=0.04,
                    T_psf=4.0,
                    T_obj=16.0,
                    noise_obj=0.01,
                    npair=100,
                    nsub_render=16,
                    dim=None,
                    prior_during=True):

    """
    notes

    nsub_render=1

        testing both prior during and not during

        the metacal is unbiased when applying the prior
        
        regular seems to be unbiased when not applying the prior

    nsub_render=16

        and rendering the metacal images without sub-pixel integration

            - during, no subpixel in metacal images
                - biased
            - during, with subpixel in metacal images
                - 1-2% biased
            - not during, with subpixel in metacal images
                - about the same
            - trying h=0.02 instead of 0.01
                - prior during does look better.  I'm actually using +/- h as
                steps, which equals the shear I'm using of 0.04, maybe that is
                key.  Or it could be even larger would be better...

                next batch looks worse though... still 1.3% biased

                - not prior during a bit more biased

    """
    from .shape import Shape
    from . import em
    import lensing

    print("nsub for rendering:",nsub_render)
    shear=Shape(shear, 0.0)
    h=0.02
    #h=shear.g1


    # PSF pars
    counts_psf=100.0
    noise_psf=0.001
    g1_psf=0.00
    g2_psf=0.00

    counts_obj=100.0

    if dim is None:
        T_tot = T_obj + T_psf
        sigma_tot=sqrt(T_tot/2.0)
        dim=int(round(2*5*sigma_tot))
    dims=[dim]*2
    print("dims:",dims)
    npix=dims[0]*dims[1]
    cen=[dims[0]/2., dims[1]/2.]
    jacob=UnitJacobian(cen[0],cen[1])
    wt_obj = zeros(dims) + 1.0/noise_obj**2


    prior,prior_gflat=get_mh_prior(T_obj, counts_obj)

    g_vals=zeros( (npair*2, 2) )
    g_err_vals=zeros(npair*2)
    gsens_vals=g_vals.copy()
    s2n_vals=zeros(npair*2)

    nretry=0
    for ii in xrange(npair):
        while True:
            try:
                if (ii % 100) == 0:
                    print("%d/%d" % (ii,npair))

                pars_obj_0 = prior.sample()
                #print(pars_obj_0)

                shape1=Shape(pars_obj_0[2], pars_obj_0[3])

                shape2=shape1.copy()
                shape2.rotate(numpy.pi/2.)

                pars_psf = [pars_obj_0[0], pars_obj_0[1], g1_psf, g2_psf,
                            T_psf, counts_psf]
                gm_psf=gmix.GMixModel(pars_psf, "gauss")
                im_psf=gm_psf.make_image(dims, jacobian=jacob, nsub=nsub_render)

                noise_im_psf=noise_psf*numpy.random.randn(npix)
                noise_im_psf = noise_im_psf.reshape(dims)
                im_psf[:,:] += noise_im_psf

                im_psf_sky,sky=em.prep_image(im_psf)
                psf_obs = Observation(im_psf_sky, jacobian=jacob)

                mc_psf=em.GMixEM(psf_obs)

                emo_guess=gm_psf.copy()
                emo_guess._data['p'] = 1.0
                emo_guess._data['row'] += 0.01*srandu()
                emo_guess._data['col'] += 0.01*srandu()
                emo_guess._data['irr'] += 0.01*srandu()
                emo_guess._data['irc'] += 0.01*srandu()
                emo_guess._data['icc'] += 0.01*srandu()

                mc_psf.go(emo_guess, sky)
                res_psf=mc_psf.get_result()
                psf_fit=mc_psf.get_gmix()
                #print('psf numiter:',res_psf['numiter'],'fdiff:',res_psf['fdiff'])

                psf_obs.set_gmix(psf_fit)

                for ipair in [1,2]:

                    noise_image = noise_obj*numpy.random.randn(npix)
                    noise_image = noise_image.reshape(dims)

                    sheared_pars = pars_obj_0.copy()
                    if ipair==1:
                        i=2*ii
                        sh = shape1.copy()
                    if ipair==2:
                        i=2*ii+1
                        sh = shape2.copy()

                    sh.shear(shear.g1, shear.g2)
                    sheared_pars[2]=sh.g1
                    sheared_pars[3]=sh.g2

                    # simulated observation, here we integrate over pixels
                    # but the obs should get psf_obs set
                    obs = _make_obs(sheared_pars, model, noise_image,
                                    jacob, wt_obj, psf_obs, nsub_render)

                    #res=_do_lm_fit(obs, prior_gflat, prior, model)
                    res=_do_lm_fit(obs, prior, prior, model, prior_during=prior_during)
                    check_g(res['g'])

                    # now metacal
                    pars_meas = res['pars'].copy()
                    pars_lo=_make_sheared_pars(pars_meas, -h, 0.0)
                    pars_hi=_make_sheared_pars(pars_meas, +h, 0.0)

                    noise_image_mc = noise_obj*numpy.random.randn(npix)
                    noise_image_mc = noise_image_mc.reshape(dims)

                    # nsub=1 here since all are observed models
                    obs_lo = _make_obs(pars_lo, model, noise_image_mc,
                                       jacob, wt_obj,
                                       psf_obs, nsub_render)
                                       #psf_obs, 1)
                    obs_hi = _make_obs(pars_hi, model, noise_image_mc,
                                       jacob, wt_obj,
                                       psf_obs, nsub_render)
                                       #psf_obs, 1)

                    #res_lo=_do_lm_fit(obs_lo, prior_gflat, prior, model)
                    res_lo=_do_lm_fit(obs_lo, prior, prior, model, prior_during=prior_during)
                    check_g(res_lo['g'])
                    #res_hi=_do_lm_fit(obs_hi, prior_gflat, prior, model)
                    res_hi=_do_lm_fit(obs_hi, prior, prior, model, prior_during=prior_during)
                    check_g(res_hi['g'])

                    pars_lo=res_lo['pars']
                    pars_hi=res_hi['pars']

                    gsens_vals[i,:] = (pars_hi[2]-pars_lo[2])/(2.*h)
                    s2n_vals[i]=res['s2n_w']

                    g_vals[i,0] = res['pars'][2]
                    g_vals[i,1] = res['pars'][3]
                    g_err_vals[i] = res['pars_err'][2]

                break

            except RetryError:
                print("retrying")
                pass
            except GMixRangeError:
                print("retrying range error")
                pass



    gsens_mean=gsens_vals.mean(axis=0)

    s2n=s2n_vals.mean()
    print('s2n:',s2n)
    print("g_sens:",gsens_mean[0])

    chunksize=int(g_vals.shape[0]/100.)
    if chunksize < 1:
        chunksize=1
    print("chunksize:",chunksize)

    shear, shear_cov = lensing.shear.shear_jackknife(g_vals,
                                                     chunksize=chunksize)
    shear_fix=shear/gsens_mean[0]
    shear_cov_fix=shear_cov/gsens_mean[0]**2

    print("%g +/- %g" % (shear[0], sqrt(shear_cov[0,0])))
    print("%g +/- %g" % (shear_fix[0], sqrt(shear_cov_fix[0,0])))
    print("nretry:",nretry)

    out={'g':g_vals,
         'g_sens':gsens_vals,
         'gsens_mean':gsens_mean,
         'shear':shear,
         'shear_cov':shear_cov,
         'shear_fix':shear_fix,
         'shear_cov_fix':shear_cov_fix,
         's2n_vals':s2n_vals,
         's2n_mean':s2n}

    return out

def test_lm_psf_simple_sub_many(num, model, g1=0.3, g2=0.0, **keys):
    used=zeros(num,dtype='i2')
    g1vals=zeros(num)
    g2vals=zeros(num)
    s2nvals=zeros(num)

    keys['verbose']=False
    for i in xrange(num):
        res=test_lm_psf_simple_sub(model, g1=g1, g2=g2, **keys)

        if 's2n_w' in res:

            pars=res['pars']
            used[i]=1
            g1vals[i]=pars[2]
            g2vals[i]=pars[3]
            s2nvals[i]=res['s2n_w']
        
    w,=where(used==1)

    g1vals=g1vals[w]
    g2vals=g2vals[w]
    s2nvals=s2nvals[w]

    g1mean=g1vals.mean()
    g2mean=g2vals.mean()
    s2nmean=s2nvals.mean()

    g1err=g1vals.std()/sqrt(num)
    g2err=g2vals.std()/sqrt(num)
    s2nerr=s2nvals.std()/sqrt(num)

    print("s2n: %g +/- %g" % (s2nmean,s2nerr))
    print("g1:  %g +/- %g" % (g1mean,g1err))
    print("g2:  %g +/- %g" % (g2mean,g2err))


def test_lm_psf_simple_sub(model,
                           nsub_render=16,
                           nsub_fit=16,
                           g1=0.0,
                           g2=0.0,
                           T=4.0,
                           flux=100.0,
                           noise=0.1,
                           verbose=True):
    """
    test levenberg marquardt fit of psf with possible sub-pixel
    integration
    """
    from numpy.random import randn
    import images

    sigma=sqrt(T/2.0)
    dim=int(round(2*5*sigma))

    dims=[dim]*2

    cen=(dim-1.)/2.

    pars=array([cen,cen,g1,g2,T,flux],dtype='f8')
    gm=gmix.GMixModel(pars, model)

    im=gm.make_image(dims, nsub=nsub_render)

    noise_im = noise*randn(dim*dim).reshape(im.shape)
    im += noise_im
    #images.view(im)

    wt=im*0 + 1.0/noise**2
    obs = Observation(im,weight=wt)

    guess = pars.copy()
    guess[0] += 0.5*srandu()
    guess[1] += 0.5*srandu()
    
    while True:
        guess[2] = g1 + 0.1*srandu()
        guess[3] = g2 + 0.1*srandu()
        g=sqrt(guess[2]**2 + guess[3]**2)
        if g < 1.0:
            break

    # note log parameters in fit!
    guess[4] += 0.02*srandu()
    guess[5] += 0.02*srandu()

    lm_pars={'maxfev': 300,
             'ftol':   1.0e-6,
             'xtol':   1.0e-6,
             'epsfcn': 1.0e-6}

    fitter=LMSimple(obs, model, nsub=nsub_fit, lm_pars=lm_pars)
    fitter.run_lm(guess)

    res=fitter.get_result()

    if verbose:
        print("flags:",res['flags'])
        print_pars(pars,            front='truth: ')
        print_pars(res['pars'],     front='fit:   ')
        print_pars(res['pars_err'], front='err:   ')
        print_pars(guess,           front='guess: ')

    return res

def test_nm_psf_coellip(g1=0.0,
                        g2=0.0,
                        T=4.0,
                        flux=100.0,
                        noise=0.01,
                        ngauss=2,
                        maxiter=4000,
                        seed=None,
                        show=False):
    """
    test nelder mead fit of turb psf with coellip 
    """
    from numpy.random import randn
    import images

    numpy.random.seed(seed)

    #ngauss=3

    sigma=sqrt(T/2.0)
    dim=int(round(2*5*sigma))

    dims=[dim]*2

    cen=(dim-1.)/2.

    pars=array([cen,cen,g1,g2,T,flux],dtype='f8')
    gm=gmix.GMixModel(pars, 'turb')

    im=gm.make_image(dims, nsub=16)

    noise_im = noise*randn(dim*dim).reshape(im.shape)
    im += noise_im
    #images.view(im)

    wt=im*0 + 1.0/noise**2
    obs = Observation(im,weight=wt)

    npars=4+ngauss*2

    Tguess=T
    Cguess=flux

    def get_guess():
        guess=zeros(npars)
        # bad guess but roughly right proportions
        if ngauss==3:
            Tfrac=array([0.5793612389470884,1.621860687127999,7.019347162356363])
            Cfrac=array([0.596510042804182,0.4034898268889178,1.303069003078001e-07])
        else:
            Tfrac=array([0.5,2.0])
            Cfrac=array([0.7,0.3])

        width=0.1
        guess[4:4+ngauss]=Tguess*Tfrac*(1.0 + width*srandu())
        guess[4+ngauss:]=Cguess*Cfrac*(1.0 + width*srandu())

        guess[0]=cen+width*srandu()
        guess[1]=cen+width*srandu()

        guess[2] = width*srandu()
        guess[3] = width*srandu()
        return guess

    guess=get_guess()
    guess_orig=guess.copy()

    fitter=MaxCoellip(obs,ngauss)

    print("running nm")
    itry=1
    tm0=time.time()
    while True:
        print("   try:",itry)
        fitter.run_max(guess, maxiter=maxiter)
        res=fitter.get_result()
        if res['flags'] == 0:
            break
        guess=get_guess()
        itry+=1
    print("time:",time.time()-tm0)

    for key in res:
        if key not in ['pars','pars_err','pars_cov','g','g_cov','x']:
            print("    %s: %s" % (key, res[key]))
    print()

    fitmix=fitter.get_gmix()
    print_pars(pars,            front='truth: ')
    print("T:",fitmix.get_T(),"count:",fitmix.get_flux())
    print_pars(res['pars'],     front='fit:   ')

    if 'pars_err' in res:
        print_pars(res['pars_err'], front='err:   ')
    else:
        print("NO ERROR PRESENT")

    print_pars(guess_orig,     front='oguess: ')

    if show:
        import images
        gm=fitter.get_gmix()
        model_im=gm.make_image(dims)
        images.compare_images(im, model_im, label1='image',label2='model')

def check_g(g):
    gtot=sqrt(g[0]**2 + g[1]**2)
    if gtot > 0.97:
        raise RetryError("bad g")

def get_g_guesses(g10, g20, width=0.01):
    while True:
        g1 = g10 + width*srandu()
        g2 = g20 + width*srandu()
        g=sqrt(g1**2 + g2**2)
        if g < 1.0:
            break

    return g1,g2

def test_nm_many(n=1000, **kw):
    import esutil as eu

    cen_offset = kw.get('cen_offset',numpy.zeros(2))

    g1vals=zeros(n)
    g2vals=zeros(n)
    g1errs=zeros(n)
    g2errs=zeros(n)
    nfevs=zeros(n,dtype='i4')
    ntrys=zeros(n,dtype='i4')

    fracprint=0.01
    np=int(n*fracprint)
    if np <= 0:
        np=1

    tm0=time.time()
    for i in xrange(n):
        if ( (i+1) % np) == 0 or i==0:
            print("%d/%d" % (i+1,n))

        #kw['cen_offset']=cen_offset + numpy.random.random(2)
        kw['cen_offset']=cen_offset + numpy.random.randn(2)
        res=test_nm('exp', **kw)
        pars, pars_err=res['pars'],res['pars_err']

        g1vals[i] = pars[2]
        g2vals[i] = pars[3]
        g1errs[i] = pars_err[2]
        g2errs[i] = pars_err[3]
        nfevs[i]=res['nfev']
        ntrys[i]=res['ntry']
    print("total time:",time.time()-tm0)

    weights=1.0/(g1errs**2 + g2errs**2)

    g1mean, g1err = eu.stat.wmom(g1vals, weights, calcerr=True)
    g2mean, g2err = eu.stat.wmom(g2vals, weights, calcerr=True)
    print("e1: %g +/- %g" % (g1mean,g1err))
    print("e2: %g +/- %g" % (g2mean,g2err))

    return {'g1':g1vals,
            'g1err':g1errs,
            'g2':g2vals,
            'g2err':g2errs,
            'nfev':nfevs,
            'ntry':ntrys}

def test_max(model, sigma=2.82, counts=100.0, noise=0.001, nimages=1,
             method='nm',
             g1=0.1,
             g2=0.05,
             sigma_fac=5.0,
             g1_psf=0.0,
             g2_psf=0.05,
             jfac=1.0,
             prior_type='flat',
             g_prior_type='flat',
             verbose=True,
             show=False,
             dims=None,
             cen_offset=None,
             aperture=None,
             do_aperture=False, # auto-calculate aperture
             maxfev=4000,
             ftol=1.e-4,
             xtol=1.e-4,
             seed=None,
             guess_quality='bad', # use 'good' to make a good guess
             do_emcee=False,
             nwalkers=80, burnin=800, nstep=800):
    """
    Fit with nelder-mead, calculating cov matrix with our code

    if do_emcee is True, compare with a mcmc fit using emcee
    """
    from . import em
    from . import joint_prior
    import time
    import images
    from .em import GMixMaxIterEM

    jfac2=jfac*jfac
    numpy.random.seed(seed)

    fmt='%12.6f'

    #
    # simulation
    #

    sigma = sigma*jfac

    # PSF pars
    counts_psf=100.0
    noise_psf=0.01
    T_psf=4.0*jfac2

    T=2.*sigma**2
    sigmatot=sqrt( (T + T_psf)/2. )

    if dims is None:
        dims=[2.*sigma_fac*sigmatot/jfac]*2

    cen_orig=array( [(dims[0]-1)/2.]*2 )

    if cen_offset is not None:
        cen = cen_orig + array( cen_offset )
    else:
        cen = cen_orig.copy()

    j=Jacobian(cen[0],cen[1], jfac, 0.0, 0.0, jfac)

    pars_psf = [0.0, 0.0, g1_psf, g2_psf, T_psf, counts_psf]
    gm_psf=gmix.GMixModel(pars_psf, "gauss")

    pars_obj = array([0.0, 0.0, g1, g2, T, counts])
    npars=pars_obj.size
    gm_obj0=gmix.GMixModel(pars_obj, model)

    gm=gm_obj0.convolve(gm_psf)

    jpsf=Jacobian(cen_orig[0], cen_orig[1], jfac, 0.0, 0.0, jfac)
    im_psf=gm_psf.make_image(dims, jacobian=jpsf, nsub=16)
    im_psf[:,:] += noise_psf*numpy.random.randn(im_psf.size).reshape(im_psf.shape)
    wt_psf=zeros(im_psf.shape) + 1./noise_psf**2

    im_obj=gm.make_image(dims, jacobian=j, nsub=16)
    im_obj[:,:] += numpy.random.normal(scale=noise, size=im_obj.shape)
    wt_obj=zeros(im_obj.shape) + 1./noise**2

    #
    # fitting
    #


    # psf fitting
    im_psf_sky,sky=em.prep_image(im_psf)
    psf_obs = Observation(im_psf_sky, jacobian=jpsf)
    mc_psf=em.GMixEM(psf_obs)

    while True:
        emo_guess=gm_psf.copy()
        emo_guess._data['p'] = 1.0
        emo_guess._data['row'] += 0.1*srandu()
        emo_guess._data['col'] += 0.1*srandu()
        emo_guess._data['irr'] += 0.5*srandu()
        emo_guess._data['irc'] += 0.1*srandu()
        emo_guess._data['icc'] += 0.5*srandu()


        try:
            mc_psf.run_em(emo_guess, sky, maxiter=2000)
            break
        except GMixMaxIterEM:
            continue

    res_psf=mc_psf.get_result()
    if verbose:
        print("dims:",dims)
        print('psf numiter:',res_psf['numiter'],'fdiff:',res_psf['fdiff'])

    psf_fit=mc_psf.get_gmix()
    
    if verbose:
        print("fit psf T:",psf_fit.get_T())

    psf_obs.set_gmix(psf_fit)

    cen_width=0.5
    if prior_type=='flat':
        pmaker=joint_prior.make_uniform_simple_sep
    else:
        pmaker=joint_prior.make_cosmos_simple_sep

    prior=pmaker([0.0,0.0], # cen
                 [cen_width]*2, #cen width
                 [-0.97,3500.], # T
                 [-0.97,1.0e9]) # counts

    if g_prior_type=='round':
        prior.g_prior=priors.GPriorBA(0.001)

    #prior=None
    obs=Observation(im_obj, weight=wt_obj, jacobian=j, psf=psf_obs)

    #
    # nm fitting
    #

    if do_aperture:
        aperture=get_edge_aperture(dims, cen)
        if verbose:
            print("Using aperture:",aperture)
    if verbose:
        print("fitting with nelder-mead")

    if method=='nm':
        max_fitter=MaxSimple(obs, model,
                            prior=prior, aperture=aperture)
    else:
        max_fitter=LMSimple(obs, model, lm_pars={'maxfev':4000},
                            prior=prior, aperture=aperture)

    guess=zeros( max_fitter.npars )
    ntry=0
    while True:
        ntry += 1
        if guess_quality=='bad':
            guess[0] = cen_width*srandu()
            guess[1] = cen_width*srandu()
            guess[2], guess[3] = get_g_guesses(0.0, 0.0, width=0.1)
            guess[4] = T*(1.0 + 0.1*srandu())
            guess[5] = counts*(1.0 + 0.1*srandu())
        else:
            guess[0] = 0.001*srandu()
            guess[1] = 0.001*srandu()
            guess[2],guess[3] = get_g_guesses(g1,g2,width=0.01)
            guess[4] = T*(1.0 + 0.01*srandu())
            guess[5] = counts*(1.0 + 0.01*srandu())

        t0=time.time()
        if method=='nm':
            max_fitter.go(guess,
                          maxfev=4000,
                          maxiter=4000,
                          xtol=xtol,
                          ftol=ftol)
        else:
            max_fitter.go(guess)

        max_res=max_fitter.get_result()
        if verbose:
            print("time for nm:", time.time()-t0)

        # we could also just check EIG_NOTFINITE but then there would
        # be no errors
        if (max_res['flags'] & 3) != 0:
            print("    did not converge, trying again with a new guess")
            print_pars(max_res['pars'],              front='    pars were:', fmt=fmt)
        elif (max_res['flags'] & EIG_NOTFINITE) != 0:
            print("    bad cov, trying again with a new guess")
            print_pars(max_res['pars'],              front='    pars were:', fmt=fmt)
        else:
            break

    max_res['ntry'] = ntry

    #
    # emcee fitting
    # 
    if do_emcee:
        if verbose:
            print("fitting with emcee")
        emcee_fitter=MCMCSimple(obs, model, nwalkers=nwalkers, prior=prior, aperture=aperture)

        guess=zeros( (nwalkers, npars) )
        guess[:,0] = 0.1*srandu(nwalkers)
        guess[:,1] = 0.1*srandu(nwalkers)

        # intentionally good guesses
        for i in xrange(nwalkers):
            guess[i,2], guess[i,3] = get_g_guesses(pars_obj[2],pars_obj[3],width=0.01)
        guess[:,4] = T*(1.0 + 0.01*srandu(nwalkers))
        guess[:,5] = counts*(1.0 + 0.01*srandu(nwalkers))

        t0=time.time()
        pos=emcee_fitter.run_mcmc(guess, burnin)
        pos=emcee_fitter.run_mcmc(pos, nstep)
        emcee_fitter.calc_result()
        if verbose:
            print("time for emcee:", time.time()-t0)

        emcee_res=emcee_fitter.get_result()

    if verbose:
        max_pars=max_res['pars']
        max_perr=max_res['pars_err']

        for key in max_res:
            if key not in ['pars','pars_err','pars_cov','g','g_cov','x']:
                print("    %s: %s" % (key, max_res[key]))

        print_pars(pars_obj,              front='true pars: ', fmt=fmt)

        if do_emcee:
            print_pars(emcee_res['pars'],     front='emcee pars:', fmt=fmt)
        print_pars(max_res['pars'],        front='max pars:  ', fmt=fmt)

        if do_emcee:
            print_pars(emcee_res['pars_err'], front='emcee err: ', fmt=fmt)
        print_pars(max_res['pars_err'],    front='max err:   ', fmt=fmt)

        Ts2n=max_pars[4]/max_perr[4]
        print("\ns2n:",max_res['s2n_w'],"Ts2n:",Ts2n)

        if do_emcee:
            print("s2n:",emcee_res['s2n_w'],"arate:",emcee_res['arate'],"tau:",emcee_res['tau'])

            if show:
                emcee_fitter.make_plots(do_residual=True,show=True,prompt=False)

        #print("\nnm cov:")
        #images.imprint(max_res['pars_cov'], fmt='%12.6g')

        if show:
            import images
            gm0=max_fitter.get_gmix()
            gm=gm0.convolve(psf_fit)
            model_im=gm.make_image(dims,jacobian=j)
            images.compare_images(im_obj, model_im, label1='image',label2='model')

    return max_res

def test_model_logpars(model, T=16.0, counts=100.0, noise=0.001, nimages=1,
                       nwalkers=80, burnin=800, nstep=800,
                       g_prior=None, show=False, **keys):
    """
    Test fitting the specified model.

    Send g_prior to do some lensfit/pqr calculations
    """
    from . import em
    from . import joint_prior
    import time

    #
    # simulation
    #

    # PSF pars
    counts_psf=100.0
    noise_psf=0.01
    g1_psf=0.05
    g2_psf=-0.01
    T_psf=4.0

    # object pars
    g1_obj=0.1
    g2_obj=0.05

    sigma=sqrt( (T + T_psf)/2. )
    dims=[2.*5.*sigma]*2
    cen=[dims[0]/2., dims[1]/2.]
    j=UnitJacobian(cen[0],cen[1])

    pars_psf = [0.0, 0.0, g1_psf, g2_psf, T_psf, counts_psf]
    gm_psf=gmix.GMixModel(pars_psf, "gauss")

    pars_obj = array([0.0, 0.0, g1_obj, g2_obj, T, counts])
    npars=pars_obj.size
    gm_obj0=gmix.GMixModel(pars_obj, model)

    gm=gm_obj0.convolve(gm_psf)

    im_psf=gm_psf.make_image(dims, jacobian=j)
    im_psf[:,:] += noise_psf*numpy.random.randn(im_psf.size).reshape(im_psf.shape)
    wt_psf=zeros(im_psf.shape) + 1./noise_psf**2

    im_obj=gm.make_image(dims, jacobian=j)
    im_obj[:,:] += noise*numpy.random.randn(im_obj.size).reshape(im_obj.shape)
    wt_obj=zeros(im_obj.shape) + 1./noise**2

    #
    # fitting
    #


    # psf using EM
    im_psf_sky,sky=em.prep_image(im_psf)
    psf_obs = Observation(im_psf_sky, jacobian=j)
    mc_psf=em.GMixEM(psf_obs)

    emo_guess=gm_psf.copy()
    emo_guess._data['p'] = 1.0
    emo_guess._data['row'] += 0.1*srandu()
    emo_guess._data['col'] += 0.1*srandu()
    emo_guess._data['irr'] += 0.5*srandu()
    emo_guess._data['irc'] += 0.1*srandu()
    emo_guess._data['icc'] += 0.5*srandu()

    mc_psf.run_em(emo_guess, sky)
    res_psf=mc_psf.get_result()
    print('psf numiter:',res_psf['numiter'],'fdiff:',res_psf['fdiff'])

    psf_fit=mc_psf.get_gmix()

    psf_obs.set_gmix(psf_fit)

    prior=joint_prior.make_erf_simple_sep([0.0,0.0],
                                          [0.1,0.1],
                                          [-5.,0.1,6.,0.1],
                                          [-0.97,0.1,1.0e9,0.25e8])
    #prior=None
    obs=Observation(im_obj, weight=wt_obj, jacobian=j, psf=psf_obs)
    mc_obj=MCMCSimple(obs, model, nwalkers=nwalkers, prior=prior,
                      use_logpars=True)

    guess=zeros( (nwalkers, npars) )
    guess[:,0] = 0.01*srandu(nwalkers)
    guess[:,1] = 0.01*srandu(nwalkers)

    # intentionally bad guesses
    guess[:,2] = 0.01*srandu(nwalkers)
    guess[:,3] = 0.01*srandu(nwalkers)
    guess[:,4] = log10( T*(1.0 + 0.01*srandu(nwalkers)) )
    guess[:,5] = counts*(1.0 + 0.01*srandu(nwalkers))

    t0=time.time()
    pos=mc_obj.run_mcmc(guess, burnin)
    pos=mc_obj.run_mcmc(pos, nstep)
    mc_obj.calc_result()
    tm=time.time()-t0

    trials=mc_obj.get_trials()
    print("T minmax:",trials[:,4].min(), trials[:,4].max())
    print("F minmax:",trials[:,5].min(), trials[:,5].max())

    res_obj=mc_obj.get_result()

    print_pars(pars_obj,            front='true pars:')
    print_pars(res_obj['pars'],     front='pars_obj: ')
    print_pars(res_obj['pars_err'], front='perr_obj: ')
    print('T: %.4g +/- %.4g' % (res_obj['pars'][4], res_obj['pars_err'][4]))
    print("s2n:",res_obj['s2n_w'],"arate:",res_obj['arate'],"tau:",res_obj['tau'])

    if show:
        import images
        mc_obj.make_plots(do_residual=True,show=True,prompt=False,**keys)

    return tm




def get_isamples_and_weights(sampler, lnprob_func):
    """
    might be bogus, fixing pars besides g

    sampler should already have trials set
    """
    samples = sampler.get_trials()
    nsample = samples.shape[0]

    proposed_lnprob = sampler.get_lnprob(samples)

    lnprob = zeros(nsample)
    for i in xrange(nsample):
        lnprob[i] = lnprob_func(samples[i,:])

    lnpdiff = lnprob - proposed_lnprob
    lnpdiff -= lnpdiff.max()
    weights = exp(lnpdiff)

    return samples, weights




def test_covsample_log(model,
                       seed=None,
                       g1_obj=0.1,
                       g2_obj=0.05,
                       T=16.0,
                       counts=100.0,
                       g1_psf=0.0,
                       g2_psf=0.0,
                       T_psf=4.0,
                       noise=0.001,
                       nwalkers=80,
                       burnin=800,
                       nstep=800,
                       nsample=None,
                       ifactor=2.0,
                       thin=2,
                       nbin=50,
                       show=False, width=1200, height=1200):
    """
    Test fitting the specified model.

    """
    from . import em
    from . import joint_prior
    import time
    from numpy.random import randn

    numpy.random.seed(seed)

    if nsample is None:
        nsample = nstep*nwalkers

    g_prior = priors.GPriorBA(0.3)
    #
    # simulation
    #

    # PSF pars
    counts_psf=100.0
    noise_psf=0.001

    sigma=sqrt( (T + T_psf)/2. )
    dims=[2.*5.*sigma]*2
    cen=[dims[0]/2., dims[1]/2.]
    j=UnitJacobian(cen[0],cen[1])

    pars_psf = [0.0, 0.0, g1_psf, g2_psf, T_psf, counts_psf]
    gm_psf=gmix.GMixModel(pars_psf, "gauss")

    pars_obj = array([0.0, 0.0, g1_obj, g2_obj, T, counts])
    npars=pars_obj.size
    gm_obj0=gmix.GMixModel(pars_obj, model)

    gm=gm_obj0.convolve(gm_psf)

    im_psf=gm_psf.make_image(dims, jacobian=j)
    im_psf[:,:] += noise_psf*numpy.random.randn(im_psf.size).reshape(im_psf.shape)
    wt_psf=zeros(im_psf.shape) + 1./noise_psf**2

    im_obj=gm.make_image(dims, jacobian=j)
    im_obj[:,:] += noise*numpy.random.randn(im_obj.size).reshape(im_obj.shape)
    wt_obj=zeros(im_obj.shape) + 1./noise**2

    #
    # fitting
    #


    # psf using EM
    im_psf_sky,sky=em.prep_image(im_psf)
    psf_obs = Observation(im_psf_sky, jacobian=j)
    mc_psf=em.GMixEM(psf_obs)

    emo_guess=gm_psf.copy()
    emo_guess._data['p'] = 1.0
    emo_guess._data['row'] += 0.1*srandu()
    emo_guess._data['col'] += 0.1*srandu()
    emo_guess._data['irr'] += 0.5*srandu()
    emo_guess._data['irc'] += 0.1*srandu()
    emo_guess._data['icc'] += 0.5*srandu()

    mc_psf.run_em(emo_guess, sky)
    res_psf=mc_psf.get_result()
    print('psf numiter:',res_psf['numiter'],'fdiff:',res_psf['fdiff'])

    psf_fit=mc_psf.get_gmix()

    psf_obs.set_gmix(psf_fit)

    cen_prior=priors.CenPrior(0.0, 0.0, 0.1, 0.1)
    #T_prior=priors.FlatPrior(-1.0, 4.0)
    T_prior=priors.TwoSidedErf(-0.5, 0.1, 4.0, 0.1)
    #F_prior=priors.FlatPrior(-0.97, 1.0e9)
    #F_prior=priors.FlatPrior(-1.0, 9.0)
    F_prior=priors.TwoSidedErf(-0.5, 0.1, 9.0, 0.1)

    prior=joint_prior.PriorSimpleSep(cen_prior,
                                     g_prior,
                                     T_prior,
                                     F_prior)

    obs=Observation(im_obj, weight=wt_obj, jacobian=j, psf=psf_obs)

    nm_fitter=MaxSimple(obs, model, prior=prior, use_logpars=True)
    nm_guess=pars_obj.copy()
    nm_guess[2] = 0.1*srandu()
    nm_guess[3] = 0.1*srandu()
    nm_guess[4:4+2] = log( nm_guess[4:4+2] )
    while True:
        nm_fitter.run_max(nm_guess, xtol=1.0e-5, ftol=1.0e-5, maxiter=4000, maxfev=4000)
        nm_res=nm_fitter.get_result()

        if nm_res['flags']==0:
            break
        
        nm_guess[0] = pars_obj[0] + 0.01*srandu()
        nm_guess[1] = pars_obj[1] + 0.01*srandu()
        nm_guess[2] = pars_obj[2] + 0.01*srandu()
        nm_guess[3] = pars_obj[3] + 0.01*srandu()
        nm_guess[4] = pars_obj[4] + 0.01*srandu()
        nm_guess[5] = pars_obj[5] + 0.01*srandu()

    nm_pars=nm_res['pars']
    nm_err=nm_res['pars_err']
    print("nfev:",nm_res['nfev'])
    print_pars(nm_pars, front="nm pars: ")

    guess=zeros( (nwalkers, npars) )
    '''
    guess[:,0] = nm_pars[0] + 0.01*srandu(nwalkers)
    guess[:,1] = nm_pars[1] + 0.01*srandu(nwalkers)

    guess[:,2] = nm_pars[2]*(1.0 + 0.01*randu(nwalkers))
    guess[:,3] = nm_pars[3]*(1.0 + 0.01*randu(nwalkers))
    guess[:,4] = nm_pars[4] + 0.01*srandu(nwalkers)
    guess[:,5] = nm_pars[5] + 0.01*srandu(nwalkers)
    '''

    guess[:,0] = nm_pars[0] + nm_err[0]*randn(nwalkers)
    guess[:,1] = nm_pars[1] + nm_err[1]*randn(nwalkers)

    guess[:,2] = nm_pars[2]*(1.0 + 0.01*randu(nwalkers))
    guess[:,3] = nm_pars[3]*(1.0 + 0.01*randu(nwalkers))
    guess[:,4] = nm_pars[4] + nm_err[4]*randn(nwalkers)
    guess[:,5] = nm_pars[5] + nm_err[5]*randn(nwalkers)


    t0=time.time()
    mcmc_fitter=MCMCSimple(obs, model, nwalkers=nwalkers, prior=prior, use_logpars=True)
    pos=mcmc_fitter.run_mcmc(guess, burnin)
    pos=mcmc_fitter.run_mcmc(pos, nstep, thin=2)
    mcmc_fitter.calc_result()
    tm=time.time()-t0

    trials=mcmc_fitter.get_trials()

    res_obj=mcmc_fitter.get_result()

    print_pars(pars_obj,            front='true pars: ')
    print_pars(nm_res['pars'],      front='pars max:  ')
    print_pars(nm_res['pars_err'],  front='perr max:  ')
    print_pars(res_obj['pars'],     front='pars mcmc: ')
    print_pars(res_obj['pars_err'], front='perr mcmc: ')
    print('T: %.4g +/- %.4g' % (res_obj['pars'][4], res_obj['pars_err'][4]))
    print("s2n:",res_obj['s2n_w'],"arate:",res_obj['arate'],"tau:",res_obj['tau'])
    print("time:",tm)
    print()

    if show:
        import biggles
        import lensing

        print("isamp nsample:",nsample)
        tm0=time.time()
        fsamples, weights = get_isamples_and_weights(mcmc_fitter,
                                                     nm_res['pars'],
                                                     nm_res['pars_cov'],
                                                     ifactor,
                                                     nsample)
        print("isamp time:",time.time()-tm0)
        print("weight max:",weights.max())
        neff=weights.sum()
        print("eff num:",neff,"frac:",neff/nsample)
        print()

        gsamples=fsamples[:,2:2+2]


        g1=trials[:,2]
        g2=trials[:,3]
        g1s=gsamples[:,0]
        g2s=gsamples[:,1]

        ming1=min( g1.min(), g1s.min() )
        maxg1=max( g1.max(), g1s.max() )
        ming2=min( g2.min(), g2s.min() )
        maxg2=max( g2.max(), g2s.max() )
        
        mcmc_fitter.make_plots(show=True)
        lin_trials=trials.copy()
        lin_trials[:,4] = 10.0**lin_trials[:,4]
        lin_trials[:,5] = 10.0**lin_trials[:,5]
        import mcmc
        mcmc.plot_results(lin_trials)
        linpars,lincov=mcmc.extract_stats(lin_trials)
        linerr=sqrt(diag(lincov))
        print_pars(linpars,front="linpars: ")
        print_pars(linerr, front="linerrs: ")


        g1plt=biggles.plot_hist(g1, nbin=nbin, min=ming1, max=maxg1,
                                color='blue',
                                xlabel='g1',
                                visible=False,
                                norm=1)
        biggles.plot_hist(g1s, nbin=nbin, min=ming1, max=maxg1,
                          color='red',
                          plt=g1plt,
                          visible=False,
                          norm=1)
        biggles.plot_hist(g1s,
                          weights=weights,
                          nbin=nbin, min=ming1, max=maxg1,
                          color='orange',
                          plt=g1plt,
                          visible=False,
                          norm=1)



        g2plt=biggles.plot_hist(g2, nbin=nbin, min=ming2, max=maxg2,
                                color='blue',
                                xlabel='g2',
                                visible=False,
                                norm=1)
        biggles.plot_hist(g2s, nbin=nbin, min=ming2, max=maxg2,
                          color='red',
                          plt=g2plt,
                          visible=False,
                          norm=1)
        biggles.plot_hist(g2s,
                          weights=weights,
                          nbin=nbin, min=ming2, max=maxg2,
                          color='orange',
                          plt=g2plt,
                          visible=False,
                          norm=1)


        g1plt.show()
        g2plt.show()

    return tm

def test_isample_many(num, *args, **kw):

    eff=zeros(num)
    nfev=zeros(num)
    for i in xrange(num):
        print("%d/%d" % (i+1,num))
        mres,ires=test_isample(*args, **kw)

        eff[i] = ires['efficiency']
        nfev[i] = mres['nfev']

    effmean=eff.mean()
    effstd=eff.std()
    print("eff: %g +/- %g" % (effmean, effstd))
    print("nfev: %g +/- %g" % (nfev.mean(), nfev.std()))

    return  effmean, effstd

def test_isample(model,
                 nsample,
                 ifactor,
                 use_asinh=False, # implies use_logpars=False
                 df=2.1,
                 seed=None,
                 g1_obj=0.1,
                 g2_obj=0.05,
                 T=16.0,
                 counts=100.0,
                 g1_psf=0.0,
                 g2_psf=0.0,
                 T_psf=4.0,
                 noise=0.001,
                 nbin=50,
                 max_fitter_type='lm',
                 show=False):
    """
    Test fitting the specified model.
    """
    from . import em
    from . import joint_prior
    import time
    from math import ceil
    from numpy.random import randn

    numpy.random.seed(seed)

    g_prior = priors.GPriorBA(0.3)

    if use_asinh:
        use_logpars=False
        asinh_pars=[4,5]
    else:
        use_logpars=True
        asinh_pars=[]
    #
    # simulation
    #

    # PSF pars
    counts_psf=100.0
    noise_psf=0.001

    sigma=sqrt( (T + T_psf)/2. )
    dims=[int(ceil(2.*5.*sigma))]*2
    cen=[dims[0]/2., dims[1]/2.]
    j=UnitJacobian(cen[0],cen[1])

    pars_psf = [0.0, 0.0, g1_psf, g2_psf, T_psf, counts_psf]
    gm_psf=gmix.GMixModel(pars_psf, "gauss")

    pars_obj = array([0.0, 0.0, g1_obj, g2_obj, T, counts])
    npars=pars_obj.size
    gm_obj0=gmix.GMixModel(pars_obj, model)

    gm=gm_obj0.convolve(gm_psf)

    im_psf=gm_psf.make_image(dims, jacobian=j)
    im_psf[:,:] += noise_psf*numpy.random.randn(im_psf.size).reshape(im_psf.shape)
    wt_psf=zeros(im_psf.shape) + 1./noise_psf**2

    im_obj=gm.make_image(dims, jacobian=j)
    im_obj[:,:] += noise*numpy.random.randn(im_obj.size).reshape(im_obj.shape)
    wt_obj=zeros(im_obj.shape) + 1./noise**2

    #
    # fitting
    #


    # psf using EM
    im_psf_sky,sky=em.prep_image(im_psf)
    psf_obs = Observation(im_psf_sky, jacobian=j)
    mc_psf=em.GMixEM(psf_obs)

    emo_guess=gm_psf.copy()
    emo_guess._data['p'] = 1.0
    emo_guess._data['row'] += 0.1*srandu()
    emo_guess._data['col'] += 0.1*srandu()
    emo_guess._data['irr'] += 0.5*srandu()
    emo_guess._data['irc'] += 0.1*srandu()
    emo_guess._data['icc'] += 0.5*srandu()

    mc_psf.run_em(emo_guess, sky)
    res_psf=mc_psf.get_result()
    print('psf numiter:',res_psf['numiter'],'fdiff:',res_psf['fdiff'])

    psf_fit=mc_psf.get_gmix()

    psf_obs.set_gmix(psf_fit)

    cen_prior=priors.CenPrior(0.0, 0.0, 0.1, 0.1)
    if use_logpars:
        T_prior=priors.TwoSidedErf(-10.0, 0.1, 13., 0.1)
        F_prior=priors.TwoSidedErf(-10.0, 0.1, 13., 0.1)
    else:
        T_prior=priors.TwoSidedErf(-1.0, 0.1, 1.0e6, 1.0e5)
        F_prior=priors.TwoSidedErf(-1.0, 0.1, 1.0e6, 1.0e5)

    prior=joint_prior.PriorSimpleSep(cen_prior,
                                     g_prior,
                                     T_prior,
                                     F_prior)

    obs=Observation(im_obj, weight=wt_obj, jacobian=j, psf=psf_obs)

    if max_fitter_type=='nm':
        max_fitter=MaxSimple(obs, model, prior=prior, use_logpars=use_logpars)
    else:
        max_fitter=LMSimple(obs, model, prior=prior, use_logpars=use_logpars,
                            lm_pars={'maxfev':4000,
                                     'ftol':1.0e-6,
                                     'xtol':1.0e-6,
                                     'epsfcn':1.0e-6})
    max_guess=pars_obj.copy()
    max_guess[2] = 0.1*srandu()
    max_guess[3] = 0.1*srandu()

    if use_logpars:
        max_guess[4:4+2] = log( max_guess[4:4+2] )
    else:
        max_guess[4:4+2] = max_guess[4:4+2]

    while True:
        tm0=time.time()
        if max_fitter_type=='nm':
            max_fitter.run_max(max_guess, xtol=1.0e-5, ftol=1.0e-5, maxiter=4000, maxfev=4000)
        else:
            max_fitter.run_lm(max_guess)
        max_res=max_fitter.get_result()
        max_tm=time.time()-tm0

        if max_res['flags']==0:
            break
        
        max_guess[0] = pars_obj[0] + 0.01*srandu()
        max_guess[1] = pars_obj[1] + 0.01*srandu()
        max_guess[2] = 0.1*srandu()
        max_guess[3] = 0.1*srandu()
        if use_logpars:
            max_guess[4] = log(pars_obj[4]) + 0.01*srandu()
            max_guess[5] = log(pars_obj[5]) + 0.01*srandu()
        else:
            max_guess[4] = pars_obj[4]*(1.0 + 0.01*srandu())
            max_guess[5] = pars_obj[5]*(1.0 + 0.01*srandu())

    max_res['lm_pars_err'] = max_res['pars_err'].copy()
    max_fitter.calc_cov(1.0e-3, 5.0)
    print("nfev:",max_res['nfev'],"s2n:",max_res['s2n_w'],"time:",max_tm)
    print_pars(max_res['pars'],      front='pars max:  ')
    print_pars(max_res['pars_err'],  front='perr max:  ')

    print()

    print("isamp nsample:",nsample)
    tm0=time.time()
    #cov = max_res['pars_cov']*ifactor**2
    cov = max_res['pars_cov']


    # student T sampler
    isampler=ISampler(max_res['pars'], cov, df, ifactor=ifactor, asinh_pars=asinh_pars)
    isampler.make_samples(nsample)
    isampler.set_iweights(max_fitter.calc_lnprob)
    isampler.calc_result()

    isamples=isampler.get_samples()
    iweights = isampler.get_iweights()
    res=isampler.get_result()

    print("T isamp time:",time.time()-tm0)
    neff=iweights.sum()
    print("T eff num:",neff,"frac:",neff/nsample)
    print_pars(res['pars'], front='pars: ')
    print_pars(res['pars_err'], front='perr: ')
    print()

    

    if show:
        import biggles
        import lensing
        import mcmc

        #mcmc.plot_results(samples)

        g1=samples[:,2]
        g2=samples[:,3]
        Tg1=isamples[:,2]
        Tg2=isamples[:,3]
        ming1=min(g1.min(), Tg1.min())
        maxg1=max(g1.max(), Tg1.max())

        ming2=min(g2.min(), Tg2.min())
        maxg2=max(g2.max(), Tg2.max())

        g1plt=biggles.plot_hist(g1,
                                nbin=nbin,
                                weights=iweights,
                                min=ming1,
                                max=maxg1,
                                color='blue',
                                xlabel='g1',
                                visible=False,
                                norm=1)
        biggles.plot_hist(g1,
                          nbin=nbin,
                          min=ming1,
                          max=maxg1,
                          color='red',
                          plt=g1plt,
                          visible=False,
                          norm=1)

        g2plt=biggles.plot_hist(g2,
                                nbin=nbin,
                                min=ming2,
                                max=maxg2,
                                weights=iweights,
                                color='blue',
                                xlabel='g2',
                                visible=False,
                                norm=1)
        biggles.plot_hist(g2,
                          nbin=nbin,
                          min=ming2,
                          max=maxg2,
                          color='red',
                          plt=g2plt,
                          visible=False,
                          norm=1)


        # from T sampler

        biggles.plot_hist(Tg1,
                          nbin=nbin,
                          min=ming1,
                          max=maxg1,
                          weights=iweights,
                          color='magenta',
                          width=2,
                          plt=g1plt,
                          visible=False,
                          norm=1)

        biggles.plot_hist(Tg2,
                          nbin=nbin,
                          min=ming2,
                          max=maxg2,
                          weights=iweights,
                          color='magenta',
                          width=2,
                          plt=g2plt,
                          visible=False,
                          norm=1)


        g1plt.show()
        g2plt.show()

    return max_res, res

def test_fracdev(fracdev=0.3,
                 noise=0.1,
                 noise_psf=0.001,
                 use_logpars=False,
                 fracdev_method='lm',
                 seed=None,
                 verbose=True,
                 show=False):
    from . import joint_prior, priors

    numpy.random.seed(seed)

    Flux = 100.0
    Texp = 16.0
    Tdev = 16.0

    g1=0.1
    g2=0.03

    Tpsf=4.0

    Tw = (1-fracdev)*Texp + fracdev*Tdev
    if verbose:
        print("g1:",g1,"g2:",g2,"Tw:",Tw)
    Ttot = Texp + Tdev + Tpsf

    sigma=sqrt(Ttot/2)

    dim=2*5*int(sigma)
    cen=[(dim-1)/2.]*2
    dims=[dim]*2

    jacobian=UnitJacobian(cen[0],cen[0])
    epars=array([0.0, 0.0, g1, g2, Texp, (1-fracdev)*Flux] )
    dpars=array([0.0, 0.0, g1, g2, Tdev,     fracdev*Flux])
    gme0 = gmix.GMixModel(epars,'exp')
    gmd0 = gmix.GMixModel(dpars,'dev')

    ppars=array( [0.0,0.0,0.0,0.0,Tpsf,1.0] )
    gmpsf = gmix.GMixModel(ppars,'gauss')

    gme=gme0.convolve(gmpsf)
    gmd=gmd0.convolve(gmpsf)

    impsf = gmpsf.make_image(dims, jacobian=jacobian)
    ime   = gme.make_image(dims, jacobian=jacobian)
    imd   = gmd.make_image(dims, jacobian=jacobian)

    impsf += numpy.random.normal(scale=noise_psf, size=impsf.shape)
    wtpsf = numpy.zeros(impsf.shape) + 1.0/noise_psf**2

    im0 = ime + imd

    im = im0 + numpy.random.normal(scale=noise, size=im0.shape)
    wt = numpy.zeros(im.shape) + 1.0/noise**2


    psf_obs = Observation(impsf, weight=wtpsf, jacobian=jacobian)
    obs = Observation(im, weight=wt, jacobian=jacobian)

    # fit psf
    pfitter=LMSimple(psf_obs, 'gauss')
    pfitter.go(ppars)

    psf_obs.set_gmix( pfitter.get_gmix() )

    # fit galaxy to exp and dev
    obs.set_psf(psf_obs)

    guess = epars.copy()
    guess[4] = Tw
    guess[5] = Flux


    g_prior = priors.ZDisk2D(0.985)
    cen_prior=priors.CenPrior(0.0, 0.0, 0.1, 0.1)
    #T_prior=priors.FlatPrior(-1.0, 1.0e6)
    #F_prior=priors.FlatPrior(-1.0, 1.0e6)

    if use_logpars:
        T_prior=priors.TwoSidedErf(-11.,1., 13.8,1.)
        F_prior=priors.TwoSidedErf(-11.,1., 13.8,1.)
        guess[4] = log(guess[4])
        guess[5] = log(guess[5])
    else:
        T_prior=priors.TwoSidedErf(0.0,0.1, 1.0e6,1.0e5)
        F_prior=priors.TwoSidedErf(0.0,0.1, 1.0e6,1.0e5)

    prior=joint_prior.PriorSimpleSep(cen_prior,
                                     g_prior,
                                     T_prior,
                                     F_prior)


    efitter=LMSimple(obs, 'exp',prior=prior,use_logpars=use_logpars)
    efitter.go(guess)

    dfitter=LMSimple(obs, 'dev',prior=prior,use_logpars=use_logpars)
    dfitter.go(guess)

    # fit for fracdev
    eres=efitter.get_result()
    dres=dfitter.get_result()


    if eres['flags'] != 0:
        raise RuntimeError("exp failed with flags %s" % eres['flags'])
    if dres['flags'] != 0:
        raise RuntimeError("dev failed with flags %s" % dres['flags'])

    efitpars = eres['pars']
    dfitpars = dres['pars']
    pefitpars = efitpars.copy()
    pdfitpars = dfitpars.copy()
    if use_logpars:
        pefitpars[4:] = exp(pefitpars[4:])
        pdfitpars[4:] = exp(pdfitpars[4:])
    
    if verbose:
        print_pars(pefitpars,front="    efitpars: ")
        print_pars(pdfitpars,front="    dfitpars: ")
        print("chi2per exp:",eres['chi2per'],'dev:',dres['chi2per'])
        print("nfev exp:",eres['nfev'],'dev:',dres['nfev'])

    '''
    tm0=time.time()
    ffitter_old = FracdevFitterMax(obs, efitpars, dfitpars,
                            use_logpars=use_logpars,
                            method=fracdev_method)

    ffitter_old.go(0.5 + 0.1*srandu())
    resold=ffitter_old.get_result()
    tm=time.time()-tm0

    if resold['flags'] != 0:
        raise RuntimeError("failed with flags: %s" %resold['flags'])
    '''
    tm0=time.time()
    ffitter_new = FracdevFitter(obs, efitpars, dfitpars,
                                   use_logpars=use_logpars)
    res=ffitter_new.get_result()
    tmnew=time.time()-tm0

    fdfit = res['fracdev']
    Fluxfit= pefitpars[5]*(1.0-fdfit)+ pdfitpars[5]*fdfit

    if verbose:
        print()
        #print("fracdev nfev:",resold['nfev'])
        print("fracdev true:",fracdev)
        #print("fracdev old:  %.3g +/- %.3g" % (resold['fracdev'],resold['fracdev_err']))
        print("fracdev fit:  %.3g +/- %.3g" % (res['fracdev'],res['fracdev_err']))

        print("flux fit:",Fluxfit)

        #print("time: %.3g tmnew: %.3g" % (tm,tmnew))

    fduse = fdfit
    #fduse = fracdev
    TdByTe=pdfitpars[4]/pefitpars[4]
    cfitter=LMComposite(obs, fduse, TdByTe,prior=prior,
                        use_logpars=use_logpars)
    cfitter.go(guess)
    
    cres=cfitter.get_result()

    if cres['flags'] != 0:
        raise RuntimeError("failed with flags %s" % cres['flags'])


    ppars = cres['pars'].copy()
    pperr = cres['pars_err'].copy()
    if use_logpars:
        ppars[4:] = exp(ppars[4:])
        pperr[4:] = ppars[4:]*pperr[4:]

    if verbose:
        print()
        print("s/n:",cres['s2n_w'],"nfev:",cres['nfev'],'chi2per:',cres['chi2per'])
        print_pars(ppars,front='    cpars: ')
        print_pars(pperr,front='    cperr: ')

    if show:
        import images

        egm0=efitter.get_gmix()
        psf_gmix = pfitter.get_gmix()
        egm=egm0.convolve(psf_gmix)

        dgm0=dfitter.get_gmix()
        psf_gmix = pfitter.get_gmix()
        dgm=dgm0.convolve(psf_gmix)

 
        cgm0=cfitter.get_gmix()
        psf_gmix = pfitter.get_gmix()
        cgm=cgm0.convolve(psf_gmix)

        eim=egm.make_image(dims, jacobian=jacobian)
        dim=dgm.make_image(dims, jacobian=jacobian)
        cim=cgm.make_image(dims, jacobian=jacobian)

        images.compare_images(im, eim,
                              label1='image',
                              label2='exp')
        images.compare_images(im, dim,
                              label1='image',
                              label2='dev')
        images.compare_images(im, cim,
                              label1='image',
                              label2='cm')

    return cres['pars']

def test_metacal(model, show=False, **kw):
    from .metacal import Metacal
    from .shape import Shape
    import images
    from .bootstrap import Bootstrapper

    max_pars={'method':'lm','lm_pars':{'maxfev':4000}}
    # galsim has a different convention from ngmix
    T_obj=4.0
    #T_obj=100.0
    psf_obs, obs=make_test_observations(model,
                                        g1_obj=0.0, g2_obj=0.0,
                                        T_obj=T_obj, **kw)
    print("dims:",obs.image.shape)
    print("cen0:", array(obs.image.shape)/2.)

    obs.set_psf(psf_obs)

    boot0=Bootstrapper(obs)
    boot0.fit_psfs('gauss', 4.0)
    boot0.fit_max('exp', max_pars)
    res0=boot0.get_max_fitter().get_result()
    print_pars(res0['pars'], front='    res0: ')

    print("making Metacal object")
    mc=Metacal(obs)

    print("getting shears")
    sval=0.01
    sh1m=Shape(-sval,  0.00 )
    sh1p=Shape( sval,  0.00 )
    sh2m=Shape( 0.00, -sval )
    sh2p=Shape( 0.00,  sval )

    print("getting galshear obs")
    R_obs1m = mc.get_obs_galshear(sh1m)
    R_obs1p = mc.get_obs_galshear(sh1p)
    R_obs2m = mc.get_obs_galshear(sh2m)
    R_obs2p = mc.get_obs_galshear(sh2p)


    # you can also get an unsheared, just convolved obs
    print("getting unsheared galshear obs")
    Runsheared, psf_unsheared= mc.get_obs_galshear(sh1p, get_unsheared=True)

    print("getting psfshear obs")
    # observations used to calculate Rpsf
    Rpsf_obs1m = mc.get_obs_psfshear(sh1m)
    Rpsf_obs1p = mc.get_obs_psfshear(sh1p)
    Rpsf_obs2m = mc.get_obs_psfshear(sh2m)
    Rpsf_obs2p = mc.get_obs_psfshear(sh2p)

    for tobs in [R_obs1m,R_obs1p,R_obs2m,R_obs2p,
                 Rpsf_obs1m,Rpsf_obs1p,Rpsf_obs2m,Rpsf_obs2p]:
        b=Bootstrapper(tobs)
        b.fit_psfs('gauss', 4.0)
        b.fit_max('exp', max_pars)
        tres=b.get_max_fitter().get_result()
        print_pars(tres['pars'], front='    tres: ')

    if show:
        width=1100
        height=900
        #images.compare_images(obs.image, R_obs1m.image, label1='im',label2='sh 1m')
        plt=images.compare_images(obs.image, R_obs1p.image, 
                                  label1='im',label2='sh 1p',
                                  width=width,height=height)
        plt.write_img(width,height,'/astro/u/esheldon/tmp/sh1p-diff.png')

def test_fit_gauss1_many(num, T=4.0, method='lm', use_errors=False, eps=None,show=False, **keys):
    import esutil as eu
    from .gexceptions import GMixMaxIterEM
    used=zeros(num,dtype='i2')
    g1vals=zeros(num)
    g2vals=zeros(num)

    Irr=zeros(num)
    Irc=zeros(num)
    Icc=zeros(num)

    Irr_err=zeros(num)
    Irc_err=zeros(num)
    Icc_err=zeros(num)


    keys['verbose']=False
    for i in xrange(num):
        try:
            fitter=test_fit_gauss1(T=T, method=method, **keys)
            if fitter is not None:
                used[i]=1
                gm=fitter.get_gmix()
                g1,g2,Ttmp=gm.get_g1g2T()

                g1vals[i]=g1
                g2vals[i]=g2

                '''
                d=gm.get_data()
                Irr[i] = d['irr'][0]
                Irc[i] = d['irc'][0]
                Icc[i] = d['icc'][0]
                '''
                res=fitter.get_result()
                pars=res['pars']
                perr=res['pars_err']
                Irr[i] = pars[3]
                Irc[i] = pars[4]
                Icc[i] = pars[5]
                Irr_err[i] = perr[3]
                Irc_err[i] = perr[4]
                Icc_err[i] = perr[5]

        except GMixMaxIterEM:
            pass

    w,=where(used==1)
    numuse=w.size
    print("used:",numuse)

    Tvals = Irr + Icc
    Tmean,Tstd,Terr=eu.stat.sigma_clip(Tvals, get_err=True)
    print("T:     %g" % T)
    print("Tmeas: %g +/- %g +/- %g" % (Tmean,Tstd,Terr))

    g1vals=g1vals[w]
    g2vals=g2vals[w]
    g=sqrt(g1vals**2 + g2vals**2)

    g1mean=g1vals.mean()
    g2mean=g2vals.mean()

    g1mean,g1std,g1err=eu.stat.sigma_clip(g1vals, get_err=True)
    g2mean,g2std,g2err=eu.stat.sigma_clip(g2vals, get_err=True)

    print("g1:  %g +/- %g" % (g1mean,g1err))
    print("g2:  %g +/- %g" % (g2mean,g2err))

    nvals={}
    types=[(Irr,Irr_err,'Irr'), 
           (Irc,Irc_err,'Irc'), 
           (Icc,Icc_err,'Icc'),
           (g,None,'|g|')]

    for t in types:
        data,errs,label=t
        if label=='|g|':
            continue

        m,s,err=eu.stat.sigma_clip(data,get_err=True)
        print("%s: %g +/- %g +/- %g" % (label,m,s,err))
        if use_errors:
            s,ssig=eu.stat.sigma_clip(errs)

        if label=='Irc':
            midval=(Tmean/2.)/numpy.abs(m)
            p=priors.LogNormal(midval,s)
            if m < 0:
                r=m + -(p.sample(numuse*10)-midval)
            else:
                r=m + p.sample(numuse*10)-midval
        else:
            p=priors.LogNormal(m,s)
            r=p.sample(numuse*10)

        nvals[label] = r

    if show or eps:
        import biggles
        from biggles import plot_hist
        tab=biggles.Table(2,2)


        grid=eu.plotting.Grid(4)
        for i,tup in enumerate(types):
            data,errs,label=tup
            
            m,s=eu.stat.sigma_clip(data)
            binsize=0.2*s
            minval=m-4*s
            maxval=m+4*s
            plt=plot_hist(data,
                          min=minval,max=maxval,
                          visible=False,
                          binsize=binsize,
                          norm=1,
                          xlabel=label)

            ptext=biggles.PlotLabel(0.9,0.9,
                                    'mn: %.3g +/- %.3g' % (m,s),
                                    halign='right')
            plt.add(ptext)
            if label in ['Irr','Icc','Irc']:
                r=nvals[label]

                plot_hist(r,
                          min=minval,max=maxval,
                          visible=False,
                          binsize=binsize,
                          color='red',
                          norm=1,
                          plt=plt)
            row,col=grid(i)
            tab[row,col] = plt

        if eps:
            tab.write_eps(eps)

        if show:
            tab.show()

def test_fit_gauss1(model='gauss',
                    method='em',
                    dopsf=False,
                    g1=0.0,
                    g2=0.0,
                    T=4.0,
                    flux=100.0,
                    noise=2.0,
                    maxiter=1000,
                    tol=1.0e-6,
                    verbose=True):
    """
    noise=2.0 is about s/n=10
    """
    from numpy.random import randn
    from . import em

    if model != 'gauss':
        dopsf=True

    if dopsf:
        nsub=4
        Tpsf=4.0
        Ttot = T+Tpsf
    else:
        nsub=1
        Ttot = T

    sigma=sqrt(Ttot/2.0)
    dim=int(round(2*5*sigma))

    dims=[dim]*2

    cen=(dim-1.)/2.

    pars=array([cen,cen,g1,g2,T,flux],dtype='f8')
    gm=gmix.GMixModel(pars, model)
    if dopsf:
        psf_pars=array([cen,cen,0.,0.,Tpsf,1.0],dtype='f8')
        gmpsf=gmix.GMixModel(psf_pars, 'gauss')
        gm=gm.convolve(gmpsf)

    im_nonoise=gm.make_image(dims, nsub=nsub)

    noise_im = noise*randn(dim*dim).reshape(im_nonoise.shape)
    im = im_nonoise + noise_im

    weight=numpy.zeros(im.shape) + 1.0/noise**2
    obsorig=Observation(im,weight=weight)

    if method=='em':
        imsky,sky=em.prep_image(im)
        obs=Observation(imsky)
        fitter=em.GMixEM(obs)

        if model=='gauss':
            guess = gm.copy()
        else:
            guess=gmpsf.copy()

        fitter.go(guess, sky, maxiter=maxiter, tol=tol)

    else:
        lm_pars={'maxfev': 4000}

        fitter=LMGaussMom(obsorig, lm_pars=lm_pars)
        guess=array([flux*(1.0 + 0.1*srandu()),
                     cen,cen,
                     T/2.*(1.0+0.1*srandu()),
                     0.1*srandu(),
                     T/2.*(1.0+0.1*srandu())])

        fitter.go(guess)

    res=fitter.get_result()
    if verbose:
        print("s2n:",gm.get_model_s2n(obsorig))
        if 'pars_cov' in res:
            import images
            import esutil as eu

            corr=eu.stat.cov2cor( res['pars_cov'][3:3+3, 3:3+3])
            images.imprint(corr)

    if res['flags']==0:
        return fitter
    else:
        return None

def test_fixT(model, verbose=True, show=False, T_obj=16.0, do_control=False, **kw):
    from .metacal import Metacal
    from .shape import Shape
    import images
    from .bootstrap import Bootstrapper

    max_pars={'method':'lm-fixT',
              'lm_pars':{'maxfev':4000}}
    psf_obs, obs=make_test_observations(model,
                                        T_obj=T_obj,
                                        **kw)

    obs.set_psf(psf_obs)

    boot=Bootstrapper(obs,use_logpars=True)
    boot.fit_psfs('gauss', 4.0)
    boot.fit_max_fixT('exp', max_pars, T_obj)
    res=boot.get_max_fitter().get_result()

    if verbose:
        print("s2n: %g nfev: %s" % (res['s2n_w'], res['nfev']))
        print_pars(res['pars'], front='    pars: ')
        print_pars(res['pars_err'], front='    perr: ')

    if do_control:
        max_pars['method']='lm'
        boot.fit_max('exp', max_pars)
        resc=boot.get_max_fitter().get_result()
        return res, resc
    else:
        return res

def test_fixT_many(num, model, **keys):
    used=zeros(num,dtype='i2')
    usedc=zeros(num,dtype='i2')
    g1vals=zeros(num)
    g2vals=zeros(num)
    g1valsc=zeros(num)
    g2valsc=zeros(num)

    keys['verbose']=False
    for i in xrange(num):
        try:
            res,resc=test_fixT(model, do_control=True, **keys)

            if res['flags']==0:
                pars=res['pars']
                used[i]=1
                g1vals[i]=pars[2]
                g2vals[i]=pars[3]

            if resc['flags']==0:
                pars=resc['pars']
                usedc[i]=1
                g1valsc[i]=pars[2]
                g2valsc[i]=pars[3]

        except BootGalFailure:
            pass
            
    w,=where(used==1)

    g1vals=g1vals[w]
    g2vals=g2vals[w]

    g1mean=g1vals.mean()
    g2mean=g2vals.mean()

    g1err=g1vals.std()/sqrt(w.size)
    g2err=g2vals.std()/sqrt(w.size)

    tup=(g1mean,g1err,g1vals.min(),g1vals.max(),
         w.size)
    print("g1:  %g +/- %g min: %g max: %g num: %s" % tup)

    w,=where(usedc==1)

    g1valsc=g1valsc[w]
    g2valsc=g2valsc[w]

    g1meanc=g1valsc.mean()
    g2meanc=g2valsc.mean()

    g1errc=g1valsc.std()/sqrt(w.size)
    g2errc=g2valsc.std()/sqrt(w.size)

    tup=(g1meanc,g1errc,g1valsc.min(),g1valsc.max(),
         w.size)
    print("g1c:  %g +/- %g min: %g max: %g num: %s" % tup)


class GPriorBAWrapper(priors.GPriorBA):
    def get_lnprob_scalar(self, gvec):
        return self.get_lnprob_scalar2d(gvec[0], gvec[1])

    def sample(self, num=1):
        assert num==1,"num==1 here"
        g1,g2=self.sample2d(num)
        return array([g1[0], g2[0]],dtype='f8')

    def fill_fdiff(self, pars, fdiff, **keys):
        index=0
        fdiff[index] = self.get_lnprob_scalar(pars)
        index += 1

        chi2 = -2*fdiff[0:index]
        chi2.clip(min=0.0, max=None, out=chi2)
        fdiff[0:index] = sqrt(chi2)

        return index



def test_gonly(model, verbose=True, show=False, T_obj=16.0, do_control=False, **kw):
    from .metacal import Metacal
    from .shape import Shape
    import images
    from .bootstrap import Bootstrapper

    use_logpars=True

    g_prior_wrap = GPriorBAWrapper(0.3)

    cen_prior=priors.CenPrior(0.0, 0.0, 0.1, 0.1)
    g_prior = priors.GPriorBA(0.3)
    T_prior=priors.FlatPrior(-10.0,15.0)
    counts_prior=priors.FlatPrior(-10.0,15.0)

 
    prior=joint_prior.PriorSimpleSep(cen_prior,
                                     g_prior,
                                     T_prior,
                                     counts_prior)

    max_pars={'method':'lm-gonly',
              'lm_pars':{'maxfev':4000}}
    mdict = make_test_observations(model,
                                   T_obj=T_obj,
                                   more=True,
                                   **kw)

    obs=mdict['obs']
    pars=mdict['pars']
    obs.set_psf(mdict['psf_obs'])

    if use_logpars:
        pars[4:] = log(pars[4:])

    boot=Bootstrapper(obs,use_logpars=use_logpars)
    boot.fit_psfs('gauss', 4.0)
    boot.fit_max_gonly('exp', max_pars, pars, prior=g_prior_wrap)
    res=boot.get_max_fitter().get_result()

    if verbose:
        print("s2n: %g nfev: %s" % (res['s2n_w'], res['nfev']))
        print_pars(res['pars'], front='    pars: ')
        print_pars(res['pars_err'], front='    perr: ')

    if do_control:
        max_pars['method']='lm'
        boot.fit_max('exp', max_pars, prior=prior)
        resc=boot.get_max_fitter().get_result()
        return res, resc
    else:
        return res

def test_gonly_many(num, model, **keys):
    used=zeros(num,dtype='i2')
    usedc=zeros(num,dtype='i2')
    g1vals=zeros(num)
    g2vals=zeros(num)
    g1valsc=zeros(num)
    g2valsc=zeros(num)

    keys['verbose']=False
    for i in xrange(num):
        try:
            res,resc=test_gonly(model, do_control=True, **keys)

            if res['flags']==0:
                pars=res['pars']
                used[i]=1
                g1vals[i]=pars[0]
                g2vals[i]=pars[1]

            if resc['flags']==0:
                pars=resc['pars']
                usedc[i]=1
                g1valsc[i]=pars[2]
                g2valsc[i]=pars[3]

        except BootGalFailure:
            pass
            
    w,=where(used==1)

    g1vals=g1vals[w]
    g2vals=g2vals[w]

    g1mean=g1vals.mean()
    g2mean=g2vals.mean()

    g1err=g1vals.std()/sqrt(w.size)
    g2err=g2vals.std()/sqrt(w.size)

    tup=(g1mean,g1err,g1vals.min(),g1vals.max(),
         w.size)
    print("g1:  %g +/- %g min: %g max: %g num: %s" % tup)

    w,=where(usedc==1)

    g1valsc=g1valsc[w]
    g2valsc=g2valsc[w]

    g1meanc=g1valsc.mean()
    g2meanc=g2valsc.mean()

    g1errc=g1valsc.std()/sqrt(w.size)
    g2errc=g2valsc.std()/sqrt(w.size)

    tup=(g1meanc,g1errc,g1valsc.min(),g1valsc.max(),
         w.size)
    print("g1c:  %g +/- %g min: %g max: %g num: %s" % tup)

    if True:
        from biggles import plot_hist
        plt=plot_hist(g1vals, nbin=50,visible=False)
        plt.write_eps('g1vals.eps')

def perturb_gmix(gm0):
    ngauss=len(gm0)

    pars=gm0.get_full_pars()
    gm=gm0.copy()

    beg=0
    for i in xrange(ngauss):
        pars[beg+0] = (1.0/ngauss)*(1.0 + 0.1*srandu())
        pars[beg+1] += 0.1*srandu()
        pars[beg+2] += 0.1*srandu()
        pars[beg+3] *= (1.0 + 0.1*srandu())
        pars[beg+4] += 0.1*srandu()
        pars[beg+5] *= (1.0 + 0.1*srandu())

        beg += 6

    gm=gmix.GMix(pars=pars)
    return gm

def guess_em_ngauss(ngauss, T):
    pars=numpy.zeros(ngauss*6)
    T2=T/2.

    beg=0
    for i in xrange(ngauss):
        pars[beg+0] = (1.0/ngauss)*(1.0 + 0.1*srandu())
        pars[beg+1] = 0.1*srandu()
        pars[beg+2] = 0.1*srandu()
        pars[beg+3] = T2*(1.0 + 0.1*srandu())
        pars[beg+4] = 0.1*T2*srandu()
        pars[beg+5] = T2*(1.0 + 0.1*srandu())

        beg += 6

    gm=gmix.GMix(pars=pars)
    return gm


def test_em_model(model,
                  ngauss=None, 
                  maxiter=4000, tol=1.0e-6,
                  ntry=10,
                  verbose=True, show=False, T_obj=16.0, do_control=False, **kw):
    from . import em

    mdict = make_test_observations(model,
                                   T_obj=T_obj,
                                   more=True,
                                   **kw)

    print_pars(mdict['pars'], front='    pars: ')
    obs=mdict['obs']

    s2n=mdict['gm_obj'].get_model_s2n(obs)

    if verbose:
        print("s2n: %g" % s2n)

    imsky,sky=em.prep_image(obs.image)

    newobs = Observation(imsky, jacobian=obs.jacobian)

    mc=em.GMixEM(newobs)

    for i in xrange(ntry):
        if ngauss is None:
            guess = perturb_gmix(mdict['gm_obj'])
        else:
            guess=guess_em_ngauss(ngauss, T_obj)

        if verbose:
            print("-")
            print(guess)

        mc.go(guess, sky, maxiter=maxiter, tol=tol)
    
        res=mc.get_result()
        if res['flags']==0:
            break

    if res['flags']==0:

        gm=mc.get_gmix()
        if show:
            import images

            model_im=gm.make_image(obs.image.shape, jacobian=obs.jacobian)
            model_im *= obs.image.sum()/model_im.sum()
            images.compare_images(obs.image, model_im)
        return gm
    else:
        return None

def test_moms(model='gauss',
              wmodel='gauss',
              wT=4.0,
              wg1=0.0,
              wg2=0.0,
              dopsf=False,
              g1=0.0,
              g2=0.0,
              T=4.0,
              flux=100.0,
              noise=2.0,
              verbose=True):
    """
    wgmix is a gaussian mixture to use for the weight

    noise=2.0 is about s/n=10
    """
    from numpy.random import randn
    from . import em

    if model != 'gauss':
        dopsf=True

    if dopsf:
        nsub=4
        Tpsf=4.0
        Ttot = T+Tpsf
    else:
        nsub=1
        Ttot = T

    sigma=sqrt(Ttot/2.0)
    dim=int(round(2*5*sigma))

    dims=[dim]*2

    cen=(dim-1.)/2.

    jacobian=UnitJacobian(cen,cen)
    pars=array([0.0,0.0,g1,g2,T,flux],dtype='f8')

    gm=gmix.GMixModel(pars, model)
    if dopsf:
        psf_pars=array([0.0,0.0,0.,0.,Tpsf,1.0],dtype='f8')
        gmpsf=gmix.GMixModel(psf_pars, 'gauss')
        gm=gm.convolve(gmpsf)

    im_nonoise=gm.make_image(dims, jacobian=jacobian, nsub=nsub)

    noise_im = noise*randn(dim*dim).reshape(im_nonoise.shape)
    im = im_nonoise + noise_im

    weight=numpy.zeros(im.shape) + 1.0/noise**2
    obs=Observation(im, jacobian=jacobian)

    # now the weight model
    wpars=array([0.0,0.0,wg1,wg2,wT,1.0],dtype='f8')
    wgm = gmix.GMixModel(wpars,wmodel)

    res=wgm.get_weighted_mom_sums(obs)

    res['skysig'] = noise
    res['Ierr'] = noise*sqrt(res['VIsum'])
    res['Terr'] = noise*sqrt(res['VTsum'])
    res['M1err'] = noise*sqrt(res['VM1sum'])
    res['M2err'] = noise*sqrt(res['VM2sum'])
    res['s2n_w'] = res['Isum']/res['Ierr']
    return res

def test_moms_many(num, method='lm', use_errors=False, eps=None,show=False, **keys):
    import esutil as eu
    from .gexceptions import GMixMaxIterEM
    used=zeros(num,dtype='i2')

    I=zeros(num)
    T=zeros(num)
    M1=zeros(num)
    M2=zeros(num)

    I_err=zeros(num)
    T_err=zeros(num)
    M1_err=zeros(num)
    M2_err=zeros(num)

    s2n=zeros(num)

    keys['verbose']=False
    for i in xrange(num):
        try:
            res=test_moms(**keys)

            if res['niter'] < res['maxiter']:
                used[i] = 1

                I[i]  = res['Isum']
                T[i]  = res['Tsum']
                M1[i] = res['M1sum']
                M2[i] = res['M2sum']

                I_err[i]  = res['VIsum']
                T_err[i]  = res['VTsum']
                M1_err[i] = res['VM1sum']
                M2_err[i] = res['VM2sum']

                s2n[i] = res['s2n_w']

        except GMixMaxIterEM:
            pass

    w,=where(used)
    I=I[w]
    T=T[w]
    M1=M1[w]
    M2=M2[w]
    I_err=I_err[w]
    T_err=T_err[w]
    M1_err=M1_err[w]
    M2_err=M2_err[w]
    s2n=s2n[w]
    
    m,s,err=eu.stat.sigma_clip(s2n,get_err=True)
    print("s/n: %g +/- %g +/- %g" % (m,s,err))
    nvals={}
    types=[(I,I_err,'I'), 
           (T,T_err,'T'), 
           (M1,M1_err,'M1'),
           (M2,M2_err,'M2')]

    for t in types:
        data,errs,label=t
        if label=='|g|':
            continue

        m,s,err=eu.stat.sigma_clip(data,get_err=True)
        print("%s: %g +/- %g +/- %g" % (label,m,s,err))
        if use_errors:
            s,ssig=eu.stat.sigma_clip(errs)

        '''
        if label=='Irc':
            midval=(Tmean/2.)/numpy.abs(m)
            p=priors.LogNormal(midval,s)
            if m < 0:
                r=m + -(p.sample(numuse*10)-midval)
            else:
                r=m + p.sample(numuse*10)-midval
        else:
            p=priors.LogNormal(m,s)
            r=p.sample(numuse*10)

        nvals[label] = r
        '''

    if show or eps:
        import biggles
        from biggles import plot_hist
        tab=biggles.Table(2,2)


        grid=eu.plotting.Grid(4)
        for i,tup in enumerate(types):
            data,errs,label=tup
            
            m,s=eu.stat.sigma_clip(data)
            binsize=0.2*s
            minval=m-4*s
            maxval=m+4*s
            plt=plot_hist(data,
                          min=minval,max=maxval,
                          visible=False,
                          binsize=binsize,
                          norm=1,
                          xlabel=label)

            ptext=biggles.PlotLabel(0.9,0.9,
                                    'mn: %.3g +/- %.3g' % (m,s),
                                    halign='right')
            plt.add(ptext)
            '''
            if label in ['Irr','Icc','Irc']:
                r=nvals[label]

                plot_hist(r,
                          min=minval,max=maxval,
                          visible=False,
                          binsize=binsize,
                          color='red',
                          norm=1,
                          plt=plt)
            '''
            row,col=grid(i)
            tab[row,col] = plt

        if eps:
            tab.write_eps(eps)

        if show:
            tab.show()


