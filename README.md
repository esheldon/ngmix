ngmix
=====

Gaussian mixture models for 2d images, implemented with python and numba.

Note this code is still under heavy development, including an alternative
implementation in C++.  No stable API is yet provided.

Also numba itself is under even heavier development, and in fact recent
releases have broken backwards compatibility.  This version only works with
the 0.11 release series.

See the caveats section below.

examples
--------

    import ngmix
    from ngmix.observation import Observation, ObsList, MultiBandObsList
    import numpy
    
    #
    # make a gaussian mixture, convolve it with a PSF and
    # render an image
    #

    # cen1,cen2,g1,g2,T,flux
    # note T=ixx+iyy
    pars=[15.0, 16.0, 0.2, -0.1, 16.0, 100.0]

    # PSF is a single gaussian
    psf_pars=[12.0, 12.0, -0.03, 0.02, 4.0, 1.0]

    gmix0=ngmix.gmix.GMixModel(pars,"exp")
    psf_gmix=ngmix.gmix.GMixModel(psf_pars,"gauss")

    gmix=gmix0.convolve(psf_gmix)

    dims=[32,32]
    image=gmix.make_image(dims, nsub=16)

    sigma=0.001
    noise = sigma*numpy.random.randn(image.size).reshape(image.shape)
    noisy_image = image + noise


    #
    # fit the data
    #
    # The code is designed to work on multiple images simulaneously, we always
    # fit in "sky coordinates".  We do this by making a jacobian to represent
    # the transformation, and make sure the center is at our best guess of the
    # object location.  In this case we make a unit jacobian.  for non-unit,
    # see the ngmix.jacobian.Jacobian class. In that case, we would also fit
    # the PSF in sky coordinates by sending a jacobian to the EM code


    # fit the PSF
    # it is best to fit the PSF using an EM algorithm
    psf_dims=[24,24]
    psf_im=psf_gmix.make_image(psf_dims, nsub=16)
    imsky,sky=ngmix.em.prep_image(psf_im)

    jacob=ngmix.jacobian.UnitJacobian(psf_pars[0], psf_pars[1])
    psf_obs=Observation(imsky, jacob=psf_jacob)

    em=ngmix.em.GMixEM(psf_obs)
    # guess truth
    guess=psf_gmix.copy()
    em.go(guess, sky, tol=1.e-5)

    psf_gmix_fit=em.get_gmix()

    psf_obs.add_gmix(psf_gmix_fit)

    # fit the galaxy.
    # the weight image for the fit
    weight=numpy.zeros(image.shape) + 1/sigma**2


    jacob=ngmix.jacobian.UnitJacobian(pars[0], pars[1])

    # we use a weight map and add the psf
    obs = Observation(image, weight=weight, jacob=jacob, psf=psf_obs)

    # Use MCMCSimple to fit using a "simple" model, either "exp" or "dev"
    # currently. You can also send a prior= keyword that takes parameters
    # and returns the log(probability)

    model="exp"
    fitter=ngmix.fitting.MCMCSimple(obs, model, nwalkers=80)

    # guess should be an array [nwalkers, npars].  It is good to
    # guess random points around your best estimate of the galaxy
    # parameters
    # note the fitter works in log10(T) and log10(flux)

    pos=fitter.run_mcmc(guess, burnin)
    pos=fitter.run_mcmc(pos, nstep)

    fitter.calc_result()
    fitter.calc_lin_result() # T, flux in linear space

    res=fitter.get_result()
    lin_res=fitter.get_lin_result()  # results in linear space
    print res['pars']
    print res['pars_perr']
    print res['pars_pcov'] # full covariance matrix

    # Fit multiple images of the same object. Send an ObsList to the fitter.
    # create an ObsList by appending Observation objects

    obs_list = ObsList()
    obs_list.append( obs1 )
    obs_list.append( obs2 )
    obs_list.append( obs3 )

    fitter=ngmix.fitting.MCMCSimple(obs_list, model, nwalkers=80)

    # you can also fit multiple bands/filters at once with fixed structural
    # parameters but different fluxes in each band

    mb_obs_list = MultiBandObsList()
    mb_obs_list.append( obs_list_g ) 
    mb_obs_list.append( obs_list_r ) 
    mb_obs_list.append( obs_list_i ) 
    mb_obs_list.append( obs_list_z ) 

    fitter=ngmix.fitting.MCMCSimple(mb_obs_list, model, nwalkers=80)

dependencies
------------

* numpy
* numba: http://numba.pydata.org/ I recommend installing the anaconda python distribution, which comes with numba https://store.continuum.io/cshop/anaconda/.  Remember to update numba with "conda update numba".  Only 0.1* series currentlyworks, as numba itself has changed dramatically.
* emcee: optional for doing MCMC fitting: http://dan.iel.fm/emcee/current/ Affine invariant MCMC sampler.

caveats
-------

* only numba 0.1* series works.
* numba is in heavy development.  Currently the JIT object code is not cached,
  so there is a slow compilation step every time the package is loaded the
  *first* time in your python session.  Caching is planned for a future numba
  release.
* numba does not yet support inlining.  Also function calls are slow.
  As a result, my numba-based code has a lot of redundancy, because the same
  bit of code must appear in multiple places.  Inlining could appear in a
  future numba release.

