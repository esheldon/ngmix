ngmix
=====

Gaussian mixtures models for 2d images, implemented with python and numba.

Note this code is still under heavy development, and no stable API is yet
provided.

examples
--------

    import ngmix
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

    # fit the PSF
    # it is best to fit the PSF using an EM algorithm
    psf_dims=[24,24]
    psf_im=psf_gmix.make_image(psf_dims, nsub=16)
    imsky,sky=ngmix.em.prep_image(psf_im)

    em=ngmix.em.GMixEM(imsky)
    guess=psf_gmix.copy()
    em.go(guess, sky, tol=1.e-5)

    psf_gmix_fit=em.get_gmix()

    # fit the galaxy.
    # the weight image for the fit
    weight=numpy.zeros(image.shape) + 1/sigma**2

    # The code is designed to work on multiple images simulaneously, we always
    # fit in "sky coordinates".  We do this by making a jacobian to represent
    # the transformation, and make sure the center is at our best guess of the
    # object location.  In this case we make a unit jacobian.  for non-unit,
    # see the ngmix.jacobian.Jacobian class. In that case, we would also fit
    # the PSF in sky coordinates by sending a jacobian to the EM code

    jacob=ngmix.jacobian.UnitJacobian(pars[0], pars[1])

    # fit using a "simple" model, either "exp" or "dev" currently
    model="exp"
    fitter=ngmix.fitting.MCMCSimple(noisy_image,weight,jacob,model,
                                    psf=psf_gmix_fit,
                                    T_guess=pars[4],
                                    counts_guess=pars[5],
                                    nwalkers=40,
                                    burnin=200,
                                    nstep=100)
    fitter.go()

    res=fitter.get_result()
    print res['pars']
    print res['perr']

    # Fit multiple images of the same object. Send the image, weight etc as
    # lists or even lists-of-lists if multiple wavelength bands are present.
    # If the input is a list of lists, each flux is fit for separately with
    # fixed structural parameters

    # 4 bands
    imlist_g = [im1, im2, im3, im4]
    # etc... for other bands. Similar for weight lists, jacobians and psfs

    im_lol = [ imlist_g, imlist_r, imlist_i, imlist_z ]
    wt_lol = [ wtlist_g, wtlist_r, wtlist_i, wtlist_z ]
    j_lol = [ jlist_g, jlist_r, jlist_i, jlist_z ]
    psf_lol = [ psf_list_g, psf_list_r, psf_list_i, psf_list_z]

    fitter=ngmix.fitting.MCMCSimple(im_lol,wt_lol,j_lol,model,
                                    psf=psf_gmix_fit, .....)

dependencies
------------

* numpy
* numba: http://numba.pydata.org/ I recommend instaling the anaconda python distribution, which comes with numba https://store.continuum.io/cshop/anaconda/.  Remember to update numba with "conda update numba".
* emcee: optional for doing MCMC fitting: http://dan.iel.fm/emcee/current/ Affine invariant MCMC sampler.

caveats
-------

* numba is in heavy development.  Currently the JIT object code is not cached,
  so there is a slow compilation step every time the package is loaded the
  *first* time in your python session.  Caching is planned for the next major
  numba release, which will shorten the startup times immensely.
* numba does not yet support inlining.  Also function calls are slow.
  As a result, my numba-based code has a lot of redundancy, because the same
  bit of code must appear in multiple places.  Inlining could appear in a
  future numba release.

