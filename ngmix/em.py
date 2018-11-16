"""
Fit an image with a gaussian mixture using the EM algorithm
"""
from __future__ import print_function, absolute_import, division

try:
    xrange
except NameError:
    xrange=range

import numpy

from . import gmix
from .gmix import GMix


from .gexceptions import GMixRangeError, GMixMaxIterEM
from .priors import srandu

from .jacobian import Jacobian, UnitJacobian

from .observation import Observation

from .em_nb import em_run

EM_RANGE_ERROR = 2**0
EM_MAXITER = 2**1

def fit_em(obs, guess, **keys):
    """
    fit the observation with EM
    """
    im,sky = prep_image(obs.image)
    newobs = Observation(im, jacobian=obs.jacobian)
    fitter=GMixEM(newobs)
    fitter.go(guess, sky, **keys)

    return fitter

def prep_image(im0):
    """
    Prep an image to fit with EM.  Make sure there are no pixels < 0

    parameters
    ----------
    image: ndarray
        2d image

    output
    ------
    new_image, sky:
        The image with new background level and the background level
    """
    im=im0.copy()

    # need no zero pixels and sky value
    im_min = im.min()
    im_max = im.max()
    sky=0.001*(im_max-im_min)

    im += (sky-im_min)

    return im, sky


class GMixEM(object):
    """
    Fit an image with a gaussian mixture using the EM algorithm

    parameters
    ----------
    obs: Observation
        An Observation object, containing the image and possibly
        non-trivial jacobian.  see ngmix.observation.Observation

        The image should not have zero or negative pixels. You can
        use the prep_image() function to ensure this.
    """
    def __init__(self, obs):

        self._obs=obs

        self._counts=obs.image.sum()

        self._gm        = None
        self._sums      = None
        self._result    = None
        self._sky_guess = None

    def get_gmix(self):
        """
        Get the gaussian mixture from the final iteration
        """
        return self._gm

    def get_result(self):
        """
        Get some stats about the processing
        """
        return self._result

    def make_image(self, counts=None):
        """
        Get an image of the best fit mixture
        """
        im=self._gm.make_image(self._obs.image.shape,
                               jacobian=self._obs.jacobian)
        if counts is not None:
            im *= (counts/im.sum())
        return im

    def go(self, gmix_guess, sky_guess, maxiter=100, tol=1.e-6):
        """
        Run the em algorithm from the input starting guesses

        parameters
        ----------
        gmix_guess: GMix
            A gaussian mixture (GMix or child class) representing
            a starting guess for the algorithm
        sky_guess: number
            A guess at the sky value
        maxiter: number, optional
            The maximum number of iterations, default 100
        tol: number, optional
            The tolerance in the moments that implies convergence,
            default 1.e-6
        """

        if hasattr(self,'_gm'):
            del self._gm

        conf=self._make_conf()
        conf['tol'] = tol
        conf['maxiter'] = maxiter
        conf['sky_guess'] = sky_guess
        conf['counts'] = self._counts
        conf['pixel_scale'] = self._obs.jacobian.get_scale()

        gm = gmix_guess.copy()
        sums = self._make_sums(len(gm))

        flags=0
        try:
            numiter, fdiff = em_run(
                conf,
                self._obs.pixels,
                sums,
                gm.get_data(),
            )

            # we have mutated the _data elements, we want to make
            # sure the pars are propagated.  Make a new full gm
            pars=gm.get_full_pars()
            self._gm=GMix(pars=pars)

            if numiter >= maxiter:
                flags = EM_MAXITER

            result={
                'flags':flags,
                'numiter':numiter,
                'fdiff':fdiff,
                'message':'OK',
            }

        except (GMixRangeError,ZeroDivisionError) as err:
            # most likely the algorithm reached an invalid gaussian
            message = str(err)
            print(message)
            result={
                'flags':EM_RANGE_ERROR,
                'message': message,
            }

        self._result = result

    run_em=go

    def _make_sums(self, ngauss):
        """
        make the sum structure
        """
        return numpy.zeros(ngauss, dtype=_sums_dtype)

    def _make_conf(self):
        """
        make the sum structure
        """
        conf_arr = numpy.zeros(1, dtype=_em_conf_dtype)
        return conf_arr[0]



_sums_dtype=[
    ('gi','f8'),

    # scratch on a given pixel
    ('trowsum','f8'),
    ('tcolsum','f8'),
    ('tu2sum','f8'),
    ('tuvsum','f8'),
    ('tv2sum','f8'),

    # sums over all pixels
    ('pnew','f8'),
    ('rowsum','f8'),
    ('colsum','f8'),
    ('u2sum','f8'),
    ('uvsum','f8'),
    ('v2sum','f8'),
]
_sums_dtype=numpy.dtype(_sums_dtype,align=True)

_em_conf_dtype=[
    ('tol','f8'),
    ('maxiter','i4'),
    ('sky_guess','f8'),
    ('counts','f8'),
    ('pixel_scale','f8'),
]
_em_conf_dtype=numpy.dtype(_em_conf_dtype,align=True)

def test_1gauss(counts=1.0,
                noise=0.0,
                T=4.0,
                maxiter=4000,
                g1=0.0,
                g2=0.0,
                show=False,
                pad=False,
                verbose=True,
                seed=31415):
    import time

    rng=numpy.random.RandomState(seed)

    sigma=numpy.sqrt(T/2)
    dim=int(2*5*sigma)
    dims=[dim]*2
    cen=[dims[0]/2., dims[1]/2.]

    jacob=UnitJacobian(row=cen[0], col=cen[1])

    pars = [0.0, 0.0, g1, g2, T, counts]
    gm=gmix.GMixModel(pars, "gauss")

    im0=gm.make_image(dims, jacobian=jacob)

    im = im0 + rng.normal(size=im0.shape, scale=noise)

    imsky,sky = prep_image(im)

    obs=Observation(imsky, jacobian=jacob)

    guess_pars = [
        srandu(rng=rng),
        srandu(rng=rng),
        0.05*srandu(rng=rng),
        0.05*srandu(rng=rng),
        T*(1.0 + 0.1*srandu(rng=rng)),
        counts*(1.0 + 0.1*srandu(rng=rng)),
    ]
    gm_guess= gmix.GMixModel(guess_pars, "gauss")

    print("gm:",gm)
    print("gm_guess:",gm_guess)

    # twice, first time numba compiles the code
    for i in xrange(2):
        tm0=time.time()
        em=GMixEM(obs)
        em.go(gm_guess, sky, maxiter=maxiter)
        tm=time.time()-tm0

    gmfit=em.get_gmix()
    res=em.get_result()

    if verbose:
        print("dims:",dims)
        print("cen:",cen)
        print('guess:')
        print(gm_guess)

        print('time:',tm,'seconds')
        print()

        print()
        print('results')
        print(res)

        print()
        print('gmix true:')
        print(gm)
        print('best fit:')
        print(gmfit)

    if show:
        import images
        imfit=gmfit.make_image(im.shape)
        imfit *= (im0.sum()/imfit.sum())

        images.compare_images(im, imfit)

    return gmfit

def test_1gauss_T_recovery(noise,
                           T=8.0,
                           counts=1.0,
                           ntrial=100,
                           show=True,
                           png=None):
    import biggles

    T_true=T

    T_meas=numpy.zeros(ntrial)
    for i in xrange(ntrial):
        while True:
            try:
                gm=test_1gauss(
                    noise=noise,
                    T=T_true,
                    counts=counts,
                    verbose=False,
                )
                T=gm.get_T()
                T_meas[i]=T
                break
            except GMixRangeError:
                pass
            except GMixMaxIterEM:
                pass

    mean=T_meas.mean()
    std=T_meas.std()
    print("<T>:",mean,"sigma(T):",std)
    binsize=0.2*std
    plt=biggles.plot_hist(T_meas, binsize=binsize, visible=False)
    p=biggles.Point(T_true, 0.0, type='filled circle', size=2, color='red')
    plt.add(p)
    plt.title='Flux: %g T: %g noise: %g' % (counts, T_true, noise)

    xmin=mean-4.0*std
    xmax=mean+4.0*std

    plt.xrange=[xmin, xmax]

    if show:
        plt.show()

    if png is not None:
        print(png)
        plt.write_img(800, 800, png)

def test_1gauss_jacob(counts_sky=100.0,
                      noise_sky=0.0,
                      maxiter=100,
                      show=False):
    import time
    #import images
    dims=[25,25]
    cen=[dims[0]/2., dims[1]/2.]

    #j1,j2,j3,j4=0.26,0.02,-0.03,0.23
    dvdrow,dvdcol,dudrow,dudcol=-0.04,-0.915,1.10,0.12
    j=Jacobian(row=cen[0],
               col=cen[1],
               dvdrow=dvdrow,
               dvdcol=dvdcol,
               dudrow=dudrow,
               dudcol=dudcol)

    jfac=j.get_scale()

    g1=0.1
    g2=0.05

    Tsky=8.0*jfac**2
    noise_pix=noise_sky/jfac**2

    pars = [0.0, 0.0, g1, g2, Tsky, counts_sky]
    gm=gmix.GMixModel(pars, "gauss")
    print('gmix true:')
    print(gm)

    im0=gm.make_image(dims, jacobian=j)

    im = im0 + noise_pix*numpy.random.randn(im0.size).reshape(dims)

    imsky,sky = prep_image(im)

    obs=Observation(imsky, jacobian=j)

    gm_guess=gm.copy()
    gm_guess._data['p']=1.0
    gm_guess._data['row'] += srandu()
    gm_guess._data['col'] += srandu()
    gm_guess._data['irr'] += srandu()
    gm_guess._data['irc'] += srandu()
    gm_guess._data['icc'] += srandu()

    print('guess:')
    print(gm_guess)

    tm0=time.time()
    em=GMixEM(obs)
    em.go(gm_guess, sky, maxiter=maxiter)
    tm=time.time()-tm0
    print('time:',tm,'seconds')

    gmfit=em.get_gmix()
    res=em.get_result()
    print('best fit:')
    print(gmfit)
    print('results')
    print(res)

    if show:
        import images
        imfit=gmfit.make_image(im.shape, jacobian=j)
        imfit *= (im0.sum()/imfit.sum())

        images.compare_images(im, imfit)

    return gmfit

def test_2gauss(counts=100.0, noise=0.0, maxiter=100,show=False):
    import time
    dims=[25,25]

    cen=(numpy.array(dims)-1.0)/2.0
    jacob=UnitJacobian(row=cen[0], col=cen[1])

    cen1=[ -3.25, -3.25]
    cen2=[ 3.0, 0.5]

    e1_1=0.1
    e2_1=0.05
    T_1=8.0
    counts_1=0.4*counts
    irr_1 = T_1/2.*(1-e1_1)
    irc_1 = T_1/2.*e2_1
    icc_1 = T_1/2.*(1+e1_1)

    e1_2=-0.2
    e2_2=-0.1
    T_2=4.0
    counts_2=0.6*counts
    irr_2 = T_2/2.*(1-e1_2)
    irc_2 = T_2/2.*e2_2
    icc_2 = T_2/2.*(1+e1_2)

    pars = [counts_1, cen1[0],cen1[1], irr_1, irc_1, icc_1,
            counts_2, cen2[0],cen2[1], irr_2, irc_2, icc_2]

    gm=gmix.GMix(pars=pars)
    print('gmix true:')
    print(gm)

    im0=gm.make_image(dims, jacobian=jacob)
    im = im0 + noise*numpy.random.randn(im0.size).reshape(dims)

    imsky,sky = prep_image(im)

    obs=Observation(imsky, jacobian=jacob)

    gm_guess=gm.copy()
    gm_guess._data['p']=[0.5,0.5]
    gm_guess._data['row'] += 4*srandu(2)
    gm_guess._data['col'] += 4*srandu(2)
    gm_guess._data['irr'] += 0.5*srandu(2)
    gm_guess._data['irc'] += 0.5*srandu(2)
    gm_guess._data['icc'] += 0.5*srandu(2)

    print('guess:')
    print(gm_guess)

    for i in xrange(2):
        tm0=time.time()
        em=GMixEM(obs)
        em.go(gm_guess, sky, maxiter=maxiter)
        tm=time.time()-tm0
    print('time:',tm,'seconds')

    gmfit=em.get_gmix()
    res=em.get_result()
    print('best fit:')
    print(gmfit)
    print('results')
    print(res)

    if show:
        import images
        imfit=gmfit.make_image(im.shape)
        imfit *= (im0.sum()/imfit.sum())

        images.compare_images(im, imfit)

    return tm
