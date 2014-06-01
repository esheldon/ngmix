"""
Fit an image with a gaussian mixture using the EM algorithm
"""
from __future__ import print_function

import numpy
import numba
from numba import jit, autojit, float64, int64

from . import gmix
from .gmix import GMix

from . import _gmix

from .gexceptions import GMixRangeError, GMixMaxIterEM
from .priors import srandu

from .jacobian import Jacobian, _jacobian

from .observation import Observation

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

        self._gm        = gmix_guess.copy()
        self._ngauss    = len(self._gm)
        self._sums      = numpy.zeros(self._ngauss, dtype=_sums_dtype)
        self._sky_guess = sky_guess
        self._maxiter   = maxiter
        self._tol       = tol

        # will raise GMixRangeError, but not GMixMaxIterEM, which
        # we handle below
        numiter, fdiff = _gmix.em_run(self._gm._data,
                                      self._obs.image,
                                      self._obs.jacobian._data,
                                      self._sums,
                                      numpy.float64(self._sky_guess),
                                      self._counts,
                                      self._tol,
                                      self._maxiter)

        self._result={'numiter':numiter,
                      'fdiff':fdiff}

        if numiter >= maxiter:
            raise GMixMaxIterEM("reached max iter: %s" % maxiter)
    # alias
    run_em=go

    def go_old(self, gmix_guess, sky_guess, maxiter=100, tol=1.e-6):
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

        self._gm        = gmix_guess.copy()
        self._ngauss    = len(self._gm)
        self._sums      = numpy.zeros(self._ngauss, dtype=_sums_dtype)
        self._sky_guess = sky_guess
        self._maxiter   = maxiter
        self._tol       = tol

        try:
            numiter, fdiff = _run_em(self._obs.image,
                                     self._gm._data,
                                     self._sums,
                                     self._obs.jacobian._data,
                                     numpy.float64(self._sky_guess),
                                     numpy.int64(self._maxiter),
                                     numpy.float64(self._tol))
                                     #_exp3_ivals[0],
                                     #_exp3_lookup)
        except ZeroDivisionError:
            raise GMixRangeError("divide by zero")

        self._result={'numiter':numiter,
                      'fdiff':fdiff}

        if numiter >= maxiter:
            raise GMixMaxIterEM("reached max iter: %s" % maxiter)

_sums=numba.struct([('gi',float64),
                    # scratch on a given pixel
                    ('trowsum',float64),
                    ('tcolsum',float64),
                    ('tu2sum',float64),
                    ('tuvsum',float64),
                    ('tv2sum',float64),
                    # sums over all pixels
                    ('pnew',float64),
                    ('rowsum',float64),
                    ('colsum',float64),
                    ('u2sum',float64),
                    ('uvsum',float64),
                    ('v2sum',float64)])

_sums_dtype=_sums.get_dtype()


'''
@autojit
def _clear_sums(sums):
    ngauss=sums.size
    for i in xrange(ngauss):
        sums[i].gi=0
        sums[i].trowsum=0
        sums[i].tcolsum=0
        sums[i].tu2sum=0
        sums[i].tuvsum=0
        sums[i].tv2sum=0
        sums[i].pnew=0
        sums[i].rowsum=0
        sums[i].colsum=0
        sums[i].u2sum=0
        sums[i].uvsum=0
        sums[i].v2sum=0

# have to send whole array
@jit(argtypes=[_gauss2d[:], int64, float64, float64, float64, float64, float64, float64])
def _gauss2d_set(self, i, p, row, col, irr, irc, icc):

    det = irr*icc - irc*irc
    if det < 1.0e-200:
        raise GMixRangeError("found det <= 0: %s" % det)

    self[i].p=p
    self[i].row=row
    self[i].col=col
    self[i].irr=irr
    self[i].irc=irc
    self[i].icc=icc

    self[i].det = det

    idet=1.0/det
    self[i].drr = irr*idet
    self[i].drc = irc*idet
    self[i].dcc = icc*idet
    self[i].norm = 1./(2*numpy.pi*numpy.sqrt(det))

    self[i].pnorm = self[i].p*self[i].norm

@autojit
def _set_gmix_from_sums(gmix, sums):
    ngauss=gmix.size 
    for i in xrange(ngauss):
        p=sums[i].pnew
        _gauss2d_set(gmix,
                     i,
                     p,
                     sums[i].rowsum/p,
                     sums[i].colsum/p,
                     sums[i].u2sum/p,
                     sums[i].uvsum/p,
                     sums[i].v2sum/p)

@autojit
def _gauss2d_verify(self):
    ngauss=self.size
    for i in xrange(ngauss):
        if self[i].det < 1.0e-200:
            raise GMixRangeError("det <= 0: %s" % self[i].det)

@jit(argtypes=[ _gauss2d[:] ])
def _get_wmomsum(self):
    ngauss=self.size
    wmom=0.0
    for i in xrange(ngauss):
        wmom += self[i].p*(self[i].irr + self[i].icc)
    return wmom


#@jit(argtypes=[float64[:,:],_gauss2d[:],_sums[:],_jacobian[:],float64,int64,float64,int64,float64[:]],
#     locals=dict(psum=float64, skysum=float64))
#def _run_em(image, gmix, sums, j, sky, maxiter, tol, i0, expvals):
@jit(argtypes=[float64[:,:],_gauss2d[:],_sums[:],_jacobian[:],float64,int64,float64],
     locals=dict(psum=float64, skysum=float64))
def _run_em(image, gmix, sums, j, sky, maxiter, tol):
    """
    this is a mess until we get inlining in numba
    """
    nrows,ncols=image.shape
    counts=numpy.sum(image)

    ngauss=gmix.size
    scale=j[0].sdet
    npoints=image.size
    area = npoints*scale*scale

    nsky = sky/counts
    psky = sky/(counts/area)

    wmomlast=-9999.0
    fdiff=9999.0

    iiter=0
    while iiter < maxiter:
        #_gauss2d_verify(gmix)

        psum=0.0
        skysum=0.0
        _clear_sums(sums)


        for row in xrange(nrows):
            u=j[0].dudrow*(row - j[0].row0) + j[0].dudcol*(0 - j[0].col0)
            v=j[0].dvdrow*(row - j[0].row0) + j[0].dvdcol*(0 - j[0].col0)
            for col in xrange(ncols):
                
                imnorm = image[row,col]/counts

                gtot=0.0
                for i in xrange(ngauss):
                    udiff = u-gmix[i].row
                    vdiff = v-gmix[i].col

                    u2 = udiff*udiff
                    v2 = vdiff*vdiff
                    uv = udiff*vdiff

                    chi2=gmix[i].dcc*u2 + gmix[i].drr*v2 - 2.0*gmix[i].drc*uv
                    sums[i].gi = gmix[i].norm*gmix[i].p*numpy.exp( -0.5*chi2 )
                    # note a bigger range is needed than for rendering since we
                    # need to sample the space
                    """
                    if chi2 < 50.0 and chi2 >= 0.0:
                        pnorm = gmix[i].pnorm
                        x = -0.5*chi2

                        # 3rd order approximation to exp
                        ival = int64(x-0.5)
                        f = x - ival
                        index = ival-i0
                        
                        expval = expvals[index]
                        fexp = (6+f*(6+f*(3+f)))*0.16666666
                        expval *= fexp

                        sums[i].gi = pnorm*expval
                    """
                    gtot += sums[i].gi

                    sums[i].trowsum = u*sums[i].gi
                    sums[i].tcolsum = v*sums[i].gi
                    sums[i].tu2sum  = u2*sums[i].gi
                    sums[i].tuvsum  = uv*sums[i].gi
                    sums[i].tv2sum  = v2*sums[i].gi

                gtot += nsky
                igrat = imnorm/gtot
                for i in xrange(ngauss):
                    # wtau is gi[pix]/gtot[pix]*imnorm[pix]
                    # which is Dave's tau*imnorm = wtau
                    wtau = sums[i].gi*igrat

                    psum += wtau
                    sums[i].pnew += wtau

                    # row*gi/gtot*imnorm
                    sums[i].rowsum += sums[i].trowsum*igrat
                    sums[i].colsum += sums[i].tcolsum*igrat
                    sums[i].u2sum  += sums[i].tu2sum*igrat
                    sums[i].uvsum  += sums[i].tuvsum*igrat
                    sums[i].v2sum  += sums[i].tv2sum*igrat

                skysum += nsky*imnorm/gtot
                u += j[0].dudcol
                v += j[0].dvdcol

        _set_gmix_from_sums(gmix, sums)

        psky = skysum
        nsky = psky/area

        wmom = _get_wmomsum(gmix)
        wmom /= psum
        fdiff = numpy.abs((wmom-wmomlast)/wmom)

        if fdiff < tol:
            break

        wmomlast = wmom
        iiter += 1

    return iiter, fdiff
'''

def test_1gauss(counts=100.0, noise=0.0, maxiter=100, show=False):
    import time
    dims=[25,25]
    cen=[dims[0]/2., dims[1]/2.]

    g1=0.1
    g2=0.05
    T=8.0

    pars = [cen[0],cen[1], g1, g2, T, counts]
    gm=gmix.GMixModel(pars, "gauss")
    print('gmix true:')
    print(gm)

    im0=gm.make_image(dims)

    im = im0 + noise*numpy.random.randn(im0.size).reshape(dims)

    imsky,sky = prep_image(im) 

    obs=Observation(imsky)

    gm_guess=gm.copy()
    gm_guess._data['p']=1.0
    gm_guess._data['row'] += 1*srandu()
    gm_guess._data['col'] += 1*srandu()
    gm_guess._data['irr'] += 0.5*srandu()
    gm_guess._data['irc'] += 0.5*srandu()
    gm_guess._data['icc'] += 0.5*srandu()

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
        imfit=gmfit.make_image(im.shape)
        imfit *= (im0.sum()/imfit.sum())

        images.compare_images(im, imfit)

def test_1gauss_jacob(counts_sky=100.0, noise_sky=0.0, maxiter=100, jfac=0.27, show=False):
    import time
    #import images
    dims=[25,25]
    cen=[dims[0]/2., dims[1]/2.]

    j=Jacobian(cen[0],cen[1], jfac, jfac*0.1, jfac*0.1, jfac)

    g1=0.1
    g2=0.05
    Tpix=8.0
    Tsky=8.0*jfac**2
    counts_pix=counts_sky/jfac**2
    noise_pix=noise_sky/jfac**2

    pars = [0.0, 0.0, g1, g2, Tsky, counts_sky]
    gm=gmix.GMixModel(pars, "gauss")
    print('gmix true:')
    print(gm)

    im0=gm.make_image(dims, jacobian=j)
    #images.view(im0)

    im = im0 + noise_pix*numpy.random.randn(im0.size).reshape(dims)

    imsky,sky = prep_image(im) 

    obs=Observation(imsky, jacobian=j)

    gm_guess=gm.copy()
    gm_guess._data['p']=1.0
    gm_guess._data['row'] += 1*srandu()
    gm_guess._data['col'] += 1*srandu()
    gm_guess._data['irr'] += 0.5*srandu()
    gm_guess._data['irc'] += 0.5*srandu()
    gm_guess._data['icc'] += 0.5*srandu()

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


def test_2gauss(counts=100.0, noise=0.0, maxiter=100,show=False):
    import time
    dims=[25,25]
    cen1=[ 0.35*dims[0], 0.35*dims[1] ]
    cen2=[ 0.6*dims[0], 0.5*dims[1] ]

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

    im0=gm.make_image(dims)
    im = im0 + noise*numpy.random.randn(im0.size).reshape(dims)

    imsky,sky = prep_image(im) 

    obs=Observation(imsky)

    gm_guess=gm.copy()
    gm_guess._data['p']=[0.5,0.5]
    gm_guess._data['row'] += 4*srandu(2)
    gm_guess._data['col'] += 4*srandu(2)
    gm_guess._data['irr'] += 0.5*srandu(2)
    gm_guess._data['irc'] += 0.5*srandu(2)
    gm_guess._data['icc'] += 0.5*srandu(2)

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
        imfit=gmfit.make_image(im.shape)
        imfit *= (im0.sum()/imfit.sum())

        images.compare_images(im, imfit)

    return tm
