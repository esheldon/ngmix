from copy import copy
import numpy
import numba
from numba import autojit, float64
from . import gmix
from .gmix import GMix, _gauss2d_set, _gauss2d, _get_wmomsum, _gauss2d_verify
from .gexceptions import GMixRangeError, GMixMaxIterEM
from .priors import srandu

class GMixEM(object):
    """
    Fit an image with a gaussian mixture using the EM algorithm

    todo: jacobian
    """
    def __init__(self, image, sky_guess, gmix_guess, jacobian=None):

        self._image=numpy.array(image, dtype='f8', copy=False)
        self._sky_guess=sky_guess

        self._gm=gmix_guess.copy()
        self._ngauss=len(self._gm)
        self._sums = numpy.zeros(self._ngauss, dtype=_sums_dtype)
        self._result=None

    def get_gmix(self):
        return self._gm
    def get_result(self):
        return self._result

    def go(self, maxiter, tol=1.e-6):
        numiter, fdiff = _run_em(self._image,
                                 self._gm._data,
                                 self._sums,
                                 numpy.float64(self._sky_guess),
                                 numpy.int64(maxiter),
                                 numpy.float64(tol))

        self._result={'numiter':numiter,
                      'fdiff':fdiff}

        if numiter == maxiter:
            raise GMixMaxIterEM("reached max iter: %s" % maxiter)


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

@autojit(locals=dict(psum=float64, skysum=float64))
def _run_em(image, gmix, sums, sky, maxiter, tol):
    """
    this is a mess until we get inlining in numba
    """
    nrows,ncols=image.shape
    counts=numpy.sum(image)

    ngauss=gmix.size
    scale=1.0
    npoints=image.size
    area = npoints*scale*scale

    nsky = sky/counts
    psky = sky/(counts/area)

    wmomlast=-9999.0
    fdiff=9999.0

    iiter=0
    while iiter < maxiter:
        _gauss2d_verify(gmix)

        psum=0.0
        skysum=0.0
        _clear_sums(sums)

        for row in xrange(nrows):
            for col in xrange(ncols):
                
                imnorm = image[row,col]/counts

                gtot=0.0
                for i in xrange(ngauss):
                    u = row-gmix[i].row
                    v = col-gmix[i].col

                    u2 = u*u
                    v2 = v*v
                    uv = u*v

                    chi2=gmix[i].dcc*u2 + gmix[i].drr*v2 - 2.0*gmix[i].drc*uv
                    sums[i].gi = gmix[i].norm*gmix[i].p*numpy.exp( -0.5*chi2 )

                    gtot += sums[i].gi

                    sums[i].trowsum = row*sums[i].gi
                    sums[i].tcolsum = col*sums[i].gi
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

    numiter=iiter+1
    return numiter, fdiff


def _prep_image(im0):

    im=im0.copy()

    # need no zero pixels and sky value
    im_min = im.min()
    im_max = im.max()
    sky=0.001*(im_max-im_min)

    im += (sky-im_min)

    return im, sky

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

def test_1gauss(counts=100.0, noise=0.0, maxiter=5000):

    dims=[25,25]
    cen=[dims[0]/2., dims[1]/2.]

    g1=0.1
    g2=0.05
    T=8.0

    pars = [cen[0],cen[1], g1, g2, T, counts]
    gm=gmix.GMixModel(pars, "gauss")
    print 'gmix true:'
    print gm

    im0=gm.make_image(dims)

    im0[:,:] += noise*numpy.random.randn(im0.size).reshape(dims)

    im,sky = _prep_image(im0) 

    gm_guess=gm.copy()
    gm_guess._data['p']=1.0
    gm_guess._data['row'] += 1*srandu()
    gm_guess._data['col'] += 1*srandu()
    gm_guess._data['irr'] += 0.5*srandu()
    gm_guess._data['irc'] += 0.5*srandu()
    gm_guess._data['icc'] += 0.5*srandu()

    print 'guess:'
    print gm_guess

    #em=GMixEMPy(im, sky)
    em=GMixEM(im, sky, gm_guess)
    em.go(maxiter)

    gmfit=em.get_gmix()
    res=em.get_result()
    print 'best fit:'
    print gmfit
    print 'results'
    print res

def test_2gauss(counts=100.0, noise=0.0, maxiter=5000,show=False):
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
    print 'gmix true:'
    print gm

    im0=gm.make_image(dims)
    im = im0 + noise*numpy.random.randn(im0.size).reshape(dims)

    imsky,sky = _prep_image(im) 

    gm_guess=gm.copy()
    gm_guess._data['p']=[0.5,0.5]
    gm_guess._data['row'] += 4*srandu(2)
    gm_guess._data['col'] += 4*srandu(2)
    gm_guess._data['irr'] += 0.5*srandu(2)
    gm_guess._data['irc'] += 0.5*srandu(2)
    gm_guess._data['icc'] += 0.5*srandu(2)

    print 'guess:'
    print gm_guess

    tm0=time.time()
    em=GMixEM(imsky, sky, gm_guess)
    em.go(maxiter)
    print 'time:',time.time()-tm0,'seconds'

    gmfit=em.get_gmix()
    res=em.get_result()
    print 'best fit:'
    print gmfit
    print 'results'
    print res

    if show:
        import images
        imfit=gmfit.make_image(im.shape)
        imfit *= (im0.sum()/imfit.sum())

        images.compare_images(im, imfit)
