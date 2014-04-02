from __future__ import print_function
import numpy
from numpy import array, zeros, exp, log10, log, dot, sqrt
import numba
from numba import float64, int64, autojit, jit
from . import fastmath
from .jacobian import Jacobian, _jacobian
from .shape import g1g2_to_e1e2, e1e2_to_g1g2

from .gexceptions import GMixRangeError, GMixFatalError

def make_gmix_model(pars, model):
    """
    get a gaussian mixture model for the given model
    """
    model_num=get_model_num(model)
    if model==GMIX_COELLIP:
        return GMixCoellip(pars)
    else:
        return GMixModel(pars, model)

class GMix(object):
    """
    A two-dimensional gaussian mixture.

    To create a specific model, use GMixModel

    parameters
    ----------
    Send either ngauss= or pars=

    ngauss: number, optional
        number of gaussians.  data will be zeroed
    pars: array-like, optional
        6*ngauss elements to fill the gaussian mixture.

    methods
    -------
    copy(self):
        make a new copy of this GMix
    convolve(psf):
        Get a new GMix that is the convolution of the GMix with the input psf
    get_T():
        get T=sum(p*T_i)/sum(p)
    get_psum():
        get sum(p)
    set_psum(psum):
        set new overall sum(p)
    get_cen():
        get cen=sum(p*cen_i)/sum(p)
    set_cen(row,col):
        set the overall center to the input.
    """
    def __init__(self, ngauss=None, pars=None):

        if ngauss is None and pars is None:
            raise GMixFatalError("send ngauss= or pars=")

        if pars is not None:
            npars = len(pars)
            if (npars % 6) != 0:
                raise GMixFatalError("len(pars) must be mutiple of 6 "
                                     "got %s" % npars)
            self._ngauss=npars/6
            self.reset()
            self.fill(pars)
        else:
            self._ngauss=ngauss
            self.reset()

    def get_data(self):
        """
        Get the underlying array
        """
        return self._data

    def get_full_pars(self):
        """
        Get a full parameter description.
           [p1,row1,col1,irr1,irc1,icc1,
            p2,row2,col2,irr2,irc2,icc2,
            ...
           ]

        """
        n=self._ngauss
        pars=numpy.zeros(n*6)
        beg=0
        for i in xrange(n):
            pars[beg+0] = self._data['p'][i]
            pars[beg+1] = self._data['row'][i]
            pars[beg+2] = self._data['col'][i]
            pars[beg+3] = self._data['irr'][i]
            pars[beg+4] = self._data['irc'][i]
            pars[beg+5] = self._data['icc'][i]
            
            beg += 6
        return pars

    def get_cen(self):
        """
        get the center position (row,col)
        """
        row,col,psum=_get_cen(self._data)
        return row,col
    
    def set_cen(self, row, col):
        """
        Move the mixture to a new center
        """
        _set_cen(self._data, row, col)

    def get_T(self):
        """
        get weighted average T sum(p*T)/sum(p)
        """
        T,psum=_get_T(self._data)
        return T

    def get_e1e2T(self):
        """
        Get e1,e2 and T for the total gmix.

        Warning: only really works if the centers are the same
        """
        e1,e2,T=_get_e1e2T(self._data)
        return e1,e2,T

    def get_g1g2T(self):
        """
        Get g1,g2 and T for the total gmix.

        Warning: only really works if the centers are the same
        """
        e1,e2,T=_get_e1e2T(self._data)
        g1,g2=e1e2_to_g1g2(e1,e2)
        return g1,g2,T

    def get_psum(self):
        """
        get sum(p)
        """
        return self._data['p'].sum()

    def set_psum(self, psum):
        """
        set a new value for sum(p)
        """
        psum0 = self._data['p'].sum()
        rat = psum/psum0
        self._data['p'] *= rat
        self._data['pnorm'] = self._data['p']*self._data['norm']

    def fill(self, pars):
        """
        fill the gaussian mixture from a 'full' parameter array.

        The length must match the internal size

        parameters
        ----------
        pars: array-like
            [p1,row1,col1,irr1,irc1,icc1,
             p2,row2,col2,irr2,irc2,icc2,
             ...]

             Should have length 6*ngauss
        """
        parr=array(pars, dtype='f8', copy=False) 
        npars=parr.size
        npars_expected = self._data.size*6
        if npars != npars_expected:
            raise GMixFatalError("expected len(pars)=%d but "
                                 "got %d" % (npars_expected,npars))
        
        _fill_full(self._data, parr)

    def copy(self):
        """
        Get a new GMix with the same parameters
        """
        gmix = GMix(self._ngauss)
        gmix._data[:] = self._data[:]
        return gmix

    def convolve(self, psf):
        """
        Get a new GMix that is the convolution of the GMix with the input psf

        parameters
        ----------
        psf: GMix object
        """
        if not isinstance(psf, GMix):
            raise TypeError("Can only convolve with another GMix "
                            " got type %s" % type(psf))

        ng=len(self)*len(psf)
        gmix = GMix(ngauss=ng)
        convolve_fill(gmix, self, psf)
        return gmix

    def make_image(self, dims, nsub=1, jacobian=None):
        """
        Render the mixture into a new image

        parameters
        ----------
        dims: 2-element sequence
            dimensions [nrows, ncols]
        nsub: integer, optional
            Defines a grid for sub-pixel integration 
        """
        image=numpy.zeros(dims, dtype='f8')
        if jacobian is not None:
            _render_jacob_fast3(self._data,
                                image,
                                nsub,
                                jacobian._data,
                                _exp3_ivals[0],
                                _exp3_lookup)
        else:
            _render_fast3(self._data, image, nsub, _exp3_ivals[0], _exp3_lookup)

        return image

    def fill_image(self, image, nsub=1, jacobian=None):
        """
        Render the mixture into the input image

        parameters
        ----------
        image: 2-d array
            the image to fill
        nsub: integer, optional
            Defines a grid for sub-pixel integration 
        """
        if jacobian is not None:
            _render_jacob_fast3(self._data,
                                image,
                                nsub,
                                jacobian._data,
                                _exp3_ivals[0],
                                _exp3_lookup)
        else:
            _render_fast3(self._data, image, nsub, _exp3_ivals[0], _exp3_lookup)

    def get_loglike(self, image, weight, jacobian=None, get_s2nsums=False):
        """
        Calculate the log likelihood

        If the function calls and error checking are bottlenecks, make the
        calls to the _loglike* functions directly.  But looks to be only
        1%% slower

        parameters
        ----------
        image: 2-d array
            the image to fill
        weight: 2-d array
            the weight image
        """
        if image.size != weight.size:
            raise ValueError("image and weight must be same shape")

        if jacobian is not None:
            if not isinstance(jacobian,Jacobian):
                raise ValueError("jacobian must be instance of Jacobian")
            loglike,s2n_numer,s2n_denom=_loglike_jacob_fast3(self._data,
                                                             image,
                                                             weight,
                                                             jacobian._data,
                                                             _exp3_ivals[0],
                                                             _exp3_lookup)
        else:
            loglike,s2n_numer,s2n_denom=_loglike_fast3(self._data,
                                                       image,
                                                       weight,
                                                       _exp3_ivals[0],
                                                       _exp3_lookup)
        if get_s2nsums:
            return loglike,s2n_numer,s2n_denom
        else:
            return loglike

    def reset(self):
        """
        Replace the data array with a zeroed one.
        """
        self._data = zeros(self._ngauss, dtype=_gauss2d_dtype)


    def get_loglike(self, row, col):
        """
        Evaluate a single row, col
        """
        #return _gauss2d_loglike(self._data, _exp3_ivals[0], _exp3_lookup, row, col)
        return _gauss2d_loglike(self._data, row, col)
    def get_like(self, row, col):
        """
        Evaluate a single row, col
        """
        #return _gauss2d_ike(self._data, _exp3_ivals[0], _exp3_lookup, row, col)
        return _gauss2d_ike(self._data, row, col)


    def __len__(self):
        return self._ngauss

    def __repr__(self):
        rep=[]
        #fmt="p: %-10.5g row: %-10.5g col: %-10.5g irr: %-10.5g irc: %-10.5g icc: %-10.5g"
        fmt="p: %-12.8g row: %-12.8g col: %-12.8g irr: %-12.8g irc: %-12.8g icc: %-12.8g"
        for i in xrange(self._ngauss):
            t=self._data[i]
            s=fmt % (t['p'],t['row'],t['col'],t['irr'],t['irc'],t['icc'])
            rep.append(s)

        rep='\n'.join(rep)
        return rep


class GMixModel(GMix):
    """
    A two-dimensional gaussian mixture created from a set of model parameters

    Inherits from the more general GMix class, and all its methods.

    parameters
    ----------
    pars: array-like
        Parameter array. The number of elements will depend
        on the model type.
    model: string or gmix type
        e.g. 'exp' or GMIX_EXP
    """
    def __init__(self, pars, model):

        self._pars       = array(pars, dtype='f8', copy=False) 
        self._model      = _gmix_model_dict[model]
        self._model_name = _gmix_string_dict[self._model]

        if self._model==GMIX_FULL:
            super(GMixModel,self).__init__(pars=self._pars) 
        else:
            self._ngauss = _gmix_ngauss_dict[self._model]
            self._npars  = _gmix_npars_dict[self._model]
            self.reset()
            self.fill(self._pars)

    def fill(self, pars):
        """
        Fill in the gaussian mixture with new parameters
        """
        if self._model==GMIX_FULL:
            super(GMixModel,self).fill(pars)
        else:
            parr=array(pars, dtype='f8', copy=False)

            if parr.size != self._npars:
                err="model '%s' requires %s pars, got %s"
                err =err % (self._model_name,self._npars, parr.size)
                raise GMixFatalError(err)

            self._pars[:] = parr[:]

            if self._model==GMIX_GAUSS:
                _fill_gauss(self._data, self._pars)
            elif self._model==GMIX_EXP:
                _fill_exp(self._data, self._pars)
            elif self._model==GMIX_DEV:
                _fill_dev(self._data, self._pars)
            elif self._model==GMIX_TURB:
                _fill_turb(self._data, self._pars)
            elif self._model==GMIX_BDC:
                _fill_bdc(self._data, self._pars)
            elif self._model==GMIX_BDF:
                _fill_bdf(self._data, self._pars)
            else:
                raise GMixFatalError("unsupported model: "
                                     "'%s'" % self._model_name)



def get_coellip_npars(ngauss):
    return 4 + 2*ngauss

def get_coellip_ngauss(npars):
    return (npars-4)/2

class GMixCoellip(GMix):
    """
    A two-dimensional gaussian mixture, each co-centeric and co-elliptical

    Inherits from the more general GMix class, and all its methods.

    parameters
    ----------
    pars: array-like
        Parameter array. The number of elements will depend
        on the model type.
    """
    def __init__(self, pars):
        self._model      = GMIX_COELLIP
        self._model_name = 'coellip'
        self.fill(pars)

    def fill(self, parsin):
        """
        Fill in the gaussian mixture with new parameters
        """

        pars = array(parsin, dtype='f8', copy=True) 

        npars=pars.size

        ncheck=npars-4
        if ( ncheck % 2 ) != 0:
            raise ValueError("coellip must have len(pars)==4+2*ngauss, got %s" % npars)

        self._pars=pars
        self._ngauss = ncheck/2
        self._npars = npars

        self.reset()

        _fill_coellip(self._data, pars)


GMIX_FULL=0
GMIX_GAUSS=1
GMIX_TURB=2
GMIX_EXP=3
GMIX_DEV=4
GMIX_BDC=5
GMIX_BDF=6
GMIX_COELLIP=7
GMIX_COELLIP4=8

_gmix_model_dict={'full':       GMIX_FULL,
                  GMIX_FULL:    GMIX_FULL,
                  'gauss':      GMIX_GAUSS,
                  GMIX_GAUSS:   GMIX_GAUSS,
                  'turb':       GMIX_TURB,
                  GMIX_TURB:    GMIX_TURB,
                  'exp':        GMIX_EXP,
                  GMIX_EXP:     GMIX_EXP,
                  'dev':        GMIX_DEV,
                  GMIX_DEV:     GMIX_DEV,
                  'bdc':        GMIX_BDC,
                  GMIX_BDC:     GMIX_BDC,
                  'bdf':        GMIX_BDF,
                  GMIX_BDF:     GMIX_BDF,
                  'coellip':    GMIX_COELLIP,
                  GMIX_COELLIP: GMIX_COELLIP,
                  'coellip4':    GMIX_COELLIP4,
                  GMIX_COELLIP4: GMIX_COELLIP4}

_gmix_string_dict={GMIX_FULL:'full',
                   'full':'full',
                   GMIX_GAUSS:'gauss',
                   'gauss':'gauss',
                   GMIX_TURB:'turb',
                   'turb':'turb',
                   GMIX_EXP:'exp',
                   'exp':'exp',
                   GMIX_DEV:'dev',
                   'dev':'dev',
                   GMIX_BDC:'bdc',
                   'bdc':'bdc',
                   GMIX_BDF:'bdf',
                   'bdf':'bdf',
                   GMIX_COELLIP:'coellip',
                   'coellip':GMIX_COELLIP,
                   GMIX_COELLIP4:'coellip4',
                   'coellip4':GMIX_COELLIP4}

_gmix_npars_dict={GMIX_GAUSS:6,
                  GMIX_TURB:6,
                  GMIX_EXP:6,
                  GMIX_DEV:6,
                  GMIX_BDC:8,
                  GMIX_BDF:7,
                  GMIX_COELLIP4:12}
_gmix_ngauss_dict={GMIX_GAUSS:1,
                   GMIX_TURB:3,
                   GMIX_EXP:6,
                   GMIX_DEV:10,
                   GMIX_BDC:16,
                   GMIX_BDF:16,
                   GMIX_COELLIP4:4}


_gauss2d=numba.struct([('p',float64),
                       ('row',float64),
                       ('col',float64),
                       ('irr',float64),
                       ('irc',float64),
                       ('icc',float64),
                       ('det',float64),
                       ('drr',float64),
                       ('drc',float64),
                       ('dcc',float64),
                       ('norm',float64),
                       ('pnorm',float64)])

_gauss2d_dtype=_gauss2d.get_dtype()


def get_model_num(model):
    """
    Get the numerical identifier for the input model,
    which could be string or number
    """
    return _gmix_model_dict[model]
def get_model_name(model):
    """
    Get the string identifier for the input model,
    which could be string or number
    """
    return _gmix_string_dict[model]

def get_model_npars(model):
    """
    Get the number of parameters for the input model,
    which could be string or number
    """
    mi=_gmix_model_dict[model]
    return _gmix_npars_dict[mi]


@autojit
def _gauss2d_verify(self):
    ngauss=self.size
    for i in xrange(ngauss):
        if self[i].det <= 0:
            raise GMixRangeError("det <= 0: %s" % self[i].det)

# have to send whole array
@jit(argtypes=[_gauss2d[:], int64, float64, float64, float64, float64, float64, float64])
def _gauss2d_set(self, i, p, row, col, irr, irc, icc):

    det = irr*icc - irc*irc
    if det <= 0.0:
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

@jit(argtypes=[ _gauss2d[:], float64[:] ] )
def _fill_coellip(self, pars):

    npars=pars.size
    ngauss = (npars-4)/2;

    row=pars[0]
    col=pars[1]
    g1=pars[2]
    g2=pars[3]

    e1,e2 = g1g2_to_e1e2(g1, g2)

    ngauss=self.size
    for i in xrange(ngauss):

        T = pars[4+i]
        counts=pars[4+ngauss+i]

        _gauss2d_set(self,
                     i,
                     counts,
                     row,
                     col, 
                     (T/2.)*(1-e1), 
                     (T/2.)*e2,
                     (T/2.)*(1+e1))



@jit(argtypes=[ _gauss2d[:], float64[:], float64[:], float64[:] ] )
def _fill_simple(self, pars, fvals, pvals):
    row=pars[0]
    col=pars[1]
    g1=pars[2]
    g2=pars[3]
    T=pars[4]
    counts=pars[5]

    e1,e2 = g1g2_to_e1e2(g1, g2)

    ngauss=self.size
    for i in xrange(ngauss):

        T_i = T*fvals[i]
        counts_i=counts*pvals[i]

        _gauss2d_set(self,
                     i,
                     counts_i,
                     row,
                     col, 
                     (T_i/2.)*(1-e1), 
                     (T_i/2.)*e2,
                     (T_i/2.)*(1+e1))


_gauss_fvals = array([1.0],dtype='f8')
_gauss_pvals = array([1.0],dtype='f8')

@jit(argtypes=[ _gauss2d[:], float64[:] ] )
def _fill_gauss(self, pars):
    _fill_simple(self, pars, _gauss_fvals, _gauss_pvals)


_exp_pvals = array([0.00061601229677880041, 
                    0.0079461395724623237, 
                    0.053280454055540001, 
                    0.21797364640726541, 
                    0.45496740582554868, 
                    0.26521634184240478],dtype='f8')
_exp_fvals = array([0.002467115141477932, 
                    0.018147435573256168, 
                    0.07944063151366336, 
                    0.27137669897479122, 
                    0.79782256866993773, 
                    2.1623306025075739],dtype='f8')


@jit(argtypes=[ _gauss2d[:], float64[:] ] )
def _fill_exp(self, pars):
    _fill_simple(self, pars, _exp_fvals, _exp_pvals)


_dev_pvals = array([6.5288960012625658e-05, 
                    0.00044199216814302695, 
                    0.0020859587871659754, 
                    0.0075913681418996841, 
                    0.02260266219257237, 
                    0.056532254390212859, 
                    0.11939049233042602, 
                    0.20969545753234975, 
                    0.29254151133139222, 
                    0.28905301416582552],dtype='f8')
_dev_fvals = 1.025*array([2.9934935706271918e-07, 
                          3.4651596338231207e-06, 
                          2.4807910570562753e-05, 
                          0.00014307404300535354, 
                          0.000727531692982395, 
                          0.003458246439442726, 
                          0.0160866454407191, 
                          0.077006776775654429, 
                          0.41012562102501476, 
                          2.9812509778548648],dtype='f8')

@jit(argtypes=[ _gauss2d[:], float64[:] ] )
def _fill_dev(self, pars):
    _fill_simple(self, pars, _dev_fvals, _dev_pvals)


_tmp_bulge_pars=zeros(_gmix_npars_dict[GMIX_DEV])
_tmp_disk_pars=zeros(_gmix_npars_dict[GMIX_EXP])
_tmp_bulge_gmix=zeros(_gmix_ngauss_dict[GMIX_DEV],dtype=_gauss2d_dtype)
_tmp_disk_gmix=zeros(_gmix_ngauss_dict[GMIX_EXP],dtype=_gauss2d_dtype)

#@jit(argtypes=[ _gauss2d[:], float64[:] ] )
def _fill_bdc(self, pars):
    """
    Fill a bulge+disk model, co-centric and co-elliptical

    pars are [c1,c2,g1,g2,Tbulge,Tdisk,Fbulge,Fdisk]
    """
    _tmp_bulge_pars[0]=pars[0]
    _tmp_bulge_pars[1]=pars[1]
    _tmp_bulge_pars[2]=pars[2]
    _tmp_bulge_pars[3]=pars[3]
    _tmp_bulge_pars[4]=pars[4]
    _tmp_bulge_pars[5]=pars[6]

    _tmp_disk_pars[0]=pars[0]
    _tmp_disk_pars[1]=pars[1]
    _tmp_disk_pars[2]=pars[2]
    _tmp_disk_pars[3]=pars[3]
    _tmp_disk_pars[4]=pars[5]
    _tmp_disk_pars[5]=pars[7]

    _fill_dev(_tmp_bulge_gmix, _tmp_bulge_pars)
    _fill_exp(_tmp_disk_gmix,  _tmp_disk_pars)

    ng_bulge=_tmp_bulge_gmix.size
    #ng_disk=_tmp_disk_gmix.size

    #for i in xrange(ng_bulge):
    #    self[i] = _tmp_bulge_gmix[i]
    #for i in xrange(ng_disk):
    #    self[ng_bulge+i] = _tmp_disk_gmix[i]
    self[0:ng_bulge] = _tmp_bulge_gmix[:]
    self[ng_bulge:]  = _tmp_disk_gmix[:]

def _fill_bdf(self, pars):
    """
    Fill a bulge+disk model, co-centric and co-elliptical
    and with Tbulge=Tdisk

    pars are [c1,c2,g1,g2,T,Fbulge,Fdisk]
    """
    _tmp_bulge_pars[:]=pars[0:6]

    _tmp_disk_pars[:]=_tmp_bulge_pars[:]
    _tmp_disk_pars[5]=pars[6]

    _fill_dev(_tmp_bulge_gmix, _tmp_bulge_pars)
    _fill_exp(_tmp_disk_gmix,  _tmp_disk_pars)

    ng_bulge=_tmp_bulge_gmix.size

    self[0:ng_bulge] = _tmp_bulge_gmix[:]
    self[ng_bulge:]  = _tmp_disk_gmix[:]



_turb_fvals = array([0.5793612389470884,1.621860687127999,7.019347162356363],dtype='f8')
_turb_pvals = array([0.596510042804182,0.4034898268889178,1.303069003078001e-07],dtype='f8')

@jit(argtypes=[ _gauss2d[:], float64[:] ] )
def _fill_turb(self, pars):
    _fill_simple(self, pars, _turb_fvals, _turb_pvals)



@jit(argtypes=[ _gauss2d[:] ])
def _get_cen(self):
    row=0.0
    col=0.0
    psum=0.0

    ngauss=self.size
    for i in xrange(ngauss):
        p=self[i].p
        row += p*self[i].row
        col += p*self[i].col
        psum += p

    row /= psum
    col /= psum

    return row, col, psum

@jit(argtypes=[ _gauss2d[:], float64, float64 ])
def _set_cen(self, row, col):

    row_cur, col_cur, _ =_get_cen(self)
    row_shift = row - row_cur
    col_shift = col - col_cur

    ngauss=self.size
    for i in xrange(ngauss):
        self[i].row += row_shift
        self[i].col += col_shift

@jit(argtypes=[ _gauss2d[:] ])
def _get_T(self):
    T=0.0
    psum=0.0

    ngauss=self.size
    for i in xrange(ngauss):
        p=self[i].p
        T += (self[i].irr + self[i].icc)*p
        psum += p

    T /= psum

    return T, psum

@jit(argtypes=[ _gauss2d[:] ])
def _get_e1e2T(self):
    e1=-9999.
    e2=-9999.
    irr=0.0
    irc=0.0
    icc=0.0

    psum=0.0

    ngauss=self.size
    for i in xrange(ngauss):
        p=self[i].p

        irr += p*self[i].irr
        irc += p*self[i].irc
        icc += p*self[i].icc

        psum += p

    ipsum = 1.0/psum
    irr *= ipsum
    irc *= ipsum
    icc *= ipsum

    T = irr + icc
    
    if T > 0:
        e1 = (icc-irr)/T
        e2 = 2*irc/T

    return e1, e2, T


@jit(argtypes=[ _gauss2d[:] ])
def _get_wmomsum(self):
    ngauss=self.size
    wmom=0.0
    for i in xrange(ngauss):
        wmom += self[i].p*(self[i].irr + self[i].icc)
    return wmom


@jit(argtypes=[_gauss2d[:], float64[:]] )
def _fill_full(self, pars):

    ngauss=self.size

    for i in xrange(ngauss): 

        beg=i*6
        _gauss2d_set(self,
                     int64(i),
                     pars[beg+0],
                     pars[beg+1],
                     pars[beg+2],
                     pars[beg+3],
                     pars[beg+4],
                     pars[beg+5])

def convolve_fill(self, gmix, psf):
    """
    Fill "self" with gmix convolved with psf
    """
    ng=len(gmix)*len(psf)
    if ng != len(self):
        raise GMixFatalError("target gmix is wrong size, %d "
                             "instead of %d" % (len(gmix),ng))

    _convolve_fill(self._data, gmix._data, psf._data)

@jit(argtypes=[ _gauss2d[:], _gauss2d[:], _gauss2d[:] ])
def _convolve_fill(self, obj_gmix, psf_gmix):
    
    nobj=obj_gmix.size
    npsf=psf_gmix.size

    psf_rowcen,psf_colcen,psf_psum = _get_cen(psf_gmix)
    psf_ipsum=1.0/psf_psum

    iself=0
    for iobj in xrange(nobj):
        for ipsf in xrange(npsf):
            p = obj_gmix[iobj].p*psf_gmix[ipsf].p*psf_ipsum

            row = obj_gmix[iobj].row + (psf_gmix[ipsf].row-psf_rowcen)
            col = obj_gmix[iobj].col + (psf_gmix[ipsf].col-psf_colcen)

            irr = obj_gmix[iobj].irr + psf_gmix[ipsf].irr
            irc = obj_gmix[iobj].irc + psf_gmix[ipsf].irc
            icc = obj_gmix[iobj].icc + psf_gmix[ipsf].icc

            _gauss2d_set(self, iself, p, row, col, irr, irc, icc)

            iself += 1

@jit(argtypes=[ _gauss2d[:], float64[:,:], int64 ])
def _render_slow(self, image, nsub):
    """
    Adds to image; make sure to zero the iamge first if that is what you want
    """
    ngauss=self.size
    nrows=image.shape[0]
    ncols=image.shape[1]

    stepsize = 1./nsub
    offset = (nsub-1)*stepsize/2.
    areafac = 1./(nsub*nsub)

    for row in xrange(nrows):
        for col in xrange(ncols):

            # we add to existing value
            model_val=image[row,col]

            tval = 0.0
            trow = row-offset
            for irowsub in xrange(nsub):
                tcol = col-offset
                for icolsub in xrange(nsub):

                    for i in xrange(ngauss):
                        u = trow - self[i].row
                        u2 = u*u
                        v = tcol - self[i].col
                        v2 = v*v

                        uv=u*v

                        chi2=self[i].dcc*u2 + self[i].drr*v2 - 2.0*self[i].drc*uv

                        if chi2 < 25.0 and chi2 >= 0.0:
                            pnorm = self[i].pnorm
                            tval += pnorm*numpy.exp( -0.5*chi2 )
                    tcol += stepsize
                trow += stepsize

            tval *= areafac
            model_val += tval
            image[row,col] = model_val


#
# create the fast lookup table for exponentials

# for this demand chi2 < 25
_exp3_ivals, _exp3_lookup = fastmath.make_exp_lookup(-26, 0)
# for this demand chi2 < 99
#_exp3_ivals, _exp3_lookup = fastmath.make_exp_lookup(-100, 0)

@jit(argtypes=[ _gauss2d[:], float64[:,:], int64, int64, float64[:] ])
def _render_fast3(self, image, nsub, i0, expvals):
    """
    Adds to image; make sure to zero the iamge first if that is what you want

    Uses 3rd order approximation to exponential function, only for negative
    arguments or zero

    This code is a mess because we can't do inlining in numba
    """
    ngauss=self.size
    nrows=image.shape[0]
    ncols=image.shape[1]

    stepsize = 1./nsub
    offset = (nsub-1)*stepsize/2.
    areafac = 1./(nsub*nsub)

    for row in xrange(nrows):
        for col in xrange(ncols):

            # we add to existing value
            model_val=image[row,col]

            tval = 0.0
            trow = row-offset
            for irowsub in xrange(nsub):
                tcol = col-offset
                for icolsub in xrange(nsub):

                    for i in xrange(ngauss):
                        u = trow - self[i].row
                        u2 = u*u
                        v = tcol - self[i].col
                        v2 = v*v

                        uv=u*v

                        chi2=self[i].dcc*u2 + self[i].drr*v2 - 2.0*self[i].drc*uv

                        if chi2 < 25.0 and chi2 >= 0.0:
                            pnorm = self[i].pnorm
                            x = -0.5*chi2

                            # 3rd order approximation to exp
                            #if x < 0.0:
                            #    ival = int64(x-0.5)
                            #else:
                            #    ival = int64(x+0.5)
                            ival = int64(x-0.5)
                            f = x - ival
                            index = ival-i0

                            expval = expvals[index]
                            fexp = (6+f*(6+f*(3+f)))*0.16666666
                            expval *= fexp

                            tval += pnorm*expval

                    tcol += stepsize
                trow += stepsize

            tval *= areafac
            model_val += tval
            image[row,col] = model_val


# evaluate a single point
#@jit(argtypes=[ _gauss2d[:], int64, float64[:], float64, float64 ])
#def _gauss2d_like(self, i0, expvals, row, col):
@jit(argtypes=[ _gauss2d[:], float64, float64 ])
def _gauss2d_like(self, row, col):

    like = 0.0

    ngauss=self.size
    for i in xrange(ngauss):

        u = row - self[i].row
        u2 = u*u
        v = col - self[i].col
        v2 = v*v

        uv=u*v

        chi2=self[i].dcc*u2 + self[i].drr*v2 - 2.0*self[i].drc*uv

        pnorm = self[i].pnorm
        tlike = pnorm*numpy.exp(-0.5*chi2)

        like += tlike

    return like

# evaluate a single point
#@jit(argtypes=[ _gauss2d[:], int64, float64[:], float64, float64 ])
#def _gauss2d_loglike(self, i0, expvals, row, col):
@jit(argtypes=[ _gauss2d[:], float64, float64 ])
def _gauss2d_loglike(self, row, col):
    #like = _gauss2d_like(self, i0, expvals, row, col)
    like = _gauss2d_like(self, row, col)
    loglike = numpy.log( like )
    return loglike



@jit(argtypes=[ _gauss2d[:], float64[:,:], int64, _jacobian[:], int64, float64[:] ])
def _render_jacob_fast3(self, image, nsub, j, i0, expvals):
    """
    Adds to image; make sure to zero the iamge first if that is what you want

    Uses 3rd order approximation to exponential function, only for negative
    arguments or zero

    This code is a mess because we can't do inlining in numba
    """
    ngauss=self.size
    nrows=image.shape[0]
    ncols=image.shape[1]

    col0=j[0].col0
    row0=j[0].row0
    dudrow=j[0].dudrow
    dudcol=j[0].dudcol
    dvdrow=j[0].dvdrow
    dvdcol=j[0].dvdcol

    stepsize = 1./nsub
    offset = (nsub-1)*stepsize/2.
    areafac = 1./(nsub*nsub)

    ustepsize = stepsize*dudcol
    vstepsize = stepsize*dvdcol

    for row in xrange(nrows):
        for col in xrange(ncols):

            # we add to existing value
            model_val=image[row,col]

            tval = 0.0
            trow = row-offset
            lowcol = col-offset

            for irowsub in xrange(nsub):
                # always start from lowcol position, then step u,v later
                u=dudrow*(trow - row0) + dudcol*(lowcol - col0)
                v=dvdrow*(trow - row0) + dvdcol*(lowcol - col0)
                for icolsub in xrange(nsub):

                    for i in xrange(ngauss):
                        udiff=u-self[i].row
                        vdiff=v-self[i].col

                        u2 = udiff*udiff
                        v2 = vdiff*vdiff
                        uv=udiff*vdiff

                        chi2=self[i].dcc*u2 + self[i].drr*v2 - 2.0*self[i].drc*uv

                        if chi2 < 25.0 and chi2 >= 0.0:
                            pnorm = self[i].pnorm
                            x = -0.5*chi2

                            # 3rd order approximation to exp
                            #if x < 0.0:
                            #    ival = int64(x-0.5)
                            #else:
                            #    ival = int64(x+0.5)
                            ival = int64(x-0.5)
                            f = x - ival
                            index = ival-i0

                            expval = expvals[index]
                            fexp = (6+f*(6+f*(3+f)))*0.16666666
                            expval *= fexp

                            tval += pnorm*expval

                    # move u and v for each "column" step
                    u += ustepsize
                    v += vstepsize

                # step to next sub-row
                trow += stepsize

            tval *= areafac
            model_val += tval
            image[row,col] = model_val


@jit(argtypes=[ _gauss2d[:], float64[:,:], float64[:,:], int64, float64[:] ])
def _loglike_fast3(self, image, weight, i0, expvals):
    """
    using 3rd order approximation to the exponential function

    This code is a mess because we can't do inlining in numba
    """
    ngauss=self.size
    nrows=image.shape[0]
    ncols=image.shape[1]

    s2n_numer=0.0
    s2n_denom=0.0
    loglike = 0.0
    for row in xrange(nrows):
        for col in xrange(ncols):

            ivar = weight[row,col]
            if ivar <= 0.0:
                continue

            model_val=0.0
            for i in xrange(ngauss):
                u = row - self[i].row
                u2 = u*u
                v = col - self[i].col
                v2 = v*v

                uv=u*v

                chi2=self[i].dcc*u2 + self[i].drr*v2 - 2.0*self[i].drc*uv

                if chi2 < 25.0 and chi2 >= 0.0:
                    pnorm = self[i].pnorm
                    x = -0.5*chi2

                    # 3rd order approximation to exp
                    #if x < 0.0:
                    #    ival = int64(x-0.5)
                    #else:
                    #    ival = int64(x+0.5)
                    ival = int64(x-0.5)
                    f = x - ival
                    index = ival-i0

                    expval = expvals[index]
                    fexp = (6+f*(6+f*(3+f)))*0.16666666
                    expval *= fexp

                    model_val += pnorm*expval

            pixval = image[row,col]
            diff = model_val-pixval
            loglike += diff*diff*ivar
            s2n_numer += pixval*model_val*ivar
            s2n_denom += model_val*model_val*ivar

    loglike *= (-0.5)

    return loglike, s2n_numer, s2n_denom

#@autojit
@jit(argtypes=[ _gauss2d[:], float64[:,:], float64[:,:], _jacobian[:], int64, float64[:] ])
def _loglike_jacob_fast3(self, image, weight, j, i0, expvals):
    """
    using 3rd order approximation to the exponential function

    This code is a mess because we can't do inlining in numba
    """
    ngauss=self.size
    nrows=image.shape[0]
    ncols=image.shape[1]

    s2n_numer=0.0
    s2n_denom=0.0
    loglike = 0.0
    for row in xrange(nrows):
        u=j[0].dudrow*(row - j[0].row0) + j[0].dudcol*(0 - j[0].col0)
        v=j[0].dvdrow*(row - j[0].row0) + j[0].dvdcol*(0 - j[0].col0)

        for col in xrange(ncols):

            ivar = weight[row,col]
            if ivar <= 0.0:
                continue

            model_val=0.0
            for i in xrange(ngauss):
                udiff=u-self[i].row
                vdiff=v-self[i].col

                u2 = udiff*udiff
                v2 = vdiff*vdiff
                uv=udiff*vdiff

                chi2=self[i].dcc*u2 + self[i].drr*v2 - 2.0*self[i].drc*uv

                if chi2 < 25.0 and chi2 >= 0.0:
                    pnorm = self[i].pnorm
                    x = -0.5*chi2

                    # 3rd order approximation to exp
                    #if x < 0.0:
                    #    ival = int64(x-0.5)
                    #else:
                    #    ival = int64(x+0.5)
                    ival = int64(x-0.5)
                    f = x - ival
                    index = ival-i0
                    
                    expval = expvals[index]
                    fexp = (6+f*(6+f*(3+f)))*0.16666666
                    expval *= fexp

                    model_val += pnorm*expval
            
            pixval = image[row,col]
            diff = model_val-pixval
            loglike += diff*diff*ivar
            s2n_numer += pixval*model_val*ivar
            s2n_denom += model_val*model_val*ivar

            u += j[0].dudcol
            v += j[0].dvdcol

    loglike *= (-0.5)

    return loglike, s2n_numer, s2n_denom

'''
@jit(argtypes=[ _gauss2d[:], float64[:,:], float64[:,:], _jacobian[:], int64, float64[:] ])
def _fluxcorr_jacob_fast3(self, image, weight, j, i0, expvals):
    """
    using 3rd order approximation to the exponential function

    This code is a mess because we can't do inlining in numba
    """
    ngauss=self.size
    nrows=image.shape[0]
    ncols=image.shape[1]

    xcorr_sum=0.0
    msq_sum=0.0


    for row in xrange(nrows):
        u=j[0].dudrow*(row - j[0].row0) + j[0].dudcol*(0 - j[0].col0)
        v=j[0].dvdrow*(row - j[0].row0) + j[0].dvdcol*(0 - j[0].col0)

        for col in xrange(ncols):

            ivar = weight[row,col]
            if ivar <= 0.0:
                continue

            model_val=0.0
            for i in xrange(ngauss):
                udiff=u-self[i].row
                vdiff=v-self[i].col

                u2 = udiff*udiff
                v2 = vdiff*vdiff
                uv=udiff*vdiff

                chi2=self[i].dcc*u2 + self[i].drr*v2 - 2.0*self[i].drc*uv

                if chi2 < 25.0 and chi2 >= 0.0:
                    pnorm = self[i].pnorm
                    x = -0.5*chi2

                    # 3rd order approximation to exp
                    ival = int64(x-0.5)
                    f = x - ival
                    index = ival-i0
                    
                    expval = expvals[index]
                    fexp = (6+f*(6+f*(3+f)))*0.16666666
                    expval *= fexp

                    model_val += pnorm*expval

            pixval = image[row,col]

            xcorr_sum += model_val*pixval*ivar
            msq_sum += model_val*model_val*ivar

            u += j[0].dudcol
            v += j[0].dvdcol


    return xcorr_sum, msq_sum
'''




@jit(argtypes=[ _gauss2d[:], float64[:,:], float64[:,:], _jacobian[:], float64[:], int64, int64, float64[:] ])
def _fdiff_jacob_fast3(self, image, weight, j, fdiff, start, i0, expvals):
    """
    using 3rd order approximation to the exponential function

    This code is a mess because we can't do inlining in numba
    """
    ngauss=self.size
    nrows=image.shape[0]
    ncols=image.shape[1]

    s2n_numer=0.0
    s2n_denom=0.0

    # starting i at start-1 simplifies logic later, because of the continue
    fdiff_i=start-1

    for row in xrange(nrows):
        u=j[0].dudrow*(row - j[0].row0) + j[0].dudcol*(0 - j[0].col0)
        v=j[0].dvdrow*(row - j[0].row0) + j[0].dvdcol*(0 - j[0].col0)

        for col in xrange(ncols):
            fdiff_i += 1

            ivar = weight[row,col]
            if ivar <= 0.0:
                continue

            ierr=numpy.sqrt(ivar)

            model_val=0.0
            for i in xrange(ngauss):
                udiff=u-self[i].row
                vdiff=v-self[i].col

                u2 = udiff*udiff
                v2 = vdiff*vdiff
                uv=udiff*vdiff

                chi2=self[i].dcc*u2 + self[i].drr*v2 - 2.0*self[i].drc*uv

                if chi2 < 25.0 and chi2 >= 0.0:
                    pnorm = self[i].pnorm
                    x = -0.5*chi2

                    # 3rd order approximation to exp
                    #if x < 0.0:
                    #    ival = int64(x-0.5)
                    #else:
                    #    ival = int64(x+0.5)
                    ival = int64(x-0.5)
                    f = x - ival
                    index = ival-i0
                    
                    expval = expvals[index]
                    fexp = (6+f*(6+f*(3+f)))*0.16666666
                    expval *= fexp

                    model_val += pnorm*expval
            
            pixval = image[row,col]
            fdiff[fdiff_i] = (model_val-pixval)*ierr
            s2n_numer += pixval*model_val*ivar
            s2n_denom += model_val*model_val*ivar

            u += j[0].dudcol
            v += j[0].dvdcol


    return s2n_numer, s2n_denom

@jit(argtypes=[ _gauss2d[:], float64[:,:], float64[:,:], _jacobian[:], float64[:], float64[:], int64, int64, float64[:] ])
def _fdiff_jacob_fast3_randomize(self, image, weight, j, fdiff, randvals, start, i0, expvals):
    """
    Add random numbers to model as well.

    using 3rd order approximation to the exponential function

    This code is a mess because we can't do inlining in numba
    """
    ngauss=self.size
    nrows=image.shape[0]
    ncols=image.shape[1]

    s2n_numer=0.0
    s2n_denom=0.0

    # starting i at start-1 simplifies logic later, because of the continue
    fdiff_i=start-1

    for row in xrange(nrows):
        u=j[0].dudrow*(row - j[0].row0) + j[0].dudcol*(0 - j[0].col0)
        v=j[0].dvdrow*(row - j[0].row0) + j[0].dvdcol*(0 - j[0].col0)

        for col in xrange(ncols):
            fdiff_i += 1

            ivar = weight[row,col]
            if ivar <= 0.0:
                continue

            # because we are adding noise to the model
            ivar *= 0.5

            ierr=numpy.sqrt(ivar)

            model_val=0.0
            for i in xrange(ngauss):
                udiff=u-self[i].row
                vdiff=v-self[i].col

                u2 = udiff*udiff
                v2 = vdiff*vdiff
                uv=udiff*vdiff

                chi2=self[i].dcc*u2 + self[i].drr*v2 - 2.0*self[i].drc*uv

                if chi2 < 25.0 and chi2 >= 0.0:
                    pnorm = self[i].pnorm
                    x = -0.5*chi2

                    # 3rd order approximation to exp
                    #if x < 0.0:
                    #    ival = int64(x-0.5)
                    #else:
                    #    ival = int64(x+0.5)
                    ival = int64(x-0.5)
                    f = x - ival
                    index = ival-i0
                    
                    expval = expvals[index]
                    fexp = (6+f*(6+f*(3+f)))*0.16666666
                    expval *= fexp

                    model_val += pnorm*expval
            
            model_val += randvals[fdiff_i]

            pixval = image[row,col]
            fdiff[fdiff_i] = (model_val-pixval)*ierr
            s2n_numer += pixval*model_val*ivar
            s2n_denom += model_val*model_val*ivar

            u += j[0].dudcol
            v += j[0].dvdcol


    return s2n_numer, s2n_denom

class GMixND(object):
    """
    Gaussian mixture in arbitrary dimensions
    """
    def __init__(self, weights, means, covars):
        # copy both to avoid it getting changed under us and to
        # make sure native byte order
        self.weights = numpy.array(weights, dtype='f8', copy=True)
        self.means=numpy.array(means, dtype='f8', copy=True)
        self.covars=numpy.array(covars, dtype='f8', copy=True)

        self.ngauss = self.weights.size

        self.ndim = means[0,:].size

        self._calc_icovars_and_norms()

        self.tmp_lnprob = zeros(self.ngauss)


    def get_prob_scalar(self, pars):
        """
        Use a numba compile function
        """
        dolog=0
        if self.ndim==3:
            return _get_gmixnd_3d(self.log_pnorms,
                                  self.means,
                                  self.icovars,
                                  self.tmp_lnprob,
                                  pars,
                                  dolog)
        elif self.ndim==2:
            return _get_gmixnd_2d(self.log_pnorms,
                                  self.means,
                                  self.icovars,
                                  self.tmp_lnprob,
                                  pars,
                                  dolog)
        else:
            raise RuntimeError("only have fast formula for 2,3 dims")

    def get_lnprob_scalar(self, pars):
        """
        (x-xmean) icovar (x-xmean)
        """
        dolog=1
        if self.ndim==3:
            return _get_gmixnd_3d(self.log_pnorms,
                                  self.means,
                                  self.icovars,
                                  self.tmp_lnprob,
                                  pars,
                                  dolog)
        elif self.ndim==2:
            return _get_gmixnd_2d(self.log_pnorms,
                                  self.means,
                                  self.icovars,
                                  self.tmp_lnprob,
                                  pars,
                                  dolog)

        else:
            raise RuntimeError("only have fast formula for 2,3 dims")
        return lnp

    def get_prob_array(self, pars):
        """
        array input
        """
        dolog=0
        n=pars.shape[0]
        p=zeros(n)
        if self.ndim==3:
            _get_gmixnd_array_3d(self.log_pnorms,
                                 self.means,
                                 self.icovars,
                                 self.tmp_lnprob,
                                 pars,
                                 dolog,
                                 p)
        elif self.ndim==2:
            _get_gmixnd_array_2d(self.log_pnorms,
                                 self.means,
                                 self.icovars,
                                 self.tmp_lnprob,
                                 pars,
                                 dolog,
                                 p)

        else:
            raise RuntimeError("only have fast formula for 2,3 dims")
        return p

    def get_lnprob_array(self, pars):
        """
        array input
        """
        dolog=1
        n=pars.shape[0]
        lnp=zeros(n)
        if self.ndim==3:
            _get_gmixnd_array_3d(self.log_pnorms,
                                 self.means,
                                 self.icovars,
                                 self.tmp_lnprob,
                                 pars,
                                 dolog,
                                 lnp)
        if self.ndim==2:
            _get_gmixnd_array_2d(self.log_pnorms,
                                 self.means,
                                 self.icovars,
                                 self.tmp_lnprob,
                                 pars,
                                 dolog,
                                 lnp)

        else:
            raise RuntimeError("only have fast formula for 2,3 dims")

        return lnp


    def _calc_icovars_and_norms(self):
        """
        Calculate the normalizations and inverse covariance matrices
        """
        from numpy import pi

        twopi = 2.0*pi

        norms = zeros(self.ngauss)
        icovars = zeros( (self.ngauss, self.ndim, self.ndim) )

        for i in xrange(self.ngauss):
            cov = self.covars[i,:,:]
            icov = numpy.linalg.inv( cov )

            det = numpy.linalg.det(cov)
            n=1.0/sqrt( twopi**self.ndim * det )

            norms[i] = n
            icovars[i,:,:] = icov

        self.norms = norms
        self.pnorms = norms*self.weights
        self.log_pnorms = log(self.pnorms)
        self.icovars = icovars


@autojit
def _get_gmixnd_array_3d(log_pnorms, means, icovars, tmp_lnprob, x, dolog, output):
    """
    Fill the output array
    """
    n=output.size
    for i in xrange(n):
        output[i] = _get_gmixnd_3d(log_pnorms, means, icovars, tmp_lnprob, x[i,:], dolog)

@autojit
def _get_gmixnd_3d(log_pnorms, means, icovars, tmp_lnprob, x, dolog):
    """
    Trying to avoid underflow
    vmax = arr.max(axis=0)
    out = np.log(np.sum(np.exp(arr - vmax), axis=0))
    out += vmax
    return out
    """
    ngauss=log_pnorms.size

    p=0.0
    lnp=0.0
    lnpmax=-9.99e9

    for i in xrange(ngauss):
        logpnorm = log_pnorms[i]

        x1diff=x[0]-means[i,0]
        x2diff=x[1]-means[i,1]
        x3diff=x[2]-means[i,2]

        chi2  = x1diff*(icovars[i,0,0]*x1diff + icovars[i,0,1]*x2diff + icovars[i,0,2]*x3diff)
        chi2 += x2diff*(icovars[i,1,0]*x1diff + icovars[i,1,1]*x2diff + icovars[i,1,2]*x3diff)
        chi2 += x3diff*(icovars[i,2,0]*x1diff + icovars[i,2,1]*x2diff + icovars[i,2,2]*x3diff)

        lnp = -0.5*chi2 + logpnorm
        if lnp > lnpmax:
            lnpmax=lnp
        tmp_lnprob[i] = lnp
        
    for i in xrange(ngauss):
        p += exp(tmp_lnprob[i] - lnpmax)

    out=0.0
    if dolog==1:
        out = log(p) + lnpmax
    else:
        out=p*exp(lnpmax)

    return out

@autojit
def _get_gmixnd_array_2d(log_pnorms, means, icovars, tmp_lnprob, x, dolog, output):
    """
    Fill the output array
    """
    n=output.size
    for i in xrange(n):
        output[i] = _get_gmixnd_2d(log_pnorms, means, icovars, tmp_lnprob, x[i,:], dolog)

@autojit
def _get_gmixnd_2d(log_pnorms, means, icovars, tmp_lnprob, x, dolog):
    """
    Trying to avoid underflow
    vmax = arr.max(axis=0)
    out = np.log(np.sum(np.exp(arr - vmax), axis=0))
    out += vmax
    return out
    """
    ngauss=log_pnorms.size

    p=0.0
    lnp=0.0
    lnpmax=-9.99e9

    for i in xrange(ngauss):
        logpnorm = log_pnorms[i]

        x1diff=x[0]-means[i,0]
        x2diff=x[1]-means[i,1]

        chi2  = x1diff*(icovars[i,0,0]*x1diff + icovars[i,0,1]*x2diff)
        chi2 += x2diff*(icovars[i,1,0]*x1diff + icovars[i,1,1]*x2diff)

        lnp = -0.5*chi2 + logpnorm
        if lnp > lnpmax:
            lnpmax=lnp
        tmp_lnprob[i] = lnp
        
    for i in xrange(ngauss):
        p += exp(tmp_lnprob[i] - lnpmax)

    out=0.0
    if dolog==1:
        out = log(p) + lnpmax
    else:
        out=p*exp(lnpmax)

    return out

