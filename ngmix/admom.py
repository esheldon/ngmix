from __future__ import print_function
import numpy

from .gmix import GMix, GMixModel
from .shape import e1e2_to_g1g2
from .observation import Observation, ObsList, MultiBandObsList
from . import _gmix

class Admom(object):
    """
    for now use default tolerances set to
    that of the fortran code, for comparison

    parameters
    ----------
    obs: Observation
        ngmix.Observation
    maxiter: integer, optional
        Maximum number of iterations
    etol: float, optional
        absolute tolerance in e1 or e2 to determine convergence
    Ttol: float, optional
        relative tolerance in T to determine convergence
    """

    def __init__(self, obs, maxiter=100, shiftmax=5.0,
                 etol=1.0e-5, Ttol=0.001, deconv=False):
        self._set_obs(obs, deconv)
        self._set_conf(maxiter, shiftmax, etol, Ttol)

    def get_result(self):
        """
        get the result
        """

        if not hasattr(self,'result'):
            raise RuntimeError("run go() first")

        return self.result

    def get_gmix(self):
        """
        get a gmix representing the best fit, normalized
        """

        pars=self.result['pars'].copy()
        pars[5]=1.0

        e1 = pars[2]/pars[4]
        e2 = pars[3]/pars[4]

        g1,g2 = e1e2_to_g1g2(e1, e2)
        pars[2] = g1
        pars[3] = g2

        return GMixModel(pars, "gauss")

    def go(self, guess_gmix):
        """
        run the adpative moments

        parameters
        ----------
        guess_gmix: ngmix.GMix
            A guess for the fitter.
        """

        if not isinstance(guess_gmix,GMix):
            raise ValueError("guess should be GMix, but got "
                             "type %s" % type(guess_gmix))


        am_result=self._get_am_result()

        if self._deconv:
            _gmix.admom_multi_deconv(
                self.conf,
                self._imlist,
                self._wtlist,
                self._psflist,
                self._jlist,
                guess_gmix._data,
                am_result,
            )
        else:
            if len(self._imlist) > 1:
                #print("using multi")
                _gmix.admom_multi(
                    self.conf,
                    self._imlist,
                    self._wtlist,
                    self._jlist,
                    guess_gmix._data,
                    am_result,
                )
            else:
                #print("using single")
                _gmix.admom(
                    self.conf,
                    self._imlist[0],
                    self._wtlist[0],
                    self._jlist[0],
                    guess_gmix._data,
                    am_result,
                )

        self._copy_result(am_result)

    def _copy_result(self, ares):
        ares=ares[0]

        res={}
        for n in ares.dtype.names:
            if n == 'sums':
                res[n] = ares[n].copy()
            elif n=='sums_cov':
                res[n] = ares[n].reshape( (6,6)).copy()
            else:
                res[n] = ares[n]

        res['s2n'] = -9999.0
        res['err'] = 9999.0
        res['e'] = [-9999.0, -9999.0]
        res['e_cov'] = numpy.diag( [9999.0]*2 )

        res['flagstr'] = _admom_flagmap[res['flags']]
        if res['flags']==0:
            # now want pars and cov for [cen1,cen2,e1,e2,T,flux]
            sums=res['sums']

            pars=res['pars']
            res['T'] = pars[4]
            if res['T'] > 0.0:
                res['e'][:] = res['pars'][2:2+2]/res['T']
            else:
                res['flags'] = 0x8

            if res['s2n_denom'] > 0:
                res['s2n'] = res['s2n_numer']/numpy.sqrt(res['s2n_denom'])

                # error on each shape component from BJ02 for gaussians
                # assumes round

                res['err'] = 2.0/res['s2n']
                res['e_cov'][:,:] = numpy.diag( [ res['err']**2 ]*2 )

                # very approximate off-diagonal terms
                #if scov[2,2] > 0 and scov[3,3] > 0:
                #    scov=res['sums_cov']
                #    cross=res['err']**2 * scov[2,3]/numpy.sqrt(scov[2,2]*scov[3,3])
                #    res['e_cov'][0,1] = cross
                #    res['e_cov'][1,0] = cross
            else:
                res['flags'] = 0x40

        self.result=res

    def _set_obs(self, obs, deconv):

        assert deconv==False,"deconv doesn't work yet"
        self._deconv=deconv

        imlist=[]
        wtlist=[]
        jlist=[]

        if deconv:
            self._psflist=[]

        if isinstance(obs,MultiBandObsList):
            mbobs=obs
            for oblist in mbobs:
                for obs in obslist:
                    imlist.append(obs.image)
                    wtlist.append(obs.weight)
                    jlist.append(obs.jacobian._data)
                    if deconv and obs.has_psf_gmix():
                        self._psflist.append(obs.psf.gmix._data)

        elif isinstance(obs, ObsList):
            obslist=obs
            for obs in obslist:
                imlist.append(obs.image)
                wtlist.append(obs.weight)
                jlist.append(obs.jacobian._data)
                if deconv and obs.has_psf_gmix():
                    self._psflist.append(obs.psf.gmix._data)

        elif isinstance(obs, Observation):
            imlist.append(obs.image)
            wtlist.append(obs.weight)
            jlist.append(obs.jacobian._data)
            if deconv and obs.has_psf_gmix():
                self._psflist.append(obs.psf.gmix._data)
        else:
            raise ValueError("obs is type '%s' but should be "
                             "Observation, ObsList, or MultiBandObsList")

        if deconv:
            np=len(self._psflist)
            ni=len(imlist)

            if np != ni:
                raise ValueError("only some of obs had psf set: %d/%d" % (np,ni))

        if len(imlist) > 1000:
            raise ValueError("currently limited to 1000 "
                             "images, got %d" % len(imlist))

        self._imlist=imlist
        self._wtlist=wtlist
        self._jlist=jlist


    def _set_conf(self, maxiter, shiftmax, etol, Ttol):
        dt=numpy.dtype(_admom_conf_dtype, align=True)
        conf=numpy.zeros(1, dtype=dt)

        conf['maxit']=maxiter
        conf['shiftmax']=shiftmax
        conf['etol']=etol
        conf['Ttol']=Ttol

        self.conf=conf

    def _get_am_result(self):
        dt=numpy.dtype(_admom_result_dtype, align=True)
        return numpy.zeros(1, dtype=dt)


_admom_conf_dtype=[
    ('maxit','i4'),
    ('shiftmax','f8'),
    ('etol','f8'),
    ('Ttol','f8'),
]
_admom_result_dtype=[
    ('flags','i4'),

    ('numiter','i4'),

    ('nimage','i4'),

    ('wsum','f8'),
    ('s2n_numer','f8'),
    ('s2n_denom','f8'),

    ('sums','f8',6),
    ('sums_cov','f8', 36),

    ('pars','f8',6),
]

_admom_flagmap={
    0:'ok',
    0x1:'edge hit', # not currently used
    0x2:'center shifted too far',
    0x4:'flux < 0',
    0x8:'T < 0',
    0x10:'determinant near zero',
    0x20:'maxit reached',
    0x40:'zero var',
}
