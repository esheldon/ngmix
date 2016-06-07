from __future__ import print_function
import numpy

from .gmix import GMix
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

    def __init__(self, obs, maxiter=200, shiftmax=5.0, etol=0.001, Ttol=0.01):
        self.obs=obs
        self._set_conf(maxiter, shiftmax, etol, Ttol)
        self._set_am_result()

    def get_result(self):
        """
        get the result
        """
        return self.result

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

        gdata = guess_gmix._data

        _gmix.admom(
            self.conf,
            self.obs.image,
            self.obs.weight,
            self.obs.jacobian._data,
            guess_gmix._data,
            self.am_result,
        )

        self._copy_result()

    def _copy_result(self):
        ares=self.am_result[0]

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

        if res['flags']==0:
            # now want pars and cov for [cen1,cen2,e1,e2,T,flux]
            sums=res['sums']

            pars = sums.copy()
            pars[0] = sums[0]/sums[5]
            pars[1] = sums[1]/sums[5]

            pars[2] = sums[2]/sums[4]
            pars[3] = sums[3]/sums[4]
            pars[4] = 2*sums[4]/sums[5]

            pars[5] = sums[5]/res['wsum']

            res['pars'] = pars
            res['e'] = pars[2:2+2]

            if res['s2n_denom'] > 0:
                res['s2n'] = res['s2n_numer']/numpy.sqrt(res['s2n_denom'])
                sigma = numpy.sqrt(pars[4]/2.0)
                # BJ02 for gaussians
                res['err'] = 2.0/res['s2n']

                res['e_cov'] = numpy.diag( [ res['err']**2 ]*2 )

                # very approximate cov
                scov=res['sums_cov']
                cross=res['err']**2 * scov[2,3]/numpy.sqrt(scov[2,2]*scov[3,3])
                res['e_cov'][0,1] = cross
                res['e_cov'][1,0] = cross


        self.result=res

    def _set_conf(self, maxiter, shiftmax, etol, Ttol):
        dt=numpy.dtype(_admom_conf_dtype, align=True)
        conf=numpy.zeros(1, dtype=dt)

        conf['maxit']=maxiter
        conf['shiftmax']=shiftmax
        conf['etol']=etol
        conf['Ttol']=Ttol

        self.conf=conf

    def _set_am_result(self):
        dt=numpy.dtype(_admom_result_dtype, align=True)
        self.am_result=numpy.zeros(1, dtype=dt)


_admom_conf_dtype=[
    ('maxit','i4'),
    ('shiftmax','f8'),
    ('etol','f8'),
    ('Ttol','f8'),
]
_admom_result_dtype=[
    ('flags','i4'),

    ('numiter','i4'),

    ('wsum','f8'),
    ('s2n_numer','f8'),
    ('s2n_denom','f8'),

    ('sums','f8',6),
    ('sums_cov','f8', 36),

]
