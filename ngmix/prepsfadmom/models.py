"""
closed-form weighted moment sums for object models

The adaptive moments machinery only consumes weighted moment sums,
which are linear in the image, so model contributions can be computed
(or subtracted) in closed form without rendering images.  For
gaussian models and fixed-ratio mixtures of gaussians the weighted
sums follow from the product-gaussian identities (Bromiley 2003).

For weight W (covariance wt_cov, unit peak, centered at the object being
measured) and a model gaussian (flux F, covariance cov_sm in the
smoothed-deconvolved plane) at offset d,

    C = wt_cov + cov_sm
    s_flux = F exp(-d^T C^-1 d / 2) / (2 pi detAtinv sqrt(det C))
    mu     = wt_cov C^-1 d              first moments
    <x x^T> = wt_cov C^-1 cov_sm + mu mu^T  second moments

in exactly the units of the admom_ksums output.

Model dicts:
    {'type': 'gauss', 'cov_sm': (2,2), 'F': (nband,)}
        a single gaussian; cov_sm is the covariance in the smoothed
        plane (galaxy covariance plus the smoothing)
    {'type': 'star', 'cov_sm': (2,2), 'F': (nband,)}
        a pre-psf delta function; cov_sm is exactly the smoothing
        covariance and is never updated
    {'type': 'exp', 'e1':, 'e2':, 'T':, 'F': (nband,)}
    {'type': 'exp', 'cov': (2,2), 'F': (nband,)}
    {'type': 'dev', ...}
        the ngmix 6-gaussian exponential expansion (or the
        10-gaussian de Vaucouleurs expansion), coelliptical with
        fixed flux fractions and size ratios; T and e (or the family
        covariance cov, of which each component covariance is a fixed
        multiple) are pre-psf.  The family covariance need not be
        positive definite: the components only enter the sums through
        their smoothed covariances, so mildly negative sizes are well
        defined, allowing unbiased scatter through zero size (see
        mixture_model_valid)
    {'type': 'bdf', 'cov': (2,2), 'F': (nband,),
     'fracdev':, 'TdByTe':}
        the composite exp plus dev model: both families share the
        center and the family covariance (so the ellipticity), the
        dev component covariance is TdByTe times the exp one, and
        fracdev is the dev share of the total flux.  At fixed
        fracdev and TdByTe this is a fixed-ratio 16-gaussian
        mixture and all of the machinery above applies

References
----------
Bromiley, P. A. (2003), Products and Convolutions of Gaussian
    Probability Density Functions, Tina-Vision Memo 2003-003,
    University of Manchester
"""
import numpy as np

from ..gmix import GMixModel
from ..moments import cov_from_e


def det2(M):
    """determinant of a 2x2 matrix"""
    return M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]


def _inv2(M):
    """inverse of a symmetric 2x2 matrix"""
    return np.array([
        [M[1, 1], -M[0, 1]],
        [-M[0, 1], M[0, 0]],
    ]) / det2(M)


def gauss_model_ksums(flux, cov_sm, dv, du, wt_cov, detAtinv):
    """
    closed-form weighted moment sums of a gaussian model

    Parameters
    ----------
    flux: float
        total flux of the model
    cov_sm: (2, 2) array
        model covariance in the smoothed-deconvolved plane
    dv, du: float
        offset of the model center from the weight center
    wt_cov: (2, 2) array
        the weight covariance
    detAtinv: float
        the k-space area factor of the epoch

    Returns
    -------
    sums: array of size 6
        [v, u, M1, M2, T, flux] sums in the units of admom_ksums
    """
    C = wt_cov + cov_sm
    Cinv = _inv2(C)
    d = np.array([dv, du])
    Cd = Cinv @ d

    # the product of the weight and model gaussians is an
    # unnormalized gaussian (see the module docstring).  Its
    # integral, the total weighted flux, is the two-gaussian
    # overlap integral: a gaussian density with the summed
    # covariance C evaluated at the center offset d, so the
    # flux falls off as a gaussian in the offset.  Units are
    # those of the admom_ksums output
    sflux = flux * np.exp(-0.5 * d @ Cd) / (
        2 * np.pi * detAtinv * np.sqrt(det2(C))
    )

    # mu: the mean of the product gaussian, wt_cov C^-1 d
    mu = wt_cov @ Cd
    # Sp: the second moments of the product gaussian
    Sp = wt_cov @ Cinv @ cov_sm
    # raw second moments about the weight center: Sp + mu mu^T
    vv = Sp[0, 0] + mu[0] * mu[0]
    vu = Sp[0, 1] + mu[0] * mu[1]
    uu = Sp[1, 1] + mu[1] * mu[1]

    return sflux * np.array([
        mu[0], mu[1], uu - vv, 2 * vu, uu + vv, 1.0,
    ])


_PROFILE_COMPS = {}


def get_profile_comps(name):
    """
    flux fractions and relative T factors of the ngmix gaussian
    expansion of the named profile ('exp': 6 gaussians, 'dev': 10
    gaussians), normalized to total T = 1

    The table is derived from a GMixModel rather than duplicating
    the expansion constants here, so it cannot drift from the
    ngmix tables; it is computed once per profile and cached.
    """
    if name not in _PROFILE_COMPS:
        gm = GMixModel([0, 0, 0, 0, 1.0, 1.0], name)
        d = gm.get_data()
        fracs = d['p'] / d['p'].sum()
        cTs = d['irr'] + d['icc']
        _PROFILE_COMPS[name] = list(zip(fracs, cTs))
    return _PROFILE_COMPS[name]


def bdf_comps(fracdev, TdByTe):
    """
    flux fractions and relative T factors of the composite exp plus
    dev mixture: the exp table scaled by 1 - fracdev plus the dev
    table with sizes scaled by TdByTe and fluxes by fracdev.  The
    scale parameter of the composite is the exp family T
    """
    return (
        [((1 - fracdev) * f, cT)
         for f, cT in get_profile_comps('exp')]
        + [(fracdev * f, TdByTe * cT)
           for f, cT in get_profile_comps('dev')]
    )


def _mixture_comps(model):
    """
    the (frac, cT) component table for a mixture model dict or
    spec; see the module docstring
    """
    if model['type'] == 'bdf':
        return bdf_comps(model['fracdev'], model['TdByTe'])
    return get_profile_comps(model['type'])


def model_gauss_components(model, Tsmooth):
    """
    the gaussian components of a model in the smoothed plane

    Parameters
    ----------
    model: dict
        model dict, see the module docstring
    Tsmooth: float
        T of the gaussian smoothing

    Returns
    -------
    fracs, cov_sm00, cov_sm01, cov_sm11: arrays
        per component flux fractions and covariances in the smoothed
        plane
    """
    if model['type'] in ('gauss', 'star'):
        c = model['cov_sm']
        return (
            np.ones(1),
            np.array([c[0, 0]]),
            np.array([c[0, 1]]),
            np.array([c[1, 1]]),
        )
    elif model['type'] in ('exp', 'dev', 'bdf'):
        if 'cov' in model:
            fam_cov = model['cov']
        else:
            fam_cov = cov_from_e(model['e1'], model['e2'], model['T'])
        smooth = Tsmooth / 2
        comps = _mixture_comps(model)
        n = len(comps)
        fracs = np.zeros(n)
        cov_sm00 = np.zeros(n)
        cov_sm01 = np.zeros(n)
        cov_sm11 = np.zeros(n)
        for k, (frac, cT) in enumerate(comps):
            fracs[k] = frac
            cov_sm00[k] = cT * fam_cov[0, 0] + smooth
            cov_sm01[k] = cT * fam_cov[0, 1]
            cov_sm11[k] = cT * fam_cov[1, 1] + smooth
        return fracs, cov_sm00, cov_sm01, cov_sm11
    else:
        raise ValueError(f"bad model type: '{model['type']}'")


def model_ksums(model, band, dv, du, wt_cov, detAtinv, Tsmooth):
    """
    closed-form weighted moment sums of an object model, dispatching
    on the model type, evaluated via the numba kernel
    """
    from .models_nb import gauss_comps_ksums

    fracs, cov_sm00, cov_sm01, cov_sm11 = model_gauss_components(model, Tsmooth)
    F = model['F'][band] * fracs
    n = F.size
    sums = np.zeros(6)
    gauss_comps_ksums(
        F, cov_sm00, cov_sm01, cov_sm11,
        np.full(n, float(dv)), np.full(n, float(du)),
        wt_cov[0, 0], wt_cov[0, 1], wt_cov[1, 1], float(detAtinv), sums,
    )
    return sums


def mixture_model_valid(model, fam_cov, wt_cov, Tsmooth):
    """
    check that a mixture family covariance gives well defined
    weighted sums: every component must have positive definite total
    covariance C = wt_cov + cT fam_cov + smoothing.  This is weaker than
    positive definiteness of fam_cov itself: the smoothing and the
    weight regularize the model family the same way they regularize
    the data, so mildly negative sizes and ellipticities slightly
    past one remain valid states.

    model is a family name ('exp', 'dev') or a model spec dict; for
    'bdf' the dict must carry 'TdByTe' (the flux split does not
    affect validity)
    """
    if isinstance(model, str):
        comps = get_profile_comps(model)
    elif model['type'] == 'bdf':
        comps = bdf_comps(model.get('fracdev', 0.5), model['TdByTe'])
    else:
        comps = get_profile_comps(model['type'])

    smooth_cov = np.diag([Tsmooth / 2, Tsmooth / 2])
    for frac, cT in comps:
        C = wt_cov + cT * fam_cov + smooth_cov
        if (C[0, 0] <= 0 or C[1, 1] <= 0
                or det2(C) <= 1.0e-6 * C[0, 0] * C[1, 1]):
            return False
    return True
