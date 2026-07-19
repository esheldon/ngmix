"""
closed-form weighted moment sums for object models

The adaptive moments machinery only consumes weighted moment sums,
which are linear in the image, so model contributions can be computed
(or subtracted) in closed form without rendering images.  For
gaussian models and fixed-ratio mixtures of gaussians the weighted
sums follow from the product-gaussian identities.

For weight W (covariance Sw, unit peak, centered at the object being
measured) and a model gaussian (flux F, covariance So in the
smoothed-deconvolved plane) at offset d,

    C = Sw + So
    s_flux = F exp(-d^T C^-1 d / 2) / (2 pi detAtinv sqrt(det C))
    mu     = Sw C^-1 d              first moments
    <x x^T> = Sw C^-1 So + mu mu^T  second moments

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
"""
import numpy as np

from ..gmix import GMixModel


def det2(M):
    """determinant of a 2x2 matrix"""
    return M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]


def inv2(M):
    """inverse of a symmetric 2x2 matrix"""
    return np.array([
        [M[1, 1], -M[0, 1]],
        [-M[0, 1], M[0, 0]],
    ]) / det2(M)


def cov_from_e(e1, e2, T):
    """covariance matrix [[Ivv, Ivu], [Ivu, Iuu]] from
    distortion-convention shape parameters, e1 = <uu - vv>/T"""
    return 0.5 * np.array([
        [T * (1 - e1), T * e2],
        [T * e2, T * (1 + e1)],
    ])


def gauss_model_ksums(flux, So, dv, du, Sw, detAtinv):
    """
    closed-form weighted moment sums of a gaussian model

    Parameters
    ----------
    flux: float
        total flux of the model
    So: (2, 2) array
        model covariance in the smoothed-deconvolved plane
    dv, du: float
        offset of the model center from the weight center
    Sw: (2, 2) array
        the weight covariance
    detAtinv: float
        the k-space area factor of the epoch

    Returns
    -------
    sums: array of size 6
        [v, u, M1, M2, T, flux] sums in the units of admom_ksums
    """
    C = Sw + So
    Cinv = inv2(C)
    d = np.array([dv, du])
    Cd = Cinv @ d

    sflux = flux * np.exp(-0.5 * d @ Cd) / (
        2 * np.pi * detAtinv * np.sqrt(det2(C))
    )

    mu = Sw @ Cd
    Sp = Sw @ Cinv @ So
    vv = Sp[0, 0] + mu[0] * mu[0]
    vu = Sp[0, 1] + mu[0] * mu[1]
    uu = Sp[1, 1] + mu[1] * mu[1]

    return sflux * np.array([
        mu[0], mu[1], uu - vv, 2 * vu, uu + vv, 1.0,
    ])


_PROFILE_COMPS = {}


def get_profile_comps(name):
    """flux fractions and relative T factors of the ngmix gaussian
    expansion of the named profile ('exp': 6 gaussians, 'dev': 10
    gaussians), normalized to total T = 1"""
    if name not in _PROFILE_COMPS:
        gm = GMixModel([0, 0, 0, 0, 1.0, 1.0], name)
        d = gm.get_data()
        fracs = d['p'] / d['p'].sum()
        cTs = d['irr'] + d['icc']
        _PROFILE_COMPS[name] = list(zip(fracs, cTs))
    return _PROFILE_COMPS[name]


def model_comps(model, Tsmooth):
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
    fracs, So00, So01, So11: arrays
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
    elif model['type'] in ('exp', 'dev'):
        if 'cov' in model:
            Sfam = model['cov']
        else:
            Sfam = cov_from_e(model['e1'], model['e2'], model['T'])
        smooth = Tsmooth / 2
        comps = get_profile_comps(model['type'])
        n = len(comps)
        fracs = np.zeros(n)
        So00 = np.zeros(n)
        So01 = np.zeros(n)
        So11 = np.zeros(n)
        for k, (frac, cT) in enumerate(comps):
            fracs[k] = frac
            So00[k] = cT * Sfam[0, 0] + smooth
            So01[k] = cT * Sfam[0, 1]
            So11[k] = cT * Sfam[1, 1] + smooth
        return fracs, So00, So01, So11
    else:
        raise ValueError(f"bad model type: '{model['type']}'")


def model_ksums(model, band, dv, du, Sw, detAtinv, Tsmooth):
    """
    closed-form weighted moment sums of an object model, dispatching
    on the model type, evaluated via the numba kernel
    """
    from .models_nb import gauss_comps_ksums

    fracs, So00, So01, So11 = model_comps(model, Tsmooth)
    F = model['F'][band] * fracs
    n = F.size
    sums = np.zeros(6)
    gauss_comps_ksums(
        F, So00, So01, So11,
        np.full(n, float(dv)), np.full(n, float(du)),
        Sw[0, 0], Sw[0, 1], Sw[1, 1], float(detAtinv), sums,
    )
    return sums


def mixture_model_valid(name, Sfam, Sw, Tsmooth):
    """
    check that a mixture family covariance gives well defined
    weighted sums: every component must have positive definite total
    covariance C = Sw + cT Sfam + smoothing.  This is weaker than
    positive definiteness of Sfam itself: the smoothing and the
    weight regularize the model family the same way they regularize
    the data, so mildly negative sizes and ellipticities slightly
    past one remain valid states.
    """
    smooth_cov = np.diag([Tsmooth / 2, Tsmooth / 2])
    for frac, cT in get_profile_comps(name):
        C = Sw + cT * Sfam + smooth_cov
        if (C[0, 0] <= 0 or C[1, 1] <= 0
                or det2(C) <= 1.0e-6 * C[0, 0] * C[1, 1]):
            return False
    return True
