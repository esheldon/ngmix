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

References
----------
Bromiley, P. A. (2003), Products and Convolutions of Gaussian
    Probability Density Functions, Tina-Vision Memo 2003-003,
    University of Manchester
"""
import numpy as np

from ..gmix.gmix_nb import (
    _pvals_exp, _fvals_exp, _pvals_dev, _fvals_dev,
)
from ..moments import cov_from_e


def det2(M):
    """determinant of a 2x2 matrix"""
    return M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]


# (flux fraction, relative T) per gaussian of the ngmix expansions;
# the fractions sum to one and the T factors are normalized to
# total T = 1
_PROFILE_COMPS = {
    'exp': list(zip(_pvals_exp.tolist(), _fvals_exp.tolist())),
    'dev': list(zip(_pvals_dev.tolist(), _fvals_dev.tolist())),
}


def get_profile_comps(name):
    """
    flux fractions and relative T factors of the ngmix gaussian
    expansion of the named profile ('exp': 6 gaussians, 'dev': 10
    gaussians), normalized to total T = 1

    These are the expansion tables of ngmix.gmix.gmix_nb, the same
    ones GMixModel uses.

    Parameters
    ----------
    name: str
        'exp' or 'dev'

    Returns
    -------
    comps: list of (frac, cT)
        one entry per gaussian: its fraction of the total flux and
        its T relative to the total T
    """
    return _PROFILE_COMPS[name]


def model_gauss_components(model, Tsmooth):
    """
    the gaussian components of a model in the smoothed plane

    exported for use in kdeblend

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
    elif model['type'] in ('exp', 'dev'):
        if 'cov' in model:
            fam_cov = model['cov']
        else:
            fam_cov = cov_from_e(model['e1'], model['e2'], model['T'])
        smooth = Tsmooth / 2
        comps = get_profile_comps(model['type'])
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

    fracs, cov_sm00, cov_sm01, cov_sm11 = model_gauss_components(
        model, Tsmooth,
    )
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

    model is a family name ('exp', 'dev') or a model dict with a
    'type' entry holding the name
    """
    if isinstance(model, str):
        comps = get_profile_comps(model)
    else:
        comps = get_profile_comps(model['type'])

    smooth_cov = np.diag([Tsmooth / 2, Tsmooth / 2])
    for frac, cT in comps:
        C = wt_cov + cT * fam_cov + smooth_cov
        if (C[0, 0] <= 0 or C[1, 1] <= 0
                or det2(C) <= 1.0e-6 * C[0, 0] * C[1, 1]):
            return False
    return True
