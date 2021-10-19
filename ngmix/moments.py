import numpy as np

from .gexceptions import GMixRangeError
from . import shape
import ngmix.flags
from .util import get_ratio_error

MOMENTS_NAME_MAP = {
    "Mv": 0,
    "Mu": 1,
    "M1": 2,
    "M2": 3,
    "MT": 4,
    "MF": 5,
}


def sigma_to_fwhm(sigma):
    """
    convert sigma to fwhm for a gaussian
    """
    return sigma * 2.3548200450309493


def T_to_fwhm(T):
    """
    convert T to fwhm for a gaussian
    """
    sigma = np.sqrt(T / 2.0)
    return sigma_to_fwhm(sigma)


def fwhm_to_sigma(fwhm):
    """
    convert fwhm to sigma for a gaussian
    """
    return fwhm / 2.3548200450309493


def fwhm_to_T(fwhm):
    """
    convert fwhm to T for a gaussian
    """
    sigma = fwhm_to_sigma(fwhm)
    return 2 * sigma ** 2


def r50_to_sigma(r50):
    """
    convert r50, the half light radius, to sigma for a gaussian

    half light radius is the radius that contains half the total light.
    For a gaussian this is fwhm/2
    """
    fwhm = 2.0 * r50
    return fwhm_to_sigma(fwhm)


def sigma_to_r50(sigma):
    """
    convert sigma to r50 for a gaussian
    """
    fwhm = sigma_to_fwhm(sigma)
    r50 = fwhm / 2.0
    return r50


def r50_to_T(r50):
    """
    convert r50, the half light radius, to T=2*sigma**2 for a gaussian

    half light radius is the radius that contains half the total light.
    For a gaussian this is fwhm/2
    """

    sigma = r50_to_sigma(r50)
    T = 2 * sigma ** 2
    return T


def T_to_r50(T):
    """
    convert T=2*sigma**2 to r50 for a gaussian
    """
    sigma = np.sqrt(T / 2.0)
    return sigma_to_r50(sigma)


def moms_to_e1e2(M1, M2, T):
    """
    convert M1, M2, T to e1,e2

    parameters
    -----------
    M1,M2,T: float or array
        M1 is <x^2 - y^2>
        M2 is <2*x*y>
        T  is <x^2 + y^2>

    returns
    -------
    e1,e2:
        M1/T M2/T also know as the standard ellipticity parameters
    """
    if isinstance(T, np.ndarray):
        (w,) = np.where(T <= 0.0)
        if w.size > 0:
            raise GMixRangeError("%d T were <= 0.0" % w.size)
    else:
        if T <= 0.0:
            raise GMixRangeError("T <= 0.0: %g" % T)

    Tinv = 1.0 / T
    e1 = M1 * Tinv
    e2 = M2 * Tinv

    return e1, e2


def get_Tround(T, g1, g2):
    """
    get the round T

    parameters
    ----------
    T: float
        <x^2> + <y^2>
    g1,g2: float
        The reduced shear style shape

    returns
    -------
    Tround: float
        The round size.
    """
    gsq = g1 ** 2 + g2 ** 2
    return T * (1 - gsq) / (1 + gsq)


def get_T(Tround, g1, g2):
    """
    get elliptical T for given Tround and g1,g2

    parameters
    ----------
    Tround: float
        <x^2> + <y^2>
    g1,g2: float
        The reduced shear style shape

    returns
    -------
    T: float
        The elliptical size <x^2 + y^2>
    """
    gsq = g1 ** 2 + g2 ** 2
    return Tround * (1 + gsq) / (1 - gsq)


def get_sheared_M1M2T(M1, M2, T, s1, s2):
    """
    Get sheared moments

    parameters
    ----------
    M1: float or array
        <x^2 - y^2>
    M2: float or array
        <2*xy>
    T:  float or array
        <x^2 + y^2>
    s1,s2: float or array
        The shear to apply.

    returns
    -------
    sheared M1, M2, T
    """

    e1, e2 = moms_to_e1e2(M1, M2, T)
    g1, g2 = shape.e1e2_to_g1g2(e1, e2)

    g1s, g2s = shape.shear_reduced(g1, g2, s1, s2)

    Tround = get_Tround(T, g1, g2)

    Ts = get_T(Tround, g1s, g2s)

    e1s, e2s = shape.g1g2_to_e1e2(g1s, g2s)

    M1s = Ts * e1s
    M2s = Ts * e2s

    return M1s, M2s, Ts


def get_sheared_g1g2T(g1, g2, T, s1, s2):
    """
    Get sheared g1, g2, T

    parameters
    ----------
    g1,g2: float or array
        The reduced shear style shape
    T: float or array
        <x^2 + y^2>
    s1,s2: float or array
        The shear to apply.

    returns
    -------
    sheared g1, g2, T
    """
    g1s, g2s = shape.shear_reduced(g1, g2, s1, s2)

    Tround = get_Tround(T, g1, g2)
    Ts = get_T(Tround, g1s, g2s)

    return g1s, g2s, Ts


def get_sheared_moments(irr, irc, icc, s1, s2):
    """
    Get sheared raw moments

    parameters
    ----------
    irr: scalar or array
        <y^2>
    irc: scalar or array
        <xy>
    icc: scalar or array
        <x^2>
    s1,s2: float or array
        The shear to apply.

    returns
    -------
    irr_s, irc_s, icc_s: scalar or array
        The sheared moments.
    """
    g1, g2, T = mom2g(irr, irc, icc)
    g1s, g2s, Ts = get_sheared_g1g2T(g1, g2, T, s1, s2)
    irr_s, irc_s, icc_s = g2mom(g1s, g2s, Ts)
    return irr_s, irc_s, icc_s


def mom2e(Irr, Irc, Icc):
    """
    Convert icc, irc, icc to e1,e2

    parameters
    ----------
    irr: scalar or array
        <y^2>
    irc: scalar or array
        <xy>
    icc: scalar or array
        <x^2>

    returns
    -------
    e1,e2: scalar or array
        The standard ellipticity parameters.
    """
    T = Irr + Icc
    e1 = (Icc - Irr) / T
    e2 = 2.0 * Irc / T

    return e1, e2, T


def mom2g(Irr, Irc, Icc):
    """
    Convert icc, irc, icc to g1,g2

    parameters
    ----------
    irr: scalar or array
        <y^2>
    irc: scalar or array
        <xy>
    icc: scalar or array
        <x^2>

    returns
    -------
    g1,g2: scalar or array
        The reduced shear style shapes.
    """
    e1, e2, T = mom2e(Irr, Irc, Icc)
    g1, g2 = shape.e1e2_to_g1g2(e1, e2)

    return g1, g2, T


def e2mom(e1, e2, T):
    """
    Convert e1,e2,T to icc, irc, icc.

    parameters
    ----------
    e1,e2: scalar or array
        The standard ellipticity parameters
    T: scalar or array
        <x^2 + y^2>

    returns
    -------
    irr: scalar or array
        <y^2>
    irc: scalar or array
        <xy>
    icc: scalar or array
        <x^2>
    """
    Irc = e2 * T / 2.0
    Icc = (1 + e1) * T / 2.0
    Irr = (1 - e1) * T / 2.0

    return Irr, Irc, Icc


def g2mom(g1, g2, T):
    """
    Convert g1,g2,T to icc, irc, icc.

    parameters
    ----------
    g1,g2: scalar or array
        The reduced shear style shapes.
    T: scalar or array
        <x^2 + y^2>

    returns
    -------
    irr: scalar or array
        <y^2>
    irc: scalar or array
        <xy>
    icc: scalar or array
        <x^2>
    """
    e1, e2 = shape.g1g2_to_e1e2(g1, g2)
    Irc = e2 * T / 2.0
    Icc = (1 + e1) * T / 2.0
    Irr = (1 - e1) * T / 2.0

    return Irr, Irc, Icc


def make_mom_result(mom, mom_cov):
    """Make a fitting results dict from a set of moments.

    Parameters
    ----------
    mom : np.ndarray
        The array of moments in the order [Mu, Mv, M1, M2, MT, MF].
    mom_cov : np.ndarray
        The array of moment covariances.

    Returns
    -------
    res : dict
        A dictionary of results.
    """
    # for safety...
    if len(mom) != 6:
        raise ValueError(
            "You must pass exactly 6 moments in the order [Mu, Mv, M1, M2, MT, MF] "
            "for ngmix.moments.make_mom_result."
        )
    if mom_cov.shape != (6, 6):
        raise ValueError(
            "You must pass a 6x6 matrix for ngmix.moments.make_mom_result."
        )

    # now finally build the outputs and their errors
    res = {}
    res["flags"] = 0
    res["flagstr"] = ""
    res["flux"] = mom[5]
    res["mom"] = mom
    res["mom_cov"] = mom_cov
    res["flux_flags"] = 0
    res["flux_flagstr"] = ""
    res["T_flags"] = 0
    res["T_flagstr"] = ""

    # we fill these in later if T > 0 and flux cov is positive
    res["flux_err"] = np.nan
    res["T"] = np.nan
    res["T_err"] = np.nan
    res["s2n"] = np.nan
    res["e1"] = np.nan
    res["e2"] = np.nan
    res["e"] = np.array([np.nan, np.nan])
    res["e_err"] = np.array([np.nan, np.nan])
    res["e_cov"] = np.diag([np.nan, np.nan])
    res["mom_err"] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

    # handle flux-only
    if mom_cov[5, 5] > 0:
        res["flux_err"] = np.sqrt(mom_cov[5, 5])
        res["s2n"] = res["flux"] / res["flux_err"]
    else:
        res["flux_flags"] |= ngmix.flags.NONPOS_VAR

    # handle flux+T only
    if np.all(np.diagonal(mom_cov)[4:6] > 0):
        if mom[5] > 0:
            res["T"] = mom[4] / mom[5]
            res["T_err"] = get_ratio_error(
                mom[4], mom[5],
                mom_cov[4, 4], mom_cov[5, 5], mom_cov[4, 5]
            )
        else:
            # flux <= 0.0
            res["T_flags"] |= ngmix.flags.NONPOS_FLUX
    else:
        res["T_flags"] |= ngmix.flags.NONPOS_VAR

    # now handle full flags
    if np.all(np.diagonal(mom_cov) > 0):
        res["mom_err"] = np.sqrt(np.diagonal(mom_cov))
    else:
        res["flags"] |= ngmix.flags.NONPOS_VAR

    if res["flags"] == 0:
        if mom[5] > 0:
            if res["T"] > 0:
                res["pars"] = np.array([
                    mom[0], mom[1],
                    mom[2]/mom[5],
                    mom[3]/mom[5],
                    mom[4]/mom[5],
                    mom[5],
                ])
                res["e1"] = mom[2] / mom[4]
                res["e2"] = mom[3] / mom[4]
                res["e"] = np.array([res["e1"], res["e2"]])
                e_err = np.zeros(2)
                e_err[0] = get_ratio_error(
                    mom[2], mom[4],
                    mom_cov[2, 2], mom_cov[4, 4], mom_cov[2, 4]
                )
                e_err[1] = get_ratio_error(
                    mom[3], mom[4],
                    mom_cov[3, 3], mom_cov[4, 4], mom_cov[3, 4]
                )
                if np.all(np.isfinite(e_err)):
                    res["e_err"] = e_err
                    res["e_cov"] = np.diag(e_err**2)
                else:
                    # bad e_err
                    res["flags"] |= ngmix.flags.NONPOS_SHAPE_VAR
            else:
                # T <= 0.0
                res["flags"] |= ngmix.flags.NONPOS_SIZE
        else:
            # flux <= 0.0
            res["flags"] |= ngmix.flags.NONPOS_FLUX

    res["flagstr"] = ngmix.flags.get_flags_str(res["flags"])
    res["flux_flagstr"] = ngmix.flags.get_flags_str(res["flux_flags"])
    res["T_flagstr"] = ngmix.flags.get_flags_str(res["T_flags"])

    return res
