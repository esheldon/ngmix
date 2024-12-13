import numpy as np

from .gexceptions import GMixRangeError
from . import shape
import ngmix.flags
from .util import get_ratio_error

MOMENTS_NAME_MAP = {
    # v
    "Mv": 0,
    # u
    "Mu": 1,
    # u^2 - v^2
    "M1": 2,
    # v * u
    "M2": 3,
    # v^2 + u^2
    "MT": 4,
    # 1 (flux)
    "MF": 5,

    # notation from piff.util.calculate_moments
    # third order

    # these are same as above but with the alternative notation
    "M00": 5,  # same as MF
    "M10": 1,  # same as Mu
    "M01": 0,  # same as Mv
    "M11": 4,  # same as MT
    "M20": 2,  # same as M1
    "M02": 3,  # same as M2

    # u * r^2
    "M21": 6,
    # v * r^2
    "M12": 7,
    # u * (u^2 - 3 * v^2)
    "M30": 8,
    # v * (3 * u^2 - v^2)
    "M03": 9,

    # fourth order
    # r^4
    "M22": 10,
    # r^2 * (u^ - v^)
    "M31": 11,
    # r^2 * 2 * u * v
    "M13": 12,
    # u^4 - 6 * u^2 * v^2 + v^4
    "M40": 13,
    # (u^2 - v^2) * 4 * u * v
    "M14": 14,

    # 6th order
    # r^6
    "M33": 15,

    # 8th order
    # r^8
    "M44": 16,
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


def make_mom_result(sums, sums_cov, sums_norm=None):
    """Make a fitting results dict from a set of unnormalized moments.

    Parameters
    ----------
    sums : np.ndarray
        The array of unnormalized moments in the order [Mv, Mu, M1, M2, MT, MF].
    sums_cov : np.ndarray
        The array of unnormalized moment covariances.
    sums_norm : float, optional
        The sum of the moment weight function itself. This is added to the output data.
        The default of None puts in NaN.

    Returns
    -------
    res : dict
        A dictionary of results.
    """
    # for safety...
    if len(sums) != 6 and len(sums) != 17:
        raise ValueError(
            "You must pass exactly 6 or 17 unnormalized moments in the order "
            "[Mv, Mu, M1, M2, MT, MF, ...] for ngmix.moments.make_mom_result."
        )
    if sums_cov.shape != (6, 6) and sums_cov.shape != (17, 17):
        raise ValueError(
            "You must pass a 6x6 or 17x17 matrix for ngmix.moments.make_mom_result."
        )

    mv_ind = MOMENTS_NAME_MAP["Mv"]
    mu_ind = MOMENTS_NAME_MAP["Mu"]
    mf_ind = MOMENTS_NAME_MAP["MF"]
    mt_ind = MOMENTS_NAME_MAP["MT"]
    m1_ind = MOMENTS_NAME_MAP["M1"]
    m2_ind = MOMENTS_NAME_MAP["M2"]

    # now finally build the outputs and their errors
    res = {}
    res["flags"] = 0
    res["flagstr"] = ""
    res["flux"] = sums[mf_ind]
    res["sums"] = sums
    res["sums_cov"] = sums_cov
    res["sums_norm"] = sums_norm if sums_norm is not None else np.nan
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
    res["sums_err"] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

    # handle flux-only
    if sums_cov[mf_ind, mf_ind] > 0:
        res["flux_err"] = np.sqrt(sums_cov[mf_ind, mf_ind])
        res["s2n"] = res["flux"] / res["flux_err"]
    else:
        res["flux_flags"] |= ngmix.flags.NONPOS_VAR

    # handle flux+T only
    if sums_cov[mf_ind, mf_ind] > 0 and sums_cov[mt_ind, mt_ind] > 0:
        if sums[mf_ind] > 0:
            res["T"] = sums[mt_ind] / sums[mf_ind]
            res["T_err"] = get_ratio_error(
                sums[mt_ind],
                sums[mf_ind],
                sums_cov[mt_ind, mt_ind],
                sums_cov[mf_ind, mf_ind],
                sums_cov[mt_ind, mf_ind],
            )
        else:
            # flux <= 0.0
            res["T_flags"] |= ngmix.flags.NONPOS_FLUX
    else:
        res["T_flags"] |= ngmix.flags.NONPOS_VAR

    # now handle full flags
    if np.all(np.diagonal(sums_cov) > 0):
        res["sums_err"] = np.sqrt(np.diagonal(sums_cov))
    else:
        res["flags"] |= ngmix.flags.NONPOS_VAR

    if res["flags"] == 0:
        if res["flux"] > 0:
            if res["T"] > 0:
                res["e1"] = sums[m1_ind] / sums[mt_ind]
                res["e2"] = sums[m2_ind] / sums[mt_ind]
                res["e"] = np.array([res["e1"], res["e2"]])

                res["pars"] = np.array([
                    sums[mv_ind],
                    sums[mu_ind],
                    res["e1"],
                    res["e2"],
                    res["T"],
                    res["flux"],
                ])

                e_err = np.zeros(2)
                e_err[0] = get_ratio_error(
                    sums[m1_ind],
                    sums[mt_ind],
                    sums_cov[m1_ind, m1_ind],
                    sums_cov[mt_ind, mt_ind],
                    sums_cov[m1_ind, mt_ind],
                )
                e_err[1] = get_ratio_error(
                    sums[m2_ind],
                    sums[mt_ind],
                    sums_cov[m2_ind, m2_ind],
                    sums_cov[mt_ind, mt_ind],
                    sums_cov[m2_ind, mt_ind]
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

    _add_moments_by_name(res)

    return res


def _add_moments_by_name(res):
    sums = res['sums']
    sums_cov = res['sums_cov']

    mf_ind = MOMENTS_NAME_MAP["MF"]
    fsum = sums[mf_ind]
    fsum_err = np.sqrt(sums_cov[mf_ind, mf_ind])

    # add in named sums normalized by flux sum (weight * image).sum()
    # we don't store flags or errors for these
    mkeys = list(MOMENTS_NAME_MAP.keys())
    for name in mkeys:
        ind = MOMENTS_NAME_MAP[name]
        if ind > sums.size-1:
            continue

        err_name = f'{name}_err'

        if name in ['MF', 'M00']:
            res[name] = fsum
            res[err_name] = fsum_err
        else:
            if fsum > 0:
                res[name] = sums[ind] / fsum
                res[err_name] = get_ratio_error(
                    sums[ind],
                    sums[mf_ind],
                    sums_cov[ind, ind],
                    sums_cov[mf_ind, mf_ind],
                    sums_cov[ind, mf_ind],
                )
            else:
                res[name] = np.nan
                res[err_name] = np.nan


def regularize_mom_shapes(res, fwhm_reg):
    """Apply regularization to the shapes computed from moments sums.

    This routine transforms the shapes as

        e_{1,2} = M_{1,2}/(T + T_reg)

    where T_reg is the T value equivalent to fwhm_reg, T is the original T value
    from the moments, and M_{1,2} are the moments for shapes e_{1,2}.

    This form of regularization is equivalent, for Gaussians, to convolving the Gaussian
    with an isotropic Gaussian smoothing kernel of size fwhm_reg.

    Parameters
    ----------
    res : dict
        The original moments result before regularization.
    fwhm_reg : float
        The regularization FWHM value. Typically this should be of order the size of
        the PSF for a pre-PSF moment.

    Returns
    -------
    res_reg : dict
        The regularized moments result. The size and flux are unchanged.
    """
    if fwhm_reg > 0:
        raw_mom = res["sums"]
        raw_mom_cov = res["sums_cov"]

        T_reg = fwhm_to_T(fwhm_reg)

        # the moments are not normalized and are sums, so convert T_reg to a sum using
        # the flux sum first via T_reg -> T_reg * raw_mom[5]
        amat = np.eye(6)
        amat[4, 5] = T_reg

        # the pre-PSF fitters do not fill out the centroid moments so hack around that
        raw_mom_orig = raw_mom.copy()
        if np.isnan(raw_mom_orig[0]):
            raw_mom[0] = 0
        if np.isnan(raw_mom_orig[1]):
            raw_mom[1] = 0
        reg_mom = np.dot(amat, raw_mom)
        if np.isnan(raw_mom_orig[0]):
            raw_mom[0] = np.nan
            reg_mom[0] = np.nan
        if np.isnan(raw_mom_orig[1]):
            raw_mom[1] = np.nan
            reg_mom[1] = np.nan

        reg_mom_cov = np.dot(amat, np.dot(raw_mom_cov, amat.T))
        momres = make_mom_result(reg_mom, reg_mom_cov)

        # use old T
        for col in ["T", "T_err", "T_flags", "T_flagstr"]:
            momres[col] = res[col]

        momres["flags"] |= res["flags"]
        momres["flagstr"] = ngmix.flags.get_flags_str(momres["flags"])

        return momres
    else:
        return res
