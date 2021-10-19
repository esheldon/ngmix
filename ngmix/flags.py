NO_ATTEMPT = 2**0
CEN_SHIFT = 2**1
NONPOS_FLUX = 2**2
NONPOS_SIZE = 2**3
LOW_DET = 2**4
MAXITER = 2**5
NONPOS_VAR = 2**6
GMIX_RANGE_ERROR = 2**7
NONPOS_SHAPE_VAR = 2**8

# flags for LM fitting diagnostics
LM_SINGULAR_MATRIX = 2 ** 9
LM_NEG_COV_EIG = 2 ** 10
LM_NEG_COV_DIAG = 2 ** 11
LM_FUNC_NOTFINITE = 2 ** 12

# for LM this indicates a the eigenvalues of the covariance cannot be found
EIG_NOTFINITE = 2 ** 13

DIV_ZERO = 2 ** 14  # division by zero
ZERO_DOF = 2 ** 15  # dof zero so can't do chi^2/dof

# these mappings keep the API the same
EM_RANGE_ERROR = GMIX_RANGE_ERROR
EM_MAXITER = MAXITER
BAD_VAR = NONPOS_VAR

NAME_MAP = {
    # no attempt was made to measure this object, usually
    # due to a previous step in the code fails.
    NO_ATTEMPT: 'no attempt',

    # flag for the center shifting too far
    # used by admom
    CEN_SHIFT: 'center shifted too far',

    NONPOS_FLUX: 'flux <= 0',
    NONPOS_SIZE: 'T <= 0',
    LOW_DET: 'determinant near zero',
    MAXITER: 'max iterations reached',
    NONPOS_VAR: 'non-positive (definite) variance',
    NONPOS_SHAPE_VAR: 'non-positive shape variance',
    GMIX_RANGE_ERROR: 'GMixRangeError raised',

    LM_SINGULAR_MATRIX: 'singular matrix in LM',
    LM_NEG_COV_EIG: 'negative covariance eigenvalue in LM',
    LM_NEG_COV_DIAG: 'negative covariance diagional value in LM',
    LM_FUNC_NOTFINITE: 'function not finite in LM',

    # for LM this indicates a the eigenvalues of the covariance cannot be found
    EIG_NOTFINITE: 'eigenvalues of covariance cannot be found in LM',

    DIV_ZERO: 'divide by zero',
    ZERO_DOF: 'degrees of freedom for it is zero (no chi^2/dof possible)',
}

INV_NAME_MAP = {}
for k, v in list(NAME_MAP.items()):
    INV_NAME_MAP[v] = k


def get_flags_str(val, name_map=None):
    """Get a descriptive string given a flag value.

    Parameters
    ----------
    val : int
        The flag value.
    name_map : dict, optional
        A dictionary mapping names to values. Default is global at
        ngmix.flags.NAME_MAP.

    Returns
    -------
    flagstr : str
        A string of descriptions for each bit separated by `|`.
    """
    if name_map is None:
        inv_name_map = INV_NAME_MAP
    else:
        inv_name_map = {}
        for k, v in name_map.items():
            inv_name_map[v] = k

    nstrs = []
    for pow in range(32):
        fval = 2**pow
        if ((val & fval) != 0):
            if fval in inv_name_map:
                nstrs.append(inv_name_map[fval])
            else:
                nstrs.append("bit 2**%d" % pow)
    return "|".join(nstrs)
