NO_ATTEMPT = 2**0
BIG_CEN_SHIFT = 2**1
NONPOS_FLUX = 2**2
NONPOS_SIZE = 2**3
LOW_DET = 2**4
MAXIT = 2**5
NONPOS_VAR = 2**6
GMIX_RANGE_ERROR = 2**7
NONPOS_SHAPE_VAR = 2**8

NAME_MAP = {
    # no attempt was made to measure this object, usually
    # due to a previous step in the code fails.
    NO_ATTEMPT: 'no attempt',

    # flag for the center shifting too far
    # used by admom
    BIG_CEN_SHIFT: 'center shifted too far',

    NONPOS_FLUX: 'flux <= 0',
    NONPOS_SIZE: 'T <= 0',
    LOW_DET: 'determinant near zero',
    MAXIT: 'maxit reached',
    NONPOS_VAR: 'non-positive variance',
    NONPOS_SHAPE_VAR: 'non-positive shape variance',
    GMIX_RANGE_ERROR: 'GMixRangeError raised',
}

for k, v in list(NAME_MAP.items()):
    NAME_MAP[v] = k


def get_procflags_str(val, name_map=None):
    """Get a descriptive string given a flag value.

    Parameters
    ----------
    val : int
        The flag value.
    name_map : dict, optional
        A dictionary mapping names to values. Default is global at
        ngmix.procflags.NAME_MAP.

    Returns
    -------
    flagstr : str
        A string of descriptions for each bit separated by `|`.
    """
    if name_map is None:
        name_map = NAME_MAP

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
