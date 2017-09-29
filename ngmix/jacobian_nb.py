import numpy
from numba import jit, njit

@njit(cache=True)
def jacobian_get_vu(jacob, row, col):
    """
    convert row,col to v,u using the input jacobian
    """

    rowdiff = row - jacob['row0'][0]
    coldiff = col - jacob['col0'][0]

    u = jacob['dudrow'][0]*rowdiff + jacob['dudcol'][0]*coldiff
    v = jacob['dvdrow'][0]*rowdiff + jacob['dvdcol'][0]*coldiff

    return v,u
