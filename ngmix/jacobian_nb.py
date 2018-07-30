from numba import njit

@njit
def jacobian_get_vu(jacob, row, col):
    """
    convert row,col to v,u using the input jacobian
    """

    rowdiff = row - jacob['row0'][0]
    coldiff = col - jacob['col0'][0]

    v = jacob['dvdrow'][0]*rowdiff + jacob['dvdcol'][0]*coldiff
    u = jacob['dudrow'][0]*rowdiff + jacob['dudcol'][0]*coldiff

    return v,u

@njit
def jacobian_get_rowcol(jacob, v, u):
    """
    convert v,u to row,col using the input jacobian
    """

    rowdiff =  jacob['dudcol'][0]*v - jacob['dvdcol'][0]*u
    coldiff = -jacob['dudrow'][0]*v + jacob['dvdrow'][0]*u

    row = jacob['row0'][0] + rowdiff/jacob['det'][0]
    col = jacob['col0'][0] + coldiff/jacob['det'][0]

    return row, col
