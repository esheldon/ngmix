def make_exp_lookup(minval=-100, maxval=100, dtype='f8'):
    """
    lookup array in range [minval,0] inclusive
    """
    nlook = abs(maxval-minval)+1
    expvals=numpy.zeros(nlook, dtype=dtype)

    ivals = numpy.arange(minval,maxval+1,dtype='i4')

    index=0

    expvals[:] = numpy.exp(ivals)

    return ivals, expvals
