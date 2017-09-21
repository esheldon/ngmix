#ifndef _PYGMIX_JACOBIAN_HEADER_GUARD
#define _PYGMIX_JACOBIAN_HEADER_GUARD

struct jacobian {
    double row0;
    double col0;

    double dvdrow;
    double dvdcol;
    double dudrow;
    double dudcol;

    double det;
    double sdet;
};

#define PYGMIX_JACOB_GETU(jacob, row, col) ({           \
    double _u_val;                                      \
    _u_val=(jacob)->dudrow*((row) - (jacob)->row0)      \
           + (jacob)->dudcol*((col) - (jacob)->col0);   \
    _u_val;                                             \
})

#define PYGMIX_JACOB_GETV(jacob, row, col) ({           \
    double _v_val;                                      \
    _v_val=(jacob)->dvdrow*((row) - (jacob)->row0)      \
           + (jacob)->dvdcol*((col) - (jacob)->col0);   \
    _v_val;                                             \
})



#endif
