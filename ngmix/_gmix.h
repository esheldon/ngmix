#ifndef _PYGMIX_HEADER_GUARD
#define _PYGMIX_HEADER_GUARD

#include <Python.h>
#include <numpy/arrayobject.h> 

#define PYGMIX_LOW_DETVAL 1.0e-200

enum PyGMix_Models {
    PyGMIX_GMIX_FULL=0,
    PyGMIX_GMIX_GAUSS=1,
    PyGMIX_GMIX_TURB=2,
    PyGMIX_GMIX_EXP=3,
    PyGMIX_GMIX_DEV=4,
    PyGMIX_GMIX_BDC=5,
    PyGMIX_GMIX_BDF=6,
    PyGMIX_GMIX_COELLIP=7,
    PyGMIX_GMIX_SERSIC=8,
    PYGMIX_GMIX_FRACDEV=9,
    PYGMIX_GMIX_SWINDLE=10,
    PYGMIX_GMIX_GAUSSMOM=11,
    PyGMIX_GMIX_COELLIP4=100
};

struct pixel {
    double u;
    double v;

    double val;

    double ierr;

    double fdiff;
};

struct coord {
    double u;
    double v;
};


struct gauss {
    double p;
    double row;
    double col;

    double irr;
    double irc;
    double icc;

    double det;

    int64_t norm_set;

    double drr;
    double drc;
    double dcc;

    double norm;
    double pnorm;
};

struct PyGMixCM {
    double fracdev;
    double TdByTe; // ratio Tdev/Texp
    double Tfactor;
    struct gauss gmix[16];
};

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

struct PyGMix_EM_Sums {
    double gi;

    // scratch on a given pixel
    double trowsum;
    double tcolsum;
    double tu2sum;
    double tuvsum;
    double tv2sum;

    // sums over all pixels
    double pnew;
    double rowsum;
    double colsum;
    double u2sum;
    double uvsum;
    double v2sum;
};


/*
 *
 * fast exponential function
 *
 */

union pygmix_fmath_di {
    double d;
    uint64_t i;
};

static inline double expd(double x)
{

// holds definition of the table and C1,C2,C3, a, ra
#include "fmath-dtbl.c"

    union pygmix_fmath_di di;
    uint64_t iax, u;
    double t, y;

    di.d = x * a + b;
    iax = dtbl[di.i & sbit_masked];

    t = (di.d - b) * ra - x;
    u = ((di.i + adj) >> sbit) << 52;
    y = (C3 - t) * (t * t) * C2 - t + C1;

    di.i = u | iax;
    return y * di.d;
}


// will check > -26 and < 0.0 so these are not actually necessary
//static int _exp3_ivals[] = {-26, -25, -24, -23, -22, -21, 
//                            -20, -19, -18, -17, -16, -15, -14,
//                            -13, -12, -11, -10,  -9,  -8,  -7,
//                            -6,  -5,  -4,  -3,  -2,  -1,   0};

/*
static int _exp3_i0=-26;
static double _exp3_lookup[] = {  5.10908903e-12,   1.38879439e-11,   3.77513454e-11,
                                  1.02618796e-10,   2.78946809e-10,   7.58256043e-10,
                                  2.06115362e-09,   5.60279644e-09,   1.52299797e-08,
                                  4.13993772e-08,   1.12535175e-07,   3.05902321e-07,
                                  8.31528719e-07,   2.26032941e-06,   6.14421235e-06,
                                  1.67017008e-05,   4.53999298e-05,   1.23409804e-04,
                                  3.35462628e-04,   9.11881966e-04,   2.47875218e-03,
                                  6.73794700e-03,   1.83156389e-02,   4.97870684e-02,
                                  1.35335283e-01,   3.67879441e-01,   1.00000000e+00};

static inline double pygmix_exp3(double x) {

    int ival = (int)(x-0.5);
    double f = x - ival;
    int index = ival-_exp3_i0;
    double expval = _exp3_lookup[index];
    expval *= (6+f*(6+f*(3+f)))*0.16666666;

    return expval;
}
*/




#define PYGMIX_MAX_CHI2 25.0
#define PYGMIX_MAX_CHI2_FAST 300.0


/*
static inline double gauss_eval(const struct gauss *gauss,
                                double row,
                                double col)
{
    double v = row-gauss->row;
    double u = col-gauss->col;
    double val=0.0;

    double chi2 =
          gauss->dcc*v*v
        + gauss->drr*u*u
        - 2.0*gauss->drc*v*u;

    if (chi2 < PYGMIX_MAX_CHI2 && chi2 >= 0.0) {
        val = gauss->pnorm*expd( -0.5*chi2 );
    }
    return val;
}

static inline double gmix_eval(const struct gauss* gmix,
                               npy_intp n_gauss,
                               double row,
                               double col)
{
    const struct gauss* gauss=NULL;
    int i=0, igauss=0;
    double vdiff=0, udiff=0, chi2=0;
    double val=0.0;

    for (i=0; i< n_gauss; i++) {
        for (igauss=0; igauss < n_gauss; igauss++) {
            gauss = &gmix[igauss];

            vdiff = row - gauss->row;
            udiff = col - gauss->col;

            chi2 =       gauss->dcc*vdiff*vdiff
                   +     gauss->drr*udiff*udiff
                   - 2.0*gauss->drc*vdiff*udiff;

            if (chi2 < PYGMIX_MAX_CHI2 && chi2 >= 0.0) {
                val += gauss->pnorm*expd( -0.5*chi2 );
            }

            //val += gauss_eval(gauss, row, col);
        }
    }
    return val;
}
*/




#define PYGMIX_GAUSS_EVAL_FULL(gauss, rowval, colval) ({       \
    double _vtmp = (rowval)-(gauss)->row;                      \
    double _utmp = (colval)-(gauss)->col;                      \
    double _g_val=0.0;                                         \
                                                               \
    double _chi2 =                                             \
          (gauss)->dcc*_vtmp*_vtmp                             \
        + (gauss)->drr*_utmp*_utmp                             \
        - 2.0*(gauss)->drc*_vtmp*_utmp;                        \
                                                               \
    if (_chi2 >= 0.0) {                                        \
        _g_val = (gauss)->pnorm*exp( -0.5*_chi2 );             \
    }                                                          \
                                                               \
    _g_val;                                                    \
})



#define PYGMIX_GAUSS_EVAL(gauss, rowval, colval) ({            \
    double _vtmp = (rowval)-(gauss)->row;                      \
    double _utmp = (colval)-(gauss)->col;                      \
    double _g_val=0.0;                                         \
                                                               \
    double _chi2 =                                             \
          (gauss)->dcc*_vtmp*_vtmp                             \
        + (gauss)->drr*_utmp*_utmp                             \
        - 2.0*(gauss)->drc*_vtmp*_utmp;                        \
                                                               \
    if (_chi2 < PYGMIX_MAX_CHI2 && _chi2 >= 0.0) {             \
        _g_val = (gauss)->pnorm*expd( -0.5*_chi2 );            \
    }                                                          \
                                                               \
    _g_val;                                                    \
})

#define PYGMIX_GAUSS_EVAL_FAST(gauss, rowval, colval) ({       \
    double _u = (rowval)-(gauss)->row;                         \
    double _v = (colval)-(gauss)->col;                         \
    double _g_val=0.0;                                         \
                                                               \
    double _chi2 =                                             \
          (gauss)->dcc*_u*_u                                   \
        + (gauss)->drr*_v*_v                                   \
        - 2.0*(gauss)->drc*_u*_v;                              \
                                                               \
    if (_chi2 < PYGMIX_MAX_CHI2_FAST && _chi2 >= 0.0) {	       \
        _g_val = (gauss)->pnorm*expd( -0.5*_chi2 );            \
    }                                                          \
                                                               \
    _g_val;                                                    \
})

// using full exp() function
#define PYGMIX_GMIX_EVAL_FULL(gmixin, n_gauss, rowval, colval) ({          \
    int _i=0;                                                            \
    double _gm_val=0.0;                                                  \
    const struct gauss* _gauss=(gmixin);                                          \
    for (_i=0; _i< (n_gauss); _i++) {                                    \
        _gm_val += PYGMIX_GAUSS_EVAL_FULL(_gauss, (rowval), (colval));   \
        _gauss++;                                                        \
    }                                                                    \
    _gm_val;                                                             \
})

// using approximate exp() function
#define PYGMIX_GMIX_EVAL(gmixin, n_gauss, rowval, colval) ({     \
    int _i=0;                                                  \
    double _gm_val=0.0;                                        \
    const struct gauss* _gauss=(gmixin);                        \
    for (_i=0; _i< (n_gauss); _i++) {                          \
        _gm_val += PYGMIX_GAUSS_EVAL(_gauss, (rowval), (colval)); \
        _gauss++;                                              \
    }                                                          \
    _gm_val;                                                   \
})

// using approximate exp() function w/ no cut in chi2
#define PYGMIX_GMIX_EVAL_FAST(gmixin, n_gauss, rowval, colval) ({		\
      int _i=0;								\
    double _gm_val=0.0;							\
    const struct gauss* _gauss=(gmixin);					\
    for (_i=0; _i< (n_gauss); _i++) {					\
        _gm_val += PYGMIX_GAUSS_EVAL_FAST(_gauss, (rowval), (colval));  \
        _gauss++;							\
    }									\
    _gm_val;								\
})

#define PYGMIX_JACOB_GETU(jacob, row, col) ({           \
    double _u_val;                                      \
    _u_val=(jacob)->dudrow*((row) - (jacob)->row0)        \
           + (jacob)->dudcol*((col) - (jacob)->col0);     \
    _u_val;                                             \
})

#define PYGMIX_JACOB_GETV(jacob, row, col) ({           \
    double _v_val;                                      \
    _v_val=(jacob)->dvdrow*((row) - (jacob)->row0)        \
           + (jacob)->dvdcol*((col) - (jacob)->col0);     \
    _v_val;                                             \
})

#define PYGMIX_PACK_RESULT3(s2n_numer, s2n_denom, npix) do {      \
    retval=PyTuple_New(3);                                        \
    long _npix = (npix);                                          \
    PyTuple_SetItem(retval,0,PyFloat_FromDouble( (s2n_numer) ));  \
    PyTuple_SetItem(retval,1,PyFloat_FromDouble( (s2n_denom) ));  \
    PyTuple_SetItem(retval,2,PyLong_FromLong(_npix));             \
} while(0)


#define PYGMIX_PACK_RESULT4(loglike, s2n_numer, s2n_denom, npix) do { \
    retval=PyTuple_New(4);                                        \
    long _npix = (npix);                                          \
    PyTuple_SetItem(retval,0,PyFloat_FromDouble( (loglike) ));    \
    PyTuple_SetItem(retval,1,PyFloat_FromDouble( (s2n_numer) ));  \
    PyTuple_SetItem(retval,2,PyFloat_FromDouble( (s2n_denom) ));  \
    PyTuple_SetItem(retval,3,PyLong_FromLong( _npix ));           \
} while(0)


// pvals->counts
// fvals->T
// pvals sum to 1
// (pvals*fvals) sum to 1

static const double PyGMix_pvals_exp[] = {
    0.00061601229677880041, 
    0.0079461395724623237, 
    0.053280454055540001, 
    0.21797364640726541, 
    0.45496740582554868, 
    0.26521634184240478};

static const double PyGMix_fvals_exp[] = {
    0.002467115141477932, 
    0.018147435573256168, 
    0.07944063151366336, 
    0.27137669897479122, 
    0.79782256866993773, 
    2.1623306025075739};

static const double PyGMix_pvals_dev[] = {
    6.5288960012625658e-05,
    0.00044199216814302695, 
    0.0020859587871659754, 
    0.0075913681418996841, 
    0.02260266219257237, 
    0.056532254390212859, 
    0.11939049233042602, 
    0.20969545753234975, 
    0.29254151133139222, 
    0.28905301416582552};

static const double PyGMix_fvals_dev[] = {
    3.068330909892871e-07,
    3.551788624668698e-06,
    2.542810833482682e-05,
    0.0001466508940804874,
    0.0007457199853069548,
    0.003544702600428794,
    0.01648881157673708,
    0.07893194619504579,
    0.4203787615506401,
    3.055782252301236};

static const double PyGMix_pvals_turb[] = {
    0.596510042804182,0.4034898268889178,1.303069003078001e-07};

static const double PyGMix_fvals_turb[] = {
    0.5793612389470884,1.621860687127999,7.019347162356363};

static const double PyGMix_pvals_gauss[] = {1.0};
static const double PyGMix_fvals_gauss[] = {1.0};

#define PYGMIX_MAXDIMS 10
#define PYGMIX_DOFFSET 2

#endif
