#ifndef _PYGMIX_GMIX_HEADER_GUARD
#define _PYGMIX_GMIX_HEADER_GUARD

#include <stdint.h>
#include "fastexp.h"

#define PYGMIX_LOW_DETVAL 1.0e-200

#define PYGMIX_MAX_CHI2 25.0
#define PYGMIX_MAX_CHI2_FAST 300.0

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


void gmix_get_cen(const struct gauss *self,
                  long n_gauss,
                  double* row,
                  double *col,
                  double *psum);

int gmix_get_e1e2T(struct gauss *gmix,
                   long n_gauss,
                   double *e1, double *e2, double *T);

int gauss2d_set_norm(struct gauss *self, int dothrow);
int gauss2d_set_norm_throw(struct gauss *self);
int gauss2d_set(struct gauss *self,
                double p,
                double row,
                double col,
                double irr,
                double irc,
                double icc);

int get_n_gauss(int model, int *status);

int gmix_fill_full(struct gauss *self,
                   long n_gauss,
                   const double* pars,
                   long n_pars);

int gmix_set_norms(struct gauss *self,
                   long n_gauss);


int gmix_set_norms_if_needed(struct gauss *self,
                             long n_gauss);

int gmix_fill_simple(struct gauss *self,
                     long n_gauss,
                     const double* pars,
                     long n_pars,
                     int model,
                     const double* fvals,
                     const double* pvals);

int gmix_fill_cm(struct PyGMixCM*self,
                 const double* pars,
                 long n_pars);


int gmix_fill_coellip(struct gauss *self,
                      long n_gauss,
                      const double* pars,
                      long n_pars);

double get_cm_Tfactor(double fracdev, double TdByTe);

int gmix_fill(struct gauss *self,
              long n_gauss,
              const double* pars,
              long n_pars,
              int model);

int convolve_fill(struct gauss *self, long self_n_gauss,
                  const struct gauss *gmix, long n_gauss,
                  const struct gauss *psf, long psf_n_gauss);


#endif
