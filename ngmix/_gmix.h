#ifndef _PYGMIX_HEADER_GUARD
#define _PYGMIX_HEADER_GUARD

#include <Python.h>
#include <numpy/arrayobject.h> 


#define PYGMIX_MAXDIMS 10
#define PYGMIX_DOFFSET 2


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






#endif
