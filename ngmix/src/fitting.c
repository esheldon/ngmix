//   Calculate the image mean, accounting for weight function.

#include "fitting.h"
#include "gmix.h"

extern PyObject* GMixRangeError;
extern PyObject* GMixFatalError;

double get_loglike_pixels(struct gauss *gmix,
                          npy_intp n_gauss,
                          const struct pixel *pixels,
                          npy_intp n_pixels,
                          int *status)
{

    npy_intp ipixel=0, igauss=0;

    struct gauss *gauss=NULL;
    const struct pixel *pixel=NULL;

    double
        model_val=0, diff=0,
        chi2=0, udiff=0, vdiff=0,
        loglike=-9999;

    if (!gmix_set_norms_if_needed(gmix, n_gauss)) {
        goto _bail;
    }

#pragma omp parallel for \
        default(none) \
        shared(gmix,pixels,n_pixels,n_gauss) \
        private(ipixel,pixel,igauss,gauss,vdiff,udiff,chi2,model_val,diff) \
		reduction(+:loglike)

    for (ipixel=0; ipixel < n_pixels; ipixel++) {
        pixel = &pixels[ipixel];

        model_val=0;
        for (igauss=0; igauss < n_gauss; igauss++) {
            gauss = &gmix[igauss];

            // v->row, u->col in gauss
            vdiff = pixel->v - gauss->row;
            udiff = pixel->u - gauss->col;

            chi2 =       gauss->dcc*vdiff*vdiff
                   +     gauss->drr*udiff*udiff
                   - 2.0*gauss->drc*vdiff*udiff;

            if (chi2 < PYGMIX_MAX_CHI2 && chi2 >= 0.0) {
                model_val += gauss->pnorm*expd( -0.5*chi2 );
            }

        }

        diff = model_val-pixel->val;
        loglike += diff*diff*pixel->ierr*pixel->ierr;

    }

    loglike *= (-0.5);

    *status=1;
_bail:
    return loglike;
}




