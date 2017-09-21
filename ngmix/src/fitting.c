//   Calculate the image mean, accounting for weight function.

#include <stdlib.h>
#include "fitting.h"

/*
   norms should be set before entry
*/
double get_loglike(const struct gauss *gmix,
                   long n_gauss,
                   const struct pixel *pixels,
                   long n_pixels)
{

    long ipixel=0;
    const struct pixel *pixel=NULL;

    double model_val=0, diff=0, loglike=-9999;

    for (ipixel=0; ipixel < n_pixels; ipixel++) {
        pixel = &pixels[ipixel];

        model_val=PYGMIX_GMIX_EVAL(gmix, n_gauss, pixel->v, pixel->u);

        diff = model_val-pixel->val;

        loglike += diff*diff*pixel->ierr*pixel->ierr;

    }

    loglike *= (-0.5);

    return loglike;
}


/*
   also get s2n sums
   norms should be set before entry
*/
void get_loglike_more(const struct gauss *gmix,
                      long n_gauss,
                      const struct pixel *pixels,
                      long n_pixels,
                      double *loglike,
                      double *s2n_numer,
                      double *s2n_denom,
                      long *npix)
{

    long ipixel=0;
    const struct pixel *pixel=NULL;

    double model_val=0, diff=0, ivar=0;


    *loglike = *s2n_numer = *s2n_denom = *npix = 0;

    for (ipixel=0; ipixel < n_pixels; ipixel++) {
        pixel = &pixels[ipixel];
        ivar = pixel->ierr*pixel->ierr;

        model_val=PYGMIX_GMIX_EVAL(gmix, n_gauss, pixel->v, pixel->u);

        diff = model_val - pixel->val;

        if (ivar > 0.0) {
            *loglike   += diff*diff*ivar;
            *s2n_numer += pixel->val * model_val * ivar;
            *s2n_denom += model_val * model_val * ivar;
            *npix += 1;
        }
    }

    *loglike *= (-0.5);

}

void fill_fdiff(struct pixel* pixels,     // array of pixels
                long n_pixels,        // number of pixels in array
                const struct gauss* gmix, // the gaussian mixture
                long n_gauss)         // number of gaussians

{

    struct pixel* pixel=NULL;
    double model_val=0;
    long ipixel=0;

    for (ipixel=0; ipixel < n_pixels; ipixel++) {
        pixel = &pixels[ipixel];

        model_val=PYGMIX_GMIX_EVAL(gmix, n_gauss, pixel->v, pixel->u);

        pixel->fdiff = (model_val-pixel->val)*pixel->ierr;
    }
}

/*
static void fill_fdiff_exp3(
    const struct pixel* pixels,        // array of pixels
    long n_pixels,                 // number of pixels in array
    double *fdiff,                     // same size as pixels
    const struct gauss* gmix, // the gaussian mixture
    long n_gauss)                  // number of gaussians

{

    const struct pixel* pixel=NULL;
    const struct gauss* gauss=NULL;

    //int ival, index;
    double
        model_val=0, vdiff=0, udiff=0, chi2=0,
        diff=0;
//        x, f, expval;
    
    long ipixel=0, igauss=0;

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
                model_val += gauss->pnorm*pygmix_exp3( -0.5*chi2 );
            }

        }

        diff = model_val-pixel->val;
        diff *= pixel->ierr;
        fdiff[ipixel] = diff;
    }

}
*/




/*
    //struct gauss *gauss=NULL;
#pragma omp parallel for \
        default(none) \
        shared(gmix,n_gauss,pixels,n_pixels) \
        private(ipixel,pixel,igauss,gauss,vdiff,udiff,chi2,model_val,diff) \
		reduction(+:loglike)
*/

/*
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
*/




