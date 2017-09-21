//   Calculate the image mean, accounting for weight function.

#include "fitting.h"
#include "gmix.h"

/*
   norms should be set before entry
*/
double get_loglike(const struct gauss *gmix,
                   npy_intp n_gauss,
                   const struct pixel *pixels,
                   npy_intp n_pixels)
{

    npy_intp ipixel=0;
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
                      npy_intp n_gauss,
                      const struct pixel *pixels,
                      npy_intp n_pixels,
                      double *loglike,
                      double *s2n_numer,
                      double *s2n_denom,
                      long *npix)
{

    npy_intp ipixel=0;
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
                npy_intp n_pixels,        // number of pixels in array
                const struct gauss* gmix, // the gaussian mixture
                npy_intp n_gauss)         // number of gaussians

{

    struct pixel* pixel=NULL;
    double model_val=0;
    npy_intp ipixel=0;

    for (ipixel=0; ipixel < n_pixels; ipixel++) {
        pixel = &pixels[ipixel];

        model_val=PYGMIX_GMIX_EVAL(gmix, n_gauss, pixel->v, pixel->u);

        pixel->fdiff = (model_val-pixel->val)*pixel->ierr;
    }
}

/*
static void fill_fdiff_exp3(
    const struct pixel* pixels,        // array of pixels
    npy_intp n_pixels,                 // number of pixels in array
    double *fdiff,                     // same size as pixels
    const struct gauss* gmix, // the gaussian mixture
    npy_intp n_gauss)                  // number of gaussians

{

    const struct pixel* pixel=NULL;
    const struct gauss* gauss=NULL;

    //int ival, index;
    double
        model_val=0, vdiff=0, udiff=0, chi2=0,
        diff=0;
//        x, f, expval;
    
    npy_intp ipixel=0, igauss=0;

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



/*
   Calculate the loglike between the gmix and the input image

   Error checking should be done in python.
*/
/*
static PyObject * PyGMix_get_loglike_image(PyObject* self, PyObject* args) {

    PyObject* gmix_obj=NULL;
    PyObject* image_obj=NULL;
    PyObject* weight_obj=NULL;
    PyObject* jacob_obj=NULL;
    npy_intp n_gauss=0, n_row=0, n_col=0, row=0, col=0;//, igauss=0;

    struct gauss *gmix=NULL;//, *gauss=NULL;
    struct jacobian *jacob=NULL;

    double data=0, ivar=0, u=0, v=0;
    double model_val=0, diff=0;
    double s2n_numer=0.0, s2n_denom=0.0, loglike = 0.0;

    long npix = 0;

    PyObject* retval=NULL;

    if (!PyArg_ParseTuple(args, (char*)"OOOO", 
                          &gmix_obj, &image_obj, &weight_obj, &jacob_obj)) {
        return NULL;
    }

    gmix=(struct gauss* ) PyArray_DATA(gmix_obj);
    n_gauss=PyArray_SIZE(gmix_obj);

    if (!gmix_set_norms_if_needed(gmix, n_gauss)) {
        return NULL;
    }

    n_row=PyArray_DIM(image_obj, 0);
    n_col=PyArray_DIM(image_obj, 1);

    jacob=(struct jacobian* ) PyArray_DATA(jacob_obj);

    for (row=0; row < n_row; row++) {
        //u=jacob->dudrow*(row - jacob->row0) + jacob->dudcol*(0 - jacob->col0);
        //v=jacob->dvdrow*(row - jacob->row0) + jacob->dvdcol*(0 - jacob->col0);
        u=PYGMIX_JACOB_GETU(jacob, row, 0);
        v=PYGMIX_JACOB_GETV(jacob, row, 0);

        for (col=0; col < n_col; col++) {

            ivar=*( (double*)PyArray_GETPTR2(weight_obj,row,col) );
            if ( ivar > 0.0) {
                data=*( (double*)PyArray_GETPTR2(image_obj,row,col) );
                model_val=PYGMIX_GMIX_EVAL(gmix, n_gauss, v, u);

                diff = model_val-data;
                loglike += diff*diff*ivar;
                s2n_numer += data*model_val*ivar;
                s2n_denom += model_val*model_val*ivar;

                npix += 1;
            }

            u += jacob->dudcol;
            v += jacob->dvdcol;

        }
    }

    loglike *= (-0.5);

    // fill in the retval
    PYGMIX_PACK_RESULT4(loglike, s2n_numer, s2n_denom, npix);
    return retval;
}

*/
