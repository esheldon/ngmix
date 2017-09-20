// Get the weighted moments of the image, using the input gaussian
// mixture as the weight function.  The moments are *not* normalized

// See the get_weighted_moments method in gmix.py

#include "../_gmix.h"
#include "gmix.h"

PyObject * PyGMix_get_weighted_moments(PyObject* self, PyObject* args) {

    PyObject* image_obj=NULL;
    PyObject* weight_obj=NULL;
    PyObject* jacob_obj=NULL;
    PyObject* gmix_obj=NULL;

    PyObject* pars_obj=NULL;
    PyObject* pcov_obj=NULL;
    double rmax=0, rmaxsq=0;

    npy_intp n_gauss=0, n_row=0, n_col=0, row=0, col=0;//, igauss=0;

    struct gauss *gmix=NULL;//, *gauss=NULL;
    struct jacobian *jacob=NULL;
    double *pars=NULL, *pcov=NULL;
    double F[6];

    double
        u=0, v=0, umod=0, vmod=0,
        wdata=0, data=0, weight=0, w2=0, wsum=0,
        s2n_numer=0, s2n_denom=0,
        ivar=0, var=0,
        ucen=0, vcen=0, psum=0, rsq=0;
    int flags=0, i=0, j=0;

    if (!PyArg_ParseTuple(args, (char*)"OOOOOOd", 
                          &image_obj,
                          &weight_obj,
                          &jacob_obj,
                          &gmix_obj,
                          &pars_obj,
                          &pcov_obj,
                          &rmax)) {
        return NULL;
    }

    gmix=(struct gauss* ) PyArray_DATA(gmix_obj);
    n_gauss=PyArray_SIZE(gmix_obj);

    if (!gmix_set_norms_if_needed(gmix, n_gauss)) {
        return NULL;
    }

    gmix_get_cen(gmix, n_gauss, &vcen, &ucen, &psum);

    n_row=PyArray_DIM(image_obj, 0);
    n_col=PyArray_DIM(image_obj, 1);

    rmaxsq=rmax*rmax;

    jacob=(struct jacobian* ) PyArray_DATA(jacob_obj);

    pars=PyArray_DATA(pars_obj); // pars[6]
    pcov=PyArray_DATA(pcov_obj); // pcov[6,6]

    for (row=0; row < n_row; row++) {
        for (col=0; col < n_col; col++) {

            // sky coordinates relative to the jacobian center
            v=PYGMIX_JACOB_GETV(jacob, row, col);
            u=PYGMIX_JACOB_GETU(jacob, row, col);

            rsq = u*u + v*v;

            if (rsq > rmaxsq) {
                continue;
            }

            data = *( (double*)PyArray_GETPTR2(image_obj,row,col) );
            ivar = *( (double*)PyArray_GETPTR2(weight_obj,row,col) );

            if (ivar <= 0.0) {
                flags = 1;
                goto _getmom_bail;
            }

            var=1.0/ivar;

            weight=PYGMIX_GMIX_EVAL(gmix, n_gauss, v, u);

            // sky coordinates relative to the gaussian mixture center
            vmod = v-vcen;
            umod = u-ucen;

            wdata = weight*data;
            w2 = weight*weight;
            wsum += weight;

            // for the s/n sums
            s2n_numer += wdata*ivar;
            s2n_denom += w2*ivar;

            F[0] = vmod;
            F[1] = umod;
            F[2] = umod*umod - vmod*vmod;
            F[3] = 2*vmod*umod;
            F[4] = umod*umod + vmod*vmod;
            F[5] = 1.0;

            for (i=0; i<6; i++) {
                pars[i] += wdata*F[i];
                for (j=0; j<6; j++) {
                    pcov[i + 6*j] += w2*var*F[i]*F[j];
                }
            }

        }
    }

_getmom_bail:

    return Py_BuildValue("iddd", flags, wsum, s2n_numer, s2n_denom);
}


/*
   weighted moments of one gaussian mixture with another
*/
PyObject * PyGMix_get_weighted_gmix_moments(PyObject* self, PyObject* args) {

    // arguments
    PyObject* gmix_obj=NULL;
    PyObject* wt_gmix_obj=NULL;
    PyObject* jacob_obj=NULL;
    int n_row=0, n_col=0;
    PyObject* pars_obj=NULL;

    // for unpacking input arrays
    npy_intp n_gauss=0, wt_n_gauss=0;
    struct gauss *gmix=NULL;
    struct gauss *wt_gmix=NULL;
    struct jacobian *jacob=NULL;
    double *pars=NULL;

    // local variables
    int i=0, row=0, col=0;
    double u=0, v=0, umod=0, vmod=0;
    double model_val=0, wmodel, weight=0, wsum=0;
    double wt_ucen=0, wt_vcen=0, wt_psum=0;
    double F[6];

    if (!PyArg_ParseTuple(
            args, (char*)"OOOiiO", 
            &gmix_obj,
            &wt_gmix_obj,
            &jacob_obj,
            &n_row,
            &n_col,
            &pars_obj
        ) ) {
        return NULL;
    }

    // unpack the input and output arrays
    gmix=(struct gauss* ) PyArray_DATA(gmix_obj);
    n_gauss=PyArray_SIZE(gmix_obj);
    if (!gmix_set_norms_if_needed(gmix, n_gauss)) {
        return NULL;
    }

    wt_gmix=(struct gauss* ) PyArray_DATA(wt_gmix_obj);
    wt_n_gauss=PyArray_SIZE(wt_gmix_obj);
    if (!gmix_set_norms_if_needed(wt_gmix, wt_n_gauss)) {
        return NULL;
    }

    jacob=(struct jacobian* ) PyArray_DATA(jacob_obj);

    pars=PyArray_DATA(pars_obj); // pars[6]

    // center of the weight gmix
    gmix_get_cen(wt_gmix, wt_n_gauss, &wt_vcen, &wt_ucen, &wt_psum);

    for (row=0; row < n_row; row++) {
        for (col=0; col < n_col; col++) {
            // sky coordinates relative to the jacobian center
            v=PYGMIX_JACOB_GETV(jacob, row, col);
            u=PYGMIX_JACOB_GETU(jacob, row, col);

            model_val=PYGMIX_GMIX_EVAL(gmix, n_gauss, v, u);

            // evaluate the gaussian mixture at the specified location
            weight=PYGMIX_GMIX_EVAL_FULL(wt_gmix, wt_n_gauss, v, u);

            // sky coordinates relative to the gaussian mixture center
            vmod = v-wt_vcen;
            umod = u-wt_ucen;

            wmodel = weight*model_val;
            wsum += weight;

            F[0] = vmod;
            F[1] = umod;
            F[2] = umod*umod - vmod*vmod;
            F[3] = 2*vmod*umod;
            F[4] = umod*umod + vmod*vmod;
            F[5] = 1.0;

            for (i=0; i<6; i++) {
                pars[i] += wmodel*F[i];
            }

        }
    }

    return Py_BuildValue("d", wsum);
}

PyObject * PyGMix_get_unweighted_moments(PyObject* self, PyObject* args) {

    PyObject
        *im_obj=NULL,
        *jacob_obj=NULL,

        *pars_obj=NULL;

    double
        u=0, v=0,
        data=0,
        F[6]={0},
        *pars=NULL;

    npy_intp n_row=0, n_col=0, row=0, col=0;

    struct jacobian *jacob=NULL;

    int i=0;

    if (!PyArg_ParseTuple(args, (char*)"OOO", 
                          &im_obj,
                          &jacob_obj,
                          &pars_obj)) {
        return NULL;
    }

    n_row=PyArray_DIM(im_obj, 0);
    n_col=PyArray_DIM(im_obj, 1);

    jacob=(struct jacobian* ) PyArray_DATA(jacob_obj);

    pars=PyArray_DATA(pars_obj); // [6]

    for (row=0; row < n_row; row++) {
        for (col=0; col < n_col; col++) {

            // sky coordinates relative to the jacobian center
            v=PYGMIX_JACOB_GETV(jacob, row, col);
            u=PYGMIX_JACOB_GETU(jacob, row, col);

            data = *( (double*)PyArray_GETPTR2(im_obj,row,col) );

            F[0] = v;
            F[1] = u;
            F[2] = u*u - v*v;
            F[3] = 2*v*u;
            F[4] = u*u + v*v;
            F[5] = 1.0;

            for (i=0; i<6; i++) {
                pars[i] += data*F[i];
            }

        }
    }

    Py_RETURN_NONE;
}




