/*
 
   Operate on padded structures that map to C structures, as created
   using numba.struct

 */

#include <Python.h>
#include <numpy/arrayobject.h> 
#include "_gmix.h"

/*
   Render the gmix in the input image, without jacobian
*/
static PyObject * PyGMix_render(PyObject* self, PyObject* args) {

    PyObject* gmix_obj=NULL;
    PyObject* image_obj=NULL;
    int nsub=0;
    npy_intp n_gauss=0, n_row=0, n_col=0, row=0, col=0;//, igauss=0;
    npy_intp rowsub=0, colsub=0;

    struct PyGMix_Gauss2D *gmix=NULL;
    double *ptr=NULL, stepsize=0, offset=0, areafac=0, tval=0, trow=0, tcol=0;

    if (!PyArg_ParseTuple(args, (char*)"OOi", &gmix_obj, &image_obj, &nsub)) {
        return NULL;
    }

    gmix=(struct PyGMix_Gauss2D* ) PyArray_DATA(gmix_obj);
    n_gauss=PyArray_SIZE(gmix_obj);

    stepsize = 1./nsub;
    offset = (nsub-1)*stepsize/2.;
    areafac = 1./(nsub*nsub);

    n_row=PyArray_DIM(image_obj, 0);
    n_col=PyArray_DIM(image_obj, 1);

    for (row=0; row < n_row; row++) {
        for (col=0; col < n_row; col++) {

            tval = 0.0;
            trow = row-offset;

            for (rowsub=0; rowsub<nsub; rowsub++) {
                tcol = col-offset;
                for (colsub=0; colsub<nsub; colsub++) {

                    tval += PYGMIX_GMIX_EVAL(gmix, n_gauss, trow, tcol);

                    tcol += stepsize;
                } // colsub

                trow += stepsize;
            } // rowsub

            // add to existing values
            ptr=(double*)PyArray_GETPTR2(image_obj,row,col);
            tval *= areafac;
            (*ptr) += tval;
        } // cols
    } // rows

    return Py_None;
}


/*
   Render the gmix in the input image, with jacobian
*/
static PyObject * PyGMix_render_jacob(PyObject* self, PyObject* args) {

    PyObject* gmix_obj=NULL;
    PyObject* image_obj=NULL;
    PyObject* jacob_obj=NULL;
    int nsub=0;
    npy_intp n_gauss=0, n_row=0, n_col=0, row=0, col=0;//, igauss=0;
    npy_intp rowsub=0, colsub=0;

    struct PyGMix_Gauss2D *gmix=NULL;
    struct PyGMix_Jacobian *jacob=NULL;

    double *ptr=NULL, u=0, v=0, stepsize=0, ustepsize=0, vstepsize=0,
           offset=0, areafac=0, tval=0,trow=0, lowcol=0;

    if (!PyArg_ParseTuple(args, (char*)"OOiO", &gmix_obj, &image_obj, &nsub, &jacob_obj)) {
        return NULL;
    }

    gmix=(struct PyGMix_Gauss2D* ) PyArray_DATA(gmix_obj);
    n_gauss=PyArray_SIZE(gmix_obj);

    jacob=(struct PyGMix_Jacobian* ) PyArray_DATA(jacob_obj);

    stepsize = 1./nsub;
    offset = (nsub-1)*stepsize/2.;
    areafac = 1./(nsub*nsub);
    // sub-steps while moving along column direction
    ustepsize = stepsize*jacob->dudcol;
    vstepsize = stepsize*jacob->dvdcol;

    n_row=PyArray_DIM(image_obj, 0);
    n_col=PyArray_DIM(image_obj, 1);

    for (row=0; row < n_row; row++) {
        for (col=0; col < n_row; col++) {

            tval = 0.0;
            trow = row-offset;
            lowcol = col-offset;

            for (rowsub=0; rowsub<nsub; rowsub++) {
                u=jacob->dudrow*(trow - jacob->row0) + jacob->dudcol*(lowcol - jacob->col0);
                v=jacob->dvdrow*(trow - jacob->row0) + jacob->dvdcol*(lowcol - jacob->col0);
                for (colsub=0; colsub<nsub; colsub++) {

                    tval += PYGMIX_GMIX_EVAL(gmix, n_gauss, u, v);

                    u += ustepsize;
                    v += vstepsize;
                } // colsub

                trow += stepsize;
            } // rowsub

            // add to existing values
            ptr=(double*)PyArray_GETPTR2(image_obj,row,col);
            tval *= areafac;
            (*ptr) += tval;
        } // cols
    } // rows

    return Py_None;
}


/*
   Calculate the loglike between the gmix and the input image
*/
static PyObject * PyGMix_get_loglike(PyObject* self, PyObject* args) {

    PyObject* gmix_obj=NULL;
    PyObject* image_obj=NULL;
    PyObject* weight_obj=NULL;
    PyObject* jacob_obj=NULL;
    npy_intp n_gauss=0, n_row=0, n_col=0, row=0, col=0;//, igauss=0;

    struct PyGMix_Gauss2D *gmix=NULL;//, *gauss=NULL;
    struct PyGMix_Jacobian *jacob=NULL;

    double data=0, ivar=0, u=0, v=0;
    double model_val=0, diff=0;
    double s2n_numer=0.0, s2n_denom=0.0, loglike = 0.0;

    PyObject* retval=NULL;

    if (!PyArg_ParseTuple(args, (char*)"OOOO", &gmix_obj, &image_obj, &weight_obj, &jacob_obj)) {
        return NULL;
    }

    gmix=(struct PyGMix_Gauss2D* ) PyArray_DATA(gmix_obj);
    n_gauss=PyArray_SIZE(gmix_obj);

    n_row=PyArray_DIM(image_obj, 0);
    n_col=PyArray_DIM(image_obj, 1);

    jacob=(struct PyGMix_Jacobian* ) PyArray_DATA(jacob_obj);

    for (row=0; row < n_row; row++) {
        u=jacob->dudrow*(row - jacob->row0) + jacob->dudcol*(0 - jacob->col0);
        v=jacob->dvdrow*(row - jacob->row0) + jacob->dvdcol*(0 - jacob->col0);

        for (col=0; col < n_row; col++) {

            ivar=*( (double*)PyArray_GETPTR2(weight_obj,row,col) );
            if ( ivar > 0.0) {
                data=*( (double*)PyArray_GETPTR2(image_obj,row,col) );
                model_val=PYGMIX_GMIX_EVAL(gmix, n_gauss, u, v);

                diff = model_val-data;
                loglike += diff*diff*ivar;
                s2n_numer += data*model_val*ivar;
                s2n_denom += model_val*model_val*ivar;
            }

            u += jacob->dudcol;
            v += jacob->dvdcol;

        }
    }

    loglike *= (-0.5);

    retval=PyTuple_New(3);
    PyTuple_SetItem(retval,0,PyFloat_FromDouble(loglike));
    PyTuple_SetItem(retval,1,PyFloat_FromDouble(s2n_numer));
    PyTuple_SetItem(retval,2,PyFloat_FromDouble(s2n_denom));
    return retval;
}

/*
   Fill the input fdiff=(model-data)/err, return s2n_numer, s2n_denom
*/
static PyObject * PyGMix_fill_fdiff(PyObject* self, PyObject* args) {

    PyObject* gmix_obj=NULL;
    PyObject* image_obj=NULL;
    PyObject* weight_obj=NULL;
    PyObject* jacob_obj=NULL;
    PyObject* fdiff_obj=NULL;
    npy_intp n_gauss=0, n_row=0, n_col=0, row=0, col=0;//, igauss=0;
    int start=0;

    struct PyGMix_Gauss2D *gmix=NULL;//, *gauss=NULL;
    struct PyGMix_Jacobian *jacob=NULL;

    double data=0, ivar=0, ierr=0, u=0, v=0, *fdiff_ptr=NULL;
    double model_val=0, diff=0;
    double s2n_numer=0.0, s2n_denom=0.0;

    PyObject* retval=NULL;

    if (!PyArg_ParseTuple(args, (char*)"OOOOOi", 
                          &gmix_obj, &image_obj, &weight_obj, &jacob_obj,
                          &fdiff_obj, &start)) {
        return NULL;
    }

    gmix=(struct PyGMix_Gauss2D* ) PyArray_DATA(gmix_obj);
    n_gauss=PyArray_SIZE(gmix_obj);

    n_row=PyArray_DIM(image_obj, 0);
    n_col=PyArray_DIM(image_obj, 1);

    jacob=(struct PyGMix_Jacobian* ) PyArray_DATA(jacob_obj);

    // we might start somewhere after the priors
    // note fdiff is 1-d
    fdiff_ptr=(double *)PyArray_GETPTR1(fdiff_obj,start);

    for (row=0; row < n_row; row++) {
        u=jacob->dudrow*(row - jacob->row0) + jacob->dudcol*(0 - jacob->col0);
        v=jacob->dvdrow*(row - jacob->row0) + jacob->dvdcol*(0 - jacob->col0);

        for (col=0; col < n_row; col++) {

            ivar=*( (double*)PyArray_GETPTR2(weight_obj,row,col) );
            if ( ivar > 0.0) {
                ierr=sqrt(ivar);

                data=*( (double*)PyArray_GETPTR2(image_obj,row,col) );
                model_val=PYGMIX_GMIX_EVAL(gmix, n_gauss, u, v);

                diff = model_val-data;

                (*fdiff_ptr) = (model_val-data)*ierr;
                s2n_numer += data*model_val*ivar;
                s2n_denom += model_val*model_val*ivar;
            } else {
                (*fdiff_ptr) = 0.0;
            }

            fdiff_ptr++;

            u += jacob->dudcol;
            v += jacob->dvdcol;

        }
    }

    retval=PyTuple_New(2);
    PyTuple_SetItem(retval,1,PyFloat_FromDouble(s2n_numer));
    PyTuple_SetItem(retval,2,PyFloat_FromDouble(s2n_denom));
    return retval;
}


static PyObject * PyGMix_test(PyObject* self, PyObject* args) {

    PyObject *gmix_obj=NULL;
    PyObject *jacob_obj=NULL;
    npy_intp n_gauss=0, n_jacob=0;

    if (!PyArg_ParseTuple(args, (char*)"OO", &gmix_obj, &jacob_obj)) {
        return NULL;
    }

    n_gauss=PyArray_SIZE(gmix_obj);
    n_jacob=PyArray_SIZE(jacob_obj);

    printf("n_gauss: %ld\n", n_gauss);
    printf("n_jacob (should be 1): %ld\n", n_jacob);
    return Py_None;
}

static PyMethodDef pygauss2d_funcs[] = {
    {"test",        (PyCFunction)PyGMix_test,         METH_VARARGS,  "test\n\nprint and return."},
    {"get_loglike", (PyCFunction)PyGMix_get_loglike,  METH_VARARGS,  "calculate likelihood\n"},
    {"fill_fdiff",  (PyCFunction)PyGMix_fill_fdiff,  METH_VARARGS,  "fill fdiff for LM\n"},
    {"render",      (PyCFunction)PyGMix_render, METH_VARARGS,  "render without jacobian\n"},
    {"render_jacob",(PyCFunction)PyGMix_render_jacob, METH_VARARGS,  "render with jacobian\n"},
    {NULL}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_gmix",      /* m_name */
        "Defines the funcs associated with gmix",  /* m_doc */
        -1,                  /* m_size */
        fitstype_funcs,      /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };
#endif

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
init_gmix(void) 
{
    PyObject* m;


#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
    if (m==NULL) {
        return NULL;
    }

#else
    m = Py_InitModule3("_gmix", pygauss2d_funcs, "Define gmix funcs.");
    if (m==NULL) {
        return;
    }
#endif

    // for numpy
    import_array();
}
