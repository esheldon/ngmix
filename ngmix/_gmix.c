/*
 
   Operate on padded structures that map to C structures, as created
   using numba.struct

 */

#include <Python.h>
#include <numpy/arrayobject.h> 
#include "_gmix.h"


static PyObject * PyGMix_loglike_jacob(PyObject* self, PyObject* args) {

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
    {"test",      (PyCFunction)PyGMix_test,      METH_VARARGS,  "test\n\nprint and return."},
    {"loglike_jacob",      (PyCFunction)PyGMix_loglike_jacob,      METH_VARARGS,  "calculate likelihood with weight map and jacobian\n"},
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
