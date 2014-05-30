/*
 
   Operate on padded structures that map to C structures, as created
   using numba.struct

 */

#include <Python.h>
#include <numpy/arrayobject.h> 
#include "_gmix.h"

/*
    convert reduced shear g1,g2 to standard ellipticity
    parameters e1,e2

    return 0 means out of range
*/
int g1g2_to_e1e2(double g1, double g2, double *e1, double *e2) {
    int status=0;
    double g=sqrt(g1*g1 + g2*g2);

    if (g >= 1) {
        fprintf(stderr,"g out of bounds: %g\n", g);
        return status;
    }
    if (g == 0.0) {
        *e1=0;
        *e2=0;
    }

    double eta = 2*atanh(g);
    double e = tanh(eta);
    if (e >= 1.) {
        // round off?
        e = 0.99999999;
    }

    double fac = e/g;

    *e1 = fac*g1;
    *e2 = fac*g2;
    
    status=1;
    return status;
}

/* 
   zero return value means bad determinant, out of range
   
*/
static int gauss2d_set(struct PyGMix_Gauss2D *self,
                       double p,
                       double row,
                       double col,
                       double irr,
                       double irc,
                       double icc) {

    int status=0;

    double det = irr*icc - irc*irc;
    if (det < 1.0e-200) {
        fprintf(stderr,"gmix error: det too low: %.16g\n", det);
        return status;
    }

    self->p=p;
    self->row=row;
    self->col=col;
    self->irr=irr;
    self->irc=irc;
    self->icc=icc;

    self->det = det;

    double idet=1.0/det;
    self->drr = irr*idet;
    self->drc = irc*idet;
    self->dcc = icc*idet;
    self->norm = 1./(2*M_PI*sqrt(det));

    self->pnorm = self->p*self->norm;

    status=1;
    return status;
}

static inline int get_ngauss(int model) {
    int status=1;
    switch (model) {
        case PyGMIX_GMIX_EXP:
            return 6;
        case PyGMIX_GMIX_DEV:
            return 10;
        case PyGMIX_GMIX_TURB:
            return 3;
        case PyGMIX_GMIX_BDC:
            return 16;
        case PyGMIX_GMIX_BDF:
            return 16;
        case PyGMIX_GMIX_SERSIC:
            return 10;
        default:
            fprintf(stderr,"cannot get ngauss for model %d\n", model);
            status=0;
    }

    return status;
}

// for counts
static const double PyGMix_pvals_exp[] = {
    0.00061601229677880041, 
    0.0079461395724623237, 
    0.053280454055540001, 
    0.21797364640726541, 
    0.45496740582554868, 
    0.26521634184240478};
// for T
static const double PyGMix_fvals_exp[] = {
    0.002467115141477932, 
    0.018147435573256168, 
    0.07944063151366336, 
    0.27137669897479122, 
    0.79782256866993773, 
    2.1623306025075739};

int gmix_fill_simple(struct PyGMix_Gauss2D *self,
                     npy_intp n_gauss,
                     const double* pars,
                     npy_intp n_pars,
                     int model,
                     const double* fvals,
                     const double* pvals) {

    int status=0;
    npy_intp i=0;
    if (n_pars != 6) {
        fprintf(stderr,"simple pars should be size 6\n");
        return status;
    }
    int n_gauss_expected=get_ngauss(model);
    if (n_gauss != n_gauss_expected) {
        fprintf(stderr,"for model %d expected %d gauss, got %ld\n",
                model, n_gauss_expected, n_gauss);
        return status;
    }

    double row=pars[0];
    double col=pars[1];
    double g1=pars[2];
    double g2=pars[3];
    double T=pars[4];
    double counts=pars[5];

    double e1,e2;
    status=g1g2_to_e1e2(g1, g2, &e1, &e2);
    if (!status) {
        return status;
    }


    for (i=0; i<n_gauss; i++) {
        double T_i_2 = 0.5*T*fvals[i];
        double counts_i=counts*pvals[i];

        gauss2d_set(&self[i],
                    counts_i,
                    row,
                    col, 
                    T_i_2*(1-e1), 
                    T_i_2*e2,
                    T_i_2*(1+e1));
    }

    status=1;
    return status;
}

int gmix_fill(struct PyGMix_Gauss2D *self,
              npy_intp n_gauss,
              const double* pars,
              npy_intp n_pars,
              int model) {

    int status=0;
    switch (model) {
        case PyGMIX_GMIX_EXP:
            status=gmix_fill_simple(self, n_gauss,
                                    pars, n_pars,
                                    model,
                                    PyGMix_fvals_exp,
                                    PyGMix_pvals_exp);
            break;
        default:
            fprintf(stderr,"gmix error: Bad gmix model: %d\n", model);
            status=0;
            break;
    }

    return status;
}

static PyObject * PyGMix_gmix_fill(PyObject* self, PyObject* args) {
    PyObject* gmix_obj=NULL;
    PyObject* pars_obj=NULL;
    int model=0;

    struct PyGMix_Gauss2D *gmix=NULL;

    if (!PyArg_ParseTuple(args, (char*)"OOi",
                          &gmix_obj, 
                          &pars_obj,
                          &model)) {

        return NULL;
    }

    gmix=(struct PyGMix_Gauss2D* ) PyArray_DATA(gmix_obj);
    npy_intp n_gauss=PyArray_SIZE(gmix_obj);

    const double *pars=(double *) PyArray_DATA(pars_obj);
    npy_intp n_pars = PyArray_SIZE(pars_obj);

    int res=gmix_fill(gmix, n_gauss, pars, n_pars, model);

    return PyInt_FromLong( (long) res );
}

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

            //printf("row: %ld col: %ld start: %d\n", row, col, start);

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
    PyTuple_SetItem(retval,0,PyFloat_FromDouble(s2n_numer));
    PyTuple_SetItem(retval,1,PyFloat_FromDouble(s2n_denom));
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

    {"gmix_fill",(PyCFunction)PyGMix_gmix_fill, METH_VARARGS,  "Fill the input gmix from the pars\n"},
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
