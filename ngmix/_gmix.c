/*
 
   Operate on padded structures that map to C structures, as created
   using numba.struct

 */

#include <Python.h>
#include <numpy/arrayobject.h> 
#include "_gmix.h"

// exceptions
static PyObject* GMixRangeError;
static PyObject* GMixFatalError;

/*
    convert reduced shear g1,g2 to standard ellipticity
    parameters e1,e2

    return 0 means out of range
*/
static int g1g2_to_e1e2(double g1, double g2, double *e1, double *e2) {
    double g=sqrt(g1*g1 + g2*g2);

    if (g >= 1) {
        PyErr_Format(GMixRangeError, "g out of bounds: %g", g);
        return 0;
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
    
    return 1;
}

static void gmix_get_cen(const struct PyGMix_Gauss2D *self,
                         npy_intp n_gauss,
                         double* row,
                         double *col,
                         double *psum)
{
    npy_intp i=0;
    *row=0;
    *col=0;
    *psum=0;

    for (i=0; i<n_gauss; i++) {
        const struct PyGMix_Gauss2D *gauss=&self[i];

        double p=gauss->p;
        *row += p*gauss->row;
        *col += p*gauss->col;
        *psum += p;
    }
    (*row) /= (*psum);
    (*col) /= (*psum);
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


    double det = irr*icc - irc*irc;
    if (det < 1.0e-200) {
        // PyErr_Format doesn't format floats
        fprintf(stderr,"gauss2d det too low: %.16g", det);
        PyErr_Format(GMixRangeError, "gauss2d det too low");
        //PyErr_Format(GMixRangeError, "gauss2d det too low: %.16g", det);
        return 0;
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

    return 1;
}

static inline int get_n_gauss(int model, int *status) {
    int n_gauss=0;

    *status=1;
    switch (model) {
        case PyGMIX_GMIX_GAUSS:
            n_gauss=1;
            break;
        case PyGMIX_GMIX_EXP:
            n_gauss=6;
            break;
        case PyGMIX_GMIX_DEV:
            n_gauss=10;
            break;
        case PyGMIX_GMIX_TURB:
            n_gauss=3;
            break;
        case PyGMIX_GMIX_BDC:
            n_gauss=16;
            break;
        case PyGMIX_GMIX_BDF:
            n_gauss=16;
            break;
        case PyGMIX_GMIX_SERSIC:
            n_gauss=10;
            break;
        default:
            PyErr_Format(GMixFatalError, 
                         "cannot get n_gauss for model %d", model);
            n_gauss=-1;
            *status=0;
    }

    return n_gauss;
}

// pvals->counts
// fvals->T
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


/*
   when an error occurs and exception is set. Use goto pattern
   for errors to simplify code.
*/
static int gmix_fill_full(struct PyGMix_Gauss2D *self,
                          npy_intp n_gauss,
                          const double* pars,
                          npy_intp n_pars)
{

    int status=0;
    npy_intp i=0, beg=0;

    if ( (n_pars % 6) != 0) {
        PyErr_Format(GMixFatalError, 
                     "full pars should be multiple of 6, got %ld", n_pars);
        goto _gmix_fill_full_bail;
    }

    for (i=0; i<n_gauss; i++) {
        beg=i*6;

        status=gauss2d_set(&self[i],
                           pars[beg+0],
                           pars[beg+1],
                           pars[beg+2],
                           pars[beg+3],
                           pars[beg+4],
                           pars[beg+5]);

        // an exception will be set
        if (!status) {
            goto _gmix_fill_full_bail;
        }

    }

    status=1;

_gmix_fill_full_bail:
    return status;
}



/*
   when an error occurs and exception is set. Use goto pattern
   for errors to simplify code.
*/
static int gmix_fill_simple(struct PyGMix_Gauss2D *self,
                            npy_intp n_gauss,
                            const double* pars,
                            npy_intp n_pars,
                            int model,
                            const double* fvals,
                            const double* pvals)
{

    int status=0;
    npy_intp i=0;

    if (n_pars != 6) {
        PyErr_Format(GMixFatalError, 
                     "simple pars should be size 6, got %ld", n_pars);
        goto _gmix_fill_simple_bail;
    }

    int n_gauss_expected=get_n_gauss(model, &status);
    if (!status) {
        goto _gmix_fill_simple_bail;
    }
    if (n_gauss != n_gauss_expected) {
        PyErr_Format(GMixFatalError, 
                     "for model %d expected %d gauss, got %ld",
                     model, n_gauss_expected, n_gauss);
        goto _gmix_fill_simple_bail;
    }

    double row=pars[0];
    double col=pars[1];
    double g1=pars[2];
    double g2=pars[3];
    double T=pars[4];
    double counts=pars[5];

    double e1,e2;
    // can set exception inside
    status=g1g2_to_e1e2(g1, g2, &e1, &e2);
    if (!status) {
        goto _gmix_fill_simple_bail;
    }


    for (i=0; i<n_gauss; i++) {
        double T_i_2 = 0.5*T*fvals[i];
        double counts_i=counts*pvals[i];

        status=gauss2d_set(&self[i],
                           counts_i,
                           row,
                           col, 
                           T_i_2*(1-e1), 
                           T_i_2*e2,
                           T_i_2*(1+e1));
        // an exception will be set
        if (!status) {
            goto _gmix_fill_simple_bail;
        }
    }

    status=1;

_gmix_fill_simple_bail:
    return status;
}

/*

   Set an exception on error

   Use goto pattern to simplify code

*/
static int gmix_fill(struct PyGMix_Gauss2D *self,
                     npy_intp n_gauss,
                     const double* pars,
                     npy_intp n_pars,
                     int model)
{

    int status=0;
    switch (model) {
        case PyGMIX_GMIX_EXP:
            status=gmix_fill_simple(self, n_gauss,
                                    pars, n_pars,
                                    model,
                                    PyGMix_fvals_exp,
                                    PyGMix_pvals_exp);
            break;
        case PyGMIX_GMIX_DEV:
            status=gmix_fill_simple(self, n_gauss,
                                    pars, n_pars,
                                    model,
                                    PyGMix_fvals_dev,
                                    PyGMix_pvals_dev);
            break;

        case PyGMIX_GMIX_TURB:
            status=gmix_fill_simple(self, n_gauss,
                                    pars, n_pars,
                                    model,
                                    PyGMix_fvals_turb,
                                    PyGMix_pvals_turb);
            break;
        case PyGMIX_GMIX_GAUSS:
            status=gmix_fill_simple(self, n_gauss,
                                    pars, n_pars,
                                    model,
                                    PyGMix_fvals_gauss,
                                    PyGMix_pvals_gauss);
            break;

        case PyGMIX_GMIX_FULL:
            status=gmix_fill_full(self, n_gauss, pars, n_pars);
            break;

        default:
            PyErr_Format(GMixFatalError, 
                         "gmix error: Bad gmix model: %d", model);
            goto _gmix_fill_bail;
            break;
    }

    status=1;

_gmix_fill_bail:
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
    if (!res) {
        // raise an exception
        return NULL;
    } else {
        return Py_None;
    }
}


static int convolve_fill(const struct PyGMix_Gauss2D *gmix, npy_intp n_gauss,
                         const struct PyGMix_Gauss2D *psf, npy_intp psf_n_gauss,
                         struct PyGMix_Gauss2D *out, npy_intp out_n_gauss)
{
    int status=0;
    npy_intp ntot, iobj=0, ipsf=0, itot=0;
    double psf_rowcen=0, psf_colcen=0, psf_psum=0;

    ntot = n_gauss*psf_n_gauss;
    if (ntot != out_n_gauss) {
        PyErr_Format(GMixRangeError, 
                     "target gmix is wrong size %ld, expected %ld",
                     out_n_gauss, ntot);
        goto _convolve_fill_bail;
    }

    gmix_get_cen(psf, psf_n_gauss, &psf_rowcen, &psf_colcen, &psf_psum);
    double psf_ipsum=1.0/psf_psum;

    itot=0;
    for (iobj=0; iobj<n_gauss; iobj++) {
        const struct PyGMix_Gauss2D *obj_gauss=&gmix[iobj];

        for (ipsf=0; ipsf<psf_n_gauss; ipsf++) {
            const struct PyGMix_Gauss2D *psf_gauss=&psf[ipsf];

            double p = obj_gauss->p*psf_gauss->p*psf_ipsum;

            double row = obj_gauss->row + (psf_gauss->row-psf_rowcen);
            double col = obj_gauss->col + (psf_gauss->col-psf_colcen);

            double irr = obj_gauss->irr + psf_gauss->irr;
            double irc = obj_gauss->irc + psf_gauss->irc;
            double icc = obj_gauss->icc + psf_gauss->icc;

            status=gauss2d_set(&out[itot], 
                               p, row, col, irr, irc, icc);
            // an exception will be set
            if (!status) {
                goto _convolve_fill_bail;
            }

            itot++;
        }
    }

    status=1;
_convolve_fill_bail:
    return status;
}

static PyObject * PyGMix_convolve_fill(PyObject* self, PyObject* args) {
    PyObject* gmix_obj=NULL;
    PyObject* psf_gmix_obj=NULL;
    PyObject* out_gmix_obj=NULL;

    struct PyGMix_Gauss2D *gmix=NULL;
    struct PyGMix_Gauss2D *psf_gmix=NULL;
    struct PyGMix_Gauss2D *out_gmix=NULL;

    if (!PyArg_ParseTuple(args, (char*)"OOO",
                          &gmix_obj, 
                          &psf_gmix_obj,
                          &out_gmix_obj)) {

        return NULL;
    }

    gmix=(struct PyGMix_Gauss2D* ) PyArray_DATA(gmix_obj);
    npy_intp n_gauss=PyArray_SIZE(gmix_obj);

    psf_gmix=(struct PyGMix_Gauss2D* ) PyArray_DATA(psf_gmix_obj);
    npy_intp psf_n_gauss =PyArray_SIZE(psf_gmix_obj);

    out_gmix=(struct PyGMix_Gauss2D* ) PyArray_DATA(out_gmix_obj);
    npy_intp out_n_gauss =PyArray_SIZE(out_gmix_obj);

    int res=convolve_fill(gmix, n_gauss,
                          psf_gmix, psf_n_gauss,
                          out_gmix, out_n_gauss);
    if (!res) {
        // raise an exception
        return NULL;
    } else {
        return Py_None;
    }
}




/*
   Render the gmix in the input image, without jacobian

   Error checking should be done in python.
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
        for (col=0; col < n_col; col++) {

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

   Error checking should be done in python.
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

    if (!PyArg_ParseTuple(args, (char*)"OOiO", 
                          &gmix_obj, &image_obj, &nsub, &jacob_obj)) {
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
        for (col=0; col < n_col; col++) {

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

   Error checking should be done in python.
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

    if (!PyArg_ParseTuple(args, (char*)"OOOO", 
                          &gmix_obj, &image_obj, &weight_obj, &jacob_obj)) {
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

        for (col=0; col < n_col; col++) {

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

   Error checking should be done in python.
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
    double model_val=0;
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

        for (col=0; col < n_col; col++) {

            ivar=*( (double*)PyArray_GETPTR2(weight_obj,row,col) );
            if ( ivar > 0.0) {
                ierr=sqrt(ivar);

                data=*( (double*)PyArray_GETPTR2(image_obj,row,col) );
                model_val=PYGMIX_GMIX_EVAL(gmix, n_gauss, u, v);

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
    PyErr_Format(GMixRangeError, "testing GMixRangeError");
    return NULL;
}

static PyMethodDef pygauss2d_funcs[] = {
    {"test",        (PyCFunction)PyGMix_test,         METH_VARARGS,  "test\n\nprint and return."},

    {"get_loglike", (PyCFunction)PyGMix_get_loglike,  METH_VARARGS,  "calculate likelihood\n"},
    {"fill_fdiff",  (PyCFunction)PyGMix_fill_fdiff,  METH_VARARGS,  "fill fdiff for LM\n"},
    {"render",      (PyCFunction)PyGMix_render, METH_VARARGS,  "render without jacobian\n"},
    {"render_jacob",(PyCFunction)PyGMix_render_jacob, METH_VARARGS,  "render with jacobian\n"},

    {"gmix_fill",(PyCFunction)PyGMix_gmix_fill, METH_VARARGS,  "Fill the input gmix from the pars\n"},
    {"convolve_fill",(PyCFunction)PyGMix_convolve_fill, METH_VARARGS,  "convolve gaussian with psf and store in output\n"},
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
    PyObject* m=NULL;


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

    /* register exceptions */
    if (GMixRangeError == NULL) {
        /* NULL = baseclass will be "exception" */
        GMixRangeError = PyErr_NewException("_gmix.GMixRangeError", NULL, NULL);
        if (GMixRangeError) {
            Py_INCREF(GMixRangeError);
            PyModule_AddObject(m, "GMixRangeError", GMixRangeError);
        } else {
            return;
        }
    }
    /* register exceptions */
    if (GMixFatalError == NULL) {
        /* NULL = baseclass will be "exception" */
        GMixFatalError = PyErr_NewException("_gmix.GMixFatalError", NULL, NULL);
        if (GMixFatalError) {
            Py_INCREF(GMixFatalError);
            PyModule_AddObject(m, "GMixFatalError", GMixFatalError);
        } else {
            return;
        }
    }


    // for numpy
    import_array();
}
