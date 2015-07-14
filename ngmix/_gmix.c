/*
 
   Instead of defining types here, represent them with numpy arrays with
   fields.  These map to unpadded structs (defined in _gmix.h).  This allows us
   to do all memory allocation in python instead of here in C.  The principle
   is to never allocate memory in this code except for tuples of scalar return
   values. The only incref we do should be for Py_None This should avoid the
   dredded memory leaks or forgotten incref/decref bugs that plague C
   extensions.

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
    double eta=0, e=0, fac=0;
    double g=sqrt(g1*g1 + g2*g2);

    if (g >= 1) {
        char gstr[25];
        snprintf(gstr,24,"%g", g);
        PyErr_Format(GMixRangeError, "g out of bounds: %s", gstr);
        return 0;
    }
    if (g == 0.0) {
        *e1=0;
        *e2=0;
    } else {

        eta = 2*atanh(g);
        e = tanh(eta);
        if (e >= 1.) {
            // round off?
            e = 0.99999999;
        }

        fac = e/g;

        *e1 = fac*g1;
        *e2 = fac*g2;
    }

    return 1;
}


/*
    convert eta1,eta2 to reduced shear g1,g2
    return 0 means out of range
*/
static int eta1eta2_to_g1g2(double eta1, double eta2, double *g1, double *g2) {
    double g=0, fac=0;
    double eta=sqrt(eta1*eta1 + eta2*eta2);
    //long double tmp=0;

    if (eta == 0.0) {
        *g1=0;
        *g2=0;
    } else {

        /*
        tmp = (long double) eta;
        tmp *= 0.5;
        g=(double) tanhl(tmp);
        */

        g=tanh(0.5*eta);

        if (g >= 1.) {
            return 0;
            // round off?
            //g = 0.99999999999999999;
        }

        fac = g/eta;

        *g1 = fac*eta1;
        *g2 = fac*eta2;
    }

    return 1;
}



static double gmix_get_T(struct PyGMix_Gauss2D *gmix,
                         npy_intp n_gauss)
{

    double T=0.0, psum=0;
    npy_intp i=0;

    for (i=0; i<n_gauss; i++) {
        struct PyGMix_Gauss2D *gauss=&gmix[i];

        psum += gauss->p;
        T += gauss->p*(gauss->irr + gauss->icc);
    }
    T /= psum;
    return T;
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
   
   note for gaussians we plan to convolve with a psf we might
   not care that det < 0, so we don't always evaluate
*/

static int gauss2d_set_norm(struct PyGMix_Gauss2D *self)
{
    int status=0;
    double idet=0;
    if (self->det < 1.0e-200) {

        // PyErr_Format doesn't format floats
        char detstr[25];
        snprintf(detstr,24,"%g", self->det);
        PyErr_Format(GMixRangeError, "gauss2d det too low: %s", detstr);
        status=0;

    } else {

        idet=1.0/self->det;
        self->drr = self->irr*idet;
        self->drc = self->irc*idet;
        self->dcc = self->icc*idet;
        self->norm = 1./(2*M_PI*sqrt(self->det));

        self->pnorm = self->p*self->norm;

        self->norm_set=1;
        status=1;
    }

    return status;

}
static int gauss2d_set(struct PyGMix_Gauss2D *self,
                       double p,
                       double row,
                       double col,
                       double irr,
                       double irc,
                       double icc) {

    // this means norm_set=0 as well as the other pieces not
    // yet calculated
    memset(self, 0, sizeof(struct PyGMix_Gauss2D));

    self->p=p;
    self->row=row;
    self->col=col;
    self->irr=irr;
    self->irc=irc;
    self->icc=icc;

    self->det = irr*icc - irc*irc;

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
        case PYGMIX_GMIX_GAUSSMOM:
            n_gauss=1;
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
static int gmix_set_norms(struct PyGMix_Gauss2D *self,
                          npy_intp n_gauss)
{

    int status=0;
    npy_intp i=0;

    for (i=0; i<n_gauss; i++) {

        status=gauss2d_set_norm(&self[i]);

        // an exception will be set
        if (!status) {
            status=0;
            goto _gmix_set_norms_bail;
        }

    }

    status=1;

_gmix_set_norms_bail:
    return status;
}

static int gmix_set_norms_if_needed(struct PyGMix_Gauss2D *self,
                                    npy_intp n_gauss)
{

    int status=1;
    if (!self->norm_set) {
        status=gmix_set_norms(self, n_gauss);
    }
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

    int status=0, n_gauss_expected=0;
    double row=0,col=0,g1=0,g2=0,
           T=0,counts=0,e1=0,e2=0,
           T_i_2=0,counts_i=0;
    npy_intp i=0;

    if (n_pars != 6) {
        PyErr_Format(GMixFatalError, 
                     "simple pars should be size 6, got %ld", n_pars);
        goto _gmix_fill_simple_bail;
    }

    n_gauss_expected=get_n_gauss(model, &status);
    if (!status) {
        goto _gmix_fill_simple_bail;
    }
    if (n_gauss != n_gauss_expected) {
        PyErr_Format(GMixFatalError, 
                     "for model %d expected %d gauss, got %ld",
                     model, n_gauss_expected, n_gauss);
        goto _gmix_fill_simple_bail;
    }

    row=pars[0];
    col=pars[1];
    g1=pars[2];
    g2=pars[3];
    T=pars[4];
    counts=pars[5];

    // can set exception inside
    status=g1g2_to_e1e2(g1, g2, &e1, &e2);
    if (!status) {
        goto _gmix_fill_simple_bail;
    }


    for (i=0; i<n_gauss; i++) {
        T_i_2 = 0.5*T*fvals[i];
        counts_i=counts*pvals[i];

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


static int gmix_fill_cm(struct PyGMixCM*self,
                               const double* pars,
                               npy_intp n_pars)
{

    int status=0;
    double
        row=0,col=0,
        g1=0,g2=0,e1=0,e2=0,
        T=0,counts=0,
        T_i_2=0,
        counts_i=0,
        f=0, p=0,
        ifracdev=0;

    npy_intp i=0;

    if (n_pars != 6) {
        PyErr_Format(GMixFatalError, 
                     "composite pars should be size 6, got %ld", n_pars);
        goto _bail;
    }

    row=pars[0];
    col=pars[1];
    g1=pars[2];
    g2=pars[3];
    T=pars[4] * self->Tfactor;
    counts=pars[5];

    ifracdev = 1.0-self->fracdev;

    // can set an exception
    status=g1g2_to_e1e2(g1, g2, &e1, &e2);
    if (!status) {
        goto _bail;
    }

    for (i=0; i<16; i++) {
        if (i < 6) {
            p=PyGMix_pvals_exp[i] * ifracdev;
            f=PyGMix_fvals_exp[i];
        } else {
            p=PyGMix_pvals_dev[i-6] * self->fracdev;
            f=PyGMix_fvals_dev[i-6] * self->TdByTe;
        }

        T_i_2 = 0.5*T*f;
        counts_i=counts*p;

        status=gauss2d_set(&self->gmix[i],
                           counts_i,
                           row,
                           col, 
                           T_i_2*(1-e1), 
                           T_i_2*e2,
                           T_i_2*(1+e1));
        // an exception will be set
        if (!status) {
            goto _bail;
        }
    }

    status=1;

_bail:
    return status;
}

static int gmix_fill_gaussmom(struct PyGMix_Gauss2D *self,
                              npy_intp n_gauss,
                              const double* pars,
                              npy_intp n_pars)
{

    int status=0;
    double row=0,col=0, M1=0, M2=0, T=0, I=0;
    double Irr=0,Irc=0, Icc=0;
    npy_intp i=0;

    if (n_pars != 6) {
        PyErr_Format(GMixFatalError, 
                     "gaussmom pars should be size 6, got %ld", n_pars);
        goto _gmix_fill_gaussmom_bail;
    }
    if (n_gauss != 1) {
        PyErr_Format(GMixFatalError, 
                     "gaussmom is for one gaussian, got got %ld", n_gauss);
        goto _gmix_fill_gaussmom_bail;
    }

    row=pars[0];
    col=pars[1];
    M1=pars[2];
    M2=pars[3];
    T=pars[4];
    I=pars[5];

    Irr = (T-M1)*0.5;
    Irc = M2*0.5;
    Icc = (T+M1)*0.5;

    status=gauss2d_set(&self[i],
                       I,
                       row,
                       col, 
                       Irr,
                       Irc,
                       Icc);

    // an exception will be set
    if (!status) {
        goto _gmix_fill_gaussmom_bail;
    }

    status=1;

_gmix_fill_gaussmom_bail:
    return status;
}


static int gmix_fill_coellip(struct PyGMix_Gauss2D *self,
                             npy_intp n_gauss,
                             const double* pars,
                             npy_intp n_pars)
{

    int status=0;//, n_gauss_expected=0;
    double row=0,col=0,g1=0,g2=0,
           T=0,Thalf=0,counts=0,e1=0,e2=0;
    npy_intp i=0;

    row=pars[0];
    col=pars[1];
    g1=pars[2];
    g2=pars[3];

    // can set exception inside
    status=g1g2_to_e1e2(g1, g2, &e1, &e2);
    if (!status) {
        goto _gmix_fill_coellip_bail;
    }


    for (i=0; i<n_gauss; i++) {
        T = pars[4+i];
        Thalf=0.5*T;
        counts=pars[4+n_gauss+i];

        status=gauss2d_set(&self[i],
                           counts,
                           row,
                           col, 
                           Thalf*(1-e1), 
                           Thalf*e2,
                           Thalf*(1+e1));
        // an exception will be set
        if (!status) {
            goto _gmix_fill_coellip_bail;
        }
    }

    status=1;

_gmix_fill_coellip_bail:
    return status;
}


static PyObject * PyGMix_get_cm_Tfactor(PyObject* self, PyObject* args) {
    double
        fracdev=0,
        TdByTe=0,
        ifracdev=0,
        Tfactor=0,
        p=0,f=0;
    long i=0;

    if (!PyArg_ParseTuple(args, (char*)"dd", &fracdev, &TdByTe)) {
        return NULL;
    }

    ifracdev = 1.0-fracdev;
    for (i=0; i<6; i++) {
        p=PyGMix_pvals_exp[i] * ifracdev;
        f=PyGMix_fvals_exp[i];

        Tfactor += p*f;
    }

    for (i=0; i<10; i++) {
        p=PyGMix_pvals_dev[i] * fracdev;
        f=PyGMix_fvals_dev[i] * TdByTe;

        Tfactor += p*f;
    }

    Tfactor = 1.0/Tfactor;

    return Py_BuildValue("d", Tfactor);
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

        case PyGMIX_GMIX_COELLIP:
            status=gmix_fill_coellip(self, n_gauss, pars, n_pars);
            break;

        case PyGMIX_GMIX_FULL:
            status=gmix_fill_full(self, n_gauss, pars, n_pars);
            break;

        case PYGMIX_GMIX_GAUSSMOM:
            status=gmix_fill_gaussmom(self, n_gauss, pars, n_pars);
            break;

        default:
            PyErr_Format(GMixFatalError, 
                         "gmix error: Bad gmix model: %d", model);
            goto _gmix_fill_bail;
            break;
    }

_gmix_fill_bail:
    return status;

}

static PyObject * PyGMix_gmix_fill(PyObject* self, PyObject* args) {
    PyObject* gmix_obj=NULL;
    PyObject* pars_obj=NULL;
    const double* pars=NULL;
    npy_intp n_gauss=0, n_pars=0;
    int res=0;

    int model=0;

    struct PyGMix_Gauss2D *gmix=NULL;

    if (!PyArg_ParseTuple(args, (char*)"OOi",
                          &gmix_obj, 
                          &pars_obj,
                          &model)) {

        return NULL;
    }

    gmix=(struct PyGMix_Gauss2D* ) PyArray_DATA(gmix_obj);
    n_gauss=PyArray_SIZE(gmix_obj);

    pars=(double *) PyArray_DATA(pars_obj);
    n_pars = PyArray_SIZE(pars_obj);

    res=gmix_fill(gmix, n_gauss, pars, n_pars, model);
    if (!res) {
        // raise an exception
        return NULL;
    } else {
        Py_INCREF(Py_None);
        return Py_None;
    }
}

/* no type checking here */
static PyObject * PyGMix_gmix_fill_cm(PyObject* self, PyObject* args) {
    PyObject* composite_obj=NULL;
    PyObject* pars_obj=NULL;
    const double* pars=NULL;
    npy_intp n_pars=0;
    int res=0;

    struct PyGMixCM* comp=NULL;

    if (!PyArg_ParseTuple(args, (char*)"OO",
                          &composite_obj, 
                          &pars_obj)) {

        return NULL;
    }

    comp=(struct PyGMixCM* ) PyArray_DATA(composite_obj);

    pars=(double *) PyArray_DATA(pars_obj);
    n_pars = PyArray_SIZE(pars_obj);

    res=gmix_fill_cm(comp, pars, n_pars);
    if (!res) {
        // raise an exception
        return NULL;
    } else {
        Py_INCREF(Py_None);
        return Py_None;
    }
}



static int convolve_fill(struct PyGMix_Gauss2D *self, npy_intp self_n_gauss,
                         const struct PyGMix_Gauss2D *gmix, npy_intp n_gauss,
                         const struct PyGMix_Gauss2D *psf, npy_intp psf_n_gauss)
{
    int status=0;
    npy_intp ntot=0, iobj=0, ipsf=0, itot=0;
    double psf_rowcen=0, psf_colcen=0, psf_psum=0, psf_ipsum=0;

    ntot = n_gauss*psf_n_gauss;
    if (ntot != self_n_gauss) {
        PyErr_Format(GMixFatalError, 
                     "target gmix is wrong size %ld, expected %ld",
                     self_n_gauss, ntot);
        goto _convolve_fill_bail;
    }

    gmix_get_cen(psf, psf_n_gauss, &psf_rowcen, &psf_colcen, &psf_psum);
    psf_ipsum=1.0/psf_psum;

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

            status=gauss2d_set(&self[itot], 
                               p, row, col, irr, irc, icc);
            // an exception will be set
            if (!status) {
                goto _convolve_fill_bail;
            }

            itot++;
        }
    }

    // we want to do this here, because it will raise an exception if there are
    // issues with the resulting gaussians
    status=gmix_set_norms(self, self_n_gauss);
    if (!status) {
        goto _convolve_fill_bail;
    }

    status=1;
_convolve_fill_bail:
    return status;
}

static PyObject * PyGMix_convolve_fill(PyObject* self, PyObject* args) {
    PyObject* self_gmix_obj=NULL;
    PyObject* gmix_obj=NULL;
    PyObject* psf_gmix_obj=NULL;

    struct PyGMix_Gauss2D *gmix=NULL;
    struct PyGMix_Gauss2D *psf_gmix=NULL;
    struct PyGMix_Gauss2D *self_gmix=NULL;
    npy_intp self_n_gauss=0, n_gauss=0, psf_n_gauss=0;
    int res=0;

    if (!PyArg_ParseTuple(args, (char*)"OOO",
                          &self_gmix_obj,
                          &gmix_obj, 
                          &psf_gmix_obj)) {

        return NULL;
    }

    self_gmix=(struct PyGMix_Gauss2D* ) PyArray_DATA(self_gmix_obj);
    self_n_gauss =PyArray_SIZE(self_gmix_obj);

    gmix=(struct PyGMix_Gauss2D* ) PyArray_DATA(gmix_obj);
    n_gauss=PyArray_SIZE(gmix_obj);

    psf_gmix=(struct PyGMix_Gauss2D* ) PyArray_DATA(psf_gmix_obj);
    psf_n_gauss =PyArray_SIZE(psf_gmix_obj);


    res=convolve_fill(self_gmix, self_n_gauss,
                      gmix, n_gauss,
                      psf_gmix, psf_n_gauss);

    if (!res) {
        // raise an exception
        return NULL;
    } else {

        Py_INCREF(Py_None);
        return Py_None;
    }
}



static PyObject * PyGMix_gmix_set_norms(PyObject* self, PyObject* args) {
    PyObject* gmix_obj=NULL;
    npy_intp n_gauss=0;
    int status=0;

    struct PyGMix_Gauss2D *gmix=NULL;

    if (!PyArg_ParseTuple(args, (char*)"O", &gmix_obj)) {
        return NULL;
    }

    gmix=(struct PyGMix_Gauss2D* ) PyArray_DATA(gmix_obj);
    n_gauss=PyArray_SIZE(gmix_obj);

    status=gmix_set_norms(gmix, n_gauss);
    if (!status) {
        // raise an exception
        return NULL;
    } else {
        Py_INCREF(Py_None);
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

    if (!gmix_set_norms_if_needed(gmix, n_gauss)) {
        return NULL;
    }

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

                    tval += PYGMIX_GMIX_EVAL_FULL(gmix, n_gauss, trow, tcol);
                    //tval += PYGMIX_GMIX_EVAL(gmix, n_gauss, trow, tcol);

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

    Py_INCREF(Py_None);
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

    if (!gmix_set_norms_if_needed(gmix, n_gauss)) {
        return NULL;
    }

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
                //u=jacob->dudrow*(trow - jacob->row0) + jacob->dudcol*(lowcol - jacob->col0);
                //v=jacob->dvdrow*(trow - jacob->row0) + jacob->dvdcol*(lowcol - jacob->col0);
                u=PYGMIX_JACOB_GETU(jacob, trow, lowcol);
                v=PYGMIX_JACOB_GETV(jacob, trow, lowcol);

                for (colsub=0; colsub<nsub; colsub++) {

                    tval += PYGMIX_GMIX_EVAL_FULL(gmix, n_gauss, u, v);
                    //tval += PYGMIX_GMIX_EVAL(gmix, n_gauss, u, v);

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

    Py_INCREF(Py_None);
    return Py_None;
}


/*
   Calculate the image mean, accounting for weight function.
*/

static PyObject * PyGMix_get_image_mean(PyObject* self, PyObject* args) {

    PyObject* image_obj=NULL;
    PyObject* weight_obj=NULL;
    npy_intp n_row=0, n_col=0, row=0, col=0;//, igauss=0;

    double data=0, ivar=0;
    double wsum=0, imsum=0, wmean=0;

    if (!PyArg_ParseTuple(args, (char*)"OO", &image_obj, &weight_obj)) {
        return NULL;
    }

    n_row=PyArray_DIM(image_obj, 0);
    n_col=PyArray_DIM(image_obj, 1);

    for (row=0; row < n_row; row++) {
        for (col=0; col < n_col; col++) {

            ivar=*( (double*)PyArray_GETPTR2(weight_obj,row,col) );
            if ( ivar > 0.0) {
                data=*( (double*)PyArray_GETPTR2(image_obj,row,col) );

                wsum += ivar;
                imsum += ivar*data;
            }
        }
    }

    if (wsum > 0) {
        wmean = imsum/wsum;
    }

    return Py_BuildValue("d", wmean);
}


/*
   Calculate the sum needed for the model s/n

   This is s2n_sum = sum(model_i^2 * ivar_i)

   The s/n will be sqrt(s2n_sum)
*/
static PyObject * PyGMix_get_model_s2n_sum(PyObject* self, PyObject* args) {

    PyObject* gmix_obj=NULL;
    PyObject* weight_obj=NULL;
    PyObject* jacob_obj=NULL;
    npy_intp n_gauss=0, n_row=0, n_col=0, row=0, col=0;//, igauss=0;

    struct PyGMix_Gauss2D *gmix=NULL;//, *gauss=NULL;
    struct PyGMix_Jacobian *jacob=NULL;

    double ivar=0, u=0, v=0;
    double model_val=0;
    double s2n_sum=0;

    if (!PyArg_ParseTuple(args, (char*)"OOO", 
                          &gmix_obj, &weight_obj, &jacob_obj)) {
        return NULL;
    }

    gmix=(struct PyGMix_Gauss2D* ) PyArray_DATA(gmix_obj);
    n_gauss=PyArray_SIZE(gmix_obj);

    if (!gmix_set_norms_if_needed(gmix, n_gauss)) {
        return NULL;
    }

    n_row=PyArray_DIM(weight_obj, 0);
    n_col=PyArray_DIM(weight_obj, 1);

    jacob=(struct PyGMix_Jacobian* ) PyArray_DATA(jacob_obj);

    for (row=0; row < n_row; row++) {
        u=PYGMIX_JACOB_GETU(jacob, row, 0);
        v=PYGMIX_JACOB_GETV(jacob, row, 0);

        for (col=0; col < n_col; col++) {

            ivar=*( (double*)PyArray_GETPTR2(weight_obj,row,col) );
            if ( ivar > 0.0) {
                model_val=PYGMIX_GMIX_EVAL(gmix, n_gauss, u, v);

                s2n_sum += model_val*model_val*ivar;
            }

            u += jacob->dudcol;
            v += jacob->dvdcol;

        }
    }

    return Py_BuildValue("d", s2n_sum);
}

/*

   Same as above but also calculating Ts2n sums

   Calculate the sum needed for the model s/n

   This is s2n_sum = sum(model_i^2 * ivar_i)

   The s/n will be sqrt(s2n_sum)

   Also the sums need for var(T)  This is Mike's approximate formula, which
   I have verified (ignoring covariance between numerator and denominator)

   var(T) =
   Sum (w_p^2 r_p^4 / var_p) / (Sum (w_p^2 / var_p))^2 + (Sum (w_p^2 r_p^2 / var_p))^2 / (Sum (w_p^2 / var_p))^3


*/

static PyObject * PyGMix_get_model_s2n_Tvar_sums(PyObject* self, PyObject* args) {

    PyObject* gmix_obj=NULL;
    PyObject* weight_obj=NULL;
    PyObject* jacob_obj=NULL;
    npy_intp n_gauss=0, n_row=0, n_col=0, row=0, col=0;//, igauss=0;

    struct PyGMix_Gauss2D *gmix=NULL;//, *gauss=NULL;
    struct PyGMix_Jacobian *jacob=NULL;

    double ivar=0, u=0, v=0;
    double model_val=0;
    double s2n_sum=0, r4sum=0, r2sum=0, r2=0, r4=0, m2=0;
    double rowcen=0, colcen=0, psum=0, rowmod=0, colmod=0;

    if (!PyArg_ParseTuple(args, (char*)"OOO", 
                          &gmix_obj, &weight_obj, &jacob_obj)) {
        return NULL;
    }

    gmix=(struct PyGMix_Gauss2D* ) PyArray_DATA(gmix_obj);
    n_gauss=PyArray_SIZE(gmix_obj);

    if (!gmix_set_norms_if_needed(gmix, n_gauss)) {
        return NULL;
    }

    gmix_get_cen(gmix, n_gauss, &rowcen, &colcen, &psum);

    n_row=PyArray_DIM(weight_obj, 0);
    n_col=PyArray_DIM(weight_obj, 1);

    jacob=(struct PyGMix_Jacobian* ) PyArray_DATA(jacob_obj);

    for (row=0; row < n_row; row++) {
        u=PYGMIX_JACOB_GETU(jacob, row, 0);
        v=PYGMIX_JACOB_GETV(jacob, row, 0);

        for (col=0; col < n_col; col++) {

            ivar=*( (double*)PyArray_GETPTR2(weight_obj,row,col) );
            if ( ivar > 0.0) {
                model_val=PYGMIX_GMIX_EVAL(gmix, n_gauss, u, v);
                m2 = model_val*model_val;

                s2n_sum += m2*ivar;

                rowmod=u-rowcen;
                colmod=v-colcen;

                r2 = rowmod*rowmod + colmod*colmod;
                r4 = r2*r2;
                r4sum += m2 * r4 * ivar;
                r2sum += m2 * r2 * ivar;
            }

            u += jacob->dudcol;
            v += jacob->dvdcol;

        }
    }

    return Py_BuildValue("ddd", s2n_sum, r2sum, r4sum);
}

/*
   with alternate weight
*/
static PyObject * PyGMix_get_model_s2n_Tvar_sums_altweight(PyObject* self, PyObject* args) {

    PyObject* gmix_obj=NULL;
    PyObject* wgmix_obj=NULL;
    PyObject* weight_obj=NULL;
    PyObject* jacob_obj=NULL;
    npy_intp n_gauss=0, wn_gauss=0, n_row=0, n_col=0, row=0, col=0;//, igauss=0;

    struct PyGMix_Gauss2D *gmix=NULL;//, *gauss=NULL;
    struct PyGMix_Gauss2D *wgmix=NULL;//, *gauss=NULL;
    struct PyGMix_Jacobian *jacob=NULL;

    double ivar=0, u=0, v=0;
    double model_val=0, wval=0;
    double s2n_sum=0, r4sum=0, r2sum=0, r2=0, r4=0, m2=0;
    double rowcen=0, colcen=0, psum=0, rowmod=0, colmod=0;

    if (!PyArg_ParseTuple(args, (char*)"OOOO", 
                          &gmix_obj, &wgmix_obj, &weight_obj, &jacob_obj)) {
        return NULL;
    }

    gmix=(struct PyGMix_Gauss2D* ) PyArray_DATA(gmix_obj);
    n_gauss=PyArray_SIZE(gmix_obj);

    wgmix=(struct PyGMix_Gauss2D* ) PyArray_DATA(wgmix_obj);
    wn_gauss=PyArray_SIZE(wgmix_obj);

    if (!gmix_set_norms_if_needed(gmix, n_gauss)) {
        return NULL;
    }
    if (!gmix_set_norms_if_needed(wgmix, wn_gauss)) {
        return NULL;
    }

    gmix_get_cen(gmix, n_gauss, &rowcen, &colcen, &psum);

    n_row=PyArray_DIM(weight_obj, 0);
    n_col=PyArray_DIM(weight_obj, 1);

    jacob=(struct PyGMix_Jacobian* ) PyArray_DATA(jacob_obj);

    for (row=0; row < n_row; row++) {
        u=PYGMIX_JACOB_GETU(jacob, row, 0);
        v=PYGMIX_JACOB_GETV(jacob, row, 0);

        for (col=0; col < n_col; col++) {

            ivar=*( (double*)PyArray_GETPTR2(weight_obj,row,col) );
            if ( ivar > 0.0) {
                model_val=PYGMIX_GMIX_EVAL(gmix, n_gauss, u, v);
                wval=PYGMIX_GMIX_EVAL(wgmix, wn_gauss, u, v);

                m2 = wval*model_val;

                s2n_sum += m2*ivar;

                rowmod=u-rowcen;
                colmod=v-colcen;

                r2 = rowmod*rowmod + colmod*colmod;
                r4 = r2*r2;
                r4sum += m2 * r4 * ivar;
                r2sum += m2 * r2 * ivar;
            }

            u += jacob->dudcol;
            v += jacob->dvdcol;

        }
    }

    return Py_BuildValue("ddd", s2n_sum, r2sum, r4sum);
}



/*

Get the weighted moments of the image, using the input gaussian
mixture as the weight function.  The moments are *not* normalized

Just iterating for the centroid, with the first location taken as the
jacobian center, so you should have a good guess

These are moments, so there cannot be masked portions of the image.

In the following, W is the weight function, I is the image, and
w is the weight map

Returns

       ucen  = sum(W*I*u)/sum(W*I)
       vcen  = sum(W*I*v)/sum(W*I)
       Isum  = sum(W*I)
       Tsum  = sum(W * I * {u^2 + v^2} )
       M1sum = sum(W * I * {u^2 - v^2} )
       M2sum = sum(W * I * 2*u*v)

where u,v are relative to the jacobian center

Also returned are sums used to calculate variances in these quantities, but
note the covariance can be significant

       VIsum  = sum(W^2)
       VTsum  = sum(W^2 * {u^2 + v^2}^2 )
       VM1sum = sum(W^2 * {u^2 - v^2}^2 )
       VM2sum = sum(W^2 * {2*u*v}^2 )

These should be multiplied by the noise^2 to turn them into proper variances

*/
static PyObject * PyGMix_get_weighted_mom_sums(PyObject* self, PyObject* args) {

    PyObject* gmix_obj=NULL;
    PyObject* image_obj=NULL;
    PyObject* jacob_obj=NULL;
    PyObject* cen_obj=NULL;
    PyObject* pars_obj=NULL;
    PyObject* pvar_obj=NULL;

    npy_intp n_gauss=0, n_row=0, n_col=0, row=0, col=0;//, igauss=0;

    struct PyGMix_Gauss2D *gmix=NULL;//, *gauss=NULL;
    struct PyGMix_Jacobian *jacob=NULL;
    double *cen=NULL, *pars=NULL, *pvar=NULL;

    double u=0, v=0, umod=0, vmod=0;
    double wdata=0, data=0, weight=0, w2=0;
    double T=0,M1=0,M2=0;
    double Isum=0, usum=0, vsum=0, Tsum=0, M1sum=0, M2sum=0;
    double VIsum=0, VTsum=0, VM1sum=0, VM2sum=0;
    double Vusum=0, Vvsum=0;
    double ucen=0, vcen=0, ucenold=0, vcenold=0;
    double centol=0, max_shift=0;
    int maxiter=0, niter=0, flags=0;

    if (!PyArg_ParseTuple(args, (char*)"OOOiddOOO", 
                          &image_obj, &gmix_obj, &jacob_obj,
                          &maxiter, &centol, &max_shift,
                          &cen_obj, &pars_obj, &pvar_obj)) {
        return NULL;
    }

    gmix=(struct PyGMix_Gauss2D* ) PyArray_DATA(gmix_obj);
    n_gauss=PyArray_SIZE(gmix_obj);

    if (!gmix_set_norms_if_needed(gmix, n_gauss)) {
        return NULL;
    }

    n_row=PyArray_DIM(image_obj, 0);
    n_col=PyArray_DIM(image_obj, 1);

    jacob=(struct PyGMix_Jacobian* ) PyArray_DATA(jacob_obj);

    // start guess at the jacobian center
    ucen=0.0;
    vcen=0.0;
    ucenold=9999;
    vcenold=9999;

    // iterate for the centroid
    for (niter=0; niter<maxiter; niter++) {

        Isum=0;
        usum=0;
        vsum=0;
        for (row=0; row < n_row; row++) {
            u=PYGMIX_JACOB_GETU(jacob, row, 0);
            v=PYGMIX_JACOB_GETV(jacob, row, 0);

            for (col=0; col < n_col; col++) {

                data=*( (double*)PyArray_GETPTR2(image_obj,row,col) );

                umod = u-ucen;
                vmod = v-vcen;
 
                weight=PYGMIX_GMIX_EVAL_FULL(gmix, n_gauss, umod, vmod);

                wdata = weight*data;

                Isum += wdata;
                usum += wdata*umod;
                vsum += wdata*vmod;

                u += jacob->dudcol;
                v += jacob->dvdcol;

            }
        }

        if (Isum <= 0.0) {
            flags = 2;
            break;
        }
        ucen = usum/Isum;
        vcen = vsum/Isum;
        //fprintf(stderr,"%.10g %.10g %.10g %.10g\n", usum, vsum, ucen, vcen);

        if ((fabs(ucen) > max_shift) || (fabs(vcen) > max_shift)) {
            flags = 4;
            break;
        }

        if ((fabs(ucen-ucenold) < centol) && (fabs(vcen-vcenold) < centol)) {
            break;
        }
        ucenold=ucen;
        vcenold=vcen;
    }

    if (niter==maxiter) {
        flags = 1;
    } else {
        niter += 1;
    }

    if (flags != 0) {
        goto _getmom_bail;
    }

    Isum=0;
    for (row=0; row < n_row; row++) {
        u=PYGMIX_JACOB_GETU(jacob, row, 0);
        v=PYGMIX_JACOB_GETV(jacob, row, 0);

        for (col=0; col < n_col; col++) {

            data=*( (double*)PyArray_GETPTR2(image_obj,row,col) );

            umod = u-ucen;
            vmod = v-vcen;

            weight=PYGMIX_GMIX_EVAL_FULL(gmix, n_gauss, umod, vmod);


            T = umod*umod + vmod*vmod;
            M1 = vmod*vmod - umod*umod;
            M2 = 2*umod*vmod;

            wdata = weight*data;
            w2 = weight*weight;

            Isum  += wdata;
            Tsum  += wdata*T;
            M1sum += wdata*M1;
            M2sum += wdata*M2;

            VIsum += w2;
            Vusum  += w2*umod*umod;
            Vvsum  += w2*vmod*vmod;
            VTsum  += w2*T*T;
            VM1sum += w2*M1*M1;
            VM2sum += w2*M2*M2;

            u += jacob->dudcol;
            v += jacob->dvdcol;

        }
    }

_getmom_bail:

    cen=PyArray_DATA(cen_obj);
    pars=PyArray_DATA(pars_obj);
    pvar=PyArray_DATA(pvar_obj);

    cen[0]=ucen;
    cen[1]=vcen;

    pars[0]=usum;
    pars[1]=vsum;
    pars[2]=M1sum;
    pars[3]=M2sum;
    pars[4]=Tsum;
    pars[5]=Isum;

    pvar[0]=Vusum;
    pvar[1]=Vvsum;
    pvar[2]=VM1sum;
    pvar[3]=VM2sum;
    pvar[4]=VTsum;
    pvar[5]=VIsum;

    return Py_BuildValue("ii",  niter, flags);
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

    long npix = 0;

    PyObject* retval=NULL;

    if (!PyArg_ParseTuple(args, (char*)"OOOO", 
                          &gmix_obj, &image_obj, &weight_obj, &jacob_obj)) {
        return NULL;
    }

    gmix=(struct PyGMix_Gauss2D* ) PyArray_DATA(gmix_obj);
    n_gauss=PyArray_SIZE(gmix_obj);

    if (!gmix_set_norms_if_needed(gmix, n_gauss)) {
        return NULL;
    }

    n_row=PyArray_DIM(image_obj, 0);
    n_col=PyArray_DIM(image_obj, 1);

    jacob=(struct PyGMix_Jacobian* ) PyArray_DATA(jacob_obj);

    for (row=0; row < n_row; row++) {
        //u=jacob->dudrow*(row - jacob->row0) + jacob->dudcol*(0 - jacob->col0);
        //v=jacob->dvdrow*(row - jacob->row0) + jacob->dvdcol*(0 - jacob->col0);
        u=PYGMIX_JACOB_GETU(jacob, row, 0);
        v=PYGMIX_JACOB_GETV(jacob, row, 0);

        for (col=0; col < n_col; col++) {

            ivar=*( (double*)PyArray_GETPTR2(weight_obj,row,col) );
            if ( ivar > 0.0) {
                data=*( (double*)PyArray_GETPTR2(image_obj,row,col) );
                model_val=PYGMIX_GMIX_EVAL(gmix, n_gauss, u, v);

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
    PYGMIX_PACK_RESULT4();
    return retval;
}

/*
   Calculate the loglike between the gmix and the input image

   only evaluate the loglike within the specified circular aperture, centered
   on the canonical center of the jacobian.  This only makes sense if the
   jacobian center is near the true center

   Error checking should be done in python.
*/
static PyObject * PyGMix_get_loglike_aper(PyObject* self, PyObject* args) {

    PyObject* gmix_obj=NULL;
    PyObject* image_obj=NULL;
    PyObject* weight_obj=NULL;
    PyObject* jacob_obj=NULL;
    double aperture=0, ap2=0, rad2=0;
    npy_intp n_gauss=0, n_row=0, n_col=0, row=0, col=0;//, igauss=0;

    struct PyGMix_Gauss2D *gmix=NULL;//, *gauss=NULL;
    struct PyGMix_Jacobian *jacob=NULL;

    double data=0, ivar=0, u=0, v=0;
    double model_val=0, diff=0;
    double s2n_numer=0.0, s2n_denom=0.0, loglike = 0.0;

    long npix=0;

    PyObject* retval=NULL;

    if (!PyArg_ParseTuple(args, (char*)"OOOOd", 
                          &gmix_obj, &image_obj, &weight_obj, &jacob_obj, &aperture)) {
        return NULL;
    }

    gmix=(struct PyGMix_Gauss2D* ) PyArray_DATA(gmix_obj);
    n_gauss=PyArray_SIZE(gmix_obj);

    if (!gmix_set_norms_if_needed(gmix, n_gauss)) {
        return NULL;
    }

    n_row=PyArray_DIM(image_obj, 0);
    n_col=PyArray_DIM(image_obj, 1);

    jacob=(struct PyGMix_Jacobian* ) PyArray_DATA(jacob_obj);

    ap2=aperture*aperture;
    for (row=0; row < n_row; row++) {
        //u=jacob->dudrow*(row - jacob->row0) + jacob->dudcol*(0 - jacob->col0);
        //v=jacob->dvdrow*(row - jacob->row0) + jacob->dvdcol*(0 - jacob->col0);
        u=PYGMIX_JACOB_GETU(jacob, row, 0);
        v=PYGMIX_JACOB_GETV(jacob, row, 0);

        for (col=0; col < n_col; col++) {

            // distance from jacobian center
            rad2=u*u + v*v;
            if (rad2 <= ap2) {

                ivar=*( (double*)PyArray_GETPTR2(weight_obj,row,col) );
                if ( ivar > 0.0) {
                    data=*( (double*)PyArray_GETPTR2(image_obj,row,col) );
                    model_val=PYGMIX_GMIX_EVAL(gmix, n_gauss, u, v);

                    diff = model_val-data;
                    loglike += diff*diff*ivar;
                    s2n_numer += data*model_val*ivar;
                    s2n_denom += model_val*model_val*ivar;

                    npix += 1;
                }

            }
            u += jacob->dudcol;
            v += jacob->dvdcol;

        }
    }

    loglike *= (-0.5);

    // fill in the retval
    PYGMIX_PACK_RESULT4();
    return retval;
}



/*
   Calculate the loglike between the input image and the model image,
   subtracting off the mean in each case

   Error checking should be done in python.
*/
static PyObject * PyGMix_get_loglike_images_margsky(PyObject* self, PyObject* args) {

    PyObject* image_obj=NULL;
    PyObject* weight_obj=NULL;
    PyObject* model_image_obj=NULL;
    npy_intp n_row=0, n_col=0, row=0, col=0;

    double image_mean=0, model_mean=0;
    double data=0, ivar=0, data_mod=0, model_mod=0;
    double model_val=0, diff=0;
    double s2n_numer=0.0, s2n_denom=0.0, loglike = 0.0;

    long npix=0;

    PyObject* retval=NULL;

    if (!PyArg_ParseTuple(args, (char*)"OdOOd", 
                          &image_obj, &image_mean, &weight_obj, &model_image_obj, &model_mean)) {
        return NULL;
    }

    n_row=PyArray_DIM(image_obj, 0);
    n_col=PyArray_DIM(image_obj, 1);

    for (row=0; row < n_row; row++) {
        for (col=0; col < n_col; col++) {

            ivar=*( (double*)PyArray_GETPTR2(weight_obj,row,col) );
            if ( ivar > 0.0) {
                data      = *( (double*)PyArray_GETPTR2(image_obj,row,col) );
                model_val = *( (double*)PyArray_GETPTR2(model_image_obj,row,col) );

                data_mod=data-image_mean;
                model_mod=model_val-model_mean;

                diff = model_mod - data_mod;
                loglike += diff*diff*ivar;
                s2n_numer += data_mod*model_mod*ivar;
                s2n_denom += model_mod*model_mod*ivar;

                npix += 1;
            }
        }
    }

    loglike *= (-0.5);

    // fill in the retval
    PYGMIX_PACK_RESULT4();
    return retval;
}


static PyObject * PyGMix_get_loglike_sub(PyObject* self, PyObject* args) {

    PyObject* gmix_obj=NULL;
    PyObject* image_obj=NULL;
    PyObject* weight_obj=NULL;
    PyObject* jacob_obj=NULL;
    npy_intp n_gauss=0, n_row=0, n_col=0, row=0, col=0, colsub=0, rowsub=0;

    struct PyGMix_Gauss2D *gmix=NULL;//, *gauss=NULL;
    struct PyGMix_Jacobian *jacob=NULL;

    double data=0, ivar=0, u=0, v=0;
    double stepsize=0, ustepsize=0, vstepsize=0, offset=0, areafac=0;
    double model_val=0, diff=0;
    double trow=0, lowcol=0, s2n_numer=0.0, s2n_denom=0.0, loglike = 0.0;
    int nsub=0;

    long npix=0;

    PyObject* retval=NULL;

    if (!PyArg_ParseTuple(args, (char*)"OOOOi", 
                          &gmix_obj, &image_obj, &weight_obj, &jacob_obj, &nsub)) {
        return NULL;
    }

    gmix=(struct PyGMix_Gauss2D* ) PyArray_DATA(gmix_obj);
    n_gauss=PyArray_SIZE(gmix_obj);

    if (!gmix_set_norms_if_needed(gmix, n_gauss)) {
        return NULL;
    }

    jacob=(struct PyGMix_Jacobian* ) PyArray_DATA(jacob_obj);

    stepsize = 1./nsub;
    offset = (nsub-1)*stepsize/2.;
    areafac = 1./(nsub*nsub);
    ustepsize = stepsize*jacob->dudcol;
    vstepsize = stepsize*jacob->dvdcol;

    n_row=PyArray_DIM(image_obj, 0);
    n_col=PyArray_DIM(image_obj, 1);


    for (row=0; row < n_row; row++) {
        for (col=0; col < n_col; col++) {

            ivar=*( (double*)PyArray_GETPTR2(weight_obj,row,col) );
            if ( ivar > 0.0) {

                npix += 1;

                trow = row-offset;
                lowcol = col-offset;

                model_val=0.;
                for (rowsub=0; rowsub<nsub; rowsub++) {
                    u=PYGMIX_JACOB_GETU(jacob, trow, lowcol);
                    v=PYGMIX_JACOB_GETV(jacob, trow, lowcol);

                    for (colsub=0; colsub<nsub; colsub++) {

                        model_val += PYGMIX_GMIX_EVAL(gmix, n_gauss, u, v);

                        u += ustepsize;
                        v += vstepsize;
                    } // colsub

                    trow += stepsize;
                } // rowsub

                model_val *= areafac;

                data=*( (double*)PyArray_GETPTR2(image_obj,row,col) );

                diff = model_val-data;
                loglike += diff*diff*ivar;
                s2n_numer += data*model_val*ivar;
                s2n_denom += model_val*model_val*ivar;
            }

        }
    }

    loglike *= (-0.5);

    // fill in the retval
    PYGMIX_PACK_RESULT4();
    return retval;
}


/*
   Calculate the loglike between the gmix and the input image

   Error checking should be done in python.
   
   logfactor = log(gamma((nu+1)/2)/(gamma(nu/2)*sqrt(pi*nu)))
*/
static PyObject * PyGMix_get_loglike_robust(PyObject* self, PyObject* args) {

    PyObject* gmix_obj=NULL;
    PyObject* image_obj=NULL;
    PyObject* weight_obj=NULL;
    PyObject* jacob_obj=NULL;
    double nu;
    npy_intp n_gauss=0, n_row=0, n_col=0, row=0, col=0;//, igauss=0;

    struct PyGMix_Gauss2D *gmix=NULL;//, *gauss=NULL;
    struct PyGMix_Jacobian *jacob=NULL;

    double data=0, ivar=0, u=0, v=0;
    double model_val=0, diff=0;
    double s2n_numer=0.0, s2n_denom=0.0, loglike=0.0, nupow=0, logfactor=0;

    long npix=0;

    PyObject* retval=NULL;

    if (!PyArg_ParseTuple(args, (char*)"OOOOd", 
                          &gmix_obj, &image_obj, &weight_obj, &jacob_obj, 
                          &nu)) {
        return NULL;
    }

    gmix=(struct PyGMix_Gauss2D* ) PyArray_DATA(gmix_obj);
    n_gauss=PyArray_SIZE(gmix_obj);

    if (!gmix_set_norms_if_needed(gmix, n_gauss)) {
        return NULL;
    }

    n_row=PyArray_DIM(image_obj, 0);
    n_col=PyArray_DIM(image_obj, 1);

    jacob=(struct PyGMix_Jacobian* ) PyArray_DATA(jacob_obj);
    
    logfactor = lgamma((nu+1.0)/2.0) - lgamma(nu/2.0) - 0.5*log(M_PI*nu);
    nupow = -0.5*(nu+1.0);
    
    for (row=0; row < n_row; row++) {
        //u=jacob->dudrow*(row - jacob->row0) + jacob->dudcol*(0 - jacob->col0);
        //v=jacob->dvdrow*(row - jacob->row0) + jacob->dvdcol*(0 - jacob->col0);
        u=PYGMIX_JACOB_GETU(jacob, row, 0);
        v=PYGMIX_JACOB_GETV(jacob, row, 0);

        for (col=0; col < n_col; col++) {

            ivar=*( (double*)PyArray_GETPTR2(weight_obj,row,col) );
            if ( ivar > 0.0) {
                data=*( (double*)PyArray_GETPTR2(image_obj,row,col) );
                model_val=PYGMIX_GMIX_EVAL(gmix, n_gauss, u, v);

                diff = model_val-data;
                loglike += logfactor + nupow*log(1.0+diff*diff*ivar/nu);
                s2n_numer += data*model_val*ivar;
                s2n_denom += model_val*model_val*ivar;

                npix += 1;
            }

            u += jacob->dudcol;
            v += jacob->dvdcol;

        }
    }

    // fill in the retval
    PYGMIX_PACK_RESULT4();

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

    long npix=0;

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

    if (!gmix_set_norms_if_needed(gmix, n_gauss)) {
        return NULL;
    }

    n_row=PyArray_DIM(image_obj, 0);
    n_col=PyArray_DIM(image_obj, 1);

    jacob=(struct PyGMix_Jacobian* ) PyArray_DATA(jacob_obj);

    // we might start somewhere after the priors
    // note fdiff is 1-d
    fdiff_ptr=(double *)PyArray_GETPTR1(fdiff_obj,start);

    for (row=0; row < n_row; row++) {
        u=PYGMIX_JACOB_GETU(jacob, row, 0);
        v=PYGMIX_JACOB_GETV(jacob, row, 0);

        for (col=0; col < n_col; col++) {

            ivar=*( (double*)PyArray_GETPTR2(weight_obj,row,col) );
            if ( ivar > 0.0) {
                ierr=sqrt(ivar);

                data=*( (double*)PyArray_GETPTR2(image_obj,row,col) );
                model_val=PYGMIX_GMIX_EVAL(gmix, n_gauss, u, v);

                (*fdiff_ptr) = (model_val-data)*ierr;
                s2n_numer += data*model_val*ivar;
                s2n_denom += model_val*model_val*ivar;

                npix += 1;
            } else {
                (*fdiff_ptr) = 0.0;
            }

            fdiff_ptr++;

            u += jacob->dudcol;
            v += jacob->dvdcol;

        }
    }

    // fill in the retval
    PYGMIX_PACK_RESULT3();
    return retval;
}

/*
   Fill the input fdiff=(model-data)/err, return s2n_numer, s2n_denom
   with sub-pixel integration

   Error checking should be done in python.
*/
static PyObject * PyGMix_fill_fdiff_sub(PyObject* self, PyObject* args) {

    PyObject* gmix_obj=NULL;
    PyObject* image_obj=NULL;
    PyObject* weight_obj=NULL;
    PyObject* jacob_obj=NULL;
    PyObject* fdiff_obj=NULL;
    npy_intp n_gauss=0, n_row=0, n_col=0, row=0, col=0;//, igauss=0;
    int start=0, nsub=0;

    long npix=0;

    struct PyGMix_Gauss2D *gmix=NULL;//, *gauss=NULL;
    struct PyGMix_Jacobian *jacob=NULL;

    double data=0, ivar=0, ierr=0, u=0, v=0, *fdiff_ptr=NULL;
    double model_val=0;
    double s2n_numer=0.0, s2n_denom=0.0;
    double stepsize=0, ustepsize=0, vstepsize=0, 
           offset=0, areafac=0, trow=0, lowcol=0;
    npy_intp rowsub=0, colsub=0;

    PyObject* retval=NULL;

    if (!PyArg_ParseTuple(args, (char*)"OOOOOii", 
                          &gmix_obj, &image_obj, &weight_obj, &jacob_obj,
                          &fdiff_obj, &start, &nsub)) {
        return NULL;
    }

    gmix=(struct PyGMix_Gauss2D* ) PyArray_DATA(gmix_obj);
    n_gauss=PyArray_SIZE(gmix_obj);

    if (!gmix_set_norms_if_needed(gmix, n_gauss)) {
        return NULL;
    }

    jacob=(struct PyGMix_Jacobian* ) PyArray_DATA(jacob_obj);

    stepsize = 1./nsub;
    offset = (nsub-1)*stepsize/2.;
    areafac = 1./(nsub*nsub);
    ustepsize = stepsize*jacob->dudcol;
    vstepsize = stepsize*jacob->dvdcol;

    n_row=PyArray_DIM(image_obj, 0);
    n_col=PyArray_DIM(image_obj, 1);


    // we might start somewhere after the priors
    // note fdiff is 1-d
    fdiff_ptr=(double *)PyArray_GETPTR1(fdiff_obj,start);

    for (row=0; row < n_row; row++) {
        for (col=0; col < n_col; col++) {

            ivar=*( (double*)PyArray_GETPTR2(weight_obj,row,col) );
            if ( ivar > 0.0) {

                npix += 1;

                trow = row-offset;
                lowcol = col-offset;

                model_val=0.;
                for (rowsub=0; rowsub<nsub; rowsub++) {
                    //u=jacob->dudrow*(trow - jacob->row0) + jacob->dudcol*(lowcol - jacob->col0);
                    //v=jacob->dvdrow*(trow - jacob->row0) + jacob->dvdcol*(lowcol - jacob->col0);
                    u=PYGMIX_JACOB_GETU(jacob, trow, lowcol);
                    v=PYGMIX_JACOB_GETV(jacob, trow, lowcol);

                    for (colsub=0; colsub<nsub; colsub++) {

                        model_val += PYGMIX_GMIX_EVAL(gmix, n_gauss, u, v);

                        u += ustepsize;
                        v += vstepsize;
                    } // colsub

                    trow += stepsize;
                } // rowsub

                model_val *= areafac;

                ierr=sqrt(ivar);
                data=*( (double*)PyArray_GETPTR2(image_obj,row,col) );

                (*fdiff_ptr) = (model_val-data)*ierr;
                s2n_numer += data*model_val*ivar;
                s2n_denom += model_val*model_val*ivar;
            } else {
                (*fdiff_ptr) = 0.0;
            }

            fdiff_ptr++;

        }
    }

    // fill in the retval
    PYGMIX_PACK_RESULT3();
    return retval;
}


/*
 *
   Expectation maximization image fitting
 *
 */


static void em_clear_sums(struct PyGMix_EM_Sums *sums, npy_intp n_gauss)
{
    memset(sums, 0, n_gauss*sizeof(struct PyGMix_EM_Sums));
}

/*
static void em_sums_print(const struct PyGMix_EM_Sums *sums, npy_intp n_gauss)
{
    npy_intp i=0;
    for (i=0; i<n_gauss; i++) {
        const struct PyGMix_EM_Sums *sum=&sums[i];

        fprintf(stderr,"%ld: %g %g %g %g %g %g %g %g %g %g %g %g\n",
                i+1,
                sum->gi,
                sum->trowsum,
                sum->tcolsum,
                sum->tu2sum,
                sum->tuvsum,
                sum->tv2sum,

                sum->pnew,
                sum->rowsum,
                sum->colsum,
                sum->u2sum,
                sum->uvsum,
                sum->v2sum);

    }
}
*/

/*

   note for em we immediately set the normalization, unlike shear measurements
   where we allow T <= 0.0

*/
static 
int em_set_gmix_from_sums(struct PyGMix_Gauss2D *gmix,
                           npy_intp n_gauss,
                           const struct PyGMix_EM_Sums *sums)
{
    int status=0;
    npy_intp i=0;
    for (i=0; i<n_gauss; i++) {
        const struct PyGMix_EM_Sums *sum=&sums[i];
        struct PyGMix_Gauss2D *gauss=&gmix[i];

        double p=sum->pnew;
        double pinv=1.0/p;

        status=gauss2d_set(gauss,
                           p,
                           sum->rowsum*pinv,
                           sum->colsum*pinv,
                           sum->u2sum*pinv,
                           sum->uvsum*pinv,
                           sum->v2sum*pinv);

        status=gauss2d_set_norm(gauss);

        // an exception will be set
        if (!status) {
            goto _em_set_gmix_from_sums_bail;
        }
    }
    status=1;

_em_set_gmix_from_sums_bail:
    return status;
}


/*
   input gmix is guess and will eventually hold the final
   stage of the iteration
*/
static int em_run(PyObject* image_obj,
                  double sky,
                  double counts,
                  const struct PyGMix_Jacobian* jacob,
                  struct PyGMix_Gauss2D *gmix, // holds the guess
                  npy_intp n_gauss,
                  struct PyGMix_EM_Sums *sums,
                  double tol,
                  long maxiter,
                  long *numiter,
                  double *frac_diff)
{
    int status=0;
    double skysum=0;
    npy_intp row=0, col=0, i=0;

    npy_intp n_row=PyArray_DIM(image_obj, 0);
    npy_intp n_col=PyArray_DIM(image_obj, 1);
    npy_intp n_points=PyArray_SIZE(image_obj);

    double scale=jacob->sdet;
    double area = n_points*scale*scale;

    double nsky = sky/counts;
    double psky = sky/(counts/area);

    double T=0, T_last=-9999.0, igrat=0;

    (*numiter)=0;
    while ( (*numiter) < maxiter) {
        skysum=0.0;
        em_clear_sums(sums, n_gauss);

        for (row=0; row<n_row; row++) {

            double u=PYGMIX_JACOB_GETU(jacob, row, 0);
            double v=PYGMIX_JACOB_GETV(jacob, row, 0);

            for (col=0; col<n_col; col++) {

                double gtot=0.0;
                double imnorm=*(double*)PyArray_GETPTR2(image_obj,row,col);

                imnorm /= counts;

                for (i=0; i<n_gauss; i++) {
                    struct PyGMix_EM_Sums *sum=&sums[i];
                    const struct PyGMix_Gauss2D *gauss=&gmix[i];

                    double udiff = u-gauss->row;
                    double vdiff = v-gauss->col;

                    double u2 = udiff*udiff;
                    double v2 = vdiff*vdiff;
                    double uv = udiff*vdiff;

                    double chi2=
                        gauss->dcc*u2 + gauss->drr*v2 - 2.0*gauss->drc*uv;

                    if (chi2 < PYGMIX_MAX_CHI2 && chi2 >= 0.0) {
                        sum->gi = gauss->pnorm*expd( -0.5*chi2 );
                    } else {
                        sum->gi = 0.0;
                    }
                    gtot += sum->gi;
                    sum->trowsum = u*sum->gi;
                    sum->tcolsum = v*sum->gi;
                    sum->tu2sum  = u2*sum->gi;
                    sum->tuvsum  = uv*sum->gi;
                    sum->tv2sum  = v2*sum->gi;

                } // gaussians

                gtot += nsky;

                if (gtot == 0) {
                    PyErr_Format(GMixRangeError, "em gtot = 0");
                    goto _em_run_bail;
                }

                igrat = imnorm/gtot;
                for (i=0; i<n_gauss; i++) {
                    struct PyGMix_EM_Sums *sum=&sums[i];

                    // wtau is gi[pix]/gtot[pix]*imnorm[pix]
                    // which is Dave's tau*imnorm = wtau
                    double wtau = sum->gi*igrat;

                    sum->pnew += wtau;

                    // row*gi/gtot*imnorm;
                    sum->rowsum += sum->trowsum*igrat;
                    sum->colsum += sum->tcolsum*igrat;
                    sum->u2sum  += sum->tu2sum*igrat;
                    sum->uvsum  += sum->tuvsum*igrat;
                    sum->v2sum  += sum->tv2sum*igrat;

                }

                skysum += nsky*imnorm/gtot;
                u += jacob->dudcol;
                v += jacob->dvdcol;
            } //cols
        } // rows


        status=em_set_gmix_from_sums(gmix, n_gauss, sums);
        if (!status) {
            goto _em_run_bail;
            break;
        }

        psky = skysum;
        nsky = psky/area;

        T = gmix_get_T(gmix, n_gauss);
        (*frac_diff) = fabs((T-T_last)/T);

        if ( (*frac_diff) < tol) {
            break;
        }

        T_last = T;

        (*numiter) += 1;

    } // iteration

    status=1;
_em_run_bail:
    return status;
}

static PyObject * PyGMix_em_run(PyObject* self, PyObject* args) {

    PyObject* gmix_obj=NULL;
    PyObject* image_obj=NULL;
    //PyObject* weight_obj=NULL;
    PyObject* jacob_obj=NULL;
    PyObject* sums_obj=NULL;
    double sky=0, counts=0, tol=0;
    long maxiter=0;
    npy_intp n_gauss=0;

    struct PyGMix_Gauss2D *gmix=NULL;//, *gauss=NULL;
    struct PyGMix_Jacobian *jacob=NULL;
    struct PyGMix_EM_Sums* sums=NULL;
    long numiter=0;
    double frac_diff=0;
    int status=0;


    // weight object is currently ignored
    if (!PyArg_ParseTuple(args, (char*)"OOOOdddl", 
                          &gmix_obj,
                          &image_obj,
                          //&weight_obj,
                          &jacob_obj,
                          &sums_obj,
                          &sky, &counts, &tol, &maxiter)) {
        return NULL;
    }

    gmix=(struct PyGMix_Gauss2D* ) PyArray_DATA(gmix_obj);
    n_gauss=PyArray_SIZE(gmix_obj);

    if (!gmix_set_norms_if_needed(gmix, n_gauss)) {
        return NULL;
    }

    jacob=(struct PyGMix_Jacobian* ) PyArray_DATA(jacob_obj);
    sums=(struct PyGMix_EM_Sums* )  PyArray_DATA(sums_obj);

    status=em_run(image_obj,
                  sky,
                  counts,
                  jacob,
                  gmix,
                  n_gauss,
                  sums,
                  tol,
                  maxiter,
                  &numiter,
                  &frac_diff);

    if (!status) {
        // raise an exception
        return NULL;
    } else {
        PyObject* retval=PyTuple_New(2);
        PyTuple_SetItem(retval,0,PyLong_FromLong(numiter));
        PyTuple_SetItem(retval,1,PyFloat_FromDouble(frac_diff));
        return retval;
    }
}

/*
   convert log pars to linear pars
   pars 4: are converted
*/
static 
PyObject * PyGMix_convert_simple_double_logpars(PyObject* self, PyObject* args) {

    PyObject* logpars_obj=NULL;
    PyObject* pars_obj=NULL;
    int i, npars;
    double *logpars=NULL, *pars=NULL;

    // weight object is currently ignored
    if (!PyArg_ParseTuple(args, (char*)"OO", 
                          &logpars_obj,
                          &pars_obj)) {
        return NULL;
    }

    logpars=PyArray_DATA(logpars_obj);
    pars=PyArray_DATA(pars_obj);

    npars=PyArray_SIZE(logpars_obj);

    for (i=0; i<npars; i++) {
        if (i < 4) {
        //if (i != 4) {
            pars[i] = logpars[i];
        } else {
            pars[i] = pow(10.0, logpars[i]);
        }
    }
    Py_RETURN_NONE;
}

/*
   convert log pars to linear pars, pulling out a specific band

   no error checking done here
*/
static 
PyObject * PyGMix_convert_simple_double_logpars_band(PyObject* self, PyObject* args) {

    PyObject* logpars_obj=NULL;
    PyObject* pars_obj=NULL;
    int band=0;
    double *logpars=NULL, *pars=NULL;

    // weight object is currently ignored
    if (!PyArg_ParseTuple(args, (char*)"OOi", 
                          &logpars_obj,
                          &pars_obj,
                          &band)) {
        return NULL;
    }

    logpars=PyArray_DATA(logpars_obj);
    pars=PyArray_DATA(pars_obj);

    pars[0] = logpars[0];
    pars[1] = logpars[1];
    pars[2] = logpars[2];
    pars[3] = logpars[3];
    pars[4] = exp( logpars[4] );
    pars[5] = exp( logpars[5+band] );
    //pars[5] = logpars[5+band];

    Py_RETURN_NONE;
}


/*
   convert eta pars to g1,g2 pars

   no error checking done here
*/
static 
PyObject * PyGMix_convert_simple_eta2g_band(PyObject* self, PyObject* args) {

    PyObject* etapars_obj=NULL;
    PyObject* gpars_obj=NULL;
    int band=0;
    double *etapars=NULL, *gpars=NULL;
    double g1=0, g2=0;
    int status=0;

    // weight object is currently ignored
    if (!PyArg_ParseTuple(args, (char*)"OOi", 
                          &etapars_obj,
                          &gpars_obj,
                          &band)) {
        return NULL;
    }

    etapars=PyArray_DATA(etapars_obj);
    gpars=PyArray_DATA(gpars_obj);

    status=eta1eta2_to_g1g2(etapars[2], etapars[3], &g1, &g2);


    gpars[0] = etapars[0];
    gpars[1] = etapars[1];
    gpars[2] = g1;
    gpars[3] = g2;
    gpars[4] = etapars[4];
    gpars[5] = etapars[5+band];

    return Py_BuildValue("i", status);
}



/*

   full-covariance, nd-gaussian evaluations

   can do either log or linear
       log_pnorms, means, icovars, tmp_lnprob, x, dolog

   make sure they are arrays from python
*/

static int gmixnd_get_prob_args_check(PyObject* log_pnorms,
                                      PyObject* means,
                                      PyObject* icovars,
                                      PyObject* tmp_lnprob,
                                      PyObject* pars,
                                      npy_intp *n_gauss,
                                      int *n_dim)
{
    int status=0, n_dim_means=0, n_dim_icovars=0;
    npy_intp n_pars=0, n_tmp=0;

    n_dim_means=PyArray_NDIM(means);
    if (n_dim_means != 2) {
        PyErr_Format(GMixFatalError, "means dim must be 2, got %d", n_dim_means);
        goto _gmixnd_get_prob_args_check_bail;
    }
    n_dim_icovars=PyArray_NDIM(icovars);
    if (n_dim_icovars != 3) {
        PyErr_Format(GMixFatalError, "icovars dim must be 3, got %d", n_dim_icovars);
        goto _gmixnd_get_prob_args_check_bail;
    }

    (*n_gauss)=PyArray_SIZE(log_pnorms);
    (*n_dim)=PyArray_DIM(means,1);

    if (*n_dim > 10) {
        PyErr_Format(GMixFatalError, "dim must be <= 10, got %d", *n_dim);
        goto _gmixnd_get_prob_args_check_bail;
    }

    n_pars=PyArray_SIZE(pars);
    if (n_pars != (*n_dim)) {
        PyErr_Format(GMixFatalError, "n_dim is %d but n_pars is %ld",
                     (*n_dim), n_pars);
        goto _gmixnd_get_prob_args_check_bail;
    }
    n_tmp=PyArray_SIZE(tmp_lnprob);
    if (n_tmp != (*n_gauss)) {
        PyErr_Format(GMixFatalError, "n_gauss is %ld but n_tmp_lnprob is %ld",
                     (*n_gauss), n_tmp);
        goto _gmixnd_get_prob_args_check_bail;
    }

    status=1;
_gmixnd_get_prob_args_check_bail:
    return status;
}

static 
PyObject * PyGMix_gmixnd_get_prob_scalar(PyObject* self, PyObject* args) {

    PyObject* log_pnorms=NULL;
    PyObject* means=NULL;
    PyObject* icovars=NULL;
    PyObject* tmp_lnprob=NULL;
    PyObject* pars=NULL;
    double* tmp_lnprob_ptr=NULL;

    // up to 10 dims allowed
    double xdiff[10];
    int dolog=0;
    npy_intp i=0, n_gauss=0;
    double p=0.0, retval=0;
    double lnpmax=-9.99e9, logpnorm=0, chi2=0, lnp=0, par=0, mean=0, icov=0;
    int n_dim=0, idim1=0, idim2=0;

    // weight object is currently ignored
    if (!PyArg_ParseTuple(args, (char*)"OOOOOi", 
                          &log_pnorms,
                          &means,
                          &icovars,
                          &tmp_lnprob,
                          &pars,
                          &dolog)) {
        return NULL;
    }


    if (!gmixnd_get_prob_args_check(log_pnorms,
                                    means,
                                    icovars,
                                    tmp_lnprob,
                                    pars,
                                    &n_gauss,
                                    &n_dim)) {
        return NULL;
    }

    tmp_lnprob_ptr = (double *) PyArray_DATA(tmp_lnprob);

    for (i=0; i<n_gauss; i++) {

        logpnorm = *(double *) PyArray_GETPTR1(log_pnorms, i);

        for (idim1=0; idim1<n_dim; idim1++) {
            par=*(double *) PyArray_GETPTR1(pars, idim1);
            mean=*(double *) PyArray_GETPTR2(means, i, idim1);

            xdiff[idim1] = par-mean;
        }

        chi2=0;
        for (idim1=0; idim1<n_dim; idim1++) {
            for (idim2=0; idim2<n_dim; idim2++) {
                icov=*(double *) PyArray_GETPTR3(icovars, i, idim1, idim2);

                chi2 += xdiff[idim1]*xdiff[idim2]*icov;
            }
        }

        lnp = -0.5*chi2 + logpnorm;
        if (lnp > lnpmax) {
            lnpmax=lnp;
        }
        tmp_lnprob_ptr[i] = lnp;
    }    

    p=0;
    for (i=0; i<n_gauss; i++) {
        p += exp(tmp_lnprob_ptr[i] - lnpmax);
    }

    if (dolog) {
        retval = log(p) + lnpmax;
    } else {
        retval = p*exp(lnpmax);
    }

    return PyFloat_FromDouble(retval);

}




static PyObject * PyGMix_test(PyObject* self, PyObject* args) {
    PyErr_Format(GMixRangeError, "testing GMixRangeError");
    return NULL;
}

static PyObject* PyGMix_erf(PyObject* self, PyObject* args)
{
    double val=0, out=0;
    long double lval=0;
    if (!PyArg_ParseTuple(args, (char*)"d", &val)) {
        return NULL;
    }

    lval=(long double) val;

    out=(double)erfl(lval);

    return Py_BuildValue("d", out);

}
static PyObject* PyGMix_erf_array(PyObject* self, PyObject* args)
{
    PyObject *arr=NULL, *out=NULL;
    double *pin=NULL, *pout=NULL, tmp=0;
    npy_intp num=0, i=0;
    long double lval=0;
    if (!PyArg_ParseTuple(args, (char*)"OO", &arr, &out)) {
        return NULL;
    }

    num=PyArray_SIZE(arr);
    for (i=0; i<num; i++) {
        pin = PyArray_GETPTR1( arr, i );
        pout = PyArray_GETPTR1( out, i );

        lval=(long double) (*pin);
        tmp=(double) erfl(lval);

        *pout = tmp;
    }

    return Py_BuildValue("");

}


static PyMethodDef pygauss2d_funcs[] = {

    {"get_image_mean", (PyCFunction)PyGMix_get_image_mean,  METH_VARARGS,  "calculate mean with weight\n"},

    {"get_model_s2n_sum", (PyCFunction)PyGMix_get_model_s2n_sum,  METH_VARARGS,  "calculate the s/n of the model\n"},
    {"get_model_s2n_Tvar_sums", (PyCFunction)PyGMix_get_model_s2n_Tvar_sums,  METH_VARARGS,  "calculate the s/n of the model\n"},
    {"get_model_s2n_Tvar_sums_altweight", (PyCFunction)PyGMix_get_model_s2n_Tvar_sums_altweight,  METH_VARARGS,  "calculate the s/n of the model\n"},

    {"get_weighted_mom_sums", (PyCFunction)PyGMix_get_weighted_mom_sums,  METH_VARARGS,  "calculate weighted mom sums\n"},

    {"get_loglike", (PyCFunction)PyGMix_get_loglike,  METH_VARARGS,  "calculate likelihood\n"},
    {"get_loglike_images_margsky", (PyCFunction)PyGMix_get_loglike_images_margsky,  METH_VARARGS,  "calculate likelihood between images, subtracting mean\n"},
    {"get_loglike_aper", (PyCFunction)PyGMix_get_loglike_aper,  METH_VARARGS,  "calculate likelihood within the specified circular aperture\n"},


    {"get_loglike_sub", (PyCFunction)PyGMix_get_loglike_sub,  METH_VARARGS,  "calculate likelihood\n"},
    {"get_loglike_robust", (PyCFunction)PyGMix_get_loglike_robust,  METH_VARARGS,  "calculate likelihood with robust metric\n"},

    {"fill_fdiff",  (PyCFunction)PyGMix_fill_fdiff,  METH_VARARGS,  "fill fdiff for LM\n"},
    {"fill_fdiff_sub",  (PyCFunction)PyGMix_fill_fdiff_sub,  METH_VARARGS,  "fill fdiff for LM with sub-pixel integration\n"},
    {"render",      (PyCFunction)PyGMix_render, METH_VARARGS,  "render without jacobian\n"},
    {"render_jacob",(PyCFunction)PyGMix_render_jacob, METH_VARARGS,  "render with jacobian\n"},

    {"gmix_fill",(PyCFunction)PyGMix_gmix_fill, METH_VARARGS,  "Fill the input gmix with the input pars\n"},
    {"gmix_fill_cm",(PyCFunction)PyGMix_gmix_fill_cm, METH_VARARGS,  "Fill the input gmix with the input pars\n"},

    {"convolve_fill",(PyCFunction)PyGMix_convolve_fill, METH_VARARGS,  "convolve gaussian with psf and store in output\n"},
    {"set_norms",(PyCFunction)PyGMix_gmix_set_norms, METH_VARARGS,  "set the normalizations used during evaluation of the gaussians\n"},

    {"em_run",(PyCFunction)PyGMix_em_run, METH_VARARGS,  "run the em algorithm\n"},

    {"get_cm_Tfactor",        (PyCFunction)PyGMix_get_cm_Tfactor,         METH_VARARGS,  "get T factor for composite model\n"},

    {"convert_simple_double_logpars",        (PyCFunction)PyGMix_convert_simple_double_logpars,         METH_VARARGS,  "convert log10 to linear.\n"},
    {"convert_simple_double_logpars_band",        (PyCFunction)PyGMix_convert_simple_double_logpars_band,         METH_VARARGS,  "convert log10 to linear with band specified.\n"},

    {"convert_simple_eta2g_band",        (PyCFunction)PyGMix_convert_simple_eta2g_band,         METH_VARARGS,  "convert eta to g, band specified.\n"},

    {"gmixnd_get_prob_scalar",        (PyCFunction)PyGMix_gmixnd_get_prob_scalar,         METH_VARARGS,  "get prob or log prob for scalar arg, nd gaussian"},

    {"test",        (PyCFunction)PyGMix_test,         METH_VARARGS,  "test\n\nprint and return."},
    {"erf",         (PyCFunction)PyGMix_erf,         METH_VARARGS,  "erf with better precision."},
    {"erf_array",         (PyCFunction)PyGMix_erf_array,         METH_VARARGS,  "erf with better precision."},
    {NULL}  /* Sentinel */
};



/*
 *
 * prior classes
 *
 */

/* class representing a 1-d normal distribution */

struct PyGMixNormal {
    PyObject_HEAD
    double cen;
    double sigma;
    double s2inv;
};

static int
PyGMixNormal_init(struct PyGMixNormal* self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, (char*)"dd", &self->cen, &self->sigma)) {
        return -1;
    }

    self->s2inv = 1.0/(self->sigma*self->sigma);

    return 0;
}

static PyObject* PyGMixNormal_get_lnprob_scalar(struct PyGMixNormal* self,
                                                 PyObject *args)
{
    double x=0, diff=0, lnp=0;
    if (!PyArg_ParseTuple(args, (char*)"d", &x)) {
        return NULL;
    }

    diff = self->cen-x;
    lnp = -0.5*diff*diff*self->s2inv;

    return PyFloat_FromDouble(lnp);

}
static PyObject* PyGMixNormal_get_prob_scalar(struct PyGMixNormal* self,
                                               PyObject *args)
{
    double x=0, diff=0, lnp=0, p=0;
    if (!PyArg_ParseTuple(args, (char*)"d", &x)) {
        return NULL;
    }

    diff = self->cen-x;
    lnp = -0.5*diff*diff*self->s2inv;
    p = exp(lnp);

    return PyFloat_FromDouble(p);

}


static PyMethodDef PyGMixNormal_methods[] = {
    {"get_lnprob_scalar", (PyCFunction)PyGMixNormal_get_lnprob_scalar, METH_VARARGS, "nget ln(prob) for the input x value."},
    {"get_prob_scalar", (PyCFunction)PyGMixNormal_get_prob_scalar, METH_VARARGS, "get prob for the input x value."},
    {NULL}  /* Sentinel */
};


static PyTypeObject PyGMixNormalType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "_gmix.Normal",            /*tp_name*/
    sizeof(struct PyGMixNormal), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    0,                         /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "PyGMix Normal distribution",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    PyGMixNormal_methods,             /* tp_methods */
    0,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    //0,     /* tp_init */
    (initproc)PyGMixNormal_init,      /* tp_init */
    0,                         /* tp_alloc */
    //PyGMixObject_new,                 /* tp_new */
    PyType_GenericNew,                 /* tp_new */
};


/* class representing a 2-d circular normal distribution */

struct PyGMixNormal2D {
    PyObject_HEAD
    double cen1;
    double cen2;
    double sigma1;
    double sigma2;
    double s2inv1;
    double s2inv2;
};

static int
PyGMixNormal2D_init(struct PyGMixNormal2D* self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, (char*)"dddd", 
                          &self->cen1, &self->cen2,
                          &self->sigma1, &self->sigma2)) {
        return -1;
    }

    self->s2inv1 = 1.0/(self->sigma1*self->sigma1);
    self->s2inv2 = 1.0/(self->sigma2*self->sigma2);

    return 0;
}

static PyObject* PyGMixNormal2D_get_lnprob_scalar(struct PyGMixNormal2D* self,
                                                  PyObject *args)
{
    double x1=0, x2=0, d1=0, d2=0, lnp=0;
    if (!PyArg_ParseTuple(args, (char*)"dd", &x1, &x2)) {
        return NULL;
    }

    d1 = self->cen1-x1;
    d2 = self->cen2-x2;
    lnp = -0.5*d1*d1*self->s2inv1 - 0.5*d2*d2*self->s2inv2;

    return PyFloat_FromDouble(lnp);

}
static PyObject* PyGMixNormal2D_get_prob_scalar(struct PyGMixNormal2D* self,
                                               PyObject *args)
{
    double x1=0, x2=0, d1=0, d2=0, lnp=0, p=0;
    if (!PyArg_ParseTuple(args, (char*)"dd", &x1, &x2)) {
        return NULL;
    }

    d1 = self->cen1-x1;
    d2 = self->cen2-x2;
    lnp = -0.5*d1*d1*self->s2inv1 - 0.5*d2*d2*self->s2inv2;

    p=exp(lnp);
    return PyFloat_FromDouble(p);
}

static PyObject* PyGMixNormal2D_get_lnprob_scalar_sep(struct PyGMixNormal2D* self,
                                                      PyObject *args)
{
    double x1=0, x2=0, d1=0, d2=0, lnp1=0, lnp2=0;
    PyObject* retval=NULL;

    if (!PyArg_ParseTuple(args, (char*)"dd", &x1, &x2)) {
        return NULL;
    }

    d1 = self->cen1-x1;
    d2 = self->cen2-x2;
    lnp1 = -0.5*d1*d1*self->s2inv1;
    lnp2 = -0.5*d2*d2*self->s2inv2;

    retval=PyTuple_New(2);
    PyTuple_SetItem(retval,0,PyFloat_FromDouble(lnp1));
    PyTuple_SetItem(retval,1,PyFloat_FromDouble(lnp2));

    return retval;
}


static PyMethodDef PyGMixNormal2D_methods[] = {
    {"get_lnprob_scalar", (PyCFunction)PyGMixNormal2D_get_lnprob_scalar, METH_VARARGS, "nget ln(prob) for the input location."},
    {"get_prob_scalar", (PyCFunction)PyGMixNormal2D_get_prob_scalar, METH_VARARGS, "get prob for the input location."},
    {"get_lnprob_scalar_sep", (PyCFunction)PyGMixNormal2D_get_lnprob_scalar_sep, METH_VARARGS, "get prob for the input location, separately for each dimension."},
    {NULL}  /* Sentinel */
};


static PyTypeObject PyGMixNormal2DType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "_gmix.Normal2D",            /*tp_name*/
    sizeof(struct PyGMixNormal2D), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    0,                         /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "PyGMix Normal2D distribution",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    PyGMixNormal2D_methods,             /* tp_methods */
    0,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    //0,     /* tp_init */
    (initproc)PyGMixNormal2D_init,      /* tp_init */
    0,                         /* tp_alloc */
    //PyGMixObject_new,                 /* tp_new */
    PyType_GenericNew,                 /* tp_new */
};


/* class representing a 2-d disk with max radius */

struct PyGMixZDisk2D {
    PyObject_HEAD
    double radius;
    double radius_sq;
};

static int
PyGMixZDisk2D_init(struct PyGMixZDisk2D* self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, (char*)"d", 
                          &self->radius)) {
        return -1;
    }

    self->radius_sq = self->radius*self->radius;
    return 0;
}

static PyObject* PyGMixZDisk2D_get_lnprob_scalar1d(struct PyGMixZDisk2D* self,
                                                   PyObject *args)
{
    double r=0;
    if (!PyArg_ParseTuple(args, (char*)"d", &r)) {
        return NULL;
    }

    if (r >= self->radius) {
        PyErr_Format(GMixRangeError, "position out of bounds");
        return NULL;
    } else {
        return PyFloat_FromDouble(0.0);
    }

}
static PyObject* PyGMixZDisk2D_get_prob_scalar1d(struct PyGMixZDisk2D* self,
                                                 PyObject *args)
{
    double r=0, retval=0;
    if (!PyArg_ParseTuple(args, (char*)"d", &r)) {
        return NULL;
    }

    if (r >= self->radius) {
        retval=0.0; 
    } else {
        retval=1.0;
    }
    return PyFloat_FromDouble(retval);
}


static PyObject* PyGMixZDisk2D_get_lnprob_scalar2d(struct PyGMixZDisk2D* self,
                                                   PyObject *args)
{
    double x=0, y=0, r2=0;
    if (!PyArg_ParseTuple(args, (char*)"dd", &x, &y)) {
        return NULL;
    }

    r2 = x*x + y*y;
    if (r2 >= self->radius_sq) {
        PyErr_Format(GMixRangeError, "position out of bounds");
        return NULL;
    } else {
        return PyFloat_FromDouble(0.0);
    }

}

static PyObject* PyGMixZDisk2D_get_prob_scalar2d(struct PyGMixZDisk2D* self,
                                                 PyObject *args)
{
    double x=0, y=0, r2=0, retval=0;
    if (!PyArg_ParseTuple(args, (char*)"dd", &x, &y)) {
        return NULL;
    }

    r2 = x*x + y*y;

    if (r2 >= self->radius_sq) {
        retval=0.0; 
    } else {
        retval=1.0;
    }
    return PyFloat_FromDouble(retval);
}

static PyObject* PyGMixZDisk2D_get_prob_array2d(struct PyGMixZDisk2D* self,
                                                PyObject *args)
{
    PyObject *xobj=NULL, *yobj=NULL, *probobj=NULL;
    npy_intp nx=0, i=0;
    double x=0, y=0, r2=0, *probptr=0;
    if (!PyArg_ParseTuple(args, (char*)"OOO", &xobj, &yobj, &probobj)) {
        return NULL;
    }

    nx=PyArray_SIZE(xobj);

    for (i=0; i<nx; i++) {
        x= *( (double*) PyArray_GETPTR1(xobj, i) );
        y= *( (double*) PyArray_GETPTR1(yobj, i) );
        probptr = (double*) PyArray_GETPTR1(probobj, i);

        r2 = x*x + y*y;

        if (r2 >= self->radius_sq) {
            *probptr=0.0; 
        } else {
            *probptr=1.0; 
        }
    }
    Py_RETURN_NONE;
}


static PyMethodDef PyGMixZDisk2D_methods[] = {
    {"get_lnprob_scalar1d", (PyCFunction)PyGMixZDisk2D_get_lnprob_scalar1d, METH_VARARGS, "0 inside disk, throw exception outside"},
    {"get_prob_scalar1d", (PyCFunction)PyGMixZDisk2D_get_prob_scalar1d, METH_VARARGS, "1 inside disk, 0 outside"},

    {"get_lnprob_scalar2d", (PyCFunction)PyGMixZDisk2D_get_lnprob_scalar2d, METH_VARARGS, "0 inside disk, throw exception outside"},
    {"get_prob_scalar2d", (PyCFunction)PyGMixZDisk2D_get_prob_scalar2d, METH_VARARGS, "1 inside disk, 0 outside"},
    {"get_prob_array2d", (PyCFunction)PyGMixZDisk2D_get_prob_array2d, METH_VARARGS, "1 inside disk, 0 outside"},
    {NULL}  /* Sentinel */
};


static PyTypeObject PyGMixZDisk2DType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "_gmix.ZDisk2D",            /*tp_name*/
    sizeof(struct PyGMixZDisk2D), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    0,                         /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "PyGMix ZDisk2D distribution",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    PyGMixZDisk2D_methods,             /* tp_methods */
    0,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    //0,     /* tp_init */
    (initproc)PyGMixZDisk2D_init,      /* tp_init */
    0,                         /* tp_alloc */
    //PyGMixObject_new,                 /* tp_new */
    PyType_GenericNew,                 /* tp_new */
};






#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_gmix",      /* m_name */
        "Defines the funcs associated with gmix",  /* m_doc */
        -1,                  /* m_size */
        pygauss2d_funcs,      /* m_methods */
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
#if PY_MAJOR_VERSION >= 3
PyInit__gmix(void)
#else
init_gmix(void) 
#endif
{
    PyObject* m=NULL;

    PyGMixNormalType.tp_new = PyType_GenericNew;
    PyGMixNormal2DType.tp_new = PyType_GenericNew;
    PyGMixZDisk2DType.tp_new = PyType_GenericNew;

#if PY_MAJOR_VERSION >= 3
    if (PyType_Ready(&PyGMixNormalType) < 0) {
        return NULL;
    }
    if (PyType_Ready(&PyGMixNormal2DType) < 0) {
        return NULL;
    }
    if (PyType_Ready(&PyGMixZDisk2DType) < 0) {
        return NULL;
    }


    m = PyModule_Create(&moduledef);
    if (m==NULL) {
        return NULL;
    }

#else
    if (PyType_Ready(&PyGMixNormalType) < 0) {
        return;
    }
    if (PyType_Ready(&PyGMixNormal2DType) < 0) {
        return;
    }
    if (PyType_Ready(&PyGMixZDisk2DType) < 0) {
        return;
    }



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
#if PY_MAJOR_VERSION >= 3
            return NULL;
#else
            return;
#endif
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
#if PY_MAJOR_VERSION >= 3
            return NULL;
#else
            return;
#endif
        }
    }


    Py_INCREF(&PyGMixNormalType);
    PyModule_AddObject(m, "Normal", (PyObject *)&PyGMixNormalType);
    Py_INCREF(&PyGMixNormal2DType);
    PyModule_AddObject(m, "Normal2D", (PyObject *)&PyGMixNormal2DType);
    Py_INCREF(&PyGMixZDisk2DType);
    PyModule_AddObject(m, "ZDisk2D", (PyObject *)&PyGMixZDisk2DType);



    // for numpy
    import_array();

#if PY_MAJOR_VERSION >= 3
    return m;
#endif

}
