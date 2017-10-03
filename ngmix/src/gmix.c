#include <Python.h>
#include "gmix.h"
#include "shapes.h"


extern PyObject* GMixRangeErrorC;
extern PyObject* GMixFatalError;

void gmix_get_cen(const struct gauss *self,
                  long n_gauss,
                  double* row,
                  double *col,
                  double *psum)
{
    long i=0;
    *row=0;
    *col=0;
    *psum=0;

    for (i=0; i<n_gauss; i++) {
        const struct gauss *gauss=&self[i];

        double p=gauss->p;
        *row += p*gauss->row;
        *col += p*gauss->col;
        *psum += p;
    }
    (*row) /= (*psum);
    (*col) /= (*psum);
}

int gmix_get_e1e2T(struct gauss *gmix,
                   long n_gauss,
                   double *e1, double *e2, double *T)
{

    int status=1;
    double row=0, col=0, psum=0;
    double T_sum=0.0, irr_sum=0, irc_sum=0, icc_sum=0;
    double rowdiff=0, coldiff=0;
    long i=0;

    gmix_get_cen(gmix, n_gauss, &row, &col, &psum);

    if (psum == 0) {
        status=0;
        return status;
    }

    for (i=0; i<n_gauss; i++) {
        struct gauss *gauss=&gmix[i];

        psum += gauss->p;

        rowdiff=gauss->row-row;
        coldiff=gauss->col-col;

        irr_sum += gauss->p*(gauss->irr + rowdiff*rowdiff);
        irc_sum += gauss->p*(gauss->irc + rowdiff*coldiff);
        icc_sum += gauss->p*(gauss->icc + coldiff*coldiff);

    }

    T_sum = irr_sum + icc_sum;
    *T = T_sum/psum;

    *e1 = (icc_sum - irr_sum)/T_sum;
    *e2 = 2.0*irc_sum/T_sum;

    return status;
}



 
//   zero return value means bad determinant, out of range
//   
//   note for gaussians we plan to convolve with a psf we might
//   not care that det < 0, so we don't always evaluate

int gauss2d_set_norm(struct gauss *self, int dothrow)
{
    int status=0;
    double idet=0;
    if (self->det < PYGMIX_LOW_DETVAL) {

        // PyErr_Format doesn't format floats
        if (dothrow) {
            char detstr[25];
            snprintf(detstr,24,"%g", self->det);
            PyErr_Format(GMixRangeErrorC, "gauss2d det too low: %s", detstr);
        }
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

int gauss2d_set_norm_throw(struct gauss *self)
{
    return gauss2d_set_norm(self, 1);
}

int gauss2d_set(struct gauss *self,
                double p,
                double row,
                double col,
                double irr,
                double irc,
                double icc) {

    // this means norm_set=0 as well as the other pieces not
    // yet calculated
    memset(self, 0, sizeof(struct gauss));

    self->p=p;
    self->row=row;
    self->col=col;
    self->irr=irr;
    self->irc=irc;
    self->icc=icc;

    self->det = irr*icc - irc*irc;

    return 1;
}

int get_n_gauss(int model, int *status) {
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


int gmix_fill_full(struct gauss *self,
                   long n_gauss,
                   const double* pars,
                   long n_pars)
{

    int status=0;
    long i=0, beg=0;

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


int gmix_set_norms(struct gauss *self,
                   long n_gauss)
{

    int status=0;
    long i=0;

    for (i=0; i<n_gauss; i++) {

        status=gauss2d_set_norm_throw(&self[i]);

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

int gmix_set_norms_if_needed(struct gauss *self,
                             long n_gauss)
{

    int status=1;
    if (!self->norm_set) {
        status=gmix_set_norms(self, n_gauss);
    }
    return status;
}



// when an error occurs and exception is set. Use goto pattern
// for errors to simplify code.

int gmix_fill_simple(struct gauss *self,
                     long n_gauss,
                     const double* pars,
                     long n_pars,
                     int model,
                     const double* fvals,
                     const double* pvals)
{

    int status=0, n_gauss_expected=0;
    double row=0,col=0,g1=0,g2=0,
           T=0,counts=0,e1=0,e2=0,
           T_i_2=0,counts_i=0;
    long i=0;

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


int gmix_fill_cm(struct PyGMixCM*self,
                 const double* pars,
                 long n_pars)
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

    long i=0;

    if (n_pars != 6) {
        PyErr_Format(GMixFatalError, 
                     "composite pars should be size 6, got %ld", n_pars);
        goto _bail;
    }

    row=pars[0];
    col=pars[1];
    g1=pars[2];
    g2=pars[3];
    if (pars[4] == 0.0) {
      T = 0.0;
    } else {
      T=pars[4] * self->Tfactor;
    }
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


int gmix_fill_coellip(struct gauss *self,
                      long n_gauss,
                      const double* pars,
                      long n_pars)
{

    int status=0;//, n_gauss_expected=0;
    double row=0,col=0,g1=0,g2=0,
           T=0,Thalf=0,counts=0,e1=0,e2=0;
    long i=0;

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


double get_cm_Tfactor(double fracdev, double TdByTe)
{
    double
        ifracdev=0,
        Tfactor=0,
        p=0,f=0;
    long i=0;

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

    return Tfactor;
}

int gmix_fill(struct gauss *self,
              long n_gauss,
              const double* pars,
              long n_pars,
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

        default:
            PyErr_Format(GMixFatalError, 
                         "gmix error: Bad gmix model: %d", model);
            goto _gmix_fill_bail;
            break;
    }

_gmix_fill_bail:
    return status;

}


int convolve_fill(struct gauss *self, long self_n_gauss,
                  const struct gauss *gmix, long n_gauss,
                  const struct gauss *psf, long psf_n_gauss)
{
    int status=0;
    long ntot=0, iobj=0, ipsf=0, itot=0;
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
        const struct gauss *obj_gauss=&gmix[iobj];

        for (ipsf=0; ipsf<psf_n_gauss; ipsf++) {
            const struct gauss *psf_gauss=&psf[ipsf];

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



