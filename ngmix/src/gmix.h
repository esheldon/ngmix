#ifndef _PYGMIX_GMIX_HEADER_GUARD
#define _PYGMIX_GMIX_HEADER_GUARD

void gmix_get_cen(const struct gauss *self,
                  npy_intp n_gauss,
                  double* row,
                  double *col,
                  double *psum);

int gmix_get_e1e2T(struct gauss *gmix,
                   npy_intp n_gauss,
                   double *e1, double *e2, double *T);

int gauss2d_set_norm(struct gauss *self, int dothrow);
int gauss2d_set_norm_throw(struct gauss *self);
int gauss2d_set(struct gauss *self,
                double p,
                double row,
                double col,
                double irr,
                double irc,
                double icc);

inline int get_n_gauss(int model, int *status);

int gmix_fill_full(struct gauss *self,
                   npy_intp n_gauss,
                   const double* pars,
                   npy_intp n_pars);

int gmix_set_norms(struct gauss *self,
                   npy_intp n_gauss);


int gmix_set_norms_if_needed(struct gauss *self,
                             npy_intp n_gauss);

int gmix_fill_simple(struct gauss *self,
                     npy_intp n_gauss,
                     const double* pars,
                     npy_intp n_pars,
                     int model,
                     const double* fvals,
                     const double* pvals);

int gmix_fill_cm(struct PyGMixCM*self,
                 const double* pars,
                 npy_intp n_pars);


int gmix_fill_coellip(struct gauss *self,
                      npy_intp n_gauss,
                      const double* pars,
                      npy_intp n_pars);

double get_cm_Tfactor(double fracdev, double TdByTe);

int gmix_fill(struct gauss *self,
              npy_intp n_gauss,
              const double* pars,
              npy_intp n_pars,
              int model);

int convolve_fill(struct gauss *self, npy_intp self_n_gauss,
                  const struct gauss *gmix, npy_intp n_gauss,
                  const struct gauss *psf, npy_intp psf_n_gauss);

double get_model_s2n_sum(struct gauss *gmix,
                         npy_intp n_gauss,
                         PyObject* weight,
                         struct jacobian* jacob,
                         int *status);

#endif
