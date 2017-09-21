#ifndef _PYGMIX_FITTING_HEADER_GUARD
#define _PYGMIX_FITTING_HEADER_GUARD

#include "../_gmix.h"

double get_loglike(const struct gauss *gmix,
                   npy_intp n_gauss,
                   const struct pixel *pixels,
                   npy_intp n_pixels);

void get_loglike_more(const struct gauss *gmix,
                      npy_intp n_gauss,
                      const struct pixel *pixels,
                      npy_intp n_pixels,
                      double *loglike,
                      double *s2n_numer,
                      double *s2n_denom,
                      long *npix);

void fill_fdiff(struct pixel* pixels,     // array of pixels
                npy_intp n_pixels,        // number of pixels in array
                const struct gauss* gmix, // the gaussian mixture
                npy_intp n_gauss);        // number of gaussians


#endif
