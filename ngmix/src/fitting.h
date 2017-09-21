#ifndef _PYGMIX_FITTING_HEADER_GUARD
#define _PYGMIX_FITTING_HEADER_GUARD

#include "../_gmix.h"

double get_loglike_pixels(struct gauss *gmix,
                          npy_intp n_gauss,
                          const struct pixel *pixels,
                          npy_intp n_pixels,
                          int *status);

#endif
