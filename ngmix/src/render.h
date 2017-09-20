#ifndef _PYGMIX_RENDER_HEADER_GUARD
#define _PYGMIX_RENDER_HEADER_GUARD

#include "../_gmix.h"

void render(const struct gauss *gmix,
            npy_intp n_gauss,
            double *image,
            const struct coord *coords,
            npy_intp n_pixels,
            const struct jacobian* jacob,
            int fast_exp);

#endif
