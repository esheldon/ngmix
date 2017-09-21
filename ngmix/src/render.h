#ifndef _PYGMIX_RENDER_HEADER_GUARD
#define _PYGMIX_RENDER_HEADER_GUARD

#include "gmix.h"
#include "pixels.h"
#include "jacobian.h"

void render(const struct gauss *gmix,
            long n_gauss,
            double *image,
            const struct coord *coords,
            long n_pixels,
            const struct jacobian* jacob,
            int fast_exp);

#endif
