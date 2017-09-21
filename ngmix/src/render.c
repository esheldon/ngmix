#include <omp.h>
#include <stdlib.h>
#include "render.h"
#include "fastexp.h"

//
// norms should be set on the gaussian mixture
//

void render(const struct gauss *gmix,
            long n_gauss,
            double *image,
            const struct coord *coords,
            long n_pixels,
            const struct jacobian* jacob,
            int fast_exp)
{

    long igauss=0, ipixel=0;

    const struct gauss *gauss=NULL;
    const struct coord *coord=NULL;

    double udiff=0, vdiff=0, model_val=0, chi2=0;

#pragma omp parallel for \
        default(none) \
        shared(gmix,coords,n_pixels,n_gauss,image) \
        private(ipixel,coord,igauss,gauss,vdiff,udiff,chi2,model_val)

    for (ipixel=0; ipixel < n_pixels; ipixel++) {
        coord = &coords[ipixel];

        model_val=0;
        for (igauss=0; igauss < n_gauss; igauss++) {
            gauss = &gmix[igauss];

            // v->row, u->col in gauss
            vdiff = coord->v - gauss->row;
            udiff = coord->u - gauss->col;

            chi2 =      gauss->dcc*vdiff*vdiff
                  +     gauss->drr*udiff*udiff
                  - 2.0*gauss->drc*vdiff*udiff;

            if (chi2 < PYGMIX_MAX_CHI2_FAST && chi2 >= 0.0) {
                model_val += gauss->pnorm*expd( -0.5*chi2 );
            }
        }

        image[ipixel] += model_val;

    }
}




