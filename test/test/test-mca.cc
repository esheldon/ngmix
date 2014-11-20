#include <cstdio>
#include "mca.h"

using namespace std;

int main(int argc, char **argv)
{

    mca::MCA fitter;

    size_t nwalkers=80;
    size_t nsteps_per_walker=800;
    size_t npars=6;

    fitter.reset(nwalkers, nsteps_per_walker, npars);

    printf("nwalkers:          %lu\n", fitter.get_nwalkers());
    printf("nsteps_per_walker: %lu\n", fitter.get_nsteps_per_walker());
    printf("nsteps_total:      %lu\n", fitter.get_nsteps_total());
    printf("npars:             %lu\n", fitter.get_npars());

    size_t walker=35;
    size_t walker_step=80;
    size_t parnum=2;

    printf("par[%lu] for walker %lu step %lu: %g\n",
           parnum, walker, walker_step,
           fitter.get_par(walker, walker_step, parnum));

    return 0;
}
