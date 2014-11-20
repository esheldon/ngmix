#include <cstdio>
#include <cmath>
#include <vector>
#include "gmix.h"
#include "image.h"
#include "mtrng.h"
#include "jacobian.h"

using namespace std;
using namespace NGMix;

using image::Image;
using jacobian::Jacobian;


GMix* make_model(vector<double>& pars, enum gmix_model model) {
    GMix* gmix=NULL;
    if (model==GMIX_EXP) {
        gmix=new GMixExp(pars);
    } else if (model==GMIX_GAUSS) {
        gmix=new GMixGauss(pars);
    }
    return gmix;
}

int main(int argc, char **argv)
{

    vector<double> pars;

    pars.push_back(16.0);
    pars.push_back(15.0);
    pars.push_back(0.2);
    pars.push_back(0.1);
    pars.push_back(8.0);
    pars.push_back(1.0);

    // not using the maker, no pointer
    GMixExp exp_gmix(pars);

    exp_gmix.print();

    // use the maker, pointer
    GMix* exp_gmix2 = make_model(pars, GMIX_EXP);

    printf("\n");
    exp_gmix2->print();

    GMix* gauss_gmix = make_model(pars, GMIX_GAUSS);

    printf("\n");
    gauss_gmix->print();


    long nrows=32, ncols=32;
    Image im(nrows, ncols);

    exp_gmix.render(im);
    im.show("/tmp/timage-32432.dat");

    delete exp_gmix2;
    delete gauss_gmix;

    return 0;
}
