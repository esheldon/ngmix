#include <cstdio>
#include <cmath>
#include <vector>
#include "gmix.h"
#include "image.h"
#include "mtrng.h"
#include "jacobian.h"

using namespace std;

using NGMix::GMix;
using NGMix::Gauss;
using image::Image;
using jacobian::Jacobian;


int main(int argc, char **argv)
{

    Gauss gempty;
    gempty.print();

    Gauss ginit(1.0, 25.0, 26.0, 4.0, 1.6, 3.7);
    ginit.print();

    Gauss gset;
    gset.set(200.0, 15.2, 14.75, 1.8, 0.25, 1.9);
    gset.print();

    vector<double> pars;

    pars.push_back(0.4);
    pars.push_back(16.0);
    pars.push_back(15.0);
    pars.push_back(8.0);
    pars.push_back(1.5);
    pars.push_back(5.0);

    pars.push_back(0.6);
    pars.push_back(8.0);
    pars.push_back(15.0);
    pars.push_back(4.0);
    pars.push_back(2.2);
    pars.push_back(7.0);

    GMix gmix(pars);

    printf("\n");
    gmix.print();


    double row=16.0, col=15.0;
    printf("eval(%g,%g) = %g\n", row, col, gmix(row,col));

    pars[1] = 18.0;
    pars[2] = 18.1;
    gmix.set_from_pars(pars);

    gmix.print();

    printf("eval(%g,%g) = %g\n", row, col, gmix(row,col));

    long nrows=32, ncols=32;
    Image im(nrows, ncols);

    gmix.render(im);
    im.show("/tmp/timage-32432.dat");

    mtrng::MtRng64 rng;
    rng.init_dev_urandom();

    double sigma=0.001;
    im.add_gaussian_noise(rng, sigma);
    im.show("/tmp/timage-32432.dat");

    double loglike=0, s2n_numer=0, s2n_denom=0;
    Image weight(nrows,ncols);
    Jacobian jacob;
    weight.add_scalar( 1.0/(sigma*sigma) );

    gmix.get_loglike(im, weight, jacob,
                     &loglike, &s2n_numer, &s2n_denom);

    double s2n = s2n_numer/sqrt(s2n_denom);
    double dof = im.get_size();
    printf("s2n:       %g\n", s2n);
    printf("loglike:   %g\n", loglike);
    printf("chi^2/dof: %g\n", loglike/(-0.5)/dof);
    return 0;
}
