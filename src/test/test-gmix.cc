#include <cstdio>
#include <vector>
#include "gmix.h"

using namespace std;

using gmix::GMix;
using gmix::Gauss;
using std::vector;

int main(int argc, char **argv)
{

    Gauss gempty;
    gempty.show();

    Gauss ginit(1.0, 25.0, 26.0, 4.0, 1.6, 3.7);
    ginit.show();

    Gauss gset;
    gset.set(200.0, 15.2, 14.75, 1.8, 0.25, 1.9);
    gset.show();

    vector<double> pars;

    pars.push_back(0.4);
    pars.push_back(16.0);
    pars.push_back(15.0);
    pars.push_back(4.0);
    pars.push_back(0.5);
    pars.push_back(3.0);

    pars.push_back(0.6);
    pars.push_back(14.0);
    pars.push_back(13.0);
    pars.push_back(2.0);
    pars.push_back(1.2);
    pars.push_back(5.0);

    GMix obj_gmix(pars);

    std::printf("\n");
    obj_gmix.show();


    double row=16.0, col=15.0;
    std::printf("eval(%g,%g) = %g\n", row, col, obj_gmix(row,col));

    pars[1] = 18.0;
    pars[2] = 18.1;
    obj_gmix.set_from_pars(pars);

    obj_gmix.show();

    std::printf("eval(%g,%g) = %g\n", row, col, obj_gmix(row,col));


    return 0;
}
