#include <cstdio>
#include <cmath>
#include "fmath.h"

using namespace std;


int main(int argc, char **argv)
{

    long n=1000;

    double dmin=-26.0;
    double dmax=0.0;

    double delta=(dmax-dmin)/n;

    double max_fracdiff=0.0;
    double x=dmin;
    for (long i=0; i<n; i++) {
        double etrue = std::exp(x);
        double eapprox = fmath::expd(x);

        double fracdiff = abs( eapprox/etrue-1);
        if (fracdiff > max_fracdiff) {
            max_fracdiff = fracdiff;
        }
        x += delta;

        std::printf("%.16g  %.16g\n", x, fracdiff);
    }

    std::printf("max fracdiff: %.16g\n", max_fracdiff);
    return 0;

}
