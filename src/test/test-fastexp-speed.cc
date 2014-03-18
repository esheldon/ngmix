#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include "fastexp.h"

using namespace std;

typedef double ftype;

int main(int argc, char **argv)
{

    ftype tot=0;
    long n=100000;
    long nrepeat=10000;

    vector<ftype> d(n);
    
    for (long i=0; i<n; i++) {
        d[i] = drand48();
    }

    time_t t1,t2;
    
    t1=clock();
    for (long irep=0; irep<nrepeat; irep++) {
        tot=0;
        for (long i=0; i<n; i++) {
            tot += exp(d[i]);
        }
    }
    t2=clock();
    double tstd = (t2-t1)/( (double)CLOCKS_PER_SEC );
    printf("time for std:  %.16g s\n", tstd);
    printf("total sum: %.16g\n", tot);



    t1=clock();
    for (long irep=0; irep<nrepeat; irep++) {
        tot=0;
        for (long i=0; i<n; i++) {
            tot += fastexp::expd(d[i]);
        }
    }
    t2=clock();
    double tfast = (t2-t1)/( (double)CLOCKS_PER_SEC );
    printf("time for fast:  %.16g s\n", tfast);
    printf("total sum: %.16g\n", tot);


    printf("fastexp is faster by %.16g\n", tstd/tfast);


    return 0;

}
