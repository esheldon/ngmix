// convert reduced shear g1,g2 to standard ellipticity
// parameters e1,e2

// return 0 means out of range

#include "../_gmix.h"

extern PyObject* GMixRangeError;
extern PyObject* GMixFatalError;


int g1g2_to_e1e2(double g1, double g2, double *e1, double *e2) {
    double eta=0, e=0, fac=0;
    double g=sqrt(g1*g1 + g2*g2);

    if (g >= 1) {
        char gstr[25];
        snprintf(gstr,24,"%g", g);
        PyErr_Format(GMixRangeError, "g out of bounds: %s", gstr);
        return 0;
    }
    if (g == 0.0) {
        *e1=0;
        *e2=0;
    } else {

        eta = 2*atanh(g);
        e = tanh(eta);
        if (e >= 1.) {
            // round off?
            e = 0.99999999;
        }

        fac = e/g;

        *e1 = fac*g1;
        *e2 = fac*g2;
    }

    return 1;
}


