#include "gauleg.h"

static const double pygmix_gl_xxi5[5] = {
    -0.906179845938664,  -0.5384693101056831,
    0,  0.5384693101056831,  0.906179845938664
};

static const double pygmix_gl_wwi5[5] = {
    0.05613434886242515,  0.1133999999968999,
    0.1347850723875167,  0.1133999999968999,  0.05613434886242515
};

static const double pygmix_gl_xxi10[10] = {
    -0.9739065285171717,  -0.8650633666889845,
    -0.6794095682990243,  -0.4333953941292472,
    -0.1488743389816312,  0.1488743389816312,
    0.4333953941292472,  0.6794095682990243,
    0.8650633666889845,  0.9739065285171717
};

static const double pygmix_gl_wwi10[10] = {
    0.06667134430868371,  0.1494513490843985,  0.2190863625152871,
    0.2692667193099917,  0.2955242247147529,  0.2955242247147529,
    0.2692667193099917,  0.2190863625152871,  0.1494513490843985,
    0.06667134430868371
};


int set_gauleg_data(int npoints, const double **xxi, const double **wwi)
{
    int status=1;
    if (npoints==5) {
        *xxi=pygmix_gl_xxi5;
        *wwi=pygmix_gl_wwi5;
    } else if (npoints==10) {
        *xxi=pygmix_gl_xxi10;
        *wwi=pygmix_gl_wwi10;
    } else {
        PyErr_Format(PyExc_ValueError,
                     "bad npoints: %d, npoints only 5,10 for now", npoints);
        status=0;
    }
    return status;
}


