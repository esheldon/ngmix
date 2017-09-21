#ifndef _PYGMIX_PIXELS_HEADER_GUARD
#define _PYGMIX_PIXELS_HEADER_GUARD

struct pixel {
    double u;
    double v;

    double val;

    double ierr;

    double fdiff;
};

struct coord {
    double u;
    double v;
};


#endif
