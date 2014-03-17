#include <cstdio>
#include "image.h"

using namespace std;

using image::Image;

int main(int argc, char **argv)
{

    Image im(2,2);

    im.resize(5,7);

    long nrows=im.get_nrows();
    long ncols=im.get_ncols();

    long i=0;
    for (long row=0; row<nrows; row++) {
        for (long col=0; col<ncols; col++) {
            im(row,col)=i;
            i++;
        }
    }

    for (long row=0; row<nrows; row++) {
        for (long col=0; col<ncols; col++) {
            printf("%5g ", im(row,col));
        }
        printf("\n");
    }

    // this is OK if you know what you are doing
    for (long row=0; row<nrows; row++) {
        for (long col=0; col<ncols; col++) {
            printf("%5g ", im.data[row][col]);
        }
        printf("\n");
    }

    return 0;
}
