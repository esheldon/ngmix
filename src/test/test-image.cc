#include <cstdio>
#include "mtrng.h"
#include "image.h"

using namespace std;

using image::Image;

int main(int argc, char **argv)
{

    Image im(2,2);

    mtrng::MtRng64 rng;

    std::printf("seeding from /dev/urandom\n");
    rng.init_dev_urandom();

    im.resize(5,5);

    im.add_constant(10);

    im.print();

    std::printf("\n");

    im.add_gaussian_noise(rng, 1.0);
    im.print();

    return 0;
}
