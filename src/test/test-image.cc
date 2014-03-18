#include <cstdio>
#include "image.h"

using namespace std;

using image::Image;

int main(int argc, char **argv)
{

    Image im(2,2);

    MtRng::MtRng64 rng;
    rng.init_from_dev_urandom();

    im.resize(5,5);

    im.add_constant(10);

    im.show();

    std::printf("\n");

    im.add_gaussian_noise(rng, 1.0);
    im.show();

    return 0;
}
