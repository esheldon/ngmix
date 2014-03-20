#include <cstdio>
#include "shape.h"

using namespace std;

using NGMix::Shape;

int main(int argc, char **argv)
{

    Shape shape1;
    Shape shape2(0.2, -0.15);
    Shape shear(0.1, 0.0);

    printf("shape1\n");
    shape1.show();

    printf("shape1 reset\n");
    shape1.set_g(.324, .1234);
    shape1.show();

    printf("shape2\n");
    shape2.show();

    shape2.shear(shear);

    printf("shear\n");
    shear.show();

    printf("shape 2 after shear\n");
    shape2.show();

    printf("e1:   %.16g   e2: %.16g\n", shape2.e1, shape2.e2);
    printf("eta1: %.16g eta2: %.16g\n", shape2.eta1, shape2.eta2);
    return 0;

}
