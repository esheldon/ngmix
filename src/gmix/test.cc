#include <cstdio>
#include "gmix.h"

using namespace std;

using gmix::GMix;
using gmix::Gauss;

int main(int argc, char **argv)
{

    Gauss gempty;
    Gauss ginit(1.0, 25.0, 26.0, 4.0, 1.6, 3.7);
    Gauss gset;

    gset.set(1.0, 25.0, 26.0, 4.0, 1.6, 3.7);

    GMix gmix;

    return 0;
}
