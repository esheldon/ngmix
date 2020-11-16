import numpy


def make_rng(rng=None):
    """
    if the input rng is None, create a new RandomState
    but with a seed generated from current numpy state
    """
    if rng is None:
        seed = numpy.random.randint(0, 2 ** 30)
        rng = numpy.random.RandomState(seed)

    return rng


def srandu(num=None, rng=None):
    """
    Generate random numbers in the symmetric distribution [-1,1]

    if the input rng is None, use the global numpy RNG state
    """
    if rng is None:
        randu = numpy.random.uniform
    else:
        randu = rng.uniform

    return randu(low=-1.0, high=1.0, size=num)
