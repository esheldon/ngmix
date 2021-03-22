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


def srandu(nrand=None, *, rng):
    """
    Generate random numbers in the symmetric distribution [-1,1]

    Parameters
    ----------
    nrand: int or None, optional
        Number of samples. If None a scalar is returned, else an array
    rng: np.random.RandomState
        The random number generator

    Returns
    -------
    samples between
    """
    randu = rng.uniform
    return randu(low=-1.0, high=1.0, size=nrand)
