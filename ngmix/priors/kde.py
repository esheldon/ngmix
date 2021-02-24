import scipy.stats


class KDE(object):
    """
    create a kde from the input data

    This class is a wrapper around scipy.stats.gaussian_kde to
    provide a uniform interface.

    parameters
    ----------
    data : array, shape (# of data points, # of dims)
        The input data to fit.
    kde_factor : str, scalar, or callable
        Any valid input to the keyword `bw_method` for the gaussian KDE.
    """
    def __init__(self, data, kde_factor):
        if len(data.shape) == 1:
            self.is_1d = True
        else:
            self.is_1d = False

        self.kde = scipy.stats.gaussian_kde(
            data.transpose(), bw_method=kde_factor,
        )

    def sample(self, nrand=None, n=None):
        """
        draw random samples from the kde.

        parameters
        ----------
        nrand: int or None
            The number of samples. If None, a single scalar sample is drawn.
            Default is None.

        returns
        -------
        samples: scalar or array-like
            The samples. The output shape depends on the dimensoon of the input data to
            the KDE. If the input data is one dimensional, then the output shape is
            (`nrand`,) or a scalar depending on if `nrand` is not None. If the input
            data has more than one dimension, then the output shape is either
            (`ndims`, `nrand`) or (`ndims`,) depending on whether or not `nrand` is
            non-None or None.
        """
        if n is None and nrand is not None:
            # if they have given nrand and not n, use that
            # this keeps the API the same but allows ppl to use the new API of nrand
            n = nrand

        if n is None:
            n = 1
            is_scalar = True
        else:
            is_scalar = False

        r = self.kde.resample(size=n).transpose()

        if self.is_1d:
            r = r[:, 0]

        if is_scalar:
            r = r[0]

        return r
