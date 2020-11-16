

class KDE(object):
    """
    create a kde from the input data

    just a wrapper around scipy.stats.gaussian_kde to
    provide a uniform interface
    """

    def __init__(self, data, kde_factor):
        import scipy.stats

        if len(data.shape) == 1:
            self.is_1d = True
        else:
            self.is_1d = False

        self.kde = scipy.stats.gaussian_kde(
            data.transpose(), bw_method=kde_factor,
        )

    def sample(self, n=None):
        """
        draw random samples from the kde
        """
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
