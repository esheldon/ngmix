__all__ = ['GMixList', 'MultiBandGMixList']

from .gmix import GMix


class GMixList(list):
    """
    Hold a list of GMix objects

    This class provides a bit of type safety and ease of type checking
    """

    def append(self, gmix):
        """
        Add a new mixture

        over-riding this for type safety
        """
        assert isinstance(gmix, GMix), "gmix should be of type GMix"
        super(GMixList, self).append(gmix)

    def __setitem__(self, index, gmix):
        """
        over-riding this for type safety
        """
        assert isinstance(gmix, GMix), "gmix should be of type GMix"
        super(GMixList, self).__setitem__(index, gmix)


class MultiBandGMixList(list):
    """
    Hold a list of lists of GMixList objects, each representing a filter
    band

    This class provides a bit of type safety and ease of type checking
    """

    def append(self, gmix_list):
        """
        add a new GMixList

        over-riding this for type safety
        """
        assert isinstance(
            gmix_list, GMixList
        ), "gmix_list should be of type GMixList"
        super(MultiBandGMixList, self).append(gmix_list)

    def __setitem__(self, index, gmix_list):
        """
        over-riding this for type safety
        """
        assert isinstance(
            gmix_list, GMixList
        ), "gmix_list should be of type GMixList"
        super(MultiBandGMixList, self).__setitem__(index, gmix_list)
