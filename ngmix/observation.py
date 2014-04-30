from numpy import ndarray
from .jacobian import Jacobian
from .gmix import GMix

class Observation(object):
    """
    Represent an observation with an image and possibly a
    weight map and jacobian

    parameters
    ----------
    image: ndarray
        The image
    weight: ndarray
        Weight map, same shape as image
    jacobian: Jacobian
        Type Jacobian or a sub-type
    psf_image: ndarray, optional
        Optional psf image
    psf_gmix: GMix, optional
        Optional GMix object representing the PSF
    """

    def __init__(self, image, weight, jacobian, psf_image=None, psf_gmix=None):
        assert isinstance(image,ndarray),"image must be of type ndarray"
        assert isinstance(weight,ndarray),"weight must be of type ndarray"
        assert isinstance(jacobian,Jacobian),"jacobian must be of type Jacobian"
        assert len(image.shape)==2,"image must be 2d"
        assert len(weight.shape)==2,"weight must be 2d"

        assert (image.shape==weight.shape),"image and weight must be same shape"

        self.image=image.astype('f8', copy=False)
        self.weight=weight.astype('f8', copy=False)
        self.jacobian=jacobian

        self.set_psf_image(psf_image)
        self.set_psf_gmix(psf_gmix)

    def set_psf_image(self,psf_image):
        """
        Set a psf image.
        """
        if psf_image is not None:
            assert isinstance(psf_image,ndarray),"psf_image must be of type ndarray"
        self.psf_image=psf_image

    def set_psf_gmix(self,psf_gmix):
        """
        Set a psf gmix.
        """
        if psf_gmix is not None:
            assert isinstance(psf_gmix,GMix),"psf_gmix must be of type GMix"
        self.psf_gmix=psf_gmix


class ObsList(list):
    """
    Hold a list of Observation objects

    This class provides a bit of type safety and ease of type checking
    """
    def append(self, obs):
        """
        Add a new observation

        over-riding this for type safety
        """
        assert isinstance(obs,Observation),"obs should be of type Observation"
        super(ObsList,self).append(obs)

    def __setitem__(self, index, obs):
        """
        over-riding this for type safety
        """
        assert isinstance(obs,Observation),"obs should be of type Observation"
        super(ObsList,self).__setitem__(index, obs)

class MultiBandObsList(list):
    """
    Hold a list of lists of ObsList objects, each representing a filter
    band

    This class provides a bit of type safety and ease of type checking
    """

    def append(self, obs_list):
        """
        Add a new ObsList

        over-riding this for type safety
        """
        assert isinstance(obs_list,ObsList),"obs_list should be of type ObsList"
        super(MultibandObsList,self).append(obs_list)

    def __setitem__(self, index, obs_list):
        """
        over-riding this for type safety
        """
        assert isinstance(obs_list,ObsList),"obs_list should be of type ObsList"
        super(MultibandObsList,self).__setitem__(index, obs_list)


