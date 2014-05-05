from numpy import ndarray
from .jacobian import Jacobian, UnitJacobian
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
    jacobian: Jacobian, optional
        Type Jacobian or a sub-type
    gmix: GMix, optional
        Optional GMix object associated with this observation
    psf: Observation, optional
        Optional psf Observation
    """

    def __init__(self, image,
                 weight=None,
                 jacobian=None,
                 gmix=None,
                 psf=None):

        assert isinstance(image,ndarray),"image must be of type ndarray"
        assert len(image.shape)==2,"image must be 2d"
        self.image=image.astype('f8', copy=False)

        self.meta={}

        self.set_jacobian(jacobian)
        self.set_weight(weight)
        self.set_gmix(gmix)
        self.set_psf(psf)

    def set_weight(self, weight):
        """
        Set the weight map.

        parameters
        ----------
        weight: ndarray (or None)
        """

        if weight is not None:
            assert isinstance(weight,ndarray),"weight must be of type ndarray"
            weight=weight.astype('f8', copy=False)

            assert len(weight.shape)==2,"weight must be 2d"
            assert (weight.shape==self.image.shape),"image and weight must be same shape"

        self.weight=weight

    def set_jacobian(self, jacobian):
        """
        Set the jacobian.

        parameters
        ----------
        jacobian: Jacobian (or None)
        """
        if jacobian is None:
            jacobian=UnitJacobian(0.0, 0.0)
        assert isinstance(jacobian,Jacobian),"jacobian must be of type Jacobian"
        self.jacobian=jacobian

    def set_psf(self,psf):
        """
        Set a psf Observation
        """
        if psf is not None:
            assert isinstance(psf,Observation),"psf must be of Observation"
        self.psf=psf

    def set_gmix(self,gmix):
        """
        Set a psf gmix.
        """
        if gmix is not None:
            assert isinstance(gmix,GMix),"gmix must be of type GMix"
        self.gmix=gmix

    def update_meta_data(self, meta_dict):
        """
        Add some metadata
        """
        self.meta.update(meta_dict)

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
        assert isinstance(obs,Observation),"obs should be of type Observation, got %s" % type(obs)
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
        super(MultiBandObsList,self).append(obs_list)

    def __setitem__(self, index, obs_list):
        """
        over-riding this for type safety
        """
        assert isinstance(obs_list,ObsList),"obs_list should be of type ObsList"
        super(MultiBandObsList,self).__setitem__(index, obs_list)


