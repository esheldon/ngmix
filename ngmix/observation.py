import numpy
from .jacobian import Jacobian, UnitJacobian
from .gmix import GMix
import copy

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
                 aperture=None,
                 psf=None):

        self.image=numpy.asanyarray(image, dtype='f8')
        assert len(image.shape)==2,"image must be 2d"

        self.meta={}

        # If jacobian is None, set UnitJacobian
        self.set_jacobian(jacobian)

        # if weight is None, set unity weights
        self.set_weight(weight)

        if gmix is not None:
            self.set_gmix(gmix)

        if aperture is not None:
            self.set_aperture(aperture)

        if psf is not None:
            self.set_psf(psf)

    def set_weight(self, weight):
        """
        Set the weight map.

        parameters
        ----------
        weight: ndarray (or None)
        """

        if weight is not None:
            weight=numpy.asanyarray(weight, dtype='f8')
            assert len(weight.shape)==2,"weight must be 2d"

            assert (weight.shape==self.image.shape),"image and weight must be same shape"

        else:
            weight = numpy.zeros(self.image.shape) + 1.0

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

    def get_jacobian(self):
        return self.jacobian.copy()

    def set_psf(self,psf):
        """
        Set a psf Observation
        """

        mess="psf must be of Observation, got %s" % type(psf)
        assert isinstance(psf,Observation),mess
        self.psf=psf

    def get_psf(self):
        """
        get the psf object
        """
        if not self.has_psf():
            raise RuntimeError("this obs has no psf set")
        return self.psf

    def has_psf(self):
        """
        does this object have a psf set?
        """
        return hasattr(self,'psf')

    def get_psf_gmix(self):
        """
        get the psf gmix if it exists
        """
        if not self.has_psf_gmix():
            raise RuntimeError("this obs has not psf set with a gmix")
        return self.psf.get_gmix()


    def has_psf_gmix(self):
        """
        does this object have a psf set, which has a gmix set?
        """
        if self.has_psf():
            return self.psf.has_gmix()
        else:
            return False


    def set_gmix(self,gmix):
        """
        Set a psf gmix.
        """
        mess="gmix must be of type GMix, got %s" % type(gmix)
        assert isinstance(gmix,GMix),mess
        self.gmix=gmix.copy()

    def get_gmix(self):
        """
        get a copy of the gmix object
        """
        if not self.has_gmix():
            raise RuntimeError("this obs has not gmix set")
        return self.gmix.copy()

    def has_gmix(self):
        """
        does this object have a gmix set?
        """
        return hasattr(self, 'gmix')

    def set_aperture(self,aperture):
        """
        Set an aperture.
        """
        if self.has_aperture():
            del self.aperture

        if aperture is not None:
            self.aperture=float(aperture)

    def get_aperture(self):
        """
        get a copy of the aperture
        """
        if not self.has_aperture():
            raise RuntimeError("this obs has no aperture set")
        return copy.copy(self.aperture)

    def has_aperture(self):
        """
        returns True if the aperture is not None
        """
        return hasattr(self,'aperture')

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

    def set_aperture(self, aper):
        """
        set aperture on all contained Observations
        """
        for obs in self:
            obs.set_aperture(aper)

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

    def set_aperture(self, aper):
        """
        set aperture on all contained Observations
        """
        for obslist in self:
            obslist.set_aperture(aper)

    def __setitem__(self, index, obs_list):
        """
        over-riding this for type safety
        """
        assert isinstance(obs_list,ObsList),"obs_list should be of type ObsList"
        super(MultiBandObsList,self).__setitem__(index, obs_list)

def get_mb_obs(obs_in):
    """
    convert the input to a MultiBandObsList

    Input should be an Observation, ObsList, or MultiBandObsList
    """

    if isinstance(obs_in,Observation):
        obs_list=ObsList()
        obs_list.append(obs_in)

        obs=MultiBandObsList()
        obs.append(obs_list)
    elif isinstance(obs_in,ObsList):
        obs=MultiBandObsList()
        obs.append(obs_in)
    elif isinstance(obs_in,MultiBandObsList):
        obs=obs_in
    else:
        raise ValueError("obs should be Observation, ObsList, or MultiBandObsList")

    return obs


