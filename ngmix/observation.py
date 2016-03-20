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
    weight: ndarray, optional
        Weight map, same shape as image
    bmask: ndarray, optional
        A bitmask array
    jacobian: Jacobian, optional
        Type Jacobian or a sub-type
    gmix: GMix, optional
        Optional GMix object associated with this observation
    psf: Observation, optional
        Optional psf Observation
    meta: dict
        Optional dictionary
    """

    def __init__(self, image,
                 weight=None,
                 bmask=None,
                 jacobian=None,
                 gmix=None,
                 aperture=None,
                 psf=None,
                 meta=None):

        self.image=None
        self.set_image(image)

        self.meta={}

        if meta is not None:
            self.update_meta_data(meta)

        # If jacobian is None, set UnitJacobian
        self.set_jacobian(jacobian)

        # if weight is None, set unity weights
        self.set_weight(weight)

        self.set_bmask(bmask)

        if gmix is not None:
            self.set_gmix(gmix)

        if aperture is not None:
            self.set_aperture(aperture)

        if psf is not None:
            self.set_psf(psf)

    def set_image(self, image):
        """
        Set the image.  If the image is being reset, must be
        same shape as previous incarnation in order to remain
        consistent

        parameters
        ----------
        image: ndarray (or None)
        """

        image_old=self.image

        # force f8 with native byte ordering, contiguous C layout
        self.image=numpy.ascontiguousarray(image,dtype='f8')

        assert len(self.image.shape)==2,"image must be 2d"

        if image_old is not None:
            mess=("old and new image must have same shape, to "
                  "maintain consistency")
            assert self.image.shape == image_old.shape,mess

    def set_weight(self, weight):
        """
        Set the weight map.

        parameters
        ----------
        weight: ndarray (or None)
        """

        if weight is not None:
            # force f8 with native byte ordering, contiguous C layout
            weight=numpy.ascontiguousarray(weight, dtype='f8')
            assert len(weight.shape)==2,"weight must be 2d"

            mess="image and weight must be same shape"
            assert (weight.shape==self.image.shape),mess

        else:
            weight = numpy.zeros(self.image.shape) + 1.0

        self.weight=weight

    def set_bmask(self, bmask):
        """
        Set the bitmask

        parameters
        ----------
        bmask: ndarray (or None)
        """

        if bmask is not None:
            # force contiguous C, but we don't know what dtype to expect
            bmask=numpy.ascontiguousarray(bmask)
            assert len(bmask.shape)==2,"bmask must be 2d"

            assert (bmask.shape==self.image.shape),"image and bmask must be same shape"

        self.bmask=bmask


    def set_jacobian(self, jacobian):
        """
        Set the jacobian.

        parameters
        ----------
        jacobian: Jacobian (or None)
        """
        if jacobian is None:
            jacobian=UnitJacobian(row=0.0, col=0.0)
        assert isinstance(jacobian,Jacobian),"jacobian must be of type Jacobian"
        self.jacobian=jacobian

    def get_jacobian(self):
        """
        get a copy of the jacobian
        """
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

    def get_s2n(self):
        """
        get the the simple s/n estimator

        sum(I)/sqrt( sum( 1/w ) ) = Isum/sqrt(Vsum)

        returns
        -------
        s2n: float
            The supid s/n estimator
        """

        Isum, Vsum, Npix = self.get_s2n_sums()
        if Vsum > 0.0:
            s2n = Isum/numpy.sqrt(Vsum)
        else:
            s2n=-9999.0
        return s2n


    def get_s2n_sums(self):
        """
        get the sums for the simple s/n estimator

        sum(I)/sqrt( sum( 1/w ) ) = Isum/sqrt(Vsum)

        returns
        -------
        Isum, Vsum, Npix
        """

        image = self.image
        weight = self.weight

        w=numpy.where(weight > 0)

        if w[0].size > 0:
            Isum = image[w].sum()
            Vsum = (1.0/weight[w]).sum()
            Npix = w[0].size
        else:
            Isum = 0.0
            Vsum = 0.0
            Npix = 0

        return Isum, Vsum, Npix

    def update_meta_data(self, meta):
        """
        Add some metadata
        """

        if not isinstance(meta,dict):
            raise TypeError("meta data must be in dictionary form")
        self.meta.update(meta)

class ObsList(list):
    """
    Hold a list of Observation objects

    This class provides a bit of type safety and ease of type checking
    """

    def __init__(self, meta=None):
        super(ObsList,self).__init__()

        self.meta={}
        if meta is not None:
            self.update_meta_data(meta)

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

    def get_s2n(self):
        """
        get the the simple s/n estimator

        sum(I)/sqrt( sum( 1/w ) ) = Isum/sqrt(Vsum)

        returns
        -------
        s2n: float
            The supid s/n estimator
        """

        Isum, Vsum, Npix = self.get_s2n_sums()
        if Vsum > 0.0:
            s2n = Isum/numpy.sqrt(Vsum)
        else:
            s2n=-9999.0
        return s2n


    def get_s2n_sums(self):
        """
        get the sums for the simple s/n estimator

        sum(I)/sqrt( sum( 1/w ) ) = Isum/sqrt(Vsum)

        returns
        -------
        Isum, Vsum, Npix
        """

        Isum = 0.0
        Vsum = 0.0
        Npix = 0

        for obs in self:
            tIsum,tVsum,tNpix = obs.get_s2n_sums()
            Isum += tIsum
            Vsum += tVsum
            Npix += tNpix

        return Isum, Vsum, Npix

    def update_meta_data(self, meta):
        """
        Add some metadata
        """

        if not isinstance(meta,dict):
            raise TypeError("meta data must be in dictionary form")
        self.meta.update(meta)

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

    def __init__(self, meta=None):
        super(MultiBandObsList,self).__init__()

        self.meta={}
        if meta is not None:
            self.update_meta_data(meta)

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

    def get_s2n(self):
        """
        get the the simple s/n estimator

        sum(I)/sqrt( sum( 1/w ) ) = Isum/sqrt(Vsum)

        returns
        -------
        s2n: float
            The supid s/n estimator
        """

        Isum, Vsum, Npix = self.get_s2n_sums()
        if Vsum > 0.0:
            s2n = Isum/numpy.sqrt(Vsum)
        else:
            s2n=-9999.0
        return s2n

    def get_s2n_sums(self):
        """
        get the sums for the simple s/n estimator

        sum(I)/sqrt( sum( 1/w ) ) = Isum/sqrt(Vsum)

        returns
        -------
        Isum, Vsum, Npix
        """

        Isum = 0.0
        Vsum = 0.0
        Npix = 0

        for obslist in self:
            tIsum,tVsum,tNpix = obslist.get_s2n_sums()
            Isum += tIsum
            Vsum += tVsum
            Npix += tNpix

        return Isum, Vsum, Npix


    def update_meta_data(self, meta):
        """
        Add some metadata
        """

        if not isinstance(meta,dict):
            raise TypeError("meta data must be in dictionary form")
        self.meta.update(meta)

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

