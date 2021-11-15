__all__ = [
    'Observation', 'ObsList', 'MultiBandObsList', 'get_mb_obs',
    'KObservation', 'KObsList', 'KMultiBandObsList',
    'make_kobs', 'get_kmb_obs',
]
import copy
import numpy as np
from .jacobian import Jacobian, UnitJacobian, DiagonalJacobian
from .gmix import GMix

from .pixels import make_pixels

DEFAULT_XINTERP = 'lanczos15'


class MetadataMixin(object):
    @property
    def meta(self):
        """
        get the metadata dictionary

        currently this simply returns a reference
        """
        return self._meta

    @meta.setter
    def meta(self, meta):
        """
        set the metadata dictionary

        This method does consistency checks and will raise a TypeError if the input is
        not None or a Python dict.
        """
        self.set_meta(meta)

    def set_meta(self, meta):
        """
        set the metadata dictionary

        This method does consistency checks and will raise a TypeError if the input is
        not None or a Python dict.
        """
        if meta is None:
            meta = {}

        if not isinstance(meta, dict):
            raise TypeError("meta data must be in "
                            "dictionary form, got %s" % type(meta))

        self._meta = meta

    def update_meta_data(self, meta):
        """
        Update the metadata dictionary

        This method does consistency checks and will raise a TypeError if the input is
        not a Python dict.
        """
        if not isinstance(meta, dict):
            raise TypeError(
                "meta data must be in dictionary form, got %s" % type(meta)
            )
        self.meta.update(meta)


class Observation(MetadataMixin):
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
    ormask: ndarray, optional
        A bitmask array
    noise: ndarray, optional
        A noise field to associate with this observation
    jacobian: Jacobian, optional
        Type Jacobian or a sub-type
    gmix: GMix, optional
        Optional GMix object associated with this observation
    psf: Observation, optional
        Optional psf Observation
    meta: dict
        Optional dictionary
    mfrac: ndarray, optional
        A masked fraction image for this observation.
    ignore_zero_weight: bool
        If True, do not store zero weight pixels in the pixels
        array.  Default is True.
    store_pixels: bool
        If True, store an array of pixels for use in fitting routines.
        If False, the ignore_zero_weight keyword is not used.
    ignore_zero_weight: bool
        Only relevant if store_pixels is True.
        If ignore_zero_weight is True, then zero-weight pixels are ignored
        when constructing the internal pixels array for fitting routines.
        If False, then zero-weight pixels are included in the internal pixels
        array.

    notes
    -----
    Updates of the internal data of ngmix.Observation will only work in
    a python context, e.g:

        with obs.writeable():
            obs.image[w] += 5
    """
    def __init__(self,
                 image,
                 weight=None,
                 bmask=None,
                 ormask=None,
                 noise=None,
                 jacobian=None,
                 gmix=None,
                 psf=None,
                 meta=None,
                 mfrac=None,
                 store_pixels=True,
                 ignore_zero_weight=True):

        self._writeable = False
        self._ignore_zero_weight = ignore_zero_weight
        self._store_pixels = store_pixels

        # pixels depends on image, weight and jacobian, so delay until all are
        # set

        self.set_image(image, update_pixels=False)

        # If these are None, they get default values

        self.set_weight(weight, update_pixels=False)
        self.set_jacobian(jacobian, update_pixels=False)

        # now image, weight, and jacobian are set, create
        # the pixel array
        self.update_pixels()

        self.set_meta(meta)

        # optional, if None nothing is set
        self.set_bmask(bmask)
        self.set_ormask(ormask)
        self.set_noise(noise)
        self.set_gmix(gmix)
        self.set_psf(psf)
        self.set_mfrac(mfrac)

    @property
    def image(self):
        """
        getter for image

        returns a read-only reference
        """
        return self._get_view(self._image)

    @image.setter
    def image(self, image):
        """
        set the image

        this does consistency checks and can trigger an update
        of the pixels array
        """
        self.set_image(image)

    @property
    def weight(self):
        """
        getter for weight

        returns a read-only reference
        """
        return self._get_view(self._weight)

    @weight.setter
    def weight(self, weight):
        """
        set the weight

        this does consistency checks and can trigger an update
        of the pixels array
        """
        self.set_weight(weight)

    @property
    def pixels(self):
        """
        getter for pixels

        this simply returns a reference.  Note the pixels array is *always*
        read only.  To reset the pixels you must reset the image/weight/jacobian
        """
        return self._pixels

    @property
    def mfrac(self):
        """
        getter for mfrac

        returns a read-only reference
        """
        return self._get_view(self._mfrac)

    @mfrac.setter
    def mfrac(self, mfrac):
        """
        set the mfrac, with consistency checks
        """
        self.set_mfrac(mfrac)

    @property
    def bmask(self):
        """
        getter for bmask

        returns a read-only reference
        """
        return self._get_view(self._bmask)

    @bmask.setter
    def bmask(self, bmask):
        """
        set the bmask, with consistency checks
        """
        self.set_bmask(bmask)

    @property
    def ormask(self):
        """
        getter for ormask

        returns a read-only reference
        """
        return self._get_view(self._ormask)

    @ormask.setter
    def ormask(self, ormask):
        """
        set the ormask
        """
        self.set_ormask(ormask)

    @property
    def noise(self):
        """
        getter for noise

        returns a read-only reference
        """
        return self._get_view(self._noise)

    @noise.setter
    def noise(self, noise):
        """
        set the noise
        """
        self.set_noise(noise)

    @property
    def jacobian(self):
        """
        get a read-only reference to the jacobian.  A new jacobian
        is made with read-only reference to underlying data
        """
        return self.get_jacobian()

    @jacobian.setter
    def jacobian(self, jacobian):
        """
        set the jacobian
        """
        self.set_jacobian(jacobian)

    @property
    def gmix(self):
        """
        get a copy of the gaussian mixture
        """
        return self.get_gmix()

    @gmix.setter
    def gmix(self, gmix):
        """
        set the gmix
        """
        self.set_gmix(gmix)

    @property
    def psf(self):
        """
        getter for psf

        currently this simply returns a reference
        """
        return self._psf

    @psf.setter
    def psf(self, psf):
        """
        set the psf
        """
        self.set_psf(psf)

    def set_image(self, image, update_pixels=True):
        """
        Set the image.  If the image is being reset, must be
        same shape as previous incarnation in order to remain
        consistent

        parameters
        ----------
        image: ndarray (or None)
            The new image.
        update_pixels: bool
            If True, update the internal pixels array. If False, do not
            do this update. Default is True.
        """

        if hasattr(self, '_image'):
            image_old = self._image
        else:
            image_old = None

        # force f8 with native byte ordering, contiguous C layout
        image = np.ascontiguousarray(image, dtype='f8')

        assert len(image.shape) == 2, "image must be 2d"

        if image_old is not None:
            mess = ("old and new image must have same shape, to "
                    "maintain consistency, got %s "
                    "vs %s" % (image.shape, image_old.shape))
            assert image.shape == image_old.shape, mess

        self._image = image

        if update_pixels:
            self.update_pixels()

    def set_weight(self, weight, update_pixels=True):
        """
        Set the weight map.

        parameters
        ----------
        weight: ndarray (or None)
            The new weight image.
        update_pixels: bool
            If True, update the internal pixels array. If False, do not
            do this update. Default is True.
        """

        image = self.image
        if weight is not None:
            # force f8 with native byte ordering, contiguous C layout
            weight = np.ascontiguousarray(weight, dtype='f8')
            assert len(weight.shape) == 2, "weight must be 2d"

            mess = "image and weight must be same shape"
            assert (weight.shape == image.shape), mess

        else:
            weight = np.zeros(image.shape) + 1.0

        self._weight = weight
        if update_pixels:
            self.update_pixels()

    def set_mfrac(self, mfrac):
        """
        Set the masked fraction entry.

        parameters
        ----------
        mfrac: ndarray (or None)
            The new masked fraction image.
        """
        if mfrac is None:
            if self.has_mfrac():
                del self._mfrac
        else:

            image = self.image

            # force contiguous C, but we don't know what dtype to expect
            mfrac = np.ascontiguousarray(mfrac)
            assert len(mfrac.shape) == 2, "mfrac must be 2d"

            assert (mfrac.shape == image.shape), \
                "image and mfrac must be same shape"

            self._mfrac = mfrac

    def has_mfrac(self):
        """
        returns True if a masked fraction image is set
        """
        if hasattr(self, '_mfrac'):
            return True
        else:
            return False

    def set_bmask(self, bmask):
        """
        Set the bitmask

        parameters
        ----------
        bmask: ndarray (or None)
            The new bit mask image.
        """
        if bmask is None:
            if self.has_bmask():
                del self._bmask
        else:

            image = self.image

            # force contiguous C, but we don't know what dtype to expect
            bmask = np.ascontiguousarray(bmask)
            assert len(bmask.shape) == 2, "bmask must be 2d"

            assert (bmask.shape == image.shape), \
                "image and bmask must be same shape"

            self._bmask = bmask

    def has_bmask(self):
        """
        returns True if a bitmask is set
        """
        if hasattr(self, '_bmask'):
            return True
        else:
            return False

    def set_ormask(self, ormask):
        """
        Set the bitmask

        parameters
        ----------
        ormask: ndarray (or None)
            The new "or" mask image.
        """
        if ormask is None:
            if self.has_ormask():
                del self._ormask
        else:

            image = self.image

            # force contiguous C, but we don't know what dtype to expect
            ormask = np.ascontiguousarray(ormask)
            assert len(ormask.shape) == 2, "ormask must be 2d"

            assert (ormask.shape == image.shape),\
                "image and ormask must be same shape"

            self._ormask = ormask

    def has_ormask(self):
        """
        returns True if a bitmask is set
        """
        if hasattr(self, '_ormask'):
            return True
        else:
            return False

    def set_noise(self, noise):
        """
        Set a noise image

        parameters
        ----------
        noise: ndarray (or None)
            The new noise image.
        """
        if noise is None:
            if self.has_noise():
                del self._noise
        else:

            image = self.image

            # force contiguous C, but we don't know what dtype to expect
            noise = np.ascontiguousarray(noise)
            assert len(noise.shape) == 2, "noise must be 2d"

            assert (noise.shape == image.shape), \
                "image and noise must be same shape"

            self._noise = noise

    def has_noise(self):
        """
        returns True if a bitmask is set
        """
        if hasattr(self, '_noise'):
            return True
        else:
            return False

    def set_jacobian(self, jacobian, update_pixels=True):
        """
        Set the jacobian.  If None is sent, a UnitJacobian is generated with
        center equal to the canonical center

        parameters
        ----------
        jacobian: Jacobian (or None)
            The new jacobian.
        update_pixels: bool
            If True, update the internal pixels array. If False, do not
            do this update. Default is True.
        """
        if jacobian is None:
            cen = (np.array(self.image.shape)-1.0)/2.0
            jac = UnitJacobian(row=cen[0], col=cen[1])
        else:
            mess = ("jacobian must be of "
                    "type Jacobian, got %s" % type(jacobian))
            assert isinstance(jacobian, Jacobian), mess
            jac = jacobian.copy()

        self._jacobian = jac

        if update_pixels:
            self.update_pixels()

    def get_jacobian(self):
        """
        get a jacobian with reference to our jacobian's data

        this is not writeable by default
        """
        j = self._jacobian.copy()
        j._data = self._get_view(self._jacobian._data)
        return j

    def set_psf(self, psf):
        """
        Set a psf Observation
        """
        if self.has_psf():
            del self._psf

        if psf is not None:
            mess = "psf must be of Observation, got %s" % type(psf)
            assert isinstance(psf, Observation), mess
            self._psf = psf

    def get_psf(self):
        """
        get the psf object
        """
        if not self.has_psf():
            raise RuntimeError("this obs has no psf set")
        return self._psf

    def has_psf(self):
        """
        does this object have a psf set?
        """
        return hasattr(self, '_psf')

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

    def set_gmix(self, gmix):
        """
        Set the gmix.

        parameters
        ----------
        gmix: ngmix.GMix
            The GMix to use to set the internal GMix.
        """

        if self.has_gmix():
            del self._gmix

        if gmix is not None:
            mess = "gmix must be of type GMix, got %s" % type(gmix)
            assert isinstance(gmix, GMix), mess
            self._gmix = gmix.copy()

    def get_gmix(self):
        """
        get a copy of the gmix object
        """
        if not self.has_gmix():
            raise RuntimeError("this obs has not gmix set")
        return self._gmix.copy()

    def has_gmix(self):
        """
        does this object have a gmix set?
        """
        return hasattr(self, '_gmix')

    def get_s2n(self):
        """
        get the the simple s/n estimator

        sum(I)/sqrt( sum( 1/w ) ) = Isum/sqrt(Vsum)

        returns
        -------
        s2n: float
            The s/n of the images. Will retun -9999 if the s/n cannot be computed.
        """

        Isum, Vsum, Npix = self.get_s2n_sums()
        if Vsum > 0.0:
            s2n = Isum/np.sqrt(Vsum)
        else:
            s2n = -9999.0
        return s2n

    def get_s2n_sums(self):
        """
        get the sums for the simple s/n estimator

        sum(I)/sqrt( sum( 1/w ) ) = Isum/sqrt(Vsum)

        returns
        -------
        Isum: float
            The value sum(I).
        Vsum: float
            The value sum(1/w)
        Npix: int
            The number of non-zero-weight pixels.
        """

        image = self.image
        weight = self.weight

        w = np.where(weight > 0)

        if w[0].size > 0:
            Isum = image[w].sum()
            Vsum = (1.0/weight[w]).sum()
            Npix = w[0].size
        else:
            Isum = 0.0
            Vsum = 0.0
            Npix = 0

        return Isum, Vsum, Npix

    def copy(self, memo=None):
        """
        make a copy of the observation
        """
        if self.has_bmask():
            bmask = self.bmask.copy()
        else:
            bmask = None

        if self.has_ormask():
            ormask = self.ormask.copy()
        else:
            ormask = None

        if self.has_noise():
            noise = self.noise.copy()
        else:
            noise = None

        if self.has_gmix():
            # makes a copy
            gmix = self.gmix
        else:
            gmix = None

        if self.has_psf():
            psf = self.psf.copy()
        else:
            psf = None

        if self.has_mfrac():
            mfrac = self.mfrac.copy()
        else:
            mfrac = None

        meta = copy.deepcopy(self._meta, memo=memo)

        return Observation(
            self.image.copy(),
            weight=self.weight.copy(),
            bmask=bmask,
            ormask=ormask,
            noise=noise,
            gmix=gmix,
            jacobian=self.jacobian,  # makes a copy internally
            meta=meta,
            psf=psf,
            mfrac=mfrac,
            store_pixels=self._store_pixels,
            ignore_zero_weight=self._ignore_zero_weight,
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        result = self.copy(memo=memo)
        memo[id(self)] = result
        return result

    def __eq__(self, obs):
        if not isinstance(obs, Observation):
            return False

        # attributes that all share
        if self.meta != obs.meta:
            return False

        # image attributes
        attrs = (
            ('image', 'array'),
            ('weight', 'array'),
            ('bmask', 'array'),
            ('ormask', 'array'),
            ('mfrac', 'array'),
            ('noise', 'array'),
            ('psf', 'obj'),
            ('gmix', 'obj'),
            ('jacobian', 'obj'),
            ('meta', 'obj'),
        )
        for attr, atype in attrs:
            has = f'has_{attr}'
            if not hasattr(self, has):
                # these are probably required
                self_has = obs_has = True
            else:
                self_has = getattr(self, has)()
                obs_has = getattr(self, has)()
            if self_has or obs_has:
                if self_has and obs_has:

                    self_data = getattr(self, attr)
                    obs_data = getattr(obs, attr)
                    if not np.all(self_data == obs_data):
                        return False
                else:
                    return False

        return True

    @property
    def store_pixels(self):
        """getter for store_pixels attribute"""
        return self._store_pixels

    @store_pixels.setter
    def store_pixels(self, store_pixels):
        """setter for store pixels

        calls update_pixels after store_pixels is set if needed
        """
        # only update if value is changed
        do_update = False if store_pixels == self._store_pixels else True
        self._store_pixels = store_pixels
        if do_update:
            self.update_pixels()

    @property
    def ignore_zero_weight(self):
        """getter for ignore_zero_weight attribute"""
        return self._ignore_zero_weight

    @ignore_zero_weight.setter
    def ignore_zero_weight(self, ignore_zero_weight):
        """setter for ignore_zero_weight

        calls update_pixels after ignore_zero_weight is set if needed
        """
        do_update = False if ignore_zero_weight == self._ignore_zero_weight else True
        self._ignore_zero_weight = ignore_zero_weight
        if do_update:
            self.update_pixels()

    def update_pixels(self):
        """
        create the pixel struct array, for efficient cache usage
        """

        if not self._store_pixels:
            self._pixels = None
            return

        pixels = make_pixels(
            self.image,
            self.weight,
            self._jacobian,
            ignore_zero_weight=self._ignore_zero_weight,
        )
        pixels.flags['WRITEABLE'] = False
        self._pixels = pixels

    def _get_view(self, data):
        """return a view of some numpy data.

        The `_writeable` attribute is set by the context management methods
        `__enter__` and `__exit__` so that the internal data of the Observation
        can only be updated in place within a context manager.
        """
        view = data.view()
        view.flags['WRITEABLE'] = self._writeable
        return view

    def writeable(self):
        """
        returns self

        This method is meant to be used when updating the data of an Observation,
        e.g.,
            with obs.writeable():
                obs.image[w] += 5
        """
        return self

    def __enter__(self):
        self._writeable = True
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._writeable = False
        self.update_pixels()


class ObsList(list, MetadataMixin):
    """
    Hold a list of Observation objects

    This class provides a bit of type safety and ease of type checking.

    parameters
    ----------
    meta: dict or None
        Any metadata keep in the `meta` attribute. Optional.
    """

    def __init__(self, meta=None):
        super(ObsList, self).__init__()

        self.set_meta(meta)

    def append(self, obs):
        """
        Add a new observation

        over-riding this for type safety

        parameters
        ----------
        obs: ngmix.Observation
            An observation. An AssertionError will be raised if `obs` is not
            an `ngmix.Observation`.
        """
        mess = "obs should be of type Observation, got %s" % type(obs)
        assert isinstance(obs, Observation), mess
        super(ObsList, self).append(obs)

    def get_s2n(self):
        """
        get the the simple s/n estimator

        sum(I)/sqrt( sum( 1/w ) ) = Isum/sqrt(Vsum)

        returns
        -------
        s2n: float
            The s/n of the images. Will retun -9999 if the s/n cannot be computed.
        """

        Isum, Vsum, Npix = self.get_s2n_sums()
        if Vsum > 0.0:
            s2n = Isum/np.sqrt(Vsum)
        else:
            s2n = -9999.0
        return s2n

    def get_s2n_sums(self):
        """
        get the sums for the simple s/n estimator

        sum(I)/sqrt( sum( 1/w ) ) = Isum/sqrt(Vsum)

        returns
        -------
        -------
        Isum: float
            The value sum(I).
        Vsum: float
            The value sum(1/w)
        Npix: int
            The number of non-zero-weight pixels.
        """

        Isum = 0.0
        Vsum = 0.0
        Npix = 0

        for obs in self:
            tIsum, tVsum, tNpix = obs.get_s2n_sums()
            Isum += tIsum
            Vsum += tVsum
            Npix += tNpix

        return Isum, Vsum, Npix

    def copy(self, memo=None):
        """
        copy all the data into a new ObsList
        """
        new_obslist = ObsList(meta=copy.deepcopy(self._meta, memo))
        for obs in self:
            new_obslist.append(obs.copy(memo=memo))
        return new_obslist

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        result = self.copy(memo=memo)
        memo[id(self)] = result
        return result

    def __eq__(self, obslist):
        if len(self) != len(obslist):
            return False

        for self_obs, obs in zip(self, obslist):
            if self_obs != obs:
                return False
        return True

    def __setitem__(self, index, obs):
        """
        over-riding this for type safety
        """
        assert isinstance(obs, Observation), \
            "obs should be of type Observation"
        super(ObsList, self).__setitem__(index, obs)


class MultiBandObsList(list, MetadataMixin):
    """
    Hold a list of lists of ObsList objects, each representing a filter
    band

    This class provides a bit of type safety and ease of type checking

    parameters
    ----------
    meta: dict or None
        Any metadata keep in the `meta` attribute. Optional.
    """

    def __init__(self, meta=None):
        super(MultiBandObsList, self).__init__()

        self.set_meta(meta)

    def append(self, obs_list):
        """
        Add a new ObsList

        over-riding this for type safety

        parameters
        ----------
        obs_list: ngmix.ObsList
            An ObsList. An AssertionError will be raised if `obs_list` is not
            an `ngmix.ObsList`.
        """
        assert isinstance(obs_list, ObsList),\
            'obs_list should be of type ObsList'
        super(MultiBandObsList, self).append(obs_list)

    def get_s2n(self):
        """
        get the the simple s/n estimator

        sum(I)/sqrt( sum( 1/w ) ) = Isum/sqrt(Vsum)

        returns
        -------
        s2n: float
            The s/n of the images. Will retun -9999 if the s/n cannot be computed.
        """

        Isum, Vsum, Npix = self.get_s2n_sums()
        if Vsum > 0.0:
            s2n = Isum/np.sqrt(Vsum)
        else:
            s2n = -9999.0
        return s2n

    def get_s2n_sums(self):
        """
        get the sums for the simple s/n estimator

        sum(I)/sqrt( sum( 1/w ) ) = Isum/sqrt(Vsum)

        returns
        -------
        Isum: float
            The value sum(I).
        Vsum: float
            The value sum(1/w)
        Npix: int
            The number of non-zero-weight pixels.
        """

        Isum = 0.0
        Vsum = 0.0
        Npix = 0

        for obslist in self:
            tIsum, tVsum, tNpix = obslist.get_s2n_sums()
            Isum += tIsum
            Vsum += tVsum
            Npix += tNpix

        return Isum, Vsum, Npix

    def copy(self, memo=None):
        """
        copy all the data into a new MultiBandObsList
        """
        new_mbobs = MultiBandObsList(meta=copy.deepcopy(self._meta, memo=memo))

        for obslist in self:
            new_mbobs.append(obslist.copy(memo=memo))

        return new_mbobs

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        result = self.copy()
        memo[id(self)] = result
        return result

    def __eq__(self, mbobs):
        if len(self) != len(mbobs):
            return False

        for self_obslist, obslist in zip(self, mbobs):
            if self_obslist != obslist:
                return False
        return True

    def __setitem__(self, index, obs_list):
        """
        over-riding this for type safety
        """
        assert isinstance(obs_list, ObsList),\
            'obs_list should be of type ObsList'
        super(MultiBandObsList, self).__setitem__(index, obs_list)


def get_mb_obs(obs_in):
    """
    convert the input to a MultiBandObsList

    parameters
    ----------
    obs_in: ngmix.Observation, ngmix.ObsList, or ngmix.MultiBandObsList
        Input data to convert to a MultiBandObsList.

    returns
    -------
    mbobs: ngmix.MultiBandObsList
        A MultiBandObsList containing the input data.
    """

    if isinstance(obs_in, Observation):
        obs_list = ObsList()
        obs_list.append(obs_in)

        obs = MultiBandObsList()
        obs.append(obs_list)

    elif isinstance(obs_in, ObsList):
        obs = MultiBandObsList()
        obs.append(obs_in)

    elif isinstance(obs_in, MultiBandObsList):
        obs = obs_in

    else:
        raise ValueError(
            'obs should be Observation, ObsList, or MultiBandObsList'
        )

    return obs


#
# k space stuff
#


class KObservation(MetadataMixin):
    """
    a k-space observation

    parameters
    ----------
    kimage: galsim.Image
        A galsim image of the observation in k-space.
    weight: galsim.Image or None
        A real galsim image of the weight map. If None, the weights are all
        set to unity. Optional.
    psf: KObservation or None
        A KObservation of the PSF. If None, no PSF is set. Optional.
    meta: dict or None
        Any metadata keep in the `meta` attribute. Optional.
    """
    def __init__(self,
                 kimage,
                 weight=None,
                 psf=None,
                 meta=None):

        self._set_image(kimage)
        self._set_weight(weight)
        self.set_psf(psf)

        self._set_jacobian()

        self.set_meta(meta)

    def _set_image(self, kimage):
        """
        set the images, ensuring consistency
        """
        import galsim

        if not isinstance(kimage, galsim.Image):
            raise ValueError("kimage must be a galsim.Image")
        if kimage.array.dtype != np.complex128:
            raise ValueError("kimage must be complex")

        self.kimage = kimage

    def _set_weight(self, weight):
        """
        set the weight, ensuring consistency with
        the images
        """
        import galsim

        if weight is None:
            weight = self.kimage.real.copy()
            weight.setZero()
            weight.array[:, :] = 1.0

        else:
            assert isinstance(weight, galsim.Image)

            if weight.array.shape != self.kimage.array.shape:
                raise ValueError("weight kimage must have "
                                 "same shape as kimage")

        self.weight = weight

    @property
    def psf(self):
        """
        getter for psf

        currently this simply returns a reference
        """
        return self._psf

    def has_psf(self):
        """
        does this object have a psf set?
        """
        return hasattr(self, '_psf')

    def set_psf(self, psf):
        """
        set the psf KObservation.  can be None

        parameters
        ----------
        psf: KObservation or None
            The PSF as a KObservation. If not None, the shape of the psf image
            should match the observation image.
        """
        if self.has_psf():
            del self._psf

        if psf is None:
            return

        assert isinstance(psf, KObservation)

        self._psf = psf

        if psf.kimage.array.shape != self.kimage.array.shape:
            raise ValueError("psf kimage must have "
                             "same shape as kimage")
        assert np.allclose(psf.kimage.scale, self.kimage.scale)

    def _set_jacobian(self):
        """
        center is always at the canonical center.

        scale is always the scale of the image
        """

        scale = self.kimage.scale

        dims = self.kimage.array.shape
        cen = np.zeros(2)
        for i in range(2):
            if (dims[i] % 2) == 0:
                cen[i] = (dims[i]-1.0)/2.0 + 0.5
            else:
                cen[i] = (dims[i]-1.0)/2.0

        self.jacobian = DiagonalJacobian(
            scale=scale,
            row=cen[0],
            col=cen[1],
        )


class KObsList(list, MetadataMixin):
    """
    Hold a list of Observation objects

    This class provides a bit of type safety and ease of type checking

    parameters
    ----------
    meta: dict or None
        Any metadata keep in the `meta` attribute. Optional.
    """

    def __init__(self, meta=None):
        super(KObsList, self).__init__()

        self.set_meta(meta)

    def append(self, kobs):
        """
        Add a new KObservation

        over-riding this for type safety

        parameters
        ----------
        kobs: ngmix.KObservation
            A KObservation. An AssertionError will be raised if `kobs` is not
            an `ngmix.KObservation`.
        """
        assert isinstance(kobs, KObservation), \
            ("kobs should be of type "
             "KObservation, got %s" % type(kobs))

        super(KObsList, self).append(kobs)

    def __setitem__(self, index, kobs):
        """
        over-riding this for type safety
        """
        assert isinstance(kobs, KObservation),\
            'kobs should be of type KObservation'
        super(KObsList, self).__setitem__(index, kobs)


class KMultiBandObsList(list, MetadataMixin):
    """
    Hold a list of lists of ObsList objects, each representing a filter band

    This class provides a bit of type safety and ease of type checking

    parameters
    ----------
    meta: dict or None
        Any metadata keep in the `meta` attribute. Optional.
    """

    def __init__(self, meta=None):
        super(KMultiBandObsList, self).__init__()

        self.set_meta(meta)

    def append(self, kobs_list):
        """
        Add a new ObsList

        over-riding this for type safety

        parameters
        ----------
        kobs_list: ngmix.KObsList
            An KObsList. An AssertionError will be raised if `kobs_list` is not
            an `ngmix.KObsList`.
        """
        assert isinstance(kobs_list, KObsList), \
            "kobs_list should be of type KObsList"
        super(KMultiBandObsList, self).append(kobs_list)

    def __setitem__(self, index, kobs_list):
        """
        over-riding this for type safety
        """
        assert isinstance(kobs_list, KObsList), \
            "kobs_list should be of type KObsList"

        super(KMultiBandObsList, self).__setitem__(index, kobs_list)


def make_iilist(obs, interp=DEFAULT_XINTERP):
    """
    make a multi-band interpolated image list, as well as the maximum of
    getGoodImageSize from each psf, and corresponding dk

    parameters
    ----------
    obs: real space obs list
        Either Observation, ObsList or MultiBandObsList
    interp: string, optional
        The x interpolant, default 'lanczos15'

    returns
    -------
    mb_iilist: list of list of dicts
        A list of list of dictionaries containing the inteprolated image data
        for each observations. The entries are

            'wcs': the galsim WCS
            'scale': pixel-scale of the WCS
            'ii': the inteprolated image
            'weight': the weight map
            'meta': the metadata
            'psf_ii': the interpolated PSF image
            'psf_weight': the PSF weight map
            'psf_meta': the PSF metadata
            'realspace_gsimage': the galsim image data
    dim: int
        The maximum good image size over all PSFs in the data.
    dk: float
        The k-space spacing corresponding to dim.
    """
    import galsim

    mb_obs = get_mb_obs(obs)

    dimlist = []
    dklist = []

    mb_iilist = []

    for band, obs_list in enumerate(mb_obs):
        iilist = []
        for obs in obs_list:

            jac = obs.jacobian
            gsimage = galsim.Image(
                obs.image,
                wcs=jac.get_galsim_wcs(),
            )
            ii = galsim.InterpolatedImage(
                gsimage,
                x_interpolant=interp,
            )
            if hasattr(ii, 'SBProfile'):
                gsvers = 1
            else:
                gsvers = 2

            if obs.has_psf():
                psf_weight = obs.psf.weight

                # normalized
                psf_gsimage = galsim.Image(
                    obs.psf.image/obs.psf.image.sum(),
                    wcs=obs.psf.jacobian.get_galsim_wcs(),
                )

                psf_ii = galsim.InterpolatedImage(
                    psf_gsimage,
                    x_interpolant=interp,
                )
                # make dimensions odd
                if gsvers == 1:
                    dim = 1 + psf_ii.SBProfile.getGoodImageSize(
                        psf_ii.nyquistScale(),
                    )
                else:
                    dim = 1 + psf_ii.getGoodImageSize(
                        psf_ii.nyquist_scale
                    )
                psf_meta = obs.psf.meta

            else:
                # make dimensions odd
                if hasattr(ii, 'SBProfile'):
                    dim = 1 + ii.SBProfile.getGoodImageSize(
                        ii.nyquistScale(),
                    )
                else:
                    dim = 1 + ii.getGoodImageSize(
                        ii.nyquist_scale,
                    )
                psf_ii = None
                psf_weight = None
                psf_meta = None

            if gsvers == 1:
                dk = ii.stepK()
            else:
                dk = ii.stepk

            dimlist.append(dim)
            dklist.append(dk)

            iilist.append({
                'wcs': jac.get_galsim_wcs(),
                'scale': jac.scale,
                'ii': ii,
                'weight': obs.weight,
                'meta': obs.meta,
                'psf_ii': psf_ii,
                'psf_weight': psf_weight,
                'psf_meta': psf_meta,
                'realspace_gsimage': gsimage,
            })

        mb_iilist.append(iilist)

    dimarr = np.array(dimlist)
    dkarr = np.array(dklist)

    imax = dimarr.argmax()

    dim = dimarr[imax]
    dk = dkarr[imax]

    return mb_iilist, dim, dk


def make_kobs(mb_obs, interp=DEFAULT_XINTERP):
    """
    make k space observations from real space observations, with common
    dimensions and dk for each band and epoch

    parameters
    ----------
    obs: real space data
        Either Observation, ObsList or MultiBandObsList
    interp: string, optional
        The x interpolant, default 'lanczos15'

    returns
    -------
    mb_kobs: KMultiBandObsList
        The k-space data.
    """

    mb_iilist, dim, dk = make_iilist(mb_obs, interp=interp)

    mb_kobs = KMultiBandObsList()

    for iilist in mb_iilist:

        kobs_list = KObsList()
        for iidict in iilist:

            kimage = iidict['ii'].drawKImage(
                nx=dim,
                ny=dim,
                scale=dk,
            )

            # need a better way to deal with weights, chi^2 etc.
            weight = kimage.real.copy()
            useweight = iidict['weight'].max()
            weight.array[:, :] = 0.5*useweight

            # parseval's theorem
            weight *= (1.0/weight.array.size)

            if iidict['psf_ii'] is not None:
                psf_kimage = iidict['psf_ii'].drawKImage(
                    nx=dim,
                    ny=dim,
                    scale=dk,
                )

                psf_useweight = iidict['psf_weight'].max()
                psf_weight = psf_kimage.real.copy()
                psf_weight.array[:, :] = 0.5*psf_useweight

                # parseval's theorem
                psf_weight *= (1.0/psf_weight.array.size)

                psf_meta = {}
                psf_meta.update(iidict['psf_meta'])
                psf_meta['ii'] = iidict['psf_ii']
                psf_kobs = KObservation(
                    psf_kimage,
                    weight=psf_weight,
                    meta=psf_meta,
                )
            else:
                psf_kobs = None

            meta = iidict['meta']
            meta['realspace_gsimage'] = iidict['realspace_gsimage']
            meta['scale'] = iidict['scale']
            kobs = KObservation(
                kimage,
                weight=weight,
                psf=psf_kobs,
                meta=meta,
            )

            kobs_list.append(kobs)

        mb_kobs.append(kobs_list)

    return mb_kobs


def get_kmb_obs(obs_in):
    """
    convert the input to a KMultiBandObsList

    parameters
    ----------
    obs_in: ngmix.KObservation, ngmix.KObsList, or ngmix.KMultiBandObsList
        Input data to convert to a KMultiBandObsList.

    returns
    -------
    kmb_obs: ngmix.KMultiBandObsList
        A KMultiBandObsList containing the input data.
    """

    if isinstance(obs_in, KObservation):
        obs_list = KObsList()
        obs_list.append(obs_in)

        obs = KMultiBandObsList()
        obs.append(obs_list)
    elif isinstance(obs_in, KObsList):
        obs = KMultiBandObsList()
        obs.append(obs_in)
    elif isinstance(obs_in, KMultiBandObsList):
        obs = obs_in
    else:
        raise ValueError("obs should be KObservation, "
                         "KObsList, or KMultiBandObsList")

    return obs
