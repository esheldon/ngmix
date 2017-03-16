import numpy
from .jacobian import Jacobian, UnitJacobian, DiagonalJacobian
from .gmix import GMix
import copy

DEFAULT_XINTERP='lanczos15'

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


#
# k space stuff
#


class KObservation(object):
    def __init__(self,
                 kimage,
                 weight=None,
                 psf=None,
                 meta=None):

        self._set_image(kimage)
        self._set_weight(weight)
        self.set_psf(psf)

        self._set_jacobian()

        self.meta={}
        if meta is not None:
            self.update_meta_data(meta)

    def _set_image(self, kimage):
        """
        set the images, ensuring consistency
        """
        import galsim

        if not isinstance(kimage,galsim.Image):
            raise ValueError("kimage must be a galsim.Image")
        if kimage.array.dtype != numpy.complex128:
            raise ValueError("kimage must be complex")

        self.kimage=kimage

    def _set_weight(self, weight):
        """
        set the weight, ensuring consistency with
        the images
        """
        import galsim

        if weight is None:
            weight = self.kimage.real.copy()
            weight.setZero()
            weight.array[:,:] = 1.0

        else:
            assert isinstance(weight, galsim.Image)

            if weight.array.shape!=self.kimage.array.shape:
                raise ValueError("weight kimage must have "
                                 "same shape as kimage")

        self.weight=weight

    def set_psf(self, psf):
        """
        set the psf KObservation.  can be None

        Shape of psf image should match the image
        """
        self.psf = psf
        if psf is None:
            return

        assert isinstance(psf, KObservation)

        if psf.kimage.array.shape!=self.kimage.array.shape:
            raise ValueError("psf kimage must have "
                             "same shape as kimage")
        assert numpy.allclose(psf.kimage.scale,self.kimage.scale)

    def _set_jacobian(self):
        """
        center is always at the canonical center.

        scale is always the scale of the image
        """

        scale=self.kimage.scale

        dims=self.kimage.array.shape
        if (dims[0] % 2) == 0:
            cen = (numpy.array(dims)-1.0)/2.0 + 0.5
        else:
            cen = (numpy.array(dims)-1.0)/2.0

        self.jacobian = DiagonalJacobian(
            scale=scale,
            row=cen[0],
            col=cen[1],
        )

    def update_meta_data(self, meta):
        """
        Add some metadata
        """

        if not isinstance(meta,dict):
            raise TypeError("meta data must be in dictionary form")
        self.meta.update(meta)


class KObsList(list):
    """
    Hold a list of Observation objects

    This class provides a bit of type safety and ease of type checking
    """

    def __init__(self, meta=None):
        super(KObsList,self).__init__()

        self.meta={}
        if meta is not None:
            self.update_meta_data(meta)

    def append(self, kobs):
        """
        Add a new KObservation

        over-riding this for type safety
        """
        assert isinstance(kobs,KObservation),\
                "kobs should be of type KObservation, got %s" % type(kobs)

        super(KObsList,self).append(kobs)

    def update_meta_data(self, meta):
        """
        Add some metadata
        """

        if not isinstance(meta,dict):
            raise TypeError("meta data must be in dictionary form")
        self.meta.update(meta)

    def __setitem__(self, index, kobs):
        """
        over-riding this for type safety
        """
        assert isinstance(kobs,KObservation),"kobs should be of type KObservation"
        super(KObsList,self).__setitem__(index, kobs)



class KMultiBandObsList(list):
    """
    Hold a list of lists of ObsList objects, each representing a filter
    band

    This class provides a bit of type safety and ease of type checking
    """

    def __init__(self, meta=None):
        super(KMultiBandObsList,self).__init__()

        self.meta={}
        if meta is not None:
            self.update_meta_data(meta)

    def append(self, kobs_list):
        """
        Add a new ObsList

        over-riding this for type safety
        """
        assert isinstance(kobs_list,KObsList),\
                "kobs_list should be of type KObsList"
        super(KMultiBandObsList,self).append(kobs_list)

    def update_meta_data(self, meta):
        """
        Add some metadata
        """

        if not isinstance(meta,dict):
            raise TypeError("meta data must be in dictionary form")
        self.meta.update(meta)

    def __setitem__(self, index, kobs_list):
        """
        over-riding this for type safety
        """
        assert isinstance(kobs_list,KObsList),\
                "kobs_list should be of type KObsList"
        super(KMultiBandObsList,self).__setitem__(index, kobs_list)



def make_iilist(obs, **kw):
    """
    make a multi-band interpolated image list, as well as the maximum of
    getGoodImageSize from each psf, and corresponding dk

    parameters
    ----------
    obs: real space obs list
        Either Observation, ObsList or MultiBandObsList
    interp: string, optional
        The x interpolant, default 'lanczos15'
    """
    import galsim

    wmult=1.0

    interp=kw.get('interp',DEFAULT_XINTERP)
    mb_obs = get_mb_obs(obs)

    dimlist=[]
    dklist=[]

    mb_iilist=[]
    for band,obs_list in enumerate(mb_obs):
        iilist=[]
        for obs in obs_list:

            gsimage = galsim.Image(
                obs.image,
                wcs=obs.jacobian.get_galsim_wcs(),
            )
            ii = galsim.InterpolatedImage(
                gsimage,
                x_interpolant=interp,
            )

            if obs.has_psf():
                psf_weight = obs.psf.weight,

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
                dim = 1 + psf_ii.SBProfile.getGoodImageSize(
                    psf_ii.nyquistScale(),
                    #wmult,
                )

            else:
                # make dimensions odd
                dim = 1 + ii.SBProfile.getGoodImageSize(
                    ii.nyquistScale(),
                    #wmult,
                )
                psf_ii=None
                psf_weight=None

            dk=ii.stepK()

            dimlist.append( dim )
            dklist.append(dk)

            iilist.append({
                'wcs':obs.jacobian.get_galsim_wcs(),
                'ii':ii,
                'weight':obs.weight,
                'psf_ii':psf_ii,
                'psf_weight':psf_weight,
            })

        mb_iilist.append(iilist)

    dimarr = numpy.array(dimlist)
    dkarr = numpy.array(dklist)

    imax = dimarr.argmax()

    dim=dimarr[imax]
    dk=dkarr[imax]

    return mb_iilist, dim, dk


def make_kobs(mb_obs, **kw):
    """
    make k space observations from real space observations, with common
    dimensions and dk for each band and epoch

    parameters
    ----------
    obs: real space obs list
        Either Observation, ObsList or MultiBandObsList
    interp: string, optional
        The x interpolant, default 'lanczos15'
    """
    import galsim

    mb_iilist, dim, dk = make_iilist(mb_obs, **kw)

    mb_kobs = KMultiBandObsList()

    for iilist in mb_iilist:

        kobs_list=KObsList()
        for iidict in iilist:

            kimage = iidict['ii'].drawKImage(
                nx=dim,
                ny=dim,
                scale=dk,
            )

            # need a better way to deal with weights, chi^2 etc.
            weight = kimage.real.copy()
            medweight = numpy.median(iidict['weight'])
            weight.array[:,:] = 0.5*medweight

            # parseval's theorem
            weight *= (1.0/weight.array.size)

            if iidict['psf_ii'] is not None:
                psf_kimage = iidict['psf_ii'].drawKImage(
                    nx=dim,
                    ny=dim,
                    scale=dk,
                )

                psf_medweight = numpy.median(iidict['psf_weight'])
                psf_weight = psf_kimage.real.copy()
                psf_weight.array[:,:] = 0.5*psf_medweight

                psf_kobs = KObservation(
                    psf_kimage,
                    weight=psf_weight,
                )
            else:
                psf_kobs=None

            kobs = KObservation(
                kimage,
                weight=weight,
                psf=psf_kobs,
            )

            kobs_list.append(kobs)

        mb_kobs.append(kobs_list)

    return mb_kobs

def get_kmb_obs(obs_in):
    """
    convert the input to a MultiBandObsList

    Input should be an KObservation, KObsList, or KMultiBandObsList
    """

    if isinstance(obs_in,KObservation):
        obs_list=KObsList()
        obs_list.append(obs_in)

        obs=KMultiBandObsList()
        obs.append(obs_list)
    elif isinstance(obs_in,KObsList):
        obs=KMultiBandObsList()
        obs.append(obs_in)
    elif isinstance(obs_in,KMultiBandObsList):
        obs=obs_in
    else:
        raise ValueError("obs should be KObservation, "
                         "KObsList, or KMultiBandObsList")

    return obs


