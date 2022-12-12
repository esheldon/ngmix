import os
import logging
import numpy as np
from meds import MEDS as _MEDS

from .observation import MultiBandObsList, Observation, ObsList
from .jacobian import Jacobian
from .gexceptions import GMixFatalError

logger = logging.getLogger(__name__)


class MultiBandNGMixMEDS(object):
    """Interface to NGMixMEDS objects in more than one band.

    Parameters
    ----------
    mlist : list of `ngmix.medsreaders.NGMixMEDS` objects
        List of the `NGMixMEDS` objects for each band.

    Attributes
    ----------
    size : int
        The number of objects in the MEDS files.

    Methods
    -------
    get_mbobs_list(indices=None, weight_type='weight')
        Get a list of `MultiBandObsList` for all or a set of objects.
    get_mbobs(iobj, weight_type='weight')
        Get a `MultiBandObsList` for a given object.
    """
    def __init__(self, mlist):
        self.mlist = mlist

    @property
    def nband(self):
        """
        number of bands
        """
        return len(self.mlist)

    @property
    def size(self):
        """Number of entries in the catalog.
        """
        return self.mlist[0].size

    def get_mbobs_list(self, indices=None, weight_type='weight', seg_type='seg'):
        """Get a list of `MultiBandObsList` for all or a set of objects.

        Parameters
        ----------
        indices : array-like, optional
            The indices of the objects to return. Default of `None` returns
            all objects.
        weight_type: string, optional
            Weight type. can be one of
                'weight': the actual weight map
                'uberseg': uberseg modified weight map
                'cweight': weight map zeroed outside the object's seg map
                'cseg': weight map zeroed outside of circular aperture that
                    doesn't touch any other object.
                'cseg-canonical': same as 'cseg' but uses the postage stamp
                    center instead of the object's position.
            Default is 'weight'
        seg_type: string, optional
            If 'seg' use the seg for each epoch, if 'cseg' use the coadd seg
            composited onto the single epoch frame

        Returns
        -------
        mbobs_list : list of ngmix.MultiBandObsList
            The list of `MultiBandObsList`s for the requested objects.
        """
        if indices is None:
            indices = np.arange(self.mlist[0].size)

        list_of_obs = []
        for iobj in indices:
            mbobs = self.get_mbobs(iobj, weight_type=weight_type, seg_type=seg_type)
            list_of_obs.append(mbobs)

        return list_of_obs

    def get_mbobs(self, iobj, weight_type='weight', seg_type='seg'):
        """Get a `MultiBandObsList` for a given object.

        Parameters
        ----------
        iobj : int
            Index of the object.
        weight_type: string, optional
            Weight type. can be one of
                'weight': the actual weight map
                'uberseg': uberseg modified weight map
                'cweight': weight map zeroed outside the object's seg map
                'cseg': weight map zeroed outside of circular aperture that
                    doesn't touch any other object.
                'cseg-canonical': same as 'cseg' but uses the postage stamp
                    center instead of the object's position.
            Default is 'weight'
        seg_type: string, optional
            If 'seg' use the seg for each epoch, if 'cseg' use the coadd seg
            composited onto the single epoch frame

        Returns
        -------
        mbobs : ngmix.MultiBandObsList
            A `MultiBandObsList` holding all of the observations for this
            object.
        """
        mbobs = MultiBandObsList()

        for m in self.mlist:
            obslist = m.get_obslist(iobj, weight_type=weight_type, seg_type=seg_type)
            mbobs.append(obslist)

        return mbobs


class NGMixMEDS(_MEDS):
    def get_obslist(self, iobj, weight_type='weight', seg_type='seg'):
        """Get an ngmix ObsList for all observations.

        Parameters
        ----------
        iobj : int
            Index of the object.
        weight_type: string, optional
            Weight type. can be one of
                'weight': the actual weight map
                'uberseg': uberseg modified weight map
                'cweight': weight map zeroed outside the object's seg map
                'cseg': weight map zeroed outside of circular aperture that
                    doesn't touch any other object.
                'cseg-canonical': same as 'cseg' but uses the postage stamp
                    center instead of the object's position.
            Default is 'weight'
        seg_type: string, optional
            If 'seg' use the seg for each epoch, if 'cseg' use the coadd seg
            composited onto the single epoch frame

        Returns
        -------
        obslist : ngmix.ObsList
            An `ObsList` of all observations.
        """
        obslist = ObsList()
        for icut in range(self._cat['ncutout'][iobj]):
            try:
                obs = self.get_obs(
                    iobj, icut, weight_type=weight_type, seg_type=seg_type,
                )
                obslist.append(obs)
            except GMixFatalError:
                logger.debug('zero weight observation found, skipping')

        if len(obslist) > 0:
            obs = obslist[0]
            if 'flux' in obs.meta:
                obslist.meta['flux'] = obs.meta['flux']
            if 'T' in obs.meta:
                obslist.meta['T'] = obs.meta['T']
        return obslist

    def get_ngmix_jacobian(self, iobj, icutout):
        """Get an ngmix.Jacobian representation.

        Parameters
        ----------
        iobj : int
            Index of the object.
        icutout : int
            Index of the cutout for this object.

        Returns
        -------
        jacob : ngmix.Jacobian
            The `Jacobian` for the cutout.
        """
        jd = self.get_jacobian(iobj, icutout)
        return Jacobian(
            row=jd['row0'],
            col=jd['col0'],
            dudrow=jd['dudrow'],
            dudcol=jd['dudcol'],
            dvdrow=jd['dvdrow'],
            dvdcol=jd['dvdcol'])

    def get_obs(self, iobj, icutout, weight_type='weight', seg_type='seg'):
        """Get an ngmix Observation.

        Parameters
        ----------
        iobj : int
            Index of the object.
        icutout : int
            Index of the cutout for this object.
        weight_type: string, optional
            Weight type. can be one of
                'weight': the actual weight map
                'uberseg': uberseg modified weight map
                'cweight': weight map zeroed outside the object's seg map
                'cseg': weight map zeroed outside of circular aperture that
                    doesn't touch any other object.
                'cseg-canonical': same as 'cseg' but uses the postage stamp
                    center instead of the object's position.
            Default is 'weight'
        seg_type: string, optional
            If 'seg' use the seg for each epoch, if 'cseg' use the coadd seg
            composited onto the single epoch frame

        Returns
        -------
        obs : ngmix.Observation
            An `Observation` for this cutout.
        """
        im = self.get_cutout(iobj, icutout, type='image')

        try:
            bmask = self.get_cutout(iobj, icutout, type='bmask')
        except Exception:
            bmask = None

        try:
            ormask = self.get_cutout(iobj, icutout, type='ormask')
        except Exception:
            ormask = None

        try:
            noise = self.get_cutout(iobj, icutout, type='noise')
        except Exception:
            noise = None

        try:
            mfrac = self.get_cutout(iobj, icutout, type='mfrac')
        except Exception:
            mfrac = None

        try:
            if seg_type == 'seg':
                seg = self.get_cutout(iobj, icutout, type='seg')
            elif seg_type == 'coadd':
                seg = self.interpolate_coadd_seg(iobj, icutout)
            else:
                raise ValueError(f'bad seg_type "{seg_type}"')
        except Exception:
            seg = None

        if weight_type == 'uberseg':
            wt = self.get_uberseg(iobj, icutout)
        elif weight_type == 'cweight':
            wt = self.get_cweight_cutout(iobj, icutout, restrict_to_seg=True)
        elif weight_type == 'weight':
            wt = self.get_cutout(iobj, icutout, type='weight')
        elif weight_type == 'cseg':
            wt = self.get_cseg_weight(iobj, icutout)
        elif weight_type == 'cseg-canonical':
            wt = self.get_cseg_weight(iobj, icutout, use_canonical_cen=True)
        else:
            raise ValueError("bad weight type '%s'" % weight_type)

        jacobian = self.get_ngmix_jacobian(iobj, icutout)
        c = self._cat

        ii = self.get_image_info()
        file_id = c['file_id'][iobj, icutout]
        file_path = os.path.basename(ii['image_path'][file_id]).strip()

        meta = dict(
            id=c['id'][iobj],
            index=iobj,
            icut=icutout,
            cutout_index=icutout,
            file_id=file_id,
            file_path=file_path,
            orig_row=c['orig_row'][iobj, icutout],
            orig_col=c['orig_col'][iobj, icutout],
            orig_start_row=c['orig_start_row'][iobj, icutout],
            orig_start_col=c['orig_start_col'][iobj, icutout],
            scale=ii['scale'][file_id],
        )

        if 'flux_auto' in c.dtype.names:
            meta['flux'] = c['flux_auto'][iobj]
        if 'x2' in c.dtype.names and 'y2' in c.dtype.names:
            meta['T'] = c['x2'][iobj] + c['y2'][iobj]
        if 'number' in c.dtype.names:
            meta['number'] = c['number'][iobj]

        if self.has_psf():
            psf_obs = self.get_psf_obs(iobj, icutout)
        else:
            psf_obs = None

        obs = Observation(
            im,
            weight=wt,
            bmask=bmask,
            ormask=ormask,
            seg=seg,
            noise=noise,
            meta=meta,
            jacobian=jacobian,
            psf=psf_obs,
            mfrac=mfrac,
        )

        return obs

    def get_psf_obs(self, iobj, icutout):
        """Get an observation of the PSF for this object.

        Parameters
        ----------
        iobj : int
            Index of the object.
        icutout : int
            Index of the cutout for this object.

        Returns
        -------
        psf : ngmix.Observation
            The PSF `Observation`.
        """
        psf_im = self.get_psf(iobj, icutout)

        # FIXME: fake the noise
        noise = psf_im.max() / 1000.0
        weight = psf_im*0 + 1.0/noise**2
        jacobian = self.get_ngmix_jacobian(iobj, icutout)

        row, col = self._get_psf_cen(iobj, icutout)
        jacobian.set_cen(
            row=row,
            col=col,
        )

        return Observation(
            psf_im,
            weight=weight,
            jacobian=jacobian,
        )

    def _get_psf_cen(self, iobj, icutout):
        """
        get the center row,col of the psf
        """
        c = self._cat

        row = c['psf_cutout_row'][iobj, icutout]
        col = c['psf_cutout_col'][iobj, icutout]
        return row, col
