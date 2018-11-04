import numpy as np
import meds

from .observation import MultiBandObsList, Observation, ObsList
from .jacobian import Jacobian


class MultiBandMEDS(object):
    """Interface to MEDS objects in more than one band for ngmix.

    Parameters
    ----------
    mlist : list of str
        List of paths to the MEDS files.

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
    def size(self):
        """Number of entries in the catalog.
        """
        return self.mlist[0].size

    def get_mbobs_list(self, indices=None, weight_type='weight'):
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
            Default is 'weight'

        Returns
        -------
        mbobs_list : list of ngmix.MultiBandObsList
            The list of `MultiBandObsList`s for the requested objects.
        """
        if indices is None:
            indices = np.arange(self.mlist[0].size)

        list_of_obs = []
        for iobj in indices:
            mbobs = self.get_mbobs(iobj, weight_type=weight_type)
            list_of_obs.append(mbobs)

        return list_of_obs

    def get_mbobs(self, iobj, weight_type='weight'):
        """Get a `MultiBandObsList` for a given object.

        Parameters
        ----------
        iobj : int
            Index of the object.
        weight_type: string, optional
            Weight type. can be one of
                'weight': the actual weight map
                'uberseg': uberseg modified weight map
            Default is 'weight'

        Returns
        -------
        mbobs : ngmix.MultiBandObsList
            A `MultiBandObsList` holding all of the observations for this
            object.
        """
        mbobs = MultiBandObsList()

        for m in self.mlist:
            obslist = m.get_obslist(iobj, weight_type=weight_type)
            mbobs.append(obslist)

        return mbobs


class MEDS(meds.MEDS):
    def get_obslist(self, iobj, weight_type='weight'):
        """Get an ngmix ObsList for all observations.

        Parameters
        ----------
        iobj : int
            Index of the object.
        weight_type: string, optional
            Weight type. can be one of
                'weight': the actual weight map
                'uberseg': uberseg modified weight map
            Default is 'weight'

        Returns
        -------
        obslist : ngmix.ObsList
            An `ObsList` of all observations.
        """
        obslist = ObsList()
        for icut in range(self._cat['ncutout'][iobj]):
            obs = self.get_obs(iobj, icut, weight_type=weight_type)
            obslist.append(obs)

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

    def get_obs(self, iobj, icutout, weight_type='weight'):
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
            Default is 'weight'

        Returns
        -------
        obs : ngmix.Observation
            An `Observation` for this cutout.
        """
        im = self.get_cutout(iobj, icutout, type='image')
        bmask = self.get_cutout(iobj, icutout, type='bmask')

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

        meta = dict(
            id=c['id'][iobj],
            index=iobj,
            icut=icutout,
            cutout_index=icutout,
            file_id=c['file_id'][iobj, icutout],
            orig_row=c['orig_row'][iobj, icutout],
            orig_col=c['orig_col'][iobj, icutout],
            orig_start_row=c['orig_start_row'][iobj, icutout],
            orig_start_col=c['orig_start_col'][iobj, icutout])

        if 'flux_auto' in c.dtype.names:
            meta['flux'] = c['flux_auto'][iobj]
        if 'x2' in c.dtype.names and 'y2' in c.dtype.names:
            meta['T'] = c['x2'][iobj] + c['y2'][iobj]
        if 'number' in c.dtype.names:
            meta['number'] = c['number'][iobj]

        psf_obs = self.get_psf_obs(iobj, icutout)
        obs = Observation(
            im,
            weight=wt,
            bmask=bmask,
            meta=meta,
            jacobian=jacobian,
            psf=psf_obs)

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

        cen = (np.array(psf_im.shape)-1.0) / 2.0
        jacobian.set_cen(
            row=cen[0],
            col=cen[1])

        return Observation(
            psf_im,
            weight=weight,
            jacobian=jacobian)
