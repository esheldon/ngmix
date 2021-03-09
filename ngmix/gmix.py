import numpy
from numpy import (
    array,
    zeros,
    sqrt,
    diag,
    isfinite,
)
from .jacobian import Jacobian, UnitJacobian
from .shape import Shape, e1e2_to_g1g2

from . import moments

from .gexceptions import GMixFatalError

from . import gmix_nb
from .gmix_nb import (
    _gmix_fill_functions,
    gmix_set_norms,
    gmix_convolve_fill,
    get_cm_Tfactor,
)
from .fitting_nb import (
    get_loglike,
    fill_fdiff,
    get_model_s2n_sum,
)

from .render_nb import render
from .pixels import make_coords


def make_gmix_model(pars, model, **kw):
    """
    get a gaussian mixture model for the given model
    """

    model = get_model_num(model)
    if model == GMIX_COELLIP:
        return GMixCoellip(pars)
    elif model == GMIX_FULL:
        return GMix(pars=pars)
    else:
        return GMixModel(pars, model)


class GMix(object):
    """
    A general two-dimensional gaussian mixture.

    parameters
    ----------
    Send either ngauss= or pars=

    ngauss: number, optional
        number of gaussians.  data will be zeroed
    pars: array-like, optional
        6*ngauss elements to fill the gaussian mixture.

    methods
    -------
    copy(self):
        make a new copy of this GMix
    convolve(psf):
        Get a new GMix that is the convolution of the GMix with the input psf
    get_T():
        get T=sum(p*T_i)/sum(p)
    get_sigma():
        get sigma=sqrt(T/2)
    get_psum():
        get sum(p)
    set_psum(psum):
        set new overall sum(p)
    get_cen():
        get cen=sum(p*cen_i)/sum(p)
    set_cen(row,col):
        set the overall center to the input.
    """

    def __init__(self, ngauss=None, pars=None):

        self._model = GMIX_FULL
        self._model_name = "full"
        self._set_fill_func()

        if ngauss is None and pars is None:
            raise GMixFatalError("send ngauss= or pars=")

        if pars is not None:
            npars = len(pars)
            if (npars % 6) != 0:
                raise GMixFatalError(
                    "len(pars) must be mutiple of 6 " "got %s" % npars
                )
            self._ngauss = npars // 6
            self._npars = npars
            self.reset()
            self._fill(pars)
        else:
            self._ngauss = ngauss
            self._npars = 6 * ngauss
            self.reset()

    def get_data(self):
        """
        Get the underlying array
        """
        return self._data

    def get_full_pars(self):
        """
        Get a full parameter description.
           [p1,row1,col1,irr1,irc1,icc1,
            p2,row2,col2,irr2,irc2,icc2,
            ...
           ]

        """

        gm = self.get_data()

        n = self._ngauss
        pars = numpy.zeros(n * 6)
        beg = 0
        for i in range(n):
            pars[beg + 0] = gm["p"][i]
            pars[beg + 1] = gm["row"][i]
            pars[beg + 2] = gm["col"][i]
            pars[beg + 3] = gm["irr"][i]
            pars[beg + 4] = gm["irc"][i]
            pars[beg + 5] = gm["icc"][i]

            beg += 6
        return pars

    def get_cen(self):
        """
        get the center position (row,col)
        """

        gm = self.get_data()
        psum = gm["p"].sum()
        rowsum = (gm["row"] * gm["p"]).sum()
        colsum = (gm["col"] * gm["p"]).sum()

        row = rowsum / psum
        col = colsum / psum

        return row, col

    def set_cen(self, row, col):
        """
        Move the mixture to a new center
        """
        gm = self.get_data()

        row0, col0 = self.get_cen()

        row_shift = row - row0
        col_shift = col - col0

        gm["row"] += row_shift
        gm["col"] += col_shift

    def get_T(self):
        """
        get weighted average T sum(p*T)/sum(p)
        """

        gm = self.get_data()

        row, col = self.get_cen()

        rowdiff = gm["row"] - row
        coldiff = gm["col"] - col

        p = gm["p"]
        ipsum = 1.0 / p.sum()

        irr = ((gm["irr"] + rowdiff ** 2) * p).sum() * ipsum
        icc = ((gm["icc"] + coldiff ** 2) * p).sum() * ipsum

        T = irr + icc

        return T

    def get_sigma(self):
        """
        get sigma=sqrt(T/2)
        """
        T = self.get_T()
        return sqrt(T / 2.0)

    def get_e1e2T(self):
        """
        Get e1,e2 and T for the total gaussian mixture.
        """

        gm = self.get_data()

        row, col = self.get_cen()

        rowdiff = gm["row"] - row
        coldiff = gm["col"] - col

        p = gm["p"]
        ipsum = 1.0 / p.sum()

        irr = ((gm["irr"] + rowdiff ** 2) * p).sum() * ipsum
        irc = ((gm["irc"] + rowdiff * coldiff) * p).sum() * ipsum
        icc = ((gm["icc"] + coldiff ** 2) * p).sum() * ipsum

        T = irr + icc

        e1 = (icc - irr) / T
        e2 = 2.0 * irc / T
        return e1, e2, T

    def get_g1g2T(self):
        """
        Get g1,g2 and T for the total gaussian mixture.
        """
        e1, e2, T = self.get_e1e2T()
        g1, g2 = e1e2_to_g1g2(e1, e2)
        return g1, g2, T

    def get_e1e2sigma(self):
        """
        Get e1,e2 and sigma for the total gmix.

        Warning: only really works if the centers are the same
        """

        e1, e2, T = self.get_e1e2T()
        sigma = sqrt(T / 2)
        return e1, e2, sigma

    def get_g1g2sigma(self):
        """
        Get g1,g2 and sigma for the total gmix.

        Warning: only really works if the centers are the same
        """
        e1, e2, T = self.get_e1e2T()
        g1, g2 = e1e2_to_g1g2(e1, e2)

        sigma = sqrt(T / 2)
        return g1, g2, sigma

    def get_flux(self):
        """
        get sum(p)
        """
        gm = self.get_data()
        return gm["p"].sum()

    # alias
    get_psum = get_flux

    def set_flux(self, psum):
        """
        set a new value for sum(p)
        """
        gm = self.get_data()

        psum0 = gm["p"].sum()
        rat = psum / psum0
        gm["p"] *= rat

        # we will need to reset the pnorm values
        gm["norm_set"] = 0

    # alias
    set_psum = set_flux

    def get_gaussap_flux(self, fwhm=None, sigma=None, T=None):
        """
        get a gaussian aperture weight flux for the mixture

        parameters
        ----------
        fwhm or sigma or T: float
            one must be sent for the gaussian weight

        Returns
        -------
        Gaussian aperture weighted flux
        """
        from numpy import linalg

        if fwhm is not None:
            sigma = moments.fwhm_to_sigma(fwhm)
        elif T is not None:
            sigma = sqrt(T / 2.0)
        elif sigma is not None:
            sigma = float(sigma)
        else:
            raise ValueError("send weight function sigma, fwhm, or T")

        sigma2 = sigma ** 2

        wt_mat = array([[sigma2, 0.0], [0.0, sigma2]],)
        wt_mat_inv = linalg.inv(wt_mat)

        gm = self.get_data()

        apflux = 0.0
        for i in range(gm.size):
            gauss = gm[i]
            det = gauss["det"]

            pval = gauss["p"]

            fac = 1.0
            if det > gmix_nb.GMIX_LOW_DETVAL:

                mat = array(
                    [
                        [gauss["irr"], gauss["irc"]],
                        [gauss["irc"], gauss["icc"]],
                    ],
                )

                try:

                    mat_inv = linalg.inv(mat)

                    invsum = mat_inv + wt_mat_inv

                    newmat = linalg.inv(invsum)
                    newdet = linalg.det(newmat)

                    fac = sqrt(newdet / det)
                    if fac > 1:
                        fac = 1

                except linalg.LinAlgError:
                    # use default fac of 1
                    pass

            apflux += pval * fac

        return apflux

    def set_norms(self):
        """
        Needed to actually evaluate the gaussian.  This is done internally
        by the c code so if all goes well you don't need to call this
        """
        gm = self.get_data()
        gmix_set_norms(gm)

    def set_norms_if_needed(self):
        """
        Needed to actually evaluate the gaussian.  This is done internally
        by the c code so if all goes well you don't need to call this
        """
        gm = self.get_data()
        if gm["norm_set"][0] == 0:
            gmix_set_norms(gm)

    def fill(self, pars):
        """
        Fill in the gaussian mixture with new parameters

        parameters
        ----------
        pars: ndarray or sequence
            The parameters
        """

        npars = len(pars)
        if npars != self._npars:
            err = "model '%s' requires %s pars, got %s"
            err = err % (self._model_name, self._npars, npars)
            raise GMixFatalError(err)

        self._fill(pars)

    def _fill(self, pars):
        """
        Fill in the gaussian mixture with new parameters, without
        error checking

        parameters
        ----------
        pars: ndarray or sequence
            The parameters
        """

        self._pars[:] = pars

        gm = self.get_data()
        self._fill_func(
            gm, self._pars,
        )

    def copy(self):
        """
        Get a new GMix with the same parameters
        """
        gmix = GMix(ngauss=self._ngauss)
        gmix._data[:] = self._data[:]
        return gmix

    def get_sheared(self, s1, s2=None):
        """
        Get a sheared version of the gaussian mixture

        call as either
            gmnew = gm.get_sheared(shape)
        or
            gmnew = gm.get_sheared(g1,g2)
        """
        if isinstance(s1, Shape):
            shear1 = s1.g1
            shear2 = s1.g2
        elif s2 is not None:
            shear1 = s1
            shear2 = s2
        else:
            raise RuntimeError("send a Shape or s1,s2")

        new_gmix = self.copy()

        ndata = new_gmix.get_data()
        ndata["norm_set"] = 0

        for i in range(len(self)):
            irr = ndata["irr"][i]
            irc = ndata["irc"][i]
            icc = ndata["icc"][i]

            irr_s, irc_s, icc_s = moments.get_sheared_moments(
                irr, irc, icc, shear1, shear2
            )

            det = irr_s * icc_s - irc_s * irc_s
            ndata["irr"][i] = irr_s
            ndata["irc"][i] = irc_s
            ndata["icc"][i] = icc_s
            ndata["det"][i] = det

        return new_gmix

    def convolve(self, psf):
        """
        Get a new GMix that is the convolution of the GMix with the input psf

        parameters
        ----------
        psf: GMix object
        """
        if not isinstance(psf, GMix):
            raise TypeError(
                "Can only convolve with another GMix "
                " got type %s" % type(psf)
            )

        ng = len(self) * len(psf)
        output = GMix(ngauss=ng)

        odata = output.get_data()
        gm = self.get_data()
        gmpsf = psf.get_data()

        gmix_convolve_fill(odata, gm, gmpsf)

        return output

    def make_image(self, dims, jacobian=None, fast_exp=False):
        """
        Render the mixture into a new image

        parameters
        ----------
        dims: 2-element sequence
            dimensions [nrows, ncols]
        fast_exp: bool, optional
            use fast, approximate exp function
        """

        dims = numpy.array(dims, ndmin=1, dtype="i8")
        if dims.size != 2:
            raise ValueError(
                "images must have two dimensions, " "got %s" % str(dims)
            )

        image = numpy.zeros(dims, dtype="f8")
        self._fill_image(image, jacobian=jacobian, fast_exp=fast_exp)
        return image

    def make_round(self, preserve_size=False):
        """
        make a round version of the mixture

        The transformation is performed as if a shear were applied,
        so

            Tround = T * (1-g^2) / (1+g^2)

        The center of all gaussians is set to the common mean

        returns
        -------
        New round gmix
        """
        # raise RuntimeError("fix round")
        from . import shape

        gm = self.copy()

        if preserve_size:
            # make sure the psf is isotropically at least as big as the largest
            # extent

            e1, e2, T = gm.get_e1e2T()

            irr, irc, icc = moments.e2mom(e1, e2, T)

            mat = numpy.zeros((2, 2))
            mat[0, 0] = irr
            mat[0, 1] = irc
            mat[1, 0] = irc
            mat[1, 1] = icc

            eigs = numpy.linalg.eigvals(mat)

            factor = eigs.max() / (T / 2.0)

        else:
            g1, g2, T = gm.get_g1g2T()
            factor = shape.get_round_factor(g1, g2)

        gdata = gm.get_data()

        # make sure the determinant gets reset
        gdata["norm_set"] = 0

        ngauss = len(gm)
        for i in range(ngauss):
            Ti = gdata["irr"][i] + gdata["icc"][i]
            gdata["irc"][i] = 0.0
            gdata["irr"][i] = 0.5 * Ti * factor
            gdata["icc"][i] = 0.5 * Ti * factor

        return gm

    def _fill_image(self, image, jacobian=None, fast_exp=False):
        """
        Internal routine.  Render the mixture into a new image.  No error
        checking on the image!  The data are *added* to the image

        parameters
        ----------
        image: 2-d double array
            image to render into
        fast_exp: bool, optional
            use fast, approximate exp function
        """

        if jacobian is None:
            cen = (numpy.array(image.shape) - 1.0) / 2.0
            jacobian = UnitJacobian(row=cen[0], col=cen[1])
        else:
            assert isinstance(jacobian, Jacobian)

        gm = self.get_data()

        coords = make_coords(image.shape, jacobian)
        render(
            gm, coords, image.ravel(), fast_exp,
        )

    def fill_fdiff(self, obs, fdiff, start=0):
        """
        Fill fdiff=(model-data)/err given the input Observation

        parameters
        ----------
        obs: Observation
            The Observation to compare with. See ngmix.observation.Observation
            The Observation must have a weight map set
        fdiff: 1-d array
            The fdiff to fill
        start: int, optional
            Where to start in the array, default 0
        """

        nuse = fdiff.size - start

        image = obs.image
        if nuse < image.size:
            raise ValueError(
                "fdiff from start must have "
                "len >= %d, got %d" % (image.size, nuse)
            )

        gm = self.get_data()
        fill_fdiff(
            gm, obs._pixels, fdiff, start,
        )

    def get_weighted_moments(self, obs, maxrad):
        """
        Get weighted moments using this mixture as the weight, including
        e1,e2,T,s2n etc.  If you just want the raw moments use
        get_weighted_sums()

        If you want the expected fluxes, you should set the flux to the inverse
        of the normalization which is 2*pi*sqrt(det)

        parameters
        ----------
        obs: Observation
            The Observation to compare with. See ngmix.observation.Observation
            The Observation must have a weight map set

        returns:
            result array with basic sums as well as summary statistics
            such as e1,e2,T,s2n etc.
        """

        res = self.get_weighted_sums(obs, maxrad)
        return get_weighted_moments_stats(res)

    def get_weighted_sums(self, obs, maxrad, res=None):
        """
        Get weighted moments using this mixture as the weight.  To
        get more summary statistics use get_weighted_moments or
        send this result to ngmix.gmix.get_weighted_moments_stats()

        parameters
        ----------
        obs: Observation
            The Observation to compare with. See ngmix.observation.Observation
            The Observation must have a weight map set
        res: result array, optional
            If sent, sums will be added to the array rather than making
            a new one
        """
        from . import admom
        from . import gmix_nb

        self.set_norms_if_needed()

        if res is None:
            dt = numpy.dtype(admom._admom_result_dtype, align=True)
            resarray = numpy.zeros(1, dtype=dt)
            res = resarray[0]

        wt_gm = self.get_data()

        # this will add to the sums
        gmix_nb.get_weighted_sums(
            wt_gm, obs.pixels, res, maxrad,
        )
        return res

    def get_model_s2n_sum(self, obs):
        """
        Get the s/n sum for the model, using only the weight
        map

            s2n_sum = sum(model_i^2 * ivar_i)

        The s/n would be sqrt(s2n_sum).  This sum can be
        added up over multiple images

        parameters
        ----------
        obs: Observation
            The Observation to compare with. See ngmix.observation.Observation
            The Observation must have a weight map set
        """

        gm = self.get_data()

        s2n_sum = get_model_s2n_sum(gm, obs.pixels)
        return s2n_sum

    def get_model_s2n(self, obs):
        """
        Get the s/n for the model, using only the weight
        map

            s2n_sum = sum(model_i^2 * ivar_i)
            s2n = sqrt(s2n_sum)

        parameters
        ----------
        obs: Observation
            The Observation to compare with. See ngmix.observation.Observation
            The Observation must have a weight map set
        """

        s2n_sum = self.get_model_s2n_sum(obs)
        s2n = sqrt(s2n_sum)
        return s2n

    def get_loglike(self, obs, more=False):
        """
        Calculate the log likelihood given the input Observation


        parameters
        ----------
        obs: Observation
            The Observation to compare with. See ngmix.observation.Observation
            The Observation must have a weight map set
        more:
            if True, return a dict with more informatioin
        """

        gm = self.get_data()
        res = get_loglike(gm, obs._pixels)

        res = pack_to_dict(res) if more else res[0]

        return res

    def reset(self):
        """
        Replace the data array with a zeroed one.
        """
        self._pars = zeros(self._npars)
        self._data = zeros(self._ngauss, dtype=_gauss2d_dtype)

    def make_galsim_object(self, Tmin=1e-6, gsparams=None):
        """
        make a galsim representation for the gaussian mixture

        Note ngmix fluxes are surface brightness, galsim are not.
        So to get agreement in a drawn image you may need to
        convert the flux using a pixel scale squared

        parameters
        ----------
        Tmin: float, optional
            Minimum size for gaussians.  Galsim doesn't allow objects
            with less than zero size because when convolving it renders
            the object
        gsparams: GSParams or dict, optional
            A GSParams object (or dict convertable to one) that sets
            certain useful parameters for GalSim renderings (e.g.
            a larger maximum FFT size)
        """

        import galsim

        if (gsparams is not None) and (
            not isinstance(gsparams, galsim.GSParams)
        ):
            if isinstance(gsparams, dict):
                # Convert to actual gsparams object
                gsparams = galsim.GSParams(**gsparams)
            else:
                raise TypeError(
                    "Only `dict` and `galsim.GSParam` types allowed"
                    " for gsparams; input has type of {}.".format(
                        type(gsparams)
                    )
                )

        data = self.get_data()

        gsobjects = []
        for i in range(len(self)):
            flux = data["p"][i]
            T = data["irr"][i] + data["icc"][i]
            if T == 0:
                T = Tmin

            e1 = (data["icc"][i] - data["irr"][i]) / T
            e2 = 2.0 * data["irc"][i] / T

            # these will most likely be sky coordinates rather than actually
            # (row,col), but I'm using those names to make it clear how we
            # reverse these for galsim below

            rowshift = data["row"][i]
            colshift = data["col"][i]

            g1, g2 = e1e2_to_g1g2(e1, e2)

            Tround = moments.get_Tround(T, g1, g2)
            if Tround < Tmin:
                Tround = Tmin

            sigma_round = sqrt(Tround / 2.0)

            gsobj = galsim.Gaussian(
                flux=flux, sigma=sigma_round, gsparams=gsparams
            )

            gsobj = gsobj.shear(g1=g1, g2=g2)

            gsobj = gsobj.shift(colshift, rowshift)

            gsobjects.append(gsobj)

        gs_obj = galsim.Add(gsobjects)

        return gs_obj

    def _set_fill_func(self):
        """
        set the function for filling the mixture
        """

        if self._model_name not in _gmix_fill_functions:
            raise ValueError("bad model: '%s'" % self._model_name)

        self._fill_func = _gmix_fill_functions[self._model_name]

    def __len__(self):
        return self._ngauss

    def __repr__(self):
        rep = []

        fmt = "p: %.4g row: %.4g col: %.4g irr: %.4g irc: %.4g icc: %.4g"
        for i in range(self._ngauss):
            t = self._data[i]
            s = fmt % (
                t["p"],
                t["row"],
                t["col"],
                t["irr"],
                t["irc"],
                t["icc"],
            )
            rep.append(s)

        rep = "\n".join(rep)
        return rep


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


class GMixModel(GMix):
    """
    A two-dimensional gaussian mixture created from a set of model parameters

    Inherits from the more general GMix class, and all its methods.

    parameters
    ----------
    pars: array-like
        Parameter array. The number of elements will depend
        on the model type.
    model: string or gmix type
        e.g. 'exp' or GMIX_EXP
    """

    def __init__(self, pars, model):
        self._do_init(pars, model)

    def _do_init(self, pars, model):
        self._model = _gmix_model_dict[model]
        self._model_name = _gmix_string_dict[self._model]

        self._ngauss = _gmix_ngauss_dict[self._model]
        self._npars = _gmix_npars_dict[self._model]

        self.reset()

        self._set_fill_func()
        self.fill(pars)

    def copy(self):
        """
        Get a new GMix with the same parameters
        """
        gmix = GMixModel(self._pars, self._model_name)
        return gmix

    def set_cen(self, row, col):
        """
        Move the mixture to a new center

        set pars as well
        """
        super(GMixModel, self).set_cen(row, col)

        pars = self._pars
        pars[0] = row
        pars[1] = col


class GMixCM(GMixModel):
    """
    Composite Model exp and dev using just fracdev

    Inherits from the more general GMix class, and all its methods.

    parameters
    ----------
    fracdev: float
        fraction of flux in the dev component
    TdByTe: float
        T_{dev}/T_{exp}
    pars: array-like
        6-parameters, same as simple models
    """

    def __init__(self, fracdev, TdByTe, pars):

        self._fracdev = fracdev
        self._TdByTe = TdByTe
        self._Tfactor = get_cm_Tfactor(fracdev, TdByTe)
        super(GMixCM, self).__init__(pars, "cm")

    def copy(self):
        """
        Get a new GMix with the same parameters
        """
        return GMixCM(self._fracdev, self._TdByTe, self._pars,)

    def _fill(self, pars):
        """
        Fill in the gaussian mixture with new parameters, with
        no error checking

        parameters
        ----------
        pars: ndarray or sequence
            The parameters
        """

        self._pars[:] = pars

        gm = self.get_data()
        self._fill_func(
            gm, self._fracdev, self._TdByTe, self._Tfactor, self._pars,
        )

    def __repr__(self):
        rep = super(GMixCM, self).__repr__()
        rep = [
            "fracdev: %g" % self._fracdev,
            "TdByTe:  %g" % self._TdByTe,
            rep,
        ]
        return "\n".join(rep)


def get_coellip_npars(ngauss):
    """
    get the number of paramters for the given ngauss
    coelliptical model
    """
    return 4 + 2 * ngauss


def get_coellip_ngauss(npars):
    """
    get the number of gaussians for the given nparameters
    coelliptical model
    """
    return (npars - 4) // 2


class GMixCoellip(GMixModel):
    """
    A two-dimensional gaussian mixture, each co-centeric and co-elliptical

    Inherits from the more general GMix class, and all its methods.

    parameters
    ----------
    pars: array-like
        Parameter array. The number of elements will depend
        on the model type.
    """

    def __init__(self, pars):

        self._model = GMIX_COELLIP
        self._model_name = "coellip"

        npars = len(pars)

        ncheck = npars - 4
        if (ncheck % 2) != 0:
            raise ValueError(
                "coellip must have len(pars)==4+2*ngauss, " "got %s" % npars
            )

        self._ngauss = ncheck // 2
        self._npars = npars

        self.reset()

        self._set_fill_func()
        self._fill(pars)

    def copy(self):
        """
        Get a new GMix with the same parameters
        """
        gmix = GMixCoellip(self._pars)
        return gmix


GMIX_FULL = 0
GMIX_GAUSS = 1
GMIX_TURB = 2
GMIX_EXP = 3
GMIX_DEV = 4
GMIX_BDC = 5
GMIX_BDF = 6
GMIX_COELLIP = 7
GMIX_CM = 9

GMIX_BD = 10

_gmix_model_dict = {
    "full": GMIX_FULL,
    GMIX_FULL: GMIX_FULL,
    "gauss": GMIX_GAUSS,
    GMIX_GAUSS: GMIX_GAUSS,
    "turb": GMIX_TURB,
    GMIX_TURB: GMIX_TURB,
    "exp": GMIX_EXP,
    GMIX_EXP: GMIX_EXP,
    "dev": GMIX_DEV,
    GMIX_DEV: GMIX_DEV,
    "bdc": GMIX_BDC,
    GMIX_BDC: GMIX_BDC,
    "bd": GMIX_BD,
    GMIX_BD: GMIX_BD,
    "bdf": GMIX_BDF,
    GMIX_BDF: GMIX_BDF,
    GMIX_CM: GMIX_CM,
    "cm": GMIX_CM,
    "coellip": GMIX_COELLIP,
    GMIX_COELLIP: GMIX_COELLIP,
}

_gmix_string_dict = {
    GMIX_FULL: "full",
    "full": "full",
    GMIX_GAUSS: "gauss",
    "gauss": "gauss",
    GMIX_TURB: "turb",
    "turb": "turb",
    GMIX_EXP: "exp",
    "exp": "exp",
    GMIX_DEV: "dev",
    "dev": "dev",
    GMIX_BDC: "bdc",
    "bdc": "bdc",
    GMIX_BD: "bd",
    "bd": "bd",
    GMIX_BDF: "bdf",
    "bdf": "bdf",
    GMIX_CM: "cm",
    "cm": "cm",
    GMIX_COELLIP: "coellip",
    "coellip": "coellip",
}


_gmix_npars_dict = {
    GMIX_GAUSS: 6,
    GMIX_TURB: 6,
    GMIX_EXP: 6,
    GMIX_DEV: 6,
    GMIX_CM: 6,
    GMIX_BD: 8,
    GMIX_BDF: 7,
    GMIX_BDC: 8,
}

_gmix_ngauss_dict = {
    GMIX_GAUSS: 1,
    "gauss": 1,
    GMIX_TURB: 3,
    "turb": 3,
    GMIX_EXP: 6,
    "exp": 6,
    GMIX_DEV: 10,
    "dev": 10,
    GMIX_CM: 16,
    GMIX_BD: 16,
    GMIX_BDF: 16,
    GMIX_BDC: 16,
    "em1": 1,
    "em2": 2,
    "em3": 3,
    "em4": 4,
    "em5": 5,
    "coellip1": 1,
    "coellip2": 2,
    "coellip3": 3,
    "coellip4": 4,
    "coellip5": 5,
}


_gauss2d_dtype = [
    ("p", "f8"),
    ("row", "f8"),
    ("col", "f8"),
    ("irr", "f8"),
    ("irc", "f8"),
    ("icc", "f8"),
    ("det", "f8"),
    ("norm_set", "i8"),
    ("drr", "f8"),
    ("drc", "f8"),
    ("dcc", "f8"),
    ("norm", "f8"),
    ("pnorm", "f8"),
]


def get_model_num(model):
    """
    Get the numerical identifier for the input model,
    which could be string or number
    """
    if model not in _gmix_model_dict:
        raise ValueError("unknown model: '%s'" % model)
    return _gmix_model_dict[model]


def get_model_name(model):
    """
    Get the string identifier for the input model,
    which could be string or number
    """
    if model not in _gmix_string_dict:
        raise ValueError("unknown model: '%s'" % model)
    return _gmix_string_dict[model]


def get_model_ngauss(model):
    """
    get the number of gaussians for the given model
    """
    if model not in _gmix_ngauss_dict:
        raise ValueError("unknown model: '%s'" % model)
    return _gmix_ngauss_dict[model]


def get_model_npars(model):
    """
    Get the number of parameters for the input model,
    which could be string or number
    """
    if model not in _gmix_model_dict:
        raise ValueError("bad model: '%s'" % model)
    mi = _gmix_model_dict[model]
    return _gmix_npars_dict[mi]


def pack_to_dict(res):
    loglike, s2n_numer, s2n_denom, npix = res
    return {
        "loglike": loglike,
        "s2n_numer": s2n_numer,
        "s2n_denom": s2n_denom,
        "npix": npix,
    }


def get_weighted_moments_stats(ares):
    """
    do some additional calculations based on the sums
    """
    from .admom import get_ratio_error

    res = {}
    for n in ares.dtype.names:
        if n == "sums":
            res[n] = ares[n].copy()
        elif n == "sums_cov":
            res[n] = ares[n].copy()
        else:
            res[n] = ares[n]

    # we always have a measure of the flux
    sums = res["sums"]
    sums_cov = res["sums_cov"]
    pars = res["pars"]

    flux_sum = sums[5]

    res["flux"] = flux_sum
    res["flux_err"] = 9999.0

    pars[5] = res["flux"]

    # these might not get filled in if T is too small
    # or if the flux variance is zero somehow
    res["T"] = -9999.0
    res["s2n"] = -9999.0
    res["e"] = array([-9999.0, -9999.0])
    res["e_err"] = 9999.0

    fvar_sum = sums_cov[5, 5]

    if fvar_sum > 0.0:

        res["flux_err"] = sqrt(fvar_sum)
        res["s2n"] = flux_sum / res["flux_err"]

    else:
        # zero var flag
        res["flags"] |= 0x40
        res["flagstr"] = "zero var"

    if res["flags"] == 0:

        if flux_sum > 0.0:
            finv = 1.0 / flux_sum

            row = sums[0] * finv
            col = sums[1] * finv
            M1 = sums[2] * finv
            M2 = sums[3] * finv
            T = sums[4] * finv

            pars[0] = row
            pars[1] = col
            pars[2] = M1
            pars[3] = M2
            pars[4] = T

            res["T"] = pars[4]

            res["T_err"] = get_ratio_error(
                sums[4],
                sums[5],
                sums_cov[4, 4],
                sums_cov[5, 5],
                sums_cov[4, 5],
            )

            if res["T"] > 0.0:
                res["e"][:] = res["pars"][2:2 + 2] / res["T"]

                e1_err = get_ratio_error(
                    sums[2],
                    sums[4],
                    sums_cov[2, 2],
                    sums_cov[4, 4],
                    sums_cov[2, 4],
                )
                e2_err = get_ratio_error(
                    sums[3],
                    sums[4],
                    sums_cov[3, 3],
                    sums_cov[4, 4],
                    sums_cov[3, 4],
                )

                if not isfinite(e1_err) or not isfinite(e2_err):
                    res["e_cov"] = diag([9999.0, 9999.0])
                else:
                    res["e_cov"] = diag([e1_err ** 2, e2_err ** 2])

            else:
                # T <= 0.0
                res["flags"] |= 0x8
                res["flagstr"] = "T <= 0.0"

        else:
            # flux <= 0.0
            res["flags"] |= 0x4
            res["flagstr"] = "flux <= 0.0"

    return res
