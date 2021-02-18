import os
import sys
import copy
import glob
import logging

import numpy as np
import galsim
import fitsio

import ngmix
from metadetect.metadetect import Metadetect

# setup logging
for lib in [__name__, 'ngmix', 'metadetect']:
    lgr = logging.getLogger(lib)
    hdr = logging.StreamHandler(sys.stdout)
    hdr.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    lgr.setLevel(logging.DEBUG)
    lgr.addHandler(hdr)


SX_CONFIG = {
    # in sky sigma
    # DETECT_THRESH
    'detect_thresh': 0.8,

    # Minimum contrast parameter for deblending
    # DEBLEND_MINCONT
    'deblend_cont': 0.00001,

    # minimum number of pixels above threshold
    # DETECT_MINAREA: 6
    'minarea': 4,

    'filter_type': 'conv',

    # 7x7 convolution mask of a gaussian PSF with FWHM = 3.0 pixels.
    'filter_kernel': [
        [0.004963, 0.021388, 0.051328, 0.068707,
         0.051328, 0.021388, 0.004963],
        [0.021388, 0.092163, 0.221178, 0.296069,
         0.221178, 0.092163, 0.021388],
        [0.051328, 0.221178, 0.530797, 0.710525,
         0.530797, 0.221178, 0.051328],
        [0.068707, 0.296069, 0.710525, 0.951108,
         0.710525, 0.296069, 0.068707],
        [0.051328, 0.221178, 0.530797, 0.710525,
         0.530797, 0.221178, 0.051328],
        [0.021388, 0.092163, 0.221178, 0.296069,
         0.221178, 0.092163, 0.021388],
        [0.004963, 0.021388, 0.051328, 0.068707,
         0.051328, 0.021388, 0.004963],
    ]
}

MEDS_CONFIG = {
    'min_box_size': 32,
    'max_box_size': 64,

    'box_type': 'iso_radius',

    'rad_min': 4,
    'rad_fac': 2,
    'box_padding': 2,
}

GAUSS_PSF = {
    'model': 'gauss',
    'ntry': 2,
    'lm_pars': {
        'maxfev': 2000,
        'ftol': 1.0e-5,
        'xtol': 1.0e-5
    }
}

TEST_METADETECT_CONFIG = {
    'weight': {
        'fwhm': 1.2,  # arcsec
    },

    'metacal': {
        'psf': 'fitgauss',
        'types': ['noshear', '1p', '1m', '2p', '2m'],
        'use_noise_image': True,
    },

    'sx': SX_CONFIG,

    'meds': MEDS_CONFIG,

    # needed for PSF symmetrization
    'psf': GAUSS_PSF,

    # check for an edge hit
    'bmask_flags': 2**30,

    # flags for mask fractions
    'star_flags': 0,
    'tapebump_flags': 0,
    'spline_interp_flags': 0,
    'noise_interp_flags': 0,
    'imperfect_flags': 0,
}


def make_sim(seed=42):
    scale = 0.25
    flux = 10.0**(0.4 * (30 - 18))
    noise = 10
    ngals = 120
    shape = 361
    buff = 60
    inner_shape = (shape - 2*buff)
    im_cen = (shape-1)/2

    rng = np.random.RandomState(seed=seed)
    us = rng.uniform(low=-inner_shape/2, high=inner_shape/2, size=ngals) * scale
    vs = rng.uniform(low=-inner_shape/2, high=inner_shape/2, size=ngals) * scale
    wcs = galsim.PixelScale(scale)
    psf = galsim.Gaussian(fwhm=0.8)

    # PSF image
    psf_img = psf.drawImage(nx=33, ny=33, wcs=wcs).array
    psf_cen = (psf_img.shape[0]-1)/2
    psf_jac = ngmix.jacobian.DiagonalJacobian(
        row=psf_cen,
        col=psf_cen,
        scale=scale,
    )
    target_s2n = 500.0
    target_noise = np.sqrt(np.sum(psf_img ** 2)) / target_s2n
    psf_obs = ngmix.Observation(
        psf_img,
        weight=np.ones_like(psf_img)/target_noise**2,
        jacobian=psf_jac,
    )

    # gals
    gals = []
    for u, v in zip(us, vs):
        gals.append(galsim.Convolve([
            galsim.Exponential(half_light_radius=0.5),
            psf,
        ]).withFlux(flux).shift(u, v))
    gals = galsim.Sum(gals)

    img = gals.drawImage(nx=shape, ny=shape, wcs=wcs).array
    img += rng.normal(size=img.shape) * noise
    nse = rng.normal(size=img.shape) * noise
    wgt = img*0 + 1.0/noise**2
    msk = np.zeros(img.shape, dtype='i4')
    im_jac = ngmix.jacobian.DiagonalJacobian(
        row=im_cen,
        col=im_cen,
        scale=scale,
    )
    obs = ngmix.Observation(
        img,
        weight=wgt,
        bmask=msk,
        ormask=msk.copy(),
        jacobian=im_jac,
        psf=psf_obs,
        noise=nse,
    )

    mbobs = ngmix.MultiBandObsList()
    obslist = ngmix.ObsList()
    obslist.append(obs)
    mbobs.append(obslist)

    return mbobs


def combine_arrays(md):
    n_data = sum(
        md.result[key].shape[0]
        for key in ["noshear", "1p", "1m", "2p", "2m"]
    )
    dtype = copy.deepcopy(list(md.result["noshear"].dtype.descr))
    dtype.append(("shear", "U10"))
    d = np.zeros(n_data, dtype=dtype)
    loc = 0
    for key in ["noshear", "1p", "1m", "2p", "2m"]:
        size = md.result[key].shape[0]
        for _col in dtype:
            col = _col[0]
            if col != "shear":
                d[col][loc:loc+size] = md.result[key][col]
            else:
                d[col][loc:loc+size] = key
        loc += size

    assert loc == n_data
    return np.sort(d, order=["sx_col", "sx_row", "shear"])


def test_mdet_regression(write=False):
    mbobs = make_sim()
    rng = np.random.RandomState(seed=42)

    md = Metadetect(TEST_METADETECT_CONFIG, mbobs, rng)
    md.go()

    all_res = combine_arrays(md)

    if write:
        print("ngmix path:", ngmix.__file__)
        pth = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "mdet_test_data_%s.fits" % ngmix.__version__,
        )
        fitsio.write(pth, all_res, clobber=True)
    else:
        pth = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "mdet_test_data_*.fits",
        )
        fnames = glob.glob(pth)
        assert len(fnames) > 0
        for fname in fnames:
            old_data = fitsio.read(fname)
            for col in old_data.dtype.names:
                if np.issubdtype(old_data[col].dtype, np.number):
                    assert np.allclose(all_res[col], old_data[col]), col


if __name__ == "__main__":
    test_mdet_regression(write=True)
