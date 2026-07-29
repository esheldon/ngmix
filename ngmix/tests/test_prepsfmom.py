import galsim
import numpy as np
import pytest
import time
from flaky import flaky
from numpy.testing import assert_allclose
from scipy.ndimage import gaussian_filter

from ngmix.prepsfmom import (
    KSigmaMom, PGaussMom,
    _build_square_apodization_mask,
    PrePSFMom,
    _gauss_kernels_cached,
    _zero_pad_and_compute_fft_cached_impl,
    _compute_cen_phase_shift,
    _compute_cen_phase_shift_orig,
)
from ngmix import Jacobian
from ngmix import Observation
import ngmix.prepsfmom
from ngmix.moments import make_mom_result
import ngmix.flags

RANDOM_TEST_FRAC = 0.5
RANDOM_TEST_SEED = 42


@pytest.mark.parametrize("row", [-0.4, 0, 1.2, 4.5])
@pytest.mark.parametrize("col", [-0.32434, 0, 1.43232, 4.56775])
@pytest.mark.parametrize("dim,msk", [
    (100, None),
    (453, None),
    (3, np.array([[True, False, True], [True, True, False], [True, True, True]])),
    (4, np.array([
        [False, True, False, True],
        [False, True, True, False],
        [True, True, True, True],
        [False, False, True, False]])),
])
def test_cen_phase_shift(row, col, msk, dim):
    np.testing.assert_allclose(
        _compute_cen_phase_shift(row, col, dim, msk=msk),
        _compute_cen_phase_shift_orig(row, col, dim, msk=msk)
    )


def _report_info(s, arr, mn, err):
    if mn is not None and err is not None:
        print(
            "%s:" % s,
            np.mean(arr), mn, np.mean(arr)/mn - 1,
            np.std(arr), err, np.std(arr)/err - 1,
            np.abs(np.mean(arr))/np.std(arr),
            flush=True,
        )
    else:
        print(
            "%s:" % s,
            np.mean(arr), None, None,
            np.std(arr), None, None,
            None,
            flush=True,
        )


def test_prepsfmom_kind(prepsfmom_caching):
    fitter = PrePSFMom(2.0, 'gauss')
    assert fitter.kind == 'pgauss'
    fitter = PrePSFMom(2.0, 'pgauss')
    assert fitter.kind == 'pgauss'
    fitter = PrePSFMom(2.0, 'ksigma')
    assert fitter.kind == 'ksigma'
    fitter = PGaussMom(2.0)
    assert fitter.kind == 'pgauss'
    fitter = KSigmaMom(2.0)
    assert fitter.kind == 'ksigma'


@pytest.mark.parametrize("cls", [KSigmaMom, PGaussMom])
def test_prepsfmom_raises_nopsf(cls, prepsfmom_caching):
    fitter = cls(20)
    obs = Observation(image=np.zeros((1000, 1000)))
    with pytest.raises(RuntimeError) as e:
        fitter.go(obs)

    assert "PSF must be set" in str(e.value)

    fitter = cls(20)
    obs = Observation(image=np.zeros((1000, 1000)))
    fitter.go(obs, no_psf=True)


@pytest.mark.parametrize("cls", [KSigmaMom, PGaussMom])
@pytest.mark.parametrize("trim_axis", [0, 1])
def test_prepsfmom_nonsquare(cls, trim_axis, prepsfmom_caching):
    """
    Test that a rectangular stamp cut from a square stamp gives the same
    result, including the errors: the pixels are identical and the noise
    scaling uses the actual pixel count
    """
    rng = np.random.RandomState(seed=10)

    image_size = 64
    ntrim = 16
    cen = (image_size - 1) / 2
    scale = 0.25

    gal = galsim.Gaussian(fwhm=0.9).shear(g1=-0.1, g2=0.2).withFlux(400)
    psf = galsim.Gaussian(fwhm=0.8).shear(g1=0.05, g2=-0.05)
    im = galsim.Convolve([gal, psf]).drawImage(
        nx=image_size, ny=image_size, scale=scale,
    ).array
    im += rng.normal(size=im.shape, scale=1.0e-4)
    wgt = np.ones_like(im) / 1.0e-4**2
    psf_im = psf.drawImage(
        nx=33, ny=33, scale=scale,
    ).array

    def _jac(row, col):
        return Jacobian(
            y=row, x=col, dudx=scale, dudy=0, dvdx=0, dvdy=scale,
        )

    psf_obs = Observation(image=psf_im, jacobian=_jac(16, 16))
    obs = Observation(
        image=im,
        weight=wgt,
        jacobian=_jac(cen, cen),
        psf=psf_obs,
    )

    # trim half the pad from each side along one axis; the object is
    # compact so the removed pixels are noise at ~1e-4 of the flux
    if trim_axis == 0:
        rim = im[ntrim:-ntrim, :]
        rwgt = wgt[ntrim:-ntrim, :]
        rjac = _jac(cen - ntrim, cen)
    else:
        rim = im[:, ntrim:-ntrim]
        rwgt = wgt[:, ntrim:-ntrim]
        rjac = _jac(cen, cen - ntrim)

    robs = Observation(image=rim, weight=rwgt, jacobian=rjac, psf=psf_obs)

    # ap_rad=0 so the only difference between the stamps is the
    # trimmed pixels themselves; the default edge taper would touch
    # different pixels in the two stamps
    fitter = cls(2.0, ap_rad=0)
    res = fitter.go(obs)
    rres = fitter.go(robs)

    assert res['flags'] == 0
    assert rres['flags'] == 0
    for key in ['flux', 'T', 'e1', 'e2']:
        assert_allclose(rres[key], res[key], rtol=0, atol=1.0e-3)
    # same noise density and same effective aperture, so the errors
    # agree despite the different pixel counts
    for key in ['flux_err', 'T_err']:
        assert_allclose(rres[key], res[key], rtol=1.0e-3, atol=0)


def _make_noise_image_data(rng, sigma):
    """
    gaussian galaxy and psf observation for the noise image tests
    """
    image_size = 48
    cen = (image_size - 1) / 2
    scale = 0.25

    gal = galsim.Gaussian(fwhm=0.9).shear(g1=-0.1, g2=0.2).withFlux(400)
    psf = galsim.Gaussian(fwhm=0.8)
    im0 = galsim.Convolve([gal, psf]).drawImage(
        nx=image_size, ny=image_size, scale=scale,
    ).array
    psf_im = psf.drawImage(nx=33, ny=33, scale=scale).array

    jac = Jacobian(y=cen, x=cen, dudx=scale, dudy=0, dvdx=0, dvdy=scale)
    pjac = Jacobian(y=16, x=16, dudx=scale, dudy=0, dvdx=0, dvdy=scale)
    psf_obs = Observation(image=psf_im, jacobian=pjac)

    wgt = np.ones_like(im0) / sigma ** 2
    return im0, wgt, jac, psf_obs


@pytest.mark.parametrize("cls", [KSigmaMom, PGaussMom])
def test_prepsfmom_noise_image_white(cls, prepsfmom_caching):
    """
    for white noise, the per-mode noise power measured from an attached
    noise image agrees with the weight map errors in the mean over
    realizations, and the measured moments are unchanged
    """
    nreal = 20
    rng = np.random.RandomState(seed=100)

    sigma = 0.06
    im0, wgt, jac, psf_obs = _make_noise_image_data(rng, sigma)
    im = im0 + rng.normal(scale=sigma, size=im0.shape)

    obs = Observation(image=im, weight=wgt, jacobian=jac, psf=psf_obs)
    res = cls(2.0).go(obs)
    assert res['flags'] == 0

    fitter = cls(2.0, use_noise_image=True)
    errs = {key: [] for key in ['flux_err', 'T_err']}
    for i in range(nreal):
        nobs = Observation(
            image=im, weight=wgt, jacobian=jac, psf=psf_obs,
            noise=rng.normal(scale=sigma, size=im.shape),
        )
        nres = fitter.go(nobs)
        assert nres['flags'] == 0

        # the noise image affects only the errors
        for key in ['flux', 'T', 'e1', 'e2']:
            assert_allclose(nres[key], res[key], rtol=0, atol=0)

        for key in errs:
            assert_allclose(nres[key], res[key], rtol=0.3, atol=0)
            errs[key].append(nres[key])

    for key in errs:
        assert_allclose(np.mean(errs[key]), res[key], rtol=0.05, atol=0)


@pytest.mark.parametrize("cls", [KSigmaMom, PGaussMom])
def test_prepsfmom_noise_image_correlated(cls, prepsfmom_caching):
    """
    for correlated noise, the per-mode errors match the observed
    scatter, which is exact up to trial statistics since the weight is
    fixed, while the white noise assumption underestimates the scatter
    badly
    """
    ntrial = 200
    filt_sigma = 1.25
    rng = np.random.RandomState(seed=3232)

    # pixel std for s/n ~ 40 with white noise; the correlated noise
    # boosts the power at the scales of the kernel by roughly the
    # effective area of the filter, so its pixel std is scaled down to
    # keep a similar effective s/n
    im0, _, jac, psf_obs = _make_noise_image_data(rng, 1.0)
    sigma_corr = np.sqrt(np.sum(im0 ** 2)) / 40.0 / 4.0
    wgt = np.ones_like(im0) / sigma_corr ** 2

    # the pixel std of a filtered unit white field, for normalization
    fac = gaussian_filter(
        rng.normal(size=(2000, 2000)), filt_sigma, mode='wrap',
    ).std()

    def _corr_noise():
        return gaussian_filter(
            rng.normal(size=im0.shape), filt_sigma, mode='wrap',
        ) * (sigma_corr / fac)

    fitter = cls(2.0, use_noise_image=True)

    fluxes, fluxerrs, e1s, e1errs = [], [], [], []
    for i in range(ntrial):
        nobs = Observation(
            image=im0 + _corr_noise(), weight=wgt, jacobian=jac,
            psf=psf_obs, noise=_corr_noise(),
        )
        nres = fitter.go(nobs)
        assert nres['flags'] == 0
        fluxes.append(nres['flux'])
        fluxerrs.append(nres['flux_err'])
        e1s.append(nres['e1'])
        e1errs.append(nres['e_err'][0])

        if i == 0:
            # the white noise errors depend only on the weight map and
            # kernels, so one fit gives them for all trials
            wres = cls(2.0).go(nobs)
            assert wres['flags'] == 0
            werr = wres['flux_err']

    assert_allclose(np.std(fluxes), np.mean(fluxerrs), rtol=0.15, atol=0)
    assert_allclose(np.std(e1s), np.mean(e1errs), rtol=0.15, atol=0)

    # the white noise assumption underestimates the scatter badly
    assert np.std(fluxes) / werr > 2.0


@pytest.mark.parametrize("cls", [KSigmaMom, PGaussMom])
def test_prepsfmom_noise_image_color(cls, prepsfmom_caching):
    """
    colors from fluxes measured with the same pre-psf aperture are
    absolutely calibrated, both the value and the error: the fixed
    aperture factor cancels in the flux ratio, the per-band psfs do
    not enter, and the flux errors are exact for the fixed weight.
    The noise is correlated and the errors use the attached noise
    images
    """
    ntrial = 200
    filt_sigma = 1.25
    rng = np.random.RandomState(seed=8712)

    image_size = 48
    cen = (image_size - 1) / 2
    scale = 0.25

    band_fluxes = [400.0, 600.0]
    gal = galsim.Gaussian(fwhm=0.9).shear(g1=-0.1, g2=0.2)
    # different psfs in the two bands; the pre-psf aperture makes the
    # color independent of them
    psfs = [
        galsim.Gaussian(fwhm=0.9).shear(g1=0.05, g2=-0.03),
        galsim.Gaussian(fwhm=0.8).shear(g1=-0.02, g2=0.04),
    ]

    jac = Jacobian(y=cen, x=cen, dudx=scale, dudy=0, dvdx=0, dvdy=scale)
    pjac = Jacobian(y=16, x=16, dudx=scale, dudy=0, dvdx=0, dvdy=scale)

    ims0 = []
    psf_obss = []
    for flux, psf in zip(band_fluxes, psfs):
        ims0.append(
            galsim.Convolve(gal.withFlux(flux), psf).drawImage(
                nx=image_size, ny=image_size, scale=scale,
            ).array
        )
        psf_im = psf.drawImage(nx=33, ny=33, scale=scale).array
        psf_obss.append(Observation(image=psf_im, jacobian=pjac))

    # effective s/n ~ 40 and 30 in the two bands; see the correlated
    # noise test for the scaling of the pixel std
    sigmas = [
        np.sqrt(np.sum(ims0[0] ** 2)) / 40.0 / 4.0,
        np.sqrt(np.sum(ims0[1] ** 2)) / 30.0 / 4.0,
    ]

    # the pixel std of a filtered unit white field, for normalization
    fac = gaussian_filter(
        rng.normal(size=(2000, 2000)), filt_sigma, mode='wrap',
    ).std()

    def _corr_noise(sigma):
        return gaussian_filter(
            rng.normal(size=ims0[0].shape), filt_sigma, mode='wrap',
        ) * (sigma / fac)

    fitter = cls(2.0, use_noise_image=True)

    color_true = -2.5 * np.log10(band_fluxes[0] / band_fluxes[1])
    colors, color_errs = [], []
    for i in range(ntrial):
        fluxes = []
        fracerr2 = 0.0
        for im0, sigma, psf_obs in zip(ims0, sigmas, psf_obss):
            obs = Observation(
                image=im0 + _corr_noise(sigma),
                weight=np.ones_like(im0) / sigma ** 2,
                jacobian=jac,
                psf=psf_obs,
                noise=_corr_noise(sigma),
            )
            res = fitter.go(obs)
            assert res['flags'] == 0
            fluxes.append(res['flux'])
            fracerr2 += (res['flux_err'] / res['flux']) ** 2

        colors.append(-2.5 * np.log10(fluxes[0] / fluxes[1]))
        color_errs.append(2.5 / np.log(10) * np.sqrt(fracerr2))

    colors = np.array(colors)

    # the aperture flux ratio equals the true flux ratio, so the mean
    # color is unbiased; the tolerance is statistical
    assert np.abs(np.mean(colors) - color_true) < (
        4 * np.std(colors) / np.sqrt(ntrial)
    )

    # the band noises are independent and the flux errors are exact,
    # so the propagated color error matches the observed scatter
    # absolutely
    assert_allclose(np.std(colors), np.mean(color_errs), rtol=0.15, atol=0)


@pytest.mark.parametrize("cls", [KSigmaMom, PGaussMom])
def test_prepsfmom_noise_image_raises(cls, prepsfmom_caching):
    """
    use_noise_image requires obs.noise to be set
    """
    rng = np.random.RandomState(seed=45)
    sigma = 0.06
    im0, wgt, jac, psf_obs = _make_noise_image_data(rng, sigma)
    obs = Observation(image=im0, weight=wgt, jacobian=jac, psf=psf_obs)
    with pytest.raises(ValueError):
        cls(2.0, use_noise_image=True).go(obs)


@pytest.mark.parametrize("cls", [KSigmaMom, PGaussMom])
def test_prepsfmom_raises_badjacob(cls, prepsfmom_caching):
    fitter = cls(1.2)

    gs_wcs = galsim.ShearWCS(
        0.2, galsim.Shear(g1=-0.1, g2=0.06)).jacobian()
    jac = Jacobian(
        y=0, x=0,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

    psf_jac = Jacobian(
        y=0, x=0,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy*2)

    obs = Observation(
        image=np.zeros((10, 10)),
        jacobian=jac,
        psf=Observation(image=np.zeros((10, 10)), jacobian=psf_jac),
    )

    with pytest.raises(RuntimeError) as e:
        fitter.go(obs)
    assert "same WCS Jacobia" in str(e.value)


@flaky(max_runs=10)
@pytest.mark.parametrize("use_cache", [True, False])
def test_prepsfmom_speed_and_cache(use_cache):
    if use_cache:
        ngmix.prepsfmom.turn_on_fft_caching()
        ngmix.prepsfmom.turn_on_kernel_caching()
    else:
        ngmix.prepsfmom.turn_off_fft_caching()
        ngmix.prepsfmom.turn_off_kernel_caching()

    image_size = 48
    psf_image_size = 53
    pixel_scale = 0.263
    fwhm = 0.9
    psf_fwhm = 0.9
    snr = 20
    mom_fwhm = 2

    rng = np.random.RandomState(seed=100)

    cen = (image_size - 1)/2
    psf_cen = (psf_image_size - 1)/2
    gs_wcs = galsim.ShearWCS(
        pixel_scale, galsim.Shear(g1=-0.1, g2=0.06)).jacobian()
    scale = np.sqrt(gs_wcs.pixelArea())
    shift = rng.uniform(low=-scale/2, high=scale/2, size=2)
    psf_shift = rng.uniform(low=-scale/2, high=scale/2, size=2)
    xy = gs_wcs.toImage(galsim.PositionD(shift))
    psf_xy = gs_wcs.toImage(galsim.PositionD(psf_shift))

    jac = Jacobian(
        y=cen + xy.y, x=cen + xy.x,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

    psf_jac = Jacobian(
        y=psf_cen + psf_xy.y, x=psf_cen + psf_xy.x,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

    gal = galsim.Gaussian(
        fwhm=fwhm
    ).shear(
        g1=-0.1, g2=0.2
    ).withFlux(
        400
    ).shift(
        dx=shift[0], dy=shift[1]
    )
    psf = galsim.Gaussian(
        fwhm=psf_fwhm
    ).shear(
        g1=0.3, g2=-0.15
    )
    im = galsim.Convolve([gal, psf]).drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs
    ).array
    noise = np.sqrt(np.sum(im**2)) / snr
    wgt = np.ones_like(im) / noise**2

    psf_im = psf.shift(
        dx=psf_shift[0], dy=psf_shift[1]
    ).drawImage(
        nx=psf_image_size,
        ny=psf_image_size,
        wcs=gs_wcs
    ).array

    # now we test the speed + caching
    _gauss_kernels_cached.cache_clear()
    _zero_pad_and_compute_fft_cached_impl.cache_clear()

    # the first fit will do numba stuff, so we exclude it
    # we also perturb the various inputs to fool our caches
    fitter = PGaussMom(
        fwhm=mom_fwhm + 1e-3,
    )

    obs = Observation(
        image=im + 1e-6,
        weight=wgt,
        jacobian=jac,
        psf=Observation(image=psf_im + 1e-8, jacobian=psf_jac),
    )

    dt = time.time()
    fitter.go(obs=obs)
    dt1 = time.time() - dt
    print("\n%0.4f ms for first fit" % (dt1*1000))

    # we miss once here for kernels, twice for images
    if use_cache:
        assert _gauss_kernels_cached.cache_info().misses == 1
        assert _zero_pad_and_compute_fft_cached_impl.cache_info().misses == 2
    else:
        assert _gauss_kernels_cached.cache_info().misses == 0
        assert _zero_pad_and_compute_fft_cached_impl.cache_info().misses == 0

    # the second fit will have numba cached, but not the other kernel and FFT caches
    fitter = PGaussMom(
        fwhm=mom_fwhm,
    )

    obs = Observation(
        image=im,
        weight=wgt,
        jacobian=jac,
        psf=Observation(image=psf_im, jacobian=psf_jac),
    )

    dt = time.time()
    fitter.go(obs=obs)
    dt2 = time.time() - dt
    print("%0.4f ms for second fit" % (dt2*1000))

    # we miss twice for kernels, total of 3 times since psf changed
    if use_cache:
        assert _gauss_kernels_cached.cache_info().misses == 2
        assert _zero_pad_and_compute_fft_cached_impl.cache_info().misses == 4
    else:
        assert _gauss_kernels_cached.cache_info().misses == 0
        assert _zero_pad_and_compute_fft_cached_impl.cache_info().misses == 0

    # now we test with full caching
    nfit = 1000
    dt = time.time()
    for _ in range(nfit):
        with obs.writeable():
            obs.image += 1e-6
        fitter.go(obs=obs)
    dt3 = time.time() - dt

    print("%0.4f ms per fit" % (dt3/nfit*1000))

    # we should never miss again for the calls above
    if use_cache:
        assert _gauss_kernels_cached.cache_info().misses == 2
        assert _zero_pad_and_compute_fft_cached_impl.cache_info().misses == 4 + nfit
    else:
        assert _gauss_kernels_cached.cache_info().misses == 0
        assert _zero_pad_and_compute_fft_cached_impl.cache_info().misses == 0

    # if numba stuff is cached this does not work so commented out
    # assert dt2 < dt1
    if use_cache:
        assert dt3/nfit < dt2*0.8
    else:
        assert dt3/nfit >= dt2*0.8


def _stack_list_of_dicts(res):  # pragma: no cover

    def _get_dtype(v):
        if isinstance(v, float):
            return ('f8',)
        elif isinstance(v, int):
            return ('i4',)
        elif isinstance(v, str):
            return ('U256',)
        elif hasattr(v, "dtype") and hasattr(v, "shape"):
            if "float" in str(v.dtype):
                dstr = "f8"
            else:
                dstr = "i8"

            if len(v.shape) == 1:
                return (dstr, v.shape[0])
            else:
                return (dstr, v.shape)
        else:
            raise RuntimeError("cannot interpret dtype of '%s'" % v)

    dtype = []
    for k, v in res[0].items():
        dtype.append((k,) + _get_dtype(v))
    d = np.zeros(len(res), dtype=dtype)
    for i in range(len(res)):
        for k, v in res[i].items():
            d[k][i] = v

    return d


@pytest.mark.parametrize("cls", [KSigmaMom, PGaussMom])
@pytest.mark.parametrize('snr', [1e1, 1e3])
@pytest.mark.parametrize('pixel_scale', [0.125, 0.25])
@pytest.mark.parametrize('fwhm,psf_fwhm', [(0.6, 0.9), (1.5, 0.9)])
@pytest.mark.parametrize('mom_fwhm', [2.0, 1.5, 1.2])
@pytest.mark.parametrize('image_size', [57, 58])
@pytest.mark.parametrize('psf_image_size', [33, 34])
@pytest.mark.parametrize('pad_factor', [3.5, 2])
def test_prepsfmom_gauss(
    pad_factor, image_size, psf_image_size, fwhm, psf_fwhm, pixel_scale, snr, mom_fwhm,
    cls, prepsfmom_caching,
):
    """fast test at a range of parameters to check that things come out ok"""
    rng = np.random.RandomState(seed=RANDOM_TEST_SEED)
    if rng.uniform() < RANDOM_TEST_FRAC:
        pytest.skip(
            "Skipping `test_prepsfmom_gauss` to save time."
        )

    rng = np.random.RandomState(seed=100)

    cen = (image_size - 1)/2
    psf_cen = (psf_image_size - 1)/2
    gs_wcs = galsim.ShearWCS(
        pixel_scale, galsim.Shear(g1=-0.1, g2=0.06)).jacobian()
    scale = np.sqrt(gs_wcs.pixelArea())
    shift = rng.uniform(low=-scale/2, high=scale/2, size=2)
    psf_shift = rng.uniform(low=-scale/2, high=scale/2, size=2)
    xy = gs_wcs.toImage(galsim.PositionD(shift))
    psf_xy = gs_wcs.toImage(galsim.PositionD(psf_shift))

    jac = Jacobian(
        y=cen + xy.y, x=cen + xy.x,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

    psf_jac = Jacobian(
        y=psf_cen + psf_xy.y, x=psf_cen + psf_xy.x,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

    gal = galsim.Gaussian(
        fwhm=fwhm
    ).shear(
        g1=-0.1, g2=0.2
    ).withFlux(
        400
    ).shift(
        dx=shift[0], dy=shift[1]
    )
    psf = galsim.Gaussian(
        fwhm=psf_fwhm
    ).shear(
        g1=0.3, g2=-0.15
    )
    im = galsim.Convolve([gal, psf]).drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs
    ).array
    noise = np.sqrt(np.sum(im**2)) / snr
    wgt = np.ones_like(im) / noise**2

    psf_im = psf.shift(
        dx=psf_shift[0], dy=psf_shift[1]
    ).drawImage(
        nx=psf_image_size,
        ny=psf_image_size,
        wcs=gs_wcs
    ).array

    fitter = cls(
        fwhm=mom_fwhm,
        pad_factor=pad_factor,
    )

    # get true flux
    im_true = gal.drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs,
    ).array
    obs = Observation(
        image=im_true,
        jacobian=jac,
    )
    res = cls(fwhm=mom_fwhm, pad_factor=pad_factor).go(obs=obs, no_psf=True)
    flux_true = res["flux"]
    T_true = res["T"]
    g1_true = res["e"][0]
    g2_true = res["e"][1]

    res = []
    for _ in range(100):
        _im = im + rng.normal(size=im.shape, scale=noise)
        obs = Observation(
            image=_im,
            weight=wgt,
            jacobian=jac,
            psf=Observation(image=psf_im, jacobian=psf_jac),
        )

        _res = fitter.go(obs=obs)
        if _res['flags'] == 0:
            res.append(_res)

    res = _stack_list_of_dicts(res)

    if np.mean(res["flux"])/np.mean(res["flux_err"]) > 7:
        print("\n")
        _report_info("snr", np.mean(res["flux"])/np.mean(res["flux_err"]), None, None)
        _report_info("flux", res["flux"], flux_true, np.mean(res["flux_err"]))
        _report_info("T", res["T"], T_true, np.mean(res["T_err"]))
        _report_info("g1", res["e"][:, 0], g1_true, np.mean(res["e_err"][0]))
        _report_info("g2", res["e"][:, 1], g2_true, np.mean(res["e_err"][1]))
        mom_cov = np.cov(res["sums"].T)
        print("sums cov ratio:\n", np.mean(res["sums_cov"], axis=0)/mom_cov, flush=True)
        assert_allclose(
            np.abs(np.mean(res["flux"]) - flux_true)/np.mean(res["flux_err"]),
            0,
            atol=4,
            rtol=0,
        )
        assert_allclose(
            np.mean(res["flux"]), flux_true, atol=0, rtol=0.1)
        assert_allclose(
            np.std(res["flux"]), np.mean(res["flux_err"]), atol=0, rtol=0.2)


@pytest.mark.parametrize("cls,mom_fwhm,snr", [
    (KSigmaMom, 2.0, 1e2),
    (PGaussMom, 2.0, 1e2),
])
@pytest.mark.parametrize('pixel_scale', [0.25])
@pytest.mark.parametrize('fwhm,psf_fwhm', [(2.0, 1.0)])
@pytest.mark.parametrize('image_size', [53])
@pytest.mark.parametrize('pad_factor', [1.5])
@pytest.mark.parametrize('fwhm_smooth', [0, 1])
def test_prepsfmom_mn_cov_psf(
    pad_factor, image_size, fwhm, psf_fwhm, pixel_scale, snr, mom_fwhm, cls,
    fwhm_smooth, prepsfmom_caching,
):
    """Slower test to make sure means and errors are right
    w/ tons of monte carlo samples.
    """
    rng = np.random.RandomState(seed=RANDOM_TEST_SEED)
    if rng.uniform() < RANDOM_TEST_FRAC:
        pytest.skip(
            "Skipping `test_prepsfmom_mn_cov_psf` to save time."
        )

    rng = np.random.RandomState(seed=100)

    cen = (image_size - 1)/2
    gs_wcs = galsim.ShearWCS(
        pixel_scale, galsim.Shear(g1=-0.1, g2=0.06)).jacobian()
    scale = np.sqrt(gs_wcs.pixelArea())
    shift = rng.uniform(low=-scale/2, high=scale/2, size=2)
    psf_shift = rng.uniform(low=-scale/2, high=scale/2, size=2)
    xy = gs_wcs.toImage(galsim.PositionD(shift))
    psf_xy = gs_wcs.toImage(galsim.PositionD(psf_shift))

    jac = Jacobian(
        y=cen + xy.y, x=cen + xy.x,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

    psf_jac = Jacobian(
        y=26 + psf_xy.y, x=26 + psf_xy.x,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

    gal = galsim.Gaussian(
        fwhm=fwhm
    ).shear(
        g1=-0.1, g2=0.2
    ).withFlux(
        400
    ).shift(
        dx=shift[0], dy=shift[1]
    )
    psf = galsim.Gaussian(
        fwhm=psf_fwhm
    ).shear(
        g1=0.3, g2=-0.15
    )
    im = galsim.Convolve([gal, psf]).drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs
    ).array
    noise = np.sqrt(np.sum(im**2)) / snr
    wgt = np.ones_like(im) / noise**2

    psf_im = psf.shift(
        dx=psf_shift[0], dy=psf_shift[1]
    ).drawImage(
        nx=53,
        ny=53,
        wcs=gs_wcs
    ).array

    fitter = cls(
        fwhm=mom_fwhm,
        pad_factor=pad_factor,
        fwhm_smooth=fwhm_smooth,
    )

    # get true flux
    im_true = gal.drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs,
    ).array
    obs = Observation(
        image=im_true,
        jacobian=jac,
    )
    res = fitter.go(obs=obs, no_psf=True)
    flux_true = res["flux"]
    T_true = res["T"]
    g1_true = res["e"][0]
    g2_true = res["e"][1]

    res = []
    for _ in range(10_000):
        _im = im + rng.normal(size=im.shape, scale=noise)
        obs = Observation(
            image=_im,
            weight=wgt,
            jacobian=jac,
            psf=Observation(image=psf_im, jacobian=psf_jac),
        )

        _res = fitter.go(obs=obs)
        if _res['flags'] == 0:
            res.append(_res)

    res = _stack_list_of_dicts(res)

    print("\n")
    _report_info("snr", np.mean(res["flux"])/np.mean(res["flux_err"]), None, None)
    _report_info("flux", res["flux"], flux_true, np.mean(res["flux_err"]))
    _report_info("T", res["T"], T_true, np.mean(res["T_err"]))
    _report_info("g1", res["e"][:, 0], g1_true, np.mean(res["e_err"][0]))
    _report_info("g2", res["e"][:, 1], g2_true, np.mean(res["e_err"][1]))
    mom_cov = np.cov(res["sums"].T)
    print("sums cov ratio:\n", np.mean(res["sums_cov"], axis=0)/mom_cov, flush=True)
    print("sums cov meas:\n", mom_cov, flush=True)
    print("sums cov pred:\n", np.mean(res["sums_cov"], axis=0), flush=True)

    assert_allclose(np.mean(res["flux"]), flux_true, atol=0, rtol=1e-2)
    assert_allclose(np.mean(res["T"]), T_true, atol=0, rtol=1e-2)
    assert_allclose(np.mean(res["e"][:, 0]), g1_true, atol=0, rtol=2e-2)
    assert_allclose(np.mean(res["e"][:, 1]), g2_true, atol=0, rtol=2e-2)

    assert_allclose(np.std(res["flux"]), np.mean(res["flux_err"]), atol=0, rtol=2e-2)
    assert_allclose(np.std(res["T"]), np.mean(res["T_err"]), atol=0, rtol=2e-2)
    assert_allclose(
        np.std(res["e"][:, 0]), np.mean(res["e_err"][:, 0]), atol=0, rtol=2e-2)
    assert_allclose(
        np.std(res["e"][:, 1]), np.mean(res["e_err"][:, 1]), atol=0, rtol=2e-2)

    assert_allclose(
        mom_cov[2:, 2:],
        np.mean(res["sums_cov"][:, 2:, 2:], axis=0),
        atol=2.5e-1,
        rtol=0,
    )

    assert_allclose(
        np.diagonal(mom_cov[2:, 2:]),
        np.diagonal(np.mean(res["sums_cov"][:, 2:, 2:], axis=0)),
        atol=0,
        rtol=2e-2,
    )


@pytest.mark.parametrize("cls,mom_fwhm,snr", [(PGaussMom, 2.0, 1e2)])
@pytest.mark.parametrize('pixel_scale', [0.25])
@pytest.mark.parametrize('fwhm,psf_fwhm', [(2.0, 1.0)])
@pytest.mark.parametrize('image_size', [53])
@pytest.mark.parametrize('pad_factor', [1.5])
def test_prepsfmom_fwhm_smooth_snr(
    pad_factor, image_size, fwhm, psf_fwhm, pixel_scale, snr, mom_fwhm, cls,
    prepsfmom_caching,
):
    rng = np.random.RandomState(seed=RANDOM_TEST_SEED)
    if rng.uniform() < RANDOM_TEST_FRAC:
        pytest.skip(
            "Skipping `test_prepsfmom_fwhm_smooth_snr` to save time."
        )

    def _run_sim_fwhm_smooth(fwhm_smooth):
        rng = np.random.RandomState(seed=100)

        cen = (image_size - 1)/2
        gs_wcs = galsim.ShearWCS(
            pixel_scale, galsim.Shear(g1=-0.1, g2=0.06)).jacobian()
        scale = np.sqrt(gs_wcs.pixelArea())
        shift = rng.uniform(low=-scale/2, high=scale/2, size=2)
        psf_shift = rng.uniform(low=-scale/2, high=scale/2, size=2)
        xy = gs_wcs.toImage(galsim.PositionD(shift))
        psf_xy = gs_wcs.toImage(galsim.PositionD(psf_shift))

        jac = Jacobian(
            y=cen + xy.y, x=cen + xy.x,
            dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
            dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

        psf_jac = Jacobian(
            y=26 + psf_xy.y, x=26 + psf_xy.x,
            dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
            dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

        gal = galsim.Gaussian(
            fwhm=fwhm
        ).shear(
            g1=-0.1, g2=0.2
        ).withFlux(
            400
        ).shift(
            dx=shift[0], dy=shift[1]
        )
        psf = galsim.Gaussian(
            fwhm=psf_fwhm
        ).shear(
            g1=0.3, g2=-0.15
        )
        im = galsim.Convolve([gal, psf]).drawImage(
            nx=image_size,
            ny=image_size,
            wcs=gs_wcs
        ).array
        noise = np.sqrt(np.sum(im**2)) / snr
        wgt = np.ones_like(im) / noise**2

        psf_im = psf.shift(
            dx=psf_shift[0], dy=psf_shift[1]
        ).drawImage(
            nx=53,
            ny=53,
            wcs=gs_wcs
        ).array

        fitter = cls(
            fwhm=mom_fwhm,
            pad_factor=pad_factor,
            fwhm_smooth=fwhm_smooth,
        )

        # get true flux
        im_true = gal.drawImage(
            nx=image_size,
            ny=image_size,
            wcs=gs_wcs,
        ).array
        obs = Observation(
            image=im_true,
            jacobian=jac,
        )
        res = fitter.go(obs=obs, no_psf=True)

        res = []
        for _ in range(1_000):
            _im = im + rng.normal(size=im.shape, scale=noise)
            obs = Observation(
                image=_im,
                weight=wgt,
                jacobian=jac,
                psf=Observation(image=psf_im, jacobian=psf_jac),
            )

            _res = fitter.go(obs=obs)
            if _res['flags'] == 0:
                res.append(_res)

        res = _stack_list_of_dicts(res)

        return np.abs(np.mean(res["e"], axis=0))/np.std(res["e"], axis=0)

    e_snr = _run_sim_fwhm_smooth(0)
    e_snr_smooth = _run_sim_fwhm_smooth(1)

    assert np.all(e_snr_smooth > e_snr)


@pytest.mark.parametrize("cls,mom_fwhm,snr", [
    (PGaussMom, 2.0, 1e2),
    (KSigmaMom, 2.0, 1e2),
])
@pytest.mark.parametrize('pixel_scale', [0.25])
@pytest.mark.parametrize('fwhm', [2])
@pytest.mark.parametrize('image_size', [53])
@pytest.mark.parametrize('pad_factor', [1.5])
@pytest.mark.parametrize('fwhm_smooth', [0, 1])
def test_prepsfmom_mn_cov_nopsf(
    pad_factor, image_size, fwhm, pixel_scale, snr, mom_fwhm, cls, fwhm_smooth,
    prepsfmom_caching,
):
    """Slower test to make sure means and errors are right
    w/ tons of monte carlo samples.
    """
    rng = np.random.RandomState(seed=RANDOM_TEST_SEED)
    if rng.uniform() < RANDOM_TEST_FRAC:
        pytest.skip(
            "Skipping `test_prepsfmom_mn_cov_nopsf` to save time."
        )

    rng = np.random.RandomState(seed=100)

    cen = (image_size - 1)/2
    gs_wcs = galsim.ShearWCS(
        pixel_scale, galsim.Shear(g1=-0.1, g2=0.06)).jacobian()
    scale = np.sqrt(gs_wcs.pixelArea())
    shift = rng.uniform(low=-scale/2, high=scale/2, size=2)
    xy = gs_wcs.toImage(galsim.PositionD(shift))

    jac = Jacobian(
        y=cen + xy.y, x=cen + xy.x,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

    gal = galsim.Gaussian(
        fwhm=fwhm
    ).shear(
        g1=-0.1, g2=0.2
    ).withFlux(
        400
    ).shift(
        dx=shift[0], dy=shift[1]
    )
    im = gal.drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs
    ).array
    noise = np.sqrt(np.sum(im**2)) / snr
    wgt = np.ones_like(im) / noise**2

    fitter = cls(
        fwhm=mom_fwhm,
        pad_factor=pad_factor,
        fwhm_smooth=fwhm_smooth,
    )

    # get true flux
    im_true = gal.drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs,
    ).array
    obs = Observation(
        image=im_true,
        jacobian=jac,
    )
    res = cls(
        fwhm=mom_fwhm, pad_factor=pad_factor,
        fwhm_smooth=fwhm_smooth,
    ).go(obs=obs, no_psf=True)
    flux_true = res["flux"]
    T_true = res["T"]
    g1_true = res["e"][0]
    g2_true = res["e"][1]

    res = []
    for _ in range(10_000):
        _im = im + rng.normal(size=im.shape, scale=noise)
        obs = Observation(
            image=_im,
            weight=wgt,
            jacobian=jac,
        )

        _res = fitter.go(obs=obs, no_psf=True)
        if _res['flags'] == 0:
            res.append(_res)

    res = _stack_list_of_dicts(res)

    print("\n")
    _report_info("snr", np.mean(res["flux"])/np.mean(res["flux_err"]), None, None)
    _report_info("flux", res["flux"], flux_true, np.mean(res["flux_err"]))
    _report_info("T", res["T"], T_true, np.mean(res["T_err"]))
    _report_info("g1", res["e"][:, 0], g1_true, np.mean(res["e_err"][0]))
    _report_info("g2", res["e"][:, 1], g2_true, np.mean(res["e_err"][1]))
    mom_cov = np.cov(res["sums"].T)
    print("sums cov ratio:\n", np.mean(res["sums_cov"], axis=0)/mom_cov, flush=True)
    print("sums cov meas:\n", mom_cov, flush=True)
    print("sums cov pred:\n", np.mean(res["sums_cov"], axis=0), flush=True)

    assert_allclose(np.mean(res["flux"]), flux_true, atol=0, rtol=1e-2)
    assert_allclose(np.mean(res["T"]), T_true, atol=0, rtol=1e-2)
    assert_allclose(np.mean(res["e"][:, 0]), g1_true, atol=0, rtol=1e-2)
    assert_allclose(np.mean(res["e"][:, 1]), g2_true, atol=0, rtol=1e-2)

    assert_allclose(np.std(res["flux"]), np.mean(res["flux_err"]), atol=0, rtol=2e-2)
    assert_allclose(np.std(res["T"]), np.mean(res["T_err"]), atol=0, rtol=2e-2)
    assert_allclose(
        np.std(res["e"][:, 0]), np.mean(res["e_err"][:, 0]), atol=0, rtol=2e-2)
    assert_allclose(
        np.std(res["e"][:, 1]), np.mean(res["e_err"][:, 1]), atol=0, rtol=2e-2)

    assert_allclose(
        mom_cov[2:, 2:],
        np.mean(res["sums_cov"][:, 2:, 2:], axis=0),
        atol=2.5e-1,
        rtol=0,
    )

    assert_allclose(
        np.diagonal(mom_cov[2:, 2:]),
        np.diagonal(np.mean(res["sums_cov"][:, 2:, 2:], axis=0)),
        atol=0,
        rtol=2e-2,
    )


def test_moments_make_mom_result_flags():
    mom = np.ones(6)
    mom_cov = np.diag(np.ones(6))

    # weird cov
    for i in range(2, 6):
        _mom_cov = mom_cov.copy()
        _mom_cov[i, i] = -1
        res = make_mom_result(mom, _mom_cov, sums_norm=1)
        assert (res["flags"] & ngmix.flags.NONPOS_VAR) != 0
        assert ngmix.flags.NAME_MAP[ngmix.flags.NONPOS_VAR] in res["flagstr"]
        if i == 5:
            assert (res["flux_flags"] & ngmix.flags.NONPOS_VAR) != 0
            assert ngmix.flags.NAME_MAP[ngmix.flags.NONPOS_VAR] in res["flux_flagstr"]
        else:
            assert res["flux_flags"] == 0
            assert res["flux_flagstr"] == ""

        if i >= 4:
            assert (res["T_flags"] & ngmix.flags.NONPOS_VAR) != 0
            assert ngmix.flags.NAME_MAP[ngmix.flags.NONPOS_VAR] in res["T_flagstr"]
        else:
            assert res["T_flags"] == 0
            assert res["T_flagstr"] == ""

    # neg flux
    _mom = mom.copy()
    _mom[5] = -1
    res = make_mom_result(_mom, mom_cov, sums_norm=1)
    assert (res["flags"] & ngmix.flags.NONPOS_FLUX) != 0
    assert ngmix.flags.NAME_MAP[ngmix.flags.NONPOS_FLUX] in res["flagstr"]
    assert res["flux_flags"] == 0
    assert res["flux_flagstr"] == ""
    assert (res["T_flags"] & ngmix.flags.NONPOS_FLUX) != 0
    assert ngmix.flags.NAME_MAP[ngmix.flags.NONPOS_FLUX] in res["T_flagstr"]

    # neg T
    _mom = mom.copy()
    _mom[4] = -1
    res = make_mom_result(_mom, mom_cov, sums_norm=1)
    assert (res["flags"] & ngmix.flags.NONPOS_SIZE) != 0
    assert ngmix.flags.NAME_MAP[ngmix.flags.NONPOS_SIZE] in res["flagstr"]
    assert res["flux_flags"] == 0
    assert res["flux_flagstr"] == ""
    assert res["T_flags"] == 0
    assert res["T_flagstr"] == ""

    # bad shape errs
    for i in [2, 3]:
        _mom_cov = mom_cov.copy()
        _mom_cov[4, i] = np.nan
        _mom_cov[i, 4] = np.nan
        res = make_mom_result(mom, _mom_cov, sums_norm=1)
        assert (res["flags"] & ngmix.flags.NONPOS_SHAPE_VAR) != 0
        assert ngmix.flags.NAME_MAP[ngmix.flags.NONPOS_SHAPE_VAR] in res["flagstr"]
        assert res["flux_flags"] == 0
        assert res["flux_flagstr"] == ""
        assert res["T_flags"] == 0
        assert res["T_flagstr"] == ""


@pytest.mark.parametrize("cls", [PGaussMom, KSigmaMom])
@pytest.mark.parametrize('pixel_scale', [0.125, 0.25])
@pytest.mark.parametrize('fwhm,psf_fwhm', [(0.6, 0.9)])
@pytest.mark.parametrize('image_size', [250])
@pytest.mark.parametrize('psf_image_size', [33, 34])
@pytest.mark.parametrize('pad_factor', [4, 3.5])
def test_prepsfmom_gauss_true_flux(
    pad_factor, psf_image_size, image_size, fwhm, psf_fwhm, pixel_scale, cls,
    prepsfmom_caching,
):
    rng = np.random.RandomState(seed=RANDOM_TEST_SEED)
    if rng.uniform() < RANDOM_TEST_FRAC:
        pytest.skip(
            "Skipping `test_prepsfmom_gauss_true_flux` to save time."
        )

    rng = np.random.RandomState(seed=100)

    snr = 1e8
    mom_fwhm = 15.0

    cen = (image_size - 1)/2
    psf_cen = (psf_image_size - 1)/2
    gs_wcs = galsim.ShearWCS(
        pixel_scale, galsim.Shear(g1=-0.1, g2=0.06)).jacobian()
    scale = np.sqrt(gs_wcs.pixelArea())
    shift = rng.uniform(low=-scale/2, high=scale/2, size=2)
    psf_shift = rng.uniform(low=-scale/2, high=scale/2, size=2)
    xy = gs_wcs.toImage(galsim.PositionD(shift))
    psf_xy = gs_wcs.toImage(galsim.PositionD(psf_shift))

    jac = Jacobian(
        y=cen + xy.y, x=cen + xy.x,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

    psf_jac = Jacobian(
        y=psf_cen + psf_xy.y, x=psf_cen + psf_xy.x,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

    gal = galsim.Gaussian(
        fwhm=fwhm
    ).shear(
        g1=-0.1, g2=0.2
    ).withFlux(
        400
    ).shift(
        dx=shift[0], dy=shift[1]
    )
    psf = galsim.Gaussian(
        fwhm=psf_fwhm
    ).shear(
        g1=0.3, g2=-0.15
    )
    im = galsim.Convolve([gal, psf]).drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs
    ).array
    noise = np.sqrt(np.sum(im**2)) / snr
    wgt = np.ones_like(im) / noise**2

    psf_im = psf.shift(
        dx=psf_shift[0], dy=psf_shift[1]
    ).drawImage(
        nx=psf_image_size,
        ny=psf_image_size,
        wcs=gs_wcs
    ).array

    fitter = cls(
        fwhm=mom_fwhm,
        pad_factor=pad_factor,
    )

    # get true flux
    im_true = gal.drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs,
    ).array
    obs = Observation(
        image=im_true,
        jacobian=jac,
    )
    res = fitter.go(obs=obs, no_psf=True)
    flux_true = res["flux"]
    assert_allclose(flux_true, 400, atol=0, rtol=5e-3)

    obs = Observation(
        image=im,
        weight=wgt,
        jacobian=jac,
        psf=Observation(image=psf_im, jacobian=psf_jac),
    )
    res = fitter.go(obs=obs)
    flux_true = res["flux"]
    assert_allclose(flux_true, 400, atol=0, rtol=5e-3)


@pytest.mark.parametrize('pixel_scale', [0.25, 0.125])
@pytest.mark.parametrize('image_size', [107])
@pytest.mark.parametrize('pad_factor', [3.5, 4])
@pytest.mark.parametrize('mom_fwhm', [2, 2.5])
@pytest.mark.parametrize('cls', [PGaussMom, KSigmaMom])
@pytest.mark.parametrize('fwhm_smooth', [0, 1.5])
def test_prepsfmom_mom_norm(
    pad_factor, image_size, pixel_scale, mom_fwhm, cls, fwhm_smooth,
    prepsfmom_caching,
):
    rng = np.random.RandomState(seed=RANDOM_TEST_SEED)
    if rng.uniform() < RANDOM_TEST_FRAC:
        pytest.skip(
            "Skipping `test_prepsfmom_mom_norm` to save time."
        )

    rng = np.random.RandomState(seed=100)

    cen = (image_size - 1)/2
    gs_wcs = galsim.ShearWCS(
        pixel_scale, galsim.Shear(g1=-0.1, g2=0.06)).jacobian()
    scale = np.sqrt(gs_wcs.pixelArea())
    shift = rng.uniform(low=-scale/2, high=scale/2, size=2)
    xy = gs_wcs.toImage(galsim.PositionD(shift))

    jac = Jacobian(
        y=cen + xy.y, x=cen + xy.x,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

    obs = Observation(
        image=np.ones((image_size, image_size)),
        jacobian=jac,
    )
    res = cls(
        fwhm=mom_fwhm, pad_factor=pad_factor, fwhm_smooth=fwhm_smooth,
    ).go(
        obs=obs, no_psf=True,
    )
    assert_allclose(res["sums_norm"], res["flux"], atol=0, rtol=2e-4)


@pytest.mark.parametrize('pixel_scale', [0.25, 0.125])
@pytest.mark.parametrize('fwhm', [2, 0.5])
@pytest.mark.parametrize('image_size', [107])
@pytest.mark.parametrize('pad_factor', [3.5, 4])
@pytest.mark.parametrize('mom_fwhm', [2, 2.5])
def test_prepsfmom_comp_to_gaussmom_simple(
    pad_factor, image_size, fwhm, pixel_scale, mom_fwhm, prepsfmom_caching,
):
    rng = np.random.RandomState(seed=RANDOM_TEST_SEED)
    if rng.uniform() < RANDOM_TEST_FRAC:
        pytest.skip(
            "Skipping `test_prepsfmom_comp_to_gaussmom_simple` to save time."
        )

    rng = np.random.RandomState(seed=100)

    cen = (image_size - 1)/2
    gs_wcs = galsim.ShearWCS(
        pixel_scale, galsim.Shear(g1=-0.1, g2=0.06)).jacobian()
    scale = np.sqrt(gs_wcs.pixelArea())
    shift = rng.uniform(low=-scale/2, high=scale/2, size=2)
    xy = gs_wcs.toImage(galsim.PositionD(shift))

    jac = Jacobian(
        y=cen + xy.y, x=cen + xy.x,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

    gal = galsim.Gaussian(
        fwhm=fwhm
    ).shear(
        g1=-0.1, g2=0.2
    ).withFlux(
        400
    ).shift(
        dx=shift[0], dy=shift[1]
    )

    # get true flux
    im_true = gal.drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs,
    ).array
    im_true_nopixel = gal.drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs,
        method="no_pixel",
    ).array

    obs = Observation(
        image=im_true,
        jacobian=jac,
    )
    obs_nopixel = Observation(
        image=im_true_nopixel,
        jacobian=jac,
    )

    res = PGaussMom(
        fwhm=mom_fwhm, pad_factor=pad_factor,
    ).go(
        obs=obs, no_psf=True, return_kernels=True,
    )

    from ngmix.gaussmom import GaussMom
    res_gmom = GaussMom(fwhm=mom_fwhm).go(obs=obs_nopixel)

    for k in sorted(res):
        if k in res_gmom:
            print("%s:" % k, res[k], res_gmom[k])

    for k in ["flux", "flux_err", "T", "T_err", "e", "e_cov"]:
        assert_allclose(res[k], res_gmom[k], atol=0, rtol=2.5e-2)


@pytest.mark.parametrize('pixel_scale', [0.25, 0.125])
@pytest.mark.parametrize('fwhm', [2, 0.5])
@pytest.mark.parametrize('image_size', [107])
@pytest.mark.parametrize('pad_factor', [3.5, 4])
@pytest.mark.parametrize('mom_fwhm', [2, 2.5])
@pytest.mark.parametrize('fwhm_smooth', [0, 1.5])
def test_prepsfmom_comp_to_gaussmom_fwhm_smooth(
    pad_factor, image_size, fwhm, pixel_scale, mom_fwhm, fwhm_smooth,
    prepsfmom_caching,
):
    rng = np.random.RandomState(seed=RANDOM_TEST_SEED)
    if rng.uniform() < RANDOM_TEST_FRAC:
        pytest.skip(
            "Skipping `test_prepsfmom_comp_to_gaussmom_fwhm_smooth` to save time."
        )

    rng = np.random.RandomState(seed=100)

    cen = (image_size - 1)/2
    gs_wcs = galsim.ShearWCS(
        pixel_scale, galsim.Shear(g1=-0.1, g2=0.06)).jacobian()
    scale = np.sqrt(gs_wcs.pixelArea())
    shift = rng.uniform(low=-scale/2, high=scale/2, size=2)
    xy = gs_wcs.toImage(galsim.PositionD(shift))

    jac = Jacobian(
        y=cen + xy.y, x=cen + xy.x,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

    gal = galsim.Gaussian(
        fwhm=fwhm
    ).shear(
        g1=-0.1, g2=0.2
    ).withFlux(
        400
    ).shift(
        dx=shift[0], dy=shift[1]
    )

    # get true flux
    im_true = gal.drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs,
    ).array
    obs = Observation(
        image=im_true,
        jacobian=jac,
    )
    res = PGaussMom(
        fwhm=mom_fwhm, pad_factor=pad_factor, fwhm_smooth=fwhm_smooth,
    ).go(
        obs=obs, no_psf=True,
    )

    from ngmix.gaussmom import GaussMom
    if fwhm_smooth > 0:
        im_true_smooth = galsim.Convolve(
            [gal, galsim.Gaussian(fwhm=fwhm_smooth)]
        ).drawImage(
            nx=image_size,
            ny=image_size,
            wcs=gs_wcs,
            method="no_pixel",
        ).array
    else:
        im_true_smooth = gal.drawImage(
            nx=image_size,
            ny=image_size,
            wcs=gs_wcs,
            method="no_pixel",
        ).array

    obs_smooth = Observation(
        image=im_true_smooth,
        jacobian=jac,
    )
    res_gmom = GaussMom(fwhm=mom_fwhm).go(obs=obs_smooth)

    for k in sorted(res):
        if k in res_gmom:
            print("%s:" % k, res[k], res_gmom[k])

    assert_allclose(res["flux"], res_gmom["flux"], atol=0, rtol=5e-4)
    assert_allclose(res["T"], res_gmom["T"], atol=0, rtol=2e-3)
    assert_allclose(res["e"], res_gmom["e"], atol=0, rtol=3e-3)
    # the errors do not match - this is because the underlying noise model is
    # different - the pure gaussian moments weight map is an error on the convolved
    # profile whereas the pre-PSF case uses error propagation through the
    # smoothing kernel treating the weight map as applying to the unconvolved profile
    # thus we do not test the errors


def _sim_apodize(flux_factor, ap_rad):
    """
    we are simulating an object at the center with a bright object right on the
    edge of the stamp.

    We then apply apodization to the image and measure the same Gaussian moment
    with either the Fourier-space code or the real-space one.

    We compare the case with zero apodization to non-zero in the test below
    and assert that with apodization the results from Fourier-space match the
    real-space results better.
    """
    rng = np.random.RandomState(seed=100)
    image_size = 53
    pixel_scale = 0.25
    fwhm = 0.9
    mom_fwhm = 2.0
    pad_factor = 4

    cen = (image_size - 1)/2
    gs_wcs = galsim.ShearWCS(
        pixel_scale, galsim.Shear(g1=-0, g2=0.0)).jacobian()
    scale = np.sqrt(gs_wcs.pixelArea())
    shift = rng.uniform(low=-scale/2, high=scale/2, size=2)
    xy = gs_wcs.toImage(galsim.PositionD(shift))

    jac = Jacobian(
        y=cen + xy.y, x=cen + xy.x,
        dudx=gs_wcs.dudx, dudy=gs_wcs.dudy,
        dvdx=gs_wcs.dvdx, dvdy=gs_wcs.dvdy)

    gal = galsim.Gaussian(
        fwhm=fwhm
    ).shear(
        g1=-0.1, g2=0.2
    ).withFlux(
        400
    ).shift(
        dx=shift[0], dy=shift[1]
    )

    # get true flux
    im_true = gal.drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs,
    ).array

    im = im_true.copy()
    im += galsim.Exponential(
        half_light_radius=fwhm
    ).shear(
        g1=-0.5, g2=0.2
    ).shift(
        cen*pixel_scale,
        0,
    ).withFlux(
        400*flux_factor
    ).drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs,
        method="real_space",
    ).array

    obs = Observation(
        image=im,
        jacobian=jac,
    )
    res = PGaussMom(fwhm=mom_fwhm, pad_factor=pad_factor, ap_rad=ap_rad).go(
        obs=obs, no_psf=True, return_kernels=True,
    )

    ap_mask = np.ones_like(im)
    if ap_rad > 0:
        _build_square_apodization_mask(ap_rad, ap_mask)

    # get true flux
    im_nopixel = gal.drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs,
        method="no_pixel",
    ).array

    im_nopixel += galsim.Exponential(
        half_light_radius=fwhm
    ).shear(
        g1=-0.5, g2=0.2
    ).shift(
        cen*pixel_scale,
        0,
    ).withFlux(
        400*flux_factor
    ).drawImage(
        nx=image_size,
        ny=image_size,
        wcs=gs_wcs,
        method="no_pixel",
    ).array

    obs_ap = Observation(
        image=im_nopixel * ap_mask,
        jacobian=jac,
    )

    from ngmix.gaussmom import GaussMom
    res_gmom = GaussMom(fwhm=mom_fwhm).go(obs=obs_ap)

    return res, res_gmom


@pytest.mark.parametrize("flux_factor", [1e2, 1e3, 1e5])
def test_prepsfmom_apodize(flux_factor, prepsfmom_caching):
    res, res_geom = _sim_apodize(flux_factor, 1.5)
    ap_diffs = np.array([
        np.abs(res[k] - res_geom[k])
        for k in ["e1", "e2", "T", "flux"]
    ])
    print("apodized:", ap_diffs)

    res, res_geom = _sim_apodize(flux_factor, 0)
    zero_diffs = np.array([
        np.abs(res[k] - res_geom[k])
        for k in ["e1", "e2", "T", "flux"]
    ])
    print("non-apodized:", zero_diffs)

    assert np.all(zero_diffs > ap_diffs)
