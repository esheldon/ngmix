import copy
import numpy as np
import ngmix

DUDROW = 0.263
DUDCOL = -0.01
DVDROW = 0.01
DVDCOL = 0.263

CUTOUT_TYPES = [
    'image',
    'weight',
    'seg',
    'bmask',
    'psf',
]


def make_fake_meds(
    fname,
    rng,
    box_size=24,
    ncutout_max=3,
    nobj=10,
    model='gauss',
    flux=100.0,
    fwhm=0.9,
    noise=1.0,
    with_psf=False,
    psf_fwhm=0.9,
    cutout_types=None,
):

    import fitsio

    if cutout_types is None:
        cutout_types = copy.deepcopy(CUTOUT_TYPES)

    with_psf = 'psf' in cutout_types

    print('writing:', fname)
    with fitsio.FITS(fname, 'rw', clobber=True) as fits:
        print('writing object_data')
        object_data, total_pixels = make_object_data(
            rng=rng, nobj=nobj, box_size=box_size,
            ncutout_max=ncutout_max,
            with_psf=with_psf,
        )
        fits.write(object_data, extname='object_data')

        print('writing image_info')
        image_info = make_image_info(nimage=ncutout_max)
        fits.write(image_info, extname='image_info')

        print('writing metadata')
        metadata = make_metadata()
        fits.write(metadata, extname='metadata')

        print('reserving mosaic images')
        reserve_mosaic_images(
            fits=fits, cutout_types=cutout_types, total_pixels=total_pixels,
        )

        print('writing cutouts')
        write_cutouts(
            fits=fits, rng=rng,
            object_data=object_data, box_size=box_size,
            cutout_types=cutout_types,
            fwhm=fwhm,
            flux=flux,
            noise=noise,
            psf_fwhm=psf_fwhm,
        )


def write_cutouts(
    fits, rng, object_data, box_size, cutout_types,
    fwhm,
    flux,
    noise,
    psf_fwhm,
):
    for cutout_type in cutout_types:

        dtype = get_dtype(cutout_type)
        extname = get_extname(cutout_type)
        hdu = fits[extname]

        for iobj in range(object_data.size):
            for icut in range(object_data['ncutout'][iobj]):
                if cutout_type == 'image':

                    row = object_data['cutout_row'][iobj, icut]
                    col = object_data['cutout_col'][iobj, icut]

                    imdata = make_model_image(
                        row=row, col=col,
                        box_size=box_size,
                        fwhm=fwhm,
                        flux=flux,
                    )
                    imdata += rng.normal(scale=noise, size=imdata.shape)

                elif cutout_type == 'noise':
                    imdata = rng.normal(scale=noise, size=imdata.shape)
                elif cutout_type == 'mfrac':
                    imdata = rng.uniform(size=imdata.shape)
                elif cutout_type == 'psf':
                    row, col = [(box_size - 1)/2]*2
                    imdata = make_model_image(
                        row=row, col=col,
                        box_size=box_size,
                        fwhm=psf_fwhm,
                        flux=1,
                    )

                    imdata += rng.normal(scale=1.0e-6, size=imdata.shape)
                elif cutout_type == 'seg':
                    imdata = np.zeros((box_size, box_size), dtype=dtype)
                    imdata[:, :] = object_data['number'][iobj]
                else:
                    imdata = np.zeros((box_size, box_size), dtype=dtype)
                    if cutout_type == 'weight':
                        imdata[:, :] = 1.0/noise**2

                hdu.write(imdata, start=object_data['start_row'][iobj, icut])


def make_model_image(row, col, box_size, fwhm, flux):

    cen = (box_size - 1)/2
    off_row = row - cen
    off_col = col - cen

    T = ngmix.moments.fwhm_to_T(fwhm)
    pars = [off_row, off_col, 0.0, 0.0, T, flux]
    gm = ngmix.GMixModel(pars, "gauss")

    jacobian = make_jacobian(row=row, col=col)
    return gm.make_image([box_size]*2, jacobian=jacobian)


def get_extname(cutout_type):
    if cutout_type == 'psf':
        extname = cutout_type
    else:
        extname = '%s_cutouts' % cutout_type
    return extname


def get_dtype(cutout_type):
    if cutout_type in ['seg', 'bmask']:
        dtype = 'i2'
    else:
        dtype = 'f4'
    return dtype


def reserve_mosaic_images(fits, cutout_types, total_pixels):
    for cutout_type in cutout_types:

        dtype = get_dtype(cutout_type)
        extname = get_extname(cutout_type)

        fits.create_image_hdu(
            img=None,
            dtype=dtype,
            dims=[total_pixels],
            extname=extname,
        )


def make_jacobian(row, col):
    return ngmix.Jacobian(
        row=row,
        col=col,
        dudrow=DUDROW,
        dudcol=DUDCOL,
        dvdrow=DVDROW,
        dvdcol=DVDCOL,
    )


def make_object_data(
    rng,
    nobj,
    box_size,
    ncutout_max,
    with_psf=False,
):
    import meds

    extra_fields = [
        ('number', 'i4'),
        ('flux_auto', 'f4'),
        ('x2', 'f4'),
        ('y2', 'f4'),
    ]
    if with_psf:
        extra_fields += [
            ('psf_box_size', 'i4'),
            ('psf_start_row', 'i4', ncutout_max),
            ('psf_cutout_row', 'f4', ncutout_max),
            ('psf_cutout_col', 'f4', ncutout_max),
        ]

    data = meds.util.get_meds_output_struct(
        nobj=nobj, ncutout_max=ncutout_max,
        extra_fields=extra_fields,
    )

    for col in data.dtype.names:
        data[col] = -9999

    data['id'] = 1 + np.arange(nobj)
    data['number'] = 1 + np.arange(nobj)
    data['box_size'] = box_size

    data['ra'] = rng.uniform(low=0, high=3, size=nobj)
    data['dec'] = rng.uniform(low=0, high=3, size=nobj)
    data['ncutout'] = rng.randint(low=1, high=ncutout_max+1, size=nobj)

    # doesn't matter
    data['orig_row'] = rng.uniform(0, 500, size=(nobj, ncutout_max))
    data['orig_col'] = rng.uniform(0, 500, size=(nobj, ncutout_max))
    data['orig_start_row'] = data['orig_row'] - box_size/2
    data['orig_start_col'] = data['orig_col'] - box_size/2

    cen = (box_size - 1)/2
    data['cutout_row'] = cen + rng.uniform(low=-0.5, high=0.5, size=(nobj, ncutout_max))
    data['cutout_col'] = cen + rng.uniform(low=-0.5, high=0.5, size=(nobj, ncutout_max))
    data['dudrow'] = DUDROW
    data['dudcol'] = DUDCOL
    data['dvdrow'] = DVDROW
    data['dvdcol'] = DVDCOL

    if with_psf:
        data['psf_box_size'] = box_size
        data['psf_cutout_row'] = cen
        data['psf_cutout_col'] = cen

    ids = np.arange(ncutout_max)
    total_pixels = 0
    current_row = 0
    npixels_per = box_size * box_size
    for iobj in range(nobj):
        ncutout = data['ncutout'][iobj]
        data['file_id'][iobj, :ncutout] = ids[:ncutout]

        for icut in range(data['ncutout'][iobj]):
            data['start_row'][iobj, icut] = current_row

            if with_psf:
                data['psf_start_row'][iobj, icut] = current_row

            current_row += npixels_per
            total_pixels += npixels_per

    return data, total_pixels


def make_image_info(nimage):
    import meds
    ii = meds.util.get_image_info_struct(nimage=nimage, path_len=10)
    ii['scale'] = 1.0
    ii['image_path'] = 'blah.fits'
    return ii


def make_metadata():
    metadata = np.zeros(1, dtype=[('medsconf', 'S10')])
    metadata['medsconf'] = 'meds01'
    return metadata
