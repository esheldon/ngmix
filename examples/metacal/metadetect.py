"""
Run metadetection example (metacalibration + detection)

For detection we use sep, with DES settings as implemented in the
sxdes package.  To install use

    conda install -c conda-forge des-sxdes


For this example we just measure simple weighted moments.
"""
import numpy as np
import ngmix
import galsim
import sxdes

SCALE = 0.263
DENSITY = 30.0  # per square arcminute
DIM_ARCMIN = 1.0
BUFF_ARCSEC = 0.1 * 60
DIM_ARCSEC = DIM_ARCMIN * 60
OFFSET_MAX = (DIM_ARCSEC - 2*BUFF_ARCSEC)/2
DIM = int(DIM_ARCSEC/SCALE)
NOBJ_MEAN = int(DIM_ARCMIN**2 * DENSITY)


def main():
    args = get_args()

    shear_true = [args.shear, 0.00]
    rng = np.random.RandomState(args.seed)

    # let's just do R11 for simplicity and to speed up this example; typically
    # the off diagonal terms are negligible, and R11 and R22 are usually
    # consistent

    dlist = []
    for i in progress(args.ntrial, miniters=10):

        obs = make_sim_obs(
            rng=rng, noise=args.noise, shear=shear_true, show=args.show,
        )
        tdata = get_all_moments(obs=obs, rng=rng)
        dlist.append(tdata)

    data = np.hstack(dlist)

    print_shear(data, shear_true)

    if args.output is not None:
        import fitsio
        print('writing:', args.output)
        fitsio.write(args.output, data, clobber=True)


def print_shear(data, shear_true):
    """
    calculate and print the shear and bias m/c

    Parameters
    ----------
    data: array
        Needs fields g, flags, s2n, T, Tpsf
    shear_true: (g1, g2)
        The true shear
    """
    w = select(data=data, shear_type='noshear')
    w_1p = select(data=data, shear_type='1p')
    w_1m = select(data=data, shear_type='1m')

    g = data['g'][w].mean(axis=0)
    gerr = data['g'][w].std(axis=0) / np.sqrt(w.size)
    g1_1p = data['g'][w_1p, 0].mean()
    g1_1m = data['g'][w_1m, 0].mean()

    g1_1p_err = data['g'][w_1p, 0].std()/np.sqrt(w_1p.size)
    g1_1m_err = data['g'][w_1m, 0].std()/np.sqrt(w_1m.size)

    R11 = (g1_1p - g1_1m)/0.02
    R11_err = np.sqrt(g1_1p_err**2 + g1_1m_err**2 - 2*0.5*g1_1m_err**2)

    shear = g / R11
    shear_err = gerr / R11

    m = shear[0]/shear_true[0]-1
    merr = shear_err[0]/shear_true[0]

    s2n = data['s2n'][w].mean()

    print('s2n: %g' % s2n)
    print('R11: %g +/- %g' % (R11, R11_err))
    print('m: %g +/- %g (99.7%% conf)' % (m, merr*3))
    print('c: %g +/- %g (99.7%% conf)' % (shear[1], shear_err[1]*3))


def select(data, shear_type):
    """
    select the data by shear type and size

    Parameters
    ----------
    data: array
        The array with fields shear_type and T
    shear_type: str
        e.g. 'noshear', '1p', etc.

    Returns
    -------
    array of indices
    """
    # raw moments, so the T is the post-psf T.  This the
    # selection is > 1.2 rather than something smaller like 0.5
    # for pre-psf T from one of the maximum likelihood fitters

    wtype, = np.where(data['shear_type'] == shear_type)

    w, = np.where(
        (data['flags'][wtype] == 0) &
        (data['T'][wtype]/data['Tpsf'][wtype] > 1.2) &
        (data['s2n'][wtype] > 10) &
        (data['s2n'][wtype] < 300) &
        (np.abs(data['g'][wtype, 0]) < 0.99) &
        (np.abs(data['g'][wtype, 1]) < 0.99)
    )

    w = wtype[w]
    print('%s kept %d/%d' % (shear_type, w.size, wtype.size))

    return w


def get_all_moments(obs, rng):
    """
    get moments for each metacal image type

    Parameters
    ----------
    obs: ngmix.Observation
        The observation to process
    rng: np.random.RandomState
        Random number generator

    Returns
    -------
    array of struct
    """
    obsdict = ngmix.metacal.get_all_metacal(
        obs=obs, rng=rng, types=['noshear', '1p', '1m'],
    )

    dlist = []
    for key, mcal_obs in obsdict.items():
        data = get_moments(mcal_obs)
        data['shear_type'] = key
        dlist.append(data)

    return np.hstack(dlist)


def get_moments(obs):
    """
    run detection and get moments for input observation

    Parameters
    ----------
    obs: ngmix.Observation
        The observation to process

    Returns
    -------
    array of struct
    """

    # measure moments with a fixed gaussian weight function, no psf correction
    weight_fwhm = 1.2
    fitter = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)
    stamp_dim = 32

    cat = run_sep(obs)
    psf_res = fitter.go(obs=obs.psf)

    dlist = []
    for iobj in range(cat.size):

        sflag, tobs = make_stamp_obs(cat[iobj], obs, stamp_dim)
        if sflag != 0:
            continue
        res = fitter.go(obs=tobs)
        st = make_struct(res)
        st['Tpsf'] = psf_res['T']
        dlist.append(st)
        # import esutil as eu
        # eu.numpy_util.ahelp(cat)
        # stop

    data = np.hstack(dlist)
    return data


def make_stamp_obs(obj_data, obs, stamp_dim):
    """
    extract a postage stamp and make an Observation

    Parameters
    ----------
    obj_data: array
        Scalar array representing an object's data from sep
    obs: Observation
        The observation from which to extract postage stamp data
    stamp_dim: int
        Dims of the stamp to extract

    Returns
    -------
    Observation
    """
    half_box_size = stamp_dim//2

    maxrow, maxcol = obs.image.shape

    row = obj_data['y'].astype('i4')
    col = obj_data['x'].astype('i4')

    start_row = row - half_box_size + 1
    end_row = row + half_box_size + 1  # plus one for slices

    start_col = col - half_box_size + 1
    end_col = col + half_box_size + 1

    if start_row < 0:
        return 1, None
        # start_row = 0
    if start_col < 0:
        return 1, None
        # start_col = 0
    if end_row > maxrow:
        return 1, None
        # end_row = maxrow
    if end_col > maxcol:
        return 1, None
        # end_col = maxcol

    image = obs.image[start_row:end_row, start_col:end_col].copy()
    weight = obs.weight[start_row:end_row, start_col:end_col].copy()

    stamp_row = obj_data['y'] - start_row
    stamp_col = obj_data['x'] - start_col
    jacobian = obs.jacobian.copy()
    jacobian.set_cen(row=stamp_row, col=stamp_col)

    stamp_obs = ngmix.Observation(
        image=image,
        weight=weight,
        jacobian=jacobian,
    )

    return 0, stamp_obs


def make_struct(res):
    """
    make the data structure

    Parameters
    ----------
    res: dict
        With keys 's2n', 'e', and 'T'

    Returns
    -------
    1-element array with fields
    """
    dt = [
        ('flags', 'i2'),
        ('shear_type', 'U7'),
        ('s2n', 'f8'),
        ('g', 'f8', 2),
        ('T', 'f8'),
        ('Tpsf', 'f8'),
    ]
    data = np.zeros(1, dtype=dt)
    data['flags'] = res['flags']
    data['s2n'] = res['s2n']
    data['g'] = res['e']
    data['T'] = res['T']

    return data


def run_sep(obs):
    """
    run sep to get detections.  We use the sxdes package
    to use DES settings

    Parameters
    ----------
    obs: ngmix.Observation
        The observation with the image to run detection on

    Returns
    -------
    cat: array with the detections and some measurements
    """
    # sep wants the image to be writeable for some reason
    noise = np.sqrt(1.0/obs.weight[0, 0])
    with obs.writeable():
        cat, seg = sxdes.run_sep(obs.image, noise)

    # from espy import images
    # images.view(seg)
    return cat


def make_sim_obs(rng, noise, shear, show=False):
    """
    simulate an exponential object with moffat psf

    the hlr of the exponential is drawn from a gaussian
    with mean 0.4 arcseconds and sigma 0.2

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    noise: float
        Noise for the image
    shear: (g1, g2)
        The shear in each component

    Returns
    -------
    ngmix.Observation
    """

    psf_noise = 1.0e-6

    nobj = rng.poisson(NOBJ_MEAN)

    psf_fwhm = 0.9

    psf = galsim.Moffat(
        beta=2.5, fwhm=psf_fwhm,
    ).shear(
        g1=0.02,
        g2=-0.01,
    )

    objs0 = []

    for i in range(nobj):
        dx, dy = rng.uniform(low=-OFFSET_MAX, high=OFFSET_MAX, size=2)
        gal_hlr = rng.normal(loc=0.4, scale=0.2)
        obj0 = galsim.Exponential(
            half_light_radius=gal_hlr,
        ).shear(
            g1=shear[0],
            g2=shear[1],
        ).shift(
            dx=dx,
            dy=dy,
        )
        objs0.append(obj0)

    objs0 = galsim.Sum(objs0)
    objs = galsim.Convolve(psf, objs0)

    psf_im = psf.drawImage(scale=SCALE).array
    im = objs.drawImage(nx=DIM, ny=DIM, scale=SCALE).array

    psf_im += rng.normal(scale=psf_noise, size=psf_im.shape)
    im += rng.normal(scale=noise, size=im.shape)

    cen = (np.array(im.shape)-1.0)/2.0
    psf_cen = (np.array(psf_im.shape)-1.0)/2.0

    jacobian = ngmix.DiagonalJacobian(
        row=cen[0] + dy/SCALE, col=cen[1] + dx/SCALE, scale=SCALE,
    )
    psf_jacobian = ngmix.DiagonalJacobian(
        row=psf_cen[0], col=psf_cen[1], scale=SCALE,
    )

    wt = im*0 + 1.0/noise**2
    psf_wt = psf_im*0 + 1.0/psf_noise**2

    psf_obs = ngmix.Observation(
        psf_im,
        weight=psf_wt,
        jacobian=psf_jacobian,
    )

    obs = ngmix.Observation(
        im,
        weight=wt,
        jacobian=jacobian,
        psf=psf_obs,
    )

    if show:
        from espy import images
        images.view(im)

    return obs


def progress(total, miniters=1):
    last_print_n = 0
    last_printed_len = 0
    sl = str(len(str(total)))
    mf = '%'+sl+'d/%'+sl+'d %3d%%'
    for i in range(total):
        yield i

        num = i+1
        if i == 0 or num == total or num - last_print_n >= miniters:
            meter = mf % (num, total, 100*float(num) / total)
            nspace = max(last_printed_len-len(meter), 0)

            print('\r'+meter+' '*nspace, flush=True, end='')
            last_printed_len = len(meter)
            if i > 0:
                last_print_n = num

    print(flush=True)


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=31415,
                        help='seed for rng')
    parser.add_argument('--ntrial', type=int, default=1000,
                        help='number of trials')
    parser.add_argument(
        '--noise', type=float, default=0.001,
        help='noise for images. Not too low to avoid bad segmentation',
    )
    parser.add_argument(
        '--shear', type=float, default=0.02,
        help=('shear for images; higher gives better s/n but '
              'can lead to some small bias (m=0.0004 '
              'shear=0.02, 0.01 for shear=0.1'),
    )

    parser.add_argument('--psf', default='gauss',
                        help='psf for reconvolution')
    parser.add_argument('--show', action='store_true')

    parser.add_argument('--output', help='write an output fits file')
    return parser.parse_args()


if __name__ == '__main__':
    main()
