import os
import pytest
import tempfile
import numpy as np


@pytest.mark.parametrize(
    'weight_type',
    ['weight', 'uberseg', 'cweight', 'cseg', 'cseg-canonical'],
)
@pytest.mark.parametrize('with_psf', [False, True])
@pytest.mark.parametrize('with_bmask', [False, True])
@pytest.mark.parametrize('with_ormask', [False, True])
@pytest.mark.parametrize('with_noise', [False, True])
@pytest.mark.parametrize('with_mfrac', [False, True])
def test_medsreaders_smoke(
    weight_type, with_psf, with_bmask, with_ormask, with_noise, with_mfrac,
):
    import ngmix.medsreaders
    from ._fakemeds import make_fake_meds

    rng = np.random.RandomState(542)

    cutout_types = ['image', 'weight', 'seg']
    if with_bmask:
        cutout_types += ['bmask']
    if with_ormask:
        cutout_types += ['ormask']
    if with_noise:
        cutout_types += ['noise']
    if with_mfrac:
        cutout_types += ['mfrac']
    if with_psf:
        cutout_types += ['psf']

    with tempfile.TemporaryDirectory() as tdir:
        fname = os.path.join(tdir, 'test-meds.fits')
        make_fake_meds(
            fname=fname, rng=rng,
            cutout_types=cutout_types,
        )

        # test reading from single meds file
        m = ngmix.medsreaders.NGMixMEDS(fname)

        for iobj in range(m.size):
            obslist = m.get_obslist(iobj, weight_type=weight_type)

            # for other weight types, we might reject some stamps
            if weight_type == 'weight':
                assert len(obslist) == m['ncutout'][iobj], ('iobj: %s' % iobj)

            for obs in obslist:
                assert obs.has_seg()
                if with_psf:
                    assert obs.has_psf()
                if with_bmask:
                    assert obs.has_bmask()
                if with_ormask:
                    assert obs.has_ormask()
                if with_mfrac:
                    assert obs.has_mfrac()
                if with_noise:
                    assert obs.has_noise()

        # test an error that can occur
        with pytest.raises(ValueError):
            m.get_obs(0, 0, weight_type='blah')

        # multiple bands; re-use the same meds file
        nband = 3
        mlist = [m]*nband
        mm = ngmix.medsreaders.MultiBandNGMixMEDS(mlist)
        assert mm.nband == nband
        assert mm.size == m.size

        # lists of mbobs.  full set versus subset
        mbobs_list = mm.get_mbobs_list(weight_type=weight_type)
        assert len(mbobs_list) == mm.size

        ind = [1, 3]
        mbobs_list = mm.get_mbobs_list(indices=ind, weight_type=weight_type)
        if weight_type == 'weight':
            assert len(mbobs_list) == len(ind)

        # read mbobs
        for i in range(m.size):
            mbobs = mm.get_mbobs(i)
            assert len(mbobs) == nband

            mbobs = mm.get_mbobs(i, weight_type=weight_type)
            if weight_type == 'weight':
                for band in range(nband):
                    assert len(mbobs[band]) == m['ncutout'][i]


def test_medsreaders_cseg():
    import ngmix.medsreaders
    from ._fakemeds import make_fake_meds

    rng = np.random.RandomState(542)

    with tempfile.TemporaryDirectory() as tdir:
        fname = os.path.join(tdir, 'test-meds.fits')
        make_fake_meds(fname=fname, rng=rng)

        # test reading from single meds file
        m = ngmix.medsreaders.NGMixMEDS(fname)

        for iobj in range(m.size):
            obslist = m.get_obslist(iobj, seg_type='seg')
            cobslist = m.get_obslist(iobj, seg_type='cseg')
            for obs, cobs in zip(obslist[1:], cobslist[1:]):
                assert np.any(obs.seg != cobs.seg)
