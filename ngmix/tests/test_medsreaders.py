import os
import pytest

@pytest.mark.skipif(
    'MEDS_TEST_FILE' not in os.environ,
    reason='MEDS_TEST_FILE environment variable not defined',
)
@pytest.mark.parametrize(
    'weight_type',
    ['weight', 'uberseg', 'cweight', 'cseg', 'cseg-canonical'],
)
def test_medsreaders_smoke(weight_type):
    import ngmix.medsreaders

    fname = os.environ['MEDS_TEST_FILE']

    # test reading from single meds file
    m = ngmix.medsreaders.NGMixMEDS(fname)

    for i in range(m.size):
        obslist = m.get_obslist(i, weight_type=weight_type)
        if weight_type == 'weight':
            assert len(obslist) == m['ncutout'][i]

    # test an error that can occur
    with pytest.raises(ValueError):
        m.get_obs(1, 1, weight_type='blah')

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
                assert len(mbobs[band] == m['ncutout'][i])
