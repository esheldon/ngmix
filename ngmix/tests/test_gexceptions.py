import pickle
import io
import pytest

from ..gexceptions import (
    NGmixBaseException,
    GMixRangeError,
    GMixFatalError,
    GMixMaxIterEM,
    BootPSFFailure,
    BootGalFailure,
)


@pytest.mark.parametrize('excp', [
    GMixRangeError,
    GMixFatalError,
    GMixMaxIterEM,
    BootPSFFailure,
    BootGalFailure,
])
def test_gexceptions_subclassing(excp):
    with pytest.raises(NGmixBaseException) as e:
        raise excp("blah blah")

    assert "blah blah" in repr(e.value)
    assert isinstance(e.value, excp)


@pytest.mark.parametrize('excp', [
    GMixRangeError,
    GMixFatalError,
    GMixMaxIterEM,
    BootPSFFailure,
    BootGalFailure,
])
def test_gexceptions_eval_repr(excp):
    e = excp("blah blah")
    et = eval(repr(e))
    assert isinstance(et, excp)
    assert repr(et) == repr(e)


@pytest.mark.parametrize('excp', [
    GMixRangeError,
    GMixFatalError,
    GMixMaxIterEM,
    BootPSFFailure,
    BootGalFailure,
])
def test_gexceptions_eval_str(excp):
    e = excp("blah blah")
    assert str(e) == "'blah blah'"


@pytest.mark.parametrize('excp', [
    GMixRangeError,
    GMixFatalError,
    GMixMaxIterEM,
    BootPSFFailure,
    BootGalFailure,
])
def test_gexceptions_pickle(excp):
    e = excp("blah blah")
    buff = io.BytesIO()
    pickle.dump(e, buff)
    buff.seek(0)
    el = pickle.load(buff)
    assert repr(e) == repr(el)
