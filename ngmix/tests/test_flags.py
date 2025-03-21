import numpy as np

import pytest
from ..flags import get_flags_str, NAME_MAP, LOW_DET


@pytest.mark.parametrize(
    "dtype", [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64,
              np.uint64]
)
def test_get_flags_str(dtype):
    """Test that get_flags_str works for different integer types."""
    # Set a high-enough bit in addition to the lower bit to ensure things work.
    val = np.array(LOW_DET, dtype=dtype)
    if val.itemsize > 4:
        val |= 2**45
    flag_str = get_flags_str(val, NAME_MAP)
    assert flag_str == NAME_MAP[LOW_DET]
