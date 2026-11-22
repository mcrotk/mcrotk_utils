"""Tests for ``mcrotk_utils.pandas_utils``' ``is_binary`` and ``num_non_binary``."""

import numpy as np
import pandas as pd
import pytest

from mcrotk_utils.pandas_utils import (
    is_binary,
    num_non_binary,
)

# --- num_non_binary and is_binary ---


@pytest.mark.parametrize(
    ("values", "count"),
    [
        ([0, 1, 1], 0),
        ([0.0, 1.0], 0),
        ([True, False], 0),
        ([0, 2, -1], 2),
        ([0, 0.5], 1),
        ([0, np.nan], 1),
        (pd.array([0, None], dtype="Int64"), 1),
        (["0", "1"], 2),
        ([], 0),
    ],
)
def test_num_non_binary_and_is_binary(values, count):
    """Counts values other than 0 and 1 (booleans count as 0 and 1; missing values are non-binary)."""
    series = pd.Series(values, dtype=object if not len(values) else None)
    assert num_non_binary(series) == count
    assert isinstance(num_non_binary(series), int)
    assert is_binary(series) == (count == 0)
