"""Checking whether a Series' values are all 0 or 1 (private).

Public through ``mcrotk_utils.pandas_utils``.
"""

import pandas as pd


def num_non_binary(series: pd.Series) -> int:
    """Return the number of values that aren't 0 or 1 (``True`` and ``False`` count as 1 and 0).

    Missing values count as non-binary. An empty Series gives 0.
    """
    return int((~series.isin([0, 1])).sum())


def is_binary(series: pd.Series) -> bool:
    """Return whether every value is 0 or 1 (or ``True`` or ``False``).

    Missing values count as non-binary. An empty Series is binary.
    """
    return num_non_binary(series) == 0
