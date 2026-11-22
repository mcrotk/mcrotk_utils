"""Adding columns derived from existing ones: joined labels, and lagged copies by period (private).

Public through ``mcrotk_utils.pandas_utils``.
"""

import operator
from collections.abc import Iterable, Sequence
from typing import SupportsIndex

import pandas as pd


def join_columns(
    df: pd.DataFrame, columns: str | Sequence[str] | None, name: str | None = None
) -> tuple[pd.DataFrame, str | None]:
    """Combine several columns into one column of labels like ``"(a, 1)"``, e.g. to group or plot by them.

    For functions that accept either one column or several, e.g. a plot's ``hue``: they can call this and
    use the one column it names from then on.

    **Arguments.**

    - ``df``: the DataFrame.
    - ``columns``: ``None``, one column name, or a list of column names.
    - ``name``: the new column's name; by default the column names in the same form, e.g. ``"(size, color)"``.

    **Result.**

    A tuple of a DataFrame and a column name:

    - ``None`` or a single name (a string): ``df`` itself and ``columns``, unchanged.
    - An empty list: ``df`` itself and ``None``.
    - A list of one name: ``df`` itself and that name.
    - A list of two or more names: a new DataFrame (``df`` isn't changed) with the joined column added, and
      the new column's name. A column of that name already in ``df`` is replaced. Each value is turned into
      text with ``str``, so missing values show as ``nan``, ``None`` or ``<NA>``, rather than making the label
      missing.
    """
    if columns is None or isinstance(columns, str):
        return df, columns
    if len(columns) <= 1:
        return df, columns[0] if columns else None
    label = name if name is not None else "(" + ", ".join(map(str, columns)) + ")"
    # map(str) works the same in every pandas version; astype(str) keeps missing values missing in pandas 3
    texts = [df[column].astype(object).map(str) for column in columns]
    joined = texts[0]
    for text in texts[1:]:
        joined = joined + ", " + text
    joined = "(" + joined + ")"
    return df.assign(**{label: joined}), label


def add_lags(
    df: pd.DataFrame, lag_col: str, num_lags: SupportsIndex, other_fixed_cols: Iterable[str] = ()
) -> pd.DataFrame:
    """Add lagged copies of the columns: for each row, the values from 1, 2, ... periods earlier.

    Rows are matched by the period column ``lag_col`` and the columns in ``other_fixed_cols``: the lag-1
    value of a row is that of the row with the same ``other_fixed_cols`` values and a period 1 lower.

    **Arguments.**

    - ``df``: the DataFrame, with at most one row for each period and ``other_fixed_cols`` values.
    - ``lag_col``: the period column, e.g. week numbers; it must support adding integers.
    - ``num_lags``: how many lags to add (0 or more); NumPy integers work too.
    - ``other_fixed_cols``: columns that identify a series, e.g. a unit; lags don't cross them.

    **Result.**

    - A new DataFrame (``df`` isn't changed) with ``df``'s columns, index and row order, followed by, for each
      lag ``k`` from 1 to ``num_lags``, a column ``<name>_lag<k>`` for each other column.
    - A lag whose earlier row doesn't exist (e.g. the first periods, or a gap) is missing, so e.g. integer
      columns' lags are floats.

    **Errors.**

    - ``ValueError``: ``num_lags`` is negative, or two rows have the same period and ``other_fixed_cols``.
    - ``TypeError``: ``num_lags`` isn't an integer.

    **Example.** ``add_lags(df, "week", 2, ["store"])`` adds ``sales_lag1`` and ``sales_lag2``: each store's
    sales one and two weeks earlier.
    """
    num_lags = operator.index(num_lags)
    if num_lags < 0:
        raise ValueError(f"num_lags must be at least 0, not {num_lags}.")
    keys = [lag_col, *other_fixed_cols]
    if df.duplicated(subset=keys).any():
        raise ValueError(f"Each combination of {keys} must appear only once, so that lags are well defined.")
    value_columns = [column for column in df.columns if column not in keys]
    with_lags = df
    for lag in range(1, num_lags + 1):
        # shifting the periods forward by lag puts each row's values on the row lag periods later
        shifted = df.assign(**{lag_col: df[lag_col] + lag})
        with_lags = with_lags.merge(shifted, on=keys, how="left", suffixes=(None, f"_lag{lag}"))
    lag_columns = [f"{column}_lag{lag}" for lag in range(1, num_lags + 1) for column in value_columns]
    # a left merge on unique keys keeps the rows in order, one each, but resets the index and (in some pandas
    # versions, for empty DataFrames) reorders the columns
    result = with_lags[[*df.columns, *lag_columns]].copy()
    result.index = df.index
    return result
