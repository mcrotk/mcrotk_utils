"""Concatenating DataFrames, keeping categorical columns categorical when their categories differ (private).

Public through ``mcrotk_utils.pandas_utils``.
"""

import warnings
from collections.abc import Hashable, Mapping, Sequence

import pandas as pd
from pandas.api.types import CategoricalDtype

from mcrotk_utils.pandas_utils._dtypes import _to_categorical, _union_categorical_dtype


def combine_dataframes(*dfs: pd.DataFrame, ignore_index: bool = False) -> pd.DataFrame:
    """Concatenate DataFrames row-wise, keeping categorical columns categorical even when their categories differ.

    ``pd.concat`` alone turns a categorical column into a plain one (e.g. strings) when the DataFrames have
    different categories for it. Here, the column instead gets all the categories: the first DataFrame's in
    their order, then each later DataFrame's new ones in their order.

    **Arguments.**

    - ``dfs``: the DataFrames, any number of them (at least one), in order. They can have different columns,
      as with ``pd.concat``.
    - ``ignore_index``: if true, the result gets a new default index (0, 1, 2, ...) instead of the inputs'
      index values.

    **Result.**

    - A new DataFrame with the rows of all the DataFrames, in order. The inputs aren't changed.
    - A column that is categorical wherever it has values stays categorical, with all the categories. It is
      ordered if it is ordered in all the DataFrames.
    - Rows from a DataFrame without one of the columns have missing values there, as with ``pd.concat``.
    - DataFrames with no rows add their categories and columns, but don't change other columns' dtypes.
      (``pd.concat`` would, e.g. turning integers into objects when an empty DataFrame's columns are objects.)

    **Warnings.**

    - ``UserWarning``: a column is categorical in some DataFrames with rows but not in others. That column
      is left to ``pd.concat``, which usually makes it non-categorical.

    **Errors.**

    - ``ValueError``: no DataFrames given.

    **Example.** ``combine_dataframes(*[pd.read_pickle(path) for path in paths])`` combines the results of
    several runs whose categorical columns each have only the categories that run saw.
    """
    if not dfs:
        raise ValueError("combine_dataframes needs at least one DataFrame.")
    dtypes = _combined_categorical_dtypes(dfs)
    # give every DataFrame the combined categories first, so pd.concat sees identical dtypes and keeps them
    with_dtypes = [_with_categorical_dtypes(df, dtypes) for df in dfs]
    combined = _concat_ignoring_empty(with_dtypes, ignore_index)
    # columns missing from some DataFrames get missing values in pd.concat, which can lose their dtype
    return _with_categorical_dtypes(combined, dtypes)


def _combined_categorical_dtypes(dfs: Sequence[pd.DataFrame]) -> dict[Hashable, CategoricalDtype]:
    """Return, for each column that is categorical wherever it has values, its dtype with all the categories.

    A column of a DataFrame with no rows only counts if it's categorical: it has no values to lose. Warns
    about columns that are categorical in some DataFrames with rows but not in others, and leaves them out.
    """
    column_dtypes_seen: dict[Hashable, list[object]] = {}
    for df in dfs:
        for column, dtype in df.dtypes.items():
            if len(df) or isinstance(dtype, CategoricalDtype):
                column_dtypes_seen.setdefault(column, []).append(dtype)
    combined = {}
    for column, dtypes in column_dtypes_seen.items():
        categoricals = [dtype for dtype in dtypes if isinstance(dtype, CategoricalDtype)]
        if len(categoricals) == len(dtypes):
            combined[column] = _union_categorical_dtype(categoricals)
        elif categoricals:
            warnings.warn(
                f"Column {column!r} is categorical in some DataFrames but not in others; "
                "it is combined by pd.concat, which usually makes it non-categorical.",
                stacklevel=3,
            )
    return combined


def _with_categorical_dtypes(df: pd.DataFrame, dtypes: Mapping[Hashable, CategoricalDtype]) -> pd.DataFrame:
    """Return a copy of ``df`` with those of its columns that are in ``dtypes`` given exactly those dtypes."""
    df = df.copy()
    for column, dtype in dtypes.items():
        if column in df.columns:
            df[column] = _to_categorical(df[column], dtype)
    return df


def _concat_ignoring_empty(dfs: Sequence[pd.DataFrame], ignore_index: bool) -> pd.DataFrame:
    """Concatenate the DataFrames with rows, keeping the columns of those without.

    DataFrames with no rows would only change the result's dtypes (pd.concat counts their columns' dtypes).
    """
    with_rows = [df for df in dfs if len(df)] or list(dfs)
    combined = pd.concat(with_rows, ignore_index=ignore_index)
    # columns that only the left-out DataFrames have, in order of first appearance, as pd.concat orders them
    all_columns = list(dict.fromkeys(column for df in dfs for column in df.columns))
    if len(all_columns) > len(combined.columns):
        combined = combined.reindex(columns=all_columns)
    return combined
