"""Merging two DataFrames on their indexes, combining chosen columns with a preferred side (private).

Public through ``mcrotk_utils.pandas_utils``.
"""

import warnings
from collections.abc import Iterable, Mapping
from typing import Literal

import pandas as pd
from pandas.api.types import CategoricalDtype

from mcrotk_utils.pandas_utils._dtypes import _to_categorical, _union_categorical_dtype, to_datetime_columns

# Suffixes merge_with_preferences gives to columns that are in both DataFrames
_SUFFIXES = ("_left", "_right")


def merge_with_preferences(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    how: Literal["left", "right", "outer", "inner"],
    preference_left: Iterable[str] = (),
    preference_right: Iterable[str] = (),
    rename_left: Mapping[str, str] | None = None,
    rename_right: Mapping[str, str] | None = None,
    datetime_cols: str | Iterable[str] = (),
) -> pd.DataFrame:
    """Merge two DataFrames on their indexes, combining chosen columns that both have into one.

    For a column in ``preference_left``, rows in both DataFrames, and rows only in ``df1``, take ``df1``'s
    value; rows only in ``df2`` take ``df2``'s. ``preference_right`` is the same the other way around.

    **Arguments.**

    - ``df1``, ``df2``: the left and right DataFrames.
    - ``how``: which rows to keep, by index, as in ``pd.merge``: ``"left"``, ``"right"``, ``"outer"`` or
      ``"inner"``.
    - ``preference_left``: columns to combine into one, preferring ``df1``'s values.
    - ``preference_right``: columns to combine into one, preferring ``df2``'s values.
    - ``rename_left``, ``rename_right``: renames applied to ``df1``'s and ``df2``'s columns first (the
      preference lists use the new names).
    - ``datetime_cols``: columns converted with ``pd.to_datetime`` in both DataFrames first, e.g. dates read
      as strings in one and as dates in the other.

    **Result.**

    - A new DataFrame; the inputs aren't changed.
    - Each preferred column appears once, under its own name, after the other columns. A row in both
      DataFrames takes the preferred value even when it's missing.
    - Preferred columns keep their dtype when it's the same in both DataFrames, including categorical (with
      the categories of both), timezone-aware and nullable dtypes. (Rows that a DataFrame lacks have missing
      values, so e.g. integers can become floats, as in ``pd.merge``.)
    - Other columns in both DataFrames appear twice, with the suffixes ``_left`` and ``_right``.

    **Warnings.**

    - ``UserWarning``: a preferred column isn't in one of the DataFrames; it's treated as all missing there.

    **Errors.**

    - ``ValueError``: a column is in both ``preference_left`` and ``preference_right``.
    - ``KeyError``: a column in ``datetime_cols`` isn't in both DataFrames.
    """
    preference_left, preference_right = list(preference_left), list(preference_right)
    in_both = sorted(set(preference_left) & set(preference_right))
    if in_both:
        raise ValueError(f"Columns can't be in both preference_left and preference_right: {in_both}.")
    preferred = preference_left + preference_right
    # rename returns new DataFrames, so the changes below don't reach the inputs
    left, right = df1.rename(columns=rename_left or {}), df2.rename(columns=rename_right or {})
    to_datetime_columns([left, right], datetime_cols)
    left = _with_columns(left, right, preferred)
    right = _with_columns(right, left, preferred)
    indicator = _unused_column_name("_merge", left, right)
    merged = left.merge(right, how=how, left_index=True, right_index=True, indicator=indicator, suffixes=_SUFFIXES)
    for column in preference_left:
        merged[column] = _preferred_values(merged, column, "left", indicator)
    for column in preference_right:
        merged[column] = _preferred_values(merged, column, "right", indicator)
    return merged.drop(columns=[indicator] + [column + suffix for column in preferred for suffix in _SUFFIXES])


def _with_columns(df: pd.DataFrame, other: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return ``df`` with any of ``columns`` it lacks added as all missing, warning about each one.

    An added column takes ``other``'s dtype for that column where it can, so combining doesn't lose it.
    """
    missing = [column for column in columns if column not in df.columns]
    if not missing:
        return df
    df = df.copy()
    for column in missing:
        warnings.warn(f"Column {column!r} not found in a DataFrame in merge_with_preferences.", stacklevel=3)
        if column in other.columns:
            # reindexing an empty slice gives missing values of the same dtype where it can hold them
            df[column] = other[column].iloc[:0].reindex(df.index)
        else:
            df[column] = pd.Series(pd.NA, index=df.index, dtype=object)
    return df


def _unused_column_name(name: str, *dfs: pd.DataFrame) -> str:
    """Return ``name``, with underscores added at the front until no DataFrame has a column of that name."""
    while any(name in df.columns for df in dfs):
        name = "_" + name
    return name


def _preferred_values(
    merged: pd.DataFrame, column: str, preferred: Literal["left", "right"], indicator: str
) -> pd.Series:
    """Return the combined values of ``column``: the preferred side's, except in rows only the other side has."""
    other = "right" if preferred == "left" else "left"
    preferred_values, other_values = _with_common_categories(
        merged[f"{column}_{preferred}"], merged[f"{column}_{other}"]
    )
    return preferred_values.where(merged[indicator] != f"{other}_only", other_values)


def _with_common_categories(first: pd.Series, second: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return the two Series, both with all the categories of the two if both are categorical.

    ``Series.where`` raises for categoricals with different categories.
    """
    if not (isinstance(first.dtype, CategoricalDtype) and isinstance(second.dtype, CategoricalDtype)):
        return first, second
    dtype = _union_categorical_dtype([first.dtype, second.dtype])
    return _to_categorical(first, dtype), _to_categorical(second, dtype)
