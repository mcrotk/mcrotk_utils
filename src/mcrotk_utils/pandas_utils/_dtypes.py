"""Column dtypes: categoricals, nullable integers and datetimes, and the categorical dtype helpers that
concatenating and merging share (private).

Public through ``mcrotk_utils.pandas_utils``.
"""

from collections.abc import Hashable, Iterable, Mapping, Sequence

import pandas as pd
from pandas.api.types import CategoricalDtype


def column_dtypes(
    categories: Mapping[str, Iterable[Hashable]] | None = None,
    nullable_ints: Iterable[str] = (),
    *,
    ordered: bool = True,
) -> dict[str, CategoricalDtype | pd.Int64Dtype]:
    """Return a mapping of columns to dtypes, for ``df.astype``: categoricals and nullable integers.

    **Arguments.**

    - ``categories``: for each categorical column, its categories, in their order.
    - ``nullable_ints``: integer columns that can have missing values; they get pandas' ``Int64`` dtype.
    - ``ordered``: whether the categoricals are ordered (by the order their categories are given in).

    **Result.**

    - A new dict from column names to dtypes; empty when no columns are given.

    **Errors.**

    - ``ValueError``: a column is in both ``categories`` and ``nullable_ints``, or a column's categories
      repeat a value.

    **Example.** ``df.astype(column_dtypes({"size": ["S", "M", "L"]}, ["count"]))``.
    """
    categories = categories or {}
    nullable_ints = list(nullable_ints)
    in_both = sorted(set(categories) & set(nullable_ints))
    if in_both:
        raise ValueError(f"Columns can't be both categorical and nullable integers: {in_both}.")
    dtypes: dict[str, CategoricalDtype | pd.Int64Dtype] = {
        column: CategoricalDtype(list(column_categories), ordered=ordered)
        for column, column_categories in categories.items()
    }
    dtypes.update({column: pd.Int64Dtype() for column in nullable_ints})
    return dtypes


def to_datetime_columns(dfs: pd.DataFrame | Iterable[pd.DataFrame], columns: str | Iterable[str]) -> None:
    """Convert columns of one or more DataFrames to datetimes with ``pd.to_datetime``, in place.

    **Arguments.**

    - ``dfs``: a DataFrame, or several (e.g. a list).
    - ``columns``: a column name, or several; each DataFrame must have them all.

    **Result.**

    - None: the DataFrames are changed in place. Empty columns become empty datetime columns.

    **Errors.**

    - ``KeyError``: a DataFrame lacks one of the columns.
    - Whatever ``pd.to_datetime`` raises for values it can't convert.
    """
    frames = [dfs] if isinstance(dfs, pd.DataFrame) else list(dfs)
    names = [columns] if isinstance(columns, str) else list(columns)
    for df in frames:
        for column in names:
            df[column] = pd.to_datetime(df[column])


def _union_categorical_dtype(dtypes: Sequence[CategoricalDtype]) -> CategoricalDtype:
    """Return a categorical dtype with all the dtypes' categories, in order of first appearance.

    It is ordered if all the dtypes are.
    """
    categories = dtypes[0].categories
    for dtype in dtypes[1:]:
        # difference with sort=False keeps the new categories in their own order
        categories = categories.append(dtype.categories.difference(categories, sort=False))
    return CategoricalDtype(categories, ordered=all(dtype.ordered for dtype in dtypes))


def _to_categorical(series: pd.Series, dtype: CategoricalDtype) -> pd.Series:
    """Return ``series`` with exactly the categorical ``dtype``: its categories, in its order.

    ``astype`` isn't enough: pandas treats unordered dtypes with the same categories in another order as the
    same dtype, so it would leave the categories in their old order.
    """
    if isinstance(series.dtype, CategoricalDtype):
        return series.cat.set_categories(dtype.categories, ordered=dtype.ordered)
    return series.astype(dtype)
