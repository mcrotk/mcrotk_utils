"""Tests for ``mcrotk_utils.pandas_utils``' dtypes: ``column_dtypes`` and ``to_datetime_columns``."""

import pandas as pd
import pytest
from pandas.api.types import CategoricalDtype

from mcrotk_utils.pandas_utils import (
    column_dtypes,
    to_datetime_columns,
)


def categories(series):
    """Return a categorical Series' categories as a list."""
    return series.cat.categories.tolist()


# --- column_dtypes ---


def test_column_dtypes_gives_ordered_categoricals_and_nullable_ints():
    """Categorical columns get ordered categorical dtypes; integer columns get Int64."""
    dtypes = column_dtypes({"size": ["S", "M", "L"]}, ["count"])
    assert dtypes == {"size": CategoricalDtype(["S", "M", "L"], ordered=True), "count": pd.Int64Dtype()}


def test_column_dtypes_work_with_astype():
    """The result converts a DataFrame's columns with astype, keeping missing values."""
    df = pd.DataFrame({"size": ["M", "S", None], "count": [1, None, 3]})
    result = df.astype(column_dtypes({"size": ["S", "M", "L"]}, ["count"]))
    assert categories(result["size"]) == ["S", "M", "L"]
    assert result["size"].min() == "S"  # ordered by the given categories, not alphabetically
    assert result["count"].dtype == pd.Int64Dtype()
    assert result["count"].isna().tolist() == [False, True, False]


def test_column_dtypes_unordered():
    """With ordered=False the categoricals are unordered."""
    dtype = column_dtypes({"size": ["S", "M"]}, ordered=False)["size"]
    assert isinstance(dtype, CategoricalDtype)
    assert not dtype.ordered


def test_column_dtypes_empty():
    """No columns give an empty dict, which leaves a DataFrame unchanged."""
    assert column_dtypes() == {}
    df = pd.DataFrame({"a": [1]})
    pd.testing.assert_frame_equal(df.astype(column_dtypes()), df)


def test_column_dtypes_empty_categories():
    """A column can have no categories, for a column whose values are all missing."""
    dtype = column_dtypes({"a": []})["a"]
    assert isinstance(dtype, CategoricalDtype)
    assert len(dtype.categories) == 0
    assert pd.Series([None, None]).astype(dtype).isna().all()


def test_column_dtypes_accepts_iterators():
    """Categories and integer columns can be any iterables."""
    dtypes = column_dtypes({"a": iter("xy")}, iter(["n"]))
    assert list(dtypes) == ["a", "n"]
    assert isinstance(dtypes["a"], CategoricalDtype)
    assert dtypes["a"].categories.tolist() == ["x", "y"]


def test_column_dtypes_rejects_column_in_both():
    """A column can't be both categorical and a nullable integer."""
    with pytest.raises(ValueError, match="both"):
        column_dtypes({"a": [1, 2]}, ["a"])


def test_column_dtypes_rejects_repeated_categories():
    """Categories must be unique."""
    with pytest.raises(ValueError):
        column_dtypes({"a": ["x", "x"]})


# --- to_datetime_columns ---


def test_to_datetime_columns_single():
    """One DataFrame and one column name."""
    df = pd.DataFrame({"d": ["2023-01-01", "2023-02-01"], "x": ["2023-01-01", "a"]})
    to_datetime_columns(df, "d")
    assert df["d"].dtype.kind == "M"
    assert df["d"].dt.month.tolist() == [1, 2]
    assert df["x"].dtype.kind != "M"  # other columns are left alone


def test_to_datetime_columns_several():
    """Several DataFrames and columns."""
    dfs = [pd.DataFrame({"d": ["2023-01-01"], "e": ["2023-01-02"]}) for _ in range(2)]
    to_datetime_columns(dfs, ["d", "e"])
    assert all(df[column].dtype.kind == "M" for df in dfs for column in ["d", "e"])


def test_to_datetime_columns_empty_column():
    """An empty column becomes an empty datetime column."""
    df = pd.DataFrame({"d": pd.Series([], dtype=object)})
    to_datetime_columns(df, "d")
    assert df["d"].dtype.kind == "M"


def test_to_datetime_columns_nothing_to_convert():
    """No columns, or no DataFrames, change nothing."""
    df = pd.DataFrame({"d": ["2023-01-01"]})
    to_datetime_columns(df, [])
    to_datetime_columns([], "d")
    assert df["d"].tolist() == ["2023-01-01"]


def test_to_datetime_columns_missing_column():
    """A DataFrame without the column is a KeyError."""
    with pytest.raises(KeyError):
        to_datetime_columns(pd.DataFrame({"d": ["2023-01-01"]}), "e")
