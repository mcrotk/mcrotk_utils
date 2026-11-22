"""Tests for ``mcrotk_utils.pandas_utils``' ``merge_with_preferences``.

Some tests compare with a straightforward reference implementation (a plain merge, then ``np.where`` on its
indicator), which is simpler but slower or less careful (e.g. with dtypes or the index), on random inputs.
"""

from typing import Literal

import numpy as np
import pandas as pd
import pytest

from mcrotk_utils.pandas_utils import (
    merge_with_preferences,
)

# The ways of merging that merge_with_preferences accepts
HOWS: list[Literal["left", "right", "outer", "inner"]] = ["left", "right", "outer", "inner"]


def categories(series):
    """Return a categorical Series' categories as a list."""
    return series.cat.categories.tolist()


# --- merge_with_preferences ---


def reference_merge_with_preferences(df1, df2, how, preference_left, preference_right):
    """A direct implementation of merge_with_preferences (np.where on the indicator), without renames or dates."""
    suffixes = ("_left", "_right")
    preferred = {"left": "left", "right": "right"}
    other = {"left": "right", "right": "left"}
    pref_list = {"left": preference_left, "right": preference_right}
    mdf = df1.merge(df2, how=how, left_index=True, right_index=True, indicator=True, suffixes=suffixes)
    for pref in ["left", "right"]:
        for col in pref_list[pref]:
            mdf[col] = np.where(
                mdf["_merge"] == f"{other[pref]}_only",
                mdf[f"{col}_{other[pref]}"],
                mdf[f"{col}_{preferred[pref]}"],
            )
    return mdf.drop(
        columns=["_merge"]
        + [
            pref + suffix
            for pref_list in [preference_left, preference_right]
            for pref in pref_list
            for suffix in suffixes
        ]
    )


def left_and_right():
    """Two DataFrames with overlapping indexes: rows 1 and 2 on the left, 2 and 3 on the right."""
    df1 = pd.DataFrame({"p": [10.0, 20.0], "q": [11.0, 21.0], "x": [1, 2], "y": ["l1", "l2"]}, index=[1, 2])
    df2 = pd.DataFrame({"p": [200.0, 300.0], "q": [201.0, 301.0], "x": [3, 4]}, index=[2, 3])
    return df1, df2


def test_merge_takes_preferred_values():
    """Rows in both take the preferred side's value; rows only in one side take that side's value."""
    df1, df2 = left_and_right()
    result = merge_with_preferences(df1, df2, "outer", preference_left=["p"], preference_right=["q"])
    assert result.index.tolist() == [1, 2, 3]
    assert result["p"].tolist() == [10.0, 20.0, 300.0]
    assert result["q"].tolist() == [11.0, 201.0, 301.0]


def test_merge_column_layout():
    """Other columns in both get suffixes, columns in one keep their names, preferred columns come last."""
    df1, df2 = left_and_right()
    result = merge_with_preferences(df1, df2, "outer", preference_left=["p"], preference_right=["q"])
    assert list(result.columns) == ["x_left", "y", "x_right", "p", "q"]


@pytest.mark.parametrize(("how", "index"), [("left", [1, 2]), ("right", [2, 3]), ("inner", [2]), ("outer", [1, 2, 3])])
def test_merge_how(how, index):
    """``how`` decides which rows are kept, as in pd.merge."""
    df1, df2 = left_and_right()
    result = merge_with_preferences(df1, df2, how, preference_left=["p"])
    assert result.index.tolist() == index


def test_merge_takes_preferred_value_even_when_missing():
    """A row in both takes the preferred value even when it's missing, rather than the other side's."""
    df1, df2 = left_and_right()
    df1.loc[2, "p"] = np.nan
    result = merge_with_preferences(df1, df2, "outer", preference_left=["p"])
    assert result["p"].isna().tolist() == [False, True, False]


def test_merge_without_preferences_is_plain_merge():
    """With no preferred columns, the result is pd.merge on the indexes with the suffixes."""
    df1, df2 = left_and_right()
    expected = df1.merge(df2, how="outer", left_index=True, right_index=True, suffixes=("_left", "_right"))
    pd.testing.assert_frame_equal(merge_with_preferences(df1, df2, "outer"), expected)


def test_merge_matches_reference_on_random_cases():
    """On random numeric DataFrames, the result is the reference implementation's."""
    rng = np.random.default_rng(20261114)
    for _ in range(200):
        dfs = []
        for _ in range(2):
            index = np.sort(rng.choice(8, size=int(rng.integers(1, 8)), replace=False))
            columns = {name: rng.normal(size=len(index)) for name in ["p", "q", "r"]}
            dfs.append(pd.DataFrame(columns, index=index))
        how = HOWS[int(rng.integers(len(HOWS)))]
        left, right = list(rng.permutation(["p", "q", "r"])[: rng.integers(0, 4)]), []
        if left and rng.integers(2):
            right = [left.pop()]
        args = (dfs[0], dfs[1], how, left, right)
        pd.testing.assert_frame_equal(merge_with_preferences(*args), reference_merge_with_preferences(*args))


def test_merge_keeps_categorical_with_different_categories():
    """A categorical column keeps being categorical, with the categories of both sides."""
    df1 = pd.DataFrame({"c": pd.Categorical(["a", "b"])}, index=[1, 2])
    df2 = pd.DataFrame({"c": pd.Categorical(["z", "y"], categories=["z", "y"])}, index=[2, 3])
    result = merge_with_preferences(df1, df2, "outer", preference_right=["c"])
    assert categories(result["c"]) == ["z", "y", "a", "b"]
    assert result["c"].tolist() == ["a", "z", "y"]


def test_merge_keeps_timezone_and_nullable_dtypes():
    """Timezone-aware and nullable integer columns keep their dtypes (np.where would make them objects)."""
    df1 = pd.DataFrame(
        {"t": pd.to_datetime(["2023-01-01", "2023-01-02"]).tz_localize("UTC"), "n": pd.array([1, 2], dtype="Int64")},
        index=[1, 2],
    )
    df2 = pd.DataFrame(
        {"t": pd.to_datetime(["2024-01-02", "2024-01-03"]).tz_localize("UTC"), "n": pd.array([20, 30], dtype="Int64")},
        index=[2, 3],
    )
    result = merge_with_preferences(df1, df2, "outer", preference_left=["t", "n"])
    assert result["t"].dt.tz is not None
    assert result["t"].dt.year.tolist() == [2023, 2023, 2024]
    assert result["n"].dtype == pd.Int64Dtype()
    assert result["n"].tolist() == [1, 2, 30]


def test_merge_renames_first():
    """Renames apply before merging, and the preference lists use the new names."""
    df1 = pd.DataFrame({"price_l": [1.0]}, index=[1])
    df2 = pd.DataFrame({"price_r": [2.0]}, index=[1])
    result = merge_with_preferences(
        df1,
        df2,
        "outer",
        preference_right=["price"],
        rename_left={"price_l": "price"},
        rename_right={"price_r": "price"},
    )
    assert list(result.columns) == ["price"]
    assert result["price"].tolist() == [2.0]


@pytest.mark.parametrize("datetime_cols", ["d", ["d"]])
def test_merge_converts_datetime_columns(datetime_cols):
    """Columns in datetime_cols are converted in both DataFrames, so strings and dates combine."""
    df1 = pd.DataFrame({"d": ["2023-01-01"]}, index=[1])
    df2 = pd.DataFrame({"d": pd.to_datetime(["2024-01-01"])}, index=[2])
    result = merge_with_preferences(df1, df2, "outer", preference_left=["d"], datetime_cols=datetime_cols)
    assert result["d"].dtype.kind == "M"
    assert result["d"].dt.year.tolist() == [2023, 2024]


def test_merge_datetime_column_missing_raises():
    """A column in datetime_cols that a DataFrame lacks is a KeyError."""
    df1 = pd.DataFrame({"d": ["2023-01-01"]}, index=[1])
    df2 = pd.DataFrame({"e": ["2023-01-01"]}, index=[1])
    with pytest.raises(KeyError):
        merge_with_preferences(df1, df2, "outer", datetime_cols=["d"])


def test_merge_preferred_column_missing_from_one_side():
    """A preferred column one side lacks is all missing there, with a warning."""
    df1, df2 = left_and_right()
    with pytest.warns(UserWarning, match="'y' not found"):
        result = merge_with_preferences(df1, df2, "outer", preference_right=["y"])
    # rows in the right take its (missing) value; the row only on the left keeps the left's
    assert result["y"].tolist()[0] == "l1"
    assert result["y"].isna().tolist() == [False, True, True]


@pytest.mark.parametrize("side", ["left", "right"])
def test_merge_preferred_column_missing_keeps_dtype(side):
    """A preferred column that one side lacks keeps the other side's dtype, e.g. categorical.

    That holds whichever side is preferred, including the side that lacks the column.
    """
    df1 = pd.DataFrame({"c": pd.Categorical(["a", "b"])}, index=[1, 2])
    df2 = pd.DataFrame({"other": [0]}, index=[3])
    preference_left, preference_right = (["c"], []) if side == "left" else ([], ["c"])
    with pytest.warns(UserWarning):
        result = merge_with_preferences(df1, df2, "outer", preference_left, preference_right)
    assert categories(result["c"]) == ["a", "b"]
    assert result["c"].isna().tolist() == [False, False, True]


def test_merge_preferred_column_missing_from_both_sides():
    """A preferred column neither side has is all missing, with a warning for each side."""
    df1, df2 = left_and_right()
    with pytest.warns(UserWarning) as record:
        result = merge_with_preferences(df1, df2, "outer", preference_left=["absent"])
    assert sum("'absent' not found" in str(warning.message) for warning in record) == 2
    assert result["absent"].isna().all()


def test_merge_rejects_column_in_both_preferences():
    """A column can't be preferred from both sides."""
    df1, df2 = left_and_right()
    with pytest.raises(ValueError, match="both"):
        merge_with_preferences(df1, df2, "outer", preference_left=["p"], preference_right=["p"])


def test_merge_leaves_inputs_unchanged():
    """Renames, datetime conversions and added columns don't reach the inputs."""
    df1 = pd.DataFrame({"d": ["2023-01-01"], "a": [1]}, index=[1])
    df2 = pd.DataFrame({"d": ["2024-01-01"]}, index=[2])
    copies = df1.copy(), df2.copy()
    with pytest.warns(UserWarning):
        merge_with_preferences(df1, df2, "outer", preference_left=["a"], rename_right={"d": "d"}, datetime_cols=["d"])
    pd.testing.assert_frame_equal(df1, copies[0])
    pd.testing.assert_frame_equal(df2, copies[1])


def test_merge_with_existing_merge_column():
    """A column named _merge (the name pd.merge uses for its indicator) is kept as data."""
    df1 = pd.DataFrame({"_merge": ["data"], "p": [1.0]}, index=[1])
    df2 = pd.DataFrame({"p": [2.0]}, index=[2])
    result = merge_with_preferences(df1, df2, "outer", preference_left=["p"])
    assert list(result.columns) == ["_merge", "p"]
    assert result["p"].tolist() == [1.0, 2.0]


# Empty DataFrames


def test_merge_with_empty_left():
    """With no rows on the left, an outer merge has the right's rows and values."""
    df1, df2 = left_and_right()
    result = merge_with_preferences(df1.iloc[:0], df2, "outer", preference_left=["p"], preference_right=["q"])
    assert result.index.tolist() == [2, 3]
    assert result["p"].tolist() == [200.0, 300.0]
    assert result["q"].tolist() == [201.0, 301.0]


def test_merge_with_both_empty():
    """Two DataFrames with no rows give one with no rows and the merged columns."""
    df1, df2 = left_and_right()
    result = merge_with_preferences(df1.iloc[:0], df2.iloc[:0], "outer", preference_left=["p"])
    assert len(result) == 0
    assert set(result.columns) == {"q_left", "q_right", "x_left", "x_right", "y", "p"}


def test_merge_with_empty_dataframes_without_columns():
    """DataFrames with no columns or rows give an empty DataFrame (with a warning for a preferred column)."""
    with pytest.warns(UserWarning):
        result = merge_with_preferences(pd.DataFrame(), pd.DataFrame(), "outer", preference_left=["p"])
    assert len(result) == 0
    assert list(result.columns) == ["p"]
