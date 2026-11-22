"""Tests for ``mcrotk_utils.pandas_utils``' added columns: ``join_columns`` and ``add_lags``.

Some tests compare with a straightforward reference implementation (for each lag, a merge with a copy whose periods
are shifted by it), which is simpler but slower or less careful (e.g. with dtypes or the index), on random inputs.
"""

import numpy as np
import pandas as pd
import pytest

from mcrotk_utils.pandas_utils import (
    add_lags,
    join_columns,
)

# --- join_columns ---


def test_join_columns_none_or_single_name():
    """None, an empty list and a single name leave the DataFrame alone."""
    df = pd.DataFrame({"a": [1]})
    for columns, expected in [(None, None), ("a", "a"), ([], None), (["a"], "a")]:
        result, name = join_columns(df, columns)
        assert result is df
        assert name == expected


def test_join_columns_two_columns():
    """Two columns give a new column of labels, named after them; the input isn't changed."""
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    result, name = join_columns(df, ["a", "b"])
    assert name == "(a, b)"
    assert result[name].tolist() == ["(1, x)", "(2, y)"]
    assert list(df.columns) == ["a", "b"]


def test_join_columns_three_columns_and_name():
    """Any number of columns, and a chosen name."""
    df = pd.DataFrame({"a": [1], "b": [2.5], "c": pd.Categorical(["z"])})
    result, name = join_columns(df, ["a", "b", "c"], name="abc")
    assert name == "abc"
    assert result["abc"].tolist() == ["(1, 2.5, z)"]


def test_join_columns_missing_values_become_text():
    """Missing values become text rather than making the label missing."""
    df = pd.DataFrame(
        {
            "f": [np.nan],
            "o": pd.Series([None], dtype=object),
            "n": pd.array([None], dtype="Int64"),
            "c": pd.Categorical([None], categories=["z"]),
        }
    )
    result, name = join_columns(df, ["f", "o", "n", "c"])
    assert result[name].tolist() == ["(nan, None, <NA>, nan)"]


def test_join_columns_replaces_existing_column():
    """A column already named like the new one is replaced; other columns are left alone."""
    df = pd.DataFrame({"a": [1], "b": [2], "newvar": [0], "(a, b)": [0]})
    result, name = join_columns(df, ["a", "b"])
    assert result[name].tolist() == ["(1, 2)"]
    assert result["newvar"].tolist() == [0]
    assert list(result.columns) == ["a", "b", "newvar", "(a, b)"]


def test_join_columns_empty_dataframe():
    """A DataFrame with no rows gets an empty joined column."""
    df = pd.DataFrame({"a": pd.Series([], dtype="int64"), "b": pd.Series([], dtype=object)})
    result, name = join_columns(df, ["a", "b"])
    assert name in result.columns
    assert len(result) == 0


# --- add_lags ---


def reference_add_lags(df, lag_col, num_lags, other_fixed_cols):
    """A direct implementation of add_lags: for each lag, merge with a copy whose periods are shifted by it.

    (The merges reset the index, so this agrees with add_lags only for a default index.)
    """
    df_wlags = df.copy()
    merge_on_cols = [lag_col] + other_fixed_cols
    for lag in range(1, num_lags + 1):
        df_lag = df.copy()
        df_lag[lag_col] = df_lag[lag_col] + lag
        df_wlags = pd.merge(df_wlags, df_lag, on=merge_on_cols, how="left", suffixes=(None, f"_lag{lag}"))
    return df_wlags


def test_add_lags_single_series():
    """Each row gets the values from 1 and 2 periods earlier; earlier periods that don't exist are missing."""
    df = pd.DataFrame({"t": [1, 2, 3], "v": [10, 20, 30]})
    result = add_lags(df, "t", 2)
    assert list(result.columns) == ["t", "v", "v_lag1", "v_lag2"]
    np.testing.assert_array_equal(result["v_lag1"], [np.nan, 10, 20])
    np.testing.assert_array_equal(result["v_lag2"], [np.nan, np.nan, 10])


def test_add_lags_dont_cross_series():
    """Lags come from the same values of other_fixed_cols."""
    df = pd.DataFrame({"t": [1, 2, 1, 2], "store": ["a", "a", "b", "b"], "v": [1.0, 2.0, 3.0, 4.0]})
    result = add_lags(df, "t", 1, ["store"])
    np.testing.assert_array_equal(result["v_lag1"], [np.nan, 1.0, np.nan, 3.0])


def test_add_lags_gap_is_missing():
    """A missing period gives a missing lag."""
    df = pd.DataFrame({"t": [1, 3], "v": [10.0, 30.0]})
    assert add_lags(df, "t", 1)["v_lag1"].isna().all()


def test_add_lags_column_order_with_several_columns():
    """Lag columns come by lag, then in the columns' order."""
    df = pd.DataFrame({"t": [1, 2], "v": [1, 2], "w": [3, 4]})
    assert list(add_lags(df, "t", 2).columns) == ["t", "v", "w", "v_lag1", "w_lag1", "v_lag2", "w_lag2"]


def test_add_lags_keeps_index_and_row_order():
    """The result has the input's index and row order, even when rows aren't sorted by period."""
    df = pd.DataFrame({"t": [3, 1, 2], "v": [30.0, 10.0, 20.0]}, index=["c", "a", "b"])
    result = add_lags(df, "t", 1)
    assert result.index.tolist() == ["c", "a", "b"]
    np.testing.assert_array_equal(result["v_lag1"], [20.0, np.nan, 10.0])


def test_add_lags_zero_lags_gives_copy():
    """No lags gives an equal copy."""
    df = pd.DataFrame({"t": [1, 2], "v": [1, 2]})
    result = add_lags(df, "t", 0)
    pd.testing.assert_frame_equal(result, df)
    assert result is not df


def test_add_lags_integer_lags_become_floats():
    """Lags of integer columns are floats, since some are missing."""
    df = pd.DataFrame({"t": [1, 2], "v": [1, 2]})
    assert add_lags(df, "t", 1)["v_lag1"].dtype == np.float64


def test_add_lags_only_key_columns():
    """With no other columns, nothing is added."""
    df = pd.DataFrame({"t": [1, 2], "store": ["a", "a"]})
    pd.testing.assert_frame_equal(add_lags(df, "t", 2, ["store"]), df)


def test_add_lags_accepts_numpy_integer():
    """num_lags can be a NumPy integer."""
    df = pd.DataFrame({"t": [1, 2], "v": [1, 2]})
    assert "v_lag1" in add_lags(df, "t", np.int64(1)).columns


def test_add_lags_leaves_input_unchanged():
    """The input DataFrame isn't changed."""
    df = pd.DataFrame({"t": [1, 2], "v": [1, 2]})
    add_lags(df, "t", 2)
    assert list(df.columns) == ["t", "v"]
    assert df["t"].tolist() == [1, 2]


def test_add_lags_matches_reference_on_random_cases():
    """On random panels (shuffled, with gaps), the result is the reference implementation's."""
    rng = np.random.default_rng(20261114)
    for _ in range(100):
        rows = [(t, store) for store in "abc" for t in range(10) if rng.random() < 0.7]
        df = pd.DataFrame(rows, columns=["t", "store"])
        df["v"] = rng.normal(size=len(df))
        df["n"] = rng.integers(0, 100, size=len(df))
        df = df.sample(frac=1, random_state=rng).reset_index(drop=True)
        num_lags = int(rng.integers(0, 4))
        pd.testing.assert_frame_equal(
            add_lags(df, "t", num_lags, ["store"]), reference_add_lags(df, "t", num_lags, ["store"])
        )


def test_add_lags_empty_dataframe():
    """A DataFrame with no rows gets the lag columns and no rows."""
    df = pd.DataFrame({"t": pd.Series([], dtype="int64"), "store": pd.Series([], dtype=object), "v": []})
    result = add_lags(df, "t", 2, ["store"])
    assert list(result.columns) == ["t", "store", "v", "v_lag1", "v_lag2"]
    assert len(result) == 0


def test_add_lags_rejects_negative_lags():
    """num_lags below 0 is a ValueError."""
    with pytest.raises(ValueError, match="at least 0"):
        add_lags(pd.DataFrame({"t": [1], "v": [1]}), "t", -1)


def test_add_lags_rejects_non_integer_lags():
    """A float num_lags is a TypeError, even a whole number."""
    with pytest.raises(TypeError):
        add_lags(pd.DataFrame({"t": [1], "v": [1]}), "t", 1.0)  # type: ignore[arg-type]


def test_add_lags_rejects_repeated_keys():
    """Two rows with the same period and fixed columns make lags ambiguous: a ValueError."""
    df = pd.DataFrame({"t": [1, 1], "store": ["a", "a"], "v": [1, 2]})
    with pytest.raises(ValueError, match="only once"):
        add_lags(df, "t", 1, ["store"])
