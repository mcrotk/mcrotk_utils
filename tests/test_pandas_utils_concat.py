"""Tests for ``mcrotk_utils.pandas_utils``' ``combine_dataframes``.

Some tests compare with a straightforward reference implementation (concatenating pairwise), which is simpler but
slower or less careful (e.g. with dtypes or the index), on random inputs.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import CategoricalDtype

from mcrotk_utils.pandas_utils import (
    combine_dataframes,
)


def categories(series):
    """Return a categorical Series' categories as a list."""
    return series.cat.categories.tolist()


# --- combine_dataframes ---


def cat_df(values, cats, ordered=False, index=None, **other_columns):
    """Build a DataFrame with a categorical column ``cat`` and any other columns."""
    data = {"cat": pd.Categorical(values, categories=cats, ordered=ordered), **other_columns}
    return pd.DataFrame(data, index=index)


def reference_combine(dfs):
    """A direct implementation of combine_dataframes: concatenate pairwise, with the union of the categories.

    Each categorical column gets both sides' categories in order of appearance, ordered only if both are.
    """
    result = dfs[0]
    for df in dfs[1:]:
        dtypes = {}
        for column in result.columns:
            if result[column].dtype.name == "category" and df[column].dtype.name == "category":
                cat_list1, cat_list2 = list(result[column].dtype.categories), list(df[column].dtype.categories)
                cat_list_both = cat_list1.copy()
                cat_list_both.extend([x for x in cat_list2 if x not in cat_list1])
                ordered = result[column].cat.ordered and df[column].cat.ordered
                dtypes[column] = CategoricalDtype(categories=cat_list_both, ordered=ordered)
        result = pd.concat([result, df]).astype(dtypes)
    return result


def test_plain_concat_loses_categories():
    """The problem combine_dataframes solves: pd.concat makes the column non-categorical."""
    combined = pd.concat([cat_df(["a"], ["a"]), cat_df(["b"], ["b"])])
    assert not isinstance(combined["cat"].dtype, CategoricalDtype)


def test_combine_keeps_categories_in_order_of_appearance():
    """The first DataFrame's categories come first, in their order, then the new ones of the next."""
    result = combine_dataframes(cat_df(["a", "b"], ["b", "a"]), cat_df(["c", "a"], ["c", "a", "d"]))
    assert categories(result["cat"]) == ["b", "a", "c", "d"]
    assert result["cat"].tolist() == ["a", "b", "c", "a"]


def test_combine_three_dataframes():
    """Any number of DataFrames can be combined."""
    result = combine_dataframes(cat_df(["a"], ["a"]), cat_df(["b"], ["b"]), cat_df(["c", "a"], ["c", "a"]))
    assert categories(result["cat"]) == ["a", "b", "c"]
    assert result["cat"].tolist() == ["a", "b", "c", "a"]


def test_combine_numeric_categories():
    """Categories needn't be strings."""
    result = combine_dataframes(cat_df([1, 2], [1, 2]), cat_df([3], [3]))
    assert categories(result["cat"]) == [1, 2, 3]
    assert result["cat"].tolist() == [1, 2, 3]


@pytest.mark.parametrize(("ordered1", "ordered2", "expected"), [(True, True, True), (True, False, False)])
def test_combine_ordered_only_if_ordered_everywhere(ordered1, ordered2, expected):
    """The combined column is ordered if it's ordered in all the DataFrames."""
    result = combine_dataframes(cat_df(["a"], ["a"], ordered1), cat_df(["b"], ["b"], ordered2))
    assert result["cat"].cat.ordered == expected


def test_combine_matches_reference_on_random_cases():
    """On random DataFrames with the same columns, the result is the reference implementation's.

    The DataFrames all have rows: with an empty one, the reference's ``astype`` can keep the wrong category order,
    depending on the pandas version (see test_combine_keeps_first_category_order_for_same_categories).
    """
    rng = np.random.default_rng(20261114)
    letters = list("abcdefg")
    for _ in range(200):
        ordered = bool(rng.integers(2))
        dfs = []
        for _ in range(int(rng.integers(2, 5))):
            cats = list(rng.permutation(letters)[: rng.integers(1, len(letters) + 1)])
            num_rows = int(rng.integers(1, 6))
            values = list(rng.choice(cats, size=num_rows))
            dfs.append(cat_df(values, cats, ordered, num=rng.integers(0, 100, size=num_rows)))
        pd.testing.assert_frame_equal(combine_dataframes(*dfs), reference_combine(dfs))


@pytest.mark.parametrize("first_values", [["a"], []])
def test_combine_keeps_first_category_order_for_same_categories(first_values):
    """The same categories in another order keep the first DataFrame's order, even when it has no rows.

    pandas treats unordered dtypes with the same categories as equal whatever their order, so astype alone
    would keep the order of whichever DataFrame pd.concat took the dtype from.
    """
    result = combine_dataframes(cat_df(first_values, ["b", "a"]), cat_df(["a", "b"], ["a", "b"]))
    assert categories(result["cat"]) == ["b", "a"]
    assert result["cat"].tolist() == [*first_values, "a", "b"]


def test_combine_single_dataframe_gives_equal_copy():
    """One DataFrame gives a copy of it."""
    df = cat_df(["a", "b"], ["a", "b"], num=[1, 2])
    result = combine_dataframes(df)
    pd.testing.assert_frame_equal(result, df)
    assert result is not df


def test_combine_needs_a_dataframe():
    """No DataFrames is a ValueError."""
    with pytest.raises(ValueError, match="at least one"):
        combine_dataframes()


def test_combine_keeps_index_unless_ignored():
    """The inputs' index values are kept, unless ignore_index is set."""
    df1, df2 = cat_df(["a"], ["a"], index=[5]), cat_df(["b"], ["b"], index=[5])
    assert combine_dataframes(df1, df2).index.tolist() == [5, 5]
    assert combine_dataframes(df1, df2, ignore_index=True).index.tolist() == [0, 1]


def test_combine_leaves_inputs_unchanged():
    """The input DataFrames keep their own categories."""
    df1, df2 = cat_df(["a"], ["a"]), cat_df(["b"], ["b"])
    combine_dataframes(df1, df2)
    assert categories(df1["cat"]) == ["a"]
    assert categories(df2["cat"]) == ["b"]


def test_combine_column_missing_from_one_dataframe():
    """Rows from a DataFrame without the column get missing values; the column stays categorical."""
    result = combine_dataframes(cat_df(["a"], ["a"], num=[1]), pd.DataFrame({"num": [2]}))
    assert categories(result["cat"]) == ["a"]
    assert result["cat"].isna().tolist() == [False, True]
    assert result["num"].tolist() == [1, 2]


def test_combine_warns_about_mismatched_column():
    """A column categorical in one DataFrame and not in another gives a warning and is left to pd.concat."""
    df1 = cat_df(["a"], ["a"], other=pd.Categorical(["x"]))
    df2 = pd.DataFrame({"cat": ["b"], "other": pd.Categorical(["y"])})
    with pytest.warns(UserWarning, match="'cat' is categorical in some DataFrames"):
        result = combine_dataframes(df1, df2)
    assert not isinstance(result["cat"].dtype, CategoricalDtype)
    assert result["cat"].tolist() == ["a", "b"]
    assert categories(result["other"]) == ["x", "y"]  # other columns are still combined


def test_combine_doesnt_warn_when_columns_match():
    """No warning when each column is categorical everywhere or nowhere."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        combine_dataframes(cat_df(["a"], ["a"], num=[1]), cat_df(["b"], ["b"], num=[2]))


# Empty DataFrames


def test_combine_starting_from_empty_dataframe():
    """Combining onto pd.DataFrame() (e.g. at the start of a loop) gives the other DataFrame unchanged."""
    df = cat_df(["a", "b"], ["b", "a"], True, num=[1, 2])
    pd.testing.assert_frame_equal(combine_dataframes(pd.DataFrame(), df), df)
    pd.testing.assert_frame_equal(combine_dataframes(df, pd.DataFrame()), df)


def test_combine_empty_object_columns_dont_change_dtypes():
    """A DataFrame with no rows and object columns changes no dtypes, and doesn't count as a mismatch."""
    df = cat_df(["a"], ["a"], num=[1])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = combine_dataframes(pd.DataFrame(columns=["cat", "num"]), df)
    pd.testing.assert_frame_equal(result, df)


def test_combine_empty_dataframe_adds_its_categories():
    """A DataFrame with no rows still contributes its categories, in their place in the order."""
    result = combine_dataframes(cat_df([], ["z", "a"]), cat_df(["a", "b"], ["a", "b"]))
    assert categories(result["cat"]) == ["z", "a", "b"]
    assert result["cat"].tolist() == ["a", "b"]


def test_combine_empty_dataframe_adds_its_columns():
    """Columns that only a DataFrame with no rows has appear, with missing values, keeping categorical."""
    empty = pd.DataFrame({"num": pd.Series([], dtype="int64"), "extra": pd.Categorical([], categories=["e"])})
    result = combine_dataframes(pd.DataFrame({"num": [1, 2]}), empty)
    assert list(result.columns) == ["num", "extra"]
    assert result["num"].tolist() == [1, 2]
    assert result["num"].dtype == np.int64
    assert categories(result["extra"]) == ["e"]
    assert result["extra"].isna().all()


def test_combine_all_dataframes_empty():
    """DataFrames that all have no rows give one with no rows, all the columns and all the categories."""
    result = combine_dataframes(cat_df([], ["a"], num=[]), cat_df([], ["b"], num=[]))
    assert len(result) == 0
    assert list(result.columns) == ["cat", "num"]
    assert categories(result["cat"]) == ["a", "b"]


def test_combine_dataframes_without_columns():
    """DataFrames with no columns or rows give an empty DataFrame."""
    result = combine_dataframes(pd.DataFrame(), pd.DataFrame())
    assert result.shape == (0, 0)


def test_combine_column_with_no_categories():
    """A categorical column with no categories (all missing) takes on the other DataFrame's categories."""
    result = combine_dataframes(cat_df([None], []), cat_df(["a"], ["a"]))
    assert categories(result["cat"]) == ["a"]
    assert result["cat"].isna().tolist() == [True, False]
