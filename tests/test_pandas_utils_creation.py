"""Tests for ``mcrotk_utils.pandas_utils``' ways of making DataFrames: ``as_dataframe`` and ``matrices_to_long_df``.

Some tests compare with a straightforward reference implementation (melting and merging each matrix), which is
simpler but slower or less careful (e.g. with dtypes or the index), on random inputs.
"""

import io

import numpy as np
import pandas as pd
import pytest

from mcrotk_utils.pandas_utils import (
    as_dataframe,
    matrices_to_long_df,
)

# A small CSV file's contents
CSV = "a,b\n1,x\n2,y\n"


# --- as_dataframe ---


def test_as_dataframe_returns_dataframe_itself():
    """A DataFrame is returned as it is, not copied."""
    df = pd.DataFrame({"a": [1]})
    assert as_dataframe(df) is df


def test_as_dataframe_returns_empty_dataframe_itself():
    """An empty DataFrame is a DataFrame too."""
    df = pd.DataFrame()
    assert as_dataframe(df) is df


@pytest.mark.parametrize("kind", ["str", "Path", "text file", "binary file"])
def test_as_dataframe_reads_csv_from_path_or_file(tmp_path, kind):
    """Paths (as strings or Paths) and open files (text or binary) are read as CSV."""
    path = tmp_path / "data.csv"
    path.write_text(CSV)
    if kind == "str":
        result = as_dataframe(str(path))
    elif kind == "Path":
        result = as_dataframe(path)
    else:
        with path.open("r" if kind == "text file" else "rb") as file:
            result = as_dataframe(file)
    assert result["a"].tolist() == [1, 2]
    assert result["b"].tolist() == ["x", "y"]


@pytest.mark.parametrize("file", [io.StringIO(CSV), io.BytesIO(CSV.encode())])
def test_as_dataframe_reads_in_memory_files(file):
    """Anything with a read method counts as an open file."""
    assert as_dataframe(file)["a"].tolist() == [1, 2]


def test_as_dataframe_header_only_gives_columns_and_no_rows():
    """A CSV file with a header and no rows gives an empty DataFrame with those columns."""
    result = as_dataframe(io.StringIO("a,b\n"))
    assert list(result.columns) == ["a", "b"]
    assert len(result) == 0


def test_as_dataframe_completely_empty_file_raises():
    """A completely empty file is pandas' EmptyDataError."""
    with pytest.raises(pd.errors.EmptyDataError):
        as_dataframe(io.StringIO(""))


def test_as_dataframe_missing_file_raises(tmp_path):
    """A path to no file is a FileNotFoundError, from pd.read_csv."""
    with pytest.raises(FileNotFoundError):
        as_dataframe(tmp_path / "missing.csv")


@pytest.mark.parametrize("source", [3, None, [1, 2], {"a": [1]}])
def test_as_dataframe_rejects_other_types(source):
    """Anything other than a DataFrame, path or file is a TypeError."""
    with pytest.raises(TypeError, match="DataFrame, a path or an open file"):
        as_dataframe(source)


# --- matrices_to_long_df ---


def reference_long_df(matrices, row_name, column_name):
    """A direct implementation of matrices_to_long_df: melt each matrix, then merge on row and column."""
    frames = [
        pd.DataFrame(matrix)
        .melt(var_name=column_name, value_name=name, ignore_index=False)
        .reset_index()
        .rename(columns={"index": row_name})
        for name, matrix in matrices.items()
    ]
    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on=[column_name, row_name])
    return result


def test_long_df_order():
    """A row per position: all rows of matrix column 0, then of column 1."""
    result = matrices_to_long_df({"v": [[1, 2], [3, 4]]}, row_name="r", column_name="c")
    assert list(result.columns) == ["r", "c", "v"]
    assert result["r"].tolist() == [0, 1, 0, 1]
    assert result["c"].tolist() == [0, 0, 1, 1]
    assert result["v"].tolist() == [1, 3, 2, 4]
    pd.testing.assert_index_equal(result.index, pd.RangeIndex(4))


def test_long_df_matches_reference_on_random_cases():
    """For random numbers of matrices, shapes and dtypes, the result is the melt-and-merge method's."""
    rng = np.random.default_rng(20261114)
    for _ in range(100):
        shape = tuple(rng.integers(1, 6, size=2))
        matrices = {}
        for number in range(int(rng.integers(1, 5))):
            kind = int(rng.integers(3))
            matrix = [rng.integers(-5, 5, size=shape), rng.normal(size=shape), rng.integers(0, 2, size=shape) == 1]
            matrices[f"m{number}"] = matrix[kind]
        result = matrices_to_long_df(matrices, row_name="r", column_name="c")
        pd.testing.assert_frame_equal(result, reference_long_df(matrices, "r", "c"))


@pytest.mark.parametrize("dtype", [np.int8, np.float32, bool])
def test_long_df_keeps_dtypes(dtype):
    """Each matrix's column keeps its dtype; row and column numbers are int64."""
    result = matrices_to_long_df({"v": np.zeros((2, 2), dtype=dtype)}, row_name="r", column_name="c")
    assert result["v"].dtype == dtype
    assert result["r"].dtype == np.int64
    assert result["c"].dtype == np.int64


@pytest.mark.parametrize("shape", [(0, 3), (2, 0), (0, 0)])
def test_long_df_empty_matrices(shape):
    """Matrices with no rows or no columns give no rows, with the columns and dtypes still there."""
    matrices = {"i": np.zeros(shape, dtype=int), "f": np.zeros(shape)}
    result = matrices_to_long_df(matrices, row_name="r", column_name="c")
    assert len(result) == 0
    assert list(result.columns) == ["r", "c", "i", "f"]
    assert result.dtypes.tolist() == [np.int64, np.int64, np.zeros(0, dtype=int).dtype, np.float64]


@pytest.mark.parametrize(
    ("matrices", "row_name", "match"),
    [
        ({}, "r", "at least one"),
        ({"v": [1, 2]}, "r", "2-D"),
        ({"v": np.zeros((2, 2, 2))}, "r", "2-D"),
        ({"v": np.zeros((2, 2)), "w": np.zeros((2, 3))}, "r", "same shape"),
        ({"v": np.zeros((2, 2))}, "c", "differ"),
        ({"r": np.zeros((2, 2))}, "r", "differ"),
    ],
)
def test_long_df_errors(matrices, row_name, match):
    """No matrices, a matrix that isn't 2-D, different shapes, or a repeated name is a ValueError."""
    with pytest.raises(ValueError, match=match):
        matrices_to_long_df(matrices, row_name=row_name, column_name="c")
