"""Making DataFrames from other things: CSV files, and matrices (private).

Public through ``mcrotk_utils.pandas_utils``.
"""

import os
from collections.abc import Mapping
from typing import IO

import numpy as np
import numpy.typing as npt
import pandas as pd


def as_dataframe(source: pd.DataFrame | str | os.PathLike[str] | IO[str] | IO[bytes]) -> pd.DataFrame:
    """Return ``source`` if it's a DataFrame, or read it as a CSV file if it's a path or an open file.

    Lets a function take either a DataFrame or a CSV file as its input.

    **Arguments.**

    - ``source``: a DataFrame, or a path (``str`` or ``pathlib.Path``) to a CSV file, or an open CSV file
      (text or binary).

    **Result.**

    - A DataFrame is returned as it is: the same object, not a copy.
    - A path or an open file is read with ``pd.read_csv``. A file with a header but no rows gives a DataFrame
      with those columns and no rows.

    **Errors.**

    - ``TypeError``: ``source`` is none of these.
    - Whatever ``pd.read_csv`` raises, e.g. ``FileNotFoundError`` for a missing file, or
      ``pd.errors.EmptyDataError`` for a completely empty file.
    """
    if isinstance(source, pd.DataFrame):
        return source
    # anything with a read method counts as an open file, as it does for pd.read_csv
    if isinstance(source, str | os.PathLike) or hasattr(source, "read"):
        return pd.read_csv(source)
    raise TypeError(f"Expected a DataFrame, a path or an open file, not {type(source).__name__}.")


def matrices_to_long_df(matrices: Mapping[str, npt.ArrayLike], *, row_name: str, column_name: str) -> pd.DataFrame:
    """Turn matrices of the same shape into one long DataFrame, with a row per matrix position.

    E.g. matrices of assignments and outcomes, with a row per time period and a column per unit, become a
    DataFrame with columns for the period, the unit, the assignment and the outcome.

    **Arguments.**

    - ``matrices``: the matrices by name, any number of them (at least one), all 2-D and of the same shape.
      Each can be a NumPy array or anything ``np.asarray`` accepts.
    - ``row_name``, ``column_name``: names for the columns holding each position's row and column number.

    **Result.**

    - A DataFrame with the columns ``row_name``, ``column_name`` and then one per matrix, in order, and a row
      for each position: all the rows of matrix column 0, then of column 1, and so on (as
      ``DataFrame.melt`` orders them). Its index is the default one (0, 1, 2, ...).
    - Row and column numbers are 64-bit integers; each matrix's column keeps the matrix's dtype.
    - Matrices with no rows or no columns give a DataFrame with those columns and no rows.

    **Errors.**

    - ``ValueError``: no matrices, a matrix that isn't 2-D, matrices of different shapes, or names that
      aren't all different.
    """
    arrays = {name: np.asarray(matrix) for name, matrix in matrices.items()}
    _check_matrices(arrays, row_name, column_name)
    num_rows, num_columns = next(iter(arrays.values())).shape
    data = {
        row_name: np.tile(np.arange(num_rows, dtype=np.int64), num_columns),
        column_name: np.repeat(np.arange(num_columns, dtype=np.int64), num_rows),
    }
    # column-major ("F") order lists each matrix column's rows in turn, matching the position columns above
    data.update({name: array.ravel(order="F") for name, array in arrays.items()})
    return pd.DataFrame(data)


def _check_matrices(arrays: Mapping[str, npt.NDArray[np.generic]], row_name: str, column_name: str) -> None:
    """Raise ``ValueError`` unless there are matrices, all 2-D and of one shape, and all the names differ."""
    if not arrays:
        raise ValueError("matrices_to_long_df needs at least one matrix.")
    shapes = {name: array.shape for name, array in arrays.items()}
    if any(len(shape) != 2 for shape in shapes.values()):
        raise ValueError(f"The matrices must be 2-D; got shapes {shapes}.")
    if len(set(shapes.values())) > 1:
        raise ValueError(f"The matrices must have the same shape; got {shapes}.")
    names = [row_name, column_name, *arrays]
    if len(set(names)) < len(names):
        raise ValueError(f"The row, column and matrix names must all differ; got {names}.")
