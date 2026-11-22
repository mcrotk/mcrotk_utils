"""Stacking possibly-sparse matrices side by side, one above another, or in a grid of blocks (private).

Public through ``mcrotk_utils.sparse_utils``.

NumPy's stacking functions can't take sparse blocks (``np.hstack`` doesn't raise an error, but returns an array of
objects), and scipy's ``hstack`` and ``vstack`` raise an error unless some block is sparse. The functions here use
scipy's when any block is sparse, and NumPy's otherwise, so the result is sparse exactly when some block is.
"""

from collections.abc import Callable, Sequence
from typing import Any, cast, get_args, overload

import numpy as np
import numpy.typing as npt
import scipy.sparse
from scipy.sparse import issparse

from mcrotk_utils.sparse_utils._typing import PsMatrix, SparseArray, SparseFormat

# scipy's sparse formats, for checking a format at run time
_SPARSE_FORMATS: frozenset[str] = frozenset(get_args(SparseFormat.__value__))

# --- Public functions ------------------------------------------------------------------------------------------


@overload
def ps_hstack(blocks: Sequence[npt.NDArray[Any]], *, format: SparseFormat | None = None) -> npt.NDArray[Any]: ...
@overload
def ps_hstack(blocks: Sequence[SparseArray], *, format: SparseFormat | None = None) -> SparseArray: ...
@overload
def ps_hstack(blocks: Sequence[PsMatrix], *, format: SparseFormat | None = None) -> PsMatrix: ...
def ps_hstack(blocks: Sequence[PsMatrix], *, format: SparseFormat | None = None) -> PsMatrix:
    """Stack 2-D blocks side by side: sparse if any block is sparse, dense otherwise.

    **Arguments.**

    - ``blocks``: the blocks, left to right (at least one): NumPy arrays and scipy sparse arrays, in any mix and
      any sparse formats. Each must be 2-D, and all must have the same number of rows.
    - ``format``: the sparse format of the result when it's sparse (e.g. ``"csr"``), as scipy's
      ``format`` arguments take it. ``None`` (the default) leaves the choice to ``scipy.sparse.hstack``, which
      gives CSR when every block is CSR, CSC when every block is CSC, and COO otherwise. Ignored for a dense
      result, since whether the result is sparse depends on the blocks; but an unknown format raises an error
      either way.

    **Result.**

    - If any block is sparse, a scipy sparse array (``scipy.sparse.hstack``); otherwise a new NumPy array
      (``np.hstack``).
    - Either way, its dtype is the blocks' common dtype, and the values are the same.

    **Errors.**

    - ``ValueError`` if there are no blocks, if a block isn't 2-D, if the numbers of rows differ, or if
      ``format`` isn't one of scipy's formats.
    - ``TypeError`` if a block is ``None`` (only ``ps_block`` takes ``None``, for a block of zeros).
    """
    return _stack(blocks, format, sparse_stack=scipy.sparse.hstack, dense_stack=np.hstack)


@overload
def ps_vstack(blocks: Sequence[npt.NDArray[Any]], *, format: SparseFormat | None = None) -> npt.NDArray[Any]: ...
@overload
def ps_vstack(blocks: Sequence[SparseArray], *, format: SparseFormat | None = None) -> SparseArray: ...
@overload
def ps_vstack(blocks: Sequence[PsMatrix], *, format: SparseFormat | None = None) -> PsMatrix: ...
def ps_vstack(blocks: Sequence[PsMatrix], *, format: SparseFormat | None = None) -> PsMatrix:
    """Stack 2-D blocks one above another: sparse if any block is sparse, dense otherwise.

    The same as ``ps_hstack``, but stacking vertically: the blocks go top to bottom, must all have the same number
    of columns, and the result comes from ``scipy.sparse.vstack`` or ``np.vstack``. ``format`` works the same way
    (with ``None``, CSR when every block is CSR, CSC when every block is CSC, COO otherwise), as do the errors.
    """
    return _stack(blocks, format, sparse_stack=scipy.sparse.vstack, dense_stack=np.vstack)


@overload
def ps_block(
    blocks: Sequence[Sequence[npt.NDArray[Any] | None]], *, format: SparseFormat | None = None
) -> npt.NDArray[Any]: ...
@overload
def ps_block(blocks: Sequence[Sequence[PsMatrix | None]], *, format: SparseFormat | None = None) -> PsMatrix: ...
def ps_block(blocks: Sequence[Sequence[PsMatrix | None]], *, format: SparseFormat | None = None) -> PsMatrix:
    """Assemble a matrix from a grid of 2-D blocks, where ``None`` is a block of zeros: sparse if any block is
    sparse, dense otherwise.

    ``ps_block([[A, B], [C, None]])`` is the matrix with ``A`` and ``B`` side by side on top, and ``C`` and a block
    of zeros below them, as in ``np.block`` and ``scipy.sparse.block_array``.

    **Arguments.**

    - ``blocks``: the grid, as a sequence of rows (lists or tuples), each a sequence of blocks. Every row must have
      the same number of blocks (at least one row of at least one block). Each block is a 2-D NumPy array, a 2-D
      scipy sparse array in any format, or ``None``. Blocks in the same row of the grid must have the same number
      of rows, and blocks in the same column the same number of columns.
    - ``format``: as for ``ps_hstack``: the sparse format of a sparse result, where ``None`` (the default) leaves
      the choice to ``scipy.sparse.block_array``, which gives CSR when every block is CSR, CSC when every block
      is CSC, and COO otherwise (including whenever there's a ``None``); ignored for a dense result, but an
      unknown format raises an error either way.

    **``None`` blocks.** A ``None`` takes its number of rows from the other blocks in its row of the grid, and its
    number of columns from the other blocks in its column, as in ``scipy.sparse.block_array``. A row of the grid
    that is all ``None`` has no rows in the result, and a column that is all ``None`` has no columns (also as in
    scipy); a grid of only ``None`` gives an empty matrix of shape (0, 0).

    **Result.**

    - If any block is sparse, a scipy sparse array (``scipy.sparse.block_array``); otherwise a new NumPy array
      (``np.block``, with the ``None`` blocks filled with zeros, since ``np.block`` itself doesn't take ``None``).
    - Either way, its dtype is the blocks' common dtype (float64 when every block is ``None``, as in scipy), and
      the values are the same.

    **Errors.**

    - ``ValueError`` if the grid is empty or its rows have different numbers of blocks, if a block isn't 2-D, if
      blocks in a row of the grid have different numbers of rows or blocks in a column different numbers of
      columns, or if ``format`` isn't one of scipy's formats.
    """
    _check_format(format)
    grid = _as_grid(blocks)
    if _any_sparse(block for row in grid for block in row):
        return _call_untyped(scipy.sparse.block_array, grid, format=format)
    # no block is sparse here, so every block is a NumPy array or None
    return np.block(_fill_zero_blocks(cast(list[list[npt.NDArray[Any] | None]], grid)))


# --- Checks ----------------------------------------------------------------------------------------------------


def _check_format(format: str | None) -> None:
    """Raise ``ValueError`` unless ``format`` is ``None`` or one of scipy's sparse formats.

    Checked even when the result will be dense, so that a wrong format is found whatever the blocks are.
    """
    if format is not None and format not in _SPARSE_FORMATS:
        raise ValueError(f"Unknown sparse format {format!r}: expected one of {sorted(_SPARSE_FORMATS)} or None")


def _check_2d(block: PsMatrix, where: str) -> None:
    """Raise ``ValueError`` unless ``block`` is 2-D (``where`` names it in the message).

    NumPy would treat a 1-D block differently from scipy (``np.hstack`` joins 1-D arrays into a longer one), so
    other shapes are refused rather than giving results that depend on whether some other block is sparse.
    """
    if block.ndim != 2:
        raise ValueError(f"{where} must be 2-D, but has shape {block.shape}")


def _any_sparse(blocks: Any) -> bool:
    """Whether any of ``blocks`` (an iterable, possibly with ``None`` entries) is a scipy sparse array."""
    return any(issparse(block) for block in blocks)


# --- Stacking along one axis -----------------------------------------------------------------------------------


def _stack(
    blocks: Sequence[PsMatrix],
    format: SparseFormat | None,
    *,
    sparse_stack: Callable[..., Any],
    dense_stack: Callable[..., npt.NDArray[Any]],
) -> PsMatrix:
    """Check the blocks and format, then stack with ``sparse_stack`` if any block is sparse, else ``dense_stack``."""
    _check_format(format)
    blocks = list(blocks)
    if not blocks:
        raise ValueError("At least one block is needed")
    for index, block in enumerate(blocks):
        if block is None:
            raise TypeError(f"blocks[{index}] is None: only ps_block takes None, for a block of zeros")
        _check_2d(block, f"blocks[{index}]")
    if _any_sparse(blocks):
        return _call_untyped(sparse_stack, blocks, format=format)
    return dense_stack(blocks)


# --- Grids of blocks -------------------------------------------------------------------------------------------


def _as_grid(blocks: Sequence[Sequence[PsMatrix | None]]) -> list[list[PsMatrix | None]]:
    """Return the grid as a list of lists (``np.block`` refuses tuples), checking its shape and every block."""
    grid = [list(row) for row in blocks]
    if not grid or not grid[0]:
        raise ValueError("The grid of blocks must have at least one row of at least one block")
    num_columns = len(grid[0])
    for row_index, row in enumerate(grid):
        if len(row) != num_columns:
            raise ValueError(f"Row {row_index} of the grid has {len(row)} blocks, but row 0 has {num_columns}")
        for column_index, block in enumerate(row):
            if block is not None:
                _check_2d(block, f"blocks[{row_index}][{column_index}]")
    return grid


def _fill_zero_blocks(grid: list[list[npt.NDArray[Any] | None]]) -> list[list[npt.NDArray[Any]]]:
    """Replace each ``None`` in a grid of dense blocks with zeros of the size its row and column of the grid give,
    in the blocks' common dtype."""
    heights = [_common_size(row, axis=0, where=f"row {index} of the grid") for index, row in enumerate(grid)]
    columns = [list(column) for column in zip(*grid, strict=True)]
    widths = [_common_size(column, axis=1, where=f"column {index} of the grid") for index, column in enumerate(columns)]
    dtype = _common_dtype([block for row in grid for block in row if block is not None])
    return [
        [
            np.zeros((heights[row_index], widths[column_index]), dtype=dtype) if block is None else block
            for column_index, block in enumerate(row)
        ]
        for row_index, row in enumerate(grid)
    ]


def _common_size(blocks: list[npt.NDArray[Any] | None], *, axis: int, where: str) -> int:
    """The size along ``axis`` shared by the blocks that aren't ``None``, or 0 if all are ``None``.

    Raises ``ValueError`` if they differ (``where`` names the row or column of the grid in the message).
    """
    sizes = {block.shape[axis] for block in blocks if block is not None}
    if len(sizes) > 1:
        what = "numbers of rows" if axis == 0 else "numbers of columns"
        raise ValueError(f"The blocks in {where} have different {what}: {sorted(sizes)}")
    return sizes.pop() if sizes else 0


def _common_dtype(blocks: list[npt.NDArray[Any]]) -> np.dtype[Any]:
    """The dtype the blocks have in common (what ``np.block`` would give them), or float64 if there are none (as
    scipy gives a grid of only ``None``)."""
    if not blocks:
        return np.dtype(np.float64)
    # the dtypes rather than the arrays, so that only the types are promoted, never their values
    return np.result_type(*(block.dtype for block in blocks))


# --- Calling scipy ---------------------------------------------------------------------------------------------


def _call_untyped(function: Callable[..., Any], *args: Any, **kwargs: Any) -> SparseArray:
    """Call one of scipy's stacking functions, without its type stubs' overloads.

    scipy-stubs types ``hstack``, ``vstack`` and ``block_array`` with an overload per sparse format and per kind of
    block, which a format chosen at run time and a mix of dense and sparse blocks can't match. Every one of them
    returns a sparse array when given sparse-array blocks.
    """
    return function(*args, **kwargs)
