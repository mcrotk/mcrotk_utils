"""Interleaving arrays along an axis, in blocks (private).

Public through ``mcrotk_utils.numpy_utils``.
"""

import operator
from collections.abc import Sequence
from typing import Any, SupportsIndex

import numpy as np
import numpy.typing as npt


def interleave(*arrays: npt.ArrayLike, blocksize: SupportsIndex = 1, axis: int = 0) -> npt.NDArray[Any]:
    """Interleave arrays along an axis, taking ``blocksize`` entries at a time from each array in turn.

    For arrays ``a`` and ``b`` along axis 0, the rows come out as ``a0, b0, a1, b1, ...`` with ``blocksize=1``,
    and as ``a0, a1, b0, b1, a2, a3, b2, b3, ...`` with ``blocksize=2``.

    The arrays may have different lengths along ``axis``: an array that runs out is skipped from then on, and
    its last block may be shorter than ``blocksize``. All other dimensions must match.

    **Arguments.**

    - ``arrays``: the arrays to interleave, any number of them (at least one). Each can be a NumPy array
      or anything ``np.asarray`` accepts, such as a nested list. All must have the same number of dimensions.
    - ``blocksize``: how many consecutive entries along ``axis`` to take from one array before moving on
      to the next. With ``blocksize=1`` the arrays alternate entry by entry; with ``blocksize=n``, ``n``
      entries at a time. A positive integer; NumPy integers work too.
    - ``axis``: the axis to interleave along; negative values count from the end, as usual in NumPy.

    **Result.**

    - A new array: never a view of an input, even when only one array is given.
    - C-ordered, whatever the axis.
    - Its dtype is the arrays' common dtype (``np.result_type``).
    - Its shape is the inputs' shape, with their lengths along ``axis`` added up.

    **Errors.**

    - ``ValueError``: no arrays given, ``blocksize`` below 1, or shapes that differ apart from ``axis``
      (including different numbers of dimensions).
    - ``np.exceptions.AxisError`` (a ``ValueError``): the arrays have no such axis, e.g. 0-d arrays.
    - ``TypeError``: ``blocksize`` isn't an integer (including floats such as ``2.0``).

    **Example.** ``interleave(x, 1 - x, blocksize=n)`` follows each block of ``n`` rows of 0/1 assignments
    with the opposite assignments.
    """
    checked_blocksize = _checked_blocksize(blocksize)
    if not arrays:
        raise ValueError("interleave needs at least one array.")
    as_arrays = [np.asarray(array) for array in arrays]
    lengths = [_length_along(array, axis) for array in as_arrays]
    _check_other_dimensions_match(as_arrays, axis)
    destinations = _destinations(lengths, checked_blocksize)
    return _write_interleaved(as_arrays, destinations, axis)


def _checked_blocksize(blocksize: SupportsIndex) -> int:
    """Return ``blocksize`` as an int, raising if it isn't a positive integer."""
    index = operator.index(blocksize)  # TypeError for floats etc.; accepts NumPy integers
    if index < 1:
        raise ValueError(f"blocksize must be at least 1, not {index}.")
    return index


def _length_along(array: npt.NDArray[Any], axis: int) -> int:
    """Return the array's length along ``axis``, raising ``np.exceptions.AxisError`` if it has no such axis."""
    # moveaxis checks the axis (including for 0-d arrays) and raises AxisError, a ValueError, rather than the
    # IndexError that indexing the shape would give
    return np.moveaxis(array, axis, 0).shape[0]


def _check_other_dimensions_match(arrays: Sequence[npt.NDArray[Any]], axis: int) -> None:
    """Raise ``ValueError`` unless the arrays have the same shape apart from ``axis``."""
    other_shapes = {np.moveaxis(array, axis, 0).shape[1:] for array in arrays}
    if len(other_shapes) > 1:
        shapes = ", ".join(str(array.shape) for array in arrays)
        raise ValueError(f"The arrays' shapes must match apart from axis {axis}; got {shapes}.")


def _destinations(lengths: Sequence[int], blocksize: int) -> list[npt.NDArray[np.intp]]:
    """Return, for each array, the positions along the axis that its entries go to in the result."""
    order = _interleaving_order(lengths, blocksize)
    # order lists the input entries by result position; inverting it gives each input entry's result position
    positions = np.empty_like(order)
    positions[order] = np.arange(order.size)
    return np.split(positions, np.cumsum(lengths)[:-1])


def _write_interleaved(
    arrays: Sequence[npt.NDArray[Any]], destinations: Sequence[npt.NDArray[np.intp]], axis: int
) -> npt.NDArray[Any]:
    """Write each array into its destinations along ``axis`` of a new array, and return that array.

    Writing straight into a preallocated result copies the data once, where concatenating and then reordering
    would copy it twice.
    """
    shape = list(arrays[0].shape)
    shape[axis] = sum(destination.size for destination in destinations)
    result = np.empty(shape, dtype=np.result_type(*arrays))
    # a view with the axis first, so each array's entries can be written by indexing the first dimension
    result_axis_first = np.moveaxis(result, axis, 0)
    for array, destination in zip(arrays, destinations, strict=True):
        result_axis_first[destination] = np.moveaxis(array, axis, 0)
    return result


def _interleaving_order(lengths: Sequence[int], blocksize: int) -> npt.NDArray[np.intp]:
    """Return the positions in the arrays' concatenation, in interleaved order.

    ``lengths`` are the arrays' lengths along the axis. Each entry is keyed by its block number, then its
    array's number, so sorting by that key takes the blocks of all arrays in turn. The sort is stable, which
    keeps the entries within each block in their original order.
    """
    num_arrays = len(lengths)
    keys = np.concatenate(
        [np.arange(length) // blocksize * num_arrays + array_num for array_num, length in enumerate(lengths)]
    )
    return np.argsort(keys, kind="stable")
