"""Comparing each entry of an array with a reference entry along an axis (private).

Public through ``mcrotk_utils.numpy_utils``.
"""

import operator
from collections.abc import Callable
from typing import Literal

import numpy as np
import numpy.typing as npt

# Which entry along the axis each entry is compared with: a neighbour ("previous" or "next"), an end ("first" or
# "last"), or the entry at a given index (a Python or NumPy integer; negative indices count from the end)
type ComparisonReference = Literal["previous", "next", "first", "last"] | int | np.integer

# How an entry is compared with its reference: by name, or a function of (values, references), called with two
# float arrays of the same shape and returning an array of that shape
type ComparisonMeasure = (
    Literal["difference", "relative", "ratio"]
    | Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.ArrayLike]
)

_NAMED_REFERENCES = ("previous", "next", "first", "last")
_NAMED_MEASURES = ("difference", "relative", "ratio")

# Marks an entry with no reference: the first entry for "previous", the last for "next"
_NO_REFERENCE = -1


def compare_along(
    values: npt.ArrayLike,
    reference: ComparisonReference = "previous",
    measure: ComparisonMeasure = "difference",
    *,
    axis: int = -1,
    skip_missing: bool = False,
) -> npt.NDArray[np.float64]:
    """Compare each entry with a reference entry along an axis, e.g. each week with the week before.

    Missing values are NaN. With ``skip_missing=False`` (the default) they keep their positions, so an entry whose
    reference is missing gets NaN: with weeks 1, 2 and 3 and week 2 missing, week 3 isn't compared with week 1.

    **Arguments.**

    - ``values``: an array, or anything ``np.asarray`` accepts, of real numbers, with at least one dimension.
    - ``reference``: which entry along ``axis`` each entry is compared with:

      - ``"previous"`` (the default): the entry before it. The first entry has none.
      - ``"next"``: the entry after it. The last entry has none.
      - ``"first"`` or ``"last"``: the first or last entry.
      - an integer (Python's or NumPy's): the entry at that index (negative indices count from the end, as usual).

    - ``measure``: how an entry ``v`` is compared with its reference ``r``:

      - ``"difference"`` (the default): ``v - r``.
      - ``"relative"``: ``(v - r) / r``, the relative change (0.05 for 5% more).
      - ``"ratio"``: ``v / r``.
      - a function, called once as ``measure(values, references)`` with two float arrays of the input's shape
        (moved so that ``axis`` is last), returning an array of that shape.

      With ``"relative"`` and ``"ratio"``, a reference of 0 gives ``inf``, ``-inf`` or NaN, as NumPy's division
      does (without its warnings).
    - ``axis``: the axis to compare along; negative values count from the end, as usual in NumPy.
    - ``skip_missing``: if true, missing entries are passed over when finding a reference: ``"previous"`` and
      ``"next"`` take the nearest entry that isn't missing, and ``"first"`` and ``"last"`` the first or last
      that isn't (separately along each line of ``axis``). No effect on an integer ``reference``.

    **Result.** A new float64 array of the input's shape. It's NaN where the entry or its reference is missing, where
    an entry has no reference, and at the reference entries themselves (with ``"first"``, ``"last"`` or an integer
    ``reference``): an entry isn't compared with itself.

    **Errors.**

    - ``ValueError``: an unknown ``reference`` or ``measure`` name, or a ``measure`` function returning an array of
      the wrong shape.
    - ``TypeError``: a ``reference`` that's neither a name nor an integer (including ``True`` and ``False``), or
      values that can't be made real floats (e.g. complex numbers).
    - ``IndexError``: an integer ``reference`` outside the axis (any integer, when the axis has length 0).
    - ``np.exceptions.AxisError`` (a ``ValueError``): no such axis, e.g. for a 0-d array.

    **Example.** ``compare_along(weekly_totals, "first", "relative")`` gives each week's change since the first
    week, as a fraction; with one row per region, ``axis=-1`` compares weeks within each region.
    """
    _check_measure(measure)
    array = _as_float_array(values)
    moved = np.moveaxis(array, axis, -1)  # checks the axis, raising AxisError
    indices = _reference_indices(moved, reference, skip_missing=skip_missing)
    result = _measured(moved, indices, measure)
    return np.moveaxis(result, -1, axis)


def _reference_indices(
    values: npt.NDArray[np.float64], reference: ComparisonReference, *, skip_missing: bool = False
) -> npt.NDArray[np.intp]:
    """Return, for each entry along the last axis, the index of its reference entry, or -1 if it has none.

    Private, but shared with ``mcrotk_utils.plotting_utils``, which also needs to know which entries are the
    references themselves (those whose index is their own position). ``values`` must already be a float array with
    the axis to compare along last.
    """
    length = values.shape[-1]
    if reference in _NAMED_REFERENCES:
        return _named_reference_indices(values, str(reference), skip_missing)
    index = _checked_index(reference, length)
    return np.full(values.shape, index, dtype=np.intp)


def _check_measure(measure: object) -> None:
    """Raise ``ValueError`` unless ``measure`` is a known name or a callable."""
    if not callable(measure) and measure not in _NAMED_MEASURES:
        names = ", ".join(map(repr, _NAMED_MEASURES))
        raise ValueError(f"measure must be one of {names} or a function, not {measure!r}.")


def _as_float_array(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return ``values`` as a float64 array; ``TypeError`` for complex values (rather than dropping imaginary parts)."""
    array = np.asarray(values)
    if np.iscomplexobj(array):
        raise TypeError("compare_along needs real numbers, not complex ones.")
    return array.astype(np.float64)


def _named_reference_indices(
    values: npt.NDArray[np.float64], reference: str, skip_missing: bool
) -> npt.NDArray[np.intp]:
    """Return the reference indices for "previous", "next", "first" or "last" along the last axis."""
    if reference in ("next", "last"):
        # The mirror image of "previous" and "first": compute those on the reversed axis, then map indices back
        mirrored = "previous" if reference == "next" else "first"
        flipped = _named_reference_indices(values[..., ::-1], mirrored, skip_missing)[..., ::-1]
        return np.where(flipped == _NO_REFERENCE, _NO_REFERENCE, values.shape[-1] - 1 - flipped)
    if reference == "previous":
        return _previous_indices(values, skip_missing)
    return _first_indices(values, skip_missing)


def _previous_indices(values: npt.NDArray[np.float64], skip_missing: bool) -> npt.NDArray[np.intp]:
    """Return the index of the entry before each one (the nearest one that isn't missing, if ``skip_missing``)."""
    positions = np.broadcast_to(np.arange(values.shape[-1], dtype=np.intp), values.shape)
    if skip_missing:
        # The latest position up to each entry whose value isn't missing, then shifted by one so that each entry
        # gets the latest one strictly before it
        present_positions = np.where(np.isnan(values), _NO_REFERENCE, positions)
        candidates = np.maximum.accumulate(present_positions, axis=-1) if values.shape[-1] else present_positions
    else:
        candidates = positions
    shifted = np.full(values.shape, _NO_REFERENCE, dtype=np.intp)
    shifted[..., 1:] = candidates[..., :-1]
    return shifted


def _first_indices(values: npt.NDArray[np.float64], skip_missing: bool) -> npt.NDArray[np.intp]:
    """Return the index of the first entry (the first that isn't missing, if ``skip_missing``) for every entry."""
    if values.shape[-1] == 0:
        return np.zeros(values.shape, dtype=np.intp)
    if skip_missing:
        # argmax finds the first True; a line that's all missing gets 0, whose value is missing anyway
        first = np.argmax(~np.isnan(values), axis=-1, keepdims=True)
    else:
        first = np.zeros((*values.shape[:-1], 1), dtype=np.intp)
    return np.broadcast_to(first, values.shape).astype(np.intp)


def _checked_index(reference: ComparisonReference, length: int) -> int:
    """Return an integer ``reference`` as a non-negative index into an axis of ``length``, checking its range."""
    if isinstance(reference, (bool, np.bool_)):
        # bool is an int subclass, but True as "the entry at index 1" would be a surprising reading
        raise TypeError(f"reference must be a name or an integer index, not {reference!r}.")
    if isinstance(reference, str):
        names = ", ".join(map(repr, _NAMED_REFERENCES))
        raise ValueError(f"reference must be one of {names} or an integer, not {reference!r}.")
    try:
        index = operator.index(reference)  # accepts NumPy integers; rejects floats etc.
    except TypeError:
        raise TypeError(f"reference must be a name or an integer index, not {reference!r}.") from None
    if not -length <= index < length:
        raise IndexError(f"reference index {index} is outside an axis of length {length}.")
    return index % length


def _measured(
    values: npt.NDArray[np.float64], indices: npt.NDArray[np.intp], measure: ComparisonMeasure
) -> npt.NDArray[np.float64]:
    """Apply ``measure`` to each entry and its reference, with NaN where there's no comparison to make."""
    positions = np.arange(values.shape[-1], dtype=np.intp)
    no_comparison = (indices == _NO_REFERENCE) | (indices == positions)  # no reference, or the reference itself
    references = np.take_along_axis(values, np.where(no_comparison, 0, indices), axis=-1) if values.size else values
    result = _apply_measure(values, references, measure)
    missing = no_comparison | np.isnan(values) | np.isnan(references)
    return np.where(missing, np.nan, result)


def _apply_measure(
    values: npt.NDArray[np.float64], references: npt.NDArray[np.float64], measure: ComparisonMeasure
) -> npt.NDArray[np.float64]:
    """Compute ``measure`` for every entry (including those that will be set to NaN afterwards)."""
    # Division by a zero reference gives inf or NaN, as documented; NumPy's warnings about it would only be noise
    with np.errstate(divide="ignore", invalid="ignore"):
        if callable(measure):
            result = measure(values, references)
        elif measure == "difference":
            return values - references
        elif measure == "relative":
            return (values - references) / references
        else:  # "ratio", by _check_measure
            return values / references
    as_array = np.asarray(result, dtype=np.float64)
    if as_array.shape != values.shape:
        raise ValueError(f"The measure function returned shape {as_array.shape}, not the values' shape {values.shape}.")
    return as_array
