"""Tests for ``mcrotk_utils.numpy_utils.compare_along``.

The expected results come from known cases and from ``reference_compare_along``, a straightforward implementation
that loops over the lines along the axis and finds each entry's reference by scanning.
"""

import math
import warnings

import numpy as np
import pytest

from mcrotk_utils.numpy_utils import ComparisonMeasure, ComparisonReference, compare_along

NAN = np.nan
NAMED_REFERENCES: list[ComparisonReference] = ["previous", "next", "first", "last"]
NAMED_MEASURES: list[ComparisonMeasure] = ["difference", "relative", "ratio"]


def reference_index(line, position, reference, skip_missing):
    """Return the index of the entry that ``line[position]`` is compared with, or None if there's none."""
    present = [not math.isnan(value) for value in line]
    length = len(line)
    if reference in ("previous", "next"):
        step = -1 if reference == "previous" else 1
        index = position + step
        while skip_missing and 0 <= index < length and not present[index]:
            index += step
        return index if 0 <= index < length else None
    if reference in ("first", "last"):
        order = range(length) if reference == "first" else range(length - 1, -1, -1)
        candidates = [i for i in order if present[i]] if skip_missing else list(order)
        return candidates[0] if candidates else None
    return reference % length


def reference_measure(value, base, measure):
    """Compare one value with its reference, as the named measures do."""
    if measure == "difference":
        return value - base
    if measure == "ratio":
        return (
            value / base if base != 0 else (math.copysign(math.inf, value) * math.copysign(1, base) if value else NAN)
        )
    # relative
    change = value - base
    return change / base if base != 0 else (math.copysign(math.inf, change) * math.copysign(1, base) if change else NAN)


def reference_compare_line(line, reference, measure, skip_missing):
    """Compare every entry of one line with its reference, entry by entry."""
    result = []
    for position, value in enumerate(line):
        index = reference_index(line, position, reference, skip_missing)
        if index is None or index == position or math.isnan(value) or math.isnan(line[index]):
            result.append(NAN)
        else:
            result.append(reference_measure(value, line[index], measure))
    return result


def reference_compare_along(values, reference, measure, axis, skip_missing):
    """Apply ``reference_compare_line`` to every line of ``values`` along ``axis``."""
    moved = np.moveaxis(np.asarray(values, dtype=float), axis, -1)
    result = np.empty(moved.shape)
    for index in np.ndindex(moved.shape[:-1]):
        result[index] = reference_compare_line(list(moved[index]), reference, measure, skip_missing)
    return np.moveaxis(result, -1, axis)


# --- Known cases ---------------------------------------------------------------------------------------------------

LINE = [1.0, 2.0, NAN, 4.0]


@pytest.mark.parametrize(
    ("reference", "skip_missing", "expected"),
    [
        ("previous", False, [NAN, 1, NAN, NAN]),  # week 3 is missing, so week 4 has no reference
        ("previous", True, [NAN, 1, NAN, 2]),  # week 4 is compared with week 2
        ("next", False, [-1, NAN, NAN, NAN]),
        ("next", True, [-1, -2, NAN, NAN]),
        ("first", False, [NAN, 1, NAN, 3]),
        ("last", False, [-3, -2, NAN, NAN]),
        (1, False, [-1, NAN, NAN, 2]),
        (-1, False, [-3, -2, NAN, NAN]),
    ],
)
def test_known_differences_along_one_line(reference, skip_missing, expected):
    """Each entry minus its reference, NaN with no reference, at the reference itself, or for missing values."""
    result = compare_along(LINE, reference, skip_missing=skip_missing)
    np.testing.assert_array_equal(result, expected)


def test_skip_missing_finds_the_first_and_last_present_entries_of_each_line():
    """With skip_missing, "first" and "last" are the first and last entries that aren't missing, line by line."""
    values = [[NAN, 2.0, 3.0, NAN], [1.0, NAN, 5.0, 6.0]]
    np.testing.assert_array_equal(
        compare_along(values, "first", skip_missing=True), [[NAN, NAN, 1, NAN], [NAN, NAN, 4, 5]]
    )
    np.testing.assert_array_equal(
        compare_along(values, "last", skip_missing=True), [[NAN, -1, NAN, NAN], [-5, NAN, -1, NAN]]
    )
    # Without it, a missing first or last entry leaves no comparisons in its line
    np.testing.assert_array_equal(compare_along(values, "first")[0], [NAN] * 4)


def test_skip_missing_has_no_effect_on_an_integer_reference():
    """An integer reference is that index, whether or not its entry is missing."""
    values = [[NAN, 2.0, 3.0], [1.0, 2.0, 4.0]]
    for skip in (False, True):
        np.testing.assert_array_equal(compare_along(values, 0, skip_missing=skip), [[NAN, NAN, NAN], [NAN, 1, 3]])


def test_measures_on_known_values():
    """difference is v - r, relative (v - r) / r, ratio v / r."""
    values = [2.0, 3.0, 6.0]
    np.testing.assert_allclose(compare_along(values, "first", "difference"), [NAN, 1, 4])
    np.testing.assert_allclose(compare_along(values, "first", "relative"), [NAN, 0.5, 2])
    np.testing.assert_allclose(compare_along(values, "first", "ratio"), [NAN, 1.5, 3])


def test_zero_references_give_infinities_or_nan_without_warnings():
    """Dividing by a zero reference gives inf, -inf or NaN, as NumPy's division does, and no warning."""
    values = [0.0, 2.0, -3.0, 0.0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        relative = compare_along(values, "first", "relative")
        ratio = compare_along(values, "first", "ratio")
    np.testing.assert_array_equal(relative, [NAN, np.inf, -np.inf, NAN])
    np.testing.assert_array_equal(ratio, [NAN, np.inf, -np.inf, NAN])


def test_axis_chooses_the_direction_of_comparison():
    """axis=0 compares down the columns, axis=-1 (the default) along the rows."""
    values = np.array([[1.0, 2.0], [4.0, 8.0]])
    np.testing.assert_array_equal(compare_along(values), [[NAN, 1], [NAN, 4]])
    np.testing.assert_array_equal(compare_along(values, axis=0), [[NAN, NAN], [3, 6]])


def test_docstring_example_changes_since_the_first_week_within_each_region():
    """compare_along(weekly_totals, "first", "relative") gives each week's change since week 1, region by region."""
    weekly_totals = np.array([[100.0, 110.0, 121.0], [50.0, 45.0, 60.0]])  # one row per region
    np.testing.assert_allclose(compare_along(weekly_totals, "first", "relative"), [[NAN, 0.1, 0.21], [NAN, -0.1, 0.2]])


def test_a_measure_function_gets_float_arrays_with_the_axis_last():
    """A measure function is called once, with values and references of the input's shape, axis moved last."""
    calls = []

    def log_ratio(values, references):
        """Record the call, and return the log of the ratio."""
        calls.append((values.shape, references.shape, values.dtype, references.dtype))
        return np.log(values / references)

    values = np.array([[1, 4], [2, 8], [4, 16]])  # integers: passed as floats
    result = compare_along(values, "first", log_ratio, axis=0)
    assert calls == [((2, 3), (2, 3), np.float64, np.float64)]
    np.testing.assert_allclose(result, np.array([[NAN, NAN], np.log([2, 2]), np.log([4, 4])]))


def test_a_measure_function_returning_the_wrong_shape_is_refused():
    """The result of a measure function must have the values' shape."""
    with pytest.raises(ValueError, match="shape"):
        compare_along([1.0, 2.0, 3.0], measure=lambda values, references: values.sum())


def test_result_is_a_new_float64_array_and_the_input_is_unchanged():
    """The result is float64 (also for integer input), a new array, and the input isn't modified."""
    values = np.array([1, 2, 4])
    result = compare_along(values)
    assert result.dtype == np.float64
    assert not np.shares_memory(result, values)
    np.testing.assert_array_equal(values, [1, 2, 4])
    np.testing.assert_array_equal(result, [NAN, 1, 2])


def test_an_empty_axis_gives_an_empty_result_for_named_references():
    """With nothing along the axis, every named reference gives an empty result."""
    for reference in NAMED_REFERENCES:
        for skip in (False, True):
            assert compare_along(np.empty((2, 0)), reference, skip_missing=skip).shape == (2, 0)


def test_numpy_integer_references_are_accepted():
    """An integer reference may be a NumPy integer."""
    np.testing.assert_array_equal(compare_along([1.0, 3.0], np.int64(0)), [NAN, 2])


# --- Errors --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("reference", ["prev", "middle", ""])
def test_unknown_reference_names_are_refused(reference):
    """Only the four names are known."""
    with pytest.raises(ValueError, match="reference must be one of"):
        compare_along([1.0, 2.0], reference)


@pytest.mark.parametrize("reference", [True, False, np.True_, 1.0, None])
def test_references_that_are_neither_names_nor_integers_are_refused(reference):
    """Booleans (although bool is an int), floats and other values are refused."""
    with pytest.raises(TypeError, match="reference must be a name or an integer"):
        compare_along([1.0, 2.0], reference)


def test_unknown_measure_names_are_refused():
    """Only the three names, or a function, are accepted."""
    with pytest.raises(ValueError, match="measure must be one of"):
        compare_along([1.0, 2.0], measure="percent")  # type: ignore[arg-type]


@pytest.mark.parametrize(("reference", "length"), [(2, 2), (-3, 2), (0, 0)])
def test_integer_references_outside_the_axis_are_refused(reference, length):
    """An index outside the axis raises IndexError, as any index does when the axis is empty."""
    with pytest.raises(IndexError, match="outside an axis"):
        compare_along(np.ones(length), reference)


def test_missing_axes_are_refused():
    """A 0-d array has no axis to compare along, and an axis beyond the dimensions doesn't exist."""
    with pytest.raises(np.exceptions.AxisError):
        compare_along(np.float64(3.0))
    with pytest.raises(ValueError):  # AxisError is a ValueError
        compare_along(np.ones((2, 3)), axis=2)


def test_complex_values_are_refused():
    """Complex numbers raise TypeError rather than losing their imaginary parts."""
    with pytest.raises(TypeError, match="real numbers"):
        compare_along([1 + 1j, 2 + 0j])


# --- Random comparisons with the reference implementation ----------------------------------------------------------


def random_values(rng):
    """Return a random array of 1 to 3 dimensions, with some missing values and some zeros."""
    shape = tuple(int(n) for n in rng.integers(0, 5, size=rng.integers(1, 4)))
    values = np.round(rng.normal(0, 3, size=shape), 1)
    values[rng.uniform(size=shape) < 0.2] = NAN
    values[rng.uniform(size=shape) < 0.05] = 0.0
    return values


def test_random_arrays_match_the_reference_implementation():
    """Every reference, measure, axis and skip_missing setting matches the loop implementation."""
    rng = np.random.default_rng(12)
    checked = 0
    for _ in range(300):
        values = random_values(rng)
        axis = int(rng.integers(-values.ndim, values.ndim))
        length = values.shape[axis]
        references: list[ComparisonReference] = [*NAMED_REFERENCES]
        if length:
            references += [int(rng.integers(-length, length))]
        for reference in references:
            for measure in NAMED_MEASURES:
                for skip in (False, True):
                    expected = reference_compare_along(values, reference, measure, axis, skip)
                    result = compare_along(values, reference, measure, axis=axis, skip_missing=skip)
                    np.testing.assert_allclose(result, expected, rtol=1e-12, equal_nan=True)
                    checked += 1
    assert checked > 5000
