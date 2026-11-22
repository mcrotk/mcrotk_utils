"""Tests for ``mcrotk_utils.misc_utils``' checks: the range checks and ``check_not_none``."""

import math
import re
import subprocess
import sys
from fractions import Fraction

import numpy as np
import pytest

from mcrotk_utils.misc_utils import check_in_range, check_nonnegative, check_not_none, check_positive

NAN = float("nan")


# --- check_in_range ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "bounds"),
    [
        (5, {}),
        (0, {"min_value": 0}),
        (1, {"max_value": 1}),
        (0.5, {"min_value": 0, "max_value": 1, "include_min": False, "include_max": False}),
        (-math.inf, {}),
        (math.inf, {"min_value": 0}),
    ],
)
def test_check_in_range_returns_values_in_range(value, bounds):
    """A value in the range comes back as it is (bounds included by default; infinities are numbers too)."""
    assert check_in_range(value, "x", **bounds) is value


def test_check_in_range_keeps_the_type():
    """The value itself is returned, so an int stays an int."""
    assert type(check_in_range(3, "n", min_value=1)) is int


@pytest.mark.parametrize(
    ("value", "bounds", "message"),
    [
        (-1, {"min_value": 0}, "x must be at least 0, not -1."),
        (0, {"min_value": 0, "include_min": False}, "x must be greater than 0, not 0."),
        (2, {"max_value": 1}, "x must be at most 1, not 2."),
        (1, {"max_value": 1, "include_max": False}, "x must be less than 1, not 1."),
        (2, {"min_value": 0, "max_value": 1}, "x must be between 0 and 1, not 2."),
        (0, {"min_value": 0, "max_value": 1, "include_min": False}, "x must be between 0 and 1 (excluding 0), not 0."),
        (
            1,
            {"min_value": 0, "max_value": 1, "include_min": False, "include_max": False},
            "x must be between 0 and 1 (excluding 0 and 1), not 1.",
        ),
    ],
)
def test_check_in_range_rejects_values_outside(value, bounds, message):
    """Outside the range (or on an excluded bound): ValueError naming the value and the range."""
    with pytest.raises(ValueError, match=re.escape(message)):
        check_in_range(value, "x", **bounds)


@pytest.mark.parametrize("bounds", [{}, {"min_value": 0}, {"max_value": 1}, {"min_value": 0, "max_value": 1}])
def test_check_in_range_rejects_nan_even_without_bounds(bounds):
    """NaN is never in range, even when there are no bounds."""
    with pytest.raises(ValueError, match="x must be"):
        check_in_range(NAN, "x", **bounds)


def test_check_in_range_accepts_other_comparable_numbers():
    """Fractions and NumPy scalars work, by comparison with the bounds."""
    assert check_in_range(Fraction(1, 3), "x", min_value=0, max_value=1) == Fraction(1, 3)
    assert check_in_range(np.int64(3), "n", min_value=1) == 3
    with pytest.raises(ValueError):
        check_in_range(np.float64("nan"), "x")


@pytest.mark.parametrize(("low", "high"), [(1, 0), (NAN, 1), (0, NAN)])
def test_check_in_range_rejects_bad_bounds(low, high):
    """NaN bounds, or a minimum above the maximum, raise ValueError (nothing could pass)."""
    with pytest.raises(ValueError):
        check_in_range(0.5, "x", min_value=low, max_value=high)


def test_check_in_range_raises_type_error_for_values_not_comparable():
    """A value that can't be compared with the bounds (e.g. a string) raises TypeError."""
    with pytest.raises(TypeError):
        check_in_range("a", "x", min_value=0)  # type: ignore[type-var]  # pyright: ignore[reportArgumentType]


# --- check_positive, check_nonnegative --------------------------------------------------------------------------------


def test_check_positive():
    """Positive values come back; 0, negatives and NaN raise ValueError with a short message."""
    assert check_positive(2.5, "sd") == 2.5
    for value in (0, -1, NAN):
        with pytest.raises(ValueError, match=re.escape(f"sd must be positive, not {value}.")):
            check_positive(value, "sd")


def test_check_nonnegative():
    """0 and positive values come back; negatives and NaN raise ValueError with a short message."""
    assert check_nonnegative(0, "n") == 0
    assert check_nonnegative(3, "n") == 3
    for value in (-1, NAN):
        with pytest.raises(ValueError, match=re.escape(f"n must be non-negative, not {value}.")):
            check_nonnegative(value, "n")


def test_shortcuts_agree_with_check_in_range():
    """check_positive and check_nonnegative accept exactly what the equivalent check_in_range calls accept."""
    for value in (-1, -0.0, 0, 1e-300, 1, math.inf, -math.inf, NAN):
        assert _passes(check_positive, value) == _passes(
            lambda v, name: check_in_range(v, name, min_value=0, include_min=False), value
        )
        assert _passes(check_nonnegative, value) == _passes(lambda v, name: check_in_range(v, name, min_value=0), value)


def _passes(check, value):
    """Return whether ``check(value, "x")`` passes."""
    try:
        check(value, "x")
    except ValueError:
        return False
    return True


@pytest.mark.parametrize(
    "call", ["check_positive(0, 'x')", "check_nonnegative(-1, 'x')", "check_in_range(2, 'x', max_value=1)"]
)
def test_checks_still_run_under_optimization(call):
    """Unlike asserts, the checks still raise under python -O."""
    code = f"from mcrotk_utils.misc_utils import *\ntry:\n    {call}\nexcept ValueError:\n    print('raised')"
    result = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True, text=True)
    assert result.stdout.strip() == "raised", result.stderr


# --- check_not_none ---------------------------------------------------------------------------------------------------


def test_check_not_none_returns_the_value():
    """A value other than None comes back as it is, including falsy values."""
    values: tuple[object, ...] = (0, "", [], False)
    for value in values:
        assert check_not_none(value, "x") is value


def test_check_not_none_raises_type_error_by_default():
    """None raises TypeError, naming the value."""
    with pytest.raises(TypeError, match="the source file must not be None"):
        check_not_none(None, "the source file")


def test_check_not_none_raises_the_chosen_error():
    """With error=, None raises that exception class instead."""
    with pytest.raises(LookupError, match="the source file must not be None"):
        check_not_none(None, "the source file", error=LookupError)
