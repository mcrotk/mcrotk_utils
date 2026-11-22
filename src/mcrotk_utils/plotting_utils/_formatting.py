"""Turning the numbers shown on bars into label texts (private)."""

import math
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

# A format for the numbers: a format string with a replacement field ("{:.1f} kg"), a format spec alone ("+.1%"),
# or a function from the number to its text
type _NumberFormat = str | Callable[[float], str]

# Significant digits that the default formats give the largest number shown
_SIGNIFICANT_DIGITS = 3


def _formatter(fmt: _NumberFormat) -> Callable[[float], str]:
    """Return a function formatting a number with ``fmt``."""
    if callable(fmt):
        return fmt
    if not isinstance(fmt, str):
        raise TypeError(f"fmt must be a string or a function, not {type(fmt).__name__}.")
    if "{" in fmt:
        return fmt.format
    return lambda number: format(number, fmt)


def _default_format(measure: object, numbers: npt.NDArray[np.float64]) -> Callable[[float], str]:
    """Return the default format for ``measure``'s results, given all the numbers to be shown.

    Percentages for relative changes and shares; for values, differences, ratios and a measure function's results,
    fixed-point numbers with thousands separators and the same number of decimals throughout, enough to give the
    largest number three significant digits. Differences and relative changes show their sign.
    """
    if measure == "relative":
        return lambda number: format(number, "+.1%")
    if measure == "share":
        return lambda number: format(number, ".1%")
    decimals = _decimals_for(numbers)
    sign = "+" if measure == "difference" else ""
    suffix = "×" if measure == "ratio" else ""
    return lambda number: f"{number:{sign},.{decimals}f}{suffix}"


def _decimals_for(numbers: npt.NDArray[np.float64]) -> int:
    """Return how many decimals give the largest finite number (in absolute value) three significant digits."""
    finite = np.abs(numbers[np.isfinite(numbers)])
    largest = float(finite.max()) if finite.size else 0.0
    if largest == 0:
        return 0
    return max(0, _SIGNIFICANT_DIGITS - 1 - math.floor(math.log10(largest)))


def _label_text(
    number: float, is_reference: bool, format_number: Callable[[float], str], missing: str, reference: str
) -> str:
    """Return one bar's label: ``reference`` for a reference bar, ``missing`` for no number, else the number."""
    if is_reference:
        return reference
    if np.isnan(number):
        return missing
    return format_number(number)
