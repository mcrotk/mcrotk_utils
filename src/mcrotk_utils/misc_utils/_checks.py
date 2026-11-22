"""Checks returning the checked value: numbers in a range (NaN never is), and values that mustn't be None (private).

Public through ``mcrotk_utils.misc_utils``. Uses only the standard library.
"""

from typing import Protocol


class _ComparableWithFloat(Protocol):
    """Anything that can be compared with a float, such as a ``Fraction`` or a NumPy scalar.

    For the range checks' types: ``float`` alone accepts ints but not these, though they work. (A comparison may
    return something other than a ``bool``, e.g. NumPy's ``np.bool``, so the results are ``object`` here.)
    """

    def __lt__(self, other: float, /) -> object: ...
    def __le__(self, other: float, /) -> object: ...
    def __gt__(self, other: float, /) -> object: ...
    def __ge__(self, other: float, /) -> object: ...


def check_in_range[N: float | _ComparableWithFloat](
    value: N,
    name: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    include_min: bool = True,
    include_max: bool = True,
) -> N:
    """Return ``value``, raising ``ValueError`` unless it lies in the given range. NaN never does.

    Unlike an ``assert``, the check also runs under ``python -O``. Returning the value allows checking inline,
    e.g. ``self.sd = check_in_range(sd, "sd", min_value=0)``.

    **Arguments.**

    - ``value``: the number to check: an int or float, or anything comparable with the bounds (e.g. a
      ``Fraction``, or a NumPy scalar).
    - ``name``: what the value is, for the error message.
    - ``min_value``, ``max_value``: the bounds; ``None`` (the default) for no bound on that side.
    - ``include_min``, ``include_max``: whether each bound itself is allowed (by default, both are).

    **Result.**

    - ``value`` itself.

    **Errors.**

    - ``ValueError``: ``value`` is outside the range or is NaN (even with no bounds); or the bounds are NaN, or
      ``min_value`` exceeds ``max_value``.
    - ``TypeError``: ``value`` can't be compared with the bounds (e.g. a string).
    """
    _check_bounds(min_value, max_value)
    above_min = min_value is None or bool(value >= min_value if include_min else value > min_value)
    below_max = max_value is None or bool(value <= max_value if include_max else value < max_value)
    _require(
        above_min and below_max and not _is_nan(value),
        value,
        name,
        _range_text(min_value, max_value, include_min, include_max),
    )
    return value


def check_positive[N: float | _ComparableWithFloat](value: N, name: str) -> N:
    """Return ``value``, raising ``ValueError`` unless it's greater than 0 (NaN isn't).

    The same as ``check_in_range(value, name, min_value=0, include_min=False)``, with a shorter message.
    """
    # written so that NaN fails too, since every comparison with NaN is false
    _require(bool(value > 0), value, name, "positive")
    return value


def check_nonnegative[N: float | _ComparableWithFloat](value: N, name: str) -> N:
    """Return ``value``, raising ``ValueError`` unless it's at least 0 (NaN isn't).

    The same as ``check_in_range(value, name, min_value=0)``, with a shorter message.
    """
    # written so that NaN fails too, since every comparison with NaN is false
    _require(bool(value >= 0), value, name, "non-negative")
    return value


def check_not_none[T](value: T | None, name: str, *, error: type[Exception] = TypeError) -> T:
    """Return ``value``, raising ``error`` (``TypeError`` by default) if it's ``None``.

    For values typed ``X | None`` that must be an ``X`` at this point: type checkers treat the result as an ``X``,
    and the error says what was missing, where an ``assert`` would vanish under ``python -O``.
    E.g. ``Path(check_not_none(inspect.getsourcefile(module), "the source file", error=LookupError)).parent``.

    **Arguments.**

    - ``value``: the value to check.
    - ``name``: what the value is, for the error message.
    - ``error``: the exception class to raise. ``TypeError`` (the default) fits a missing argument, as Python
      raises it when ``None`` reaches code needing something else; another class can say more, e.g.
      ``LookupError`` when ``None`` means that something wasn't found.

    **Result.**

    - ``value`` itself.

    **Errors.**

    - ``error``: ``value`` is ``None``.
    """
    if value is None:
        raise error(f"{name} must not be None.")
    return value


def _check_bounds(min_value: float | None, max_value: float | None) -> None:
    """Raise ``ValueError`` if either bound is NaN or ``min_value`` exceeds ``max_value``."""
    if _is_nan(min_value) or _is_nan(max_value):
        raise ValueError(f"The bounds must not be NaN, not {min_value} and {max_value}.")
    if min_value is not None and max_value is not None and min_value > max_value:
        raise ValueError(f"min_value ({min_value}) must not exceed max_value ({max_value}).")


def _require(condition: bool, value: object, name: str, requirement: str) -> None:
    """Raise ``ValueError`` saying that ``name`` must be ``requirement``, unless ``condition`` holds."""
    if not condition:
        raise ValueError(f"{name} must be {requirement}, not {value}.")


def _range_text(min_value: float | None, max_value: float | None, include_min: bool, include_max: bool) -> str:
    """Describe the range for an error message, e.g. "at least 0" or "between 0 and 1 (excluding 0)"."""
    if min_value is None and max_value is None:
        return "a number other than NaN"
    if max_value is None:
        return f"at least {min_value}" if include_min else f"greater than {min_value}"
    if min_value is None:
        return f"at most {max_value}" if include_max else f"less than {max_value}"
    excluded = [str(bound) for bound, included in ((min_value, include_min), (max_value, include_max)) if not included]
    return f"between {min_value} and {max_value}" + (f" (excluding {' and '.join(excluded)})" if excluded else "")


def _is_nan(value: object) -> bool:
    """Return whether ``value`` is NaN, for any type (NaN is the only value that isn't equal to itself)."""
    return value != value
