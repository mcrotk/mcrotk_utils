"""Formatting for output: the widest formatted value, durations in words, and text without punctuation (private).

Public through ``mcrotk_utils.misc_utils``. Uses only the standard library.
"""

import math
import operator
import string
import unicodedata
from collections.abc import Iterable, Iterator, Mapping
from datetime import timedelta
from fractions import Fraction

# --- Text -------------------------------------------------------------------------------------------------------------


def max_width(values: object, *, format_spec: str = "") -> int:
    """Return the length of the longest single value in ``values`` when formatted, e.g. for aligning printed output.

    ``values`` can be a single value or a container of them, nested to any depth. E.g. ``max_width([5, -120])`` is
    4, ``max_width(["a", "bcd"])`` is 3, and ``f"{x:>{max_width(xs)}}"`` right-aligns each ``x`` of ``xs``.

    **Arguments.**

    - ``values``: what to measure:

      - strings and bytes are single values (so ``max_width("hello")`` is 5), not containers of characters;
      - mappings (such as dicts) contribute their values, not their keys (pass ``d.keys()`` to measure the keys);
      - other iterables (lists, tuples, sets, ranges, generators, NumPy arrays, pandas Series) contribute their
        elements. Iterating a pandas DataFrame gives its column names, so ``max_width(df)`` measures those: pass
        ``df.to_numpy()`` or ``df.stack()`` for the cells;
      - anything else is a single value.

    - ``format_spec``: how each value is formatted, as in ``format(value, format_spec)``: e.g. ``","`` for
      thousands separators or ``".2f"`` for two decimals. The default, ``""``, is the same as ``str``.

    **Result.**

    - The number of characters in the longest formatted value; 0 if there are no values. Characters are counted,
      which isn't the display width of wide characters (e.g. CJK characters or emoji).

    **Errors.**

    - ``ValueError`` or ``TypeError``: ``format_spec`` doesn't suit a value (e.g. ``".2f"`` for a string).
    """
    return max((len(format(value, format_spec)) for value in _single_values(values)), default=0)


def _single_values(values: object) -> Iterator[object]:
    """Yield the single values in ``values``, looking inside containers at any depth (a mapping's values)."""
    # strings and bytes are iterable, but are single values here
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        yield values
        return
    items = values.values() if isinstance(values, Mapping) else values
    for item in items:
        yield from _single_values(item)


# Built once, rather than on every call
_ASCII_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def remove_punctuation(text: str, *, unicode: bool = False) -> str:
    """Return ``text`` with its punctuation characters removed, wherever they are.

    **Arguments.**

    - ``text``: the text.
    - ``unicode``: ``False`` (the default) removes the ASCII punctuation characters of ``string.punctuation``:
      ``!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~``. ``True`` also removes every Unicode punctuation character (category
      ``P``: e.g. curly quotes, dashes, ``¿``, ``«»``, ``…``). Both remove all of ``string.punctuation``, including
      the 9 characters Unicode counts as symbols rather than punctuation (``$+<=>^`|~``).

    **Result.**

    - The text without those characters; everything else (letters, digits, whitespace, other symbols) is kept.
    """
    without_ascii = text.translate(_ASCII_PUNCTUATION_TABLE)
    if not unicode:
        return without_ascii
    return "".join(char for char in without_ascii if not unicodedata.category(char).startswith("P"))


# --- Durations --------------------------------------------------------------------------------------------------------


_SECONDS_PER_MINUTE = 60
_MINUTES_PER_HOUR = 60


def format_duration(duration: float | timedelta, *, digits: int = 0, aligned: bool = False) -> str:
    """Return a duration in words, e.g. ``"1 hour, 2 minutes, 3.5 seconds"``.

    Hours appear only when there are any, and minutes when there are any or there are hours; seconds always do.
    There are no days (``25 hours``). E.g. ``format_duration(3661)`` is ``"1 hour, 1 minute, 1 second"`` and
    ``format_duration(90.25, digits=1)`` is ``"1 minute, 30.2 seconds"``.

    **Arguments.**

    - ``duration``: a number of seconds (an int, float or ``Fraction``) or a ``timedelta``. Negative durations get
      a leading minus sign.
    - ``digits``: how many decimals to show for the seconds (0, the default, for whole seconds). The duration is
      rounded (half to even) to that many decimals before being split into hours, minutes and seconds, so
      rounding carries over: 59.9996 seconds with 3 digits is ``"1 minute, 0.000 seconds"``.
    - ``aligned``: ``True`` pads every number to at least 2 characters and every unit's word but the last to the
      width of its plural, so that lines with the same units line up (all but the last word), e.g.
      ``" 1 hour,   1 minute,   1 second"`` and ``" 2 hours,  2 minutes,  2 seconds"``. ``False`` (the default)
      gives the compact form.

    **Result.**

    - The duration in words. Units are singular only when their number is exactly "1" as shown (``"1 second"``,
      but ``"1.0 seconds"`` and ``"1.5 seconds"``).

    **Errors.**

    - ``ValueError``: ``duration`` isn't finite, or ``digits`` is negative.
    - ``TypeError``: ``digits`` isn't an integer.
    """
    digits = _checked_digits(digits)
    seconds = _exact_seconds(duration)
    sign = "-" if seconds < 0 else ""
    # whole units of 10 ** -digits seconds: rounding the total first lets the rounding carry into minutes and hours
    units = round(abs(seconds) * 10**digits)
    parts = _duration_parts(units, digits, aligned)
    return sign + (" " if aligned else ", ").join(parts)


def _checked_digits(digits: int) -> int:
    """Return ``digits`` as an int, raising if it isn't a non-negative integer."""
    index = operator.index(digits)  # TypeError for floats etc.
    if index < 0:
        raise ValueError(f"digits must be non-negative, not {index}.")
    return index


def _exact_seconds(duration: float | timedelta) -> Fraction:
    """Return the duration in seconds, exactly, raising ``ValueError`` if it isn't finite."""
    if isinstance(duration, timedelta):
        # whole microseconds, exactly (total_seconds() would round to a float)
        return Fraction(duration // timedelta(microseconds=1), 1_000_000)
    if not math.isfinite(duration):
        raise ValueError(f"duration must be finite, not {duration}.")
    return Fraction(duration)


def _duration_parts(units: int, digits: int, aligned: bool) -> list[str]:
    """Return the hours, minutes and seconds parts of a duration of ``units`` units of ``10 ** -digits`` seconds."""
    scale = 10**digits
    total_minutes, second_units = divmod(units, _SECONDS_PER_MINUTE * scale)
    hours, minutes = divmod(total_minutes, _MINUTES_PER_HOUR)
    whole_seconds, fraction = divmod(second_units, scale)
    seconds_text = f"{whole_seconds:{2 if aligned else ''}}" + (f".{fraction:0{digits}d}" if digits else "")
    parts = []
    if hours:
        parts.append(_unit_part(f"{hours:{2 if aligned else ''}}", hours == 1, "hour", aligned, last=False))
    if hours or minutes:
        parts.append(_unit_part(f"{minutes:{2 if aligned else ''}}", minutes == 1, "minute", aligned, last=False))
    singular = digits == 0 and whole_seconds == 1
    parts.append(_unit_part(seconds_text, singular, "second", aligned, last=True))
    return parts


def _unit_part(number: str, singular: bool, unit: str, aligned: bool, last: bool) -> str:
    """Return one part, e.g. ``"2 hours"``; when aligned and not last, with its comma and padding to the plural's width.

    In the compact form the parts' commas come from joining them.
    """
    word = unit if singular else unit + "s"
    if not aligned or last:
        return f"{number} {word}"
    # padded to the width of the plural and its comma, so that "hour," and "hours," take the same space
    return f"{number} {word + ',':<{len(unit) + 2}}"
