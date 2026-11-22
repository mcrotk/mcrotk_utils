"""Tests for ``mcrotk_utils.misc_utils``' formatting: ``max_width``, ``remove_punctuation`` and ``format_duration``."""

import math
import re
import string
from datetime import timedelta
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from mcrotk_utils.misc_utils import format_duration, max_width, remove_punctuation

NAN = float("nan")


# --- max_width --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "width"),
    [
        (7, 1),
        (-120, 4),
        ([5, -120], 4),
        ((1, 22), 2),
        ({5, 999}, 3),
        (range(1000), 3),
        ([3.5, 10], 3),
        (["a", "bcd"], 3),
        ("hello", 5),
        (b"ab", len(str(b"ab"))),
        ([[1, 2], [3333, [55555]]], 5),
        ((x for x in [1, 22, 333]), 3),
        (10**15, 16),
        (10**400, 401),
    ],
)
def test_max_width_of_single_values_and_containers(values, width):
    """Single values, flat and nested containers; strings and bytes count as single values."""
    assert max_width(values) == width


def test_max_width_of_mappings_uses_values_at_any_depth():
    """Mappings contribute their values, not their keys, at any depth; keys can be measured explicitly."""
    assert max_width({"a": 12, "b": [1, -3000]}) == 5
    assert max_width({"a_long_key": {"x": 5}}) == 1
    assert max_width({"key": "longer value"}) == 12
    assert max_width({"a_long_key": 5}.keys()) == 10


@pytest.mark.parametrize("values", [[], {}, [[], {}], (), range(0)])
def test_max_width_of_nothing_is_zero(values):
    """No values at all give 0."""
    assert max_width(values) == 0


def test_max_width_with_format_spec():
    """Each value is formatted with format_spec before being measured."""
    assert max_width([1234567], format_spec=",") == 9
    assert max_width([3.14159, -10.5], format_spec=".2f") == 6
    assert max_width([5, 12], format_spec="03d") == 3


def test_max_width_with_unsuitable_format_spec_raises():
    """A format_spec that doesn't suit a value raises."""
    with pytest.raises(ValueError):
        max_width(["a"], format_spec=".2f")


def test_max_width_of_numpy_and_pandas():
    """NumPy arrays (of any dimension) and Series contribute their elements; a DataFrame its column names."""
    assert max_width(np.array([1, -100])) == 4
    assert max_width(np.array([[1, 2], [3, 40000]])) == 5
    assert max_width(np.array([1234567]), format_spec=",") == 9
    assert max_width(pd.Series([10, -2000])) == 5
    df = pd.DataFrame({"a": [123456]})
    assert max_width(df) == 1
    assert max_width(df.to_numpy()) == 6
    assert max_width(df.stack()) == 6


def test_max_width_counts_characters():
    """Characters are counted, not display width (a wide character counts as one)."""
    assert max_width(["日本"]) == 2


# --- remove_punctuation -----------------------------------------------------------------------------------------------


def test_remove_punctuation_removes_ascii_punctuation_only_by_default():
    """Every character of string.punctuation goes; letters, digits, whitespace and other characters stay."""
    assert remove_punctuation(f"a{string.punctuation}b 1\t2–é") == "ab 1\t2–é"
    assert remove_punctuation("") == ""


def test_remove_punctuation_unicode():
    """unicode=True also removes Unicode punctuation, but keeps other symbols, letters and whitespace."""
    text = "it’s “quoted” — en–dash, ¿qué? «x» … €5 ©"
    assert remove_punctuation(text) == "it’s “quoted” — en–dash ¿qué «x» … €5 ©"
    assert remove_punctuation(text, unicode=True) == "its quoted  endash qué x  €5 ©"


def test_remove_punctuation_unicode_includes_ascii_symbols():
    """unicode=True still removes the ASCII characters that Unicode counts as symbols, not punctuation."""
    symbols = "$+<=>^`|~"
    assert all(not unicodedata_category(char).startswith("P") for char in symbols)
    assert remove_punctuation(f"a{symbols}b", unicode=True) == "ab"
    assert remove_punctuation(f"a{string.punctuation}b", unicode=True) == "ab"


def unicodedata_category(char):
    """Return the Unicode category of ``char``."""
    import unicodedata

    return unicodedata.category(char)


# --- format_duration --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("duration", "digits", "text"),
    [
        (0, 0, "0 seconds"),
        (1, 0, "1 second"),
        (5, 0, "5 seconds"),
        (59.7, 0, "1 minute, 0 seconds"),
        (0.5, 0, "0 seconds"),
        (1.5, 0, "2 seconds"),
        (1.5, 1, "1.5 seconds"),
        (1, 1, "1.0 seconds"),
        (61, 0, "1 minute, 1 second"),
        (120, 0, "2 minutes, 0 seconds"),
        (3600, 0, "1 hour, 0 minutes, 0 seconds"),
        (3661, 0, "1 hour, 1 minute, 1 second"),
        (7322.25, 2, "2 hours, 2 minutes, 2.25 seconds"),
        (90.25, 1, "1 minute, 30.2 seconds"),
        (59.9996, 3, "1 minute, 0.000 seconds"),
        (90000, 0, "25 hours, 0 minutes, 0 seconds"),
        (-5, 0, "-5 seconds"),
        (-3661, 0, "-1 hour, 1 minute, 1 second"),
        (Fraction(1, 3), 2, "0.33 seconds"),
    ],
)
def test_format_duration(duration, digits, text):
    """Examples: rounding first (with carry, half to even), units shown, singulars, signs."""
    assert format_duration(duration, digits=digits) == text


def test_format_duration_of_timedelta():
    """A timedelta gives the same as its number of seconds, exactly to the microsecond."""
    assert format_duration(timedelta(hours=1, minutes=2, seconds=3.25), digits=2) == "1 hour, 2 minutes, 3.25 seconds"
    assert format_duration(timedelta(seconds=-5)) == "-5 seconds"
    assert format_duration(timedelta(days=1, microseconds=1), digits=6) == "24 hours, 0 minutes, 0.000001 seconds"


@pytest.mark.parametrize(
    ("duration", "digits", "text"),
    [
        (3661, 0, " 1 hour,   1 minute,   1 second"),
        (7322, 0, " 2 hours,  2 minutes,  2 seconds"),
        (7322.25, 2, " 2 hours,  2 minutes,  2.25 seconds"),
        (5, 0, " 5 seconds"),
        (65, 0, " 1 minute,   5 seconds"),
    ],
)
def test_format_duration_aligned(duration, digits, text):
    """aligned=True pads numbers to 2 characters and unit words to their plural's width."""
    assert format_duration(duration, digits=digits, aligned=True) == text


def test_format_duration_aligned_lines_line_up():
    """Aligned durations with the same units and under 100 hours line up: all but the last word has one length."""
    rng = np.random.default_rng(0)
    for digits in range(3):
        lengths = {
            len(format_duration(float(seconds), digits=digits, aligned=True).rsplit(" ", 1)[0])
            for seconds in rng.uniform(3600, 99 * 3600, 200)
        }
        assert len(lengths) == 1


def test_format_duration_aligned_is_the_compact_form_padded():
    """Collapsing the aligned form's spaces gives the compact form, apart from its commas' placement."""
    rng = np.random.default_rng(1)
    for seconds in rng.uniform(-1e6, 1e6, 300):
        for digits in range(3):
            aligned = format_duration(float(seconds), digits=digits, aligned=True)
            compact = format_duration(float(seconds), digits=digits)
            assert " ".join(aligned.replace("- ", "-").split()) == compact


def test_format_duration_round_trips():
    """On random durations, the words give back the duration rounded (half to even) to ``digits`` decimals."""
    rng = np.random.default_rng(2)
    for seconds in np.concatenate([rng.uniform(-1e6, 1e6, 300), rng.uniform(0, 120, 300), rng.integers(0, 10**6, 50)]):
        for digits in range(5):
            text = format_duration(seconds.item(), digits=digits)
            assert parse_duration(text) == round(Fraction(seconds.item()), digits)


def parse_duration(text):
    """Return the number of seconds that a compact duration text states, exactly."""
    sign = -1 if text.startswith("-") else 1
    hours = re.search(r"(\d+) hours?\b", text)
    minutes = re.search(r"(\d+) minutes?\b", text)
    seconds = re.search(r"([\d.]+) seconds?\b", text)
    assert seconds is not None
    # hours appear only when there are any, and minutes when there are any or there are hours
    assert hours is None or int(hours.group(1)) > 0
    assert minutes is not None if hours else minutes is None or int(minutes.group(1)) > 0
    total = Fraction(seconds.group(1))
    total += 60 * int(minutes.group(1)) if minutes else 0
    total += 3600 * int(hours.group(1)) if hours else 0
    return sign * total


@pytest.mark.parametrize("duration", [math.inf, -math.inf, NAN])
def test_format_duration_rejects_non_finite_durations(duration):
    """Infinite or NaN durations raise ValueError."""
    with pytest.raises(ValueError, match="duration must be finite"):
        format_duration(duration)


def test_format_duration_checks_digits():
    """Negative digits raise ValueError; non-integer digits raise TypeError."""
    with pytest.raises(ValueError, match="digits must be non-negative"):
        format_duration(5, digits=-1)
    with pytest.raises(TypeError):
        format_duration(5, digits=1.0)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
