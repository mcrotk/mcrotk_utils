"""Tests for ``mcrotk_utils.misc_utils``' sorting: ``sort_with_directions``."""

import itertools
import math
import random
from collections.abc import Callable, Sequence
from datetime import date
from functools import cmp_to_key
from typing import Any, NamedTuple

import pytest

from mcrotk_utils.misc_utils import sort_with_directions


class Person(NamedTuple):
    """A record to sort in the examples."""

    name: str
    city: str
    score: int


PEOPLE = [
    Person("carol", "Boston", 88),
    Person("alice", "Austin", 92),
    Person("bob", "Boston", 92),
    Person("dave", "Austin", 75),
    Person("erin", "Chicago", 88),
]

WORDS = ["pear", "fig", "apple", "kiwi", "plum", "date"]


def reference_sort_by_comparison[T](
    items: Sequence[T], key: Callable[[T], Sequence[Any]], reverse: Sequence[bool]
) -> list[T]:
    """Sort with a comparison function: compare the keys entry by entry, flipping the result for descending ones."""

    def compare(x: T, y: T) -> int:
        """Return -1, 0 or 1 as x comes before, ties with, or comes after y."""
        for a, b, descending in zip(key(x), key(y), reverse, strict=True):
            if a != b:
                ascending_result = 1 if a > b else -1
                return -ascending_result if descending else ascending_result
        return 0

    return sorted(items, key=cmp_to_key(compare))


def reference_sort_by_negation(items: Sequence[Sequence[float]], reverse: Sequence[bool]) -> list[Sequence[float]]:
    """Sort tuples of numbers by negating the entries that are to be descending."""

    def negated(row: Sequence[float]) -> tuple[float, ...]:
        """The row with its descending entries negated."""
        return tuple(-x if descending else x for x, descending in zip(row, reverse, strict=True))

    return sorted(items, key=negated)


# Examples


def test_score_descending_then_name_ascending():
    """Highest score first, ties by name from A to Z."""
    result = sort_with_directions(PEOPLE, key=lambda p: (p.score, p.name), reverse=[True, False])
    assert [p.name for p in result] == ["alice", "bob", "carol", "erin", "dave"]


def test_three_keys_in_mixed_directions():
    """City A to Z, then score highest first, then name A to Z."""
    result = sort_with_directions(PEOPLE, key=lambda p: (p.city, p.score, p.name), reverse=[False, True, False])
    assert [p.name for p in result] == ["alice", "dave", "bob", "carol", "erin"]


def test_two_text_keys_in_opposite_directions():
    """Strings can't be negated, but can be sorted descending: length shortest first, ties from Z to A."""
    result = sort_with_directions(WORDS, key=lambda w: (len(w), w), reverse=[False, True])
    assert result == ["fig", "plum", "pear", "kiwi", "date", "apple"]


def test_value_count_pairs_most_common_first():
    """(value, count) pairs, most common first, ties by value."""
    pairs = [(2, 30), (3, 59), (4, 92), (5, 104), (6, 130), (7, 176), (8, 133), (9, 110), (10, 76), (11, 55)]
    result = sort_with_directions(pairs, key=lambda p: (p[1], p[0]), reverse=[True, False])
    assert result[:4] == [(7, 176), (8, 133), (6, 130), (9, 110)]


def test_dates_descending():
    """Entries of any comparable type work, e.g. dates: newest first, ties by name."""
    events = [("b", date(2024, 1, 1)), ("a", date(2025, 6, 1)), ("c", date(2024, 1, 1))]
    result = sort_with_directions(events, key=lambda e: (e[1], e[0]), reverse=[True, False])
    assert result == [("a", date(2025, 6, 1)), ("b", date(2024, 1, 1)), ("c", date(2024, 1, 1))]


# Compared with other ways of sorting


@pytest.mark.parametrize("reverse", list(itertools.product([False, True], repeat=4)))
def test_matches_a_comparison_function(reverse):
    """Random rows with many ties, in every combination of four directions, match sorting with a comparison."""
    rng = random.Random(sum(2**i for i, r in enumerate(reverse) if r))
    rows = [
        (rng.randint(0, 3), rng.choice(["a", "b", "ab", "ba"]), rng.random() < 0.5, rng.choice([1.5, -2.0]), i)
        for i in range(300)
    ]

    def key(row: tuple[int, str, bool, float, int]) -> tuple[int, str, bool, float]:
        """The first four entries (the fifth records the input order)."""
        return row[:4]

    assert sort_with_directions(rows, key=key, reverse=reverse) == reference_sort_by_comparison(rows, key, reverse)


@pytest.mark.parametrize("reverse", list(itertools.product([False, True], repeat=3)))
def test_matches_negating_numbers(reverse):
    """For numbers, matches sorting with the descending entries negated."""
    rng = random.Random(len(reverse) + sum(reverse))
    rows = [(rng.randint(0, 4), rng.uniform(-1, 1), rng.randint(-2, 2)) for _ in range(200)]
    assert sort_with_directions(rows, key=lambda r: r, reverse=reverse) == reference_sort_by_negation(rows, reverse)


@pytest.mark.parametrize("reverse", [False, True])
def test_a_single_bool_is_sorted(reverse):
    """With a single bool, the result is sorted(items, key=key, reverse=reverse), ties in input order."""
    rows = [(i % 3, -i) for i in range(20)]
    assert sort_with_directions(rows, key=lambda r: r[:1], reverse=reverse) == sorted(
        rows, key=lambda r: r[:1], reverse=reverse
    )


def test_a_single_bool_doesnt_check_the_keys():
    """With a single bool, key is used as sorted uses it: any comparable value, of any length."""
    assert sort_with_directions(["bb", "a", "ccc"], key=lambda w: w, reverse=True) == ["ccc", "bb", "a"]
    assert sort_with_directions([2, 1], key=lambda n: (n,) * n) == [1, 2]


# Ties, inputs and results


@pytest.mark.parametrize("reverse", list(itertools.product([False, True], repeat=2)))
def test_ties_keep_the_input_order(reverse):
    """Items whose entries are all equal keep their order from the input, in every direction."""
    rows = [(i % 2, i % 3, i) for i in range(30)]
    result = sort_with_directions(rows, key=lambda r: r[:2], reverse=reverse)
    for _, group in itertools.groupby(result, key=lambda r: r[:2]):
        positions = [r[2] for r in group]
        assert positions == sorted(positions)


def test_all_entries_descending():
    """All entries descending is the reverse order of the keys, with ties still in input order."""
    rows = [(1, "a", 0), (2, "b", 1), (1, "a", 2), (2, "a", 3)]
    result = sort_with_directions(rows, key=lambda r: r[:2], reverse=[True, True])
    assert result == [(2, "b", 1), (2, "a", 3), (1, "a", 0), (1, "a", 2)]


def test_a_generator_keeps_every_item():
    """A generator (or other one-pass iterator) gives the same result as a list: no item is lost."""
    expected = sort_with_directions(WORDS, key=lambda w: (len(w), w), reverse=[False, True])
    assert sort_with_directions((w for w in WORDS), key=lambda w: (len(w), w), reverse=[False, True]) == expected
    assert sort_with_directions(iter(WORDS), key=lambda w: (len(w), w), reverse=[False, True]) == expected
    assert sort_with_directions(map(str, WORDS), key=lambda w: (len(w), w), reverse=[False, True]) == expected


@pytest.mark.parametrize("reverse", [[True], [False, True], False, True])
def test_empty_input(reverse):
    """No items give an empty list, without calling key."""

    def key(item: object) -> tuple[object, ...]:
        """Fail if called."""
        raise AssertionError("key was called")

    assert sort_with_directions([], key=key, reverse=reverse) == []
    assert sort_with_directions(iter([]), key=key, reverse=reverse) == []


def test_one_item():
    """One item gives a list of that item."""
    assert sort_with_directions(["x"], key=lambda w: (w, len(w)), reverse=[True, False]) == ["x"]


def test_no_directions_keeps_the_input_order():
    """An empty reverse, with keys of no entries, compares nothing, so the input order is kept."""
    assert sort_with_directions([3, 1, 2], key=lambda n: (), reverse=[]) == [3, 1, 2]


def test_key_is_called_once_per_item():
    """key is called exactly once for each item, however many entries and passes."""
    calls = []

    def key(n: int) -> tuple[int, int, int]:
        """Record the call, and return entries that need several passes."""
        calls.append(n)
        return (n % 3, n % 5, n)

    items = list(range(100, 0, -1))
    sort_with_directions(items, key=key, reverse=[True, False, True])
    assert sorted(calls) == sorted(items)


def test_returns_a_new_list_and_leaves_the_input_alone():
    """The result is a new list; the input list isn't changed."""
    items = [3, 1, 2]
    result = sort_with_directions(items, key=lambda n: (n,), reverse=[False])
    assert result == [1, 2, 3]
    assert items == [3, 1, 2]
    assert result is not items


def test_keys_can_be_lists_or_other_sequences():
    """key may return a list or another sequence (such as a range) rather than a tuple."""
    assert sort_with_directions([1, 3, 2], key=lambda n: [n % 2, n], reverse=[True, False]) == [1, 3, 2]
    assert sort_with_directions([2, 1, 3], key=lambda n: range(n, n + 2), reverse=[True, True]) == [3, 2, 1]


def test_reverse_can_be_a_tuple():
    """reverse may be any sequence of bools, e.g. a tuple."""
    assert sort_with_directions(WORDS, key=lambda w: (len(w), w), reverse=(False, True)) == sort_with_directions(
        WORDS, key=lambda w: (len(w), w), reverse=[False, True]
    )


def test_nan_entries_keep_every_item():
    """NaN entries can leave the order not fully sorted (as with sorted), but no item is lost or repeated."""
    rows = [(math.nan, 1), (2.0, 2), (math.nan, 3), (1.0, 4), (3.0, 5)]
    result = sort_with_directions(rows, key=lambda r: r, reverse=[True, False])
    assert sorted(result, key=lambda r: r[1]) == rows


# Errors


@pytest.mark.parametrize("length", [1, 3])
def test_wrong_number_of_entries_raises(length):
    """A key with fewer or more entries than reverse raises ValueError naming the item's position (not just the
    first item's key is checked)."""

    def key(n: int) -> tuple[int, ...]:
        """Two entries, except for the item 7."""
        return (n,) * (length if n == 7 else 2)

    with pytest.raises(ValueError, match=f"key returned {length} entries for the item at position 3, but reverse"):
        sort_with_directions([1, 2, 3, 7, 4], key=key, reverse=[True, False])


@pytest.mark.parametrize("key_value", ["ab", b"ab", 5, None, {1, 2}])
def test_key_must_return_a_sequence_of_entries(key_value):
    """With a sequence reverse, a key that isn't a sequence, or is a string or bytes, raises TypeError."""
    with pytest.raises(TypeError, match=r"key must return a sequence of entries, .* at position 0"):
        sort_with_directions([1, 2], key=lambda n: key_value, reverse=[True, False])


@pytest.mark.parametrize("reverse", ["TF", b"\x01", 5, None, 1.0])
def test_reverse_must_be_a_bool_or_a_sequence_of_bools(reverse):
    """A reverse that isn't a bool or a sequence raises TypeError (strings aren't sequences of directions)."""
    with pytest.raises(TypeError, match="reverse must be a bool or a sequence of bools"):
        sort_with_directions([1, 2], key=lambda n: (n,), reverse=reverse)


@pytest.mark.parametrize("reverse", [[1, 0], [True, None], [True, "desc"]])
def test_reverse_must_contain_only_bools(reverse):
    """Directions other than True and False (such as 1 and 0) raise TypeError."""
    with pytest.raises(TypeError, match="reverse must contain only True and False"):
        sort_with_directions([1, 2], key=lambda n: (n, n), reverse=reverse)
