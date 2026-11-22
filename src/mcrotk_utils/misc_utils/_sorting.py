"""Sorting by several keys, each in its own direction (private).

Public through ``mcrotk_utils.misc_utils``. Uses only the standard library.
"""

import itertools
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, overload

if TYPE_CHECKING:
    # The type sorted's key returns; typeshed's, so only for type checkers (hence quoted where used: Python 3.12
    # evaluates annotations when the function is defined)
    from _typeshed import SupportsRichComparison


# With one direction, key is anything sorted accepts; with one direction per entry, it returns a sequence of entries
@overload
def sort_with_directions[T](
    items: Iterable[T], /, key: Callable[[T], "SupportsRichComparison"], reverse: bool = False
) -> list[T]: ...
@overload
def sort_with_directions[T](
    items: Iterable[T], /, key: Callable[[T], Sequence[Any]], reverse: Sequence[bool]
) -> list[T]: ...
def sort_with_directions[T](
    items: Iterable[T],
    /,
    key: Callable[[T], Any],
    reverse: bool | Sequence[bool] = False,
) -> list[T]:
    """Return ``items`` sorted by several keys, each ascending or descending on its own.

    For example, score from highest to lowest, with ties broken by name from A to Z:
    ``sort_with_directions(people, key=lambda p: (p.score, p.name), reverse=[True, False])``. The built-in
    ``sorted`` has one ``reverse`` for the whole sort, and negating a descending key (``-p.score``) only works for
    numbers; this works for entries of any type that can be compared, such as strings or dates.

    Items whose entries are all equal keep their order from ``items`` (the sort is stable). ``key`` is called once
    per item. Entries that don't compare consistently, such as NaN, can give an order that isn't fully sorted, as
    with ``sorted``.

    **Arguments.**

    - ``items``: any iterable, including a generator.
    - ``key``: a function returning a tuple (or another sequence, such as a list) of entries for each item, from
      the most important to the least.
    - ``reverse``: ``True`` for descending and ``False`` for ascending, either once for the whole sort, which is
      then exactly ``sorted(items, key=key, reverse=reverse)``, or one per entry of the key, in the same order.

    **Result.**

    - A new list of the items, in sorted order.

    **Errors.**

    - ``TypeError``: ``reverse`` isn't a bool or a sequence of bools; or (with a sequence ``reverse``) ``key``
      returned something other than a sequence of entries for an item (a string counts as one entry, not a
      sequence of characters, so it must be wrapped, e.g. ``(name,)``).
    - ``ValueError``: with a sequence ``reverse``, ``key`` returned a different number of entries for an item than
      ``reverse`` has, naming the item's position.
    """
    if isinstance(reverse, bool):
        return sorted(items, key=key, reverse=reverse)
    directions = _directions(reverse)
    keyed = [(_entries(key(item), position, len(directions)), item) for position, item in enumerate(items)]
    # One stable pass per run of neighboring entries with the same direction, least important run first: each pass
    # keeps ties in the order of the passes before it. Faster than a comparison function through cmp_to_key, as the
    # comparisons in each pass happen within sort itself rather than in a Python function.
    for start, stop, descending in reversed(_runs(directions)):
        _sort_by_entries(keyed, start, stop, descending)
    return [item for _, item in keyed]


def _directions(reverse: object) -> tuple[bool, ...]:
    """Return ``reverse`` as a tuple of bools, raising ``TypeError`` if it isn't a sequence of bools."""
    if isinstance(reverse, str | bytes) or not isinstance(reverse, Iterable):
        raise TypeError(f"reverse must be a bool or a sequence of bools, not {type(reverse).__name__}.")
    directions = tuple(reverse)
    for direction in directions:
        if not isinstance(direction, bool):
            raise TypeError(f"reverse must contain only True and False, not {direction!r}.")
    return directions


def _entries(key_value: object, position: int, length: int) -> tuple[Any, ...]:
    """Return one item's key as a tuple, checking it's a sequence (not a string) with ``length`` entries."""
    # Strings and bytes are sequences, but splitting them into characters would sort by the wrong thing
    if isinstance(key_value, str | bytes | bytearray) or not isinstance(key_value, Sequence):
        raise TypeError(
            f"key must return a sequence of entries, such as a tuple, not {type(key_value).__name__} "
            f"(for the item at position {position})."
        )
    if len(key_value) != length:
        raise ValueError(
            f"key returned {len(key_value)} entries for the item at position {position}, "
            f"but reverse has {length} directions."
        )
    return tuple(key_value)


def _runs(directions: Sequence[bool]) -> list[tuple[int, int, bool]]:
    """Split the entry positions into runs of neighboring entries with the same direction: (start, stop, direction)."""
    runs: list[tuple[int, int, bool]] = []
    start = 0
    for descending, group in itertools.groupby(directions):
        stop = start + len(list(group))
        runs.append((start, stop, descending))
        start = stop
    return runs


def _sort_by_entries(keyed: list[tuple[tuple[Any, ...], Any]], start: int, stop: int, descending: bool) -> None:
    """Stably sort (key, item) pairs in place by entries ``start`` to ``stop`` of their keys."""
    if stop - start == 1:
        # Comparing the entry itself is faster than comparing a one-entry tuple (about 10 ms against 17 ms for
        # 100,000 items), and gives the same order
        keyed.sort(key=lambda pair: pair[0][start], reverse=descending)
    else:
        keyed.sort(key=lambda pair: pair[0][start:stop], reverse=descending)
