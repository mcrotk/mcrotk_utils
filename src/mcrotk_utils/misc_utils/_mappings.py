"""Dicts built from other collections, raising on duplicates rather than silently losing entries (private).

Public through ``mcrotk_utils.misc_utils``. Uses only the standard library.
"""

from collections.abc import Hashable, Iterable, Mapping


def invert_dict[K, V: Hashable](mapping: Mapping[K, V]) -> dict[V, K]:
    """Return a dict mapping each value of ``mapping`` back to its key, raising if any value appears twice.

    A plain ``{v: k for k, v in mapping.items()}`` silently keeps only the last of the keys sharing a value.

    **Arguments.**

    - ``mapping``: a dict or other mapping, with hashable values.

    **Result.**

    - A new dict, in ``mapping``'s order.

    **Errors.**

    - ``ValueError``: two keys have the same value (or equal ones, such as ``1`` and ``1.0``), naming both keys.
    - ``TypeError``: a value isn't hashable.
    """
    inverted: dict[V, K] = {}
    for key, value in mapping.items():
        if value in inverted:
            raise ValueError(f"The value {value!r} belongs to more than one key: {inverted[value]!r} and {key!r}.")
        inverted[value] = key
    return inverted


def index_map[T: Hashable](items: Iterable[T]) -> dict[T, int]:
    """Return a dict mapping each item to its position in ``items``, raising if any item appears twice.

    The fast alternative to calling ``list.index`` repeatedly. A plain ``{x: i for i, x in enumerate(items)}``
    silently keeps only the last position of a repeated item.

    **Arguments.**

    - ``items``: any iterable of hashable items, e.g. a list, or an ``Enum`` class (its members, without aliases).

    **Result.**

    - A new dict from item to position (counting from 0), in the items' order.

    **Errors.**

    - ``ValueError``: an item appears twice (or equal ones do, such as ``1`` and ``1.0``), naming both positions.
    - ``TypeError``: an item isn't hashable.
    """
    positions: dict[T, int] = {}
    for position, item in enumerate(items):
        if item in positions:
            raise ValueError(f"{item!r} appears more than once: at positions {positions[item]} and {position}.")
        positions[item] = position
    return positions
