"""Tests for ``mcrotk_utils.misc_utils``' mappings: ``invert_dict`` and ``index_map``."""

import enum

import pytest

from mcrotk_utils.misc_utils import index_map, invert_dict


def test_invert_dict():
    """Values map back to their keys, in the mapping's order, in a new dict."""
    original = {"a": 1, "b": 2, "c": 3}
    inverted = invert_dict(original)
    assert inverted == {1: "a", 2: "b", 3: "c"}
    assert list(inverted) == [1, 2, 3]
    assert original == {"a": 1, "b": 2, "c": 3}
    assert invert_dict({}) == {}


@pytest.mark.parametrize("mapping", [{"a": 1, "b": 1}, {"a": 1, "b": 1.0}, {"a": 1, "b": True}])
def test_invert_dict_raises_on_shared_values(mapping):
    """Two keys with the same (or equal) values raise ValueError naming both keys."""
    with pytest.raises(ValueError, match="belongs to more than one key: 'a' and 'b'"):
        invert_dict(mapping)


def test_invert_dict_raises_on_unhashable_values():
    """Unhashable values raise TypeError."""
    with pytest.raises(TypeError):
        invert_dict({"a": [1]})  # type: ignore[type-var]  # pyright: ignore[reportArgumentType]


def test_index_map():
    """Each item maps to its position, in order, from any iterable."""
    assert index_map(["x", "y", "z"]) == {"x": 0, "y": 1, "z": 2}
    assert index_map(iter("ab")) == {"a": 0, "b": 1}
    assert index_map([]) == {}


def test_index_map_of_an_enum_skips_aliases():
    """An Enum class gives its members' positions; aliases aren't members, so they don't count as repeats."""

    class Colour(enum.Enum):
        RED = 1
        GREEN = 2
        CRIMSON = 1  # an alias of RED

    assert index_map(Colour) == {Colour.RED: 0, Colour.GREEN: 1}


@pytest.mark.parametrize("items", [["x", "y", "x"], [1, 2, 1.0]])
def test_index_map_raises_on_repeated_items(items):
    """A repeated (or equal) item raises ValueError naming both positions."""
    with pytest.raises(ValueError, match="appears more than once: at positions 0 and 2"):
        index_map(items)


def test_index_map_raises_on_unhashable_items():
    """Unhashable items raise TypeError."""
    with pytest.raises(TypeError):
        index_map([[1]])  # type: ignore[type-var]  # pyright: ignore[reportArgumentType]
