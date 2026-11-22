"""Tests for the ``mcrotk_utils.plotting_utils`` package itself: its exports.

(``label_bars`` is tested in ``test_plotting_utils_labels.py``, and the type aliases in ``test_typing.py``,
with the package's other type checks.)
"""

import importlib
import pkgutil

import pytest

from mcrotk_utils import plotting_utils


def public_names_of_modules():
    """Return the public names defined in the package's (private) modules, with each one's module."""
    names = {}
    for module_info in pkgutil.iter_modules(plotting_utils.__path__):
        module = importlib.import_module(f"mcrotk_utils.plotting_utils.{module_info.name}")
        for name, value in vars(module).items():
            # names defined in the module itself (not imported into it), and not private
            defined_here = getattr(value, "__module__", module.__name__) == module.__name__
            if not name.startswith("_") and defined_here and not isinstance(value, type(importlib)):
                names[name] = module
    return names


def test_all_lists_exactly_the_public_names_of_the_modules():
    """__all__ names every public name defined in the modules, and nothing else, once each."""
    assert sorted(plotting_utils.__all__) == sorted(public_names_of_modules())
    assert len(set(plotting_utils.__all__)) == len(plotting_utils.__all__)


@pytest.mark.parametrize("name", plotting_utils.__all__)
def test_every_exported_name_comes_from_its_module(name):
    """Each public name is the object defined in its (private) module."""
    assert getattr(plotting_utils, name) is getattr(public_names_of_modules()[name], name)


def test_star_import_gives_the_exported_names():
    """from mcrotk_utils.plotting_utils import * gives exactly the public names."""
    namespace: dict[str, object] = {}
    exec("from mcrotk_utils.plotting_utils import *", namespace)
    assert set(namespace) - {"__builtins__"} == set(plotting_utils.__all__)
