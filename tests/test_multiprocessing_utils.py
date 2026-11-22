"""Tests for the ``mcrotk_utils.multiprocessing_utils`` package itself: its exports, and lazy loading.

(The functions themselves are tested in ``test_multiprocessing_utils_*.py``, one file per internal module.)
"""

import importlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from mcrotk_utils import multiprocessing_utils
from mcrotk_utils.multiprocessing_utils import parallel_map_with_rngs, split_for_workers


def test_all_lists_exactly_the_exported_names():
    """__all__ (a literal list, for type checkers) names exactly the names the package maps to its modules."""
    assert sorted(multiprocessing_utils.__all__) == sorted(multiprocessing_utils._MODULE_OF_NAME)
    assert len(set(multiprocessing_utils.__all__)) == len(multiprocessing_utils.__all__)


@pytest.mark.parametrize("name", multiprocessing_utils.__all__)
def test_every_exported_name_comes_from_its_module(name):
    """Each public name is the object defined in its (private) module."""
    module_name = multiprocessing_utils._MODULE_OF_NAME[name]
    module = importlib.import_module(f"mcrotk_utils.multiprocessing_utils.{module_name}")
    assert getattr(multiprocessing_utils, name) is getattr(module, name)


def test_unknown_names_raise_attribute_error():
    """A name the package doesn't export raises AttributeError (so hasattr and getattr defaults work)."""
    with pytest.raises(AttributeError, match="has no attribute 'no_such_function'"):
        multiprocessing_utils.no_such_function  # noqa: B018
    assert not hasattr(multiprocessing_utils, "no_such_function")


def test_dir_lists_the_exported_names_before_they_are_loaded():
    """dir() includes every public name even before any is used (checked in a fresh interpreter)."""
    code = (
        "import sys, mcrotk_utils.multiprocessing_utils as m; "
        "print(sorted(set(m.__all__) - set(dir(m))), 'numpy' in sys.modules)"
    )
    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert output == "[] False"


def test_the_splitting_functions_dont_load_numpy():
    """Importing the splitting functions loads only their module, not numpy (checked in a fresh interpreter)."""
    code = (
        "import sys; from mcrotk_utils.multiprocessing_utils import split_for_workers, split_evenly; "
        "print(len(split_for_workers(range(5), 2)), 'numpy' in sys.modules)"
    )
    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert output == "2 False"


def test_star_import_gives_the_exported_names():
    """from mcrotk_utils.multiprocessing_utils import * gives exactly the public names."""
    namespace: dict[str, object] = {}
    exec("from mcrotk_utils.multiprocessing_utils import *", namespace)
    assert set(namespace) - {"__builtins__"} == set(multiprocessing_utils.__all__)


def batch_total(key, items, rng):
    """Sum a batch's items plus a draw, tagged with the batch's group."""
    return key, sum(items) + rng.random()


def test_batches_can_be_run_with_parallel_map_with_rngs():
    """split_for_workers' batches feed parallel_map_with_rngs: one call per batch, each with its own generator."""
    batches = split_for_workers(range(10), 4, key=lambda n: n % 2)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = parallel_map_with_rngs(
            batch_total, [b.key for b in batches], [b.items for b in batches], executor=executor, rng=1
        )
    rngs = np.random.default_rng(1).spawn(len(batches))
    assert results == [(b.key, sum(b.items) + rng.random()) for b, rng in zip(batches, rngs, strict=True)]
    assert sorted(n for b in batches for n in b.items) == list(range(10))
