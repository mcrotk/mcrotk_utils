"""Helpers for running work in parallel: a parallel map giving each call its own random generator, and
splitting work among workers.

Import everything from here (e.g. ``from mcrotk_utils.multiprocessing_utils import parallel_map_with_rngs``): how
the code is split into modules is internal, and may change.

- ``parallel_map_with_rngs``: call a function once per set of arguments in worker processes (or any executor or
  pool), passing each call its own generator spawned from one seed, so the results don't depend on how the calls
  are split among workers. Needs numpy (``mcrotk-utils[numpy]``).
- ``split_evenly``: split a list into a number of consecutive parts whose lengths differ by at most one.
- ``allocate_workers``: allocate workers among groups of jobs that can't be mixed, minimizing the most jobs any
  worker gets.
- ``split_for_workers``: split items into batches for workers, keeping groups apart (``Batch``), with the largest
  batch as small as possible.

Names are loaded lazily, when first used, so that each needs only its own dependencies: the splitting functions
need only the standard library.
"""

import importlib
from typing import TYPE_CHECKING, Any

# Every public name, with the (private) module defining it
_MODULE_OF_NAME = {
    "parallel_map_with_rngs": "_mapping",
    "Batch": "_splitting",
    "allocate_workers": "_splitting",
    "split_evenly": "_splitting",
    "split_for_workers": "_splitting",
}

# A literal list, as type checkers require (a test checks it matches _MODULE_OF_NAME)
__all__ = ["Batch", "allocate_workers", "parallel_map_with_rngs", "split_evenly", "split_for_workers"]

if TYPE_CHECKING:
    # what type checkers and IDEs see; at run time the names come from __getattr__ below
    from mcrotk_utils.multiprocessing_utils._mapping import parallel_map_with_rngs
    from mcrotk_utils.multiprocessing_utils._splitting import (
        Batch,
        allocate_workers,
        split_evenly,
        split_for_workers,
    )


def __getattr__(name: str) -> Any:
    """Return a public name, importing the module that defines it on first use (PEP 562).

    Importing only that module means only its dependencies are needed (e.g. not numpy for ``split_evenly``).
    """
    module_name = _MODULE_OF_NAME.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module_name}"), name)
    # cache it, so later lookups don't go through __getattr__
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the public names along with the module's own, as if they had all been imported."""
    return sorted({*globals(), *__all__})
