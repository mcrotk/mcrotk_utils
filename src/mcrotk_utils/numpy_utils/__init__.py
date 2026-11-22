"""NumPy helpers: type aliases for arrays and random seeds, interleaving arrays in blocks, and comparing entries with
reference entries along an axis.

Needs numpy, which ``mcrotk-utils`` installs only with its ``numpy`` extra (``mcrotk-utils[numpy]``). Import
everything from here (e.g. ``from mcrotk_utils.numpy_utils import interleave``): how the code is split into modules
is internal, and may change.

- Type aliases: ``NDArrayInt`` and ``NDArrayFloat`` (arrays of integers or floats of any width) and ``SeedLike``
  (what ``np.random.default_rng`` accepts as a seed).
- ``interleave``: interleave arrays along an axis, taking a block of entries at a time from each in turn.
- ``compare_along``: compare each entry with a reference entry along an axis (the previous, next, first or last
  one, or one at a given index), as a difference, relative change or ratio; with the type aliases
  ``ComparisonReference`` and ``ComparisonMeasure`` for its arguments.
"""

# Checked once here: Python runs a package's __init__ before any of its modules, so this covers them all
try:
    import numpy  # noqa: F401
except ModuleNotFoundError as error:
    # numpy is optional for the package as a whole, so say which extra provides it, not just that it's missing
    raise ModuleNotFoundError(
        "mcrotk_utils.numpy_utils needs numpy: install mcrotk-utils with its extra, mcrotk-utils[numpy]",
        name="numpy",
    ) from error

# Plain imports, not lazy ones as in stats_utils: every module needs numpy, the package's one dependency, so
# loading names lazily would save no imports
from mcrotk_utils.numpy_utils._comparisons import ComparisonMeasure, ComparisonReference, compare_along
from mcrotk_utils.numpy_utils._interleaving import interleave
from mcrotk_utils.numpy_utils._typing import NDArrayFloat, NDArrayInt, SeedLike

__all__ = [
    "ComparisonMeasure",
    "ComparisonReference",
    "NDArrayFloat",
    "NDArrayInt",
    "SeedLike",
    "compare_along",
    "interleave",
]
