"""Helpers for code that works with either dense or sparse matrices ("possibly sparse"): converting to dense, and
stacking blocks, sparse when any block is.

Needs scipy, which ``mcrotk-utils`` installs only with its ``scipy`` extra (``mcrotk-utils[scipy]``, which brings
numpy). Import everything from here (e.g. ``from mcrotk_utils.sparse_utils import ps_hstack``): how the code is
split into modules is internal, and may change.

The sparse inputs are scipy's sparse arrays (``csr_array``, ``coo_array`` and the other formats), not its older
sparse matrices (``csr_matrix`` etc.).

- Type aliases: ``SparseArray`` (a scipy sparse array in any format), ``PsMatrix`` (a NumPy array or a scipy sparse
  array) and ``SparseFormat`` (scipy's names for its sparse formats, as the stacking functions' ``format`` argument
  takes them).
- ``as_dense``: a NumPy array, whether given a sparse array or not.
- ``ps_hstack``, ``ps_vstack`` and ``ps_block``: stack matrices side by side, one above another, or in a grid
  (where ``None`` is a block of zeros); the result is sparse if any block is sparse, and dense otherwise.
"""

# Checked once here: Python runs a package's __init__ before any of its modules, so this covers them all
try:
    import scipy.sparse  # noqa: F401
except ModuleNotFoundError as error:
    # scipy is optional for the package as a whole, so say which extra provides it, not just that it's missing
    raise ModuleNotFoundError(
        "mcrotk_utils.sparse_utils needs scipy: install mcrotk-utils with its extra, mcrotk-utils[scipy]",
        name="scipy",
    ) from error

# Plain imports, not lazy ones as in stats_utils: every module needs scipy, so loading names lazily would save no
# imports
from mcrotk_utils.sparse_utils._conversion import as_dense
from mcrotk_utils.sparse_utils._stacking import ps_block, ps_hstack, ps_vstack
from mcrotk_utils.sparse_utils._typing import PsMatrix, SparseArray, SparseFormat

__all__ = ["PsMatrix", "SparseArray", "SparseFormat", "as_dense", "ps_block", "ps_hstack", "ps_vstack"]
