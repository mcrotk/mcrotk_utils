"""Converting a possibly-sparse matrix to a dense array (private).

Public through ``mcrotk_utils.sparse_utils``.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.sparse import issparse

from mcrotk_utils.sparse_utils._typing import SparseArray


def as_dense(array: npt.ArrayLike | SparseArray) -> npt.NDArray[Any]:
    """Return ``array`` as a dense NumPy array, whether it's a scipy sparse array or not.

    Neither NumPy nor scipy has one function for this: a NumPy array has no ``toarray`` method, and
    ``np.asarray`` given a sparse array doesn't raise an error but returns a 0-d array of objects holding it.

    **Arguments.**

    - ``array``: a scipy sparse array in any format, or anything ``np.asarray`` accepts (a NumPy array, a nested
      list, ...).

    **Result.**

    - For a sparse array, a new dense array with the same shape, dtype and values.
    - Otherwise, what ``np.asarray`` gives: a NumPy array is returned as it is (the same object, not a copy).
    """
    if issparse(array):
        return array.toarray()
    return np.asarray(array)
