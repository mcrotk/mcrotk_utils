"""Type aliases for sparse and possibly-sparse matrices, and for scipy's sparse formats (private).

Public through ``mcrotk_utils.sparse_utils``.
"""

from typing import Any, Literal

import numpy.typing as npt
from scipy.sparse import bsr_array, coo_array, csc_array, csr_array, dia_array, dok_array, lil_array

# A scipy sparse array in any format. Spelled out as the union of scipy's sparse array classes because scipy's
# type stubs (scipy-stubs) make the common base class, sparray, a bare mixin without shape, dtype or toarray, so
# values typed with it couldn't use them. scipy's older sparse matrices (spmatrix) are left out: they follow
# np.matrix's rules (e.g. * multiplies matrices), not those of arrays
type SparseArray = bsr_array | coo_array | csc_array | csr_array | dia_array | dok_array | lil_array

# A dense NumPy array or a scipy sparse array ("possibly sparse")
type PsMatrix = npt.NDArray[Any] | SparseArray

# scipy's names for its sparse formats, as its functions' ``format`` arguments take them
type SparseFormat = Literal["bsr", "coo", "csc", "csr", "dia", "dok", "lil"]
