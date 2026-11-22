"""Type aliases for arrays of integers or floats, and for random seeds (private).

Public through ``mcrotk_utils.numpy_utils``.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

# Arrays of integers or floats of any width (np.integer rather than np.int_, so that int32 arrays match too)
type NDArrayInt = npt.NDArray[np.integer]
type NDArrayFloat = npt.NDArray[np.floating]

# What np.random.default_rng accepts as a seed, for functions that take an ``rng`` argument and pass it on to
# default_rng. Leaves out the legacy np.random.RandomState.
type SeedLike = (
    int | Sequence[int] | NDArrayInt | np.random.SeedSequence | np.random.BitGenerator | np.random.Generator | None
)
