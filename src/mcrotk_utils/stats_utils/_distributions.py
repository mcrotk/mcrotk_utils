"""Parameters of normal (and lognormal) distributions, which can generate values (private).

Public through ``mcrotk_utils.stats_utils``. Needs only numpy.
"""

from dataclasses import dataclass
from typing import overload

try:
    import numpy as np
    import numpy.typing as npt
except ModuleNotFoundError as error:
    # these are optional for the package as a whole, so say which extra provides them, not just that one is missing
    raise ModuleNotFoundError(
        "mcrotk_utils.stats_utils' NormalParams needs numpy: install mcrotk-utils with its extra, mcrotk-utils[numpy]",
        # the top-level package (e.g. scipy rather than scipy.stats), which is what's installed
        name=error.name.partition(".")[0] if error.name else None,
    ) from error

from mcrotk_utils.numpy_utils import SeedLike


@dataclass(frozen=True)
class NormalParams:
    """A normal distribution's mean and standard deviation, checked, and able to generate values.

    The same pair also defines a lognormal distribution: that of values whose logarithm is normal with this mean and
    standard deviation (so ``mean`` and ``sd`` are then the logarithm's, not the values'). ``generate_normal`` and
    ``generate_lognormal`` draw from the two.

    Immutable (and so hashable). E.g. ``NormalParams(mean=-2.5, sd=0.1).generate_lognormal(10, rng=seed)``.

    **Arguments.**

    - ``mean``: the mean, a finite number. The default is 0.
    - ``sd``: the standard deviation, a finite number of at least 0 (0 gives a distribution with no randomness,
      e.g. to switch a random component off). The default is 1.

    **Errors.**

    - ``ValueError``: ``mean`` or ``sd`` isn't finite, or ``sd`` is negative.
    """

    mean: float = 0.0
    sd: float = 1.0

    def __post_init__(self) -> None:
        """Check the parameters (a frozen dataclass can't fix them afterwards, so they must be right from the start)."""
        if not (np.isfinite(self.mean) and np.isfinite(self.sd)):
            raise ValueError(f"mean and sd must be finite, not {self.mean} and {self.sd}.")
        if self.sd < 0:
            raise ValueError(f"sd must be non-negative, not {self.sd}.")

    @overload
    def generate_normal(self, size: None = None, *, rng: SeedLike = None) -> float: ...
    @overload
    def generate_normal(self, size: int | tuple[int, ...], *, rng: SeedLike = None) -> npt.NDArray[np.float64]: ...
    def generate_normal(
        self, size: int | tuple[int, ...] | None = None, *, rng: SeedLike = None
    ) -> float | npt.NDArray[np.float64]:
        """Return values drawn from the normal distribution with this mean and standard deviation.

        **Arguments.**

        - ``size``: how many values: ``None`` (the default) for a single one, an int for a 1-D array, or a tuple
          for an array of that shape.
        - ``rng``: a seed or ``np.random.Generator``, so that the values can be reproduced (a Generator continues
          its stream).

        **Result.**

        - A ``float`` for ``size=None``; otherwise a NumPy float array of that size. The same values as
          ``np.random.default_rng(rng).normal(mean, sd, size)``.
        """
        return _float_if_scalar(np.random.default_rng(rng).normal(self.mean, self.sd, size), size)

    @overload
    def generate_lognormal(self, size: None = None, *, rng: SeedLike = None) -> float: ...
    @overload
    def generate_lognormal(self, size: int | tuple[int, ...], *, rng: SeedLike = None) -> npt.NDArray[np.float64]: ...
    def generate_lognormal(
        self, size: int | tuple[int, ...] | None = None, *, rng: SeedLike = None
    ) -> float | npt.NDArray[np.float64]:
        """Return values drawn from the lognormal distribution whose logarithm has this mean and standard deviation.

        **Arguments.**

        - ``size``: how many values: ``None`` (the default) for a single one, an int for a 1-D array, or a tuple
          for an array of that shape.
        - ``rng``: a seed or ``np.random.Generator``, so that the values can be reproduced (a Generator continues
          its stream).

        **Result.**

        - A positive ``float`` for ``size=None``; otherwise a NumPy float array of that size. The same values as
          ``np.random.default_rng(rng).lognormal(mean, sd, size)``, which are ``exp`` of ``generate_normal``'s
          with the same ``rng``.
        """
        return _float_if_scalar(np.random.default_rng(rng).lognormal(self.mean, self.sd, size), size)


def _float_if_scalar(
    draws: float | npt.NDArray[np.float64], size: int | tuple[int, ...] | None
) -> float | npt.NDArray[np.float64]:
    """Return ``draws`` as a ``float`` when no ``size`` was asked for (numpy gives a NumPy float), otherwise as is."""
    return float(draws) if size is None else np.asarray(draws, dtype=np.float64)
