"""Input checks shared by the ``stats_utils`` modules (private)."""

try:
    import numpy as np
    import numpy.typing as npt
except ModuleNotFoundError as error:
    # these are optional for the package as a whole, so say which extra provides them, not just that one is missing
    raise ModuleNotFoundError(
        "mcrotk_utils.stats_utils needs numpy: install mcrotk-utils with its extra, mcrotk-utils[numpy]",
        # the top-level package (e.g. scipy rather than scipy.stats), which is what's installed
        name=error.name.partition(".")[0] if error.name else None,
    ) from error


def _as_1d_floats(array: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    """Return ``array`` as a 1-D float array (pandas ``NA`` becoming NaN), or raise ``ValueError`` if it isn't 1-D."""
    # pandas' nullable columns with missing values convert to floats only through to_numpy with a value for NA
    # (np.asarray can do it since pandas 3, but raises in pandas 2)
    to_numpy = getattr(array, "to_numpy", None)
    floats = to_numpy(dtype=np.float64, na_value=np.nan) if to_numpy else np.asarray(array, dtype=np.float64)
    if floats.ndim != 1:
        raise ValueError(f"{name} must be 1-D, but has {floats.ndim} dimensions.")
    return floats
