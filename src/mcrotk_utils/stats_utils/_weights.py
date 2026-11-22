"""Weighted means, normalized weights, and the HC1 inflation factor of a weighted mean's standard error (private).

Public through ``mcrotk_utils.stats_utils``. Needs only numpy.
"""

try:
    import numpy as np
    import numpy.typing as npt
except ModuleNotFoundError as error:
    # these are optional for the package as a whole, so say which extra provides them, not just that one is missing
    raise ModuleNotFoundError(
        "mcrotk_utils.stats_utils' weighted means need numpy: install mcrotk-utils with its extra, mcrotk-utils[numpy]",
        # the top-level package (e.g. scipy rather than scipy.stats), which is what's installed
        name=error.name.partition(".")[0] if error.name else None,
    ) from error

from mcrotk_utils.stats_utils._validation import _as_1d_floats


def normalize_weights(weights: npt.ArrayLike, normalization: float = 1) -> npt.NDArray[np.float64]:
    """Return ``weights`` scaled to add up to ``normalization``.

    E.g. ``[1, 1, 2]`` gives ``[0.25, 0.25, 0.5]``, or ``[1, 1, 2]`` with ``normalization=4``.

    Computed as ``weights * normalization / sum(weights)``, but without overflowing: the weights are first scaled
    by a power of 2, which is exact, so weights whose sum is too big for a float (e.g. ``[1e308, 1e308]``) still
    work, and integer weights are converted to floats first, so ``weights * normalization`` can't overflow either.
    Otherwise the result is the same, to the bit. Like any floating-point result, the normalized weights may add
    up to ``normalization`` only to within rounding (about 1e-16, relative).

    **Arguments.**

    - ``weights``: the weights, as a 1-D array or list (integers or floats; pandas Series and nullable columns
      work too, though the result is a NumPy array, without the index). They must be finite and non-negative, and
      can't all be 0.
    - ``normalization``: what the weights should add up to: a positive finite number. The default is 1.

    **Result.**

    - A 1-D NumPy float array, the same length as ``weights``.

    **Errors.**

    - ``ValueError``: ``weights`` isn't 1-D, is empty, has a NaN, infinite or negative weight, or adds up to 0;
      or ``normalization`` isn't positive and finite.
    """
    if not (np.isfinite(normalization) and normalization > 0):
        raise ValueError(f"normalization must be positive and finite, not {normalization}.")
    scaled = _scaled_weights(_checked_weights(weights))
    return scaled * normalization / np.sum(scaled)


def weighted_mean(values: npt.ArrayLike, weights: npt.ArrayLike | None = None, *, skip_na: bool = False) -> float:
    """Return the weighted mean of ``values``: ``sum(values * weights) / sum(weights)``.

    Like ``np.average``, but missing values are handled explicitly: by default a NaN (or a pandas ``NA``) raises,
    and with ``skip_na=True``, each value or weight that's missing is dropped together with its pair, so a skipped
    value's weight doesn't count either. E.g. values ``[1, NaN, 3]`` with weights ``[1, 1, 1]`` give 2 with
    ``skip_na=True``. Weights adding up to 0 raise, rather than giving NaN or ``ZeroDivisionError``.

    **Arguments.**

    - ``values``: the values, as a 1-D array or list (integers, floats or booleans; pandas Series and nullable
      columns work too). They may be infinite: then the mean is too, or NaN if there are both ``inf`` and ``-inf``.
    - ``weights``: a weight for each value, matched by position (not by a pandas index), as for ``values``. They
      must be finite and non-negative, and can't all be 0. The default, ``None``, weights the values equally.
    - ``skip_na``: whether to drop each value or weight that's NaN (or ``NA``), with its pair, rather than raise.

    **Result.**

    - The weighted mean, as a ``float``.

    **Errors.**

    - ``ValueError``:

      - ``values`` or ``weights`` isn't 1-D, or they differ in length;
      - a value or weight is NaN, unless ``skip_na`` is ``True``;
      - there are no values (left);
      - a weight is infinite or negative, or the weights add up to 0.
    """
    value_array, weight_array = _paired_values_and_weights(values, weights, skip_na)
    # scaling the weights by a power of 2 is exact, and keeps their sum from overflowing
    return float(np.average(value_array, weights=_scaled_weights(_checked_weights(weight_array))))


def hc1_inflation_factor(values: npt.ArrayLike, weights: npt.ArrayLike, *, skip_na: bool = False) -> float:
    """Return how much larger the HC1 (robust) standard error of a weighted mean is than the ordinary one.

    The ratio of the two standard errors of ``weighted_mean(values, weights)``: the heteroskedasticity-robust one
    (HC1) over the ordinary one, which treats the weights as precision weights (inverse variances). A factor above
    1 means the ordinary standard error understates the uncertainty, given how the weights and the values' spread
    vary together.

    It's the same as fitting an intercept-only weighted least-squares regression and dividing its HC1 standard
    error by its ordinary one (in statsmodels, ``smf.wls("y ~ 1", weights=w).fit()``, then
    ``HC1_se["Intercept"] / bse["Intercept"]``), but computed directly, without fitting a model:

    ``sqrt(n * sum(w**2 * e**2) / (sum(w) * sum(w * e**2)))``

    with ``e`` the residuals from the weighted mean and ``n`` the number of values. Every row counts in ``n``,
    including any with zero weight.

    With equal weights it's 1. If every residual has the same size, it's the square root of Kish's design effect,
    ``n * sum(w**2) / sum(w)**2``, so it can be read as that design effect, extended to a spread that varies.

    **Arguments.**

    - ``values``: the values, as a 1-D array or list (integers, floats or booleans; pandas Series and nullable
      columns work too). They must be finite.
    - ``weights``: a weight for each value, matched by position (not by a pandas index). They must be finite and
      non-negative, and can't all be 0.
    - ``skip_na``: whether to drop each value or weight that's NaN (or ``NA``), with its pair, rather than raise.

    **Result.**

    - The factor, as a ``float``.

    **Errors.**

    - ``ValueError``:

      - ``values`` or ``weights`` isn't 1-D, or they differ in length;
      - a value or weight is NaN, unless ``skip_na`` is ``True``;
      - a value is infinite; a weight is infinite or negative, or the weights add up to 0;
      - there are fewer than two values (left), or every value with a non-zero weight equals the weighted mean,
        so both standard errors are 0.
    """
    value_array, weight_array = _paired_values_and_weights(values, weights, skip_na)
    if not np.isfinite(value_array).all():
        raise ValueError("The values must be finite.")
    if len(value_array) < 2:
        raise ValueError("There must be at least two values, for a standard error.")
    # the factor doesn't change when the weights are scaled, so they're scaled by a power of 2 to avoid overflow
    scaled = _scaled_weights(_checked_weights(weight_array))
    residuals = value_array - np.average(value_array, weights=scaled)
    weighted_squares = scaled * residuals**2
    if not weighted_squares.any():
        raise ValueError("Every value with a non-zero weight equals the weighted mean, so both standard errors are 0.")
    ratio = len(value_array) * np.sum(scaled * weighted_squares) / (np.sum(scaled) * np.sum(weighted_squares))
    return float(np.sqrt(ratio))


def _paired_values_and_weights(
    values: npt.ArrayLike, weights: npt.ArrayLike | None, skip_na: bool
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return ``values`` and ``weights`` (equal ones if ``None``) as 1-D float arrays, without missing pairs.

    Raises ``ValueError`` if either isn't 1-D, they differ in length, a value or weight is missing and not
    ``skip_na``, or there are no values (left).
    """
    value_array = _as_1d_floats(values, "values")
    if weights is None:
        weight_array = np.ones(len(value_array))
    else:
        weight_array = _as_1d_floats(weights, "weights")
        if len(weight_array) != len(value_array):
            raise ValueError(f"values and weights differ in length: {len(value_array):,} and {len(weight_array):,}.")
    num_given = len(value_array)
    value_array, weight_array = _without_missing_pairs(value_array, weight_array, skip_na)
    if len(value_array) == 0:
        raise ValueError("There are no values" + (" without a NaN value or weight." if num_given else "."))
    return value_array, weight_array


def _without_missing_pairs(
    values: npt.NDArray[np.float64], weights: npt.NDArray[np.float64], skip_na: bool
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return ``values`` and ``weights`` without the pairs where either is NaN; raise if any is and not ``skip_na``."""
    missing = np.isnan(values) | np.isnan(weights)
    if missing.any():
        if not skip_na:
            raise ValueError(f"{int(missing.sum()):,} values or weights are NaN; pass skip_na=True to drop them.")
        return values[~missing], weights[~missing]
    return values, weights


def _checked_weights(weights: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return ``weights`` as a 1-D float array, raising ``ValueError`` unless they're usable as weights.

    That is: non-empty, finite, non-negative, and not all 0.
    """
    weight_array = _as_1d_floats(weights, "weights")
    if len(weight_array) == 0:
        raise ValueError("There are no weights.")
    if not np.isfinite(weight_array).all():
        # a NaN weight would make the result NaN, and an infinite one makes every other weight 0 and the sum infinite
        raise ValueError("The weights must be finite (not NaN or infinite).")
    if (weight_array < 0).any():
        # a negative weight makes a weighted mean no longer a mean (it can lie outside the values' range)
        raise ValueError("The weights must be non-negative.")
    if not weight_array.any():
        raise ValueError("The weights add up to 0.")
    return weight_array


def _scaled_weights(weights: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return ``weights`` scaled by a power of 2 so the largest is between 0.5 and 1 (exact, barring underflow).

    Multiplying by a power of 2 only changes a float's exponent, so ratios between the weights (and so normalized
    weights and weighted means) are unchanged, while their sum can no longer overflow. Weights too small relative
    to the largest underflow to 0, as they would once normalized anyway.
    """
    _, exponent = np.frexp(weights.max())
    return np.ldexp(weights, -exponent)
