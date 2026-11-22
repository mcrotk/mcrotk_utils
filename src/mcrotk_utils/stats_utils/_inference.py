"""Hypothesis types, p-values and confidence intervals, from empirical distributions or the t distribution (private).

Public through ``mcrotk_utils.stats_utils``. Needs scipy (and numpy, which scipy brings).
"""

from collections.abc import Sequence
from enum import Enum
from typing import Any, Literal, SupportsFloat, overload

try:
    import numpy as np
    import numpy.typing as npt
    from scipy.stats import t as student_t
except ModuleNotFoundError as error:
    # these are optional for the package as a whole, so say which extra provides them, not just that one is missing
    raise ModuleNotFoundError(
        "mcrotk_utils.stats_utils' p-values and confidence intervals need scipy: "
        "install mcrotk-utils with its extra, mcrotk-utils[scipy]",
        # the top-level package (e.g. scipy rather than scipy.stats), which is what's installed
        name=error.name.partition(".")[0] if error.name else None,
    ) from error

from mcrotk_utils.misc_utils import remove_punctuation
from mcrotk_utils.stats_utils._validation import _as_1d_floats

# The significance level used when none is given: confidence intervals are 95% by default
DEFAULT_ALPHA = 0.05

# The methods np.quantile accepts for estimating a quantile from values (all available since numpy 1.22)
type QuantileMethod = Literal[
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
    "lower",
    "higher",
    "midpoint",
    "nearest",
]

# Words naming each one-sided direction. Each is accepted alone, after "one-sided" or "one-tailed", or before
# "-tailed" (e.g. "upper", "one-sided upper", "one-tailed upper", "upper-tailed"). "greater"/"less" are scipy's
# names and "larger"/"smaller" statsmodels'.
_UPPER_WORDS = ("upper", "right", "increase", "greater", "larger")
_LOWER_WORDS = ("lower", "left", "decrease", "less", "smaller")


def _one_sided_aliases(words: tuple[str, ...]) -> frozenset[str]:
    """Return the normalized names (see ``_normalized``) accepted for a one-sided direction named by ``words``."""
    return frozenset(
        alias for word in words for alias in (word, f"onesided{word}", f"onetailed{word}", f"{word}tailed")
    )


def _normalized(name: str) -> str:
    """Return ``name`` lowercased, without punctuation or whitespace, and with the digits 1 and 2 spelled out.

    So that, e.g., ``"Two-Sided"``, ``"2 sided"`` and ``"two_sided"`` all become ``"twosided"``.
    """
    stripped = "".join(remove_punctuation(name.lower()).split())
    return stripped.replace("1", "one").replace("2", "two")


class HypothesisType(Enum):
    """Which alternative hypothesis a test has: two-sided, or one-sided in the upper or lower tail.

    For a statistic such as a difference or a coefficient, the upper tail tests for an increase (the statistic is
    larger than under the null hypothesis), and the lower tail for a decrease.

    **Creating one.** ``HypothesisType(value)`` accepts:

    - a member, which is returned as it is;
    - a member's value: ``"two-sided"``, ``"one-sided upper"`` or ``"one-sided lower"``;
    - a name, in any case and ignoring punctuation and whitespace, with the digits 1 and 2 read as "one" and "two":

      - two-sided: ``"two-sided"`` or ``"two-tailed"`` (so also ``"2-sided"``, ``"Two Tailed"``, ...);
      - upper: ``"upper"``, ``"right"``, ``"increase"``, ``"greater"`` or ``"larger"``, each alone, after
        ``"one-sided"`` or ``"one-tailed"``, or before ``"-tailed"`` (so ``"one-sided increase"``,
        ``"1-tailed right"``, ``"upper-tailed"``, ...);
      - lower: likewise with ``"lower"``, ``"left"``, ``"decrease"``, ``"less"`` or ``"smaller"``.

    Anything else raises ``ValueError``, including ``None`` and other non-strings: functions that take a
    hypothesis type give it a default (usually ``TWO_SIDED``) rather than accepting ``None`` for one.
    """

    TWO_SIDED = "two-sided"
    ONE_SIDED_UPPER = "one-sided upper"
    ONE_SIDED_LOWER = "one-sided lower"

    @classmethod
    def _missing_(cls, value: object) -> "HypothesisType":
        """Return the member for an alternative name (see the class docstring), or raise ``ValueError``."""
        # Enum calls this only for values that aren't a member or a member's value
        if isinstance(value, str):
            normalized = _normalized(value)
            if normalized in ("twosided", "twotailed"):
                return cls.TWO_SIDED
            if normalized in _one_sided_aliases(_UPPER_WORDS):
                return cls.ONE_SIDED_UPPER
            if normalized in _one_sided_aliases(_LOWER_WORDS):
                return cls.ONE_SIDED_LOWER
        raise ValueError(
            f"Unrecognized hypothesis type {value!r}: expected two-sided, or one-sided upper or lower "
            "(see the HypothesisType docstring for the names accepted)."
        )

    @property
    def scipy_alternative(self) -> Literal["two-sided", "greater", "less"]:
        """The name scipy's tests use for this hypothesis, in their ``alternative`` argument.

        ``"two-sided"``, ``"greater"`` or ``"less"``.
        """
        match self:
            case HypothesisType.TWO_SIDED:
                return "two-sided"
            case HypothesisType.ONE_SIDED_UPPER:
                return "greater"
            case HypothesisType.ONE_SIDED_LOWER:
                return "less"

    @property
    def statsmodels_alternative(self) -> Literal["two-sided", "larger", "smaller"]:
        """The name statsmodels' tests use for this hypothesis, in their ``alternative`` argument.

        ``"two-sided"``, ``"larger"`` or ``"smaller"``.
        """
        match self:
            case HypothesisType.TWO_SIDED:
                return "two-sided"
            case HypothesisType.ONE_SIDED_UPPER:
                return "larger"
            case HypothesisType.ONE_SIDED_LOWER:
                return "smaller"

    @classmethod
    def get_array(cls, values: Any = "two-sided", desired_length: int | None = None) -> npt.NDArray[np.object_]:
        """Return an array of members, one per test, from one value for all tests or one value per test.

        For running several tests at once (e.g. one per coefficient), each with its own hypothesis type or all
        with the same one.

        **Arguments.**

        - ``values``: one value, or a list or array of values, each of them anything ``HypothesisType(value)``
          accepts (see the class docstring). Nested lists and arrays of more than one dimension are flattened.
          The default is two-sided.
        - ``desired_length``: how many tests there are, or ``None`` if not known.

        **Result.**

        - A 1-D NumPy array of members, with dtype ``object``.
        - One value (including the default, or a list of one value) is repeated ``desired_length``
          times, or once if ``desired_length`` is ``None``. With ``desired_length=0``, the array is empty.
        - Several values (or none, as in an empty list) give one member each, in order.

        **Errors.**

        - ``ValueError``: ``desired_length`` is negative; several values (or none) are given and their number
          isn't ``desired_length``; or a value isn't recognized.
        """
        # values is typed Any because it can be one value or nested lists or arrays of them, which no precise
        # type describes well
        if desired_length is not None and desired_length < 0:
            raise ValueError(f"desired_length must be non-negative, not {desired_length}.")
        # dtype=object keeps strings and members as they are (rather than making a fixed-width string array), and
        # makes a single value a 0-d array, which reshape then turns into one entry, as it flattens nested values
        flat = np.asarray(values, dtype=object).reshape(-1)
        if len(flat) == 1:
            return np.full(1 if desired_length is None else desired_length, cls(flat[0]), dtype=object)
        if desired_length is not None and len(flat) != desired_length:
            raise ValueError(
                f"Got {len(flat):,} hypothesis types; expected either 1 or desired_length, {desired_length:,}."
            )
        # dtype=object also gives an empty list an object array, not numpy's default float one
        return np.array([cls(value) for value in flat], dtype=object)


def p_value_from_empirical_dist(
    value: SupportsFloat,
    dist_data: npt.ArrayLike,
    hypothesis_type: HypothesisType | str = HypothesisType.TWO_SIDED,
    *,
    include_observed: bool = True,
    skip_na: bool = False,
) -> float:
    """Return the p-value of an observed statistic, taking ``dist_data`` as the statistic's null distribution.

    ``dist_data`` is an array of values making up the empirical distribution of the statistic under the null
    hypothesis, and ``value`` is the statistic observed. For example, in a permutation or randomization test,
    ``dist_data`` holds the statistic recomputed under random re-assignments.

    The p-value is the fraction of values at least as extreme as ``value``, so values equal to it (ties) count on
    both sides: for an upper alternative, the fraction of values ``>= value``; for a lower one, ``<= value``; for
    a two-sided one, twice the smaller of the two, capped at 1. E.g. with values ``[0, 1, 1, 2]`` and ``value=1``,
    ``include_observed=False`` gives 3/4 for an upper or a lower alternative, and 1 for a two-sided one.

    By default the observed value is included as one of the values, giving ``(k + 1) / (B + 1)`` for ``k`` of
    ``B`` values at least as extreme, rather than ``k / B``. When the values are random draws from the null
    distribution (as in a permutation test with random re-assignments), the observed statistic is, under the
    null hypothesis, just as likely as any of them, so this makes the test valid: it rejects at most a fraction
    ``alpha`` of the time at level ``alpha``, whatever the number of values and however many ties there are, and
    a p-value is never 0.

    **Arguments.**

    - ``value``: the observed statistic: a number, including a NumPy scalar.
    - ``dist_data``: the values making up the null distribution, as a 1-D array or list (integers or floats).
    - ``hypothesis_type``: the alternative hypothesis, as a ``HypothesisType`` or any name it accepts. Upper
      means the alternative is that the statistic is larger than under the null. The default is two-sided.
    - ``include_observed``: whether to include the observed value as one of the values (see above). Set it to
      ``False`` when ``dist_data`` already includes it, e.g. when it's the statistic under every possible assignment,
      the observed one among them; ``(k + 1) / (B + 1)`` would then count the observed one twice.
    - ``skip_na``: whether to drop NaN values (``B`` is then the number of values left), rather than raise.

    **Result.**

    - The p-value, as a ``float`` between 0 and 1 (above 0 when ``include_observed`` is ``True``).

    **Errors.**

    - ``ValueError``:

      - ``value`` is NaN;
      - a value in ``dist_data`` is NaN, unless ``skip_na`` is ``True``;
      - ``dist_data`` is empty (or, with ``skip_na``, has only NaN values) or isn't 1-D;
      - ``hypothesis_type`` isn't recognized.
    """
    hypothesis_type = HypothesisType(hypothesis_type)
    value = _observed_value(value)
    sorted_values = np.sort(_distribution_values(dist_data, skip_na))
    # int() because the counts are NumPy integers, which would make the p-value a NumPy float rather than a float
    at_or_above = int(_count_above(sorted_values, value, inclusive=True))
    at_or_below = int(_count_below(sorted_values, value, inclusive=True))
    upper = _fraction_at_least_as_extreme(at_or_above, len(sorted_values), include_observed)
    lower = _fraction_at_least_as_extreme(at_or_below, len(sorted_values), include_observed)
    match hypothesis_type:
        case HypothesisType.ONE_SIDED_UPPER:
            return upper
        case HypothesisType.ONE_SIDED_LOWER:
            return lower
        case HypothesisType.TWO_SIDED:
            # counting ties on both sides makes the two tails add up to more than 1, so doubling can pass 1
            return min(1.0, 2 * min(upper, lower))


def _observed_value(value: SupportsFloat) -> float:
    """Return ``value`` as a float, or raise ``ValueError`` if it's NaN.

    Every comparison with NaN is false, so a NaN would otherwise count as more extreme than no value, and get the
    smallest p-value possible.
    """
    value = float(value)
    if np.isnan(value):
        raise ValueError("The observed value is NaN.")
    return value


def _distribution_values(dist_data: npt.ArrayLike, skip_na: bool) -> npt.NDArray[np.float64]:
    """Return ``dist_data`` as a 1-D float array, without NaNs if ``skip_na``; raise ``ValueError`` otherwise.

    Raises if ``dist_data`` isn't 1-D, has NaN values and ``skip_na`` is false, or has no values (left).
    """
    values = _as_1d_floats(dist_data, "dist_data")
    is_nan = np.isnan(values)
    if is_nan.any():
        # NaN values would count as never extreme while still counting towards B (and np.quantile would return
        # NaN), so they're dropped or rejected
        if not skip_na:
            raise ValueError(f"dist_data has {int(is_nan.sum()):,} NaN values; pass skip_na=True to drop them.")
        values = values[~is_nan]
    if len(values) == 0:
        # a p-value or quantile from no values would mean nothing (and k / B would be 0 / 0)
        raise ValueError("dist_data has no values" + (" that aren't NaN." if is_nan.any() else "."))
    return values


def _count_below(
    sorted_values: npt.NDArray[np.float64], values: npt.ArrayLike, inclusive: bool
) -> npt.NDArray[np.intp]:
    """Return, for each value, how many of ``sorted_values`` are below it (or at or below it, if ``inclusive``)."""
    # in sorted values, the position where a value would go first (left) or last (right) is how many are below it
    # (left) or at or below it (right)
    return np.searchsorted(sorted_values, values, side="right" if inclusive else "left")


def _count_above(
    sorted_values: npt.NDArray[np.float64], values: npt.ArrayLike, inclusive: bool
) -> npt.NDArray[np.intp]:
    """Return, for each value, how many of ``sorted_values`` are above it (or at or above it, if ``inclusive``)."""
    # the values above a value (or at or above it) are those not at or below it (or below it); an integer
    # subtraction, so exact
    return len(sorted_values) - _count_below(sorted_values, values, not inclusive)


def _fraction_at_least_as_extreme(count: int, num_values: int, include_observed: bool) -> float:
    """Return ``count / num_values``, or ``(count + 1) / (num_values + 1)`` when the observed value is included."""
    extra = 1 if include_observed else 0
    return (count + extra) / (num_values + extra)


def ci_from_empirical_dist(
    dist_data: npt.ArrayLike,
    alpha: float = DEFAULT_ALPHA,
    hypothesis_type: HypothesisType | str = HypothesisType.TWO_SIDED,
    *,
    bounds: tuple[float, float] | Literal["values"] = (-np.inf, np.inf),
    quantile_method: QuantileMethod = "inverted_cdf",
    skip_na: bool = False,
) -> npt.NDArray[np.float64]:
    """Return a confidence interval from the quantiles of an empirical distribution of an estimate.

    ``dist_data`` is an array of values making up the empirical distribution of an estimate (e.g. bootstrap
    estimates), and the interval is the percentile interval: its ends are the distribution's quantiles. Each
    one-sided interval is open at one end, which ``bounds`` sets.

    - Two-sided: from the ``alpha / 2`` quantile to the ``1 - alpha / 2`` quantile.
    - One-sided upper (the alternative that the quantity is larger): from the ``alpha`` quantile to the upper
      bound.
    - One-sided lower: from the lower bound to the ``1 - alpha`` quantile.

    E.g. with the values 1 to 100 and ``alpha=0.05``, the two-sided interval is ``[3, 98]``, and the one-sided
    upper one ``[5, inf]``; with ``bounds="values"``, ``[5, 100]``.

    **Arguments.**

    - ``dist_data``: the values making up the distribution, as a 1-D array or list (integers or floats).
    - ``alpha``: one minus the confidence level, between 0 and 1: 0.05 (the default) for a 95% interval.
    - ``hypothesis_type``: which interval, as a ``HypothesisType`` or any name it accepts. The default is
      two-sided.
    - ``bounds``: where a one-sided interval's open end goes; it doesn't affect a two-sided interval.

      - ``(low, high)``: the lower bound is ``low`` and the upper one ``high``. The default, ``(-inf, inf)``,
        leaves the open end unbounded. Finite bounds are for quantities with natural limits, e.g. ``(0, 1)`` for a
        probability; every value must then lie within them.
      - ``"values"``: the smallest or largest value, taking the values as the whole distribution, so that it
        ends there.

    - ``quantile_method``: how ``np.quantile`` estimates a quantile from the values (its ``method``
      argument; named for what it does, since "method" alone could mean the interval's method, e.g. percentile or
      BCa). The default, ``"inverted_cdf"``, gives one of the values: the smallest one with at least the given
      fraction of the values at or below it, i.e. the inverse of ``empirical_cdf``.
    - ``skip_na``: whether to drop NaN values, rather than raise.

    **Result.**

    - A NumPy array of two floats: the interval's lower and upper ends.

    **Errors.**

    - ``ValueError``:

      - ``alpha`` isn't between 0 and 1;
      - a value in ``dist_data`` is NaN, unless ``skip_na`` is ``True``;
      - ``dist_data`` is empty (or, with ``skip_na``, has only NaN values) or isn't 1-D;
      - ``bounds`` isn't ``"values"`` or a pair ``(low, high)`` with ``low <= high``, or a value lies outside it;
      - ``hypothesis_type`` isn't recognized.
    """
    hypothesis_type = HypothesisType(hypothesis_type)
    _check_probability(alpha, "alpha")
    values = _distribution_values(dist_data, skip_na)
    lower_bound, upper_bound = _interval_bounds(values, bounds)
    tail = alpha / 2 if hypothesis_type is HypothesisType.TWO_SIDED else alpha
    # np.quantile picks the same value for 1 - tail as exact arithmetic would (checked for common alphas and up
    # to 2,000 values), so the subtraction's round-off doesn't matter here
    lower = (
        lower_bound
        if hypothesis_type is HypothesisType.ONE_SIDED_LOWER
        else float(np.quantile(values, tail, method=quantile_method))
    )
    upper = (
        upper_bound
        if hypothesis_type is HypothesisType.ONE_SIDED_UPPER
        else float(np.quantile(values, 1 - tail, method=quantile_method))
    )
    return np.array([lower, upper])


def _check_probability(number: npt.ArrayLike, name: str) -> None:
    """Raise ``ValueError`` unless ``number`` (or each number in it) is between 0 and 1 (inclusive)."""
    numbers = np.asarray(number, dtype=np.float64)
    # written so that NaN fails too
    if not np.all((numbers >= 0) & (numbers <= 1)):
        raise ValueError(f"{name} must be between 0 and 1, not {number!r}.")


def _interval_bounds(
    values: npt.NDArray[np.float64], bounds: tuple[float, float] | Literal["values"]
) -> tuple[float, float]:
    """Return the lower and upper bounds for one-sided intervals, checking that ``values`` lie within them."""
    if isinstance(bounds, str):
        if bounds != "values":
            raise ValueError(f'bounds must be "values" or a pair (low, high), not {bounds!r}.')
        return float(values.min()), float(values.max())
    low, high = (float(bound) for bound in bounds)
    # written so that NaN bounds fail too
    if not low <= high:
        raise ValueError(f"bounds must be a pair (low, high) with low <= high, not {bounds!r}.")
    outside = (values < low) | (values > high)
    if outside.any():
        # values outside natural limits mean either the values or the limits are wrong
        raise ValueError(f"{int(outside.sum()):,} values lie outside bounds {bounds!r}, e.g. {values[outside][0]}.")
    return low, high


@overload
def empirical_cdf(
    value: float | np.number[Any], dist_data: npt.ArrayLike, *, inclusive: bool = True, skip_na: bool = False
) -> float: ...
@overload
def empirical_cdf(
    value: npt.NDArray[Any] | Sequence[float],
    dist_data: npt.ArrayLike,
    *,
    inclusive: bool = True,
    skip_na: bool = False,
) -> npt.NDArray[np.float64]: ...
def empirical_cdf(
    value: float | np.number[Any] | npt.NDArray[Any] | Sequence[float],
    dist_data: npt.ArrayLike,
    *,
    inclusive: bool = True,
    skip_na: bool = False,
) -> float | npt.NDArray[np.float64]:
    """Return the fraction of ``dist_data`` at or below ``value`` (below it, if not ``inclusive``).

    The empirical distribution function of ``dist_data``, at ``value``: by default ``P(X <= value)``, the usual
    definition of a CDF, and with ``inclusive=False``, ``P(X < value)``. E.g. with ``dist_data`` ``[1, 2, 2, 3]``, the
    CDF at 2 is 3/4, or 1/4 with ``inclusive=False``.

    **Arguments.**

    - ``value``: where to evaluate it: a number (including a NumPy scalar), or a list or array of numbers.
    - ``dist_data``: the values making up the distribution, as a 1-D array or list (integers or floats).
    - ``inclusive``: whether values equal to ``value`` count. The default is ``True``.
    - ``skip_na``: whether to drop NaN values from ``dist_data`` (the fraction is then of the values left), rather
      than raise.

    **Result.**

    - For a number ``value``, a ``float`` between 0 and 1.
    - For a list or array, a NumPy float array of the same shape, with the fraction for each value.

    **Errors.**

    - ``ValueError``: ``value`` is (or has) a NaN; a value in ``dist_data`` is NaN, unless ``skip_na`` is
      ``True``; ``dist_data`` is empty (or, with ``skip_na``, has only NaN values) or isn't 1-D.
    """
    sorted_values = np.sort(_distribution_values(dist_data, skip_na))
    counts = _count_below(sorted_values, _values_to_evaluate(value), inclusive)
    return _fractions(counts, len(sorted_values), scalar=np.ndim(value) == 0)


@overload
def empirical_sf(
    value: float | np.number[Any], dist_data: npt.ArrayLike, *, inclusive: bool = False, skip_na: bool = False
) -> float: ...
@overload
def empirical_sf(
    value: npt.NDArray[Any] | Sequence[float],
    dist_data: npt.ArrayLike,
    *,
    inclusive: bool = False,
    skip_na: bool = False,
) -> npt.NDArray[np.float64]: ...
def empirical_sf(
    value: float | np.number[Any] | npt.NDArray[Any] | Sequence[float],
    dist_data: npt.ArrayLike,
    *,
    inclusive: bool = False,
    skip_na: bool = False,
) -> float | npt.NDArray[np.float64]:
    """Return the fraction of ``dist_data`` above ``value`` (at or above it, if ``inclusive``).

    The empirical survival function of ``dist_data``, at ``value``: by default ``P(X > value)``, the usual definition,
    so that with both defaults ``empirical_cdf + empirical_sf`` is 1; with ``inclusive=True``, ``P(X >= value)``.
    E.g. with ``dist_data`` ``[1, 2, 2, 3]``, the survival function at 2 is 1/4, or 3/4 with ``inclusive=True``.

    The values are counted directly, not computed as 1 minus the CDF, which would add round-off: e.g.
    ``1 - 19/20`` is 0.050000000000000044, where counting gives 0.05.

    **Arguments.**

    - ``value``: where to evaluate it: a number (including a NumPy scalar), or a list or array of numbers.
    - ``dist_data``: the values making up the distribution, as a 1-D array or list (integers or floats).
    - ``inclusive``: whether values equal to ``value`` count. The default is ``False``.
    - ``skip_na``: whether to drop NaN values from ``dist_data`` (the fraction is then of the values left), rather
      than raise.

    **Result.**

    - For a number ``value``, a ``float`` between 0 and 1.
    - For a list or array, a NumPy float array of the same shape, with the fraction for each value.

    **Errors.**

    - ``ValueError``: ``value`` is (or has) a NaN; a value in ``dist_data`` is NaN, unless ``skip_na`` is
      ``True``; ``dist_data`` is empty (or, with ``skip_na``, has only NaN values) or isn't 1-D.
    """
    sorted_values = np.sort(_distribution_values(dist_data, skip_na))
    counts = _count_above(sorted_values, _values_to_evaluate(value), inclusive)
    return _fractions(counts, len(sorted_values), scalar=np.ndim(value) == 0)


def _values_to_evaluate(value: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return ``value`` as a float array (0-d for a number), or raise ``ValueError`` if it has a NaN."""
    values = np.asarray(value, dtype=np.float64)
    # every comparison with NaN is false, so a NaN would get a fraction that means nothing
    if np.isnan(values).any():
        raise ValueError("A value to evaluate at is NaN.")
    return values


def _fractions(counts: npt.NDArray[np.intp], num_values: int, scalar: bool) -> float | npt.NDArray[np.float64]:
    """Return ``counts / num_values``: a ``float`` if ``scalar``, otherwise a float array."""
    fractions = np.asarray(counts / num_values, dtype=np.float64)
    return float(fractions) if scalar else fractions


# --- The t distribution -------------------------------------------------------------------------------------------

# One hypothesis type for every test, or one per test (anything HypothesisType.get_array accepts)
type HypothesisTypes = HypothesisType | str | Sequence[HypothesisType | str] | npt.NDArray[Any]


@overload
def p_value_from_t_stat(
    t_stat: float | np.number[Any], df: float | np.number[Any], hypothesis_type: HypothesisType | str = ...
) -> float: ...
@overload
def p_value_from_t_stat(
    t_stat: npt.NDArray[Any] | Sequence[float], df: npt.ArrayLike, hypothesis_type: HypothesisTypes = ...
) -> npt.NDArray[np.float64]: ...
@overload
def p_value_from_t_stat(
    t_stat: float | np.number[Any], df: npt.NDArray[Any] | Sequence[float], hypothesis_type: HypothesisTypes = ...
) -> npt.NDArray[np.float64]: ...
def p_value_from_t_stat(
    t_stat: npt.ArrayLike, df: npt.ArrayLike, hypothesis_type: HypothesisTypes = HypothesisType.TWO_SIDED
) -> float | npt.NDArray[np.float64]:
    """Return the p-value of a t-statistic with ``df`` degrees of freedom, or of each of several.

    For an upper alternative, the probability of a t-statistic at least as large; for a lower one, at least as
    small; for a two-sided one, twice the probability beyond ``abs(t_stat)``. Computed from the survival function
    (``scipy.stats.t.sf``), never as 1 minus the CDF, so small p-values keep their precision (e.g. about 6.9e-28 for
    ``t_stat=40`` with 30 degrees of freedom, where ``1 - cdf`` would round to 0). With infinite degrees of freedom
    it's the normal distribution's p-value (a z-test).

    **Arguments.**

    - ``t_stat``: a t-statistic, or a list or array of them (one test each).
    - ``df``: the degrees of freedom: a positive number (``np.inf`` allowed), or an array of them, paired with the
      t-statistics by numpy's broadcasting (e.g. one per t-statistic).
    - ``hypothesis_type``: the alternative, as a ``HypothesisType`` or any name it accepts: one, applied to every
      test, or a list or array with one per test. The default is two-sided.

    **Result.**

    - A ``float`` when ``t_stat`` and ``df`` are single numbers; otherwise a NumPy float array of their broadcast
      shape, one p-value per test.

    **Errors.**

    - ``ValueError``: a t-statistic is NaN; a degrees of freedom is NaN or not positive; ``t_stat`` and ``df``
      don't broadcast together; or ``hypothesis_type`` isn't recognized, or is a list whose length isn't the number
      of tests.
    """
    t_stats, dfs = _t_stats_and_dfs(t_stat, df)
    # one member per test, in the tests' shape (get_array repeats a single value, or checks the number given)
    types = HypothesisType.get_array(hypothesis_type, desired_length=t_stats.size).reshape(t_stats.shape)
    two_sided, lower = types == HypothesisType.TWO_SIDED, types == HypothesisType.ONE_SIDED_LOWER
    # each test's tail: beyond |t| (doubled) for two-sided, below t for lower (the upper tail beyond -t, by symmetry)
    tail_start = np.where(two_sided, np.abs(t_stats), np.where(lower, -t_stats, t_stats))
    p_values = np.where(two_sided, 2.0, 1.0) * student_t.sf(tail_start, dfs)
    return float(p_values) if p_values.ndim == 0 else np.asarray(p_values, dtype=np.float64)


def _t_stats_and_dfs(
    t_stat: npt.ArrayLike, df: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the t-statistics and degrees of freedom as float arrays of their broadcast shape, checked.

    Raises ``ValueError`` for a NaN t-statistic, degrees of freedom that are NaN or not positive, or shapes that
    don't broadcast.
    """
    t_stats, dfs = np.asarray(t_stat, dtype=np.float64), _checked_dfs(df)
    if np.isnan(t_stats).any():
        raise ValueError("A t-statistic is NaN.")
    try:
        broadcast_t, broadcast_df = np.broadcast_arrays(t_stats, dfs)
    except ValueError as error:
        raise ValueError(f"t_stat and df don't match in shape: {t_stats.shape} and {dfs.shape}.") from error
    return broadcast_t, broadcast_df


def negate_ci(ci: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return the confidence interval for the negated quantity: ``[-upper, -lower]`` for an interval ``[lower, upper]``.

    E.g. to turn an interval for ``A - B`` into one for ``B - A``. One-sided intervals keep their infinite end on
    the other side: ``[1, inf]`` becomes ``[-inf, -1]``.

    **Arguments.**

    - ``ci``: one interval ``[lower, upper]``, or several, with each interval along the last axis (e.g. the
      ``(n, 2)`` rows from ``ci_from_t_dist`` or ``ci_from_empirical_dist``).

    **Result.**

    - A NumPy float array of the same shape, each interval negated.

    **Errors.**

    - ``ValueError``: the last axis doesn't have length 2, an end is NaN, or an interval's lower end is above its
      upper end (so it isn't an interval).
    """
    intervals = np.asarray(ci, dtype=np.float64)
    if intervals.ndim == 0 or intervals.shape[-1] != 2:
        raise ValueError(f"ci must have [lower, upper] along its last axis, not shape {intervals.shape}.")
    if np.isnan(intervals).any():
        raise ValueError("ci has a NaN end.")
    if (intervals[..., 0] > intervals[..., 1]).any():
        raise ValueError("ci has an interval whose lower end is above its upper end.")
    # swap each interval's two ends (the last axis), then negate
    return -intervals[..., ::-1]


def _checked_dfs(df: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return the degrees of freedom as a float array, raising ``ValueError`` unless each is positive (or inf)."""
    dfs = np.asarray(df, dtype=np.float64)
    # written so that NaN degrees of freedom fail too
    if not np.all(dfs > 0):
        raise ValueError("The degrees of freedom must be positive (np.inf is allowed).")
    return dfs


def ci_from_t_dist(
    estimate: npt.ArrayLike,
    se: npt.ArrayLike,
    df: npt.ArrayLike,
    alpha: npt.ArrayLike = DEFAULT_ALPHA,
    hypothesis_type: HypothesisTypes = HypothesisType.TWO_SIDED,
) -> npt.NDArray[np.float64]:
    """Return the t-based confidence interval ``estimate ± t * se``, or one for each of several estimates.

    ``t`` is the t distribution's quantile with ``df`` degrees of freedom: for a two-sided interval, the one with
    ``alpha / 2`` above it, so the interval is ``[estimate - t * se, estimate + t * se]``; for a one-sided interval,
    the one with ``alpha`` above it, and the interval is open at one end: ``[estimate - t * se, inf]`` for an upper
    alternative (the quantity is larger), ``[-inf, estimate + t * se]`` for a lower one. Each interval contains
    exactly the values the corresponding t-test (``p_value_from_t_stat``) doesn't reject at level ``alpha``.

    The quantile comes from the survival function (``scipy.stats.t.isf(tail)``, not ``ppf(1 - tail)``), so a
    small ``alpha`` keeps its precision, and an infinite end is exactly infinite. With infinite degrees of freedom
    it's the normal distribution's interval (a z-interval).

    **Arguments.**

    - ``estimate``: the point estimate, or a list or array of them (one interval each).
    - ``se``: the estimate's standard error (the standard deviation of the estimate, not of the data), at least 0.
    - ``df``: the degrees of freedom: positive (``np.inf`` allowed).
    - ``alpha``: one minus the confidence level, between 0 and 1: 0.05 (the default) for 95% intervals.
    - ``hypothesis_type``: which interval, as a ``HypothesisType`` or any name it accepts: one, applied to every
      interval, or a list or array with one per interval. The default is two-sided.

    ``estimate``, ``se``, ``df`` and ``alpha`` can each be one number or one per interval: they're paired by
    numpy's broadcasting.

    **Result.**

    - A NumPy float array whose last axis is ``[lower, upper]``: shape ``(2,)`` for one interval, ``(n, 2)`` for
      ``n`` of them (one row each), and in general the inputs' broadcast shape followed by 2.

    **Errors.**

    - ``ValueError``: an estimate or standard error isn't finite, or a standard error is negative; a degrees of
      freedom is NaN or not positive; an ``alpha`` isn't between 0 and 1; the inputs don't broadcast together; or
      ``hypothesis_type`` isn't recognized, or is a list whose length isn't the number of intervals.
    """
    _check_probability(alpha, "alpha")
    estimates, ses, dfs, alphas = _interval_inputs(estimate, se, df, alpha)
    types = HypothesisType.get_array(hypothesis_type, desired_length=estimates.size).reshape(estimates.shape)
    tail = np.where(types == HypothesisType.TWO_SIDED, alphas / 2, alphas)
    # se * t, but 0 whenever se is 0 (where t can be infinite, for alpha 0, and 0 * inf would be NaN)
    half_width = np.multiply(student_t.isf(tail, dfs), ses, out=np.zeros_like(ses), where=ses > 0)
    lower = np.where(types == HypothesisType.ONE_SIDED_LOWER, -np.inf, estimates - half_width)
    upper = np.where(types == HypothesisType.ONE_SIDED_UPPER, np.inf, estimates + half_width)
    return np.stack([lower, upper], axis=-1)


def _interval_inputs(
    estimate: npt.ArrayLike, se: npt.ArrayLike, df: npt.ArrayLike, alpha: npt.ArrayLike
) -> list[npt.NDArray[np.float64]]:
    """Return the estimates, standard errors, degrees of freedom and alphas as float arrays of one broadcast shape.

    Raises ``ValueError`` for a non-finite estimate or standard error, a negative standard error, bad degrees of
    freedom, or shapes that don't broadcast.
    """
    estimates, ses = np.asarray(estimate, dtype=np.float64), np.asarray(se, dtype=np.float64)
    if not (np.isfinite(estimates).all() and np.isfinite(ses).all()):
        raise ValueError("The estimates and standard errors must be finite (not NaN or infinite).")
    if (ses < 0).any():
        raise ValueError("The standard errors must be non-negative.")
    arrays = [estimates, ses, _checked_dfs(df), np.asarray(alpha, dtype=np.float64)]
    try:
        return list(np.broadcast_arrays(*arrays))
    except ValueError as error:
        shapes = ", ".join(str(array.shape) for array in arrays)
        raise ValueError(f"estimate, se, df and alpha don't match in shape: {shapes}.") from error
