"""Tests for ``mcrotk_utils.stats_utils``' weighted means, normalized weights and HC1 inflation factor.

Besides fixed cases, some tests compare with independent implementations: numpy, scipy and statsmodels where
they compute the same thing, and otherwise straightforward reference implementations written here.
"""

import numpy as np
import pandas as pd
import pytest

from mcrotk_utils.stats_utils import (
    HypothesisType,
    hc1_inflation_factor,
    normalize_weights,
    weighted_mean,
)

UPPER, LOWER, TWO_SIDED = HypothesisType.ONE_SIDED_UPPER, HypothesisType.ONE_SIDED_LOWER, HypothesisType.TWO_SIDED


# --- normalize_weights -----------------------------------------------------------------------------------------


def plain_normalization(weights, normalization=1):
    """The formula ``normalize_weights`` computes, applied directly (without its scaling against overflow)."""
    return weights * normalization / np.sum(weights)


def random_weights(seed):
    """Yield random non-negative weights: uniform, over many orders of magnitude, and small integers."""
    rng = np.random.default_rng(seed)
    for _ in range(200):
        size = rng.integers(1, 100)
        yield rng.uniform(0, 1, size)
        yield 10 ** rng.uniform(-8, 8, size)
        yield rng.integers(0, 10, size) + np.eye(1, size, 0, dtype=int)[0]  # at least one non-zero


def test_normalize_weights_docstring_examples():
    """[1, 1, 2] gives [0.25, 0.25, 0.5], or [1, 1, 2] when normalizing to 4."""
    assert list(normalize_weights([1, 1, 2])) == [0.25, 0.25, 0.5]
    assert list(normalize_weights([1, 1, 2], 4)) == [1, 1, 2]


@pytest.mark.parametrize("normalization", [1, 4, 0.3, 1000.0])
def test_normalize_weights_matches_the_plain_formula_to_the_bit(normalization):
    """Where the plain formula doesn't overflow, the result is the same, bit for bit (scaling by 2**k is exact)."""
    for weights in random_weights(9):
        new = normalize_weights(weights, normalization)
        assert np.array_equal(new, plain_normalization(weights.astype(float), normalization))


def test_normalize_weights_adds_up_to_the_normalization_within_rounding():
    """The normalized weights add up to the normalization, to within about 1e-16 relative."""
    for weights in random_weights(10):
        for normalization in (1, 7.5):
            assert abs(np.sum(normalize_weights(weights, normalization)) - normalization) <= 1e-15 * normalization


def test_normalize_weights_doesnt_overflow():
    """Weights whose sum overflows, and large integer weights, work (the plain formula gives wrong values)."""
    huge = np.array([1e308, 1e308])
    assert list(normalize_weights(huge)) == [0.5, 0.5]
    with np.errstate(over="ignore"):
        assert list(plain_normalization(huge)) == [0, 0]  # the sum overflows to inf
    large_ints = np.array([2**62, 2**62], dtype=np.int64)
    assert list(normalize_weights(large_ints, 4)) == [2, 2]
    assert list(plain_normalization(large_ints, 4)) != [2, 2]  # weights * 4 overflows int64
    # tiny (subnormal) weights are scaled up, exactly
    assert list(normalize_weights([5e-324, 5e-324])) == [0.5, 0.5]


def test_normalize_weights_result_and_inputs():
    """The result is a 1-D float array; lists, integers and pandas Series (index dropped) work; zeros are allowed."""
    result = normalize_weights(pd.Series([1, 3], index=["a", "b"]))
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert list(result) == [0.25, 0.75]
    assert list(normalize_weights(pd.Series([1, 0, 3], dtype="Int64"))) == [0.25, 0, 0.75]


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ([1.0, np.nan], "must be finite"),
        ([1.0, np.inf], "must be finite"),
        (pd.Series([1, None], dtype="Int64"), "must be finite"),
        ([1.0, -0.5], "non-negative"),
        ([0, 0], "add up to 0"),
        ([], "no weights"),
        ([[1.0, 2.0]], "must be 1-D"),
    ],
)
def test_normalize_weights_rejects_unusable_weights(weights, message):
    """NaN, infinite or negative weights, all zeros, none, or not 1-D, raise ValueError."""
    with pytest.raises(ValueError, match=message):
        normalize_weights(weights)


@pytest.mark.parametrize("normalization", [0, -1, np.inf, np.nan])
def test_normalize_weights_rejects_bad_normalizations(normalization):
    """The normalization must be positive and finite."""
    with pytest.raises(ValueError, match="normalization must be positive and finite"):
        normalize_weights([1.0, 2.0], normalization)


# --- weighted_mean -----------------------------------------------------------------------------------------------


def test_weighted_mean_matches_np_average():
    """On valid inputs, the result is np.average's (and np.mean's without weights)."""
    rng = np.random.default_rng(12)
    for weights in random_weights(13):
        values = rng.normal(scale=10, size=len(weights))
        assert weighted_mean(values, weights) == pytest.approx(np.average(values, weights=weights), rel=1e-14)
        assert weighted_mean(values) == pytest.approx(np.mean(values), rel=1e-14)


def test_weighted_mean_docstring_example_and_skipping_pairs():
    """With skip_na, a missing value is dropped with its weight: [1, NaN, 3] with equal weights gives 2."""
    assert weighted_mean([1.0, np.nan, 3.0], [1.0, 1.0, 1.0], skip_na=True) == 2
    # a missing weight drops its value
    assert weighted_mean([1.0, 100.0, 3.0], [1.0, np.nan, 1.0], skip_na=True) == 2
    # pandas nullable columns: the missing value's weight is dropped too (pandas' own sums would skip only the NA,
    # e.g. (s * w).sum() / w.sum() gives 4/3)
    values, weights = pd.Series([1, None, 3], dtype="Int64"), pd.Series([1.0, 1.0, 1.0])
    assert weighted_mean(values, weights, skip_na=True) == 2
    assert (values * weights).sum() / weights.sum() == pytest.approx(4 / 3)


@pytest.mark.parametrize(
    ("values", "weights"),
    [([1.0, np.nan], [1.0, 1.0]), ([1.0, 2.0], [1.0, np.nan]), (pd.Series([1, None], dtype="Int64"), None)],
)
def test_weighted_mean_rejects_missing_values_by_default(values, weights):
    """A NaN (or NA) value or weight raises ValueError unless skip_na."""
    with pytest.raises(ValueError, match="values or weights are NaN; pass skip_na=True"):
        weighted_mean(values, weights)


def test_weighted_mean_with_no_values_left_raises():
    """No values, or none left after skipping missing ones, raise ValueError."""
    with pytest.raises(ValueError, match=r"There are no values\.$"):
        weighted_mean([])
    with pytest.raises(ValueError, match="no values without a NaN value or weight"):
        weighted_mean([np.nan, 1.0], [1.0, np.nan], skip_na=True)


@pytest.mark.parametrize(
    ("weights", "message"),
    [([0.0, 0.0], "add up to 0"), ([1.0, -1.0], "non-negative"), ([1.0, np.inf], "must be finite")],
)
def test_weighted_mean_rejects_unusable_weights(weights, message):
    """Weights adding up to 0 (np.average: ZeroDivisionError), negative or infinite weights raise ValueError."""
    with pytest.raises(ValueError, match=message):
        weighted_mean([1.0, 2.0], weights)


def test_weighted_mean_rejects_mismatched_or_non_1d_inputs():
    """values and weights must be 1-D and the same length."""
    with pytest.raises(ValueError, match="differ in length: 2 and 3"):
        weighted_mean([1.0, 2.0], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="values must be 1-D"):
        weighted_mean([[1.0, 2.0]])
    with pytest.raises(ValueError, match="weights must be 1-D"):
        weighted_mean([1.0, 2.0], [[1.0, 2.0]])


def test_weighted_mean_with_infinite_values():
    """Infinite values give an infinite mean, or NaN with both signs."""
    assert weighted_mean([1.0, np.inf], [1.0, 1.0]) == np.inf
    with np.errstate(invalid="ignore"):  # numpy warns about inf - inf
        assert np.isnan(weighted_mean([-np.inf, np.inf]))


def test_weighted_mean_doesnt_overflow_with_huge_weights():
    """Weights whose sum overflows still work (np.average gives NaN)."""
    assert weighted_mean([1.0, 3.0], [1e308, 1e308]) == 2
    with np.errstate(over="ignore", invalid="ignore"):
        assert np.isnan(np.average([1.0, 3.0], weights=[1e308, 1e308]))


def test_weighted_mean_inputs_and_result():
    """Booleans, integers and pandas Series work; Series are matched by position, not index; the result is a float."""
    assert weighted_mean([True, False, True, True]) == 0.75
    result = weighted_mean(pd.Series([1, 2], index=[5, 6]), pd.Series([1, 3], index=[6, 5]))
    assert result == 1.75
    assert type(result) is float


def test_weighted_mean_skip_na_is_keyword_only():
    """skip_na must be given by keyword."""
    with pytest.raises(TypeError):
        weighted_mean([1.0], [1.0], True)  # type: ignore[call-arg]


# --- hc1_inflation_factor ----------------------------------------------------------------------------------------


def statsmodels_hc1_factor(values, weights):
    """The factor from statsmodels: fit the intercept-only WLS model, and divide its HC1 SE by its ordinary SE."""
    smf = pytest.importorskip("statsmodels.formula.api")
    fit = smf.wls("y ~ 1", data=pd.DataFrame({"y": values}), weights=weights).fit()
    return fit.HC1_se["Intercept"] / fit.bse["Intercept"]


def random_values_and_weights(seed, count=300):
    """Yield random values and weights: various sizes, weight patterns, and constant or varying spreads."""
    rng = np.random.default_rng(seed)
    for i in range(count):
        size = int(rng.integers(2, 200))
        weights = [rng.uniform(0.1, 5, size), rng.integers(1, 10, size).astype(float), 10 ** rng.uniform(-3, 3, size)]
        spread = 1 if i % 2 else rng.uniform(0.1, 10, size)
        yield rng.normal(size=size) * spread, weights[i % 3]


def test_hc1_factor_matches_statsmodels():
    """The formula gives what fitting the intercept-only WLS model in statsmodels gives (to round-off)."""
    for values, weights in random_values_and_weights(14):
        assert hc1_inflation_factor(values, weights) == pytest.approx(statsmodels_hc1_factor(values, weights), rel=1e-9)


def test_hc1_factor_counts_zero_weight_rows_like_statsmodels():
    """Every row counts in n, including zero-weight ones, as in statsmodels; the factor is continuous as a weight
    goes to 0."""
    rng = np.random.default_rng(15)
    values, weights = rng.normal(size=20), rng.uniform(0.5, 2, 20)
    with_zero = hc1_inflation_factor(np.append(values, 0.5), np.append(weights, 0.0))
    assert with_zero == pytest.approx(statsmodels_hc1_factor(np.append(values, 0.5), np.append(weights, 0.0)), rel=1e-9)
    assert with_zero == pytest.approx(hc1_inflation_factor(np.append(values, 0.5), np.append(weights, 1e-12)))
    assert with_zero != pytest.approx(hc1_inflation_factor(values, weights))


def test_hc1_factor_is_1_with_equal_weights():
    """With equal weights, the robust and ordinary standard errors agree."""
    for values, _weights in random_values_and_weights(16, 50):
        assert hc1_inflation_factor(values, np.full(len(values), 3.0)) == pytest.approx(1, rel=1e-12)


def test_hc1_factor_with_equal_size_residuals_is_the_root_of_kishs_design_effect():
    """If every residual has the same size, the factor is sqrt(n * sum(w**2) / sum(w)**2)."""
    weights = np.array([1.0, 3.0, 2.0, 2.0, 5.0, 5.0])
    values = 4 + 0.7 * np.array([1, 1, -1, -1, 1, -1])  # weighted mean 4, every residual +-0.7
    kish = len(weights) * np.sum(weights**2) / np.sum(weights) ** 2
    assert hc1_inflation_factor(values, weights) == pytest.approx(np.sqrt(kish), rel=1e-12)


def test_hc1_factor_doesnt_change_when_the_weights_are_scaled():
    """Only the weights' ratios matter, so huge or tiny weights work too."""
    values, weights = next(random_values_and_weights(17))
    factor = hc1_inflation_factor(values, weights)
    for scale in (1e-300, 1e-6, 7.0, 1e300):
        assert hc1_inflation_factor(values, weights * scale) == pytest.approx(factor, rel=1e-12)


def test_hc1_factor_skips_missing_pairs():
    """With skip_na, pairs with a missing value or weight are dropped; without it, they raise."""
    values, weights = [1.0, np.nan, 3.0, 4.0, 10.0], [1.0, 2.0, np.nan, 1.0, 3.0]
    assert hc1_inflation_factor(values, weights, skip_na=True) == hc1_inflation_factor(
        [1.0, 4.0, 10.0], [1.0, 1.0, 3.0]
    )
    with pytest.raises(ValueError, match="values or weights are NaN; pass skip_na=True"):
        hc1_inflation_factor(values, weights)


@pytest.mark.parametrize(
    ("values", "weights", "message"),
    [
        ([1.0], [1.0], "at least two values"),
        ([1.0, np.nan, 2.0], [1.0, 1.0, np.nan], "NaN"),
        ([2.0, 2.0, 2.0], [1.0, 2.0, 3.0], "both standard errors are 0"),
        ([2.0, 2.0, 5.0], [1.0, 1.0, 0.0], "both standard errors are 0"),
        ([1.0, np.inf], [1.0, 1.0], "values must be finite"),
        ([1.0, 2.0], [1.0, -1.0], "non-negative"),
        ([1.0, 2.0], [0.0, 0.0], "add up to 0"),
        ([1.0, 2.0], [1.0, 1.0, 1.0], "differ in length"),
        ([[1.0, 2.0]], [1.0, 1.0], "values must be 1-D"),
    ],
)
def test_hc1_factor_rejects_unusable_inputs(values, weights, message):
    """Fewer than two values, all residuals 0, infinite values, bad weights or shapes raise ValueError."""
    with pytest.raises(ValueError, match=message):
        hc1_inflation_factor(values, weights)


def test_hc1_factor_with_too_few_values_left_after_skipping():
    """Skipping missing pairs can leave fewer than two values, which raises."""
    with pytest.raises(ValueError, match="at least two values"):
        hc1_inflation_factor([1.0, np.nan, 3.0], [1.0, 1.0, np.nan], skip_na=True)


def test_hc1_factor_inputs_and_result():
    """pandas Series (matched by position) and integers work; the result is a float; skip_na is keyword-only."""
    result = hc1_inflation_factor(pd.Series([1, 2, 4], index=[3, 2, 1]), pd.Series([1, 2, 3], index=[1, 2, 3]))
    assert type(result) is float
    assert result == hc1_inflation_factor([1.0, 2.0, 4.0], [1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        hc1_inflation_factor([1.0, 2.0], [1.0, 1.0], True)  # type: ignore[call-arg]
