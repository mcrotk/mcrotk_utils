"""Tests for ``mcrotk_utils.stats_utils``' hypothesis types, p-values, confidence intervals and empirical CDFs.

Besides fixed cases, some tests compare with independent implementations: numpy, scipy and statsmodels where
they compute the same thing, and otherwise straightforward reference implementations written here.
"""

from typing import get_args

import numpy as np
import pytest
from scipy.stats import norm, percentileofscore, ttest_1samp
from scipy.stats import t as student_t

from mcrotk_utils.stats_utils import (
    DEFAULT_ALPHA,
    HypothesisType,
    QuantileMethod,
    ci_from_empirical_dist,
    ci_from_t_dist,
    empirical_cdf,
    empirical_sf,
    negate_ci,
    p_value_from_empirical_dist,
    p_value_from_t_stat,
)
from mcrotk_utils.stats_utils._inference import _LOWER_WORDS, _UPPER_WORDS, _one_sided_aliases

UPPER, LOWER, TWO_SIDED = HypothesisType.ONE_SIDED_UPPER, HypothesisType.ONE_SIDED_LOWER, HypothesisType.TWO_SIDED


def spellings(words):
    """Return many ways of writing ``words`` (a list of words): cases, separators, digits, punctuation around."""
    joined = []
    for separator in ("", "-", "_", ".", "-_"):
        joined.append(separator.join(words))
    variants = set()
    for text in joined:
        for cased in (text, text.upper(), text.title(), text.capitalize()):
            variants |= {cased, cased.replace("one", "1").replace("two", "2"), f"({cased})", f"{cased}!"}
            variants.add(cased.replace("One", "1").replace("Two", "2").replace("ONE", "1").replace("TWO", "2"))
    return sorted(variants)


# The words naming each one-sided direction, as the HypothesisType docstring lists them
DIRECTION_WORDS = {
    UPPER: ("upper", "right", "increase", "greater", "larger"),
    LOWER: ("lower", "left", "decrease", "less", "smaller"),
}

# Every accepted name, as words, for each member: two-sided, and each direction word alone, after "one sided" or
# "one tailed", or before "tailed"
NAME_WORDS = {
    TWO_SIDED: [["two", "sided"], ["two", "tailed"]],
    **{
        member: [
            form
            for word in words
            for form in ([word], ["one", "sided", word], ["one", "tailed", word], [word, "tailed"])
        ]
        for member, words in DIRECTION_WORDS.items()
    },
}


# --- HypothesisType: creating one ------------------------------------------------------------------------------------


@pytest.mark.parametrize("member", list(HypothesisType))
def test_members_and_values_give_the_member(member):
    """A member is returned as it is, and its value gives it."""
    assert HypothesisType(member) is member
    assert HypothesisType(member.value) is member


def test_values():
    """The members' values are as documented."""
    assert [member.value for member in HypothesisType] == ["two-sided", "one-sided upper", "one-sided lower"]


@pytest.mark.parametrize("value", [None, "", 0, 0.0, False, [], ()])
def test_falsy_values_raise(value):
    """None and other falsy values raise: they aren't hypothesis types (functions default to TWO_SIDED instead)."""
    with pytest.raises(ValueError, match="Unrecognized hypothesis type"):
        HypothesisType(value)


def test_accepts_every_spelling_of_every_name():
    """Every name gives its member, in any case, with any separators or punctuation, and with digits for one/two."""
    checked = 0
    for member, names in NAME_WORDS.items():
        for words in names:
            for text in spellings(words):
                assert HypothesisType(text) is member, text
                checked += 1
    assert checked > 1000


@pytest.mark.parametrize(("member", "words"), [(m, w) for m, names in NAME_WORDS.items() for w in names])
def test_accepts_names_with_whitespace(member, words):
    """Whitespace is ignored, so "One Sided Upper" and "two tailed" work."""
    assert HypothesisType(" ".join(words)) is member
    assert HypothesisType(" ".join(words).title()) is member
    assert HypothesisType(f" {'  '.join(words)}\t") is member


@pytest.mark.parametrize(
    "value",
    ["sideways", "one-sided", "one sided", "three-sided", "upperlower", "uppers", "two-sided upper", "1", 1, 2.5],
)
def test_unrecognized_values_raise(value):
    """Anything else raises ValueError, naming the value."""
    with pytest.raises(ValueError, match="Unrecognized hypothesis type"):
        HypothesisType(value)


# --- HypothesisType: names for libraries -----------------------------------------------------------------------------


def test_scipy_alternative():
    """The names scipy uses (two-sided, greater, less), which its tests accept with the expected meaning."""
    assert [m.scipy_alternative for m in HypothesisType] == ["two-sided", "greater", "less"]
    larger_than_0 = [0.8, 1.3, 0.2, 1.9, 1.1]
    p_values = {m: ttest_1samp(larger_than_0, 0, alternative=m.scipy_alternative).pvalue for m in HypothesisType}
    assert p_values[UPPER] < p_values[TWO_SIDED] < p_values[LOWER]


def test_statsmodels_alternative():
    """The names statsmodels uses (two-sided, larger, smaller), which its tests accept with the expected meaning."""
    assert [m.statsmodels_alternative for m in HypothesisType] == ["two-sided", "larger", "smaller"]
    weightstats = pytest.importorskip("statsmodels.stats.weightstats")
    larger_than_0 = np.array([0.8, 1.3, 0.2, 1.9, 1.1])
    p_values = {m: weightstats.ztest(larger_than_0, alternative=m.statsmodels_alternative)[1] for m in HypothesisType}
    assert p_values[UPPER] < p_values[TWO_SIDED] < p_values[LOWER]


# --- HypothesisType.get_array ----------------------------------------------------------------------------------------


def check_members_array(array, expected):
    """Check that ``array`` is a 1-D object array of exactly the members ``expected``."""
    assert isinstance(array, np.ndarray)
    assert array.dtype == object
    assert array.ndim == 1
    assert list(array) == expected
    assert all(type(item) is HypothesisType for item in array)


@pytest.mark.parametrize("value", ["upper", UPPER, ["upper"], np.array(["upper"])])
def test_get_array_repeats_one_value(value):
    """One value (or a list of one) is repeated desired_length times, or once without desired_length."""
    member = HypothesisType(value[0] if isinstance(value, list | np.ndarray) else value)
    check_members_array(HypothesisType.get_array(value, 3), [member] * 3)
    check_members_array(HypothesisType.get_array(value), [member])
    check_members_array(HypothesisType.get_array(value, 0), [])


def test_get_array_default_is_two_sided():
    """Without values, the default is two-sided."""
    check_members_array(HypothesisType.get_array(desired_length=2), [TWO_SIDED, TWO_SIDED])
    check_members_array(HypothesisType.get_array(), [TWO_SIDED])


@pytest.mark.parametrize("values", [None, [None], ["upper", None]])
def test_get_array_rejects_none(values):
    """None isn't a hypothesis type, alone or in a list."""
    with pytest.raises(ValueError, match="Unrecognized hypothesis type None"):
        HypothesisType.get_array(values)


def test_get_array_converts_several_values_in_order():
    """Several values give one member each, in order, with or without desired_length."""
    values = ["upper", LOWER, "two-sided", "one-sided increase"]
    expected = [UPPER, LOWER, TWO_SIDED, UPPER]
    check_members_array(HypothesisType.get_array(values), expected)
    check_members_array(HypothesisType.get_array(values, 4), expected)
    check_members_array(HypothesisType.get_array(tuple(values)), expected)


def test_get_array_flattens_nested_values():
    """Nested lists and arrays of more than one dimension are flattened."""
    check_members_array(HypothesisType.get_array([["upper", "lower"], ["left", "right"]]), [UPPER, LOWER, LOWER, UPPER])
    check_members_array(HypothesisType.get_array(np.array([["upper"], ["lower"]]), 2), [UPPER, LOWER])


@pytest.mark.parametrize("values", [[], (), np.array([])])
def test_get_array_with_no_values_is_empty(values):
    """No values give an empty object array (not numpy's default float one), and need desired_length 0."""
    check_members_array(HypothesisType.get_array(values), [])
    check_members_array(HypothesisType.get_array(values, 0), [])
    with pytest.raises(ValueError, match="expected either 1 or desired_length"):
        HypothesisType.get_array(values, 2)


def test_get_array_rejects_a_wrong_number_of_values():
    """Several values must number desired_length."""
    with pytest.raises(ValueError, match="Got 2 hypothesis types; expected either 1 or desired_length, 3"):
        HypothesisType.get_array(["upper", "lower"], 3)


def test_get_array_rejects_a_negative_length():
    """desired_length can't be negative."""
    with pytest.raises(ValueError, match="non-negative"):
        HypothesisType.get_array("upper", -1)


def test_get_array_rejects_unrecognized_values():
    """A value HypothesisType doesn't recognize raises its ValueError."""
    with pytest.raises(ValueError, match="Unrecognized hypothesis type 'sideways'"):
        HypothesisType.get_array(["upper", "sideways"])


@pytest.mark.parametrize(
    ("values", "desired_length", "expected"),
    [
        ("two-sided", None, [TWO_SIDED]), ("two-sided", 4, [TWO_SIDED] * 4), ("upper", 3, [UPPER] * 3),
        ("Left", None, [LOWER]), (["upper"], 2, [UPPER] * 2), (["upper", "lower"], None, [UPPER, LOWER]),
        (["upper", "lower"], 2, [UPPER, LOWER]), (["2-tailed", "right", "lower"], 3, [TWO_SIDED, UPPER, LOWER]),
        (np.array(["upper", "left"]), 2, [UPPER, LOWER]),
    ],
)  # fmt: skip
def test_get_array_cases(values, desired_length, expected):
    """One value repeated, or several converted in order, as the docstring says."""
    check_members_array(HypothesisType.get_array(values, desired_length), expected)


def test_no_name_is_accepted_for_two_members():
    """No name is in more than one member's set of aliases, so none is ambiguous."""
    upper = _one_sided_aliases(_UPPER_WORDS)
    lower = _one_sided_aliases(_LOWER_WORDS)
    two_sided = {"twosided", "twotailed"}
    assert not upper & lower
    assert not two_sided & (upper | lower)
    assert len(upper) == len(lower) == 4 * 5


# --- p_value_from_empirical_dist ------------------------------------------------------------------------------------


def reference_p_value(value, draws, hypothesis_type, include_observed=True):
    """A direct implementation of the p-value: count the draws at least as extreme, one by one."""
    extra = 1 if include_observed else 0
    upper = (sum(1 for draw in draws if draw >= value) + extra) / (len(draws) + extra)
    lower = (sum(1 for draw in draws if draw <= value) + extra) / (len(draws) + extra)
    return {UPPER: upper, LOWER: lower, TWO_SIDED: min(1.0, 2 * min(upper, lower))}[hypothesis_type]


def rejection_rate(values, alpha, hypothesis_type, p_value=None):
    """Return how often a test at level ``alpha`` rejects when the null hypothesis holds, exactly.

    Under the null hypothesis the observed statistic and the draws are exchangeable: given the set ``values`` of
    all of them, each is equally likely to be the observed one, with the rest as the draws. So the probability of
    rejecting is the fraction of the ``values`` that give a p-value of at most ``alpha`` when taken as observed.
    """
    p_value = p_value or p_value_from_empirical_dist
    values = np.asarray(values)
    rejections = [p_value(values[i], np.delete(values, i), hypothesis_type) <= alpha for i in range(len(values))]
    return np.mean(rejections)


@pytest.mark.parametrize("hypothesis_type", list(HypothesisType))
@pytest.mark.parametrize("include_observed", [True, False])
def test_p_value_matches_reference_on_random_cases(hypothesis_type, include_observed):
    """On random continuous draws and draws with many ties, the p-value is the reference implementation's."""
    rng = np.random.default_rng(20261117)
    for _ in range(300):
        size = rng.integers(1, 40)
        draws = rng.normal(size=size) if rng.integers(2) else rng.integers(-2, 3, size=size).astype(float)
        value = rng.normal(scale=2) if rng.integers(2) else float(rng.integers(-3, 4))
        new = p_value_from_empirical_dist(value, draws, hypothesis_type, include_observed=include_observed)
        assert new == pytest.approx(reference_p_value(value, draws, hypothesis_type, include_observed), rel=1e-12)


def test_p_value_counts_ties_on_both_sides():
    """The docstring's example: ties count as at least as extreme, for either direction."""
    draws = [0, 1, 1, 2]
    assert p_value_from_empirical_dist(1, draws, UPPER, include_observed=False) == 3 / 4
    assert p_value_from_empirical_dist(1, draws, LOWER, include_observed=False) == 3 / 4
    assert p_value_from_empirical_dist(1, draws, TWO_SIDED, include_observed=False) == 1
    # with the observed value counted as a draw: (3 + 1) / (4 + 1)
    assert p_value_from_empirical_dist(1, draws, UPPER) == 4 / 5
    assert p_value_from_empirical_dist(1, draws, LOWER) == 4 / 5
    assert p_value_from_empirical_dist(1, draws) == 1


def test_p_value_counts_the_observed_value_as_a_draw_by_default():
    """(k + 1) / (B + 1) by default, so never 0; k / B with include_observed=False."""
    draws = np.arange(10)
    assert p_value_from_empirical_dist(100, draws, UPPER) == 1 / 11
    assert p_value_from_empirical_dist(100, draws, UPPER, include_observed=False) == 0
    assert p_value_from_empirical_dist(100, draws, LOWER) == 1
    assert p_value_from_empirical_dist(100, draws, TWO_SIDED) == 2 / 11
    assert p_value_from_empirical_dist(-100, draws, LOWER) == 1 / 11
    assert p_value_from_empirical_dist(4.5, draws, UPPER) == 6 / 11
    assert p_value_from_empirical_dist(4.5, draws, UPPER, include_observed=False) == 5 / 10


def test_p_value_without_counting_the_observed_value_for_a_complete_list():
    """When dist_data already includes the observed assignment (all assignments listed), include_observed=False."""
    every_assignment = [1.0, 2.0, 3.0, 4.0]  # the observed statistic, 4.0, is one of them
    assert p_value_from_empirical_dist(4.0, every_assignment, UPPER, include_observed=False) == 1 / 4
    assert p_value_from_empirical_dist(4.0, every_assignment, UPPER) == 2 / 5  # counts the observed one twice


@pytest.mark.parametrize("hypothesis_type", list(HypothesisType))
@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1, 0.2, 0.5])
def test_p_value_is_valid_under_the_null_without_ties(hypothesis_type, alpha):
    """With distinct values, a test at level alpha rejects at most a fraction alpha of the time, for any B."""
    for num_draws in range(1, 120):
        values = np.arange(num_draws + 1, dtype=float)
        assert rejection_rate(values, alpha, hypothesis_type) <= alpha + 1e-12, num_draws


@pytest.mark.parametrize("hypothesis_type", list(HypothesisType))
@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1, 0.2])
def test_p_value_is_valid_under_the_null_with_ties(hypothesis_type, alpha):
    """With many ties (a statistic taking few values), the test still rejects at most a fraction alpha."""
    rng = np.random.default_rng(1117)
    for _ in range(100):
        values = rng.integers(0, rng.integers(2, 6), size=rng.integers(2, 60)).astype(float)
        assert rejection_rate(values, alpha, hypothesis_type) <= alpha + 1e-12, values


def test_p_value_without_the_observed_value_can_reject_too_often():
    """k / B (include_observed=False) can reject more than alpha of the time under the null hypothesis, which the
    default (k + 1) / (B + 1) avoids: with 21 distinct values at alpha 0.1, 3/21 of the time against 2/21."""

    def k_over_b(value, draws, hypothesis_type):
        """The p-value without including the observed value."""
        return p_value_from_empirical_dist(value, draws, hypothesis_type, include_observed=False)

    assert rejection_rate(np.arange(21.0), 0.1, UPPER, k_over_b) == pytest.approx(3 / 21)
    assert rejection_rate(np.arange(21.0), 0.1, UPPER) == pytest.approx(2 / 21)


def test_p_value_is_a_probability_and_never_zero_by_default():
    """The p-value is a float between 0 and 1, above 0 when counting the observed value."""
    rng = np.random.default_rng(17)
    for _ in range(300):
        dist_data = rng.integers(-3, 4, size=rng.integers(1, 30))
        value = rng.integers(-5, 6)
        for hypothesis_type in HypothesisType:
            p = p_value_from_empirical_dist(value, dist_data, hypothesis_type)
            assert type(p) is float
            assert 0 < p <= 1
            assert 0 <= p_value_from_empirical_dist(value, dist_data, hypothesis_type, include_observed=False) <= 1


def test_p_value_default_is_two_sided():
    """Without a hypothesis type, the test is two-sided."""
    assert p_value_from_empirical_dist(9, np.arange(10)) == p_value_from_empirical_dist(9, np.arange(10), TWO_SIDED)
    assert p_value_from_empirical_dist(9, np.arange(10)) == 4 / 11


@pytest.mark.parametrize(
    ("name", "member"),
    [("greater", UPPER), ("one-sided increase", UPPER), ("1-sided lower", LOWER), ("Two Tailed", TWO_SIDED)],
)
def test_p_value_accepts_hypothesis_type_names(name, member):
    """Names HypothesisType accepts work as well as members."""
    draws = [0.0, 1.0, 2.0, 3.0, 5.0]
    assert p_value_from_empirical_dist(2.5, draws, name) == p_value_from_empirical_dist(2.5, draws, member)


def test_p_value_accepts_lists_and_integers():
    """dist_data can be a list or an integer array, and value an integer."""
    assert p_value_from_empirical_dist(2, [1, 2, 3], UPPER) == 3 / 4
    assert p_value_from_empirical_dist(2, np.array([1, 2, 3]), UPPER) == 3 / 4


@pytest.mark.parametrize("skip_na", [False, True])
def test_p_value_rejects_a_nan_value(skip_na):
    """A NaN observed value raises ValueError, even with skip_na (there's nothing to test)."""
    with pytest.raises(ValueError, match="observed value is NaN"):
        p_value_from_empirical_dist(np.nan, [1.0, 2.0], skip_na=skip_na)


@pytest.mark.parametrize("hypothesis_type", list(HypothesisType))
def test_p_value_rejects_nan_draws_by_default(hypothesis_type):
    """NaN draws raise ValueError, naming how many and how to drop them."""
    with pytest.raises(ValueError, match="2 NaN values; pass skip_na=True"):
        p_value_from_empirical_dist(1.0, [1.0, np.nan, 2.0, np.nan], hypothesis_type)


@pytest.mark.parametrize("hypothesis_type", list(HypothesisType))
@pytest.mark.parametrize("include_observed", [True, False])
def test_p_value_with_skip_na_drops_nan_draws(hypothesis_type, include_observed):
    """With skip_na, NaN draws are dropped and B is the number of draws left."""
    with_nans = [0.0, np.nan, 1.0, 2.0, np.nan, 3.0]
    without = [0.0, 1.0, 2.0, 3.0]
    for value in (-1.0, 1.5, 2.0, 4.0):
        new = p_value_from_empirical_dist(
            value, with_nans, hypothesis_type, include_observed=include_observed, skip_na=True
        )
        assert new == p_value_from_empirical_dist(value, without, hypothesis_type, include_observed=include_observed)
    assert p_value_from_empirical_dist(2.0, with_nans, UPPER, skip_na=True) == 3 / 5


def test_p_value_with_skip_na_and_only_nan_draws_raises():
    """With skip_na, draws that are all NaN leave no draws, which raises ValueError."""
    with pytest.raises(ValueError, match="no values that aren't NaN"):
        p_value_from_empirical_dist(1.0, [np.nan, np.nan], skip_na=True)


def test_p_value_skip_na_changes_nothing_without_nans():
    """skip_na makes no difference when there are no NaN draws."""
    draws = [0.0, 1.0, 1.0, 2.0]
    for hypothesis_type in HypothesisType:
        assert p_value_from_empirical_dist(1.0, draws, hypothesis_type, skip_na=True) == p_value_from_empirical_dist(
            1.0, draws, hypothesis_type
        )


@pytest.mark.parametrize("include_observed", [True, False])
def test_p_value_rejects_empty_draws(include_observed):
    """No draws raise ValueError, even though counting the observed value alone would give 1."""
    with pytest.raises(ValueError, match="no values"):
        p_value_from_empirical_dist(1.0, [], include_observed=include_observed)
    with pytest.raises(ValueError, match="no values"):
        p_value_from_empirical_dist(1.0, np.array([]), UPPER, include_observed=include_observed)


def test_p_value_rejects_draws_that_arent_1d():
    """dist_data must be 1-D: a 2-D array or a scalar raises ValueError."""
    with pytest.raises(ValueError, match="1-D, but has 2 dimensions"):
        p_value_from_empirical_dist(1.0, [[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="1-D, but has 0 dimensions"):
        p_value_from_empirical_dist(1.0, 2.0)


@pytest.mark.parametrize("hypothesis_type", ["sideways", None])
def test_p_value_rejects_unrecognized_hypothesis_types(hypothesis_type):
    """A hypothesis type HypothesisType doesn't accept raises its ValueError."""
    with pytest.raises(ValueError, match="Unrecognized hypothesis type"):
        p_value_from_empirical_dist(1.0, [1.0, 2.0], hypothesis_type)


def test_p_value_flags_are_keyword_only():
    """include_observed and skip_na must be given by keyword, so a positional True can't be mistaken for another."""
    with pytest.raises(TypeError):
        p_value_from_empirical_dist(1.0, [1.0, 2.0], UPPER, False)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        p_value_from_empirical_dist(1.0, [1.0, 2.0], UPPER, True, False)  # type: ignore[call-arg]


# --- ci_from_empirical_dist ------------------------------------------------------------------------------------


def reference_ci(dist_data, alpha, hypothesis_type, bounds, method: QuantileMethod):
    """A direct implementation of the interval: quantiles at the tail probabilities, and the open end from bounds."""
    tail = alpha / 2 if hypothesis_type is TWO_SIDED else alpha
    low_end, high_end = (np.min(dist_data), np.max(dist_data)) if bounds == "values" else bounds
    lower = low_end if hypothesis_type is LOWER else np.quantile(dist_data, tail, method=method)
    upper = high_end if hypothesis_type is UPPER else np.quantile(dist_data, 1 - tail, method=method)
    return np.array([lower, upper], dtype=float)


# Some of the methods np.quantile accepts, for comparisons with the reference implementation
METHODS: list[QuantileMethod] = ["inverted_cdf", "linear", "nearest", "hazen"]


def random_distributions(seed):
    """Yield random distributions: continuous ones and ones with many ties, of 1 to 60 values."""
    rng = np.random.default_rng(seed)
    for _ in range(150):
        size = rng.integers(1, 61)
        yield rng.normal(size=size)
        yield rng.integers(-3, 4, size=size).astype(float)


def test_default_alpha():
    """DEFAULT_ALPHA is 0.05, for 95% intervals, and is ci_from_empirical_dist's default."""
    assert DEFAULT_ALPHA == 0.05
    values = np.arange(1.0, 101.0)
    assert np.array_equal(ci_from_empirical_dist(values), ci_from_empirical_dist(values, 0.05))


def test_quantile_methods_are_all_accepted_by_numpy():
    """Every method in QuantileMethod works with np.quantile (and the interval), in this numpy version."""
    values = np.arange(10.0)
    for method in get_args(QuantileMethod.__value__):
        np.quantile(values, 0.3, method=method)
        ci_from_empirical_dist(values, quantile_method=method)
    assert len(get_args(QuantileMethod.__value__)) == 13


@pytest.mark.parametrize("hypothesis_type", list(HypothesisType))
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("bounds", [(-np.inf, np.inf), "values"])
def test_ci_matches_reference_on_random_cases(hypothesis_type, method, bounds):
    """The interval is the reference implementation's, for every hypothesis type, these methods, and both the
    default (unbounded) and "values" bounds, on continuous values and values with many ties."""
    rng = np.random.default_rng(5)
    for dist_data in random_distributions(2026):
        for alpha in (0.05, 0.1, 0.01, float(rng.uniform())):
            new = ci_from_empirical_dist(dist_data, alpha, hypothesis_type, bounds=bounds, quantile_method=method)
            assert np.array_equal(new, reference_ci(dist_data, alpha, hypothesis_type, bounds, method)), (
                dist_data,
                alpha,
            )


def test_ci_docstring_example():
    """Values 1 to 100, alpha 0.05: two-sided [3, 98]; upper [5, inf], or [5, 100] with bounds="values"."""
    values = np.arange(1, 101)
    assert list(ci_from_empirical_dist(values)) == [3, 98]
    assert list(ci_from_empirical_dist(values, 0.05, UPPER)) == [5, np.inf]
    assert list(ci_from_empirical_dist(values, 0.05, UPPER, bounds="values")) == [5, 100]
    assert list(ci_from_empirical_dist(values, 0.05, LOWER)) == [-np.inf, 95]
    assert list(ci_from_empirical_dist(values, 0.05, LOWER, bounds="values")) == [1, 95]


def test_ci_with_natural_bounds():
    """Finite bounds give a one-sided interval's open end, e.g. (0, 1) for a probability."""
    values = np.linspace(0.2, 0.6, 41)
    upper = ci_from_empirical_dist(values, 0.1, UPPER, bounds=(0, 1))
    lower = ci_from_empirical_dist(values, 0.1, LOWER, bounds=(0, 1))
    assert upper[1] == 1
    assert upper[0] == ci_from_empirical_dist(values, 0.1, UPPER)[0]
    assert lower[0] == 0
    assert lower[1] == ci_from_empirical_dist(values, 0.1, LOWER)[1]
    # values on the bounds themselves are within them
    assert list(ci_from_empirical_dist([0.0, 0.5, 1.0], 0.5, UPPER, bounds=(0, 1))) == [0.5, 1.0]


@pytest.mark.parametrize("bounds", [(-np.inf, np.inf), "values", (-10, 10)])
def test_ci_bounds_dont_affect_two_sided_intervals(bounds):
    """A two-sided interval is the same whatever the bounds."""
    for dist_data in random_distributions(3):
        assert np.array_equal(ci_from_empirical_dist(dist_data, bounds=bounds), ci_from_empirical_dist(dist_data))


def test_ci_with_values_outside_bounds_raises():
    """Values outside finite bounds raise ValueError, saying how many and giving one, even for two-sided intervals."""
    with pytest.raises(ValueError, match=r"2 values lie outside bounds \(0, 1\), e.g. 1.2"):
        ci_from_empirical_dist([0.5, 1.2, -0.1], 0.05, UPPER, bounds=(0, 1))
    with pytest.raises(ValueError, match="1 values lie outside bounds"):
        ci_from_empirical_dist([0.5, 1.2], bounds=(0, 1))


@pytest.mark.parametrize("bounds", ["draws", (1, 0), (np.nan, 1)])
def test_ci_with_invalid_bounds_raises(bounds):
    """bounds must be "values" or a pair (low, high) with low <= high."""
    with pytest.raises(ValueError, match="bounds must be"):
        ci_from_empirical_dist([0.5], 0.05, UPPER, bounds=bounds)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, np.nan])
def test_ci_with_alpha_outside_0_to_1_raises(alpha):
    """alpha must be between 0 and 1."""
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        ci_from_empirical_dist([1.0, 2.0], alpha)


def test_ci_with_alpha_0_or_1():
    """alpha can be 0 (the whole range of the values, two-sided) or 1 (a single quantile)."""
    values = np.arange(1.0, 11.0)
    assert list(ci_from_empirical_dist(values, 0)) == [1, 10]
    assert list(ci_from_empirical_dist(values, 1)) == [5, 5]
    assert list(ci_from_empirical_dist(values, 0, UPPER)) == [1, np.inf]


def test_ci_default_method_gives_values():
    """The default method gives one of the values: the smallest with at least that fraction at or below it.

    That is, the inverse of empirical_cdf.
    """
    for dist_data in random_distributions(8):
        for alpha in (0.05, 0.2, 0.5):
            lower, upper = ci_from_empirical_dist(dist_data, alpha)
            for end, fraction in ((lower, alpha / 2), (upper, 1 - alpha / 2)):
                assert end in dist_data
                at_or_below = np.mean(dist_data[:, None] <= np.sort(dist_data)[None, :], axis=0)
                assert end == np.sort(dist_data)[np.argmax(at_or_below >= fraction)]
                # the inverse of empirical_cdf: the CDF reaches the fraction at the end, and not at any smaller value
                assert empirical_cdf(end, dist_data) >= fraction
                assert all(empirical_cdf(v, dist_data) < fraction for v in dist_data[dist_data < end])


def test_ci_passes_the_method_to_numpy():
    """quantile_method is np.quantile's method: "linear" interpolates between values."""
    values = np.arange(1.0, 101.0)
    assert list(ci_from_empirical_dist(values, quantile_method="linear")) == pytest.approx([3.475, 97.525])


@pytest.mark.parametrize("hypothesis_type", list(HypothesisType))
def test_ci_rejects_nan_values_by_default(hypothesis_type):
    """NaN values raise ValueError, naming how many and how to drop them."""
    with pytest.raises(ValueError, match="1 NaN values; pass skip_na=True"):
        ci_from_empirical_dist([1.0, np.nan, 2.0], 0.05, hypothesis_type)


@pytest.mark.parametrize("hypothesis_type", list(HypothesisType))
@pytest.mark.parametrize("bounds", [(-np.inf, np.inf), "values"])
def test_ci_with_skip_na_drops_nan_values(hypothesis_type, bounds):
    """With skip_na, NaN values are dropped (including for the "values" bounds)."""
    with_nans = [3.0, np.nan, 1.0, 2.0, np.nan, 5.0, 4.0]
    expected = ci_from_empirical_dist([3.0, 1.0, 2.0, 5.0, 4.0], 0.2, hypothesis_type, bounds=bounds)
    assert np.array_equal(
        ci_from_empirical_dist(with_nans, 0.2, hypothesis_type, bounds=bounds, skip_na=True), expected
    )


def test_ci_with_only_nan_values_or_no_values_raises():
    """No values (left) raise ValueError."""
    with pytest.raises(ValueError, match="no values that aren't NaN"):
        ci_from_empirical_dist([np.nan], skip_na=True)
    with pytest.raises(ValueError, match="no values"):
        ci_from_empirical_dist([])


def test_ci_rejects_values_that_arent_1d():
    """dist_data must be 1-D."""
    with pytest.raises(ValueError, match="1-D, but has 2 dimensions"):
        ci_from_empirical_dist([[1.0, 2.0], [3.0, 4.0]])


def test_ci_result_and_inputs():
    """The result is a float array of two ends; dist_data can be a list of integers; names work as hypothesis types."""
    ci = ci_from_empirical_dist([1, 2, 3, 4], 0.5, "greater")
    assert isinstance(ci, np.ndarray)
    assert ci.dtype == np.float64
    assert ci.shape == (2,)
    assert list(ci) == [2, np.inf]
    with pytest.raises(ValueError, match="Unrecognized hypothesis type"):
        ci_from_empirical_dist([1.0], 0.05, "sideways")


def test_ci_flags_are_keyword_only():
    """bounds, quantile_method and skip_na must be given by keyword."""
    with pytest.raises(TypeError):
        ci_from_empirical_dist([1.0, 2.0], 0.05, UPPER, "values")  # type: ignore[call-arg]


# --- empirical_cdf and empirical_sf ------------------------------------------------------------------------------


def scores_for(values, rng):
    """Return scores to evaluate at: every value (so, ties), values between them, and some outside their range."""
    return np.concatenate([values, rng.uniform(values.min() - 1, values.max() + 1, size=10)])


@pytest.mark.parametrize(("inclusive", "kind"), [(True, "weak"), (False, "strict")])
def test_cdf_matches_scipy(inclusive, kind):
    """inclusive=True is scipy's percentileofscore with kind="weak" (<=), and inclusive=False with kind="strict" (<),
    as a fraction rather than a percentage."""
    rng = np.random.default_rng(4)
    for values in random_distributions(41):
        for score in scores_for(values, rng):
            new = empirical_cdf(score, values, inclusive=inclusive)
            assert new == pytest.approx(percentileofscore(values, score, kind=kind) / 100, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize(("inclusive", "kind"), [(False, "weak"), (True, "strict")])
def test_sf_is_the_complement_of_scipys(inclusive, kind):
    """inclusive=False (>) is 1 minus percentileofscore's kind="weak", and inclusive=True (>=) 1 minus kind="strict",
    to within the round-off of that subtraction."""
    rng = np.random.default_rng(5)
    for values in random_distributions(42):
        for score in scores_for(values, rng):
            new = empirical_sf(score, values, inclusive=inclusive)
            assert new == pytest.approx(1 - percentileofscore(values, score, kind=kind) / 100, rel=1e-12, abs=1e-15)


def test_cdf_and_sf_docstring_examples():
    """Values [1, 2, 2, 3] at 2: CDF 3/4 (1/4 not inclusive); survival function 1/4 (3/4 inclusive)."""
    values = [1, 2, 2, 3]
    assert empirical_cdf(2, values) == 3 / 4
    assert empirical_cdf(2, values, inclusive=False) == 1 / 4
    assert empirical_sf(2, values) == 1 / 4
    assert empirical_sf(2, values, inclusive=True) == 3 / 4


def test_sf_is_counted_without_round_off():
    """The survival function counts values rather than subtracting from 1, so it's exact: 1/20 above 18 is 0.05,
    where 1 - 19/20 would give 0.050000000000000044."""
    values = np.arange(20)
    assert empirical_sf(18, values) == 0.05
    assert 1 - 19 / 20 != 0.05


def test_cdf_and_sf_defaults_add_up_to_1():
    """With both defaults, CDF + survival function = 1 (each value is either at or below the score, or above it)."""
    for values in random_distributions(43):
        scores = scores_for(values, np.random.default_rng(1))
        total = empirical_cdf(scores, values) + empirical_sf(scores, values)
        assert np.allclose(total, 1, rtol=0, atol=1e-15)
        # and the counts behind them add up exactly
        assert np.array_equal(
            empirical_cdf(scores, values) * len(values) + empirical_sf(scores, values) * len(values),
            np.full(len(scores), float(len(values))),
        )


def test_cdf_and_sf_at_and_beyond_the_ends():
    """Below every value the CDF is 0 and the survival function 1; at or above the largest, 1 and 0."""
    values = [1.0, 2.0, 3.0]
    assert empirical_cdf(0, values) == 0 and empirical_sf(0, values) == 1
    assert empirical_cdf(3, values) == 1 and empirical_sf(3, values) == 0
    assert empirical_cdf(np.inf, values) == 1 and empirical_cdf(-np.inf, values) == 0


@pytest.mark.parametrize("function", [empirical_cdf, empirical_sf])
def test_cdf_and_sf_scalar_scores_give_floats(function):
    """A number (Python or NumPy, integer or float) gives a float."""
    for score in (2, 2.0, np.int64(2), np.float64(2.0), np.float32(2.0)):
        result = function(score, [1, 2, 2, 3])
        assert type(result) is float
        assert result == function(2.0, [1, 2, 2, 3])


@pytest.mark.parametrize("function", [empirical_cdf, empirical_sf])
def test_cdf_and_sf_array_scores_give_arrays_of_the_same_shape(function):
    """A list or array of scores gives a float array of the same shape, one fraction per score."""
    values = [1, 2, 2, 3]
    scores = np.array([[0.0, 2.0], [2.5, 3.0]])
    result = function(scores, values)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.shape == (2, 2)
    assert [function(float(score), values) for score in scores.ravel()] == list(result.ravel())
    assert list(function([0.0, 2.0], values)) == [function(0.0, values), function(2.0, values)]
    assert function([], values).shape == (0,)


def test_cdf_gives_a_rejection_rate():
    """A use: the fraction of p-values below alpha (a rejection rate), with inclusive=False."""
    p_values = np.array([0.01, 0.04, 0.05, 0.2, 0.6])
    assert empirical_cdf(0.05, p_values, inclusive=False) == 2 / 5


@pytest.mark.parametrize("function", [empirical_cdf, empirical_sf])
def test_cdf_and_sf_reject_nan(function):
    """NaN scores raise; NaN values raise unless skip_na, which drops them."""
    with pytest.raises(ValueError, match="A value to evaluate at is NaN"):
        function(np.nan, [1.0, 2.0])
    with pytest.raises(ValueError, match="A value to evaluate at is NaN"):
        function([1.0, np.nan], [1.0, 2.0])
    with pytest.raises(ValueError, match="dist_data has 1 NaN values; pass skip_na=True"):
        function(1.0, [1.0, np.nan, 2.0])
    assert function(1.5, [1.0, np.nan, 2.0, np.nan], skip_na=True) == function(1.5, [1.0, 2.0]) == 0.5
    with pytest.raises(ValueError, match="dist_data has no values that aren't NaN"):
        function(1.0, [np.nan], skip_na=True)


@pytest.mark.parametrize("function", [empirical_cdf, empirical_sf])
def test_cdf_and_sf_reject_empty_or_non_1d_values(function):
    """values must be 1-D with at least one value."""
    with pytest.raises(ValueError, match="dist_data has no values"):
        function(1.0, [])
    with pytest.raises(ValueError, match="dist_data must be 1-D, but has 2 dimensions"):
        function(1.0, [[1.0, 2.0]])


@pytest.mark.parametrize("function", [empirical_cdf, empirical_sf])
def test_cdf_and_sf_flags_are_keyword_only(function):
    """inclusive and skip_na must be given by keyword."""
    with pytest.raises(TypeError):
        function(1.0, [1.0, 2.0], True)


# --- p_value_from_t_stat -----------------------------------------------------------------------------------------


def scipy_t_p_value(t_stat, df, hypothesis_type):
    """The p-value from scipy's t distribution directly: upper tail, lower tail (CDF), or twice the tail beyond |t|."""
    if hypothesis_type is UPPER:
        return student_t.sf(t_stat, df)
    if hypothesis_type is LOWER:
        return student_t.cdf(t_stat, df)
    return 2 * student_t.sf(abs(t_stat), df)


@pytest.mark.parametrize("hypothesis_type", list(HypothesisType))
def test_t_p_value_matches_scipys_t_distribution(hypothesis_type):
    """Each test's p-value is the tail probability from scipy's t distribution, for many t-statistics and dfs."""
    rng = np.random.default_rng(50)
    for _ in range(300):
        t_stat, df = rng.normal(scale=3), rng.choice([1, 2.5, 5, 30, 1000, np.inf])
        expected = scipy_t_p_value(t_stat, df, hypothesis_type)
        assert p_value_from_t_stat(t_stat, df, hypothesis_type) == pytest.approx(expected, rel=1e-12, abs=1e-300)


@pytest.mark.parametrize("hypothesis_type", list(HypothesisType))
def test_t_p_value_matches_scipys_one_sample_t_test(hypothesis_type):
    """For a one-sample t-test's statistic, it's what scipy's ttest_1samp gives, with the same alternative."""
    rng = np.random.default_rng(51)
    for _ in range(50):
        sample = rng.normal(loc=rng.normal(scale=0.5), size=int(rng.integers(3, 40)))
        result = ttest_1samp(sample, 0, alternative=hypothesis_type.scipy_alternative)
        p_value = p_value_from_t_stat(float(result.statistic), len(sample) - 1, hypothesis_type)
        assert p_value == pytest.approx(result.pvalue, rel=1e-10)


def test_t_p_value_keeps_precision_for_small_p_values():
    """The tail is computed directly, not as 1 - cdf: t = 40 with 30 df gives about 6.9e-28, not 0."""
    p_value = p_value_from_t_stat(40.0, 30, UPPER)
    assert p_value == pytest.approx(6.863e-28, rel=1e-3)
    assert 1 - student_t.cdf(40.0, 30) == 0


def test_t_p_value_with_infinite_df_is_the_normals():
    """With infinite degrees of freedom it's the normal distribution's p-value."""
    assert p_value_from_t_stat(1.96, np.inf) == pytest.approx(2 * norm.sf(1.96), rel=1e-14)
    assert p_value_from_t_stat(-2.5, np.inf, LOWER) == pytest.approx(norm.cdf(-2.5), rel=1e-14)


def test_t_p_value_two_sided_is_at_most_1():
    """A t-statistic of 0 gives a two-sided p-value of exactly 1, and one-sided ones of 1/2."""
    assert p_value_from_t_stat(0.0, 10) == 1
    assert p_value_from_t_stat(0.0, 10, UPPER) == p_value_from_t_stat(0.0, 10, LOWER) == 0.5


def test_t_p_value_arrays_apply_one_type_to_every_test():
    """An array of t-statistics with a single hypothesis type gives each test's p-value with that type."""
    t_stats = np.array([-3.0, 0.0, 2.0])
    for hypothesis_type in HypothesisType:
        result = p_value_from_t_stat(t_stats, 10, hypothesis_type)
        expected = [scipy_t_p_value(value, 10, hypothesis_type) for value in t_stats]
        assert result == pytest.approx(expected, rel=1e-12)


def test_t_p_value_arrays_take_one_type_per_test():
    """A list or array of hypothesis types (members or names) gives each test its own."""
    t_stats = [-3.0, 0.0, 2.0, 2.0]
    types: list[HypothesisType | str] = [LOWER, TWO_SIDED, "greater", "one-sided lower"]
    expected = [scipy_t_p_value(value, 8, HypothesisType(kind)) for value, kind in zip(t_stats, types, strict=True)]
    assert p_value_from_t_stat(t_stats, 8, types) == pytest.approx(expected, rel=1e-12)
    assert p_value_from_t_stat(t_stats, 8, np.array(types, dtype=object)) == pytest.approx(expected, rel=1e-12)


def test_t_p_value_arrays_pair_t_stats_with_dfs():
    """An array of degrees of freedom is paired with the t-statistics (or broadcast against them)."""
    expected = [scipy_t_p_value(2.0, df, TWO_SIDED) for df in (2, 20, np.inf)]
    assert p_value_from_t_stat([2.0, 2.0, 2.0], [2, 20, np.inf]) == pytest.approx(expected, rel=1e-12)
    assert p_value_from_t_stat(2.0, [2, 20, np.inf]) == pytest.approx(expected, rel=1e-12)
    two_d = p_value_from_t_stat(np.array([[1.0, 2.0], [3.0, 4.0]]), 5)
    assert two_d.shape == (2, 2)
    assert two_d[1, 0] == pytest.approx(scipy_t_p_value(3.0, 5, TWO_SIDED), rel=1e-12)


def test_t_p_value_result_types():
    """A single t-statistic and df give a float; anything else a float array (empty for no t-statistics)."""
    assert type(p_value_from_t_stat(2.0, 10)) is float
    assert type(p_value_from_t_stat(np.float64(2.0), np.int64(10))) is float
    result = p_value_from_t_stat([2.0], 10)
    assert isinstance(result, np.ndarray) and result.dtype == np.float64 and result.shape == (1,)
    assert p_value_from_t_stat([], 10).shape == (0,)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((np.nan, 10), "t-statistic is NaN"),
        (([1.0, np.nan], 10), "t-statistic is NaN"),
        ((1.0, 0), "degrees of freedom must be positive"),
        ((1.0, -2), "degrees of freedom must be positive"),
        ((1.0, np.nan), "degrees of freedom must be positive"),
        (([1.0, 2.0], [5, 6, 7]), "don't match in shape"),
        (([1.0, 2.0], 10, [UPPER, LOWER, LOWER]), "Got 3 hypothesis types; expected either 1 or desired_length, 2"),
        ((1.0, 10, "sideways"), "Unrecognized hypothesis type"),
        ((1.0, 10, None), "Unrecognized hypothesis type"),
    ],
)
def test_t_p_value_rejects_bad_inputs(arguments, message):
    """NaN t-statistics, bad degrees of freedom, mismatched shapes or hypothesis types raise ValueError."""
    with pytest.raises(ValueError, match=message):
        p_value_from_t_stat(*arguments)


# --- ci_from_t_dist ---------------------------------------------------------------------------------------------


def test_t_interval_two_sided_matches_scipys_t_interval():
    """A two-sided interval is scipy's t.interval with confidence 1 - alpha, location the estimate, scale the SE."""
    rng = np.random.default_rng(60)
    for _ in range(300):
        estimate, se = rng.normal(scale=5), rng.uniform(0.01, 3)
        df, alpha = rng.choice([1, 3.5, 10, 100, np.inf]), rng.uniform(0.001, 0.5)
        expected = student_t.interval(1 - alpha, df, loc=estimate, scale=se)
        assert ci_from_t_dist(estimate, se, df, alpha) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("hypothesis_type", list(HypothesisType))
def test_t_interval_matches_statsmodels_one_sample_intervals(hypothesis_type):
    """For a sample mean, it's statsmodels' tconfint_mean with the same alternative (one-sided ends infinite)."""
    weightstats = pytest.importorskip("statsmodels.stats.weightstats")
    rng = np.random.default_rng(61)
    for _ in range(50):
        sample = rng.normal(loc=rng.normal(), size=int(rng.integers(3, 40)))
        stats = weightstats.DescrStatsW(sample)
        expected = stats.tconfint_mean(alpha=0.1, alternative=hypothesis_type.statsmodels_alternative)
        result = ci_from_t_dist(stats.mean, stats.std_mean, len(sample) - 1, 0.1, hypothesis_type)
        assert result == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("hypothesis_type", list(HypothesisType))
def test_t_interval_contains_exactly_the_values_the_t_test_doesnt_reject(hypothesis_type):
    """A value is inside the interval exactly when the t-test of it gives a p-value above alpha."""
    rng = np.random.default_rng(62)
    estimate, se, df, alpha = 1.3, 0.4, 12, 0.05
    lower, upper = ci_from_t_dist(estimate, se, df, alpha, hypothesis_type)
    for value in rng.uniform(-1, 4, size=500):
        p_value = p_value_from_t_stat((estimate - value) / se, df, hypothesis_type)
        assert (lower <= value <= upper) == (p_value > alpha), value


def test_t_interval_one_sided_ends():
    """Upper (the quantity is larger): [estimate - t * se, inf], with t the quantile with alpha above it; lower:
    [-inf, estimate + t * se]."""
    quantile = student_t.isf(0.05, 9)
    assert list(ci_from_t_dist(2.0, 0.5, 9, 0.05, UPPER)) == [2.0 - quantile * 0.5, np.inf]
    assert list(ci_from_t_dist(2.0, 0.5, 9, 0.05, LOWER)) == [-np.inf, 2.0 + quantile * 0.5]


def test_t_interval_keeps_precision_for_small_alpha_and_uses_the_normal_with_infinite_df():
    """A tiny alpha gives a finite interval (the quantile isn't computed from 1 - alpha); infinite df gives the
    normal (z) interval."""
    lower, upper = ci_from_t_dist(0.0, 1.0, 10, 1e-10)
    assert np.isfinite(upper) and upper == pytest.approx(student_t.isf(5e-11, 10), rel=1e-12) and lower == -upper
    assert ci_from_t_dist(0.0, 1.0, np.inf, 0.05) == pytest.approx([-norm.isf(0.025), norm.isf(0.025)], rel=1e-14)


def test_t_interval_edge_values():
    """se 0 gives a zero-width interval (even with alpha 0); alpha 0 an infinite one; alpha 1 (two-sided) the
    estimate alone."""
    assert list(ci_from_t_dist(2.0, 0.0, 5)) == [2.0, 2.0]
    assert list(ci_from_t_dist(2.0, 0.0, 5, 0.0)) == [2.0, 2.0]
    assert list(ci_from_t_dist(2.0, 1.0, 5, 0.0)) == [-np.inf, np.inf]
    assert list(ci_from_t_dist(2.0, 1.0, 5, 1.0)) == [2.0, 2.0]


def test_t_interval_many_at_once():
    """Arrays give one row [lower, upper] per interval; estimate, se, df and alpha pair up by broadcasting, and
    hypothesis types are one for all or one per interval."""
    estimates, ses, dfs, alphas = np.array([1.0, -2.0, 0.5]), np.array([0.2, 1.0, 0.1]), np.array([5, 30, np.inf]), 0.1
    types: list[HypothesisType | str] = [TWO_SIDED, "greater", LOWER]
    result = ci_from_t_dist(estimates, ses, dfs, alphas, types)
    assert result.shape == (3, 2) and result.dtype == np.float64
    for row, (estimate, se, df, kind) in zip(result, zip(estimates, ses, dfs, types, strict=True), strict=True):
        assert list(row) == list(ci_from_t_dist(estimate, se, df, alphas, kind))
    # one type for all, and an alpha per interval
    same = ci_from_t_dist(estimates, ses, 10, [0.01, 0.05, 0.1])
    assert same.shape == (3, 2)
    assert list(same[1]) == list(ci_from_t_dist(-2.0, 1.0, 10, 0.05))
    # a single interval keeps shape (2,), and more dimensions add a final axis of 2
    assert ci_from_t_dist(1.0, 0.2, 5).shape == (2,)
    assert ci_from_t_dist(np.zeros((2, 3)), 1.0, 5).shape == (2, 3, 2)
    assert ci_from_t_dist([], 1.0, 5).shape == (0, 2)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((np.nan, 1.0, 5), "estimates and standard errors must be finite"),
        ((1.0, np.inf, 5), "estimates and standard errors must be finite"),
        ((1.0, -0.1, 5), "standard errors must be non-negative"),
        ((1.0, 1.0, 0), "degrees of freedom must be positive"),
        ((1.0, 1.0, np.nan), "degrees of freedom must be positive"),
        ((1.0, 1.0, 5, 1.5), "alpha must be between 0 and 1"),
        ((1.0, 1.0, 5, [0.05, np.nan]), "alpha must be between 0 and 1"),
        (([1.0, 2.0], [1.0, 1.0, 1.0], 5), "don't match in shape"),
        (([1.0, 2.0], 1.0, 5, 0.05, [UPPER, LOWER, LOWER]), "Got 3 hypothesis types"),
        ((1.0, 1.0, 5, 0.05, "sideways"), "Unrecognized hypothesis type"),
    ],
)
def test_t_interval_rejects_bad_inputs(arguments, message):
    """Non-finite estimates or SEs, negative SEs, bad df or alpha, mismatched shapes or hypothesis types raise."""
    with pytest.raises(ValueError, match=message):
        ci_from_t_dist(*arguments)


# --- negate_ci ---------------------------------------------------------------------------------------------------


def test_negate_ci_one_interval():
    """[lower, upper] becomes [-upper, -lower]; one-sided intervals keep their infinite end, on the other side."""
    assert list(negate_ci([1.0, 3.0])) == [-3.0, -1.0]
    assert list(negate_ci([1.0, np.inf])) == [-np.inf, -1.0]
    assert list(negate_ci([-np.inf, 2.0])) == [-2.0, np.inf]
    assert list(negate_ci([2.0, 2.0])) == [-2.0, -2.0]


def test_negate_ci_many_intervals():
    """Each interval along the last axis is negated: rows of (n, 2), and more dimensions."""
    intervals = np.array([[1.0, 3.0], [-2.0, 0.5], [4.0, 6.0]])
    assert np.array_equal(negate_ci(intervals), [[-3.0, -1.0], [-0.5, 2.0], [-6.0, -4.0]])
    stacked = np.stack([intervals, intervals + 1])
    assert np.array_equal(negate_ci(stacked)[1], negate_ci(intervals + 1))
    assert negate_ci(np.empty((0, 2))).shape == (0, 2)


def test_negate_ci_of_a_t_interval_is_the_interval_of_the_negated_estimate():
    """Negating ci_from_t_dist's intervals gives those for the negated estimates, with upper and lower swapped."""
    estimates, ses = np.array([1.0, -0.5, 2.0]), np.array([0.3, 0.2, 1.0])
    for kind, swapped in ((TWO_SIDED, TWO_SIDED), (UPPER, LOWER), (LOWER, UPPER)):
        negated = negate_ci(ci_from_t_dist(estimates, ses, 10, 0.05, kind))
        assert np.allclose(negated, ci_from_t_dist(-estimates, ses, 10, 0.05, swapped), rtol=1e-15)


def test_negate_ci_twice_gives_the_interval_back():
    """Negating twice gives the original."""
    intervals = np.array([[1.0, 3.0], [-np.inf, 0.5]])
    assert np.array_equal(negate_ci(negate_ci(intervals)), intervals)


@pytest.mark.parametrize(
    ("ci", "message"),
    [
        (1.0, "along its last axis"),
        ([1.0, 2.0, 3.0], "along its last axis"),
        ([[1.0, 2.0, 3.0]], "along its last axis"),
        ([1.0, np.nan], "NaN end"),
        ([3.0, 1.0], "lower end is above its upper end"),
        ([[1.0, 2.0], [5.0, 4.0]], "lower end is above its upper end"),
    ],
)
def test_negate_ci_rejects_non_intervals(ci, message):
    """A last axis not of length 2, a NaN end, or a lower end above the upper end raise ValueError."""
    with pytest.raises(ValueError, match=message):
        negate_ci(ci)
