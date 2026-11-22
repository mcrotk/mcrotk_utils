"""Tests for ``mcrotk_utils.stats_utils``' random covariance matrices and correlation matrices."""

import warnings

import numpy as np
import pytest
from scipy.stats import ortho_group

from mcrotk_utils.stats_utils import (
    HypothesisType,
    NormalParams,
    correlation_from_covariance,
    random_covariance_from_eigenvalues,
    random_covariance_from_eigenvalues_and_variances,
    random_covariance_from_variances,
    random_covariance_matrix,
)
from mcrotk_utils.stats_utils._covariance import _spread_diagonal_until_majorizing, _transfer_diagonal_to_target

UPPER, LOWER, TWO_SIDED = HypothesisType.ONE_SIDED_UPPER, HypothesisType.ONE_SIDED_LOWER, HypothesisType.TWO_SIDED


# --- Random covariance matrices ------------------------------------------------------------------------------------


def sorted_eigenvalues(matrix):
    """Return a symmetric matrix's eigenvalues, in increasing order."""
    return np.linalg.eigvalsh(matrix)


def assert_eigenvalues(matrix, expected):
    """Check that the matrix's eigenvalues are ``expected`` (in any order), to within rounding."""
    scale = max(1.0, float(np.max(expected)))
    assert np.allclose(sorted_eigenvalues(matrix), np.sort(expected), rtol=0, atol=1e-10 * scale)


def assert_symmetric(matrix):
    """Check that the matrix is exactly symmetric."""
    assert np.array_equal(matrix, matrix.T)


def feasible_eigenvalues_and_variances(seed, size, alpha):
    """Return eigenvalues and variances that fit together: variances between a typical diagonal and the boundary.

    The variances mix the diagonal of a random matrix with the eigenvalues (``alpha=0``, well inside what's
    possible) with a permutation of the eigenvalues themselves (``alpha=1``, on the boundary).
    """
    rng = np.random.default_rng(seed)
    eigenvalues = rng.gamma(2.0, size=size)
    typical = np.diag(random_covariance_from_eigenvalues(eigenvalues, rng=rng))
    variances = (1 - alpha) * typical + alpha * rng.permutation(eigenvalues)
    # make the totals agree exactly, as the mixture does only to rounding
    return eigenvalues, variances * (np.sum(eigenvalues) / np.sum(variances))


def test_from_eigenvalues_has_the_eigenvalues_and_is_symmetric():
    """The matrix has exactly the given eigenvalues (to rounding), and is exactly symmetric."""
    rng = np.random.default_rng(30)
    for size in [1, 2, 3, 5, 10, 30]:
        eigenvalues = rng.lognormal(size=size)
        matrix = random_covariance_from_eigenvalues(eigenvalues, rng=rng)
        assert matrix.shape == (size, size)
        assert_eigenvalues(matrix, eigenvalues)
        assert_symmetric(matrix)


def test_from_eigenvalues_is_q_diag_qt_with_a_haar_matrix():
    """It's Q @ diag(eigenvalues) @ Q.T, with Q from scipy's ortho_group (Haar), drawn from the given rng."""
    eigenvalues = np.array([1.0, 4.0, 9.0, 16.0])
    q = ortho_group.rvs(4, random_state=np.random.default_rng(31))
    new = random_covariance_from_eigenvalues(eigenvalues, rng=np.random.default_rng(31))
    assert np.allclose(new, q @ np.diag(eigenvalues) @ q.T, rtol=1e-13, atol=1e-13)


def test_from_eigenvalues_variances_lie_between_the_eigenvalues():
    """Each variance lies between the smallest and largest eigenvalue, and they add up to the eigenvalues' sum."""
    rng = np.random.default_rng(32)
    for _ in range(50):
        eigenvalues = rng.lognormal(size=8)
        diagonal = np.diag(random_covariance_from_eigenvalues(eigenvalues, rng=rng))
        assert np.all(diagonal >= eigenvalues.min() - 1e-12) and np.all(diagonal <= eigenvalues.max() + 1e-12)
        assert np.sum(diagonal) == pytest.approx(np.sum(eigenvalues), rel=1e-12)


def test_from_eigenvalues_allows_zero_eigenvalues():
    """Zero eigenvalues give a singular matrix, positive semi-definite to within rounding."""
    matrix = random_covariance_from_eigenvalues([0.0, 0.0, 2.0, 5.0], rng=33)
    assert_eigenvalues(matrix, [0.0, 0.0, 2.0, 5.0])
    assert sorted_eigenvalues(matrix).min() >= -1e-12


def test_random_covariance_functions_are_reproducible_from_a_seed():
    """The same seed gives the same matrix; another seed gives another."""
    eigenvalues, variances = feasible_eigenvalues_and_variances(34, 5, 0.3)
    for make in (
        lambda seed: random_covariance_from_eigenvalues(eigenvalues, rng=seed),
        lambda seed: random_covariance_from_variances(variances, rng=seed),
        lambda seed: random_covariance_from_eigenvalues_and_variances(eigenvalues, variances, rng=seed),
        lambda seed: random_covariance_matrix(5, rng=seed),
    ):
        assert np.array_equal(make(1), make(1))
        assert not np.array_equal(make(1), make(2))


def test_from_variances_has_exactly_the_variances():
    """The diagonal is exactly the given variances; the matrix is exactly symmetric and positive semi-definite."""
    rng = np.random.default_rng(35)
    for size in [1, 2, 3, 5, 10, 30]:
        variances = rng.lognormal(size=size)
        matrix = random_covariance_from_variances(variances, rng=rng)
        assert np.array_equal(np.diag(matrix), variances)
        assert_symmetric(matrix)
        assert sorted_eigenvalues(matrix).min() >= -1e-12 * variances.max()


def test_from_variances_uses_the_correlation_eigenvalues():
    """The correlation matrix has the given eigenvalues, scaled to add up to the number of variables."""
    variances = np.array([1.0, 4.0, 9.0, 0.25])
    matrix = random_covariance_from_variances(variances, correlation_eigenvalues=[4.0, 2.0, 1.0, 1.0], rng=36)
    sd = np.sqrt(variances)
    assert_eigenvalues(matrix / np.outer(sd, sd), np.array([4.0, 2.0, 1.0, 1.0]) * 4 / 8)


def test_from_variances_with_equal_correlation_eigenvalues_is_uncorrelated():
    """Equal correlation eigenvalues give the identity correlation, so a diagonal covariance matrix."""
    variances = np.array([1.0, 4.0, 9.0])
    matrix = random_covariance_from_variances(variances, correlation_eigenvalues=[5.0, 5.0, 5.0], rng=37)
    assert np.allclose(matrix, np.diag(variances), atol=1e-12)


def test_from_variances_default_correlation_eigenvalues_are_scaled_dirichlet_draws():
    """By default the correlation eigenvalues are Dirichlet(1, ..., 1) draws times the number of variables."""
    variances = np.array([2.0, 3.0, 5.0, 7.0, 11.0])
    expected = np.random.default_rng(38).dirichlet(np.ones(5)) * 5
    matrix = random_covariance_from_variances(variances, rng=np.random.default_rng(38))
    sd = np.sqrt(variances)
    assert_eigenvalues(matrix / np.outer(sd, sd), expected)


def test_from_variances_allows_zero_variances():
    """A zero variance gives a zero row and column."""
    matrix = random_covariance_from_variances([0.0, 1.0, 2.0], rng=39)
    assert np.array_equal(matrix[0], [0.0, 0.0, 0.0]) and np.array_equal(matrix[:, 0], [0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    ("correlation_eigenvalues", "message"),
    [([1.0, 1.0], "has 2 values, but there are 3 variables"), ([0.0, 0.0, 0.0], "can't all be 0"),
     ([1.0, -1.0, 3.0], "non-negative")],
)  # fmt: skip
def test_from_variances_rejects_bad_correlation_eigenvalues(correlation_eigenvalues, message):
    """correlation_eigenvalues must have one non-negative value per variable, not all 0."""
    with pytest.raises(ValueError, match=message):
        random_covariance_from_variances([1.0, 2.0, 3.0], correlation_eigenvalues=correlation_eigenvalues)


@pytest.mark.parametrize("size", [2, 3, 5, 10, 20])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 0.9, 0.99, 1.0])
def test_with_eigenvalues_and_variances_has_both(size, alpha):
    """The matrix has the given eigenvalues (to rounding) and exactly the given diagonal, up to the boundary."""
    for seed in range(5):
        eigenvalues, variances = feasible_eigenvalues_and_variances(1000 * size + seed, size, alpha)
        matrix = random_covariance_from_eigenvalues_and_variances(eigenvalues, variances, rng=seed)
        assert np.array_equal(np.diag(matrix), variances)
        assert_symmetric(matrix)
        assert_eigenvalues(matrix, eigenvalues)


def test_with_eigenvalues_and_equal_variances():
    """Equal variances fit any eigenvalues with the same total (a scaled correlation matrix)."""
    rng = np.random.default_rng(40)
    for _ in range(20):
        eigenvalues = rng.lognormal(size=6)
        variances = np.full(6, np.mean(eigenvalues))
        matrix = random_covariance_from_eigenvalues_and_variances(eigenvalues, variances, rng=rng)
        assert np.array_equal(np.diag(matrix), variances)
        assert_eigenvalues(matrix, eigenvalues)


def test_with_eigenvalues_and_variances_on_the_boundary_is_nearly_diagonal():
    """When the variances are a permutation of the eigenvalues, the only such matrix is diagonal."""
    eigenvalues = np.array([5.0, 1.0, 3.0, 2.0])
    matrix = random_covariance_from_eigenvalues_and_variances(eigenvalues, [2.0, 3.0, 5.0, 1.0], rng=41)
    assert np.allclose(matrix, np.diag([2.0, 3.0, 5.0, 1.0]), atol=1e-6)


@pytest.mark.parametrize(
    ("eigenvalues", "variances"),
    [([1.0, 1.0], [2.0, 0.0]), ([3.0, 1.0], [1.0, 1.0]), ([4.0, 1.0, 1.0], [3.5, 2.0, 0.5]), ([2.0], [1.0])],
)
def test_with_eigenvalues_and_variances_rejects_impossible_combinations(eigenvalues, variances):
    """Variances more spread out than the eigenvalues, or a different total, raise ValueError."""
    with pytest.raises(ValueError, match="majorized by the eigenvalues"):
        random_covariance_from_eigenvalues_and_variances(eigenvalues, variances)


def test_with_eigenvalues_and_variances_tolerance_and_one_variable():
    """Totals differing within tol are accepted (the diagonal is still exact); one variable needs equal values."""
    matrix = random_covariance_from_eigenvalues_and_variances([2.0, 1.0], [1.5, 1.5 + 1e-14], rng=42)
    assert np.array_equal(np.diag(matrix), [1.5, 1.5 + 1e-14])
    assert np.array_equal(random_covariance_from_eigenvalues_and_variances([2.0], [2.0]), [[2.0]])


def test_with_eigenvalues_and_variances_rejects_different_lengths():
    """eigenvalues and variances must have the same length."""
    with pytest.raises(ValueError, match="differ in length: 2 and 3"):
        random_covariance_from_eigenvalues_and_variances([1.0, 2.0], [1.0, 1.0, 1.0])


@pytest.mark.parametrize(
    "function",
    [
        random_covariance_from_eigenvalues,
        random_covariance_from_variances,
        lambda values: random_covariance_from_eigenvalues_and_variances(values, values),
    ],
)
@pytest.mark.parametrize(
    ("values", "message"),
    [([], "is empty"), ([1.0, np.nan], "must be finite"), ([1.0, np.inf], "must be finite"),
     ([1.0, -1.0], "must be non-negative"), ([[1.0, 2.0]], "must be 1-D")],
)  # fmt: skip
def test_random_covariance_functions_reject_bad_values(function, values, message):
    """Eigenvalues and variances must be 1-D, non-empty, finite and non-negative."""
    with pytest.raises(ValueError, match=message):
        function(values)


def test_random_covariance_matrix_dispatches_to_the_case_functions():
    """Given eigenvalues, variances or both, it's the corresponding function, with the same rng."""
    eigenvalues, variances = feasible_eigenvalues_and_variances(43, 4, 0.2)
    assert np.array_equal(
        random_covariance_matrix(eigenvalues=eigenvalues, rng=7), random_covariance_from_eigenvalues(eigenvalues, rng=7)
    )
    assert np.array_equal(
        random_covariance_matrix(variances=variances, rng=7), random_covariance_from_variances(variances, rng=7)
    )
    assert np.array_equal(
        random_covariance_matrix(eigenvalues=eigenvalues, variances=variances, rng=7),
        random_covariance_from_eigenvalues_and_variances(eigenvalues, variances, rng=7),
    )


@pytest.mark.parametrize(
    ("arguments", "params"),
    [({}, NormalParams()), ({"lognormal_params_eigenvalues": NormalParams(2.0)}, NormalParams(2.0)),
     ({"lognormal_params_eigenvalues": NormalParams(-1.0, 0.3)}, NormalParams(-1.0, 0.3))],
)  # fmt: skip
def test_random_covariance_matrix_draws_lognormal_eigenvalues(arguments, params):
    """With nothing given, or lognormal_params_eigenvalues, the eigenvalues are the parameters' lognormal draws
    (the default NormalParams() when nothing is given), drawn before the matrix."""
    expected = params.generate_lognormal(6, rng=np.random.default_rng(44))
    assert_eigenvalues(random_covariance_matrix(6, rng=44, **arguments), expected)


@pytest.mark.parametrize("params", [NormalParams(), NormalParams(2.0), NormalParams(-1.0, 0.3)])
def test_random_covariance_matrix_draws_lognormal_variances(params):
    """With lognormal_params_variances, the variances are the parameters' lognormal draws, drawn first."""
    expected = params.generate_lognormal(6, rng=np.random.default_rng(45))
    matrix = random_covariance_matrix(6, lognormal_params_variances=params, rng=45)
    assert np.array_equal(np.diag(matrix), expected)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"eigenvalues": [1.0, 2.0], "lognormal_params_eigenvalues": NormalParams()},
         "Give either eigenvalues or lognormal_params_eigenvalues"),
        ({"variances": [1.0, 2.0], "lognormal_params_variances": NormalParams()},
         "Give either variances or lognormal_params_variances"),
        ({"lognormal_params_eigenvalues": NormalParams(), "variances": [1.0, 2.0]}, "can't be combined"),
        ({"lognormal_params_variances": NormalParams(), "eigenvalues": [1.0, 2.0]}, "can't be combined"),
        ({"size": 2, "lognormal_params_eigenvalues": NormalParams(), "lognormal_params_variances": NormalParams()},
         "can't be combined"),
        ({}, "Give size"),
        ({"lognormal_params_eigenvalues": NormalParams(1.0)}, "Give size"),
        ({"size": 3, "eigenvalues": [1.0, 2.0]}, "must agree on the number of variables"),
        ({"eigenvalues": [1.0, 2.0], "variances": [1.0, 1.0, 1.0]}, "must agree on the number of variables"),
    ],
)  # fmt: skip
def test_random_covariance_matrix_rejects_conflicting_or_missing_arguments(arguments, message):
    """Conflicting options, a generated set with the other kind, or a missing or mismatched size raise ValueError."""
    with pytest.raises(ValueError, match=message):
        random_covariance_matrix(**arguments)


def test_random_covariance_matrix_size_can_be_given_with_values():
    """size may be given along with eigenvalues or variances, if it matches."""
    assert random_covariance_matrix(3, eigenvalues=[1.0, 2.0, 3.0], rng=46).shape == (3, 3)
    assert random_covariance_matrix(3, variances=[1.0, 2.0, 3.0], rng=46).shape == (3, 3)


def test_transfer_stage_reaches_the_targets_within_p_minus_1_rotations(monkeypatch):
    """The second stage alone matches the diagonal to the targets (by rank), with at most p - 1 rotations, and keeps
    the eigenvalues (checked before the final step that sets the diagonal exactly, which could hide a shortfall)."""
    import mcrotk_utils.stats_utils._covariance as covariance_module

    calls = []
    rotate = covariance_module._rotate_to_set_diagonal

    def counting_rotate(*args):
        """Record the rotation, then apply it."""
        calls.append(args)
        rotate(*args)

    monkeypatch.setattr(covariance_module, "_rotate_to_set_diagonal", counting_rotate)
    for size in [2, 3, 5, 10, 20]:
        for seed, alpha in enumerate([0.0, 0.5, 0.9, 0.99]):
            eigenvalues, variances = feasible_eigenvalues_and_variances(2000 * size + seed, size, alpha)
            target = np.sort(variances)[::-1]
            atol = 1e-12 * max(1.0, float(np.sum(eigenvalues)))
            matrix = random_covariance_from_eigenvalues(eigenvalues, rng=seed)
            _spread_diagonal_until_majorizing(matrix, target, atol)
            calls.clear()
            order = _transfer_diagonal_to_target(matrix, target, atol)
            assert len(calls) <= size - 1
            assert np.allclose(np.diag(matrix)[order], target, rtol=0, atol=10 * atol)
            assert_eigenvalues(matrix, eigenvalues)


# --- correlation_from_covariance ---------------------------------------------------------------------------------


def random_covariances(seed, count=300):
    """Yield random covariance matrices of 1 to 8 variables, with variances over many orders of magnitude."""
    rng = np.random.default_rng(seed)
    for _ in range(count):
        size = int(rng.integers(1, 9))
        factors = rng.normal(size=(size, size)) * rng.lognormal(0, 3, size)[:, None]
        yield factors @ factors.T


def test_correlation_is_the_covariance_over_the_standard_deviations():
    """Each entry is the covariance over the product of the two standard deviations (to rounding)."""
    for cov in random_covariances(70):
        sd = np.sqrt(np.diag(cov))
        assert np.allclose(correlation_from_covariance(cov), cov / np.outer(sd, sd), rtol=1e-12, atol=1e-14)


def test_correlation_matches_statsmodels():
    """It agrees with statsmodels' cov2corr on ordinary covariance matrices."""
    moment_helpers = pytest.importorskip("statsmodels.stats.moment_helpers")
    for cov in random_covariances(71, 50):
        assert np.allclose(correlation_from_covariance(cov), moment_helpers.cov2corr(cov), rtol=1e-12, atol=1e-14)


def test_correlation_is_exactly_a_valid_correlation_matrix():
    """The diagonal is exactly 1, the matrix exactly symmetric, and every entry in [-1, 1], even where the plain
    division misses by rounding."""
    misses = 0
    for cov in random_covariances(72):
        correlation = correlation_from_covariance(cov)
        assert np.all(np.diag(correlation) == 1.0)
        assert np.array_equal(correlation, correlation.T)
        assert np.all(np.abs(correlation) <= 1.0)
        sd = np.sqrt(np.diag(cov))
        plain = cov / np.outer(sd, sd)
        misses += int(np.sum(np.diag(plain) != 1.0) + np.sum(np.abs(plain) > 1.0))
    assert misses > 100  # the plain division does miss, so the corrections are exercised


def test_correlation_round_trips_to_the_covariance():
    """np.outer(sd, sd) * correlation gives back the covariance matrix (to rounding)."""
    for cov in random_covariances(73):
        sd = np.sqrt(np.diag(cov))
        assert np.allclose(np.outer(sd, sd) * correlation_from_covariance(cov), cov, rtol=1e-12, atol=1e-12 * cov.max())


def test_correlation_of_a_known_matrix():
    """Variances 4 and 9 with covariance 3 give correlation 0.5; perfectly correlated variables give 1."""
    assert np.array_equal(correlation_from_covariance([[4.0, 3.0], [3.0, 9.0]]), [[1.0, 0.5], [0.5, 1.0]])
    assert np.array_equal(correlation_from_covariance([[1.0, 2.0], [2.0, 4.0]]), [[1.0, 1.0], [1.0, 1.0]])
    assert np.array_equal(correlation_from_covariance([[2.5]]), [[1.0]])


def test_correlation_recovers_the_correlation_used_to_build_a_covariance():
    """For a covariance built from a correlation matrix, it gives back that correlation (its eigenvalues included)."""
    variances = np.array([1.0, 4.0, 9.0, 0.25])
    cov = random_covariance_from_variances(variances, correlation_eigenvalues=[2.0, 1.0, 0.6, 0.4], rng=74)
    assert_eigenvalues(correlation_from_covariance(cov), [2.0, 1.0, 0.6, 0.4])


def test_correlation_with_zero_variances_warns_and_uses_the_convention():
    """Constants get correlation 0 with everything and 1 on the diagonal, with a warning naming them; the result is
    still a valid correlation matrix and round-trips to the covariance."""
    cov = np.array([[4.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 9.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    with pytest.warns(UserWarning, match=r"Variables \[1, 3\] have zero variance"):
        correlation = correlation_from_covariance(cov)
    expected = np.array([[1.0, 0.0, 1 / 3, 0.0], [0.0, 1.0, 0.0, 0.0], [1 / 3, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    assert np.allclose(correlation, expected, rtol=1e-15)
    assert sorted_eigenvalues(correlation).min() >= 0
    sd = np.sqrt(np.diag(cov))
    assert np.array_equal(np.outer(sd, sd) * correlation, np.outer(sd, sd) * expected)
    with pytest.warns(UserWarning, match=r"Variables \[0\]"):
        assert np.array_equal(correlation_from_covariance([[0.0]]), [[1.0]])


def test_correlation_doesnt_warn_without_zero_variances():
    """No warning for ordinary covariance matrices."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        correlation_from_covariance([[4.0, 3.0], [3.0, 9.0]])


def test_correlation_leaves_the_input_unchanged():
    """The input array isn't modified."""
    cov = np.array([[4.0, 3.0], [3.0, 9.0]])
    correlation_from_covariance(cov)
    assert np.array_equal(cov, [[4.0, 3.0], [3.0, 9.0]])


@pytest.mark.parametrize(
    ("covariance", "message"),
    [
        ([1.0, 2.0], "square 2-D array"),
        ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], "square 2-D array"),
        ([[1.0, np.nan], [np.nan, 1.0]], "must be finite"),
        ([[np.inf, 0.0], [0.0, 1.0]], "must be finite"),
        ([[-1.0, 0.0], [0.0, 1.0]], "negative variance"),
        ([[1.0, 0.5], [0.4, 1.0]], "isn't symmetric"),
        ([[1.0, 2.0], [2.0, 1.0]], "larger than the product of the standard deviations"),
        ([[0.0, 0.5], [0.5, 1.0]], "larger than the product of the standard deviations"),
    ],
)
def test_correlation_rejects_matrices_that_arent_covariance_matrices(covariance, message):
    """Non-square, non-finite or asymmetric input, negative variances, or covariances beyond the bound raise."""
    with pytest.raises(ValueError, match=message):
        correlation_from_covariance(covariance)


def test_correlation_accepts_rounding_level_asymmetry_and_excess():
    """Asymmetry or a correlation above 1 at the level of rounding is accepted (and corrected)."""
    cov = np.array([[1.0, 1.0 + 1e-14], [1.0, 1.0]])
    assert np.array_equal(correlation_from_covariance(cov), [[1.0, 1.0], [1.0, 1.0]])
    # rounding-level asymmetry below 1 still gives an exactly symmetric result
    correlation = correlation_from_covariance(np.array([[1.0, 0.5 + 1e-14], [0.5, 1.0]]))
    assert np.array_equal(correlation, correlation.T)
    # a constant with rounding-level covariances still gets exactly 0 correlations
    with pytest.warns(UserWarning, match="zero variance"):
        correlation = correlation_from_covariance(np.array([[0.0, 1e-12], [1e-12, 1.0]]))
    assert np.array_equal(correlation, [[1.0, 0.0], [0.0, 1.0]])
