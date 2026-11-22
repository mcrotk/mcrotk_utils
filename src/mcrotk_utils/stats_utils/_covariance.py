"""Random covariance matrices with given eigenvalues and/or variances, and correlation matrices from covariance
matrices (private).

Public through ``mcrotk_utils.stats_utils``. Needs scipy (and numpy, which scipy brings).
"""

import warnings

try:
    import numpy as np
    import numpy.typing as npt
    from scipy.stats import ortho_group, random_correlation
except ModuleNotFoundError as error:
    # these are optional for the package as a whole, so say which extra provides them, not just that one is missing
    raise ModuleNotFoundError(
        "mcrotk_utils.stats_utils' covariance matrices need scipy: "
        "install mcrotk-utils with its extra, mcrotk-utils[scipy]",
        # the top-level package (e.g. scipy rather than scipy.stats), which is what's installed
        name=error.name.partition(".")[0] if error.name else None,
    ) from error

from mcrotk_utils.numpy_utils import SeedLike
from mcrotk_utils.stats_utils._distributions import NormalParams
from mcrotk_utils.stats_utils._validation import _as_1d_floats

# --- Random covariance matrices ------------------------------------------------------------------------------------
#
# Three constructions: a Haar-random orthogonal matrix for given eigenvalues; a random correlation matrix scaled by
# the standard deviations for given variances; and a generalized Bendel-Mickey algorithm (Givens rotations that fix
# the diagonal while keeping the eigenvalues) for both, possible exactly when the variances are majorized by the
# eigenvalues (the Schur-Horn theorem). References:
#
# - R. B. Bendel and M. R. Mickey, "Population correlation matrices for sampling experiments," Comm. Statist.
#   Simul. Comput. 7 (1978).
# - P. I. Davies and N. J. Higham, "Numerically stable generation of correlation matrices and their factors," BIT
#   40 (2000): the algorithm behind scipy.stats.random_correlation.
# - N. N. Chan and K.-H. Li, "Diagonal elements and eigenvalues of a real symmetric matrix," J. Math. Anal. Appl.
#   91 (1983): a constructive Schur-Horn theorem.
# - A. W. Marshall, I. Olkin and B. C. Arnold, "Inequalities: Theory of Majorization and Its Applications," 2nd
#   ed. (2011), Lemma 2.B.1: the transfers between diagonal entries.
# - F. Mezzadri, "How to generate random matrices from the classical compact groups," Notices AMS 54 (2007): Haar
#   orthogonal matrices.


def random_covariance_matrix(
    size: int | None = None,
    *,
    eigenvalues: npt.ArrayLike | None = None,
    variances: npt.ArrayLike | None = None,
    lognormal_params_eigenvalues: NormalParams | None = None,
    lognormal_params_variances: NormalParams | None = None,
    rng: SeedLike = None,
) -> npt.NDArray[np.float64]:
    """Return a random covariance matrix, with given or random eigenvalues, variances, or both.

    The eigenvalues and the variances (the diagonal) can each be given, generated as lognormal, or left free:

    - ``eigenvalues``: the matrix has exactly these eigenvalues (see ``random_covariance_from_eigenvalues``).
    - ``variances``: the matrix has exactly this diagonal (see ``random_covariance_from_variances``).
    - Both: the matrix has both (see ``random_covariance_from_eigenvalues_and_variances``, which says when that's
      possible).
    - ``lognormal_params_eigenvalues``, instead of ``eigenvalues``: the eigenvalues are drawn as lognormal, with
      these parameters for their logarithm (``NormalParams.generate_lognormal``).
    - ``lognormal_params_variances``, instead of ``variances``: likewise for the variances.
    - None of these: the eigenvalues are drawn as lognormal with the default ``NormalParams()``, i.e. their
      logarithm standard normal.

    Drawing both the eigenvalues and the variances, or drawing one of them while giving the other, raises
    ``ValueError``: once one of them is set, the other is determined as the matrix is generated, and setting both at
    random would almost never give a combination that's possible (see
    ``random_covariance_from_eigenvalues_and_variances``).

    **Arguments.**

    - ``size``: the number of variables (rows and columns). Needed only when neither ``eigenvalues`` nor
      ``variances`` is given; otherwise it must match their length if given.
    - ``eigenvalues``, ``variances``: 1-D arrays or lists of non-negative finite numbers.
    - ``lognormal_params_eigenvalues``, ``lognormal_params_variances``: ``NormalParams``, as above.
    - ``rng``: a seed or ``np.random.Generator`` (anything ``np.random.default_rng`` accepts), so that the result
      can be reproduced. The random values are drawn in this order: lognormal eigenvalues or variances (if
      generated), then the matrix.

    **Result.**

    - A symmetric ``size`` × ``size`` NumPy float array, positive semi-definite (to within rounding).

    **Errors.**

    - ``ValueError``: ``eigenvalues`` given with ``lognormal_params_eigenvalues``, or ``variances`` with
      ``lognormal_params_variances``; a generated set combined with the other kind; no ``size`` when it's needed,
      or one that doesn't match; or anything the function for the case raises.
    """
    generator = np.random.default_rng(rng)
    draw_eigenvalues = _check_given_or_drawn(eigenvalues, lognormal_params_eigenvalues, "eigenvalues")
    draw_variances = _check_given_or_drawn(variances, lognormal_params_variances, "variances")
    if (draw_eigenvalues or draw_variances) and (
        eigenvalues is not None or variances is not None or (draw_eigenvalues and draw_variances)
    ):
        raise ValueError(
            "Random eigenvalues or variances can't be combined with the other kind (given or random): the other is "
            "determined as the matrix is generated, and the two would almost never fit together."
        )
    size = _covariance_size(size, eigenvalues, variances)
    if lognormal_params_variances is not None:
        variances = lognormal_params_variances.generate_lognormal(size, rng=generator)
    elif eigenvalues is None and variances is None:
        # drawn eigenvalues are the default, with the default parameters when none are given
        eigenvalues = (lognormal_params_eigenvalues or NormalParams()).generate_lognormal(size, rng=generator)
    if eigenvalues is not None and variances is not None:
        return random_covariance_from_eigenvalues_and_variances(eigenvalues, variances, rng=generator)
    if eigenvalues is not None:
        return random_covariance_from_eigenvalues(eigenvalues, rng=generator)
    if variances is not None:
        return random_covariance_from_variances(variances, rng=generator)
    raise RuntimeError("Neither eigenvalues nor variances were given or drawn; this shouldn't happen.")


def _check_given_or_drawn(given: npt.ArrayLike | None, params: NormalParams | None, name: str) -> bool:
    """Return whether ``name`` is to be drawn (``params`` given); raise ``ValueError`` if it's also given."""
    if params is not None and given is not None:
        raise ValueError(f"Give either {name} or lognormal_params_{name}, not both.")
    return params is not None


def _covariance_size(size: int | None, *arrays: npt.ArrayLike | None) -> int:
    """Return the number of variables: the given arrays' length, or ``size``; raise ``ValueError`` if they differ."""
    lengths = {len(np.asarray(array)) for array in arrays if array is not None}
    if size is not None:
        lengths.add(size)
    if not lengths:
        raise ValueError("Give size (the number of variables) when neither eigenvalues nor variances is given.")
    if len(lengths) > 1:
        raise ValueError(f"size, eigenvalues and variances must agree on the number of variables, not {lengths}.")
    return lengths.pop()


def random_covariance_from_eigenvalues(eigenvalues: npt.ArrayLike, *, rng: SeedLike = None) -> npt.NDArray[np.float64]:
    """Return a random covariance matrix with exactly the given eigenvalues and uniformly random eigenvectors.

    ``Q @ diag(eigenvalues) @ Q.T`` with ``Q`` a Haar-random (uniformly distributed) orthogonal matrix. The
    variances (the diagonal) aren't controlled: each is a weighted average of the eigenvalues, so it lies between
    the smallest and the largest, and they add up to the eigenvalues' sum.

    **Arguments.**

    - ``eigenvalues``: a 1-D array or list of non-negative finite numbers, one per variable.
    - ``rng``: a seed or ``np.random.Generator``, so that the result can be reproduced.

    **Result.**

    - A symmetric NumPy float array, ``len(eigenvalues)`` square, exactly symmetric.

    **Errors.**

    - ``ValueError``: ``eigenvalues`` is empty, not 1-D, or has a negative, NaN or infinite value.
    """
    eigs = _nonnegative_vector(eigenvalues, "eigenvalues")
    if len(eigs) == 1:
        # scipy's ortho_group needs at least two dimensions (before scipy 1.12); one variable has nothing to rotate
        return eigs.reshape(1, 1).copy()
    q = ortho_group.rvs(len(eigs), random_state=np.random.default_rng(rng))
    # q * eigs scales column i of q by eigenvalue i, leaving one matrix product instead of two
    return _symmetrized((q * eigs) @ q.T)


def random_covariance_from_variances(
    variances: npt.ArrayLike,
    *,
    correlation_eigenvalues: npt.ArrayLike | None = None,
    rng: SeedLike = None,
) -> npt.NDArray[np.float64]:
    """Return a random covariance matrix with exactly the given variances (diagonal).

    ``diag(s) @ R @ diag(s)``, with ``s`` the standard deviations (square roots of the variances) and ``R`` a
    random correlation matrix from ``scipy.stats.random_correlation``. ``R``'s eigenvalues set how strongly the
    variables are correlated (all equal: not at all; very unequal: strongly), and can be given; by default
    they're drawn from a Dirichlet distribution with all parameters 1, scaled to add up to the number of
    variables. The matrix's own eigenvalues aren't controlled; they add up to the variances' sum.

    **Arguments.**

    - ``variances``: a 1-D array or list of non-negative finite numbers, one per variable.
    - ``correlation_eigenvalues``: the eigenvalues for ``R``, non-negative and not all 0, one per variable; they're
      scaled to add up to the number of variables, as a correlation matrix's eigenvalues must.
    - ``rng``: a seed or ``np.random.Generator``, so that the result can be reproduced.

    **Result.**

    - A symmetric NumPy float array, ``len(variances)`` square, exactly symmetric and with exactly the given
      diagonal.

    **Errors.**

    - ``ValueError``: ``variances`` or ``correlation_eigenvalues`` is empty, not 1-D, or has a negative, NaN or
      infinite value; they differ in length; or ``correlation_eigenvalues`` are all 0.
    """
    var = _nonnegative_vector(variances, "variances")
    generator = np.random.default_rng(rng)
    if len(var) == 1:
        # scipy's random_correlation needs at least two variables; one variable's covariance is its variance
        return var.reshape(1, 1).copy()
    correlation = random_correlation.rvs(
        _correlation_eigenvalues(correlation_eigenvalues, len(var), generator),
        random_state=generator,
        # the eigenvalues add up to the number of variables only to rounding, which can exceed the default 1e-13
        tol=1e-8,
    )
    sd = np.sqrt(var)
    # scaling rows and columns elementwise is O(p**2), unlike products with diagonal matrices
    return _with_diagonal(_symmetrized(sd[:, None] * correlation * sd[None, :]), var)


def _correlation_eigenvalues(
    given: npt.ArrayLike | None, size: int, generator: np.random.Generator
) -> npt.NDArray[np.float64]:
    """Return eigenvalues for a random correlation matrix: ``given`` or Dirichlet draws, scaled to sum to ``size``."""
    if given is None:
        eigs = generator.dirichlet(np.ones(size))
    else:
        eigs = _nonnegative_vector(given, "correlation_eigenvalues")
        if len(eigs) != size:
            raise ValueError(f"correlation_eigenvalues has {len(eigs):,} values, but there are {size:,} variables.")
        if not eigs.any():
            raise ValueError("correlation_eigenvalues can't all be 0.")
    return eigs * (size / np.sum(eigs))


def random_covariance_from_eigenvalues_and_variances(
    eigenvalues: npt.ArrayLike, variances: npt.ArrayLike, *, rng: SeedLike = None, tol: float = 1e-12
) -> npt.NDArray[np.float64]:
    """Return a random covariance matrix with exactly the given eigenvalues and variances (diagonal).

    Such a matrix exists only if the variances are majorized by the eigenvalues (the Schur-Horn theorem): both add
    up to the same total, and for every ``k``, the ``k`` largest variances add up to at most the ``k`` largest
    eigenvalues. So the eigenvalues must be at least as spread out as the variances; e.g. equal variances work
    with any eigenvalues that have the same total.

    Starts from ``random_covariance_from_eigenvalues`` and applies Givens rotations, which keep the eigenvalues, to
    set the diagonal: Jacobi rotations first spread the diagonal out until the variances are majorized by it (none
    are needed if they already are), then at most ``len(variances) - 1`` rotations each move part of one diagonal
    entry to another, until it matches the variances (a generalization of the Bendel-Mickey algorithm that
    ``scipy.stats.random_correlation`` uses). The randomness comes from the starting matrix; no particular
    distribution is claimed. Near the boundary of what's possible (variances almost as spread out as the
    eigenvalues), the result is necessarily close to diagonal, and the first stage takes longer.

    **Arguments.**

    - ``eigenvalues``, ``variances``: 1-D arrays or lists of non-negative finite numbers, the same length, with
      the variances majorized by the eigenvalues.
    - ``rng``: a seed or ``np.random.Generator``, so that the result can be reproduced.
    - ``tol``: the relative tolerance for checking the majorization and the totals, as a fraction of the total.

    **Result.**

    - A symmetric NumPy float array, ``len(eigenvalues)`` square, exactly symmetric and with exactly the given
      diagonal; its eigenvalues equal the given ones to within rounding.

    **Errors.**

    - ``ValueError``: either is empty, not 1-D, or has a negative, NaN or infinite value; they differ in length;
      or the variances aren't majorized by the eigenvalues (including different totals).
    """
    eigs = _nonnegative_vector(eigenvalues, "eigenvalues")
    var = _nonnegative_vector(variances, "variances")
    if len(eigs) != len(var):
        raise ValueError(f"eigenvalues and variances differ in length: {len(eigs):,} and {len(var):,}.")
    target = np.sort(var)[::-1]
    atol = tol * max(1.0, float(np.sum(eigs)))
    if abs(np.sum(eigs) - np.sum(var)) > atol or not _majorized(target, eigs, atol):
        raise ValueError(
            "No matrix has these eigenvalues and variances: the variances must be majorized by the eigenvalues "
            "(the same total, and the k largest variances adding up to at most the k largest eigenvalues)."
        )
    matrix = random_covariance_from_eigenvalues(eigs, rng=rng)
    _spread_diagonal_until_majorizing(matrix, target, atol)
    order = _transfer_diagonal_to_target(matrix, target, atol)
    # a permutation similarity puts each variance at its variable's position (the k-th largest variance went to
    # position order[k])
    positions = np.empty(len(var), dtype=np.intp)
    positions[np.argsort(-var, kind="stable")] = order
    return _with_diagonal(_symmetrized(matrix[np.ix_(positions, positions)]), var)


def _majorized(target_sorted: npt.NDArray[np.float64], values: npt.NDArray[np.float64], atol: float) -> bool:
    """Return whether the target (sorted in decreasing order) is majorized by ``values``, to within ``atol``.

    That is, whether each partial sum of the k largest targets is at most that of the k largest values (equal
    totals are checked separately).
    """
    return bool(np.all(np.cumsum(target_sorted) <= np.cumsum(np.sort(values)[::-1]) + atol))


def _spread_diagonal_until_majorizing(
    matrix: npt.NDArray[np.float64], target_sorted: npt.NDArray[np.float64], atol: float
) -> None:
    """In place: apply Jacobi rotations until the target is majorized by the matrix's diagonal.

    Each rotation zeroes the largest off-diagonal entry, replacing the two diagonal entries with the eigenvalues of
    their 2 x 2 block, which are more spread out; in the limit the diagonal is the eigenvalues, which majorize the
    target, so this ends.
    """
    # a full Jacobi diagonalization needs a few sweeps of p**2 / 2 rotations; far more means something's wrong
    max_rotations = 50 * len(matrix) ** 2
    for _ in range(max_rotations):
        if _majorized(target_sorted, np.diag(matrix), atol):
            return
        off_diagonal = np.abs(np.triu(matrix, 1))
        i, j = (int(index) for index in np.unravel_index(np.argmax(off_diagonal), off_diagonal.shape))
        middle = (matrix[i, i] + matrix[j, j]) / 2
        radius = np.hypot((matrix[i, i] - matrix[j, j]) / 2, matrix[i, j])
        # the larger of the two entries moves to the block's larger eigenvalue, the smaller to its smaller one
        _rotate_to_set_diagonal(matrix, i, j, middle + radius if matrix[i, i] >= matrix[j, j] else middle - radius)
    raise RuntimeError("The diagonal didn't come to majorize the variances; this shouldn't happen.")


def _transfer_diagonal_to_target(
    matrix: npt.NDArray[np.float64], target_sorted: npt.NDArray[np.float64], atol: float
) -> npt.NDArray[np.intp]:
    """In place: move amounts between diagonal entries until, by rank, they equal the target; return the ranking.

    Returns the positions of the diagonal entries in decreasing order (the k-th one now holds the k-th largest
    target). Each step (a "Robin Hood" transfer, Marshall-Olkin Lemma 2.B.1) takes the last rank whose entry
    exceeds its target and the first later rank whose entry falls short, and moves the smaller of the excess and
    the shortfall from one to the other with a Givens rotation; each step fixes at least one entry, so at most
    p - 1 are needed.
    """
    order = np.argsort(-np.diag(matrix), kind="stable")
    for _ in range(len(matrix) - 1):
        gap = np.diag(matrix)[order] - target_sorted
        over = np.nonzero(gap > atol)[0]
        if over.size == 0:
            break
        j = over[-1]
        # majorization guarantees a shortfall after the last excess
        k = j + 1 + np.nonzero(gap[j + 1 :] < -atol)[0][0]
        delta = min(float(gap[j]), float(-gap[k]))
        source, destination = int(order[j]), int(order[k])
        _rotate_to_set_diagonal(matrix, source, destination, float(matrix[source, source]) - delta)
    return order


def _rotate_to_set_diagonal(matrix: npt.NDArray[np.float64], i: int, j: int, value: float) -> None:
    """In place: apply a Givens rotation in the (i, j) plane so that ``matrix[i, i]`` becomes ``value``.

    The rotation keeps the eigenvalues and ``matrix[i, i] + matrix[j, j]``. ``value`` must lie between the
    eigenvalues of the 2 x 2 block (always true for the callers). With ``t = tan(angle)``, the angle solves
    ``(a_jj - value) t**2 - 2 a_ij t + (a_ii - value) = 0``; the smaller root is taken in a form that avoids
    cancellation.
    """
    a_ii, a_jj, a_ij = matrix[i, i], matrix[j, j], matrix[i, j]
    # the discriminant is non-negative in exact arithmetic; rounding can make it slightly negative
    discriminant = max(a_ij * a_ij - (a_jj - value) * (a_ii - value), 0.0)
    denominator = a_ij + np.copysign(np.sqrt(discriminant), a_ij)
    t = 0.0 if denominator == 0.0 else (a_ii - value) / denominator
    c = 1.0 / np.sqrt(1.0 + t * t)
    s = c * t
    column_i, column_j = matrix[:, i].copy(), matrix[:, j].copy()
    matrix[:, i], matrix[:, j] = c * column_i - s * column_j, s * column_i + c * column_j
    row_i, row_j = matrix[i, :].copy(), matrix[j, :].copy()
    matrix[i, :], matrix[j, :] = c * row_i - s * row_j, s * row_i + c * row_j


def _nonnegative_vector(values: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    """Return ``values`` as a 1-D float array, raising ``ValueError`` unless it's non-empty, finite and non-negative."""
    vector = _as_1d_floats(values, name)
    if len(vector) == 0:
        raise ValueError(f"{name} is empty.")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must be finite (not NaN or infinite).")
    if (vector < 0).any():
        raise ValueError(f"{name} must be non-negative.")
    return vector


def _symmetrized(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return ``(matrix + matrix.T) / 2``: products are symmetric only to rounding, and this makes them exactly so."""
    return (matrix + matrix.T) / 2


def _with_diagonal(matrix: npt.NDArray[np.float64], diagonal: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return ``matrix`` with its diagonal set to ``diagonal`` exactly (in place), correcting rounding-level errors."""
    np.fill_diagonal(matrix, diagonal)
    return matrix


# --- Correlation from covariance ---------------------------------------------------------------------------------

# Relative tolerances for checking a covariance matrix: symmetry, and correlations within [-1, 1], allowing for
# rounding but not for a matrix that isn't a covariance matrix
_SYMMETRY_RTOL = 1e-10
_CORRELATION_EXCESS_RTOL = 1e-8


def correlation_from_covariance(covariance: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return the correlation matrix of a covariance matrix: each covariance over the two standard deviations.

    The result is a valid correlation matrix, exactly: its diagonal is exactly 1, it's exactly symmetric, and its
    entries lie in [-1, 1] (the division alone can miss these by rounding, e.g. give 1.0000000000000002, which makes
    ``sqrt(1 - r**2)`` NaN).

    A variable with zero variance (a constant) has no defined correlation with anything (0 / 0). By convention, its
    correlations with the other variables are set to 0, and with itself to 1, with a ``UserWarning`` naming such
    variables. That keeps the result a valid correlation matrix (as if the constant were replaced by an independent
    variable), consistent with its covariances, which are 0; and ``np.outer(sd, sd) * correlation`` still gives back
    the covariance matrix, since the constant's standard deviation of 0 zeroes its row and column.

    **Arguments.**

    - ``covariance``: a square 2-D array (or nested list) of finite numbers: symmetric (to within rounding), with
      non-negative variances on the diagonal, and no covariance larger in size than the two standard deviations'
      product (to within rounding).

    **Result.**

    - A NumPy float array of the same shape. The input isn't changed.

    **Warnings.**

    - ``UserWarning``: some variables have zero variance (see above).

    **Errors.**

    - ``ValueError``: ``covariance`` isn't a square 2-D array, has a NaN or infinite entry or a negative variance,
      isn't symmetric, or has a covariance larger in size than the two standard deviations' product (so it isn't a
      covariance matrix).
    """
    cov = _checked_covariance(covariance)
    sd = np.sqrt(np.diag(cov))
    constant = sd == 0
    _check_cauchy_schwarz(cov, sd)
    if constant.any():
        warnings.warn(
            f"Variables {np.flatnonzero(constant).tolist()} have zero variance; their correlations are undefined and "
            "set to 0 (and 1 on the diagonal).",
            UserWarning,
            stacklevel=2,
        )
    # dividing by 1 instead of 0 for constants avoids 0/0; their rows and columns are then set to 0 explicitly
    divisor_sd = np.where(constant, 1.0, sd)
    correlation = _symmetrized(cov / np.outer(divisor_sd, divisor_sd))
    correlation[constant, :] = 0.0
    correlation[:, constant] = 0.0
    # the division can overshoot [-1, 1] by rounding (larger excesses were rejected above)
    np.clip(correlation, -1.0, 1.0, out=correlation)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def _checked_covariance(covariance: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return ``covariance`` as a float array, raising ``ValueError`` unless it's square, finite, symmetric (to
    within rounding) and has non-negative variances."""
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"covariance must be a square 2-D array, not one of shape {cov.shape}.")
    if not np.isfinite(cov).all():
        raise ValueError("covariance must be finite (no NaN or infinite entries).")
    if (np.diag(cov) < 0).any():
        raise ValueError("covariance has a negative variance on its diagonal.")
    scale = float(np.max(np.abs(cov), initial=0.0))
    if not np.allclose(cov, cov.T, rtol=0.0, atol=_SYMMETRY_RTOL * scale):
        raise ValueError("covariance isn't symmetric.")
    return cov


def _check_cauchy_schwarz(cov: npt.NDArray[np.float64], sd: npt.NDArray[np.float64]) -> None:
    """Raise ``ValueError`` if a covariance exceeds the product of the two standard deviations beyond rounding.

    A covariance matrix satisfies ``|cov[j, k]| <= sd[j] * sd[k]`` (so correlations lie in [-1, 1], and a constant
    has zero covariances); a clear excess means the matrix isn't one.
    """
    bound = np.outer(sd, sd)
    scale = float(np.max(np.abs(cov), initial=0.0))
    if (np.abs(cov) > bound * (1 + _CORRELATION_EXCESS_RTOL) + _CORRELATION_EXCESS_RTOL * scale).any():
        raise ValueError("covariance has a covariance larger than the product of the standard deviations.")
