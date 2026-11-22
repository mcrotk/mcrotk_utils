"""Bisection to where a monotonic function crosses a target value (private).

Public through ``mcrotk_utils.misc_utils``. Uses only the standard library.
"""

import math
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, overload

from mcrotk_utils.misc_utils._checks import _is_nan, check_nonnegative
from mcrotk_utils.misc_utils._functions import identity

type StopReason = Literal["exact", "value_tol", "arg_tol", "precision", "max_iter"]


@dataclass(frozen=True)
class BisectionPoint[A: float, R]:
    """A point where the function has been evaluated in ``bisect_to_target``.

    - ``arg``: the argument.
    - ``result``: what the function returned for it (e.g. an expensive solution).
    - ``value``: the number compared with the target, computed from ``result``.
    """

    arg: A
    result: R
    value: float


@dataclass(frozen=True)
class BisectionResult[A: float, R]:
    """The result of ``bisect_to_target``: the two points closest to the target, one on each side of it.

    - ``target``: the target value.
    - ``below``: the evaluated point with ``value <= target`` closest to the crossing.
    - ``above``: the evaluated point with ``value >= target`` closest to the crossing.
    - ``iterations``: how many times the function was evaluated, not counting the two ends.
    - ``stop_reason``: why the search stopped: ``"exact"`` (a point's value equals the target; ``below`` and
      ``above`` are then that point), ``"value_tol"`` or ``"arg_tol"`` (the values or arguments are within the
      tolerance), ``"precision"`` (no argument lies between the two points: adjacent floats, or adjacent integers
      in integer mode), or ``"max_iter"`` (the iteration limit was reached).
    - ``closest`` (property): whichever of ``below`` and ``above`` has the value nearer the target (``below`` on a
      tie).
    """

    target: float
    below: BisectionPoint[A, R]
    above: BisectionPoint[A, R]
    iterations: int
    stop_reason: StopReason

    @property
    def closest(self) -> BisectionPoint[A, R]:
        """Return whichever of ``below`` and ``above`` has the value nearer the target (``below`` on a tie)."""
        if self.target - self.below.value <= self.above.value - self.target:
            return self.below
        return self.above


# Python ints at both ends give integer mode; anything else gives float mode. Overloads (rather than one type
# parameter for both ends) type mixed ends, such as 0 and 10.5, as float mode, as they run.
@overload
def bisect_to_target(
    func: Callable[[int], float],
    target: float,
    low: int | BisectionPoint[int, float],
    high: int | BisectionPoint[int, float],
    *,
    value_tol: float | None = None,
    arg_tol: float | None = None,
    max_iter: int | None = None,
) -> BisectionResult[int, float]: ...
@overload
def bisect_to_target(
    func: Callable[[float], float],
    target: float,
    low: float | BisectionPoint[float, float],
    high: float | BisectionPoint[float, float],
    *,
    value_tol: float | None = None,
    arg_tol: float | None = None,
    max_iter: int | None = None,
) -> BisectionResult[float, float]: ...
@overload
def bisect_to_target[R](
    func: Callable[[int], R],
    target: float,
    low: int | BisectionPoint[int, R],
    high: int | BisectionPoint[int, R],
    *,
    value: Callable[[R], float],
    value_tol: float | None = None,
    arg_tol: float | None = None,
    max_iter: int | None = None,
) -> BisectionResult[int, R]: ...
@overload
def bisect_to_target[R](
    func: Callable[[float], R],
    target: float,
    low: float | BisectionPoint[float, R],
    high: float | BisectionPoint[float, R],
    *,
    value: Callable[[R], float],
    value_tol: float | None = None,
    arg_tol: float | None = None,
    max_iter: int | None = None,
) -> BisectionResult[float, R]: ...
def bisect_to_target(
    func: Callable[[Any], Any],
    target: float,
    low: Any,
    high: Any,
    *,
    value: Callable[[Any], float] = identity,
    value_tol: float | None = None,
    arg_tol: float | None = None,
    max_iter: int | None = None,
) -> BisectionResult[Any, Any]:
    """Find where a monotonic function crosses a target value, by bisection, returning the points on each side.

    Each step evaluates ``func`` midway between the closest points found so far below and above the target, and
    keeps the new point in place of the one on its side. So ``below.value <= target <= above.value`` throughout.

    **What it does that scipy's root finders don't.** ``scipy.optimize.bisect`` and ``brentq`` find where a
    continuous function is zero (``brentq`` faster, for smooth functions), stopping when the argument is known
    closely enough. This function also:

    - stops when the function's *values* are within a tolerance (``value_tol``), not only its arguments;
    - carries an expensive intermediate result along: ``func`` returns anything (e.g. a fitted model or an
      optimization's solution), ``value`` extracts the number from it, and the results of the final points are
      returned, so nothing needs recomputing;
    - returns both sides of the target, so the caller knows which side each point is on (e.g. "the largest
      setting whose cost doesn't exceed the budget" is ``below`` for an increasing cost);
    - can resume or refine a search from points already evaluated (pass a result's ``below`` and ``above`` as
      ``low`` and ``high``), and accepts already-evaluated ends;
    - reports why it stopped, rather than raising at the iteration limit (which would lose the work done);
    - bisects over integers when both ends are ints (e.g. the smallest ``n`` with ``f(n) >= target``);
    - works on functions with jumps: it always ends, at the latest when no argument is left between the points.

    **Arguments.**

    - ``func``: the function, of one argument. It should be monotonic (increasing or decreasing) between ``low``
      and ``high``; if it isn't, the search finds *a* crossing of the target, not necessarily the only one.
    - ``target``: the value to find.
    - ``low``, ``high``: the ends of the interval to search, in either order: numbers, or points already
      evaluated (``BisectionPoint``, e.g. from a previous result), which aren't evaluated again. Ints at both ends
      give integer mode: only integer arguments are tried. Otherwise the ends must be finite.
    - ``value``: computes the number to compare with the target from what ``func`` returns. The default,
      ``identity``, is for functions returning that number directly.
    - ``value_tol``: stop when ``above.value - below.value <= value_tol``. ``None`` (the default) for no value
      tolerance.
    - ``arg_tol``: stop when the distance between ``below.arg`` and ``above.arg`` is at most ``arg_tol``.
      ``None`` (the default) for no argument tolerance. With neither tolerance, the search continues until no
      argument is left between the points (full float precision, or adjacent integers).
    - ``max_iter``: the most evaluations of ``func`` to make, not counting the ends; ``None`` (the default) for no
      limit (the search ends anyway: after at most about 2,100 steps for floats).

    **Result.**

    - A ``BisectionResult``: ``below`` and ``above`` (each a ``BisectionPoint`` with ``arg``, ``result`` and
      ``value``), ``iterations``, ``stop_reason``, and the ``closest`` of the two points. The tolerances are
      checked before each evaluation, so the ends alone can satisfy them (``iterations`` is then 0).

    **Errors.**

    - ``ValueError``: the target doesn't lie between the ends' values (inclusive), or the ends' arguments are
      equal (unless a value equals the target) or not finite; ``func`` gives a NaN value; a tolerance is negative
      or NaN, or ``max_iter`` is negative.
    """
    _check_bisection_options(value_tol, arg_tol, max_iter)
    first, second = (_evaluated(end, func, value) for end in (low, high))
    for point in (first, second):
        if point.value == target:
            return BisectionResult(target, point, point, 0, "exact")
    below, above = _sides(first, second, target)
    integer_mode = isinstance(below.arg, int) and isinstance(above.arg, int)
    iterations = 0
    while True:
        stop_reason = _tolerance_reached(below, above, value_tol, arg_tol)
        if stop_reason is None and max_iter is not None and iterations >= max_iter:
            stop_reason = "max_iter"
        middle_arg = None if stop_reason else _midpoint(below.arg, above.arg, integer_mode)
        if middle_arg is None:
            return BisectionResult(target, below, above, iterations, stop_reason or "precision")
        middle = _evaluated(middle_arg, func, value)
        iterations += 1
        if middle.value == target:
            return BisectionResult(target, middle, middle, iterations, "exact")
        if middle.value < target:
            below = middle
        else:
            above = middle


def _check_bisection_options(value_tol: float | None, arg_tol: float | None, max_iter: int | None) -> None:
    """Raise ``ValueError`` for a negative or NaN tolerance or a negative ``max_iter``."""
    if value_tol is not None:
        check_nonnegative(value_tol, "value_tol")
    if arg_tol is not None:
        check_nonnegative(arg_tol, "arg_tol")
    if max_iter is not None:
        check_nonnegative(operator.index(max_iter), "max_iter")


def _evaluated(
    end: float | BisectionPoint[Any, Any], func: Callable[[Any], Any], value: Callable[[Any], float]
) -> BisectionPoint[Any, Any]:
    """Return ``end`` as an evaluated point: as it is if it's one already, otherwise by calling ``func``."""
    if isinstance(end, BisectionPoint):
        point = end
    else:
        result = func(end)
        point = BisectionPoint(end, result, value(result))
    if _is_nan(point.value):
        raise ValueError(f"The function's value at {point.arg} is NaN.")
    return point


def _sides(
    first: BisectionPoint[Any, Any], second: BisectionPoint[Any, Any], target: float
) -> tuple[BisectionPoint[Any, Any], BisectionPoint[Any, Any]]:
    """Return the two ends as (below, above) the target, raising ``ValueError`` if they don't bracket it."""
    for point in (first, second):
        if not isinstance(point.arg, int) and not math.isfinite(point.arg):
            raise ValueError(f"The ends' arguments must be finite, not {point.arg}.")
    if first.arg == second.arg:
        raise ValueError(f"The ends' arguments must differ, but both are {first.arg}.")
    below, above = sorted((first, second), key=lambda point: point.value)
    if not below.value <= target <= above.value:
        raise ValueError(
            f"The target ({target}) must lie between the ends' values ({below.value} and {above.value}), "
            "so that the function crosses it."
        )
    return below, above


def _tolerance_reached(
    below: BisectionPoint[Any, Any], above: BisectionPoint[Any, Any], value_tol: float | None, arg_tol: float | None
) -> StopReason | None:
    """Return which tolerance the two points are within, if any (the value tolerance first)."""
    if value_tol is not None and above.value - below.value <= value_tol:
        return "value_tol"
    if arg_tol is not None and abs(above.arg - below.arg) <= arg_tol:
        return "arg_tol"
    return None


def _midpoint(a: float, b: float, integer_mode: bool) -> float | None:
    """Return an argument strictly between ``a`` and ``b``, or ``None`` if there is none (adjacent values)."""
    low, high = min(a, b), max(a, b)
    if integer_mode:
        return (low + high) // 2 if high - low > 1 else None
    middle = low + (high - low) / 2
    if not math.isfinite(middle):
        # high - low overflowed (ends near opposite limits of the float range); halving each first can't
        middle = low / 2 + high / 2
    return middle if low < middle < high else None
