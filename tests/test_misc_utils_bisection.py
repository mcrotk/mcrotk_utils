"""Tests for ``mcrotk_utils.misc_utils``' bisection: ``bisect_to_target`` and its result classes."""

import math
import sys

import pytest

from mcrotk_utils.misc_utils import BisectionPoint, BisectionResult, bisect_to_target

NAN = float("nan")


class Counted:
    """Wrap a function, recording the arguments it's called with."""

    def __init__(self, func):
        """Wrap ``func``."""
        self.func = func
        self.args = []

    def __call__(self, arg):
        """Record ``arg`` and return the function's result."""
        self.args.append(arg)
        return self.func(arg)


def check_bracket(result):
    """Assert the invariants: below.value <= target <= above.value, and the values are the function's."""
    assert result.below.value <= result.target <= result.above.value


def test_bisection_with_argument_tolerance():
    """Stops once the arguments are within arg_tol, with the crossing between them."""
    result = bisect_to_target(lambda x: x**3, 2.0, 0.0, 2.0, arg_tol=1e-9)
    check_bracket(result)
    assert result.stop_reason == "arg_tol"
    assert abs(result.above.arg - result.below.arg) <= 1e-9
    assert result.below.arg <= 2 ** (1 / 3) <= result.above.arg


def test_bisection_with_value_tolerance():
    """Stops once the values are within value_tol."""
    result = bisect_to_target(lambda x: 1000 * x, 500.5, 0.0, 1.0, value_tol=0.01)
    check_bracket(result)
    assert result.stop_reason == "value_tol"
    assert result.above.value - result.below.value <= 0.01


def test_bisection_without_tolerances_goes_to_float_precision():
    """With no tolerances, the search ends with adjacent floats around the crossing."""
    result = bisect_to_target(lambda x: x**3, 2.0, 0.0, 2.0)
    check_bracket(result)
    assert result.stop_reason in ("precision", "exact")
    if result.stop_reason == "precision":
        assert math.nextafter(result.below.arg, math.inf) == result.above.arg


def test_bisection_ends_on_functions_with_jumps():
    """A jump bigger than the value tolerance still ends: at the jump, with adjacent floats."""
    result = bisect_to_target(math.floor, 2.5, 0.0, 10.0, value_tol=0.1)
    assert result.stop_reason == "precision"
    assert (result.below.value, result.above.value) == (2, 3)
    assert result.above.arg == 3.0
    assert result.below.arg == math.nextafter(3.0, 0.0)


def test_bisection_ends_within_about_2100_steps():
    """The worst case for floats (the whole float range, jumping at the smallest positive float) ends in time."""
    largest = sys.float_info.max
    result = bisect_to_target(lambda x: float(x >= 5e-324), 0.5, -largest, largest)
    assert result.stop_reason == "precision"
    assert (result.below.arg, result.above.arg) == (0.0, 5e-324)
    assert result.iterations <= 2100


def test_bisection_of_a_decreasing_function():
    """For a decreasing function, below is the point with the larger argument."""
    result = bisect_to_target(lambda x: -x, -0.3, 0.0, 1.0, arg_tol=1e-6)
    check_bracket(result)
    assert result.below.arg > result.above.arg
    assert result.above.arg <= 0.3 <= result.below.arg


def test_bisection_ends_in_either_order():
    """The ends can be given in either order, with the same result."""
    forward = bisect_to_target(lambda x: x**2, 2.0, 0.0, 2.0, arg_tol=1e-6)
    backward = bisect_to_target(lambda x: x**2, 2.0, 2.0, 0.0, arg_tol=1e-6)
    assert (forward.below, forward.above) == (backward.below, backward.above)


def test_bisection_exact_hit_at_an_end():
    """A target equal to an end's value stops at once, with that end on both sides."""
    func = Counted(lambda x: x)
    result = bisect_to_target(func, 0.0, 0.0, 1.0)
    assert result.stop_reason == "exact"
    assert result.below is result.above
    assert result.below.arg == 0.0
    assert result.iterations == 0


def test_bisection_exact_hit_in_the_middle():
    """A point whose value equals the target stops the search, on both sides."""
    result = bisect_to_target(lambda x: x, 1.0, 0.0, 2.0)
    assert result.stop_reason == "exact"
    assert result.below is result.above
    assert (result.below.arg, result.iterations) == (1.0, 1)


def test_bisection_iteration_limit():
    """At max_iter evaluations, the search returns a result (not an error) saying so."""
    func = Counted(lambda x: x**3)
    result = bisect_to_target(func, 2.0, 0.0, 2.0, max_iter=5)
    check_bracket(result)
    assert (result.stop_reason, result.iterations) == ("max_iter", 5)
    assert len(func.args) == 2 + 5


def test_bisection_tolerances_can_hold_at_the_ends():
    """The tolerances are checked before each evaluation, so the ends alone can satisfy them."""
    result = bisect_to_target(lambda x: x, 0.5, 0.0, 1.0, value_tol=2.0)
    assert (result.stop_reason, result.iterations) == ("value_tol", 0)


def test_bisection_carries_results():
    """func can return anything; value extracts the number; the final points keep their results."""
    func = Counted(lambda x: {"setting": x, "cost": 100 * x})
    result = bisect_to_target(func, 37.0, 0.0, 1.0, value=lambda r: r["cost"], value_tol=0.5)
    check_bracket(result)
    assert result.below.result == {"setting": result.below.arg, "cost": result.below.value}
    assert result.below.value == result.below.result["cost"]


def test_bisection_resumes_from_evaluated_points():
    """A previous result's points can be passed as the ends: they aren't evaluated again."""
    func = Counted(lambda x: x**3)
    first = bisect_to_target(func, 2.0, 0.0, 2.0, arg_tol=1e-2)
    calls_before = len(func.args)
    refined = bisect_to_target(func, 2.0, first.below, first.above, arg_tol=1e-8)
    check_bracket(refined)
    assert len(func.args) - calls_before == refined.iterations
    assert first.below.arg <= refined.below.arg <= refined.above.arg <= first.above.arg
    assert abs(refined.above.arg - refined.below.arg) <= 1e-8


def test_bisection_accepts_evaluated_ends():
    """An end given as a BisectionPoint is used as it is, without calling func on it."""
    func = Counted(lambda x: x)
    bisect_to_target(func, 0.5, BisectionPoint(0.0, 0.0, 0.0), 1.0, arg_tol=0.1)
    assert 0.0 not in func.args


def test_bisection_integer_mode():
    """Int ends give integer arguments only, ending at adjacent integers: e.g. the smallest n with n**2 >= 50."""
    func = Counted(lambda n: n * n)
    result = bisect_to_target(func, 50, 0, 100)
    assert (result.below.arg, result.above.arg, result.stop_reason) == (7, 8, "precision")
    assert all(type(arg) is int for arg in func.args)


def test_bisection_mixed_ends_use_floats():
    """An int and a float end give float mode."""
    func = Counted(lambda x: x * x)
    bisect_to_target(func, 50, 0, 10.5, max_iter=3)
    assert any(type(arg) is float for arg in func.args[2:])


def test_bisection_of_a_non_monotonic_function_finds_a_crossing():
    """For a function that isn't monotonic, the result still brackets a crossing of the target."""
    result = bisect_to_target(math.sin, 0.0, 1.0, 30.0, arg_tol=1e-9)
    check_bracket(result)
    crossing = round(result.below.arg / math.pi) * math.pi
    assert abs(result.below.arg - crossing) < 1e-8


def test_bisection_handles_the_whole_float_range():
    """Ends at the limits of the float range don't overflow the midpoint."""
    largest = sys.float_info.max
    result = bisect_to_target(lambda x: x, 1.0, -largest, largest, max_iter=5000)
    assert result.stop_reason == "exact"


def test_bisection_closest():
    """closest is the side whose value is nearer the target, below on a tie."""
    below, above = BisectionPoint(0.0, None, 1.0), BisectionPoint(1.0, None, 4.0)
    assert BisectionResult(2.0, below, above, 0, "max_iter").closest is below
    assert BisectionResult(3.0, below, above, 0, "max_iter").closest is above
    assert BisectionResult(2.5, below, above, 0, "max_iter").closest is below


@pytest.mark.parametrize(
    ("low", "high", "target", "message"),
    [
        (0.0, 1.0, 5.0, "must lie between the ends' values"),
        (0.0, 1.0, -1.0, "must lie between the ends' values"),
        (1.0, 1.0, 0.5, "must differ"),
        (0.0, math.inf, 0.5, "must be finite"),
        (NAN, 1.0, 0.5, "NaN"),
    ],
)
def test_bisection_errors_for_bad_ends(low, high, target, message):
    """The ends must differ, be finite, and have values on either side of the target."""
    with pytest.raises(ValueError, match=message):
        bisect_to_target(lambda x: x, target, low, high)


def test_bisection_error_for_nan_values():
    """A NaN value from the function raises ValueError."""
    with pytest.raises(ValueError, match="NaN"):
        bisect_to_target(lambda x: NAN if 0.4 < x < 0.6 else x, 0.7, 0.0, 1.0)


@pytest.mark.parametrize(
    "options", [{"value_tol": -1.0}, {"arg_tol": -1.0}, {"value_tol": NAN}, {"arg_tol": NAN}, {"max_iter": -1}]
)
def test_bisection_errors_for_bad_options(options):
    """Negative or NaN tolerances, and a negative max_iter, raise ValueError."""
    with pytest.raises(ValueError):
        bisect_to_target(lambda x: x, 0.5, 0.0, 1.0, **options)
