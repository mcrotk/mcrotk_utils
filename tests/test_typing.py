"""Checks of types, for the type checkers (pyright and mypy) as well as at runtime.

``assert_type`` does nothing at runtime; the type checkers fail if the type differs.
"""

import inspect
import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from typing import Any, assert_type

import matplotlib

matplotlib.use("Agg")  # before pyplot: no windows

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.sparse
from matplotlib.axes import Axes
from matplotlib.text import Annotation

from mcrotk_utils.misc_utils import (
    BisectionPoint,
    BisectionResult,
    bisect_to_target,
    check_in_range,
    check_nonnegative,
    check_not_none,
    check_positive,
    identity,
    sort_with_directions,
)
from mcrotk_utils.multiprocessing_utils import (
    Batch,
    allocate_workers,
    parallel_map_with_rngs,
    split_evenly,
    split_for_workers,
)
from mcrotk_utils.numpy_utils import (
    ComparisonMeasure,
    ComparisonReference,
    NDArrayFloat,
    NDArrayInt,
    SeedLike,
    compare_along,
    interleave,
)
from mcrotk_utils.plotting_utils import BarLabelMeasure, BarLabelPosition, BarLabelReference, label_bars
from mcrotk_utils.sparse_utils import (
    PsMatrix,
    SparseArray,
    SparseFormat,
    as_dense,
    ps_block,
    ps_hstack,
    ps_vstack,
)


def make_rng(rng: SeedLike = None) -> np.random.Generator:
    """Pass a seed on to default_rng: type checkers verify that SeedLike fits default_rng's signature."""
    return np.random.default_rng(rng)


def test_seed_like_accepts_what_default_rng_accepts():
    """Every kind of seed in SeedLike works with default_rng."""
    seeds: list[SeedLike] = [
        None,
        42,
        [1, 2, 3],
        np.array([1, 2, 3]),
        np.random.SeedSequence(42),
        np.random.PCG64(42),
        np.random.default_rng(42),
    ]
    for seed in seeds:
        assert isinstance(make_rng(seed), np.random.Generator)


def test_array_aliases_accept_any_width():
    """NDArrayInt and NDArrayFloat match arrays of any integer or float width."""
    ints: list[NDArrayInt] = [np.zeros(2, dtype=np.int32), np.zeros(2, dtype=np.int64)]
    floats: list[NDArrayFloat] = [np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float64)]
    assert len(interleave(*ints)) == 4
    assert len(interleave(*floats)) == 4


def test_range_checks_keep_the_type_of_any_comparable_number():
    """The range checks return the value's own type, for ints, floats, Fractions and NumPy scalars."""
    assert_type(check_in_range(3, "n", min_value=1), int)
    assert_type(check_positive(2.5, "x"), float)
    assert_type(check_nonnegative(Fraction(1, 3), "x"), Fraction)
    assert_type(check_positive(np.int64(3), "n"), np.int64)
    assert_type(check_in_range(np.float64(0.5), "p", min_value=0, max_value=1), np.float64)


def test_check_not_none_narrows_optional_types():
    """check_not_none turns ``X | None`` into ``X`` for type checkers."""
    source = inspect.getsourcefile(json)
    assert_type(source, str | None)
    assert_type(check_not_none(source, "json's source file"), str)


def test_identity_keeps_the_type():
    """identity is generic: the result has its argument's type."""
    assert_type(identity(3), int)
    assert_type(identity("a"), str)


def test_sort_with_directions_keeps_the_item_type():
    """The result is a list of the items' type, with one direction or one per entry; one direction takes any key."""
    words = ["pear", "fig"]
    assert_type(sort_with_directions(words, key=lambda w: (len(w), w), reverse=[False, True]), list[str])
    assert_type(sort_with_directions(iter([3, 1]), key=lambda n: (n,), reverse=(True,)), list[int])
    assert_type(sort_with_directions(words, key=len, reverse=True), list[str])
    assert_type(sort_with_directions(words, key=lambda w: w), list[str])


def square(n: int) -> int:
    """Square an int (typed, for the type checkers)."""
    return n * n


def square_float(x: float) -> float:
    """Square a float (typed, for the type checkers)."""
    return x * x


def cost_report(x: float) -> dict[str, float]:
    """Return an expensive-looking result containing a cost."""
    return {"setting": x, "cost": 100 * x}


def test_bisection_result_types():
    """Int ends give int arguments; float or mixed ends give floats; value= gives the function's result type."""
    integer = bisect_to_target(square, 50, 0, 100)
    assert_type(integer, BisectionResult[int, float])
    assert type(integer.below.arg) is int
    assert_type(bisect_to_target(square_float, 50, 0.0, 10.0), BisectionResult[float, float])
    mixed = bisect_to_target(square_float, 50, 0, 10.5)
    assert_type(mixed, BisectionResult[float, float])
    assert type(mixed.below.arg) is float
    with_result = bisect_to_target(cost_report, 37, 0.0, 1.0, value=lambda r: r["cost"], value_tol=1)
    assert_type(with_result, BisectionResult[float, dict[str, float]])
    assert_type(with_result.below.result, dict[str, float])


def test_bisection_points_resume_searches():
    """A previous result's points can be passed back as ends, and int points pass where float ones are expected."""
    previous = bisect_to_target(square, 50, 0, 100)
    assert_type(bisect_to_target(square, 55, previous.below, previous.above), BisectionResult[int, float])
    point: BisectionPoint[float, float] = previous.below
    assert point.arg == 7


def scaled_draw(scale: float, rng: np.random.Generator) -> float:
    """One scaled draw, for parallel_map_with_rngs' result type."""
    return scale * rng.random()


def test_parallel_map_with_rngs_returns_a_list_of_the_function_results():
    """parallel_map_with_rngs returns list[R] for a function returning R."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        result = parallel_map_with_rngs(scaled_draw, [1.0, 2.0], executor=executor, rng=0)
    assert_type(result, list[float])
    assert len(result) == 2


def test_parallel_map_with_rngs_accepts_pools_as_well_as_executors():
    """Type checkers accept a multiprocessing.Pool, which isn't an Executor, as the executor."""
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        result = parallel_map_with_rngs(scaled_draw, [1.0], executor=pool, rng=0)
    assert_type(result, list[float])


def test_splitting_functions_keep_the_item_and_key_types():
    """split_evenly and split_for_workers keep the items' type, and the key function's for the groups."""
    assert_type(split_evenly([1, 2, 3], 2), list[list[int]])
    assert_type(allocate_workers([3, 1], 2), list[int])
    without_key = split_for_workers(["a", "b"], 2)
    assert_type(without_key, list[Batch[None, str]])
    with_key = split_for_workers(["a1", "b1"], 2, key=lambda item: item[0])
    assert_type(with_key, list[Batch[str, str]])
    assert [batch.key for batch in with_key] == ["a", "b"]


def test_stacking_types_follow_the_blocks():
    """Dense blocks give an ndarray type, sparse ones a SparseArray, and a mix PsMatrix, for type checkers."""
    dense = np.ones((2, 2))
    sparse = scipy.sparse.csr_array(dense)
    mixed: list[PsMatrix] = [dense, sparse]
    assert_type(ps_hstack([dense, dense]), npt.NDArray[Any])
    assert_type(ps_vstack([sparse, sparse]), SparseArray)
    assert_type(ps_hstack(mixed), PsMatrix)
    assert_type(ps_block([[dense, None], [None, dense]]), npt.NDArray[Any])
    assert_type(ps_block([[sparse, None], [None, dense]]), PsMatrix)
    assert_type(as_dense(sparse), npt.NDArray[Any])
    assert_type(as_dense([[1, 2]]), npt.NDArray[Any])


def test_sparse_array_values_have_the_array_attributes():
    """A value typed SparseArray or PsMatrix has shape, ndim and dtype for type checkers (scipy-stubs' sparray,
    a bare mixin, doesn't)."""
    values: list[PsMatrix] = [np.ones((2, 3)), scipy.sparse.coo_array(np.ones((2, 3)))]
    for value in values:
        assert (value.shape, value.ndim, value.dtype) == ((2, 3), 2, np.float64)


def test_sparse_format_values_are_accepted_as_formats():
    """A variable typed SparseFormat can be passed as format (a plain str couldn't)."""
    format: SparseFormat = "csc"
    result = ps_vstack([scipy.sparse.csr_array(np.ones((1, 2))), np.ones((1, 2))], format=format)
    # a PsMatrix has no format until it's narrowed to a sparse array
    assert not isinstance(result, np.ndarray)
    assert result.format == "csc"


def test_compare_along_returns_a_float64_array_and_takes_typed_arguments():
    """compare_along's result is typed as a float64 array; variables typed with its aliases can be passed."""
    reference: ComparisonReference = "first"
    measure: ComparisonMeasure = "relative"
    result = compare_along([1.0, 2.0], reference, measure)
    assert_type(result, npt.NDArray[np.float64])
    assert result.dtype == np.float64


def test_label_bars_returns_annotations_and_takes_typed_arguments():
    """label_bars returns lists of Annotations; variables typed with its aliases can be passed."""
    _, ax = plt.subplots()
    ax.bar(["a", "b"], [1.0, 2.0])
    measure: BarLabelMeasure = "difference"
    reference: BarLabelReference = "a"
    position: BarLabelPosition = "inside"
    labels = label_bars(ax, measure, reference, position=position)
    assert_type(labels, list[list[Annotation]])
    assert isinstance(labels[0][0], Annotation)
    plt.close("all")


def test_label_bars_on_several_axes_returns_labels_by_axes():
    """Given several axes (an array, a list or a figure), label_bars returns one list of labels per axes.

    (matplotlib's stubs type plt.subplots' array of axes as Any, which matches both of label_bars' overloads, so
    it's given its type here.)
    """
    figure, untyped_axes = plt.subplots(1, 2)
    axes: npt.NDArray[Any] = untyped_axes
    for ax in axes:
        ax.bar(["a", "b"], [1.0, 2.0])
    by_axes = label_bars(axes, "difference")
    assert_type(by_axes, list[list[list[Annotation]]])
    axes_list: list[Axes] = list(axes)
    assert_type(label_bars(axes_list), list[list[list[Annotation]]])
    assert_type(label_bars(figure), list[list[list[Annotation]]])
    assert isinstance(by_axes[1][0][1], Annotation)
    plt.close("all")
