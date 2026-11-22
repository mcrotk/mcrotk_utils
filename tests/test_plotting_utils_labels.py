"""Tests for ``mcrotk_utils.plotting_utils.label_bars``.

The bars are drawn with matplotlib, pandas and seaborn (on the non-interactive Agg backend). The numbers in the
labels are checked against the data (group means, as seaborn's bars show them) and against ``compare_along`` on the
table of bar values (``compare_along`` itself is tested in ``test_numpy_utils_comparisons.py``); the labels'
positions against the bars' and error bars' coordinates.
"""

from typing import Any

import matplotlib

matplotlib.use("Agg")  # before pyplot: no windows

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns
from matplotlib.container import BarContainer
from matplotlib.text import Annotation

from mcrotk_utils.numpy_utils import compare_along
from mcrotk_utils.plotting_utils import label_bars

NAN = np.nan
WEEKS = ["w1", "w2", "w3"]
GROUPS = ["control", "test"]


def exact(number):
    """A format giving every digit, so the tests can read the numbers back."""
    return f"{number:.17g}"


def numbers(labels):
    """Return the labels' numbers as an array of lists (one per container), NaN for empty labels."""
    return [[float(label.get_text()) if label.get_text() else NAN for label in series] for series in labels]


def texts(labels):
    """Return the labels' texts, one list per container."""
    return [[label.get_text() for label in series] for series in labels]


@pytest.fixture(autouse=True)
def close_figures():
    """Close every figure after each test, so they don't pile up."""
    yield
    plt.close("all")


@pytest.fixture
def orders():
    """Orders per store by week and group: 20 rows per combination."""
    rng = np.random.default_rng(3)
    rows = [
        {"week": week, "group": group, "orders": rng.normal(100 + 5 * w + 3 * g, 10)}
        for w, week in enumerate(WEEKS)
        for g, group in enumerate(GROUPS)
        for _ in range(20)
    ]
    return pd.DataFrame(rows)


def bar_containers(ax):
    """The axes' bar containers (leaving out error-bar containers), typed as such."""
    return [container for container in ax.containers if isinstance(container, BarContainer)]


def means_table(df, rows="group", columns="week"):
    """The mean orders as a table of groups by weeks: the values seaborn's bars show."""
    return df.groupby([rows, columns]).orders.mean().unstack().to_numpy()


def seaborn_axes(df, **kwargs):
    """Draw seaborn's bar plot of orders by week and group, and return its axes."""
    _, ax = plt.subplots()
    return sns.barplot(data=df, x="week", y="orders", hue="group", ax=ax, **kwargs)


# --- What the labels show ------------------------------------------------------------------------------------------


def test_values_are_the_bars_values_from_seaborn(orders):
    """By default each seaborn bar is labelled with its value, the group mean."""
    labels = label_bars(seaborn_axes(orders), fmt=exact)
    np.testing.assert_allclose(numbers(labels), means_table(orders))


@pytest.mark.parametrize("reference", ["previous", "next", "first", "last", 0, -1, 1])
@pytest.mark.parametrize("measure", ["difference", "relative", "ratio"])
def test_comparisons_along_categories_are_within_each_series(orders, reference, measure):
    """along="categories" compares weeks within each group, as compare_along does along the table's rows."""
    labels = label_bars(seaborn_axes(orders), measure, reference, fmt=exact)
    expected = compare_along(means_table(orders), reference, measure, axis=1)
    np.testing.assert_allclose(numbers(labels), expected, equal_nan=True)


@pytest.mark.parametrize("reference", ["previous", "next", "first", "last", 0, -1])
def test_comparisons_along_series_are_within_each_category(orders, reference):
    """along="series" compares groups within each week, as compare_along does down the table's columns."""
    labels = label_bars(seaborn_axes(orders), "difference", reference, along="series", fmt=exact)
    expected = compare_along(means_table(orders), reference, "difference", axis=0)
    np.testing.assert_allclose(numbers(labels), expected, equal_nan=True)


def test_shares_are_of_the_series_total_or_the_category_total(orders):
    """share divides by the series' total along categories, by the category's total along series."""
    table = means_table(orders)
    along_categories = label_bars(seaborn_axes(orders), "share", fmt=exact)
    np.testing.assert_allclose(numbers(along_categories), table / table.sum(axis=1, keepdims=True))
    along_series = label_bars(seaborn_axes(orders), "share", along="series", fmt=exact)
    np.testing.assert_allclose(numbers(along_series), table / table.sum(axis=0, keepdims=True))


def test_a_measure_function_is_applied_as_in_compare_along(orders):
    """A function of (values, references) works as in compare_along."""
    labels = label_bars(seaborn_axes(orders), lambda v, r: v - 2 * r, "first", fmt=exact)
    expected = compare_along(means_table(orders), "first", lambda v, r: v - 2 * r, axis=1)
    np.testing.assert_allclose(numbers(labels), expected, equal_nan=True)


def test_reference_and_skip_missing_are_ignored_for_values_and_shares(orders):
    """measure="value" and "share" don't use the reference."""
    ax = seaborn_axes(orders)
    assert texts(label_bars(ax, "value", "last", fmt=exact)) == texts(label_bars(ax, "value", fmt=exact))
    assert texts(label_bars(ax, "share", 1, fmt=exact)) == texts(label_bars(ax, "share", fmt=exact))


# --- References by name --------------------------------------------------------------------------------------------


def test_a_series_named_in_seaborns_legend_can_be_the_reference(orders):
    """seaborn's containers have no labels, but its legend names the hue levels in the same order."""
    labels = label_bars(seaborn_axes(orders), "difference", "control", along="series", fmt=exact)
    expected = compare_along(means_table(orders), 0, "difference", axis=0)
    np.testing.assert_allclose(numbers(labels), expected, equal_nan=True)


def test_a_category_named_by_its_tick_label_can_be_the_reference(orders):
    """A category's tick label names it as the reference."""
    labels = label_bars(seaborn_axes(orders), "difference", "w2", fmt=exact)
    expected = compare_along(means_table(orders), 1, "difference", axis=1)
    np.testing.assert_allclose(numbers(labels), expected, equal_nan=True)


def test_pandas_columns_name_the_series():
    """pandas labels each container with its column, which can be the reference."""
    wide = pd.DataFrame({"control": [1.0, 2.0], "test": [3.0, 5.0]}, index=["w1", "w2"])
    ax = wide.plot.bar()
    labels = label_bars(ax, "ratio", "control", along="series", fmt=exact)
    np.testing.assert_allclose(numbers(labels), [[NAN, NAN], [3.0, 2.5]], equal_nan=True)


def test_reference_names_that_match_nothing_or_several_are_refused(orders):
    """An unknown name, or one shared by two categories, raises ValueError listing the names."""
    with pytest.raises(ValueError, match="matches no series; the series names are"):
        label_bars(seaborn_axes(orders), "difference", "treatment", along="series")
    _, ax = plt.subplots()
    ax.bar([0, 1, 2], [1, 2, 3], tick_label=["a", "b", "a"])
    with pytest.raises(ValueError, match="matches more than one category"):
        label_bars(ax, "difference", "a")


def test_series_without_known_names_need_an_index(orders):
    """Without container labels or a legend, a series can only be the reference by index."""
    ax = seaborn_axes(orders, legend=False)
    with pytest.raises(ValueError, match="series names aren't known"):
        label_bars(ax, "difference", "control", along="series")
    label_bars(ax, "difference", 0, along="series")  # an index still works


def test_the_named_references_take_precedence_over_labels():
    """A category labelled "last" isn't the reference "last": the names mean positions."""
    _, ax = plt.subplots()
    ax.bar(["last", "middle", "first"], [1.0, 2.0, 4.0])
    labels = label_bars(ax, "difference", "last", fmt=exact)
    np.testing.assert_allclose(numbers(labels), [[-3.0, -2.0, NAN]], equal_nan=True)


def test_index_references_outside_the_table_are_refused(orders):
    """An index beyond the categories or series raises IndexError."""
    with pytest.raises(IndexError):
        label_bars(seaborn_axes(orders), "difference", 3)
    with pytest.raises(IndexError):
        label_bars(seaborn_axes(orders), "difference", -3, along="series")


# --- Missing bars --------------------------------------------------------------------------------------------------


@pytest.fixture
def orders_without_test_in_w2(orders):
    """The orders with the test group missing in week 2, so seaborn draws no bar there."""
    return orders[~((orders.week == "w2") & (orders.group == "test"))]


def test_a_missing_bar_breaks_comparisons_unless_skip_missing(orders_without_test_in_w2):
    """Without skip_missing, w3 of test isn't compared with w1; with it, it is."""
    df = orders_without_test_in_w2
    table = df.groupby(["group", "week"]).orders.mean().unstack().reindex(columns=WEEKS).to_numpy()
    ax = seaborn_axes(df)
    assert [len(c) for c in ax.containers if isinstance(c, BarContainer)] == [3, 2]  # the bar is left out
    for skip in (False, True):
        labels = label_bars(ax, "difference", skip_missing=skip, fmt=exact)
        expected = compare_along(table, "previous", axis=1, skip_missing=skip)
        assert len(labels[1]) == 2  # one label per bar drawn
        np.testing.assert_allclose(numbers(labels)[1], expected[1][[0, 2]], equal_nan=True)
    assert np.isnan(numbers(label_bars(ax, "difference", fmt=exact))[1][1])
    assert not np.isnan(numbers(label_bars(ax, "difference", skip_missing=True, fmt=exact))[1][1])


def test_bars_drawn_with_missing_values_get_the_missing_label():
    """A bar drawn with a NaN value has nothing to show; its label sits at its base."""
    _, ax = plt.subplots()
    ax.bar(["a", "b", "c"], [1.0, NAN, 3.0])
    labels = label_bars(ax, missing_label="n/a")
    assert texts(labels) == [["1.00", "n/a", "3.00"]]
    assert labels[0][1].xy == (1.0, 0.0)


def test_missing_and_reference_labels():
    """Bars with nothing to show get missing_label; reference bars get reference_label, or else missing_label."""
    _, ax = plt.subplots()
    ax.bar(["a", "b", "c"], [1.0, 2.0, 4.0])
    assert texts(label_bars(ax, "difference", fmt="+g")) == [["", "+1", "+2"]]
    assert texts(label_bars(ax, "difference", fmt="+g", missing_label="–")) == [["–", "+1", "+2"]]
    assert texts(label_bars(ax, "difference", "last", fmt="+g", missing_label="–")) == [["-3", "-2", "–"]]
    assert texts(label_bars(ax, "difference", "last", fmt="+g", missing_label="–", reference_label="ref")) == [
        ["-3", "-2", "ref"]
    ]
    # reference_label is only for the reference bars themselves, not for bars with no previous one
    assert texts(label_bars(ax, "difference", fmt="+g", reference_label="ref")) == [["", "+1", "+2"]]


# --- Formats -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("measure", "reference", "expected"),
    [
        ("value", "previous", ["1,500", "-25", "3,000"]),  # 3,000 has 3 significant digits with no decimals
        ("difference", "first", ["", "-1,525", "+1,500"]),
        ("relative", "first", ["", "-101.7%", "+100.0%"]),
        ("ratio", "first", ["", "-0.02×", "2.00×"]),  # decimals from the largest, 2.00
        ("share", "previous", ["33.5%", "-0.6%", "67.0%"]),
    ],
)
def test_default_formats(measure, reference, expected):
    """Thousands separators and shared decimals for numbers (sign for differences), percentages, ratios with ×."""
    _, ax = plt.subplots()
    ax.bar(["a", "b", "c"], [1500.0, -25.0, 3000.0])
    assert texts(label_bars(ax, measure, reference)) == [expected]


def test_default_decimals_give_the_largest_number_three_significant_digits():
    """The decimals are the same for every label, enough for three significant digits on the largest."""
    _, ax = plt.subplots()
    ax.bar(["a", "b", "c"], [0.01234, 0.5, 1.25])
    assert texts(label_bars(ax)) == [["0.01", "0.50", "1.25"]]
    zeros = ax.bar(["a", "b", "c"], [0.0, 0.0, 0.0])
    assert texts(label_bars(ax, containers=[zeros])) == [["0", "0", "0"]]


@pytest.mark.parametrize(
    ("fmt", "expected"),
    [
        ("{:.1f} kg", ["2.0 kg", "3.5 kg"]),  # a format string with a replacement field
        ("+.2f", ["+2.00", "+3.50"]),  # a format spec alone
        (lambda number: f"<{number:g}>", ["<2>", "<3.5>"]),  # a function
    ],
)
def test_fmt_forms(fmt, expected):
    """fmt can be a format string, a format spec, or a function."""
    _, ax = plt.subplots()
    ax.bar(["a", "b"], [2.0, 3.5])
    assert texts(label_bars(ax, fmt=fmt)) == [expected]


def test_basis_points_with_a_format_function():
    """The docstring's example: a difference of proportions in basis points."""
    _, ax = plt.subplots()
    ax.bar(["control", "test"], [0.30, 0.3125])
    labels = label_bars(ax, "difference", "first", fmt=lambda d: f"{d * 1e4:+.0f} bps")
    assert texts(labels) == [["", "+125 bps"]]


def test_fmt_of_another_type_is_refused():
    """fmt must be a string or a function."""
    _, ax = plt.subplots()
    ax.bar(["a"], [1.0])
    with pytest.raises(TypeError, match="fmt must be a string or a function"):
        label_bars(ax, fmt=3)  # type: ignore[call-overload]


# --- Where the labels go -------------------------------------------------------------------------------------------


def error_bar_tops(ax):
    """The highest point of the lines over each bar of each seaborn container (its error bar)."""
    tops = []
    for container in bar_containers(ax):
        series = []
        for patch in container.patches:
            low, high = patch.get_x(), patch.get_x() + patch.get_width()
            ys = [
                y
                for line in ax.lines
                for x, y in zip(line.get_xdata(), line.get_ydata(), strict=True)
                if low <= x <= high and np.isfinite(y)
            ]
            series.append(max(ys))
        tops.append(series)
    return tops


@pytest.mark.parametrize("capsize", [0, 0.1])
def test_outside_labels_go_beyond_seaborns_error_bars(orders, capsize):
    """Labels outside the bars are anchored at the top of the error bar over each bar (with or without caps)."""
    ax = seaborn_axes(orders, capsize=capsize)
    labels = label_bars(ax, padding=4)
    for series, tops in zip(labels, error_bar_tops(ax), strict=True):
        for label, top in zip(series, tops, strict=True):
            assert label.xy[1] == pytest.approx(top)
            assert label.xyann == (0.0, 4.0)
            assert label.get_verticalalignment() == "bottom"
            assert label.get_horizontalalignment() == "center"


def test_without_avoid_lines_labels_are_at_the_bar_ends(orders):
    """avoid_lines=False anchors outside labels at the bars' ends, at the middle of each bar."""
    ax = seaborn_axes(orders)
    labels = label_bars(ax, avoid_lines=False)
    for container, series in zip(bar_containers(ax), labels, strict=True):
        for patch, label in zip(container.patches, series, strict=True):
            assert label.xy == pytest.approx((patch.get_x() + patch.get_width() / 2, patch.get_height()))


def test_matplotlib_error_bars_are_avoided_on_both_sides_of_zero():
    """matplotlib's yerr error bars (a LineCollection and cap markers) are avoided, below negative bars too."""
    _, ax = plt.subplots()
    ax.bar(["a", "b"], [3.0, -2.0], yerr=[0.5, 0.4], capsize=3)
    ax.axhline(0)  # in axes coordinates across the plot: not an error bar
    labels = label_bars(ax)[0]
    assert labels[0].xy == pytest.approx((0.0, 3.5))
    assert (labels[0].get_verticalalignment(), labels[0].xyann) == ("bottom", (0.0, 3.0))
    assert labels[1].xy == pytest.approx((1.0, -2.4))
    assert (labels[1].get_verticalalignment(), labels[1].xyann) == ("top", (0.0, -3.0))


def test_inside_and_center_positions():
    """inside: at the end, offset back into the bar; center: at the middle of the bar, not offset."""
    _, ax = plt.subplots()
    ax.bar(["a", "b"], [4.0, -2.0], yerr=[1.0, 1.0])
    inside = label_bars(ax, position="inside", padding=2)[0]
    assert [label.xy for label in inside] == [(0.0, 4.0), (1.0, -2.0)]  # error bars don't matter inside
    assert [label.xyann for label in inside] == [(0.0, -2.0), (0.0, 2.0)]
    assert [label.get_verticalalignment() for label in inside] == ["top", "bottom"]
    center = label_bars(ax, position="center")[0]
    assert [label.xy for label in center] == [(0.0, 2.0), (1.0, -1.0)]
    assert [(label.xyann, label.get_verticalalignment()) for label in center] == [((0.0, 0.0), "center")] * 2


def test_horizontal_bars_are_labelled_to_their_right_or_left():
    """For horizontal bars, outside labels go right of positive bars and left of negative ones."""
    ax = pd.DataFrame({"x": [2.0, -1.0]}, index=["a", "b"]).plot.barh(xerr=[[0.5, 0.5]])
    labels = label_bars(ax, padding=5)[0]
    assert labels[0].xy == pytest.approx((2.5, 0.0))
    assert (labels[0].get_horizontalalignment(), labels[0].get_verticalalignment(), labels[0].xyann) == (
        "left",
        "center",
        (5.0, 0.0),
    )
    assert labels[1].xy == pytest.approx((-1.5, 1.0))
    assert (labels[1].get_horizontalalignment(), labels[1].xyann) == ("right", (-5.0, 0.0))


def test_horizontal_seaborn_bars_are_read_along_the_y_axis(orders):
    """seaborn's horizontal bars give the same table, with the categories on the y axis."""
    _, ax = plt.subplots()
    sns.barplot(data=orders, y="week", x="orders", hue="group", orient="h", ax=ax)
    labels = label_bars(ax, "difference", "first", fmt=exact)
    expected = compare_along(means_table(orders), "first", axis=1)
    np.testing.assert_allclose(numbers(labels), expected, equal_nan=True)
    assert {label.get_horizontalalignment() for series in labels for label in series} == {"left"}


def test_an_inverted_value_axis_flips_the_labels_on_screen():
    """With the value axis inverted, a positive bar grows downwards, so its outside label goes below it."""
    _, ax = plt.subplots()
    ax.bar(["a"], [2.0])
    ax.invert_yaxis()
    label = label_bars(ax, padding=3)[0][0]
    assert (label.xy, label.xyann, label.get_verticalalignment()) == ((0.0, 2.0), (0.0, -3.0), "top")


def test_text_options_are_passed_to_annotate():
    """Extra keyword arguments style every label, as ax.annotate's do."""
    _, ax = plt.subplots()
    ax.bar(["a", "b"], [1.0, 2.0])
    labels = label_bars(ax, color="white", fontsize=7, fontweight="bold")[0]
    assert all(label.get_color() == "white" and label.get_fontsize() == 7 for label in labels)
    assert all(label.get_fontweight() == "bold" for label in labels)


# --- Result, containers and kinds of plot ---------------------------------------------------------------------------


def test_result_has_one_annotation_per_bar_on_the_axes(orders):
    """One list per container, one Annotation per bar in the container's order, all added to the axes."""
    ax = seaborn_axes(orders)
    labels = label_bars(ax)
    assert [len(series) for series in labels] == [len(c) for c in ax.containers]
    assert all(isinstance(label, Annotation) and label in ax.texts for series in labels for label in series)


def test_containers_restrict_the_bars_and_the_series_compared():
    """With containers given, only those are labelled, and comparisons along series are among them."""
    wide = pd.DataFrame({"a": [1.0, 2.0], "b": [4.0, 4.0], "c": [9.0, 7.0]}, index=["w1", "w2"])
    ax = wide.plot.bar()
    containers = [c for c in ax.containers if isinstance(c, BarContainer)]
    labels = label_bars(ax, "difference", "first", along="series", containers=containers[1:], fmt=exact)
    assert len(labels) == 2
    np.testing.assert_allclose(numbers(labels), [[NAN, NAN], [5.0, 3.0]], equal_nan=True)
    by_name = label_bars(ax, "difference", "c", along="series", containers=containers[1:], fmt=exact)
    np.testing.assert_allclose(numbers(by_name), [[-5.0, -3.0], [NAN, NAN]], equal_nan=True)


def test_pandas_stacked_bars_compare_segments():
    """Stacked bars give each segment's value; shares along series are each segment's share of its stack."""
    wide = pd.DataFrame({"a": [1.0, 3.0], "b": [3.0, 1.0]}, index=["w1", "w2"])
    ax = wide.plot.bar(stacked=True)
    np.testing.assert_allclose(numbers(label_bars(ax, fmt=exact)), [[1.0, 3.0], [3.0, 1.0]])
    shares = label_bars(ax, "share", along="series", position="center", fmt=exact)
    np.testing.assert_allclose(numbers(shares), [[0.25, 0.75], [0.75, 0.25]])
    assert shares[1][0].xy == (0.0, 2.5)  # the middle of the upper segment, from 1 to 4


def test_seaborn_without_hue_is_one_series(orders):
    """Without hue, seaborn draws one container: the categories are compared within it."""
    _, ax = plt.subplots()
    sns.barplot(data=orders, x="week", y="orders", ax=ax)
    labels = label_bars(ax, "relative", "first", fmt=exact)
    weekly = orders.groupby("week").orders.mean().to_numpy()
    np.testing.assert_allclose(numbers(labels), [compare_along(weekly, "first", "relative")], equal_nan=True)


def regional_orders(orders):
    """The orders, split into two regions with an extra 'region' column."""
    return orders.assign(region=np.tile(["n", "s"], len(orders) // 2))


def test_a_seaborn_grid_is_labelled_one_axes_at_a_time(orders):
    """For catplot's grid, each axes can be labelled in turn; each subplot is compared within itself."""
    df = regional_orders(orders)
    grid = sns.catplot(data=df, x="week", y="orders", hue="group", col="region", kind="bar")
    for ax, region in zip(grid.axes.flat, ["n", "s"], strict=True):
        labels = label_bars(ax, "difference", fmt=exact)
        expected = compare_along(means_table(df[df.region == region]), "previous", axis=1)
        np.testing.assert_allclose(numbers(labels), expected, equal_nan=True)


# --- Several axes (grids) ------------------------------------------------------------------------------------------


def test_a_seaborn_grid_is_labelled_in_one_call(orders):
    """Given the grid, each subplot is labelled and compared within itself, and the labels come by subplot.

    The series are named by the grid's legend, which is the figure's, not each axes'.
    """
    df = regional_orders(orders)
    grid = sns.catplot(data=df, x="week", y="orders", hue="group", col="region", kind="bar")
    labels = label_bars(grid, "difference", "control", along="series", fmt=exact)
    assert len(labels) == 2
    for subplot_labels, region in zip(labels, ["n", "s"], strict=True):
        means = means_table(df[df.region == region])
        np.testing.assert_allclose(numbers(subplot_labels)[1], means[1] - means[0])


def test_the_default_format_is_shared_across_the_subplots():
    """The decimals come from the numbers on every subplot, so they match; alone, each subplot gets its own."""
    figure, axes = plt.subplots(1, 2)
    axes[0].bar(["a", "b"], [10.0, 16.71])
    axes[1].bar(["a", "b"], [10.0, 22.6])
    together = label_bars(axes, "difference")
    assert [texts(subplot) for subplot in together] == [[["", "+6.7"]], [["", "+12.6"]]]
    plt.close(figure)
    _, alone = plt.subplots()
    alone.bar(["a", "b"], [10.0, 16.71])
    assert texts(label_bars(alone, "difference")) == [["", "+6.71"]]


def test_fmt_applies_to_every_subplot():
    """A format given is used on every subplot."""
    _, axes = plt.subplots(1, 2)
    axes[0].bar(["a"], [1.5])
    axes[1].bar(["a"], [250.0])
    assert [texts(subplot) for subplot in label_bars(axes, fmt="{:.2f}")] == [[["1.50"]], [["250.00"]]]


def test_arrays_lists_and_figures_give_the_axes_row_by_row():
    """A 2-D array is read row by row, as are nested lists and a figure's axes; axes without bars get no labels."""
    figure, axes = plt.subplots(2, 2)
    for value, ax in zip([1.0, 2.0, 3.0], axes.flat, strict=False):  # the last subplot is left empty
        ax.bar(["a"], [value])
    expected = [[["1"]], [["2"]], [["3"]], []]
    for several in [axes, axes.tolist(), list(axes.flat), figure, (ax for ax in axes.flat)]:
        labelled = label_bars(several, fmt="{:.0f}")
        assert [texts(subplot) for subplot in labelled] == expected
        for subplot in labelled:
            for annotation in (label for series in subplot for label in series):
                annotation.remove()


def test_names_are_looked_up_in_each_subplot():
    """A reference name must be found in every subplot with bars."""
    _, axes = plt.subplots(1, 2)
    axes[0].bar(["a", "b"], [1.0, 2.0])
    axes[1].bar(["c", "d"], [1.0, 2.0])
    with pytest.raises(ValueError, match="reference 'a' matches no category"):
        label_bars(axes, "difference", "a")


def test_a_figure_legend_names_series_only_when_it_is_the_only_one():
    """With no legend on the axes, the figure's legend names the series, unless the figure has several."""
    figure, ax = plt.subplots()
    ax.bar(["a"], [1.0])
    ax.bar(["a"], [2.0], bottom=1.0)
    for container in ax.containers:
        container.set_label("_unlabelled")
    handles = [container.patches[0] for container in bar_containers(ax)]
    figure.legend(handles, ["low", "high"])
    assert texts(label_bars(ax, "difference", "low", along="series")) == [[""], ["+1.00"]]
    figure.legend(handles, ["x", "y"])
    with pytest.raises(ValueError, match="series names aren't known"):
        label_bars(ax, "difference", "low", along="series")


def test_several_axes_without_any_bars_are_refused():
    """With several axes, at least one must have bars."""
    _, axes = plt.subplots(1, 2)
    with pytest.raises(ValueError, match="none of the axes have bar containers"):
        label_bars(axes)
    with pytest.raises(ValueError, match="none of the axes have bar containers"):
        label_bars([])


def test_containers_are_only_for_a_single_axes():
    """containers name bars on one axes, so they can't be given with several."""
    _, axes = plt.subplots(1, 2)
    containers = [axes[0].bar(["a"], [1.0])]
    with pytest.raises(ValueError, match="containers can only be given for a single axes"):
        label_bars(axes, containers=containers)


def test_several_axes_must_hold_only_axes():
    """Anything other than axes among the several is refused."""
    _, ax = plt.subplots()
    ax.bar(["a"], [1.0])
    with pytest.raises(TypeError, match="not to hold str"):
        label_bars([ax, "b"])


def test_the_docstring_examples(orders):
    """Change since the first week within each group, and test against control in each week."""
    ax = sns.barplot(data=orders, x="week", y="orders", hue="group")
    since_first = label_bars(ax, "relative", "first", fmt=exact)
    np.testing.assert_allclose(
        numbers(since_first), compare_along(means_table(orders), "first", "relative"), equal_nan=True
    )
    against_control = label_bars(ax, "difference", "control", along="series", fmt=exact)
    np.testing.assert_allclose(
        numbers(against_control)[1], means_table(orders)[1] - means_table(orders)[0], equal_nan=True
    )


# --- Matching bars to categories -----------------------------------------------------------------------------------


def test_bars_at_numeric_positions_need_match_order():
    """Bars that aren't each at their own tick can't be matched by ticks; match="order" numbers them by position."""
    _, ax = plt.subplots()
    ax.bar([1.0, 2.0, 7.0], [1.0, 2.0, 4.0])
    ax.set_xticks([0.0, 5.0, 10.0])
    with pytest.raises(ValueError, match="match='order'"):
        label_bars(ax, "difference")
    labels = label_bars(ax, "difference", match="order", fmt=exact)
    np.testing.assert_allclose(numbers(labels), [[NAN, 1.0, 2.0]], equal_nan=True)


def test_match_order_numbers_bars_by_position_not_drawing_order():
    """With match="order", "previous" is the bar to the left, whatever order the bars were drawn in."""
    _, ax = plt.subplots()
    ax.bar([3.0, 1.0, 2.0], [30.0, 10.0, 20.0])
    labels = label_bars(ax, "difference", match="order", fmt=exact)
    np.testing.assert_allclose(numbers(labels), [[10.0, NAN, 10.0]], equal_nan=True)


def test_reference_names_need_tick_matching():
    """With match="order" the categories have no names, so a category can only be the reference by index."""
    _, ax = plt.subplots()
    ax.bar(["a", "b"], [1.0, 2.0])
    with pytest.raises(ValueError, match="category names aren't known"):
        label_bars(ax, "difference", "a", match="order")


def test_categories_are_the_ticks_with_bars():
    """Ticks without bars aren't categories, so "previous" skips them."""
    _, ax = plt.subplots()
    ax.bar([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    np.testing.assert_allclose(numbers(label_bars(ax, "difference", fmt=exact)), [[NAN, 1.0, 2.0]], equal_nan=True)


def test_two_bars_of_one_series_at_one_tick_are_refused():
    """Two bars of one series nearest the same tick can't be told apart by ticks."""
    _, ax = plt.subplots()
    ax.bar([0.9, 1.1], [1.0, 2.0], width=0.1)
    ax.set_xticks([0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="nearest the same tick"):
        label_bars(ax)


def test_axes_without_ticks_need_match_order():
    """With no ticks on the category axis, the bars can only be matched by order."""
    _, ax = plt.subplots()
    ax.bar(["a", "b"], [1.0, 2.0])
    ax.set_xticks([])
    with pytest.raises(ValueError, match="no ticks"):
        label_bars(ax)
    assert texts(label_bars(ax, match="order")) == [["1.00", "2.00"]]


# --- Errors --------------------------------------------------------------------------------------------------------


def test_axes_without_bars_are_refused():
    """There must be bars to label."""
    _, ax = plt.subplots()
    ax.plot([1, 2])
    with pytest.raises(ValueError, match="no bars to label"):
        label_bars(ax)


def test_a_mix_of_vertical_and_horizontal_bars_is_refused():
    """The bars must all go the same way."""
    _, ax = plt.subplots()
    ax.bar([0, 1], [1.0, 2.0])
    ax.barh([0, 1], [1.0, 2.0])
    with pytest.raises(ValueError, match="all be vertical or all horizontal"):
        label_bars(ax)


def test_containers_that_arent_bar_containers_are_refused():
    """containers must be BarContainers."""
    _, ax = plt.subplots()
    ax.bar(["a"], [1.0], yerr=[0.1])
    error_bars: list[Any] = [c for c in ax.containers if not isinstance(c, BarContainer)]
    with pytest.raises(TypeError, match="BarContainers"):
        label_bars(ax, containers=error_bars)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"measure": "percent"}, "measure must be one of"),
        ({"along": "rows"}, "along must be"),
        ({"position": "above"}, "position must be one of"),
        ({"match": "labels"}, "match must be"),
        ({"measure": "difference", "reference": "prev"}, "matches no category"),
    ],
)
def test_unknown_choices_are_refused(kwargs, message):
    """Unknown names for the measure, direction, position or matching raise ValueError."""
    _, ax = plt.subplots()
    ax.bar(["a", "b"], [1.0, 2.0])
    with pytest.raises(ValueError, match=message):
        label_bars(ax, **kwargs)


def test_numpy_integer_references_are_indices():
    """A NumPy integer reference is an index, as in compare_along."""
    _, ax = plt.subplots()
    ax.bar(["a", "b"], [1.0, 3.0])
    np.testing.assert_allclose(numbers(label_bars(ax, "difference", np.int64(0), fmt=exact)), [[NAN, 2.0]])
