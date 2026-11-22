"""Labelling the bars on matplotlib axes with their values, shares, or comparisons with other bars (private).

Public through ``mcrotk_utils.plotting_utils``.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast, overload

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.text import Annotation

from mcrotk_utils.numpy_utils import ComparisonMeasure, ComparisonReference, compare_along

# Private, but shared: which entries are the references themselves, to give them their own label
from mcrotk_utils.numpy_utils._comparisons import _reference_indices
from mcrotk_utils.plotting_utils._bar_table import _BarTable, _CategoryMatching, _has_bars, _read_bar_table
from mcrotk_utils.plotting_utils._formatting import _default_format, _formatter, _label_text, _NumberFormat
from mcrotk_utils.plotting_utils._placement import BarLabelPosition, _check_position, _line_points, _placement

# What each label shows: the bar's value, its share of the total, a comparison with a reference bar, or a function
# of (values, references) as for compare_along
type BarLabelMeasure = (
    Literal["value", "share", "difference", "relative", "ratio"]
    | Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.ArrayLike]
)

# Which bar each bar is compared with: as for compare_along, or a category's or series' name
type BarLabelReference = Literal["previous", "next", "first", "last"] | int | np.integer | str


class _HasAxes(Protocol):
    """Something holding several axes as its ``axes``: a seaborn grid (e.g. from ``catplot``) or a figure."""

    @property
    def axes(self) -> npt.NDArray[Any] | list[Axes]: ...


# Several axes, for labelling a grid of plots at once: an array of any shape (as from plt.subplots), any iterable
# (nested to any depth), or a seaborn grid or matplotlib figure
type _SeveralAxes = npt.NDArray[Any] | Iterable[Any] | _HasAxes

_MEASURES = ("value", "share", "difference", "relative", "ratio")
_NAMED_REFERENCES = ("previous", "next", "first", "last")
_DIRECTIONS = ("categories", "series")


@overload
def label_bars(
    ax: Axes,
    measure: BarLabelMeasure = "value",
    reference: BarLabelReference = "previous",
    *,
    along: Literal["categories", "series"] = "categories",
    skip_missing: bool = False,
    fmt: _NumberFormat | None = None,
    missing_label: str = "",
    reference_label: str | None = None,
    position: BarLabelPosition = "outside",
    padding: float = 3.0,
    avoid_lines: bool = True,
    containers: Sequence[BarContainer] | None = None,
    match: _CategoryMatching = "ticks",
    **text_kwargs: Any,
) -> list[list[Annotation]]: ...


@overload
def label_bars(
    ax: _SeveralAxes,
    measure: BarLabelMeasure = "value",
    reference: BarLabelReference = "previous",
    *,
    along: Literal["categories", "series"] = "categories",
    skip_missing: bool = False,
    fmt: _NumberFormat | None = None,
    missing_label: str = "",
    reference_label: str | None = None,
    position: BarLabelPosition = "outside",
    padding: float = 3.0,
    avoid_lines: bool = True,
    containers: Sequence[BarContainer] | None = None,
    match: _CategoryMatching = "ticks",
    **text_kwargs: Any,
) -> list[list[list[Annotation]]]: ...


def label_bars(
    ax: Axes | _SeveralAxes,
    measure: BarLabelMeasure = "value",
    reference: BarLabelReference = "previous",
    *,
    along: Literal["categories", "series"] = "categories",
    skip_missing: bool = False,
    fmt: _NumberFormat | None = None,
    missing_label: str = "",
    reference_label: str | None = None,
    position: BarLabelPosition = "outside",
    padding: float = 3.0,
    avoid_lines: bool = True,
    containers: Sequence[BarContainer] | None = None,
    match: _CategoryMatching = "ticks",
    **text_kwargs: Any,
) -> list[list[Annotation]] | list[list[list[Annotation]]]:
    """Label the bars on ``ax`` with their values, their shares of a total, or comparisons with other bars.

    Works on bars drawn by anything that uses matplotlib's ``bar`` or ``barh``: seaborn's ``barplot`` (with or
    without ``hue``), pandas' ``plot.bar`` and ``plot.barh``, and matplotlib itself; vertical or horizontal. The bars
    form a table: each *series* (a hue level, a DataFrame column, or one ``bar`` call) has at most one bar in each
    *category* (a position on the category axis, such as a week). Comparisons go along one direction of that table:
    within each series from one category to another (``along="categories"``, e.g. each week against the week
    before), or within each category from one series to another (``along="series"``, e.g. test against control).

    A grid of plots (e.g. from seaborn's ``catplot`` or ``plt.subplots``) can be labelled in one call: each subplot
    is labelled as above, its bars compared only with others in the same subplot, and the default format is chosen
    from the numbers on all of them, so the subplots' labels match (e.g. the same number of decimals).

    **Arguments.**

    - ``ax``: the matplotlib axes with the bars; or several, for a grid: an array of axes of any shape (as
      ``plt.subplots`` returns), a list or other iterable of them (nested to any depth), or a seaborn grid or a
      matplotlib figure (whose ``axes`` are used). Axes without bars, such as an empty subplot, are skipped.
    - ``measure``: what each label shows:

      - ``"value"`` (the default): the bar's value.
      - ``"share"``: the bar's share of the total of its series (``along="categories"``) or of its category
        (``along="series"``, e.g. for stacked bars).
      - ``"difference"``, ``"relative"`` (the relative change, as a percentage by default) or ``"ratio"``: a
        comparison with the reference bar, as in ``compare_along``.
      - a function of ``(values, references)``, as in ``compare_along``.

    - ``reference``: which bar each bar is compared with, along the chosen direction: ``"previous"`` (the default),
      ``"next"``, ``"first"`` or ``"last"``; an index (of the category or series, in order along the axis or in
      the series' order; negative counts from the end); or a category's tick label (``"w1"``) or a series' name
      (``"control"``: a container's label, as pandas and ``bar(label=...)`` set them, or else the legend's
      entries, as seaborn sets them). The names ``"previous"`` etc. take precedence over labels; use an index for a
      category or series that has one of those names. Ignored for ``"value"`` and ``"share"``.
    - ``along``: ``"categories"`` (the default) or ``"series"``, the direction to compare in, as above.
    - ``skip_missing``: as in ``compare_along``: if true, a missing bar is passed over when finding a reference,
      so with week 2 missing, week 3 is compared with week 1. By default its label is ``missing_label`` instead.
    - ``fmt``: the numbers' format: a format string with a replacement field (``"{:.1f} kg"``), a format spec
      alone (``"+.1%"``), or a function from the number to its text (e.g. ``lambda d: f"{d * 1e4:+.0f} bps"`` for
      a difference of proportions in basis points). By default, percentages for ``"relative"`` (with a sign) and
      ``"share"``; otherwise numbers with thousands separators and enough decimals, the same for every bar, to give
      the largest number three significant digits, with a sign for ``"difference"`` and ``×`` after a ratio.
    - ``missing_label``: the label of a bar with nothing to show, e.g. the first bar for ``"previous"`` (default:
      no text).
    - ``reference_label``: the label of the reference bars themselves (``"first"``, ``"last"``, an index or a
      label), e.g. ``"base"``; by default ``missing_label``.
    - ``position``: ``"outside"`` (the default) beyond the end of each bar, ``"inside"`` just within it, or
      ``"center"`` at its middle. Labels outside the bars may need room: e.g. ``ax.margins(y=0.1)``.
    - ``padding``: the distance in points between a label and the bar's end (or the lines it's kept clear of).
    - ``avoid_lines``: for ``"outside"``, keep each label beyond any lines drawn across the bar's width, such as
      error bars (seaborn's, or matplotlib's ``yerr``/``xerr``), so they don't overlap.
    - ``containers``: the bar containers to label (default: every one on ``ax``); comparisons ``along="series"``
      are then among these only. Only for a single axes.
    - ``match``: how bars are matched to categories: ``"ticks"`` (the default), by the nearest tick on the category
      axis, which suits categorical plots and allows missing bars; or ``"order"``, by their order within each
      series, for bars at arbitrary numeric positions (each series must then have a bar in every category).
    - ``text_kwargs``: passed on to ``ax.annotate`` for every label, e.g. ``fontsize="small"`` or ``color="white"``.

    **Result.** The labels (matplotlib ``Annotation`` objects), one list per container, with one label per bar in
    the container's order; a bar with nothing to show has an empty label (or ``missing_label``). For several axes,
    one such list per axes, in order (row by row for a 2-D array), with an empty list for axes without bars.

    **Errors.**

    - ``ValueError``: no bars (on any of the axes), ``containers`` given with several axes, a mix of vertical and
      horizontal bars (on one axes), an unknown ``measure``, ``along``,
      ``position`` or ``match``, a ``reference`` label that matches no category or series (or several, or names
      that aren't known), or bars that can't be matched to ticks (pass ``match="order"``); and as for
      ``compare_along``.
    - ``TypeError``: a container that isn't a ``BarContainer``, an ``ax`` that holds something other than axes, or
      a ``fmt`` that's neither a string nor a function.
    - ``IndexError``: an index ``reference`` outside the categories or series.

    **Example.** With seaborn, ``ax = sns.barplot(data=df, x="week", y="orders", hue="group")``, then
    ``label_bars(ax, "relative", "first")`` labels each week with its change since the first week, within each
    group; ``label_bars(ax, "difference", "control", along="series")`` labels each test bar with its difference
    from control in the same week. For a grid with a subplot per region,
    ``grid = sns.catplot(data=df, x="week", y="orders", hue="group", col="region", kind="bar")``, then
    ``label_bars(grid, "relative", "control", along="series")`` labels every subplot.
    """
    _check_choices(measure, along, position)
    options = _LabelOptions(
        measure=measure,
        reference=reference,
        axis=1 if along == "categories" else 0,  # the table's axis that the comparisons run along
        skip_missing=skip_missing,
        fmt=fmt,
        missing_label=missing_label,
        reference_label=missing_label if reference_label is None else reference_label,
        position=position,
        padding=padding,
        avoid_lines=avoid_lines,
        match=match,
        text_kwargs=text_kwargs,
    )
    if isinstance(ax, Axes):
        return _label_axes([_contents(ax, containers, options)], options)[0]
    if containers is not None:
        raise ValueError("containers can only be given for a single axes, not for several.")
    return _label_several(_flat_axes(ax), options)


@dataclass(frozen=True)
class _LabelOptions:
    """``label_bars``' arguments that apply to every axes, with ``along`` as a table axis."""

    measure: BarLabelMeasure
    reference: BarLabelReference
    axis: int
    skip_missing: bool
    fmt: _NumberFormat | None
    missing_label: str
    reference_label: str
    position: BarLabelPosition
    padding: float
    avoid_lines: bool
    match: _CategoryMatching
    text_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _AxesContents:
    """One axes' bars, as a table, with the number to show for each bar and which bars are references."""

    ax: Axes
    table: _BarTable
    numbers: npt.NDArray[np.float64]
    is_reference: npt.NDArray[np.bool_]


def _contents(ax: Axes, containers: Sequence[BarContainer] | None, options: _LabelOptions) -> _AxesContents:
    """Read the bars on ``ax`` and work out the number each label shows."""
    table = _read_bar_table(ax, containers, options.match)
    numbers, is_reference = _numbers(
        table, options.measure, options.reference, axis=options.axis, skip_missing=options.skip_missing
    )
    return _AxesContents(ax, table, numbers, is_reference)


def _label_several(every_axes: list[Axes], options: _LabelOptions) -> list[list[list[Annotation]]]:
    """Label the axes with bars among ``every_axes`` together, giving an empty list for each axes without bars."""
    with_bars = [ax for ax in every_axes if _has_bars(ax)]
    if not with_bars:
        raise ValueError("There are no bars to label: none of the axes have bar containers.")
    labelled = iter(_label_axes([_contents(ax, None, options) for ax in with_bars], options))
    # in the order given, with the axes without bars in their places
    return [next(labelled) if _has_bars(ax) else [] for ax in every_axes]


def _label_axes(every_contents: list[_AxesContents], options: _LabelOptions) -> list[list[list[Annotation]]]:
    """Label the bars on each axes, with one format for all of them; return the labels, by axes and series."""
    if options.fmt is not None:
        format_number = _formatter(options.fmt)
    else:
        # chosen from every axes' numbers together, so that a grid's subplots show them alike
        all_numbers = np.concatenate([contents.numbers.ravel() for contents in every_contents])
        format_number = _default_format(options.measure, all_numbers)
    return [_annotate(contents, format_number, options) for contents in every_contents]


def _annotate(
    contents: _AxesContents, format_number: Callable[[float], str], options: _LabelOptions
) -> list[list[Annotation]]:
    """Add a label to each bar on one axes; return the labels, one list per series."""
    ax, table = contents.ax, contents.table
    points = _line_points(ax) if options.avoid_lines and options.position == "outside" else None
    value_axis_inverted = ax.yaxis_inverted() if table.vertical else ax.xaxis_inverted()
    labels = []
    for series in table.bars:
        series_labels = []
        for bar in series:
            index = (bar.series, bar.category)
            text = _label_text(
                contents.numbers[index],
                contents.is_reference[index],
                format_number,
                options.missing_label,
                options.reference_label,
            )
            place = _placement(
                bar,
                options.position,
                options.padding,
                vertical=table.vertical,
                value_axis_inverted=value_axis_inverted,
                points=points,
            )
            series_labels.append(
                ax.annotate(
                    text,
                    xy=place.xy,
                    xytext=place.offset,
                    textcoords="offset points",
                    ha=place.horizontal_alignment,
                    va=place.vertical_alignment,
                    **options.text_kwargs,
                )
            )
        labels.append(series_labels)
    return labels


def _flat_axes(several: _SeveralAxes) -> list[Axes]:
    """Return the axes in ``several`` as a flat list, in order (row by row for a 2-D array)."""
    if not isinstance(several, Iterable):
        # a seaborn grid or a figure: its axes
        several = several.axes
    flat: list[Axes] = []
    for item in several:
        if isinstance(item, Axes):
            flat.append(item)
        elif isinstance(item, Iterable) and not isinstance(item, str | bytes):
            # a row of a 2-D array, or a nested list
            flat.extend(_flat_axes(item))
        else:
            raise TypeError(f"ax must be matplotlib axes, or several of them, not to hold {type(item).__name__}.")
    return flat


def _check_choices(measure: object, along: str, position: str) -> None:
    """Raise ``ValueError`` for an unknown ``measure``, ``along`` or ``position``."""
    if not callable(measure) and measure not in _MEASURES:
        raise ValueError(f"measure must be one of {', '.join(map(repr, _MEASURES))} or a function, not {measure!r}.")
    if along not in _DIRECTIONS:
        raise ValueError(f"along must be 'categories' or 'series', not {along!r}.")
    _check_position(position)


def _numbers(
    table: _BarTable, measure: BarLabelMeasure, reference: BarLabelReference, *, axis: int, skip_missing: bool
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Return the number to show for each cell of the table, and which cells are the reference bars themselves."""
    no_references = np.zeros(table.values.shape, dtype=bool)
    if measure == "value":
        return table.values, no_references
    if measure == "share":
        with np.errstate(divide="ignore", invalid="ignore"):  # an all-missing or zero total gives NaN or inf
            return table.values / np.nansum(table.values, axis=axis, keepdims=True), no_references
    resolved = _resolved_reference(table, reference, axis)
    # "value" and "share" are handled above, so what's left is a comparison measure
    comparison = cast(ComparisonMeasure, measure)
    numbers = compare_along(table.values, resolved, comparison, axis=axis, skip_missing=skip_missing)
    return numbers, _reference_cells(table.values, resolved, axis=axis, skip_missing=skip_missing)


def _resolved_reference(table: _BarTable, reference: BarLabelReference, axis: int) -> ComparisonReference:
    """Return ``reference`` with a category or series name replaced by its index."""
    if not isinstance(reference, str) or reference in _NAMED_REFERENCES:
        # A named reference or an index (type checkers can't narrow a str by membership); compare_along checks it
        return cast(ComparisonReference, reference)
    names = table.category_labels if axis == 1 else table.series_labels
    kind = "category" if axis == 1 else "series"
    if names is None:
        raise ValueError(f"The {kind} names aren't known, so reference {reference!r} can't be found; pass an index.")
    matches = [index for index, name in enumerate(names) if name == reference]
    if len(matches) != 1:
        found = "no" if not matches else "more than one"
        raise ValueError(f"reference {reference!r} matches {found} {kind}; the {kind} names are {names}.")
    return matches[0]


def _reference_cells(
    values: npt.NDArray[np.float64], reference: ComparisonReference, *, axis: int, skip_missing: bool
) -> npt.NDArray[np.bool_]:
    """Return which cells are reference bars themselves: present bars that are their own reference."""
    moved = np.moveaxis(values, axis, -1)
    indices = _reference_indices(moved, reference, skip_missing=skip_missing)
    own = indices == np.arange(moved.shape[-1])
    return np.moveaxis(own & ~np.isnan(moved), -1, axis)
