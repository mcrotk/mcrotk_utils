"""Reading the bars drawn on matplotlib axes into a table of series by categories (private).

Every library tested (matplotlib's ``bar``/``barh``, pandas' ``plot.bar``, seaborn's ``barplot`` and ``catplot``)
records each series of bars (each hue level, or each DataFrame column) as one ``BarContainer`` on the axes, in
series order. Bars of different series in the same category are side by side (dodged) around the category's
position, and a missing combination is left out rather than drawn with no height, so bars are matched to categories
by position, not by their index in the container.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.container import BarContainer
from matplotlib.legend import Legend
from matplotlib.patches import Rectangle

# How bars are matched to categories: by the nearest tick on the category axis, or by their order in each series
type _CategoryMatching = Literal["ticks", "order"]


@dataclass(frozen=True)
class _Bar:
    """One bar: its patch, where it sits in the table, its value, and its extent along both axes."""

    patch: Rectangle
    series: int
    category: int
    value: float
    span: tuple[float, float]  # along the category axis: from one side of the bar to the other
    base: float  # along the value axis: where the bar starts
    end: float  # along the value axis: where it ends (below the base for a negative bar)


@dataclass(frozen=True)
class _BarTable:
    """The bars on an axes: by series (one per container), and their values as a series-by-category table."""

    bars: list[list[_Bar]]  # one list per container, in the container's order
    values: npt.NDArray[np.float64]  # shape (series, categories); NaN where a series has no bar in a category
    category_labels: list[str] | None  # the categories' tick labels, when bars were matched to ticks
    series_labels: list[str] | None  # the series' names, when known
    vertical: bool


def _read_bar_table(
    ax: Axes, containers: Sequence[BarContainer] | None = None, match: _CategoryMatching = "ticks"
) -> _BarTable:
    """Return the bars of ``containers`` (by default all the bar containers on ``ax``) as a ``_BarTable``."""
    chosen = _chosen_containers(ax, containers)
    vertical = _is_vertical(chosen)
    category_axis = ax.xaxis if vertical else ax.yaxis
    geometries = [[_geometry(patch, vertical) for patch in container.patches] for container in chosen]
    centers = [[(low + high) / 2 for (low, high), _, _ in series] for series in geometries]
    if match == "ticks":
        categories, category_labels = _categories_by_ticks(centers, category_axis)
    elif match == "order":
        categories, category_labels = _categories_by_order(centers), None
    else:
        raise ValueError(f"match must be 'ticks' or 'order', not {match!r}.")
    bars = _bars(chosen, geometries, categories)
    return _BarTable(
        bars=bars,
        values=_values_table(bars, _category_count(categories)),
        category_labels=category_labels,
        series_labels=_series_labels(ax, chosen),
        vertical=vertical,
    )


def _has_bars(ax: Axes) -> bool:
    """Return whether ``ax`` has any bar containers (e.g. not an empty subplot of a grid)."""
    return any(isinstance(container, BarContainer) for container in ax.containers)


def _chosen_containers(ax: Axes, containers: Sequence[BarContainer] | None) -> list[BarContainer]:
    """Return the containers to label: those given, or every bar container on ``ax``; raise if there are none."""
    chosen = [c for c in ax.containers if isinstance(c, BarContainer)] if containers is None else list(containers)
    if not chosen:
        raise ValueError("There are no bars to label: the axes have no bar containers (or none were given).")
    for container in chosen:
        if not isinstance(container, BarContainer):
            raise TypeError(f"containers must be matplotlib BarContainers, not {type(container).__name__}.")
    return chosen


def _is_vertical(containers: Sequence[BarContainer]) -> bool:
    """Whether the bars are vertical; raise if some containers are vertical and others horizontal."""
    orientations = {getattr(container, "orientation", None) or "vertical" for container in containers}
    if len(orientations) > 1:
        raise ValueError("The bars must all be vertical or all horizontal, not a mix.")
    return orientations == {"vertical"}


def _geometry(patch: Rectangle, vertical: bool) -> tuple[tuple[float, float], float, float]:
    """Return a bar's span along the category axis, and its base and end along the value axis."""
    x, y, width, height = patch.get_x(), patch.get_y(), patch.get_width(), patch.get_height()
    if vertical:
        span, base, length = (x, x + width), y, height
    else:
        span, base, length = (y, y + height), x, width
    # A bar drawn with a missing (NaN) value has no end: put its end at its base, where its label (if any) goes
    end = base + length if np.isfinite(length) else base
    return (min(span), max(span)), base, end


def _categories_by_ticks(centers: list[list[float]], axis: Axis) -> tuple[list[list[int]], list[str]]:
    """Match each bar to the nearest tick, and return the bars' category indices and the categories' labels.

    The categories are the ticks that have at least one bar, in increasing position. Raise ``ValueError`` if a bar
    isn't clearly nearer one tick than the next (more than halfway between ticks, as for bars at arbitrary numeric
    positions) or two bars of one series share a tick.
    """
    ticks = np.asarray(axis.get_majorticklocs(), dtype=float)
    labels = [text.get_text() for text in axis.get_majorticklabels()]
    if ticks.size == 0:
        raise ValueError("The category axis has no ticks to match the bars to; pass match='order'.")
    order = np.argsort(ticks)
    ticks, labels = ticks[order], [labels[i] for i in order] if len(labels) == len(order) else [""] * len(order)
    tolerance = np.min(np.diff(ticks)) / 2 if ticks.size > 1 else np.inf
    tick_indices = [[_nearest_tick(center, ticks, tolerance) for center in series] for series in centers]
    for series in tick_indices:
        if len(set(series)) < len(series):
            raise ValueError("Two bars of one series are nearest the same tick; pass match='order'.")
    used = sorted({index for series in tick_indices for index in series})
    category_of_tick = {tick_index: category for category, tick_index in enumerate(used)}
    categories = [[category_of_tick[index] for index in series] for series in tick_indices]
    return categories, [labels[index] for index in used]


def _nearest_tick(center: float, ticks: npt.NDArray[np.float64], tolerance: float) -> int:
    """Return the index of the tick nearest ``center``, raising if it's not within ``tolerance``."""
    index = int(np.argmin(np.abs(ticks - center)))
    if not abs(ticks[index] - center) < tolerance:
        raise ValueError(f"The bar at {center:g} isn't clearly at one tick; pass match='order'.")
    return index


def _categories_by_order(centers: list[list[float]]) -> list[list[int]]:
    """Number each series' bars by position: its lowest bar is category 0, the next category 1, and so on."""
    return [list(np.argsort(np.argsort(series, kind="stable"))) if series else [] for series in centers]


def _category_count(categories: list[list[int]]) -> int:
    """Return the number of categories: one more than the largest index of any bar."""
    return max((max(series) + 1 for series in categories if series), default=0)


def _bars(
    containers: list[BarContainer],
    geometries: list[list[tuple[tuple[float, float], float, float]]],
    categories: list[list[int]],
) -> list[list[_Bar]]:
    """Return the ``_Bar`` for every patch of every container."""
    bars = []
    for series, (container, geometry, series_categories) in enumerate(
        zip(containers, geometries, categories, strict=True)
    ):
        values = _container_values(container, geometry)
        bars.append(
            [
                _Bar(patch, series, int(category), value, span, base, end)
                for patch, (span, base, end), category, value in zip(
                    container.patches, geometry, series_categories, values, strict=True
                )
            ]
        )
    return bars


def _container_values(container: BarContainer, geometry: list[tuple[tuple[float, float], float, float]]) -> list[float]:
    """Return the values the bars show: the container's ``datavalues``, or their lengths when it has none."""
    datavalues = getattr(container, "datavalues", None)
    if datavalues is not None and len(datavalues) == len(geometry):
        return [float(value) for value in datavalues]
    return [end - base for _, base, end in geometry]


def _values_table(bars: list[list[_Bar]], category_count: int) -> npt.NDArray[np.float64]:
    """Return the bars' values as a series-by-category array, NaN where a series has no bar."""
    table = np.full((len(bars), category_count), np.nan)
    for series in bars:
        for bar in series:
            table[bar.series, bar.category] = bar.value
    return table


def _series_labels(ax: Axes, containers: list[BarContainer]) -> list[str] | None:
    """Return the series' names: the containers' labels, or else the legend's texts; ``None`` if neither works.

    pandas and ``ax.bar(..., label=...)`` label their containers; seaborn leaves them as ``_container0`` etc. (a
    leading underscore is matplotlib's mark for "no label"), but its legend lists the hue levels in the same order
    as its containers. For a grid (e.g. seaborn's ``catplot``), that legend is the figure's rather than the axes'.
    """
    labels = [str(container.get_label()) for container in containers]
    if not any(label.startswith("_") for label in labels):
        return labels
    legend = _legend_of(ax)
    all_containers = [c for c in ax.containers if isinstance(c, BarContainer)]
    if legend is None or len(legend.get_texts()) != len(all_containers):
        return None
    legend_texts = [text.get_text() for text in legend.get_texts()]
    by_container = {id(container): text for container, text in zip(all_containers, legend_texts, strict=True)}
    return (
        [by_container[id(container)] for container in containers]
        if all(id(container) in by_container for container in containers)
        else None
    )


def _legend_of(ax: Axes) -> Legend | None:
    """Return the legend for ``ax``'s bars: its own, or else its figure's only legend (as seaborn's grids draw)."""
    legend = ax.get_legend()
    if legend is not None:
        return legend
    figure_legends = ax.figure.legends if ax.figure is not None else []
    return figure_legends[0] if len(figure_legends) == 1 else None
