"""Where a bar's label goes: its anchor point, its offset in points, and its alignment (private)."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from mcrotk_utils.plotting_utils._bar_table import _Bar

# Where labels go: beyond the end of each bar, just inside its end, or at its middle
type BarLabelPosition = Literal["outside", "inside", "center"]

_POSITIONS = ("outside", "inside", "center")


@dataclass(frozen=True)
class _Placement:
    """A label's anchor (in data coordinates), its offset from it (in points), and its alignment."""

    xy: tuple[float, float]
    offset: tuple[float, float]
    horizontal_alignment: str
    vertical_alignment: str


def _check_position(position: str) -> None:
    """Raise ``ValueError`` unless ``position`` is one of the known positions."""
    if position not in _POSITIONS:
        raise ValueError(f"position must be one of {', '.join(map(repr, _POSITIONS))}, not {position!r}.")


def _line_points(ax: Axes) -> npt.NDArray[np.float64]:
    """Return the points of every line drawn on ``ax`` in data coordinates, as an array of shape (n, 2).

    Error bars are lines: seaborn draws each as a ``Line2D`` (with its caps), and matplotlib's ``errorbar`` (used by
    ``ax.bar(yerr=...)`` and pandas) as a ``LineCollection`` plus ``Line2D`` markers for the caps. Lines in other
    coordinates, such as ``axhline``'s (whose x runs across the axes, whatever the data), are left out.
    """
    points: list[npt.NDArray[np.float64]] = [np.empty((0, 2))]
    for line in ax.lines:
        if line.get_transform() == ax.transData:
            xs = np.asarray(line.get_xdata(), dtype=float)
            ys = np.asarray(line.get_ydata(), dtype=float)
            points.append(np.column_stack(np.broadcast_arrays(xs, ys)))
    for collection in ax.collections:
        if isinstance(collection, LineCollection) and collection.get_transform() == ax.transData:
            points.extend(np.asarray(segment, dtype=float).reshape(-1, 2) for segment in collection.get_segments())
    stacked = np.concatenate(points)
    return stacked[np.all(np.isfinite(stacked), axis=1)]


def _placement(
    bar: _Bar,
    position: BarLabelPosition,
    padding: float,
    *,
    vertical: bool,
    value_axis_inverted: bool,
    points: npt.NDArray[np.float64] | None,
) -> _Placement:
    """Return where to put ``bar``'s label.

    ``points`` are the line points to keep an outside label clear of (see ``_line_points``), or ``None`` to ignore
    lines; only those within the bar's span along the category axis count.
    """
    outward = 1.0 if bar.end >= bar.base else -1.0  # along the value axis, from the base towards the end
    on_screen = -outward if value_axis_inverted else outward  # the same direction as it appears
    if position == "center":
        return _placed((bar.base + bar.end) / 2, 0.0, bar, vertical, alignment=0.0)
    if position == "inside":
        return _placed(bar.end, -on_screen * padding, bar, vertical, alignment=-on_screen)
    anchor = _clear_of_lines(bar, outward, vertical, points) if points is not None else bar.end
    return _placed(anchor, on_screen * padding, bar, vertical, alignment=on_screen)


def _clear_of_lines(bar: _Bar, outward: float, vertical: bool, points: npt.NDArray[np.float64]) -> float:
    """Return the farthest point outwards along the value axis of the bar's end and the lines within its span."""
    category_coordinates, value_coordinates = (points[:, 0], points[:, 1]) if vertical else (points[:, 1], points[:, 0])
    low, high = bar.span
    within = value_coordinates[(category_coordinates >= low) & (category_coordinates <= high)]
    candidates = np.append(within, bar.end)
    return float(np.max(candidates) if outward > 0 else np.min(candidates))


def _placed(anchor: float, offset: float, bar: _Bar, vertical: bool, *, alignment: float) -> _Placement:
    """Return the _placement for a label anchored at ``anchor`` on the value axis, at the middle of the bar's span.

    ``alignment`` says which way the text extends from its anchor on screen along the value axis: positive for
    up or right, negative for down or left, 0 for centered.
    """
    middle = sum(bar.span) / 2
    if vertical:
        vertical_alignment = "bottom" if alignment > 0 else "top" if alignment < 0 else "center"
        return _Placement((middle, anchor), (0.0, offset), "center", vertical_alignment)
    horizontal_alignment = "left" if alignment > 0 else "right" if alignment < 0 else "center"
    return _Placement((anchor, middle), (offset, 0.0), horizontal_alignment, "center")
