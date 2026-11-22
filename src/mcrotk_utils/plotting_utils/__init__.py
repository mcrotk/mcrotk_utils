"""Helpers for matplotlib plots: labelling bars with their values, shares, or comparisons with other bars.

Needs matplotlib, which ``mcrotk-utils`` installs only with its ``matplotlib`` extra (``mcrotk-utils[matplotlib]``,
which brings numpy). Works on plots made with matplotlib, pandas or seaborn. Import everything from here (e.g.
``from mcrotk_utils.plotting_utils import label_bars``): how the code is split into modules is internal, and may
change.

- ``label_bars``: label each bar on an axes with its value, its share of a total, or its difference, relative
  change or ratio from a reference bar (the previous, next, first or last one, or a named category or series),
  along the categories (e.g. week over week) or across the series (e.g. test against control); labels go outside
  the bars (clear of error bars), inside them, or at their middle.
- Type aliases for ``label_bars``' arguments: ``BarLabelMeasure``, ``BarLabelReference`` and ``BarLabelPosition``.
"""

# Checked once here: Python runs a package's __init__ before any of its modules, so this covers them all
try:
    import matplotlib  # noqa: F401
except ModuleNotFoundError as error:
    # matplotlib is optional for the package as a whole, so say which extra provides it, not just that it's missing
    raise ModuleNotFoundError(
        "mcrotk_utils.plotting_utils needs matplotlib: install mcrotk-utils with its extra, mcrotk-utils[matplotlib]",
        name="matplotlib",
    ) from error

# Plain imports, not lazy ones as in stats_utils: every module needs matplotlib, so loading names lazily would save
# no imports
from mcrotk_utils.plotting_utils._labels import BarLabelMeasure, BarLabelReference, label_bars
from mcrotk_utils.plotting_utils._placement import BarLabelPosition

__all__ = ["BarLabelMeasure", "BarLabelPosition", "BarLabelReference", "label_bars"]
