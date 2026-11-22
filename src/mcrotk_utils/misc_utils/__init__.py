"""General helpers using only the standard library: checks, formatting, mappings, persistence, sorting and bisection.

Needs no third-party packages, so it works without any of ``mcrotk-utils``' extras. Import everything from here
(e.g. ``from mcrotk_utils.misc_utils import format_duration``): how the code is split into modules is internal, and
may change.

- Checks that return the checked value: ``check_in_range``, ``check_positive``, ``check_nonnegative`` and
  ``check_not_none``.
- Formatting: ``max_width`` (for aligning printed values), ``format_duration`` (a number of seconds in words) and
  ``remove_punctuation``.
- Mappings: ``invert_dict`` and ``index_map``, which raise on duplicates rather than silently losing entries.
- Persistence: ``write_atomically`` (write a file so that its path always holds a complete version).
- ``identity``.
- ``sort_with_directions``: sort by several keys, each ascending or descending on its own.
- ``bisect_to_target`` (with ``BisectionPoint``, ``BisectionResult`` and ``StopReason``): where a monotonic function
  crosses a target value.
"""

# Plain imports, not lazy ones as in stats_utils: nothing here has optional dependencies, and importing everything
# takes a few milliseconds
from mcrotk_utils.misc_utils._bisection import BisectionPoint, BisectionResult, StopReason, bisect_to_target
from mcrotk_utils.misc_utils._checks import check_in_range, check_nonnegative, check_not_none, check_positive
from mcrotk_utils.misc_utils._formatting import format_duration, max_width, remove_punctuation
from mcrotk_utils.misc_utils._functions import identity
from mcrotk_utils.misc_utils._mappings import index_map, invert_dict
from mcrotk_utils.misc_utils._persistence import write_atomically
from mcrotk_utils.misc_utils._sorting import sort_with_directions

__all__ = [
    "BisectionPoint",
    "BisectionResult",
    "StopReason",
    "bisect_to_target",
    "check_in_range",
    "check_nonnegative",
    "check_not_none",
    "check_positive",
    "format_duration",
    "identity",
    "index_map",
    "invert_dict",
    "max_width",
    "remove_punctuation",
    "sort_with_directions",
    "write_atomically",
]
