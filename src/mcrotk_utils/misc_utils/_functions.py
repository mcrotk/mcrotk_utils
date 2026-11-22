"""Small general-purpose functions (private).

Public through ``mcrotk_utils.misc_utils``. Uses only the standard library.
"""


def identity[T](value: T, /) -> T:
    """Return ``value`` unchanged: a default for arguments taking a function, where "do nothing" is wanted.

    Unlike ``lambda x: x``, it can be pickled (e.g. sent to worker processes), and type checkers keep the type
    (``identity(3)`` is an ``int``).
    """
    return value
