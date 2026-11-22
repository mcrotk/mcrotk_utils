"""pandas helpers: making, concatenating, merging and adding to DataFrames, and dtypes for their columns.

Needs pandas, which ``mcrotk-utils`` installs only with its ``pandas`` extra (``mcrotk-utils[pandas]``). Import
everything from here (e.g. ``from mcrotk_utils.pandas_utils import combine_dataframes``): how the code is split into
modules is internal, and may change.

- Making DataFrames: ``as_dataframe`` (a DataFrame, or a CSV file read into one) and ``matrices_to_long_df`` (named
  matrices into one long DataFrame).
- Dtypes: ``column_dtypes`` (categorical and nullable integer dtypes for ``astype``) and ``to_datetime_columns``.
- Concatenating: ``combine_dataframes``, which keeps categorical columns categorical when their categories differ.
- Merging: ``merge_with_preferences``, which merges on the index, combining chosen columns with a preferred side.
- Adding columns: ``join_columns`` (several columns into one of labels) and ``add_lags`` (lagged copies of columns,
  by period).
- ``is_binary`` and ``num_non_binary``.
"""

# Checked once here: Python runs a package's __init__ before any of its modules, so this covers them all
try:
    import pandas  # noqa: F401
except ModuleNotFoundError as error:
    # pandas is optional for the package as a whole, so say which extra provides it, not just that it's missing
    raise ModuleNotFoundError(
        "mcrotk_utils.pandas_utils needs pandas: install mcrotk-utils with its extra, mcrotk-utils[pandas]",
        name="pandas",
    ) from error

# Plain imports, not lazy ones as in stats_utils: every module needs pandas, the package's one dependency, so
# loading names lazily would save no imports
from mcrotk_utils.pandas_utils._binary import is_binary, num_non_binary
from mcrotk_utils.pandas_utils._columns import add_lags, join_columns
from mcrotk_utils.pandas_utils._concat import combine_dataframes
from mcrotk_utils.pandas_utils._creation import as_dataframe, matrices_to_long_df
from mcrotk_utils.pandas_utils._dtypes import column_dtypes, to_datetime_columns
from mcrotk_utils.pandas_utils._merging import merge_with_preferences

__all__ = [
    "add_lags",
    "as_dataframe",
    "column_dtypes",
    "combine_dataframes",
    "is_binary",
    "join_columns",
    "matrices_to_long_df",
    "merge_with_preferences",
    "num_non_binary",
    "to_datetime_columns",
]
