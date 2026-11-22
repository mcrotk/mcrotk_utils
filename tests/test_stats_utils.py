"""Tests for the ``mcrotk_utils.stats_utils`` package itself: its exports, and what its modules share.

(The functions themselves are tested in ``test_stats_utils_*.py``, one file per internal module.)
"""

import importlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from mcrotk_utils import stats_utils
from mcrotk_utils.stats_utils import (
    HypothesisType,
    ci_from_empirical_dist,
    empirical_cdf,
    empirical_sf,
    p_value_from_empirical_dist,
    weighted_mean,
)

UPPER, LOWER, TWO_SIDED = HypothesisType.ONE_SIDED_UPPER, HypothesisType.ONE_SIDED_LOWER, HypothesisType.TWO_SIDED


# --- Exports ---------------------------------------------------------------------------------------------------


def test_all_lists_exactly_the_exported_names():
    """__all__ (a literal list, for type checkers) names exactly the names the package maps to its modules."""
    assert sorted(stats_utils.__all__) == sorted(stats_utils._MODULE_OF_NAME)
    assert len(set(stats_utils.__all__)) == len(stats_utils.__all__)


@pytest.mark.parametrize("name", stats_utils.__all__)
def test_every_exported_name_comes_from_its_module(name):
    """Each public name is the object defined in its (private) module."""
    module = importlib.import_module(f"mcrotk_utils.stats_utils.{stats_utils._MODULE_OF_NAME[name]}")
    assert getattr(stats_utils, name) is getattr(module, name)


def test_unknown_names_raise_attribute_error():
    """A name the package doesn't export raises AttributeError (so hasattr and getattr defaults work)."""
    with pytest.raises(AttributeError, match="has no attribute 'no_such_function'"):
        stats_utils.no_such_function  # noqa: B018
    assert not hasattr(stats_utils, "no_such_function")


def test_dir_lists_the_exported_names_before_they_are_loaded():
    """dir() includes every public name, as if they had all been imported, even before any is used (checked in a
    fresh interpreter, since using a name caches it in the module)."""
    code = (
        "import sys, mcrotk_utils.stats_utils as s; print(sorted(set(s.__all__) - set(dir(s))), 'scipy' in sys.modules)"
    )
    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert output == "[] False"


def test_star_import_gives_the_exported_names():
    """from mcrotk_utils.stats_utils import * gives exactly the public names."""
    namespace: dict[str, object] = {}
    exec("from mcrotk_utils.stats_utils import *", namespace)
    assert set(namespace) - {"__builtins__"} == set(stats_utils.__all__)


# --- Shared input handling -------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean"])
def test_pandas_nullable_missing_values_count_as_nan(dtype):
    """pandas' NA in nullable columns is treated as NaN by every function taking values (in pandas 2 and 3)."""
    with_na = pd.Series([1, None, 0, 1], dtype=dtype)
    without = [1.0, 0.0, 1.0]
    assert weighted_mean(with_na, skip_na=True) == weighted_mean(without)
    assert empirical_cdf(0.5, with_na, skip_na=True) == empirical_cdf(0.5, without)
    assert empirical_sf(0.5, with_na, skip_na=True) == empirical_sf(0.5, without)
    assert p_value_from_empirical_dist(1, with_na, skip_na=True) == p_value_from_empirical_dist(1, without)
    assert np.array_equal(ci_from_empirical_dist(with_na, skip_na=True), ci_from_empirical_dist(without))
    with pytest.raises(ValueError, match="1 NaN values; pass skip_na=True"):
        empirical_cdf(0.5, with_na)
