"""The package without its optional dependencies: run in a subprocess with them blocked."""

import subprocess
import sys
import textwrap

import pytest

_BLOCK = """
import sys
for name in {blocked!r}:
    sys.modules[name] = None  # makes "import name" raise ImportError, as if it weren't installed
"""


def run(code, blocked=("numpy",)):
    """Run ``code`` in a new Python with ``blocked`` modules unavailable; return its output."""
    script = _BLOCK.format(blocked=list(blocked)) + textwrap.dedent(code)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_package_imports_without_numpy():
    """Importing the package itself doesn't need any optional dependency."""
    assert run("import mcrotk_utils; print('ok')") == "ok"


# The package itself, and one of its private modules: the package's check runs before any of its modules
@pytest.mark.parametrize("module", ["mcrotk_utils.numpy_utils", "mcrotk_utils.numpy_utils._interleaving"])
def test_numpy_utils_names_the_extra_when_numpy_is_missing(module):
    """Importing numpy_utils, or any of its modules, without numpy says which extra to install."""
    output = run(f"""
        try:
            import {module}
        except ModuleNotFoundError as error:
            print(error.name, "|", error)
    """)
    assert output.startswith("numpy |")
    assert "mcrotk-utils[numpy]" in output


def test_multiprocessing_utils_splitting_needs_no_optional_dependency():
    """The multiprocessing_utils package and its splitting functions import and work without numpy."""
    output = run("""
        from mcrotk_utils.multiprocessing_utils import allocate_workers, split_evenly, split_for_workers
        print(split_evenly(range(5), 2), allocate_workers([4, 1], 3), len(split_for_workers(range(5), 2)))
    """)
    assert output == "[[0, 1, 2], [3, 4]] [2, 1] 2"


# The name through the package (loaded lazily), and its private module
@pytest.mark.parametrize(
    "statement",
    [
        "from mcrotk_utils.multiprocessing_utils import parallel_map_with_rngs",
        "import mcrotk_utils.multiprocessing_utils._mapping",
    ],
)
def test_parallel_map_with_rngs_names_the_extra_when_numpy_is_missing(statement):
    """Importing parallel_map_with_rngs without numpy says which extra to install, naming the function."""
    output = run(f"""
        try:
            {statement}
        except ModuleNotFoundError as error:
            print(error.name, "|", error)
    """)
    assert output.startswith("numpy |")
    assert "parallel_map_with_rngs" in output
    assert "mcrotk-utils[numpy]" in output


def test_misc_utils_needs_no_optional_dependency():
    """misc_utils imports and works with numpy, pandas and scipy all unavailable."""
    output = run(
        """
        from mcrotk_utils.misc_utils import bisect_to_target, format_duration, max_width
        result = bisect_to_target(lambda n: n * n, 50, 0, 100)
        print(max_width([1, [22, {"k": 333}]]), format_duration(61), result.above.arg)
        """,
        blocked=("numpy", "pandas", "scipy"),
    )
    assert output == "3 1 minute, 1 second 8"


def test_package_imports_without_pandas():
    """Importing the package itself doesn't need pandas either."""
    assert run("import mcrotk_utils; print('ok')", blocked=("pandas",)) == "ok"


# The package itself, and one of its private modules: the package's check runs before any of its modules
@pytest.mark.parametrize("module", ["mcrotk_utils.pandas_utils", "mcrotk_utils.pandas_utils._concat"])
def test_pandas_utils_names_the_extra_when_pandas_is_missing(module):
    """Importing pandas_utils, or any of its modules, without pandas says which extra to install."""
    output = run(
        f"""
        try:
            import {module}
        except ModuleNotFoundError as error:
            print(error.name, "|", error)
        """,
        blocked=("pandas",),
    )
    assert output.startswith("pandas |")
    assert "mcrotk-utils[pandas]" in output


def test_stats_utils_imports_without_numpy_or_scipy():
    """Importing the stats_utils package itself needs nothing: its names are loaded when first used."""
    assert run("import mcrotk_utils.stats_utils; print('ok')", blocked=("numpy", "scipy")) == "ok"


def test_stats_utils_weighted_means_and_normal_params_need_only_numpy():
    """weighted_mean and NormalParams work without scipy (only the numpy extra installed)."""
    output = run(
        """
        from mcrotk_utils.stats_utils import NormalParams, hc1_inflation_factor, normalize_weights, weighted_mean
        print(weighted_mean([1.0, 3.0]), NormalParams(1.0, 0.0).generate_normal())
        """,
        blocked=("scipy",),
    )
    assert output == "2.0 1.0"


@pytest.mark.parametrize("name", ["weighted_mean", "NormalParams"])
def test_stats_utils_names_the_numpy_extra_when_numpy_is_missing(name):
    """The numpy-only names say to install the numpy extra when numpy is missing."""
    output = run(
        f"""
        try:
            from mcrotk_utils.stats_utils import {name}
        except ModuleNotFoundError as error:
            print(error.name, "|", error)
        """,
        blocked=("numpy",),
    )
    assert output.startswith("numpy |")
    assert "mcrotk-utils[numpy]" in output


@pytest.mark.parametrize("name", ["p_value_from_t_stat", "empirical_cdf", "HypothesisType", "random_covariance_matrix"])
def test_stats_utils_names_the_scipy_extra_when_scipy_is_missing(name):
    """The names that need scipy say to install the scipy extra when it's missing."""
    output = run(
        f"""
        try:
            from mcrotk_utils.stats_utils import {name}
        except ModuleNotFoundError as error:
            print(error.name, "|", error)
        """,
        blocked=("scipy",),
    )
    assert output.startswith("scipy |")
    assert "mcrotk-utils[scipy]" in output


# The package itself, and one of its private modules: the package's check runs before any of its modules
@pytest.mark.parametrize("module", ["mcrotk_utils.sparse_utils", "mcrotk_utils.sparse_utils._stacking"])
def test_sparse_utils_names_the_extra_when_scipy_is_missing(module):
    """Importing sparse_utils, or any of its modules, without scipy says which extra to install."""
    output = run(
        f"""
        try:
            import {module}
        except ModuleNotFoundError as error:
            print(error.name, "|", error)
        """,
        blocked=("scipy",),
    )
    assert output.startswith("scipy |")
    assert "mcrotk-utils[scipy]" in output


def test_package_imports_without_matplotlib():
    """Importing the package itself doesn't need matplotlib."""
    assert run("import mcrotk_utils; print('ok')", blocked=("matplotlib",)) == "ok"


# The package itself, and one of its private modules: the package's check runs before any of its modules
@pytest.mark.parametrize("module", ["mcrotk_utils.plotting_utils", "mcrotk_utils.plotting_utils._labels"])
def test_plotting_utils_names_the_extra_when_matplotlib_is_missing(module):
    """Importing plotting_utils, or any of its modules, without matplotlib says which extra to install."""
    output = run(
        f"""
        try:
            import {module}
        except ModuleNotFoundError as error:
            print(error.name, "|", error)
        """,
        blocked=("matplotlib",),
    )
    assert output.startswith("matplotlib |")
    assert "mcrotk-utils[matplotlib]" in output
