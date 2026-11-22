# mcrotk-utils

Utilities for general Python, `numpy`, `pandas`, sparse matrices, statistics, and plotting, one module per topic. The package isn't published on PyPI; install it from GitHub (see below). When part of it grows big and general enough to release, that part moves out into its own package (as the caching code did with [`mcrotk-cache-utils`](https://pypi.org/project/mcrotk-cache-utils/)) and is deleted from here. Released packages aren't re-exported from here: import them under their own names.

The package is shared as it is, without promises of stability: its interfaces can change between versions, so a project that depends on it may want to pin a tagged version (see below).

## Modules

The package has one module per topic, each with its own README listing what it contains, with examples. Each module's third-party dependencies come with an extra of the same name. Import from the module itself; the package's `__init__.py` imports nothing, so a project only needs the extras for the modules it uses.

| Module | Requires | Description |
|---|---|---|
| [`misc_utils`](src/mcrotk_utils/misc_utils/README.md) | nothing | Checks, formatting, mappings, atomic file writes, sorting by keys in different directions, and bisection to a target value |
| [`multiprocessing_utils`](src/mcrotk_utils/multiprocessing_utils/README.md) | `numpy`, for the parallel map only | A parallel map giving each call its own random generator, and splitting work among workers |
| [`numpy_utils`](src/mcrotk_utils/numpy_utils/README.md) | `numpy` | Interleaving arrays, comparing entries along an axis, and type aliases for arrays and seeds |
| [`pandas_utils`](src/mcrotk_utils/pandas_utils/README.md) | `pandas` | Making, concatenating, merging, and adding to DataFrames, and dtypes for their columns |
| [`plotting_utils`](src/mcrotk_utils/plotting_utils/README.md) | `matplotlib` | Labelling bars with their values, shares, or comparisons with other bars |
| [`sparse_utils`](src/mcrotk_utils/sparse_utils/README.md) | `scipy` | Converting and stacking matrices that may be dense or sparse |
| [`stats_utils`](src/mcrotk_utils/stats_utils/README.md) | `scipy`, or `numpy` for some | Weighted means, p-values and confidence intervals, and random covariance matrices |

Imports look like this:

```python
from mcrotk_utils.misc_utils import check_positive, format_duration
from mcrotk_utils.numpy_utils import SeedLike, interleave
from mcrotk_utils.pandas_utils import combine_dataframes
from mcrotk_utils.plotting_utils import label_bars
from mcrotk_utils.sparse_utils import as_dense, ps_hstack
from mcrotk_utils.stats_utils import p_value_from_empirical_dist, weighted_mean
```

## Using it in a project

A project lists the package with the extras it needs, and tells uv where to get it in `[tool.uv.sources]`. There are three options: the latest commit on GitHub, a tagged version from GitHub, or a local checkout as an editable install.

```toml
[project]
dependencies = [
    "mcrotk-utils[numpy,pandas,scipy,matplotlib]",
]

[tool.uv.sources]
# one of:
mcrotk-utils = { git = "https://github.com/mcrotk/mcrotk_utils" }  # the latest commit on the default branch
# mcrotk-utils = { git = "https://github.com/mcrotk/mcrotk_utils", tag = "vX.Y.Z" }  # a tagged version
# mcrotk-utils = { path = "../mcrotk_utils", editable = true }  # a local checkout, relative to the project
```

The options differ in when the project picks up changes:

- **The latest commit:** `uv lock` records the commit that was latest at the time in the project's `uv.lock`, and the project stays on it until `uv lock --upgrade-package mcrotk-utils` moves it to the latest again. This includes changes made since the last tagged version.
- **A tagged version:** replace `vX.Y.Z` with one of the repository's tags (`git tag -l -n99` lists them with what each version adds). The project stays on that version until the tag is changed.
- **A local checkout:** changes in the checkout show up immediately.

uv writes these for you: `uv add "mcrotk-utils[numpy] @ git+https://github.com/mcrotk/mcrotk_utils"` writes the first form (add `--tag vX.Y.Z` for the second), and `uv add --editable <path to the checkout> --extra numpy` the third; list more extras as `[numpy,pandas]` or with more `--extra` options. A project meant for release on PyPI can't depend on this package (PyPI rejects path and git dependencies), so it copies the few helpers it needs as private code instead.

## License

MIT; see [`LICENSE`](LICENSE).

## Development

The checks are run locally (there's no CI):

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run mypy
```

The tests include each module's README: its examples are run, and every public name must be listed in it. The images in the READMEs are drawn from the examples shown with them; after changing one of those examples, redraw the images with `uv run python scripts/make_readme_images.py`.
