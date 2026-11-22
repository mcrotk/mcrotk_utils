"""Statistics helpers: weighted means, hypothesis types, p-values and confidence intervals (from empirical
distributions or the t distribution), normal parameters, and random covariance and correlation matrices.

Import everything from here (e.g. ``from mcrotk_utils.stats_utils import weighted_mean``): how the code is split
into modules is internal, and may change.

Names are loaded lazily, when first used, so that each needs only its own dependencies: the weighted means and
``NormalParams`` need only numpy (``mcrotk-utils[numpy]``); everything else needs scipy (``mcrotk-utils[scipy]``,
which brings numpy).
"""

import importlib
from typing import TYPE_CHECKING, Any

# Every public name, with the (private) module defining it
_MODULE_OF_NAME = {
    "DEFAULT_ALPHA": "_inference",
    "HypothesisType": "_inference",
    "QuantileMethod": "_inference",
    "ci_from_empirical_dist": "_inference",
    "ci_from_t_dist": "_inference",
    "empirical_cdf": "_inference",
    "empirical_sf": "_inference",
    "negate_ci": "_inference",
    "p_value_from_empirical_dist": "_inference",
    "p_value_from_t_stat": "_inference",
    "hc1_inflation_factor": "_weights",
    "normalize_weights": "_weights",
    "weighted_mean": "_weights",
    "NormalParams": "_distributions",
    "correlation_from_covariance": "_covariance",
    "random_covariance_from_eigenvalues": "_covariance",
    "random_covariance_from_eigenvalues_and_variances": "_covariance",
    "random_covariance_from_variances": "_covariance",
    "random_covariance_matrix": "_covariance",
}

# A literal list, as type checkers require (a test checks it matches _MODULE_OF_NAME)
__all__ = [
    "DEFAULT_ALPHA",
    "HypothesisType",
    "NormalParams",
    "QuantileMethod",
    "ci_from_empirical_dist",
    "ci_from_t_dist",
    "correlation_from_covariance",
    "empirical_cdf",
    "empirical_sf",
    "hc1_inflation_factor",
    "negate_ci",
    "normalize_weights",
    "p_value_from_empirical_dist",
    "p_value_from_t_stat",
    "random_covariance_from_eigenvalues",
    "random_covariance_from_eigenvalues_and_variances",
    "random_covariance_from_variances",
    "random_covariance_matrix",
    "weighted_mean",
]

if TYPE_CHECKING:
    # what type checkers and IDEs see; at run time the names come from __getattr__ below
    from mcrotk_utils.stats_utils._covariance import (
        correlation_from_covariance,
        random_covariance_from_eigenvalues,
        random_covariance_from_eigenvalues_and_variances,
        random_covariance_from_variances,
        random_covariance_matrix,
    )
    from mcrotk_utils.stats_utils._distributions import NormalParams
    from mcrotk_utils.stats_utils._inference import (
        DEFAULT_ALPHA,
        HypothesisType,
        QuantileMethod,
        ci_from_empirical_dist,
        ci_from_t_dist,
        empirical_cdf,
        empirical_sf,
        negate_ci,
        p_value_from_empirical_dist,
        p_value_from_t_stat,
    )
    from mcrotk_utils.stats_utils._weights import hc1_inflation_factor, normalize_weights, weighted_mean


def __getattr__(name: str) -> Any:
    """Return a public name, importing the module that defines it on first use (PEP 562).

    Importing only that module means only its dependencies are needed (e.g. not scipy for ``weighted_mean``).
    """
    module_name = _MODULE_OF_NAME.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module_name}"), name)
    # cache it, so later lookups don't go through __getattr__
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the public names along with the module's own, as if they had all been imported."""
    return sorted({*globals(), *__all__})
