# `mcrotk_utils.stats_utils`

Statistics helpers: weighted means, p-values and confidence intervals (from empirical distributions such as permutation or bootstrap results, or from the t distribution), normal and lognormal parameters, and random covariance and correlation matrices. Most of it needs `scipy` (`mcrotk-utils[scipy]`, which brings `numpy`); the weighted means and `NormalParams` need only `numpy` (`mcrotk-utils[numpy]`). Names are loaded when first used, so importing those doesn't need `scipy`. Each function's docstring has the details: its arguments, result, and errors.

[Back to the package README](../../../README.md)

## Contents

The names are grouped below by what they're for.

### Weighted means

These need only `numpy`.

- `weighted_mean`: the weighted mean, with missing values either raising an error or skipped together with their weights.
- `normalize_weights`: weights scaled to add up to a given total (1 by default).
- `hc1_inflation_factor`: how much larger the robust (HC1) standard error of a weighted mean is than the ordinary one, computed directly rather than by fitting a regression.

### Hypotheses

These describe the test being done.

- `HypothesisType`: the alternative hypothesis: two-sided, or one-sided in the upper or lower tail. Functions taking one also accept names for it, read leniently (e.g. `"two-tailed"`, `"greater"`, or `"one-sided decrease"`).
- `DEFAULT_ALPHA`: the significance level used by default, 0.05.

### Empirical distributions

These work from an array of values making up a distribution, such as a statistic recomputed under random re-assignments (a permutation test) or an estimate recomputed on bootstrap samples.

- `empirical_cdf` and `empirical_sf`: the fraction of the values at or below a value, or above it.
- `p_value_from_empirical_dist`: the p-value of an observed statistic, by default counting the observed value as one of the values, which keeps the test valid and the p-value above 0.
- `ci_from_empirical_dist`: a percentile confidence interval.
- `QuantileMethod`: the type of `ci_from_empirical_dist`'s `quantile_method` argument (`numpy`'s quantile methods).

### The t distribution

These work for one test or interval, or for several at once, each with its own hypothesis type if wanted.

- `p_value_from_t_stat`: the p-value of a t-statistic, keeping its precision when it's tiny.
- `ci_from_t_dist`: the confidence interval `estimate ± t * se`, or a one-sided interval.
- `negate_ci`: the interval for the negated quantity, `[-upper, -lower]`, e.g. for B minus A from an interval for A minus B.

### Normal and lognormal values

This needs only `numpy`.

- `NormalParams`: a normal distribution's mean and standard deviation, checked, generating normal values, or lognormal ones whose logarithm has this mean and standard deviation.

### Random covariance matrices

These generate covariance matrices for simulations, controlling their eigenvalues, their variances (the diagonal), or both.

- `random_covariance_matrix`: one function for all the cases, with the eigenvalues and variances each given, drawn as lognormal, or left free.
- `random_covariance_from_eigenvalues`: exactly the given eigenvalues, with uniformly random eigenvectors.
- `random_covariance_from_variances`: exactly the given variances, with a random correlation matrix.
- `random_covariance_from_eigenvalues_and_variances`: exactly the given eigenvalues and variances, when that's possible.
- `correlation_from_covariance`: the correlation matrix of a covariance matrix, exactly valid: a diagonal of exactly 1, exactly symmetric, and entries within [-1, 1] despite rounding.

## Examples

These examples show the main functions.

### Weighted means

A weighted mean, without and with a missing value skipped (its weight is dropped too):

```pycon
>>> import numpy as np
>>> from mcrotk_utils.stats_utils import weighted_mean
>>> weighted_mean([1.0, 2.0, 4.0], [1, 1, 2])
2.75
>>> weighted_mean([1.0, np.nan, 4.0], [1, 1, 2], skip_na=True)
3.0
```

### A permutation test

Does the treated group have a higher mean? The null distribution of the difference in means comes from shuffling the group labels 999 times:

```pycon
>>> from mcrotk_utils.stats_utils import p_value_from_empirical_dist
>>> rng = np.random.default_rng(2)
>>> control, treated = rng.normal(10, 2, size=20), rng.normal(11, 2, size=20)
>>> observed = float(treated.mean() - control.mean())
>>> pooled = np.concatenate([control, treated])
>>> null_differences = []
>>> for _ in range(999):
...     shuffled = rng.permutation(pooled)
...     null_differences.append(shuffled[20:].mean() - shuffled[:20].mean())
>>> round(observed, 2), int(np.sum(np.array(null_differences) >= observed))
(1.21, 29)
```

The observed difference counts as one of the draws, since under the null hypothesis it's just as likely as any of them. That keeps the test valid, and the p-value above 0: it's (29 + 1) / (999 + 1).

```pycon
>>> p_value_from_empirical_dist(observed, null_differences, "greater")
0.03
```

### The t distribution

Two intervals in one call, each with its own hypothesis type: a two-sided 95% interval for an estimate of 10 with a standard error of 2, and a one-sided one (for an increase) for an estimate of 3 with a standard error of 1.5, both with 20 degrees of freedom:

```pycon
>>> from mcrotk_utils.stats_utils import ci_from_t_dist, p_value_from_t_stat
>>> intervals = ci_from_t_dist([10.0, 3.0], [2.0, 1.5], 20, hypothesis_type=["two-sided", "upper"])
>>> intervals.round(2)
array([[ 5.83, 14.17],
       [ 0.41,   inf]])
```

Each interval holds exactly the values that the matching t-test doesn't reject: testing the first estimate against the interval's lower end gives a p-value of exactly 5%.

```pycon
>>> round(p_value_from_t_stat((10.0 - intervals[0, 0]) / 2.0, 20), 6)
0.05
```

### Random covariance matrices

A random covariance matrix with given eigenvalues:

```pycon
>>> from mcrotk_utils.stats_utils import random_covariance_matrix
>>> covariance = random_covariance_matrix(eigenvalues=[1, 2, 3], rng=0)
>>> np.linalg.eigvalsh(covariance).round(6)
array([1., 2., 3.])
```
