# `mcrotk_utils.numpy_utils`

`numpy` helpers: interleaving arrays in blocks, comparing entries with reference entries along an axis (such as each week with the week before), and type aliases for arrays and random seeds. It needs `numpy` (`mcrotk-utils[numpy]`). Each function's docstring has the details: its arguments, result, and errors.

[Back to the package README](../../../README.md)

## Contents

The names are grouped below by what they're for.

### Array operations

These work along any axis of arrays with any number of dimensions.

- `interleave`: interleave arrays along an axis, taking a block of entries at a time from each in turn. The arrays may have different lengths along the axis.
- `compare_along`: compare each entry with a reference entry along an axis (the previous, next, first, or last one, or one at a given index), as a difference, a relative change, a ratio, or with any function. Missing values either keep their places or are skipped over when finding a reference.
- `ComparisonReference` and `ComparisonMeasure`: the types of `compare_along`'s `reference` and `measure` arguments.

### Type aliases

These are for annotating code that uses `numpy`.

- `NDArrayInt` and `NDArrayFloat`: arrays of integers or floats of any width.
- `SeedLike`: anything `np.random.default_rng` accepts as a seed: `None`, an integer or a sequence or array of them, a `SeedSequence`, a `BitGenerator`, or a `Generator`.

## Examples

These examples show the two functions.

### `interleave`

Here the rows of `assignments` are periods and its columns are units, with 1 for treated and 0 for not. Interleaving it with its opposite, two rows at a time, follows each pair of periods with the same two periods switched:

```pycon
>>> import numpy as np
>>> from mcrotk_utils.numpy_utils import compare_along, interleave
>>> assignments = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]])
>>> interleave(assignments, 1 - assignments, blocksize=2)
array([[1, 0, 1],
       [0, 1, 1],
       [0, 1, 0],
       [1, 0, 0],
       [1, 1, 0],
       [0, 0, 1],
       [0, 0, 1],
       [1, 1, 0]])
```

### `compare_along`

The relative change from each week to the next, for two regions (one per row), with a missing week in the first:

```pycon
>>> weekly = np.array([[100, 110, np.nan, 121], [50, 55, 60, 66]])
>>> compare_along(weekly, "previous", "relative").round(3)
array([[  nan, 0.1  ,   nan,   nan],
       [  nan, 0.1  , 0.091, 0.1  ]])
```

With `skip_missing=True`, the week after the missing one is compared with the week before it instead:

```pycon
>>> compare_along(weekly, "previous", "relative", skip_missing=True).round(3)
array([[  nan, 0.1  ,   nan, 0.1  ],
       [  nan, 0.1  , 0.091, 0.1  ]])
```

Each week as a ratio to the first week (the first week itself, being the reference, gets NaN):

```pycon
>>> compare_along(weekly, "first", "ratio").round(3)
array([[ nan, 1.1 ,  nan, 1.21],
       [ nan, 1.1 , 1.2 , 1.32]])
```

Along the other axis, `axis=0`, each region is compared with the first region in the same week:

```pycon
>>> compare_along(weekly, "first", "ratio", axis=0).round(3)
array([[  nan,   nan,   nan,   nan],
       [0.5  , 0.5  ,   nan, 0.545]])
```
