# `mcrotk_utils.misc_utils`

General helpers that need only the standard library: checks that return the value they check, formatting, mappings that refuse duplicates, writing files atomically, sorting by several keys in different directions, and finding where a function crosses a target value. It needs none of the package's extras. Each function's docstring has the details: its arguments, result, and errors.

[Back to the package README](../../../README.md)

## Contents

The names are grouped below by what they're for.

### Checks

These return the value they check, so they can be used inline (e.g. `self.n = check_positive(n, "n")`). When the check fails, they raise an error naming the argument. Unlike an `assert`, they also run under `python -O`.

- `check_in_range`: the value lies in a range, with each end optional, and inclusive or exclusive.
- `check_positive` and `check_nonnegative`: the value is greater than 0, or at least 0 (NaN is neither).
- `check_not_none`: the value isn't `None`. Type checkers then treat a value of type `X | None` as an `X`.

### Formatting

These turn values into text.

- `max_width`: the length of the longest value when formatted, e.g. for aligning printed columns.
- `format_duration`: a number of seconds (or a `timedelta`) in words, such as "1 hour, 2 minutes, 3.5 seconds".
- `remove_punctuation`: text without its punctuation, either ASCII punctuation only or all of Unicode's.

### Mappings

These build dicts, and raise an error on duplicates rather than silently keeping only one of them.

- `invert_dict`: a dict mapping each value back to its key.
- `index_map`: a dict mapping each item to its position.

### Files and functions

These are small building blocks.

- `write_atomically`: write a file so that its path always holds a complete version: the previous one until the new one is fully written. An interruption never leaves a half-written file, so a file's existence shows that its write finished.
- `identity`: return the argument unchanged, as a default for arguments that take a function.

### Sorting

Python's `sorted` has one `reverse` for the whole sort, and negating a key to sort it the other way only works for numbers.

- `sort_with_directions`: sort by several keys, each ascending or descending on its own, for keys of any type that can be compared, such as strings or dates.

### Bisection

`bisect_to_target` finds where a monotonic function crosses a target value. Unlike `scipy`'s root finders, it can stop when the function's values are close enough (not only its arguments), carry an expensive result (such as a fitted model) along so nothing needs recomputing, return the points on both sides of the target, search over integers, and resume from points already evaluated.

- `bisect_to_target`: the search itself.
- `BisectionResult`: what it returns: the evaluated points just below and just above the target, the closest of the two, the number of evaluations, and why it stopped.
- `BisectionPoint`: one evaluated point: its argument, what the function returned, and the value compared with the target.
- `StopReason`: the type of the reasons the search can stop for.

## Examples

These examples show the functions most worth knowing about.

### Sorting in different directions

Tasks by due date, latest first, with ties broken by name from A to Z. Negating the key, the usual trick for sorting one key in reverse, doesn't work for dates or strings:

```pycon
>>> from datetime import date
>>> from mcrotk_utils.misc_utils import sort_with_directions
>>> tasks = [
...     ("write report", date(2026, 3, 2)),
...     ("send invites", date(2026, 3, 9)),
...     ("order food", date(2026, 3, 2)),
...     ("book venue", date(2026, 3, 9)),
... ]
>>> for name, due in sort_with_directions(tasks, key=lambda task: (task[1], task[0]), reverse=[True, False]):
...     print(due, name)
2026-03-09 book venue
2026-03-09 send invites
2026-03-02 order food
2026-03-02 write report
```

### Bisection over integers

With integer ends, `bisect_to_target` tries only integers. Here it finds the smallest sample size giving a two-sided z-test at the 5% level a power of at least 80%, for an effect of 0.3 standard deviations. Power increases with the sample size, so that's the point just above the target:

```pycon
>>> from statistics import NormalDist
>>> from mcrotk_utils.misc_utils import bisect_to_target
>>> normal = NormalDist()
>>> def power(n):
...     return normal.cdf(0.3 * n**0.5 - normal.inv_cdf(0.975))
>>> result = bisect_to_target(power, 0.8, 1, 1000)
>>> result.above.arg, round(result.above.value, 3)
(88, 0.804)
>>> result.below.arg, round(result.below.value, 3)
(87, 0.799)
```

### Bisection with expensive evaluations

When each evaluation is expensive, the function can return everything it computed, with `value` picking out the number to compare with the target. Here a stand-in for fitting a model searches for the smallest penalty whose error is at most 0.25, stopping once the errors on the two sides are within 0.01 of each other. The error decreases as the penalty grows, so that fit is the one just below the target, and it's returned without being recomputed:

```pycon
>>> def fit(penalty):
...     return {"penalty": penalty, "error": 1 / (1 + penalty)}
>>> result = bisect_to_target(fit, 0.25, 0.0, 10.0, value=lambda model: model["error"], value_tol=0.01)
>>> model = result.below.result
>>> model["penalty"], round(model["error"], 4)
(3.125, 0.2424)
>>> result.iterations, result.stop_reason
(6, 'value_tol')
```

### Checks and formatting

The checks return the value, so they can be used inline, or raise an error naming the argument:

```pycon
>>> from mcrotk_utils.misc_utils import check_positive, format_duration
>>> check_positive(3, "n")
3
>>> check_positive(-1, "n")
Traceback (most recent call last):
...
ValueError: n must be positive, not -1.
```

`format_duration` gives a number of seconds in words, e.g. for reporting how long a run took:

```pycon
>>> format_duration(3723.5, digits=1)
'1 hour, 2 minutes, 3.5 seconds'
```
