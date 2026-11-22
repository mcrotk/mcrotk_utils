# `mcrotk_utils.multiprocessing_utils`

Helpers for running work in parallel: a parallel map that gives each call its own random generator, and functions for splitting work among workers. `parallel_map_with_rngs` needs `numpy` (`mcrotk-utils[numpy]`); the splitting functions need only the standard library. Names are loaded when first used, so importing one doesn't need the others' dependencies. Each function's docstring has the details: its arguments, result, and errors.

[Back to the package README](../../../README.md)

## Contents

The names fall into two groups: the parallel map, and splitting work.

### Parallel map with random generators

Simulations run in parallel need random numbers that don't depend on how the calls happen to be spread over the workers.

- `parallel_map_with_rngs`: call a function once per set of arguments, in worker processes by default, or in any executor or pool. Each call gets its own random generator, spawned from one seed, so the results depend only on the seed and the arguments: not on the number of workers, the chunk size, or which worker runs which call.

### Splitting work

These split jobs among workers so that the largest share is as small as possible, for work that's handed out in batches.

- `split_evenly`: split a list into consecutive parts whose lengths differ by at most one.
- `allocate_workers`: allocate workers among groups of jobs that can't be mixed, minimizing the most jobs any worker gets.
- `split_for_workers`: split items into batches for workers, keeping groups apart, with the largest batch as small as possible.
- `Batch`: one batch from `split_for_workers`: its group, its number within the group, and its items.

## Examples

These examples show the main functions.

### A parallel map with random generators

With a thread pool (so that the function can be defined right here), the results are the same for two workers and for three:

```pycon
>>> from concurrent.futures import ThreadPoolExecutor
>>> from mcrotk_utils.multiprocessing_utils import parallel_map_with_rngs
>>> def noisy(x, rng):
...     return x + rng.normal()
>>> with ThreadPoolExecutor(max_workers=2) as pool:
...     with_two = parallel_map_with_rngs(noisy, [1, 2, 3], executor=pool, rng=42)
>>> with ThreadPoolExecutor(max_workers=3) as pool:
...     with_three = parallel_map_with_rngs(noisy, [1, 2, 3], executor=pool, rng=42)
>>> [round(result, 3) for result in with_two]
[1.418, 3.254, 1.351]
>>> with_two == with_three
True
```

Without an `executor`, the calls run in a new process pool. The function, its arguments, and its results must then be picklable, so the function is defined at the top level of a module. The calls must also start under `if __name__ == "__main__":`, because new worker processes start by importing the main module: that's how they start on macOS and Windows, and on Linux from Python 3.14. A complete script:

```python
from mcrotk_utils.multiprocessing_utils import parallel_map_with_rngs


def simulate(level, rng):
    """Simulate one scenario: here, the mean of 1,000 draws around its level."""
    return float(rng.normal(level, 1, size=1000).mean())


if __name__ == "__main__":
    results = parallel_map_with_rngs(simulate, [0.0, 1.0, 2.0], max_workers=3, rng=2026)
    print(results)
```

### Splitting work among workers

Splitting nine jobs from two groups among four workers, without mixing the groups: the group of seven gets three workers and the group of two gets one, so no worker gets more than three jobs. `allocate_workers` gives just the allocation:

```pycon
>>> from mcrotk_utils.multiprocessing_utils import allocate_workers, split_for_workers
>>> jobs = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "b1", "b2"]
>>> for batch in split_for_workers(jobs, 4, key=lambda job: job[0]):
...     print(batch.key, batch.index, batch.items)
a 0 ['a1', 'a2', 'a3']
a 1 ['a4', 'a5']
a 2 ['a6', 'a7']
b 0 ['b1', 'b2']
>>> allocate_workers([7, 2], 4)
[3, 1]
```
