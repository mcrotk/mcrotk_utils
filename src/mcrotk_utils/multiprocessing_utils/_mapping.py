"""Parallel maps with reproducible random numbers: each call gets its own random generator (private).

Public through ``mcrotk_utils.multiprocessing_utils``. Needs numpy.
"""

import functools
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Protocol

try:
    import numpy as np
except ModuleNotFoundError as error:
    # numpy is optional for the package as a whole, so say which extra provides it, not just that it's missing
    raise ModuleNotFoundError(
        "mcrotk_utils.multiprocessing_utils' parallel_map_with_rngs needs numpy: install mcrotk-utils with its "
        "extra, mcrotk-utils[numpy]",
        name="numpy",
    ) from error

from mcrotk_utils.numpy_utils import SeedLike


class _SupportsMap(Protocol):
    """Anything with a ``map(func, iterable)`` method returning the results in order (executors and pools)."""

    def map(self, func: Callable[[Any], Any], iterable: Iterable[Any], /) -> Iterable[Any]:
        """Return ``func(item)`` for each item, in order."""
        ...


def parallel_map_with_rngs[R](
    func: Callable[..., R],
    *iterables: Iterable[Any],
    num_calls: int | None = None,
    max_workers: int | None = None,
    executor: _SupportsMap | None = None,
    chunksize: int | None = None,
    rng: SeedLike = None,
) -> list[R]:
    """Call ``func`` once per set of arguments, in parallel, giving each call its own random generator.

    Call ``i`` is ``func(*args_i, rng=rngs[i])``, where ``args_i`` holds the ``i``-th item of each of
    ``iterables`` (as with ``map``), and ``rngs = np.random.default_rng(rng).spawn(n)`` gives statistically
    independent generators, one per call, in call order. So the results depend only on ``rng`` and the
    arguments: not on the number of workers, ``chunksize``, the executor or pool, or which worker runs which
    call. With no ``iterables``, ``func(rng=rngs[i])`` is called ``num_calls`` times.

    - ``iterables``: must all have the same length (``ValueError`` otherwise). They're read in full before any
      call starts, so they may be iterators.
    - ``num_calls``: the number of calls. Needed when there are no ``iterables``; with them, it's optional, and a
      ``ValueError`` if it differs from their length.
    - ``max_workers``: the number of worker processes (default: ``ProcessPoolExecutor``'s, the number of CPUs
      the process may use, from Python 3.13). Can't be combined with ``executor``.
    - ``executor``: run the calls here instead of in a new ``ProcessPoolExecutor``: anything with a
      ``map(func, iterable)`` method returning the results in order. It's left running, for the caller to shut
      down.

      - Works (tested): ``concurrent.futures``' ``ProcessPoolExecutor`` (also with an ``initializer``, e.g. to
        set up logging in each worker) and ``ThreadPoolExecutor``; ``multiprocessing.Pool``; pathos'
        ``ProcessPool`` and ``ThreadPool``.
      - Doesn't work, so refused with a ``TypeError``: pathos' ``ParallelPool`` (and its subclasses). It sends
        functions to its workers as source code, so it can't send the ``functools.partial`` that each call is
        wrapped in: its workers fail, and its ``map`` returns ``None`` for those calls, or never returns when
        there are more calls than workers.
      - Others with such a ``map`` should work, but aren't tested.
    - ``chunksize``: calls sent to a worker process at a time, passed to ``map`` as ``chunksize=`` (so the
      executor or pool must accept it). By default it isn't passed, so the executor's or pool's own default
      applies: 1 for ``ProcessPoolExecutor``, and a size from the number of calls and workers for
      ``multiprocessing.Pool``. Larger chunks reduce the overhead per call when there are many short calls;
      thread pools ignore it.
    - ``rng``: the seed the generators are spawned from: anything ``np.random.default_rng`` accepts. The same
      int (or sequence of ints) gives the same results every time. A ``Generator``, ``BitGenerator`` or
      ``SeedSequence`` keeps count of what has been spawned from it, so passing the same one again gives new
      generators (the ones that spawning more at once would have given); a generator's own stream isn't drawn
      from, so it can be used as before.

    Returns the results in call order. A new process pool is shut down before returning, after its workers have
    finished. With the standard library's process pools (the default, ``ProcessPoolExecutor`` and
    ``multiprocessing.Pool``), ``func``, its arguments and its results must be picklable, so ``func`` can be a
    module-level function, a bound method of a picklable object, or a ``functools.partial`` of one, but not a
    lambda or a nested function. Thread pools and pathos' ``ProcessPool`` (which pickles with ``dill``) also
    accept lambdas and nested functions. An exception raised by a call is raised here.
    """
    calls = _argument_tuples(iterables, num_calls)
    if executor is not None and max_workers is not None:
        raise ValueError("Give max_workers or executor, not both: an executor sets its own number of workers")
    if executor is not None:
        _check_not_parallel_pool(executor)
    rngs = np.random.default_rng(rng).spawn(len(calls))
    # One iterable of (arguments, generator) pairs, since some pools' map (e.g. multiprocessing.Pool's) takes only
    # one; and a partial of a module-level function, rather than a lambda, so that it can be pickled for processes
    items = list(zip(calls, rngs, strict=True))
    call = functools.partial(_call_with_rng, func)
    if executor is not None:
        return _map(executor, call, items, chunksize)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        return _map(pool, call, items, chunksize)


def _map[R](mapper: _SupportsMap, call: Callable[[Any], R], items: list[Any], chunksize: int | None) -> list[R]:
    """Map ``call`` over ``items`` with ``mapper``, passing ``chunksize`` only if given (not every map takes it)."""
    if chunksize is None:
        return list(mapper.map(call, items))
    # Executor.map and Pool.map both take chunksize as a keyword; the protocol leaves it out, as not every map has it
    return list(mapper.map(call, items, chunksize=chunksize))  # type: ignore[call-arg]


def _check_not_parallel_pool(executor: object) -> None:
    """Raise ``TypeError`` for pathos' ``ParallelPool``, which can't run the calls (see ``parallel_map_with_rngs``)."""
    # Recognized by module and name, rather than by isinstance, so that pathos needn't be imported (or installed)
    for cls in type(executor).__mro__:
        if cls.__module__ == "pathos.parallel" and cls.__qualname__ == "ParallelPool":
            raise TypeError(
                "pathos' ParallelPool can't run parallel_map_with_rngs' calls: it sends functions as source code, "
                "which a functools.partial doesn't have (use pathos' ProcessPool instead)"
            )


def _argument_tuples(iterables: tuple[Iterable[Any], ...], num_calls: int | None) -> list[tuple[Any, ...]]:
    """Return the arguments of each call as a tuple, checking the lengths against each other and ``num_calls``."""
    if num_calls is not None and num_calls < 0:
        raise ValueError(f"num_calls must be nonnegative, not {num_calls}")
    if not iterables:
        if num_calls is None:
            raise ValueError("Give iterables of arguments or num_calls: with neither, the number of calls is unknown")
        return [()] * num_calls
    # strict, so that arguments of different lengths raise rather than silently dropping calls
    calls = list(zip(*iterables, strict=True))
    if num_calls is not None and num_calls != len(calls):
        raise ValueError(f"num_calls is {num_calls}, but the iterables give {len(calls)} calls")
    return calls


def _call_with_rng[R](func: Callable[..., R], item: tuple[tuple[Any, ...], np.random.Generator]) -> R:
    """Call ``func`` with one call's arguments and its generator (run in the worker)."""
    args, rng = item
    return func(*args, rng=rng)
