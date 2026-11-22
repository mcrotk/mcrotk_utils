"""Tests for ``parallel_map_with_rngs`` (``mcrotk_utils.multiprocessing_utils._mapping``).

Most tests run the calls in a ``ThreadPoolExecutor``, which is quick; the ones about processes (the default
pool, pickling, initializers) use real worker processes. Functions run in workers are defined at module level,
so that worker processes can import them.
"""

import functools
import multiprocessing
import os
import pickle
import signal
import subprocess
import sys
import textwrap
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pytest

from mcrotk_utils.multiprocessing_utils import _mapping, parallel_map_with_rngs

# ---- functions and objects run in workers ----


def draw(rng):
    """One uniform draw."""
    return rng.random()


def scaled_draws(scale, size, rng):
    """``size`` scaled normal draws, as a list."""
    return list(scale * rng.normal(size=size))


def sleep_then_return(seconds, value, rng):
    """Sleep, then return ``value`` (to make calls finish out of order)."""
    time.sleep(seconds)
    return value


def fail_on(bad, value, rng):
    """Raise for ``bad``, otherwise return ``value``."""
    if value == bad:
        raise RuntimeError(f"failed on {value}")
    return value


@dataclass
class Scaler:
    """A picklable object whose method takes an ``rng``."""

    scale: float

    def draw(self, rng):
        """One scaled uniform draw."""
        return self.scale * rng.random()


_worker_state: dict[str, float] = {}


def set_worker_offset(offset):
    """Executor initializer: store an offset in each worker."""
    _worker_state["offset"] = offset


def draw_plus_offset(rng):
    """One draw plus the offset set by the initializer."""
    return _worker_state["offset"] + rng.random()


def reference(func, iterables, num_calls, seed):
    """Call ``func`` in a plain loop with ``default_rng(seed).spawn(n)``, as ``parallel_map_with_rngs`` documents."""
    calls = list(zip(*iterables, strict=True)) if iterables else [()] * num_calls
    rngs = np.random.default_rng(seed).spawn(len(calls))
    return [func(*args, rng=rng) for args, rng in zip(calls, rngs, strict=True)]


@pytest.fixture
def threads():
    """A thread pool for quick tests."""
    with ThreadPoolExecutor(max_workers=4) as executor:
        yield executor


# ---- what each call gets ----


def test_calls_get_the_items_of_the_iterables_and_spawned_generators(threads):
    """Call i is func(*args_i, rng=rngs[i]), with rngs = default_rng(rng).spawn(n)."""
    scales, sizes = [1.0, 10.0, 100.0], [2, 3, 1]
    result = parallel_map_with_rngs(scaled_draws, scales, sizes, executor=threads, rng=42)
    assert result == reference(scaled_draws, [scales, sizes], None, 42)


def test_without_iterables_func_is_called_num_calls_times_with_rng_only(threads):
    """With no iterables, func(rng=rngs[i]) is called num_calls times."""
    assert parallel_map_with_rngs(draw, num_calls=5, executor=threads, rng=3) == reference(draw, [], 5, 3)


def test_num_calls_may_be_given_with_iterables_when_it_matches(threads):
    """With iterables, num_calls is optional, and accepted when it equals their length."""
    with_count = parallel_map_with_rngs(scaled_draws, [1.0, 2.0], [1, 1], num_calls=2, executor=threads, rng=0)
    assert with_count == parallel_map_with_rngs(scaled_draws, [1.0, 2.0], [1, 1], executor=threads, rng=0)


def test_iterators_are_accepted(threads):
    """The iterables may be iterators (e.g. generators): they're read in full first."""
    scales = (s for s in [1.0, 2.0, 3.0])
    sizes = iter([1, 2, 3])
    result = parallel_map_with_rngs(scaled_draws, scales, sizes, executor=threads, rng=5)
    assert result == reference(scaled_draws, [[1.0, 2.0, 3.0], [1, 2, 3]], None, 5)


def test_results_come_back_in_call_order(threads):
    """Results are in call order even when later calls finish first."""
    delays = [0.2, 0.1, 0.0, 0.05]
    assert parallel_map_with_rngs(sleep_then_return, delays, "abcd", executor=threads, rng=0) == list("abcd")


def test_zero_calls_give_an_empty_list(threads):
    """No calls (num_calls=0, or empty iterables) give []."""
    assert parallel_map_with_rngs(draw, num_calls=0, executor=threads) == []
    assert parallel_map_with_rngs(scaled_draws, [], [], executor=threads) == []
    assert parallel_map_with_rngs(draw, num_calls=0) == []


# ---- reproducibility ----


def test_results_dont_depend_on_workers_chunksize_or_executor(threads):
    """The same seed gives the same results with any number of workers, chunksize, or executor."""
    scales = list(range(1, 9))
    sizes = [1, 2] * 4
    expected = reference(scaled_draws, [scales, sizes], None, 11)
    for max_workers, chunksize in [(1, 1), (3, 1), (3, 4)]:
        result = parallel_map_with_rngs(
            scaled_draws, scales, sizes, max_workers=max_workers, chunksize=chunksize, rng=11
        )
        assert result == expected
    assert parallel_map_with_rngs(scaled_draws, scales, sizes, executor=threads, chunksize=3, rng=11) == expected


@pytest.mark.parametrize("seed", [17, [17, 18]])
def test_the_same_int_or_ints_give_the_same_results(threads, seed):
    """An int, or a sequence of ints, gives the same results every time."""
    first = parallel_map_with_rngs(draw, num_calls=4, executor=threads, rng=seed)
    assert parallel_map_with_rngs(draw, num_calls=4, executor=threads, rng=seed) == first
    assert first == reference(draw, [], 4, seed)


@pytest.mark.parametrize("make_seed", [np.random.default_rng, np.random.PCG64, np.random.SeedSequence])
def test_a_generator_bit_generator_or_seed_sequence_gives_new_generators_each_time(threads, make_seed):
    """Passing the same Generator, BitGenerator or SeedSequence again gives the next generators spawned from it."""
    seed = make_seed(23)
    first = parallel_map_with_rngs(draw, num_calls=3, executor=threads, rng=seed)
    second = parallel_map_with_rngs(draw, num_calls=3, executor=threads, rng=seed)
    # the second call's generators are the next three children: spawning six at once gives both calls' worth
    assert first + second == reference(draw, [], 6, make_seed(23))


def test_a_generators_own_stream_isnt_drawn_from(threads):
    """Spawning from a Generator leaves its own stream where it was."""
    rng = np.random.default_rng(23)
    parallel_map_with_rngs(draw, num_calls=3, executor=threads, rng=rng)
    assert rng.random() == np.random.default_rng(23).random()


# ---- errors ----


def test_iterables_of_different_lengths_raise(threads):
    """Iterables of different lengths raise ValueError, rather than dropping calls."""
    with pytest.raises(ValueError):
        parallel_map_with_rngs(scaled_draws, [1.0, 2.0], [1], executor=threads)


def test_num_calls_different_from_the_iterables_raises(threads):
    """num_calls that differs from the iterables' length raises ValueError."""
    with pytest.raises(ValueError, match="num_calls is 3, but the iterables give 2 calls"):
        parallel_map_with_rngs(scaled_draws, [1.0, 2.0], [1, 1], num_calls=3, executor=threads)


def test_neither_iterables_nor_num_calls_raises(threads):
    """With no iterables, num_calls is needed."""
    with pytest.raises(ValueError, match="Give iterables of arguments or num_calls"):
        parallel_map_with_rngs(draw, executor=threads)


def test_negative_num_calls_raises(threads):
    """num_calls must be nonnegative."""
    with pytest.raises(ValueError, match="nonnegative"):
        parallel_map_with_rngs(draw, num_calls=-1, executor=threads)


def test_max_workers_with_executor_raises(threads):
    """max_workers can't be combined with executor."""
    with pytest.raises(ValueError, match="not both"):
        parallel_map_with_rngs(draw, num_calls=2, max_workers=2, executor=threads)


def test_an_exception_in_a_call_is_raised(threads):
    """An exception raised by a call is raised by parallel_map_with_rngs, with threads or processes."""
    with pytest.raises(RuntimeError, match="failed on 2"):
        parallel_map_with_rngs(fail_on, [2] * 4, range(4), executor=threads)
    with pytest.raises(RuntimeError, match="failed on 2"):
        parallel_map_with_rngs(fail_on, [2] * 4, range(4), max_workers=2)


# ---- executors and processes ----


def test_the_default_pool_gets_max_workers_and_is_shut_down_after_its_workers_finish(monkeypatch):
    """Without an executor, a ProcessPoolExecutor gets max_workers, and is shut down (waiting) before returning."""
    created = []

    class RecordingPool(ThreadPoolExecutor):
        """A thread pool standing in for ProcessPoolExecutor, recording how it's made and shut down."""

        def __init__(self, max_workers=None):
            """Record max_workers."""
            super().__init__(max_workers=max_workers)
            self.max_workers_given = max_workers
            self.shutdown_wait = None
            created.append(self)

        def shutdown(self, wait=True, *, cancel_futures=False):
            """Record whether shutdown waits for the workers."""
            self.shutdown_wait = wait
            super().shutdown(wait=wait, cancel_futures=cancel_futures)

    monkeypatch.setattr(_mapping, "ProcessPoolExecutor", RecordingPool)
    parallel_map_with_rngs(draw, num_calls=3, rng=0)
    parallel_map_with_rngs(draw, num_calls=3, max_workers=2, rng=0)
    assert [pool.max_workers_given for pool in created] == [None, 2]
    assert [pool.shutdown_wait for pool in created] == [True, True]


def test_a_given_executor_is_left_running():
    """A caller's executor isn't shut down: it can be used again."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        parallel_map_with_rngs(draw, num_calls=2, executor=executor, rng=0)
        assert executor.submit(threading.get_ident).result() is not None


class SerialMapper:
    """The least an executor needs: map(func, iterable) returning results in order, recording its calls."""

    def __init__(self):
        """Start with no recorded keyword arguments."""
        self.kwargs_seen = []

    def map(self, func, iterable, **kwargs):
        """Map in this process, recording the keyword arguments; reject any but chunksize."""
        if set(kwargs) - {"chunksize"}:
            raise TypeError(f"unexpected keyword arguments {kwargs}")
        self.kwargs_seen.append(kwargs)
        return map(func, iterable)


class MapWithoutChunksize:
    """An object whose map takes no chunksize."""

    def map(self, func, iterable):
        """Map in this process."""
        return map(func, iterable)


def test_anything_with_an_ordered_map_can_run_the_calls():
    """executor can be any object with map(func, iterable) returning results in order."""
    scales, sizes = [1.0, 2.0, 3.0], [2, 1, 2]
    expected = reference(scaled_draws, [scales, sizes], None, 6)
    assert parallel_map_with_rngs(scaled_draws, scales, sizes, executor=SerialMapper(), rng=6) == expected
    assert parallel_map_with_rngs(scaled_draws, scales, sizes, executor=MapWithoutChunksize(), rng=6) == expected


def test_chunksize_is_passed_to_map_only_when_given():
    """chunksize is passed as a keyword when given, and not at all by default."""
    mapper = SerialMapper()
    parallel_map_with_rngs(draw, num_calls=3, executor=mapper, rng=0)
    parallel_map_with_rngs(draw, num_calls=3, executor=mapper, chunksize=2, rng=0)
    assert mapper.kwargs_seen == [{}, {"chunksize": 2}]


def test_chunksize_with_a_map_that_doesnt_take_it_raises():
    """A chunksize given for a map that doesn't take one raises the map's TypeError."""
    with pytest.raises(TypeError):
        parallel_map_with_rngs(draw, num_calls=2, executor=MapWithoutChunksize(), chunksize=2)


def test_a_multiprocessing_pool_can_run_the_calls_and_is_left_running():
    """A multiprocessing.Pool, which isn't a concurrent.futures executor, works, with or without chunksize."""
    expected = reference(draw, [], 10, 12)
    with multiprocessing.get_context("spawn").Pool(2) as pool:
        assert parallel_map_with_rngs(draw, num_calls=10, executor=pool, rng=12) == expected
        assert parallel_map_with_rngs(draw, num_calls=10, executor=pool, chunksize=3, rng=12) == expected
        # still usable: the function didn't close it
        assert pool.apply(abs, (-1,)) == 1


def test_an_executor_with_an_initializer_sets_up_each_worker():
    """A ProcessPoolExecutor with an initializer (e.g. for logging) runs it in each worker before the calls."""
    with ProcessPoolExecutor(max_workers=2, initializer=set_worker_offset, initargs=(100.0,)) as executor:
        result = parallel_map_with_rngs(draw_plus_offset, num_calls=4, executor=executor, rng=9)
    assert result == [100.0 + value for value in reference(draw, [], 4, 9)]


def test_bound_methods_and_partials_work_with_processes():
    """With processes, func can be a bound method of a picklable object, or a functools.partial."""
    expected = [10.0 * value for value in reference(draw, [], 3, 4)]
    assert parallel_map_with_rngs(Scaler(10.0).draw, num_calls=3, max_workers=2, rng=4) == expected
    partial_result = parallel_map_with_rngs(functools.partial(scaled_draws, 2.0), [1, 2], max_workers=2, rng=4)
    assert partial_result == reference(functools.partial(scaled_draws, 2.0), [[1, 2]], None, 4)


def test_a_lambda_cant_be_sent_to_processes():
    """With processes, func must be picklable, which a lambda isn't."""
    # PicklingError from Python 3.14; Python 3.12 and 3.13 raise AttributeError for a lambda defined in a function
    with pytest.raises((pickle.PicklingError, AttributeError)):
        parallel_map_with_rngs(lambda rng: rng.random(), num_calls=2, max_workers=2)


def test_a_lambda_works_with_threads(threads):
    """Threads don't pickle, so any callable works there."""
    assert parallel_map_with_rngs(lambda rng: rng.random(), num_calls=2, executor=threads, rng=1) == reference(
        draw, [], 2, 1
    )


# ---- pathos (a dev dependency: skipped where it isn't installed, as in the oldest-versions run) ----


def worker_pid(rng):
    """Return the id of the process running the call."""
    return os.getpid()


@pytest.fixture
def pathos_pools():
    """pathos' pools module, or skip the test if pathos isn't installed."""
    return pytest.importorskip("pathos.pools")


def close_pathos_pool(pool):
    """Close a pathos pool, wait for its workers, and clear it from pathos' cache of pools."""
    pool.close()
    pool.join()
    pool.clear()


@pytest.mark.parametrize("pool_name", ["ProcessPool", "ThreadPool"])
def test_pathos_process_and_thread_pools_can_run_the_calls(pathos_pools, pool_name):
    """pathos' ProcessPool and ThreadPool work, with or without chunksize, and are left running."""
    scales, sizes = [float(s) for s in range(1, 13)], [1, 2, 3] * 4
    expected = reference(scaled_draws, [scales, sizes], None, 5)
    pool = getattr(pathos_pools, pool_name)(nodes=2)
    try:
        assert parallel_map_with_rngs(scaled_draws, scales, sizes, executor=pool, rng=5) == expected
        assert parallel_map_with_rngs(scaled_draws, scales, sizes, executor=pool, chunksize=3, rng=5) == expected
        with pytest.raises(RuntimeError, match="failed on 2"):
            parallel_map_with_rngs(fail_on, [2] * 4, range(4), executor=pool)
    finally:
        close_pathos_pool(pool)


def test_pathos_process_pool_runs_the_calls_in_other_processes_and_takes_lambdas(pathos_pools):
    """pathos' ProcessPool uses worker processes, and can send lambdas (it pickles with dill)."""
    pool = pathos_pools.ProcessPool(nodes=2)
    try:
        pids = set(parallel_map_with_rngs(worker_pid, num_calls=20, executor=pool, rng=0))
        assert pids and os.getpid() not in pids
        result = parallel_map_with_rngs(lambda rng: rng.random(), num_calls=4, executor=pool, rng=3)
        assert result == reference(draw, [], 4, 3)
    finally:
        close_pathos_pool(pool)


def test_pathos_parallel_pool_is_refused(pathos_pools):
    """pathos' ParallelPool raises TypeError rather than running the calls."""
    pool = pathos_pools.ParallelPool(nodes=2)
    try:
        with pytest.raises(TypeError, match="ParallelPool can't run"):
            parallel_map_with_rngs(draw, num_calls=2, executor=pool, rng=0)
    finally:
        close_pathos_pool(pool)


class FakeParallelPool:
    """Stands in for pathos' ParallelPool (same module and name), recording any use of map."""

    def __init__(self):
        """Start with no calls of map."""
        self.map_called = False

    def map(self, func, iterable):
        """Record the call."""
        self.map_called = True
        return []


FakeParallelPool.__module__ = "pathos.parallel"
FakeParallelPool.__qualname__ = "ParallelPool"


class FakeParallelPoolSubclass(FakeParallelPool):
    """A subclass of the stand-in."""


@pytest.mark.parametrize("pool_class", [FakeParallelPool, FakeParallelPoolSubclass])
def test_parallel_pool_and_its_subclasses_are_refused_without_importing_pathos(pool_class):
    """The check goes by module and name (so it runs without pathos), covers subclasses, and comes before map."""
    pool = pool_class()
    with pytest.raises(TypeError, match="ParallelPool can't run"):
        parallel_map_with_rngs(draw, num_calls=2, executor=pool, rng=0)
    assert not pool.map_called


def add(a, b):
    """Add, for functools.partial."""
    return a + b


def test_pathos_parallel_pool_gives_none_for_a_partial(pathos_pools):
    """Why ParallelPool is refused: its workers fail on a functools.partial, and its map returns None for them."""
    pool = pathos_pools.ParallelPool(nodes=2)
    try:
        assert pool.map(add, [1, 2], [10, 10]) == [11, 12]  # a plain function works
        assert pool.map(functools.partial(add, 10), [1, 2]) == [None, None]
    finally:
        close_pathos_pool(pool)


_PARALLEL_POOL_SCRIPT = """
import functools
from pathos.pools import ParallelPool


def add(a, b):
    return a + b


if __name__ == "__main__":
    pool = ParallelPool(nodes=2)
    print("plain function:", pool.map(add, [1, 2], [10, 10]), flush=True)
    pool.map(functools.partial(add, 10), [1, 2, 3])
    print("partial finished", flush=True)
"""


def test_pathos_parallel_pool_never_returns_with_more_partial_calls_than_workers(pathos_pools, tmp_path):
    """Why ParallelPool is refused: with more calls of a partial than workers, its map doesn't return."""
    script = tmp_path / "parallel_pool.py"
    script.write_text(textwrap.dedent(_PARALLEL_POOL_SCRIPT))
    # A new session, so that the pool's worker processes can be killed along with the script
    process = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "plain function: [11, 12]"  # the pool itself works
        # the map of the partial, which would take milliseconds, is still running seconds later
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=3)
    finally:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
