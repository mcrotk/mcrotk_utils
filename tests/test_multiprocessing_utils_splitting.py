"""Tests for ``split_evenly``, ``allocate_workers`` and ``split_for_workers``
(``mcrotk_utils.multiprocessing_utils._splitting``).

``allocate_workers`` is compared with ``reference_best_max_jobs``, a brute-force search over every allocation.
"""

import dataclasses
import itertools
import math
import random

import numpy as np
import pytest

from mcrotk_utils.multiprocessing_utils import Batch, allocate_workers, split_evenly, split_for_workers


def max_jobs(jobs, workers):
    """The most jobs any worker gets, with each group's jobs split evenly among its workers."""
    return max((math.ceil(j / w) for j, w in zip(jobs, workers, strict=True) if j > 0), default=0)


def reference_best_max_jobs(jobs, num_workers):
    """The smallest possible max_jobs, found by trying every allocation (one worker or more per group with jobs)."""
    choices = [range(1, num_workers + 1) if j > 0 else [0] for j in jobs]
    return min(max_jobs(jobs, w) for w in itertools.product(*choices) if sum(w) <= num_workers)


# ---- split_evenly ----


def test_split_evenly_example():
    """7 items in 3 parts give consecutive lists of 3, 2 and 2 items, longer ones first."""
    assert split_evenly("abcdefg", 3) == [["a", "b", "c"], ["d", "e"], ["f", "g"]]


@pytest.mark.parametrize(("num_items", "num_parts"), list(itertools.product(range(12), range(1, 8))))
def test_split_evenly_matches_numpy_array_split(num_items, num_parts):
    """The parts are those of np.array_split: consecutive, lengths differing by at most one, longer ones first."""
    parts = split_evenly(range(num_items), num_parts)
    assert [list(part) for part in np.array_split(np.arange(num_items), num_parts)] == parts
    assert len(parts) == num_parts


def test_split_evenly_gives_empty_last_parts_with_fewer_items_than_parts():
    """With fewer items than parts, the last parts are empty."""
    assert split_evenly([1, 2], 4) == [[1], [2], [], []]
    assert split_evenly([], 2) == [[], []]


def test_split_evenly_takes_any_iterable_and_doesnt_copy():
    """Any iterable works (e.g. a generator), and the parts hold the original items, not copies."""
    items = [[1], [2], [3]]
    parts = split_evenly((item for item in items), 2)
    assert parts == [[[1], [2]], [[3]]]
    assert parts[0][0] is items[0]


@pytest.mark.parametrize("num_parts", [0, -1])
def test_split_evenly_needs_a_positive_number_of_parts(num_parts):
    """num_parts must be positive."""
    with pytest.raises(ValueError, match="num_parts"):
        split_evenly([1, 2], num_parts)


# ---- allocate_workers ----


def random_cases(count, seed):
    """Random (jobs, num_workers) cases, with enough workers for every group with jobs."""
    rng = random.Random(seed)
    cases = []
    for _ in range(count):
        jobs = [rng.choice([0, 1, 2, 3, 5, 8, 13, 21]) for _ in range(rng.randint(1, 4))]
        num_workers = rng.randint(sum(j > 0 for j in jobs), 10)
        cases.append((jobs, num_workers))
    return cases


@pytest.mark.parametrize(("jobs", "num_workers"), random_cases(300, seed=0))
def test_allocate_workers_minimizes_the_most_jobs_any_worker_gets(jobs, num_workers):
    """The allocation's most jobs per worker is the smallest any allocation reaches (brute force)."""
    assert max_jobs(jobs, allocate_workers(jobs, num_workers)) == reference_best_max_jobs(jobs, num_workers)


@pytest.mark.parametrize(("jobs", "num_workers"), random_cases(300, seed=1))
def test_allocate_workers_uses_every_worker_but_never_more_than_a_groups_jobs(jobs, num_workers):
    """Groups with jobs get at least one worker and at most one per job, groups without get none, and every
    worker is used unless there are more workers than jobs."""
    workers = allocate_workers(jobs, num_workers)
    assert all((1 <= w <= j) if j > 0 else w == 0 for j, w in zip(jobs, workers, strict=True))
    assert sum(workers) == min(num_workers, sum(jobs))


def test_allocate_workers_gives_each_further_worker_to_the_group_with_the_most_jobs_per_worker():
    """After one each, workers go one at a time to the group with the most jobs per worker, the first on a tie."""
    # [1, 1] -> second (10 vs 5) -> [1, 2], tie at 5 -> first -> [2, 2] -> second (5 vs 2.5) -> [2, 3]
    # -> second (3.3 vs 2.5) -> [2, 4], tie at 2.5 -> first -> [3, 4] -> second -> [3, 5] -> second -> [3, 6],
    # tie at 5/3 -> first -> [4, 6]
    assert allocate_workers([5, 10], 10) == [4, 6]
    assert allocate_workers([4, 4], 3) == [2, 1]


def test_allocate_workers_with_no_groups_or_no_jobs():
    """No groups give an empty list; groups without jobs get no workers."""
    assert allocate_workers([], 3) == []
    assert allocate_workers([0, 0], 3) == [0, 0]


def test_allocate_workers_needs_a_worker_for_each_group_with_jobs():
    """Fewer workers than groups with jobs raise ValueError."""
    with pytest.raises(ValueError, match="2 workers can't cover 3 groups with jobs"):
        allocate_workers([1, 0, 4, 2], 2)


def test_allocate_workers_rejects_negative_jobs():
    """A negative number of jobs raises ValueError."""
    with pytest.raises(ValueError, match="number of jobs"):
        allocate_workers([3, -1], 4)


# ---- split_for_workers ----


def test_without_a_key_split_for_workers_splits_evenly_without_empty_batches():
    """With no key, the items are one group, split evenly into at most num_workers batches, none empty."""
    assert split_for_workers(range(7), 3) == [Batch(None, 0, [0, 1, 2]), Batch(None, 1, [3, 4]), Batch(None, 2, [5, 6])]
    assert split_for_workers(range(2), 5) == [Batch(None, 0, [0]), Batch(None, 1, [1])]
    assert split_for_workers([], 3) == []


def test_split_for_workers_example_with_groups():
    """Groups in order of first appearance, items in order, workers allocated by allocate_workers."""
    items = ["a1", "b1", "a2", "a3", "b2", "a4", "c1"]
    batches = split_for_workers(items, 4, key=lambda item: item[0])
    # 4 a's, 2 b's and 1 c: one worker each, and the fourth to "a" (4 jobs per worker against 2 and 1)
    assert batches == [
        Batch("a", 0, ["a1", "a2"]),
        Batch("a", 1, ["a3", "a4"]),
        Batch("b", 0, ["b1", "b2"]),
        Batch("c", 0, ["c1"]),
    ]


@pytest.mark.parametrize("seed", range(40))
def test_split_for_workers_keeps_groups_apart_with_the_smallest_largest_batch(seed):
    """Every item lands in one batch of its own group, in order, with no empty batch, at most num_workers
    batches, and the largest batch as small as possible (brute force)."""
    rng = random.Random(seed)
    items = [(rng.choice("xyz"), n) for n in range(rng.randint(1, 30))]
    num_groups = len({group for group, _ in items})
    num_workers = rng.randint(num_groups, 8)
    batches = split_for_workers(items, num_workers, key=lambda item: item[0])
    assert all(batch.items and all(item[0] == batch.key for item in batch.items) for batch in batches)
    assert len(batches) <= num_workers
    for group in {group for group, _ in items}:
        group_batches = [batch for batch in batches if batch.key == group]
        assert [batch.index for batch in group_batches] == list(range(len(group_batches)))
        assert [item for batch in group_batches for item in batch.items] == [item for item in items if item[0] == group]
    jobs = [sum(item[0] == group for item in items) for group in dict.fromkeys(group for group, _ in items)]
    assert max(len(batch.items) for batch in batches) == reference_best_max_jobs(jobs, num_workers)


def test_split_for_workers_with_fewer_workers_than_groups_gives_one_batch_per_group():
    """With fewer workers than groups, each group is one batch."""
    batches = split_for_workers([1, 2, 3, 4, 5], 2, key=lambda n: n % 3)
    assert batches == [Batch(1, 0, [1, 4]), Batch(2, 0, [2, 5]), Batch(0, 0, [3])]


@pytest.mark.parametrize("num_workers", [0, -2])
def test_split_for_workers_needs_a_positive_number_of_workers(num_workers):
    """num_workers must be positive."""
    with pytest.raises(ValueError, match="num_workers"):
        split_for_workers([1, 2], num_workers)


def test_batches_are_frozen():
    """A Batch can't be changed (its fields, that is: its list of items is an ordinary list)."""
    batch = Batch("k", 0, [1])
    with pytest.raises(dataclasses.FrozenInstanceError):
        batch.index = 1  # type: ignore[misc]
