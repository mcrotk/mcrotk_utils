"""Splitting work among workers: lists into nearly equal parts, workers among groups, items into batches (private).

Public through ``mcrotk_utils.multiprocessing_utils``. Uses only the standard library.
"""

import heapq
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, overload

from mcrotk_utils.misc_utils import check_nonnegative, check_positive


def split_evenly[T](items: Iterable[T], num_parts: int) -> list[list[T]]:
    """Split ``items`` into ``num_parts`` consecutive lists whose lengths differ by at most one.

    The longer lists come first: e.g. 7 items in 3 parts give lists of 3, 2 and 2 items. With fewer items than
    parts, the last lists are empty. The items keep their order, and aren't copied. ``items`` may be any iterable;
    ``num_parts`` must be positive (``ValueError`` otherwise).
    """
    check_positive(num_parts, "num_parts")
    items = list(items)
    quotient, remainder = divmod(len(items), num_parts)
    parts, start = [], 0
    for index in range(num_parts):
        # the first `remainder` parts take one of the leftover items each
        size = quotient + (1 if index < remainder else 0)
        parts.append(items[start : start + size])
        start += size
    return parts


def allocate_workers(jobs: Sequence[int], num_workers: int) -> list[int]:
    """Allocate workers among groups of jobs that can't be mixed, minimizing the most jobs any worker gets.

    ``jobs[i]`` is the number of jobs in group ``i``; returns the number of workers for each group. A group's jobs
    are split as evenly as possible among its workers (as ``split_evenly`` does), so the most jobs any worker gets
    is the largest ``ceil(jobs[i] / workers[i])``, and the allocation makes that as small as possible.

    Each group with jobs gets one worker, and then each further worker goes to the group with the most jobs per
    worker at that point (the first such group, on a tie). That minimizes the most jobs any worker gets, and uses
    every worker, except that no group gets more workers than jobs (so a group with no jobs gets none, and with
    more workers than jobs in all, some workers are left over).

    Raises ``ValueError`` if a number of jobs is negative, or if there are fewer workers than groups with jobs
    (each needs one).
    """
    for count in jobs:
        check_nonnegative(count, "number of jobs")
    workers = [1 if count > 0 else 0 for count in jobs]
    if sum(workers) > num_workers:
        raise ValueError(f"{num_workers} workers can't cover {sum(workers)} groups with jobs: each needs one")
    # A heap of the groups that can still use a worker, most jobs per worker first (Fractions, so ties are exact;
    # then the group's index, so the first group wins a tie)
    heap = [(-Fraction(count, 1), index) for index, count in enumerate(jobs) if count > 1]
    heapq.heapify(heap)
    for _ in range(num_workers - sum(workers)):
        if not heap:
            break
        _, index = heapq.heappop(heap)
        workers[index] += 1
        if workers[index] < jobs[index]:
            heapq.heappush(heap, (-Fraction(jobs[index], workers[index]), index))
    return workers


@dataclass(frozen=True)
class Batch[K, T]:
    """One worker's share of the items, from ``split_for_workers``.

    - ``key``: the group the items belong to (``None`` when no ``key`` function was given).
    - ``index``: the batch's number within its group, from 0.
    - ``items``: the items, in their original order.
    """

    key: K
    index: int
    items: list[T]


@overload
def split_for_workers[T](items: Iterable[T], num_workers: int, *, key: None = None) -> list[Batch[None, T]]: ...
@overload
def split_for_workers[T, K: Hashable](
    items: Iterable[T], num_workers: int, *, key: Callable[[T], K]
) -> list[Batch[K, T]]: ...
def split_for_workers[T](
    items: Iterable[T], num_workers: int, *, key: Callable[[T], Any] | None = None
) -> list[Batch[Any, T]]:
    """Split ``items`` into batches for ``num_workers`` workers, keeping groups apart, with the largest batch as
    small as possible.

    Items with the same ``key(item)`` form a group (all the items form one group when ``key`` isn't given), and
    no batch mixes groups. The workers are allocated among the groups by ``allocate_workers``, and each group's
    items are split among its workers by ``split_evenly``, so the largest batch is as small as it can be.

    Returns the batches, group by group in the order each group first appears in ``items``, and in order within
    each group; each item keeps its place relative to the others in its group. No batch is empty: with fewer
    items than workers, there are fewer batches than workers. With fewer workers than groups, each group is one
    batch, so there are more batches than workers (a pool runs the extra ones as workers become free).
    ``num_workers`` must be positive (``ValueError`` otherwise).
    """
    check_positive(num_workers, "num_workers")
    groups: dict[Any, list[T]] = {}
    for item in items:
        groups.setdefault(None if key is None else key(item), []).append(item)
    if num_workers < len(groups):
        return [Batch(group_key, 0, group_items) for group_key, group_items in groups.items()]
    workers = allocate_workers([len(group_items) for group_items in groups.values()], num_workers)
    return [
        Batch(group_key, index, part)
        for (group_key, group_items), group_workers in zip(groups.items(), workers, strict=True)
        for index, part in enumerate(split_evenly(group_items, group_workers))
    ]
