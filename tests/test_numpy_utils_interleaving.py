"""Tests for ``mcrotk_utils.numpy_utils.interleave``."""

import numpy as np
import pytest

from mcrotk_utils.numpy_utils import interleave


def rows(*labels):
    """Build a 2-column array whose row ``i`` is ``[labels[i], labels[i]]``, so row order is easy to read."""
    return np.array([[label, label] for label in labels])


def test_blocksize_one_alternates_rows():
    """With the default blocksize, rows alternate between the arrays."""
    result = interleave(rows(0, 1, 2), rows(10, 11, 12))
    np.testing.assert_array_equal(result, rows(0, 10, 1, 11, 2, 12))


def test_blocks_alternate():
    """With a larger blocksize, whole blocks alternate."""
    result = interleave(rows(0, 1, 2, 3), rows(10, 11, 12, 13), blocksize=2)
    np.testing.assert_array_equal(result, rows(0, 1, 10, 11, 2, 3, 12, 13))


def test_first_array_one_block_longer():
    """An extra block in the first array comes last."""
    result = interleave(rows(0, 1, 2, 3, 4, 5), rows(10, 11, 12, 13), blocksize=2)
    np.testing.assert_array_equal(result, rows(0, 1, 10, 11, 2, 3, 12, 13, 4, 5))


def test_uneven_lengths_skip_exhausted_arrays_and_allow_partial_blocks():
    """An array that runs out is skipped, and a last block can be shorter than the blocksize."""
    result = interleave(rows(0, 1, 2), rows(10, 11, 12, 13, 14, 15, 16), blocksize=2)
    np.testing.assert_array_equal(result, rows(0, 1, 10, 11, 2, 12, 13, 14, 15, 16))


def test_more_than_two_arrays():
    """Blocks are taken from each array in turn, in argument order."""
    result = interleave([0, 1], [10, 11], [20, 21])
    np.testing.assert_array_equal(result, [0, 10, 20, 1, 11, 21])


def test_single_array_is_unchanged():
    """With one array there is nothing to interleave."""
    np.testing.assert_array_equal(interleave(rows(0, 1, 2), blocksize=2), rows(0, 1, 2))


def test_single_array_result_is_a_copy():
    """Even with one array, the result is a new array rather than a view of the input."""
    a = rows(0, 1, 2)
    assert not np.shares_memory(interleave(a), a)


def test_empty_array_is_skipped():
    """An array of length zero along the axis contributes nothing."""
    result = interleave(rows(0, 1), np.empty((0, 2), dtype=int))
    np.testing.assert_array_equal(result, rows(0, 1))


@pytest.mark.parametrize("axis", [1, -1])
def test_other_axis(axis):
    """Interleaving along columns, including with a negative axis."""
    a = np.array([[0, 1], [2, 3]])
    b = np.array([[10, 11], [12, 13]])
    np.testing.assert_array_equal(interleave(a, b, axis=axis), [[0, 10, 1, 11], [2, 12, 3, 13]])


def test_paired_assignments():
    """Switchback-style assignments: each block of 0/1 assignments is followed by the opposite assignments."""
    x = np.array([[0, 1], [1, 1]])
    np.testing.assert_array_equal(interleave(x, 1 - x, blocksize=2), [[0, 1], [1, 1], [1, 0], [0, 0]])


def test_equal_lengths_not_divisible_by_blocksize():
    """Equal lengths that the blocksize doesn't divide: every array ends with a partial block."""
    result = interleave([0, 1, 2, 3, 4], [10, 11, 12, 13, 14], blocksize=2)
    np.testing.assert_array_equal(result, [0, 1, 10, 11, 2, 3, 12, 13, 4, 14])


def test_blocksize_larger_than_every_length():
    """A blocksize larger than every array puts each whole array in one block, so they come one after another."""
    result = interleave([0, 1], [10, 11, 12], blocksize=5)
    np.testing.assert_array_equal(result, [0, 1, 10, 11, 12])


def test_three_arrays_with_blocks():
    """Three equal arrays with a blocksize above 1: a block from each in turn."""
    result = interleave([0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23], blocksize=2)
    np.testing.assert_array_equal(result, [0, 1, 10, 11, 20, 21, 2, 3, 12, 13, 22, 23])


def test_three_arrays_with_uneven_lengths():
    """Three arrays of different lengths: each is skipped once it runs out, and last blocks may be partial."""
    result = interleave([0, 1, 2], [10, 11, 12, 13, 14], [20], blocksize=2)
    np.testing.assert_array_equal(result, [0, 1, 10, 11, 20, 2, 12, 13, 14])


def test_uneven_lengths_along_another_axis():
    """Uneven lengths and partial blocks along axis 1, where the result is written through a moved-axis view."""
    a = np.array([[0, 1, 2], [100, 101, 102]])
    b = np.array([[10, 11, 12, 13, 14], [110, 111, 112, 113, 114]])
    expected = [[0, 1, 10, 11, 2, 12, 13, 14], [100, 101, 110, 111, 102, 112, 113, 114]]
    np.testing.assert_array_equal(interleave(a, b, blocksize=2, axis=1), expected)


def test_three_dimensional_arrays():
    """Arrays with three dimensions, interleaved along the middle axis."""
    a = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    expected = [
        [[0, 1], [100, 101], [2, 3], [102, 103]],
        [[4, 5], [104, 105], [6, 7], [106, 107]],
    ]
    np.testing.assert_array_equal(interleave(a, a + 100, axis=1), expected)


def test_all_arrays_empty():
    """Arrays that are all empty along the axis give an empty result with the other dimensions kept."""
    result = interleave(np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=int))
    assert result.shape == (0, 2)


# Two arrays of rows, of equal lengths or with the first one block longer, and lengths divisible by the
# blocksize, with the first column of the expected output. Row i of the first array is i; of the second, 10 + i.
_KNOWN_CASES = [
    # (blocksize, rows in first array, rows in second array, first column of the output)
    (1, 3, 3, [0, 10, 1, 11, 2, 12]),
    (1, 4, 3, [0, 10, 1, 11, 2, 12, 3]),
    (2, 4, 4, [0, 1, 10, 11, 2, 3, 12, 13]),
    (2, 6, 4, [0, 1, 10, 11, 2, 3, 12, 13, 4, 5]),
    (3, 6, 3, [0, 1, 2, 10, 11, 12, 3, 4, 5]),
]


@pytest.mark.parametrize(("blocksize", "length_a", "length_b", "expected"), _KNOWN_CASES)
def test_known_cases(blocksize, length_a, length_b, expected):
    """Two arrays of rows, with whole blocks, give the rows in the expected order."""
    a = rows(*range(length_a))
    b = rows(*range(10, 10 + length_b))
    np.testing.assert_array_equal(interleave(a, b, blocksize=blocksize), rows(*expected))


def reference_interleave(arrays, blocksize, axis):
    """Interleave the simple way: split each array into blocks along the axis, then take blocks in turn."""
    axis_first = [np.moveaxis(array, axis, 0) for array in arrays]
    blocks = [[array[start : start + blocksize] for start in range(0, len(array), blocksize)] for array in axis_first]
    num_rounds = max(len(array_blocks) for array_blocks in blocks)
    in_turn = [array_blocks[i] for i in range(num_rounds) for array_blocks in blocks if i < len(array_blocks)]
    combined = np.concatenate(in_turn) if in_turn else axis_first[0][:0]
    return np.moveaxis(combined, 0, axis)


def random_case(rng):
    """Draw random arrays (1-4 of them, 1-3 dimensions, lengths 0-7 along the axis), a blocksize and an axis."""
    ndim = int(rng.integers(1, 4))
    axis = int(rng.integers(-ndim, ndim))
    other_dims = rng.integers(1, 4, size=ndim).tolist()
    arrays = []
    for array_num in range(int(rng.integers(1, 5))):
        shape = list(other_dims)
        shape[axis] = int(rng.integers(0, 8))
        # distinct values per array, so a wrong order can't go unnoticed
        arrays.append(1000 * array_num + np.arange(int(np.prod(shape))).reshape(shape))
    return arrays, int(rng.integers(1, 6)), axis


def test_matches_reference_on_random_cases():
    """On many random combinations of arrays, lengths, blocksizes, dimensions and axes, matches the reference."""
    rng = np.random.default_rng(20261112)
    for _ in range(500):
        arrays, blocksize, axis = random_case(rng)
        expected = reference_interleave(arrays, blocksize, axis)
        shapes = [array.shape for array in arrays]
        np.testing.assert_array_equal(
            interleave(*arrays, blocksize=blocksize, axis=axis),
            expected,
            err_msg=f"shapes {shapes}, blocksize {blocksize}, axis {axis}",
        )


@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.float32, bool])
def test_dtype_is_kept(dtype):
    """Arrays of one dtype give a result of that dtype (not, e.g., floats for integers)."""
    a = np.zeros((2, 3), dtype=dtype)
    assert interleave(a, a).dtype == dtype


def test_mixed_dtypes_are_promoted():
    """Arrays of different dtypes give their common dtype, as with np.concatenate."""
    assert interleave(np.array([1, 2]), np.array([0.5, 1.5])).dtype == np.float64


def test_rejects_no_arrays():
    """At least one array is needed."""
    with pytest.raises(ValueError, match="at least one array"):
        interleave()


@pytest.mark.parametrize("blocksize", [0, -1])
def test_rejects_blocksize_below_one(blocksize):
    """A blocksize below 1 is an error."""
    with pytest.raises(ValueError, match="blocksize"):
        interleave([1, 2], [3, 4], blocksize=blocksize)


def test_rejects_non_integer_blocksize():
    """A float blocksize is a TypeError, even when it's a whole number."""
    with pytest.raises(TypeError):
        interleave([1, 2], [3, 4], blocksize=2.0)  # type: ignore[arg-type]


def test_accepts_numpy_integer_blocksize():
    """A NumPy integer works as a blocksize."""
    np.testing.assert_array_equal(interleave([1, 2], [3, 4], blocksize=np.int64(2)), [1, 2, 3, 4])


def test_rejects_mismatched_other_dimensions():
    """Dimensions other than the axis must match."""
    with pytest.raises(ValueError):
        interleave(np.zeros((2, 3)), np.zeros((2, 4)))


def test_rejects_zero_dimensional_arrays():
    """Scalars have no axis to interleave along."""
    with pytest.raises(ValueError):
        interleave(np.array(1), np.array(2))


def test_rejects_axis_out_of_range():
    """An axis the arrays don't have is a ValueError (np.exceptions.AxisError)."""
    with pytest.raises(ValueError):
        interleave(np.zeros((2, 3)), np.zeros((2, 3)), axis=2)


def test_rejects_different_numbers_of_dimensions():
    """Arrays with different numbers of dimensions can't be interleaved."""
    with pytest.raises(ValueError):
        interleave(np.zeros((2, 3)), np.zeros(2))


def test_result_is_c_contiguous_along_other_axes():
    """The result is a new C-ordered array, not a transposed view, even when interleaving along axis 1."""
    result = interleave(np.zeros((3, 2)), np.ones((3, 2)), axis=1)
    assert result.flags.c_contiguous
