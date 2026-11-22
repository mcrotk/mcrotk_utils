"""Tests for ``mcrotk_utils.sparse_utils``' stacking functions: ``ps_hstack``, ``ps_vstack`` and ``ps_block``.

The expected results come from NumPy's ``hstack``, ``vstack`` and ``block`` on the blocks made dense, and from
scipy's ``block_array`` for grids with ``None`` blocks.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import scipy.sparse
from scipy.sparse import issparse, sparray

from mcrotk_utils.sparse_utils import PsMatrix, SparseArray, SparseFormat, as_dense, ps_block, ps_hstack, ps_vstack

FORMATS: list[SparseFormat] = ["bsr", "coo", "csc", "csr", "dia", "dok", "lil"]

# Blocks for the stacking functions: 2 rows each for ps_hstack, and their transposes (2 columns) for ps_vstack
LEFT = np.array([[1, 0, 2], [0, 3, 0]])
MIDDLE = np.array([[0.5], [-1.5]])
RIGHT = np.array([[0, 4], [5, 0]])


def to_sparse(array, format: SparseFormat = "csr") -> SparseArray:
    """Return ``array`` as a scipy sparse array in ``format``."""
    return scipy.sparse.coo_array(array).asformat(format)


def is_sparse_array(value) -> bool:
    """Whether ``value`` is a scipy sparse array (not a sparse matrix).

    A function rather than an inline ``isinstance``, which would narrow the type to scipy-stubs' bare ``sparray``
    mixin, without ``dtype`` or ``toarray``.
    """
    return isinstance(value, sparray)


def transposed(blocks):
    """Transpose every block (sparse or dense), turning blocks for ps_hstack into blocks for ps_vstack."""
    return [block.T for block in blocks]


# Each stacking function, with NumPy's own and the way to turn ps_hstack's example blocks into blocks for it
STACKERS = [
    pytest.param(ps_hstack, np.hstack, lambda blocks: blocks, id="ps_hstack"),
    pytest.param(ps_vstack, np.vstack, transposed, id="ps_vstack"),
]


# --- ps_hstack and ps_vstack -----------------------------------------------------------------------------------


@pytest.mark.parametrize(("stack", "numpy_stack", "arrange"), STACKERS)
def test_dense_blocks_give_what_numpy_gives(stack, numpy_stack, arrange):
    """With no sparse block, the result is NumPy's: a new NumPy array with its values and dtype."""
    blocks = arrange([LEFT, MIDDLE, RIGHT])
    result = stack(blocks)
    expected = numpy_stack(blocks)
    assert type(result) is np.ndarray
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(("stack", "numpy_stack", "arrange"), STACKERS)
def test_a_single_dense_block_gives_a_new_array(stack, numpy_stack, arrange):
    """Even one dense block gives a new array, not the block itself or a view of it."""
    (block,) = arrange([LEFT])
    result = stack([block])
    assert not np.shares_memory(result, block)
    np.testing.assert_array_equal(result, block)


@pytest.mark.parametrize(("stack", "numpy_stack", "arrange"), STACKERS)
@pytest.mark.parametrize("format", FORMATS)
@pytest.mark.parametrize("sparse_position", [0, 1, 2])
def test_any_sparse_block_gives_a_sparse_result(stack, numpy_stack, arrange, format, sparse_position):
    """One sparse block, in any format and position, makes the result a sparse array with NumPy's values and
    dtype."""
    dense_blocks = [LEFT, MIDDLE, RIGHT]
    blocks: list[PsMatrix] = list(dense_blocks)
    blocks[sparse_position] = to_sparse(blocks[sparse_position], format)
    result = stack(arrange(blocks))
    expected = numpy_stack(arrange(dense_blocks))
    assert is_sparse_array(result)
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result.toarray(), expected)


@pytest.mark.parametrize(("stack", "numpy_stack", "arrange"), STACKERS)
@pytest.mark.parametrize(
    "dtypes",
    [(np.int64, np.bool_), (np.int64, np.float32), (np.int8, np.int16), (np.float32, np.complex64)],
    ids=str,
)
def test_dense_and_sparse_results_have_the_same_dtype(stack, numpy_stack, arrange, dtypes):
    """The result's dtype is the blocks' common dtype, whether or not a block is sparse."""
    blocks = arrange([LEFT.astype(dtypes[0]), MIDDLE.astype(dtypes[1])])
    expected = numpy_stack(blocks).dtype
    assert stack(blocks).dtype == expected
    assert stack([to_sparse(blocks[0]), blocks[1]]).dtype == expected
    assert stack([blocks[0], to_sparse(blocks[1])]).dtype == expected


@pytest.mark.parametrize(("stack", "numpy_stack", "arrange"), STACKERS)
@pytest.mark.parametrize("format", ["csr", "csc"])
def test_by_default_all_csr_or_all_csc_blocks_keep_their_format(stack, numpy_stack, arrange, format):
    """With format=None, blocks that are all CSR give CSR, and all CSC give CSC (scipy's choice)."""
    # arranged before being made sparse, since transposing a CSR array gives a CSC one
    blocks = [to_sparse(block, format) for block in arrange([LEFT, RIGHT])]
    assert stack(blocks).format == format


@pytest.mark.parametrize(("stack", "numpy_stack", "arrange"), STACKERS)
@pytest.mark.parametrize(
    "blocks", [[to_sparse(LEFT, "csr"), MIDDLE], [to_sparse(LEFT, "csr"), to_sparse(MIDDLE, "csc")]], ids=str
)
def test_by_default_other_sparse_results_are_coo(stack, numpy_stack, arrange, blocks):
    """With format=None, mixed formats, or sparse and dense blocks together, give COO (scipy's choice)."""
    assert stack(arrange(blocks)).format == "coo"


@pytest.mark.parametrize(("stack", "numpy_stack", "arrange"), STACKERS)
@pytest.mark.parametrize("format", FORMATS)
def test_format_sets_the_format_of_a_sparse_result(stack, numpy_stack, arrange, format):
    """A format gives a sparse result in that format."""
    result = stack(arrange([to_sparse(LEFT), MIDDLE]), format=format)
    assert result.format == format
    np.testing.assert_array_equal(result.toarray(), numpy_stack(arrange([LEFT, MIDDLE])))


@pytest.mark.parametrize(("stack", "numpy_stack", "arrange"), STACKERS)
def test_format_is_ignored_for_a_dense_result(stack, numpy_stack, arrange):
    """A format doesn't make a dense result sparse."""
    result = stack(arrange([LEFT, MIDDLE]), format="csr")
    assert type(result) is np.ndarray
    np.testing.assert_array_equal(result, numpy_stack(arrange([LEFT, MIDDLE])))


@pytest.mark.parametrize(("stack", "numpy_stack", "arrange"), STACKERS)
@pytest.mark.parametrize("sparse", [False, True])
def test_unknown_formats_raise_whether_or_not_a_block_is_sparse(stack, numpy_stack, arrange, sparse):
    """An unknown format raises ValueError, even when the result would be dense and ignore it."""
    first = to_sparse(LEFT) if sparse else LEFT
    with pytest.raises(ValueError, match="Unknown sparse format 'CSR'"):
        stack(arrange([first, MIDDLE]), format="CSR")


@pytest.mark.parametrize(("stack", "numpy_stack", "arrange"), STACKERS)
def test_no_blocks_raise(stack, numpy_stack, arrange):
    """An empty list of blocks raises ValueError."""
    with pytest.raises(ValueError, match="At least one block is needed"):
        stack([])


@pytest.mark.parametrize(("stack", "numpy_stack", "arrange"), STACKERS)
@pytest.mark.parametrize("bad_block", [np.ones(2), np.ones((2, 1, 1)), np.float64(1.0)], ids=["1-D", "3-D", "0-D"])
@pytest.mark.parametrize("sparse", [False, True])
def test_blocks_that_are_not_2d_raise(stack, numpy_stack, arrange, bad_block, sparse):
    """A block that isn't 2-D raises ValueError naming it, whether or not another block is sparse."""
    first = to_sparse(LEFT) if sparse else LEFT
    blocks = [*arrange([first]), bad_block]
    with pytest.raises(ValueError, match=r"blocks\[1\] must be 2-D"):
        stack(blocks)


@pytest.mark.parametrize(("stack", "numpy_stack", "arrange"), STACKERS)
@pytest.mark.parametrize("sparse", [False, True])
def test_blocks_of_different_sizes_raise(stack, numpy_stack, arrange, sparse):
    """Blocks whose numbers of rows (ps_hstack) or columns (ps_vstack) differ raise ValueError."""
    first = to_sparse(LEFT) if sparse else LEFT
    with pytest.raises(ValueError):
        stack(arrange([first, np.ones((3, 1))]))


@pytest.mark.parametrize(("stack", "numpy_stack", "arrange"), STACKERS)
def test_none_blocks_raise(stack, numpy_stack, arrange):
    """None (a block of zeros in ps_block) raises TypeError here, since its size couldn't be known."""
    with pytest.raises(TypeError, match=r"blocks\[1\] is None: only ps_block takes None"):
        stack([*arrange([LEFT]), None])


# --- What the stacking functions work around (as the module docstring says) -----------------------------------


def test_numpy_stacking_of_sparse_blocks_gives_an_array_of_objects():
    """np.hstack doesn't raise an error for sparse blocks, but returns a 1-D array of objects."""
    blocks: list[Any] = [to_sparse(LEFT), to_sparse(RIGHT)]  # Any, since type checkers would rightly refuse them
    result = np.hstack(blocks)
    assert result.dtype == object
    assert result.shape == (2,)


@pytest.mark.parametrize("scipy_stack", [scipy.sparse.hstack, scipy.sparse.vstack])
def test_scipy_stacking_raises_unless_some_block_is_sparse(scipy_stack):
    """scipy's hstack and vstack raise ValueError when no block is sparse."""
    with pytest.raises(ValueError):
        scipy_stack([LEFT, LEFT])


def test_numpy_joins_1d_arrays_into_a_longer_one():
    """np.hstack joins 1-D arrays end to end, unlike scipy, which is why blocks must be 2-D."""
    assert np.hstack([np.ones(2), np.ones(3)]).shape == (5,)


def test_np_block_takes_none_as_an_object():
    """np.block doesn't raise for None, but returns an array of objects, so ps_block fills zeros itself."""
    result = np.block([[LEFT, MIDDLE], [np.ones((1, 3)), None]])
    assert result.dtype == object
    assert result[2, 3] is None


# --- ps_block: known cases -------------------------------------------------------------------------------------

A = np.arange(6).reshape(2, 3)
B = np.full((2, 1), 7)
C = np.full((1, 3), 8)
A_B_C_ZERO = np.array([[0, 1, 2, 7], [3, 4, 5, 7], [8, 8, 8, 0]])


@pytest.mark.parametrize("sparse", [False, True])
def test_docstring_example(sparse):
    """[[A, B], [C, None]] has A and B on top, and C and a block of zeros below."""
    first = to_sparse(A) if sparse else A
    result = ps_block([[first, B], [C, None]])
    assert issparse(result) == sparse
    np.testing.assert_array_equal(as_dense(result), A_B_C_ZERO)


def test_dense_blocks_give_what_np_block_gives():
    """With no sparse block and no None, the result is np.block's: a new NumPy array with its values and dtype."""
    grid: list[list[npt.NDArray[Any]]] = [[A, B.astype(np.float32)], [C, np.zeros((1, 1), dtype=np.int8)]]
    result = ps_block(grid)
    expected = np.block(grid)
    assert type(result) is np.ndarray
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result, expected)


def test_a_single_dense_block_gives_a_new_array_in_ps_block():
    """Even a grid of one dense block gives a new array."""
    result = ps_block([[A]])
    assert not np.shares_memory(result, A)
    np.testing.assert_array_equal(result, A)


@pytest.mark.parametrize("format", FORMATS)
def test_any_sparse_block_gives_a_sparse_result_in_ps_block(format):
    """One sparse block, in any format, makes the result a sparse array."""
    result = ps_block([[A, B], [C, to_sparse(np.ones((1, 1), dtype=int), format)]])
    assert is_sparse_array(result)
    np.testing.assert_array_equal(as_dense(result), np.block([[A, B], [C, np.ones((1, 1), dtype=int)]]))


def test_dense_zero_blocks_take_the_blocks_common_dtype():
    """In a dense result, the zeros take the blocks' common dtype: integer blocks stay integer."""
    result = ps_block([[A, B], [C, None]])
    assert result.dtype == np.result_type(A, B, C)
    assert ps_block([[A.astype(np.int8), None], [None, B.astype(np.int16)]]).dtype == np.int16


@pytest.mark.parametrize("sparse", [False, True])
def test_none_blocks_take_their_size_from_their_row_and_column(sparse):
    """A None takes its rows from its row of the grid and its columns from its column."""
    first = to_sparse(A) if sparse else A
    result = ps_block([[first, None], [None, np.ones((4, 5))]])
    assert result.shape == (6, 8)
    expected = np.zeros((6, 8))
    expected[:2, :3] = A
    expected[2:, 3:] = 1
    np.testing.assert_array_equal(as_dense(result), expected)


@pytest.mark.parametrize("sparse", [False, True])
def test_a_column_of_only_none_has_no_columns(sparse):
    """A column of the grid that is all None has width 0 (as in scipy)."""
    first = to_sparse(A) if sparse else A
    result = ps_block([[first, None], [C, None]])
    np.testing.assert_array_equal(as_dense(result), np.vstack([A, C]))


@pytest.mark.parametrize("sparse", [False, True])
def test_a_row_of_only_none_has_no_rows(sparse):
    """A row of the grid that is all None has height 0 (as in scipy)."""
    first = to_sparse(A) if sparse else A
    result = ps_block([[first, B], [None, None]])
    np.testing.assert_array_equal(as_dense(result), np.hstack([A, B]))


@pytest.mark.parametrize("grid", [[[None]], [[None, None], [None, None]]], ids=["1x1", "2x2"])
def test_a_grid_of_only_none_is_empty_float64(grid):
    """A grid of only None gives a dense, empty (0, 0) float64 array (scipy's shape and dtype)."""
    result = ps_block(grid)
    assert type(result) is np.ndarray
    assert result.shape == (0, 0)
    assert result.dtype == np.float64
    scipy_result = scipy.sparse.block_array(grid)
    assert (scipy_result.shape, scipy_result.dtype) == ((0, 0), np.float64)


@pytest.mark.parametrize("sparse", [False, True])
def test_tuples_work_as_rows_and_grids(sparse):
    """The grid and its rows can be tuples as well as lists (np.block itself takes only lists)."""
    first = to_sparse(A) if sparse else A
    result = ps_block(((first, B), (C, None)))
    np.testing.assert_array_equal(as_dense(result), A_B_C_ZERO)


@pytest.mark.parametrize("format", ["csr", "csc"])
def test_by_default_all_csr_or_all_csc_blocks_keep_their_format_in_ps_block(format):
    """With format=None, a grid whose blocks are all CSR gives CSR, and all CSC gives CSC (scipy's choice)."""
    grid = [[to_sparse(A, format), to_sparse(B, format)], [to_sparse(C, format), to_sparse(np.ones((1, 1)), format)]]
    result = ps_block(grid)
    assert not isinstance(result, np.ndarray)
    assert result.format == format


@pytest.mark.parametrize(
    "grid",
    [
        [[to_sparse(A, "csr"), B]],
        [[to_sparse(A, "csr"), to_sparse(B, "csc")]],
        [[to_sparse(A, "csr"), to_sparse(B, "csr")], [to_sparse(C, "csr"), None]],
        [[to_sparse(A, "lil")]],
    ],
    ids=["with dense", "csr and csc", "csr with None", "lil"],
)
def test_by_default_other_sparse_ps_block_results_are_coo(grid):
    """With format=None, mixed blocks, a None, or other formats give COO (scipy's choice)."""
    result = ps_block(grid)
    assert not isinstance(result, np.ndarray)
    assert result.format == "coo"


@pytest.mark.parametrize("format", FORMATS)
def test_format_sets_the_format_of_a_sparse_ps_block_result(format):
    """A format gives a sparse result in that format."""
    result = ps_block([[to_sparse(A), B], [C, None]], format=format)
    assert not isinstance(result, np.ndarray)
    assert result.format == format
    np.testing.assert_array_equal(result.toarray(), A_B_C_ZERO)


def test_format_is_ignored_for_a_dense_ps_block_result():
    """A format doesn't make a dense result sparse."""
    result = ps_block([[A, B], [C, None]], format="csr")
    assert type(result) is np.ndarray
    np.testing.assert_array_equal(result, A_B_C_ZERO)


@pytest.mark.parametrize("sparse", [False, True])
def test_unknown_formats_raise_in_ps_block(sparse):
    """An unknown format raises ValueError, whether or not a block is sparse."""
    first = to_sparse(A) if sparse else A
    unknown_format: Any = "xyz"  # Any, since type checkers would rightly refuse it
    with pytest.raises(ValueError, match="Unknown sparse format 'xyz'"):
        ps_block([[first]], format=unknown_format)


# --- ps_block: errors ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("grid", [[], [[]], ()], ids=["no rows", "empty row", "empty tuple"])
def test_empty_grids_raise(grid):
    """A grid without a block raises ValueError."""
    with pytest.raises(ValueError, match="at least one row of at least one block"):
        ps_block(grid)


@pytest.mark.parametrize("sparse", [False, True])
def test_rows_with_different_numbers_of_blocks_raise(sparse):
    """Rows of the grid with different numbers of blocks raise ValueError."""
    first = to_sparse(A) if sparse else A
    with pytest.raises(ValueError, match="Row 1 of the grid has 2 blocks, but row 0 has 1"):
        ps_block([[first], [C, None]])


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("bad_block", [np.ones(3), np.ones((1, 3, 1))], ids=["1-D", "3-D"])
def test_blocks_that_are_not_2d_raise_in_ps_block(sparse, bad_block):
    """A block that isn't 2-D raises ValueError naming its place in the grid."""
    first = to_sparse(A) if sparse else A
    with pytest.raises(ValueError, match=r"blocks\[1\]\[0\] must be 2-D"):
        ps_block([[first], [bad_block]])


@pytest.mark.parametrize("sparse", [False, True])
def test_blocks_in_a_row_with_different_numbers_of_rows_raise(sparse):
    """Blocks in the same row of the grid with different numbers of rows raise ValueError."""
    first = to_sparse(A) if sparse else A
    with pytest.raises(ValueError, match="row"):
        ps_block([[first, np.ones((3, 1))]])


@pytest.mark.parametrize("sparse", [False, True])
def test_blocks_in_a_column_with_different_numbers_of_columns_raise(sparse):
    """Blocks in the same column of the grid with different numbers of columns raise ValueError."""
    first = to_sparse(A) if sparse else A
    with pytest.raises(ValueError, match="column"):
        ps_block([[first], [np.ones((1, 4))]])


# --- ps_block: random grids, compared with scipy ---------------------------------------------------------------


def random_grid(rng):
    """Return a random grid of blocks for ps_block, and whether any block in it is sparse.

    The grid has 1 to 3 rows and columns of blocks; each grid row has a random height and each grid column a random
    width (0 to 3, so empty blocks occur too). Each block is None, a dense array, or a sparse array in a random
    format, with a random dtype and about half its entries zero.
    """
    heights = rng.integers(0, 4, size=rng.integers(1, 4))
    widths = rng.integers(0, 4, size=rng.integers(1, 4))
    dtypes = [np.int64, np.float64, np.float32, np.bool_]
    any_sparse = False
    grid = []
    for height in heights:
        row: list[PsMatrix | None] = []
        for width in widths:
            kind = rng.choice(["none", "dense", "sparse"])
            if kind == "none":
                row.append(None)
                continue
            values = rng.integers(-3, 4, size=(height, width)) * (rng.random((height, width)) < 0.5)
            dense = values.astype(dtypes[rng.integers(len(dtypes))])
            if kind == "sparse":
                row.append(to_sparse(dense, FORMATS[rng.integers(len(FORMATS))]))
                any_sparse = True
            else:
                row.append(dense)
        grid.append(row)
    return grid, any_sparse


def test_random_grids_match_scipy_block_array():
    """For random grids of None, dense and sparse blocks, ps_block has the values, shape and dtype that
    scipy.sparse.block_array gives (with every block made sparse), and is sparse exactly when some block is."""
    rng = np.random.default_rng(20261121)
    num_dense = num_sparse = 0
    for _ in range(500):
        grid, any_sparse = random_grid(rng)
        result = ps_block(grid)
        all_sparse = [[None if block is None else to_sparse(block) for block in row] for row in grid]
        expected = scipy.sparse.block_array(all_sparse)
        assert issparse(result) == any_sparse
        assert result.shape == expected.shape
        assert result.dtype == expected.dtype
        np.testing.assert_array_equal(as_dense(result), expected.toarray())
        num_sparse += any_sparse
        num_dense += not any_sparse
    # both paths were exercised
    assert num_dense > 50
    assert num_sparse > 50
