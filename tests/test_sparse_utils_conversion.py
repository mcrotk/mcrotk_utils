"""Tests for ``mcrotk_utils.sparse_utils.as_dense``."""

import numpy as np
import pytest
import scipy.sparse

from mcrotk_utils.sparse_utils import as_dense

FORMATS = ["bsr", "coo", "csc", "csr", "dia", "dok", "lil"]
DENSE = np.array([[1.5, 0.0, 2.0], [0.0, 0.0, -3.0]])


def to_sparse(array, format):
    """Return ``array`` as a scipy sparse array in ``format`` (e.g. ``"csr"``)."""
    return scipy.sparse.coo_array(array).asformat(format)


@pytest.mark.parametrize("format", FORMATS)
def test_sparse_arrays_become_dense_with_the_same_shape_dtype_and_values(format):
    """A sparse array in any format comes back as a NumPy array with its shape, dtype and values."""
    sparse = to_sparse(DENSE.astype(np.float32), format)
    result = as_dense(sparse)
    assert type(result) is np.ndarray
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, DENSE.astype(np.float32))


@pytest.mark.parametrize("format", FORMATS)
def test_the_dense_result_is_new(format):
    """Changing the dense result leaves the sparse array as it was."""
    sparse = to_sparse(DENSE, format)
    result = as_dense(sparse)
    result[:] = 7.0
    np.testing.assert_array_equal(sparse.toarray(), DENSE)


def test_a_numpy_array_is_returned_as_it_is():
    """A NumPy array comes back as the same object, not a copy (as np.asarray does)."""
    assert as_dense(DENSE) is DENSE


def test_other_array_likes_go_through_np_asarray():
    """Anything np.asarray accepts, such as a nested list, gives what np.asarray gives."""
    result = as_dense([[1, 2], [3, 4]])
    assert type(result) is np.ndarray
    np.testing.assert_array_equal(result, np.array([[1, 2], [3, 4]]))
    assert result.dtype == np.array([[1, 2], [3, 4]]).dtype


# --- The gap as_dense fills (as its docstring says) -----------------------------------------------------------


def test_numpy_arrays_have_no_toarray_method():
    """A NumPy array has no toarray method, so sparse.toarray() can't be used on both kinds."""
    assert not hasattr(DENSE, "toarray")


def test_np_asarray_wraps_a_sparse_array_in_a_0d_object_array():
    """np.asarray doesn't convert a sparse array or raise an error: it returns a 0-d array of objects holding it."""
    sparse = scipy.sparse.csr_array(DENSE)
    wrapped = np.asarray(sparse)
    assert wrapped.shape == ()
    assert wrapped.dtype == object
    assert wrapped.item() is sparse
