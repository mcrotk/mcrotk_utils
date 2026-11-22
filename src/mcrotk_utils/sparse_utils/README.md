# `mcrotk_utils.sparse_utils`

Helpers for code that takes matrices that may be dense or sparse ("possibly sparse", hence the `ps_` prefix): converting either to a dense array, and stacking blocks into a matrix that's sparse when any block is. It needs `scipy` (`mcrotk-utils[scipy]`, which brings `numpy`). The sparse inputs are `scipy`'s sparse arrays (`csr_array`, `coo_array`, and the other formats), not its older sparse matrices (`csr_matrix` and so on). Each function's docstring has the details: its arguments, result, and errors.

[Back to the package README](../../../README.md)

## Contents

The names are grouped below by what they're for.

### Conversion

`numpy` arrays have no `toarray` method, and `np.asarray` doesn't convert a sparse array (it wraps it in a 0-d array of objects), so neither library has one function for this.

- `as_dense`: a `numpy` array, whether given a sparse array or anything `np.asarray` accepts.

### Stacking

These stack 2-D blocks, which may be `numpy` arrays or `scipy` sparse arrays in any format. The result is a `scipy` sparse array if any block is sparse, and a `numpy` array otherwise.

- `ps_hstack`: stack blocks side by side.
- `ps_vstack`: stack blocks one above another.
- `ps_block`: assemble a matrix from a grid of blocks, in which `None` is a block of zeros.

### Type aliases

These are for annotating code that takes either kind of matrix.

- `PsMatrix`: a `numpy` array or a `scipy` sparse array.
- `SparseArray`: a `scipy` sparse array in any format.
- `SparseFormat`: `scipy`'s names for its sparse formats (`"csr"`, `"coo"`, and so on), as the stacking functions' `format` argument takes them.

## Examples

These examples show the kind of code the module is for: functions that take a matrix that may be dense or sparse, and keep it that way. Adding a column of ones to a matrix:

```pycon
>>> import numpy as np
>>> from scipy import sparse
>>> from mcrotk_utils.sparse_utils import as_dense, ps_block, ps_hstack
>>> def add_intercept(X):
...     return ps_hstack([np.ones((X.shape[0], 1)), X], format="csr")
>>> X = np.array([[0.0, 1.0], [2.0, 0.0], [0.0, 3.0]])
>>> add_intercept(X)
array([[1., 0., 1.],
       [1., 2., 0.],
       [1., 0., 3.]])
```

Given a sparse matrix, the same function gives a sparse result, in the format asked for, without ever making the matrix dense:

```pycon
>>> with_intercept = add_intercept(sparse.csr_array(X))
>>> sparse.issparse(with_intercept), with_intercept.format
(True, 'csr')
>>> np.array_equal(as_dense(with_intercept), add_intercept(X))
True
```

`as_dense` converts either kind. `np.asarray` doesn't convert a sparse array: it wraps it in a 0-d array of objects.

```pycon
>>> np.asarray(sparse.csr_array(X)).shape
()
>>> as_dense(sparse.csr_array(X)).shape
(3, 2)
```

`ps_block` assembles a grid of blocks, with `None` for a block of zeros. Here one block is sparse, so the result is too:

```pycon
>>> matrix = ps_block([[X, sparse.eye_array(3, format="csr")], [None, np.ones((1, 3))]])
>>> sparse.issparse(matrix), matrix.shape
(True, (4, 5))
>>> as_dense(matrix)
array([[0., 1., 1., 0., 0.],
       [2., 0., 0., 1., 0.],
       [0., 3., 0., 0., 1.],
       [0., 0., 1., 1., 1.]])
```
