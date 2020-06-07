import numpy as np
from scipy import sparse


def generate_matrix(size: int) -> np.matrix:
    """ generate a matrix for first problem """
    matrix = np.zeros((size, size))
    for row in range(size):
        col = size - 1 - row
        if col < row:
            matrix[row, col] = 1
        if col > row:
            matrix[row, col] = -1
    return matrix


def generate_sparse_matrix(size: int) -> np.matrix:
    """ generates sparse matrix for first problem """
    data, rows, cols = [], [], []
    for row in range(size):
        rows.append(row)
        col = size - 1 - row
        cols.append(col)
        if col < row:
            data.append(1)
        if col > row:
            data.append(-1)
    return sparse.coo_matrix((data, (rows, cols)), shape=(size, size))


def generate_random_matrix(size: int) -> np.matrix:
    """ generates a random matrix for second problem """
    a = np.random.uniform(-5, 5, (size, size))
    b = np.random.uniform(-5, 5, (size, size))
    # skew-symmetric
    for row in range(size):
        for col in range(row):
            b[col, row] = -b[row, col]
    d = np.diag(np.random.uniform(0, .3, size))
    q = np.random.uniform(-500, 0, size)
    return a.dot(a.T) + b + d, q


def generate_tridiagonal_matrix(size: int) -> np.matrix:
    """ generate a triagonal matrix for fourth problem """
    d = np.zeros((size, size))
    for row in range(size):
        d[row, row] = 4
        if row != 0:
            d[row, row - 1] = 1
        if row + 1 != size:
            d[row, row + 1] = -2
    c = -np.ones(size)
    return d, c


def generate_sparse_tridiagonal_matrix(size: int) -> np.matrix:
    """ generates a sparse tridiagonal matrix for fourth problem """
    data, rows, cols = [], [], []
    for row in range(size):
        if row != 0:
            rows.append(row)
            cols.append(row - 1)
            data.append(1)
        rows.append(row)
        cols.append(row)
        data.append(4)
        if row + 1 != size:
            rows.append(row)
            cols.append(row + 1)
            data.append(-2)
    d = sparse.coo_matrix((data, (rows, cols)), shape=(size, size))
    c = -np.ones(size)
    return d, c
