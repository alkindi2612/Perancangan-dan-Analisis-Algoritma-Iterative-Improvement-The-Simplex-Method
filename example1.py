import numpy as np

def simplex(A, b, c):

    m, n = A.shape
    A = np.hstack((A, np.eye(m)))
    c = np.concatenate((c, np.zeros(m)))
    N = range(n)
    B = range(n, n + m)

    tableau = np.zeros((m+1, n+m+1))
    tableau[:-1, :n] = A
    tableau[:-1, -1] = b
    tableau[-1, :n] = c
    basis = list(B)

    while True:

        j = np.argmin(tableau[-1,:-1])

        if tableau[-1,j] >= 0:
            x = np.zeros(n)
            x[basis] = tableau[:-1, -1]
            z = tableau[-1, -1]
            return x, z

        ratios = tableau[:-1, -1] / tableau[:-1, j]
        i = np.argmin(ratios)

        if np.all(tableau[:,j] <= 0):
            raise Exception('Masalah tidak memiliki solusi optimal')

        basis[i] = j
        pivot_row = tableau[i, :] / tableau[i, j]
        tableau[:, :] -= np.outer(tableau[:, j], pivot_row)
        tableau[i, :] = pivot_row

    return None, None
