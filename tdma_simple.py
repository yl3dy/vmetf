import numpy as np

def TDMA_solve(main_diag, upper_diag, lower_diag, b):
    """Tridiagonal linear system solver."""
    n = len(main_diag)
    c = np.empty(n-1)
    d = np.empty(n)
    solution = np.empty(n)

    # forward
    c[0] = upper_diag[0] / main_diag[0]
    d[0] = b[0] / main_diag[0]
    for i in range(1, n):
        if i < n - 1:
            c[i] = upper_diag[i] / (main_diag[i] - c[i-1]*lower_diag[i-1])
        d[i] = (b[i] - d[i-1]*lower_diag[i-1]) / (main_diag[i] - c[i-1]*lower_diag[i-1])

    # backward substitution
    solution[n-1] = d[n-1]
    for i in reversed(range(n-1)):
        solution[i] = d[i] - c[i]*solution[i+1]

    return solution

