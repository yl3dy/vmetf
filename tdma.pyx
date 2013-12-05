import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
def TDMA_solve(np.ndarray[DTYPE_t, ndim=1] main_diag, np.ndarray[DTYPE_t, ndim=1] upper_diag,
               np.ndarray[DTYPE_t, ndim=1] lower_diag, np.ndarray[DTYPE_t, ndim=1] b):
    """Tridiagonal linear system solver."""
    cdef int n = len(main_diag)
    cdef int i
    cdef np.ndarray[DTYPE_t, ndim=1] c = np.empty(n-1, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] d = np.empty(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] solution = np.empty(n, dtype=DTYPE)

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

# vim: set tw=0:
