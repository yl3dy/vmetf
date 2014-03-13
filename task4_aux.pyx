"""Auxiliary stuff for task4 which needs accelerating."""

import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def smoothing(np.ndarray[DTYPE_t, ndim=1] rho, np.ndarray[DTYPE_t, ndim=1] u, np.ndarray[DTYPE_t, ndim=1] e, double q):
    """Smoothing for 2nd order methods."""
    cdef int grid_size = len(rho)
    cdef np.ndarray[DTYPE_t, ndim=1] rho_smooth = rho.copy()
    cdef np.ndarray[DTYPE_t, ndim=1] u_smooth = u.copy()
    cdef np.ndarray[DTYPE_t, ndim=1] e_smooth = e.copy()

    cdef int i
    cdef double D_m, D_mm, D_p, D_pp
    cdef double Q_rho, Q_u, Q_e
    for i in range(2, grid_size - 2):
        # Here are values for rho
        D_m = rho[i] - rho[i-1]
        D_mm = rho[i-1] - rho[i-2]
        D_p = rho[i+1] - rho[i]
        D_pp = rho[i+2] - rho[i+1]

        Q_rho = Q_u = Q_e = 0
        if D_p*D_m <= 0 or D_p*D_pp <= 0:
            Q_rho += D_p
            Q_u += u[i+1] - u[i]
            Q_e += e[i+1] - e[i]
        if D_p*D_m <= 0 or D_m*D_mm <= 0:
            Q_rho -= D_m
            Q_u -= u[i] - u[i-1]
            Q_e -= e[i] - e[i-1]

        rho_smooth[i] += q * Q_rho
        u_smooth[i] += q * Q_u
        e_smooth[i] += q * Q_e

    return rho_smooth, u_smooth, e_smooth
