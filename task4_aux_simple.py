"""Auxiliary stuff for task4 which needs accelerating.

This is pure Python version. Cython version is in task4_aux_c.pyx

"""

def smoothing(rho, u, e, q):
    """Smoothing for 2nd order methods."""
    grid_size = len(rho)
    rho_smooth = rho.copy()
    u_smooth = u.copy()
    e_smooth = e.copy()
    def get_D_p(f, i):
        return f[i+1] - f[i]
    def get_D_m(f, i):
        return f[i] - f[i-1]

    for i in range(2, grid_size - 2):
        # Here are values for rho
        D_m = get_D_m(rho, i)
        D_mm = rho[i-1] - rho[i-2]
        D_p = get_D_p(rho, i)
        D_pp = rho[i+2] - rho[i+1]

        Q_rho = Q_u = Q_e = 0
        if D_p*D_m <= 0 or D_p*D_pp <= 0:
            Q_rho += D_p
            Q_u += get_D_p(u, i)
            Q_e += get_D_p(e, i)
        if D_p*D_m <= 0 or D_m*D_mm <= 0:
            Q_rho -= D_m
            Q_u -= get_D_m(u, i)
            Q_e -= get_D_m(e, i)

        rho_smooth[i] += q * Q_rho
        u_smooth[i] += q * Q_u
        e_smooth[i] += q * Q_e

    return rho_smooth, u_smooth, e_smooth
