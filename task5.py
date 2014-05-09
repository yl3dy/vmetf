#!/usr/bin/python3

from numpy import linspace, float64, copy, ones, empty
import matplotlib.pyplot as plt
from tdma import TDMA_solve
from scipy.integrate import cumtrapz

# Physical parameters
NU = 1
# Modelic
# X
X_NODES = 100
X_MIN = 0
X_MAX = 100
X_VALS, DX = linspace(X_MIN, X_MAX, num=X_NODES, retstep=True)
# Y
Y_NODES = 100
Y_MIN = 0
Y_MAX = 1
Y_VALS, DY = linspace(Y_MIN, Y_MAX, num=Y_NODES, retstep=True)
THETA = 0.5
ITER_NUM = 5    # for iterative method
# Initial conditions
P0 = 0
U0 = 1
V0 = 0


def exact():
    pass

def integrate_v(u_next, u_prev, i):
    dudx = - (u_next[i, :] - u_prev[i, :]) / DX
    v = cumtrapz(dudx, Y_VALS, initial=0)
    return v


def inexact():
    u_prev = ones([X_NODES, Y_NODES], dtype=float64) * U0
    v_prev = ones([X_NODES, Y_NODES], dtype=float64) * V0
    u_next, v_next = copy(u_prev), copy(v_prev)
    u_star, v_star = copy(u_prev[0, :]), copy(v_prev[0, :])

    # Matrix for selected n (~x)
    main_diag = empty(Y_NODES)
    upper_diag, lower_diag = empty(Y_NODES - 1), empty(Y_NODES - 1)
    free_coefs = empty(Y_NODES)

    for i in range(1, len(X_VALS) - 1):
        # Reset u_* and v_*
        u_star = u_prev[i, :]
        v_star = v_prev[i, :]

        for _ in range(ITER_NUM):
            # Fill the matrix
            main_diag[1:-1] = u_star[1:-1] / DX + 2 * THETA * NU / DY**2    # from u^{n+1}_i
            main_diag[0], main_diag[-1] = 1, 1
            upper_diag[1:] = v_star[2:] * THETA / (2*DY) - NU*THETA / DY**2   # from u^{n+1}_{i+1}
            upper_diag[0] = 0
            lower_diag[:-1] = -v_star[:-2]*THETA / (2*DY) - THETA*NU / DY**2   # from u^{n+1}_{i-1}
            lower_diag[-1] = 0

            # from u^n_i
            free_coefs[1:-1] = u_prev[i, 1:-1] * (u_star[1:-1] / DX - 2*(1 - THETA)*NU / DY**2)
            # from u^n_{i+1}
            free_coefs[1:-1] += u_prev[i, 2:] * ((1 - THETA)*NU / DY**2 - 0.5*(1 - THETA)*v_star[1:-1] / DY)
            # from u^n_{i-1}
            free_coefs[1:-1] += u_prev[i, :-2] * (0.5*v_star[1:-1]*(1 - THETA) / DY + NU*(1-THETA) / DY**2)
            # boundary conditions
            free_coefs[0] = 0
            free_coefs[-1] = U0

            u_next[i, :] = TDMA_solve(main_diag, upper_diag, lower_diag, free_coefs)
            v_next[i, :] = integrate_v(u_next, u_prev, i)
            u_star = THETA*u_next[i, :] + (1-THETA)*u_prev[i, :]
            v_star = THETA*v_next[i, :] + (1-THETA)*v_prev[i, :]

            u_prev = copy(u_next)
            v_prev = copy(v_next)

    return u_next[50, :]


def main():
    #P_exact = exact()
    u_inexact = inexact()
    print(u_inexact[0], u_inexact[-1])
    #plt.plot(Y_VALS, P_exact, Y_VALS, P_inexact)
    plt.plot(Y_VALS, u_inexact)
    #plt.legend(('Exact', 'Approximate'))
    plt.show()

if __name__ == '__main__':
    main()
