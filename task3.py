#!/usr/bin/python3

from numpy import empty, linspace, ones
import matplotlib.pyplot as plt

# Size of main computation area
L_X = 1
L_Y = 1
X_DENSITY = 10
Y_DENSITY = 10
# Boundary conditions
T_NORTH = 2
T_SOUTH = 2
T_EAST = 2
T_WEST = 2
# Power of heat sources
S_C = 5
S_P = 10
S = lambda T: S_C + S_P*T
# Accuracy
EPS = 1e-4

def TDMA_solve(main_diag, upper_diag, lower_diag, b):
    """Tridiagonal linear system solver."""
    n = len(main_diag)
    c, d, solution = empty(n - 1), empty(n), empty(n)

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

def set_initial_conditions():
    """Set initial conditions for solver."""
    x_grid = linspace(0, L_X, X_DENSITY)
    y_grid = linspace(0, L_Y, Y_DENSITY)
    T_field = ones([X_DENSITY, Y_DENSITY])
    return x_grid, y_grid, T_field

def pretty_plot(T_field):
    plt.imshow(T_field, origin='lower')
    plt.show()

def solver():
    """Entry point for solving."""
    x_grid, y_grid, T_field_prev = set_initial_conditions()
    T_field_next = T_field_prev.copy()

    pretty_plot(T_field_prev)

if __name__ == '__main__':
    solver()
