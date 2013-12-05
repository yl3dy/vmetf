#!/usr/bin/python3

from numpy import empty, linspace, ones
import numpy as np
import matplotlib.pyplot as plt

# Size of main computation area
L_X = 1
L_Y = 1
X_DENSITY = 100
Y_DENSITY = 100
# Boundary conditions
T_NORTH = 2.0
T_SOUTH = 2.0
T_EAST = 2.0
T_WEST = 2.0
# Power of heat sources
S_C = 5
S_P = 10
S = lambda T: S_C + S_P*T
# Iterative parameters
EPS = 1e-4
TAU = 0.001

# Very small number for lambdas
L_EPS = 1e-20

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

def set_lambdas_raw():
    """Set thermal conductivity field."""
    lambdas_raw = ones([X_DENSITY, Y_DENSITY]) * 1.0
    lambdas_raw[:20, :20] = 1e20
    return lambdas_raw

def set_initial_conditions():
    """Set initial conditions for solver."""
    x_grid = linspace(0, L_X, X_DENSITY)
    y_grid = linspace(0, L_Y, Y_DENSITY)
    T_field = ones([X_DENSITY, Y_DENSITY])

    # calculate lambdas
    lambdas_raw = set_lambdas_raw()
    lambdas_x = empty([X_DENSITY - 1, Y_DENSITY - 2])
    lambdas_y = empty([X_DENSITY - 2, Y_DENSITY - 1])
    lambdas_x = ((2 * lambdas_raw[1:, 1:-1] * lambdas_raw[:-1, 1:-1] + L_EPS) /
                 (lambdas_raw[1:, 1:-1] + lambdas_raw[:-1, 1:-1] + L_EPS))
    lambdas_y = ((2 * lambdas_raw[1:-1, 1:] * lambdas_raw[1:-1, :-1] + L_EPS) /
                 (lambdas_raw[1:-1, 1:] + lambdas_raw[1:-1, :-1] + L_EPS))

    # set boundary conditions
    T_field[:, -1] = T_NORTH
    T_field[:, 0] = T_SOUTH
    T_field[0, :] = T_WEST
    T_field[-1, :] = T_EAST

    return x_grid, y_grid, lambdas_x, lambdas_y, T_field

def accuracy_acquired(T_next, T_prev):
    """Estimate difference between last 2 iterations."""
    return True

def pretty_plot(T_field):
    """Make a pretty heatmap plot."""
    plt.imshow(T_field.T, origin='lower', interpolation='none')
    plt.show()

def solver():
    """Entry point for solving."""
    x_grid, y_grid, lambdas_x, lambdas_y, T_field_prev = set_initial_conditions()
    T_field_next = T_field_prev.copy()          # for n+1 step
    T_field_halfnext = T_field_prev.copy()      # for n+1/2 step
    b_x, b_y = empty(X_DENSITY), empty(Y_DENSITY)    # free coefs for TDMA
    # TD matrix diagonals
    main_diag_x = empty(X_DENSITY)
    upper_diag_x, lower_diag_x = empty(X_DENSITY-1), empty(X_DENSITY-1)
    main_diag_y = main_diag_x.copy()
    upper_diag_y, lower_diag_y = upper_diag_x.copy(), lower_diag_x.copy()

    for iteration in range(100):
        # step in X direction
        for k in range(1, Y_DENSITY-1):
            #b_x[1:-1] = T_field_prev[1:-1, k] * 2/TAU + \
                        #(lambdas_y[:, k]/(y_grid[k] - y_grid[k+1])**2 * (T_field_prev[2:, k+1] - T_field_prev[1:-1, k])) - \
                        #(lambdas_y[:, k-1]/(y_grid[k-1] - y_grid[k])**2 * (T_field_prev[1:-1, k] - T_field_prev[:-2, k-1]))
            b_x[1:-1] = T_field_prev[1:-1, k] * 2/TAU
            b_x[0], b_x[-1] = T_field_prev[0, k], T_field_prev[-1, k]
            main_diag_x[0], main_diag_x[-1] = 1, 1
            main_diag_x[1:-1] = 2 / TAU + \
                                lambdas_x[:-1, k-1]/(x_grid[1:-1] - x_grid[:-2])**2 + \
                                lambdas_x[1:, k-1]/(x_grid[2:] - x_grid[1:-1])**2
            upper_diag_x[0] = 0
            upper_diag_x[1:] = -lambdas_x[1:, k-1] / (x_grid[2:] - x_grid[1:-1])**2
            lower_diag_x[-1] = 0
            lower_diag_x[:-1] = -lambdas_x[:-1, k-1] / (x_grid[1:-1] - x_grid[:-2])**2
            T_field_halfnext[:, k] = TDMA_solve(main_diag_x, upper_diag_x, lower_diag_x, b_x)

        # step in Y direction
        for i in range(1, X_DENSITY-1):
            #b_y[1:-1] = T_field_halfnext[i, 1:-1] * 2/TAU + \
                        #(lambdas_x[i, :]/(x_grid[i] - x_grid[i+1])**2 * (T_field_halfnext[i+1, 2:] - T_field_halfnext[i, 1:-1])) - \
                        #(lambdas_x[i-1, :]/(x_grid[i-1] - x_grid[i])**2 * (T_field_halfnext[i, 1:-1] - T_field_halfnext[i-1, :-2]))
            b_y[1:-1] = T_field_halfnext[i, 1:-1] * 2/TAU
            b_y[0], b_y[-1] = T_field_halfnext[i, 0], T_field_halfnext[i, -1]
            main_diag_y[0], main_diag_y[-1] = 1, 1
            main_diag_y[1:-1] = 2 / TAU + \
                                lambdas_y[i-1, :-1]/(y_grid[1:-1] - y_grid[:-2])**2 + \
                                lambdas_y[i-1, 1:]/(y_grid[2:] - y_grid[1:-1])**2
            upper_diag_y[0] = 0
            upper_diag_y[1:] = -lambdas_y[i-1, 1:] / (y_grid[2:] - y_grid[1:-1])**2
            lower_diag_y[-1] = 0
            lower_diag_y[:-1] = -lambdas_y[i-1, :-1] / (y_grid[1:-1] - y_grid[:-2])**2
            T_field_next[i, :] = TDMA_solve(main_diag_y, upper_diag_y, lower_diag_y, b_y)

        T_field_prev = T_field_next.copy()

    print(T_field_next)
    print('Min value: {}, max value: {}'.format(np.min(T_field_next.ravel()), np.max(T_field_next.ravel())))
    pretty_plot(T_field_next)


if __name__ == '__main__':
    solver()
