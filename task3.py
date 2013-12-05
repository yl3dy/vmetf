#!/usr/bin/python3

# With Cython
import pyximport; pyximport.install()
from tdma import TDMA_solve

# Without Cython
#from tdma_simple import TDMA_solve

from numpy import empty, linspace, ones, zeros
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
# Additional boundary conditions
BOX_SIZE_1 = [0.2, 0.2]     # box size for 1st type boundary
BOX_COORDS_2 = [[0.3, 0.3], [0.7, 0.7]]
T_2 = 2.5     # used in 2nd and 4th method (temp along y axis)
T_1 = 2.5     # used in 1st and 4th method (temp along x axis)
# Iterative parameters
EPS = 1e-4
TAU = 0.0001
ITER_NUM = 100
# Very small number for lambdas
L_EPS = 1e-20

# Precalculate useful constants
BOX_1 = [BOX_SIZE_1[0]/L_X * X_DENSITY, BOX_SIZE_1[1]/L_Y * Y_DENSITY]
BOX_2 = [[BOX_COORDS_2[0][0]/L_X*X_DENSITY, BOX_COORDS_2[0][1]/L_Y*Y_DENSITY],
         [BOX_COORDS_2[1][0]/L_X*X_DENSITY, BOX_COORDS_2[1][1]/L_Y*Y_DENSITY]]

def set_initial_conditions(boundary_type):
    """Set initial conditions for solver."""
    x_grid = linspace(0, L_X, X_DENSITY)
    y_grid = linspace(0, L_Y, Y_DENSITY)
    T_field = ones([X_DENSITY, Y_DENSITY])
    # set boundary conditions
    T_field[:, -1] = T_NORTH
    T_field[:, 0] = T_SOUTH
    T_field[0, :] = T_WEST
    T_field[-1, :] = T_EAST
    if boundary_type == 1:
        T_field[:BOX_1[0], :BOX_1[1]] = T_1

    # set raw lambdas
    lambdas_raw = ones([X_DENSITY, Y_DENSITY])*10
    if boundary_type == 1:
        lambdas_raw[:BOX_1[0], :BOX_1[1]] = 1e20
    elif boundary_type == 2:
        lambdas_raw[BOX_2[0][0]:BOX_2[1][0], BOX_2[0][1]:BOX_2[1][1]] = 1e15
    elif boundary_type in [3, 4]:
        lambdas_raw[:BOX_1[0], :BOX_1[1]] = 0

    # heat sources
    s_c = zeros([X_DENSITY, Y_DENSITY])
    s_p = zeros([X_DENSITY, Y_DENSITY])
    if boundary_type == 2:
        s_c[BOX_2[0][0]:BOX_2[1][0], BOX_2[0][1]:BOX_2[1][1]] = 1e20 * T_2
        s_p[BOX_2[0][0]:BOX_2[1][0], BOX_2[0][1]:BOX_2[1][1]] = -1e20
    elif boundary_type == 4:
        s_p[BOX_1[0], :BOX_1[1]+1] = -1e20
        s_p[:BOX_1[0]+1, BOX_1[1]] = -1e20
        s_c[BOX_1[0], :BOX_1[1]+1] = 1e20 * T_1
        s_c[:BOX_1[0]+1, BOX_1[1]] = 1e20 * T_2

    # calculate lambdas +/-
    lambdas_x = empty([X_DENSITY - 1, Y_DENSITY - 2])
    lambdas_y = empty([X_DENSITY - 2, Y_DENSITY - 1])
    lambdas_x = ((2 * lambdas_raw[1:, 1:-1] * lambdas_raw[:-1, 1:-1] + L_EPS) /
                 (lambdas_raw[1:, 1:-1] + lambdas_raw[:-1, 1:-1] + L_EPS))
    lambdas_y = ((2 * lambdas_raw[1:-1, 1:] * lambdas_raw[1:-1, :-1] + L_EPS) /
                 (lambdas_raw[1:-1, 1:] + lambdas_raw[1:-1, :-1] + L_EPS))

    return x_grid, y_grid, lambdas_x, lambdas_y, T_field, s_c, s_p

def accuracy_acquired(T_next, T_prev):
    """Estimate difference between last 2 iterations."""
    return True

def pretty_plot(T_field):
    """Make a pretty heatmap plot."""
    plt.imshow(T_field.T, origin='lower', interpolation='none')
    plt.show()

def solver(boundary_type):
    """Entry point for solving."""
    x_grid, y_grid, lambdas_x, lambdas_y, T_field_prev, s_c, s_p = set_initial_conditions(boundary_type)
    T_field_next = T_field_prev.copy()          # for n+1 step
    T_field_halfnext = T_field_prev.copy()      # for n+1/2 step
    b_x, b_y = empty(X_DENSITY), empty(Y_DENSITY)    # free coefs for TDMA
    # TD matrix diagonals
    main_diag_x = empty(X_DENSITY)
    upper_diag_x, lower_diag_x = empty(X_DENSITY-1), empty(X_DENSITY-1)
    main_diag_y = main_diag_x.copy()
    upper_diag_y, lower_diag_y = upper_diag_x.copy(), lower_diag_x.copy()

    for iteration in range(ITER_NUM):
        # step in X direction
        for k in range(1, Y_DENSITY-1):
            #b_x[1:-1] = T_field_prev[1:-1, k] * 2/TAU - s_c[1:-1, k] + \
                        #(lambdas_y[:, k]/(y_grid[k] - y_grid[k+1])**2 * (T_field_prev[2:, k+1] - T_field_prev[1:-1, k])) - \
                        #(lambdas_y[:, k-1]/(y_grid[k-1] - y_grid[k])**2 * (T_field_prev[1:-1, k] - T_field_prev[:-2, k-1]))
            b_x[1:-1] = T_field_prev[1:-1, k] * 2/TAU - s_c[1:-1, k]
            b_x[0], b_x[-1] = T_field_prev[0, k], T_field_prev[-1, k]
            main_diag_x[0], main_diag_x[-1] = 1, 1
            main_diag_x[1:-1] = 2 / TAU + s_p[1:-1, k] + \
                                lambdas_x[:-1, k-1]/(x_grid[1:-1] - x_grid[:-2])**2 + \
                                lambdas_x[1:, k-1]/(x_grid[2:] - x_grid[1:-1])**2
            upper_diag_x[0] = 0
            upper_diag_x[1:] = -lambdas_x[1:, k-1] / (x_grid[2:] - x_grid[1:-1])**2
            lower_diag_x[-1] = 0
            lower_diag_x[:-1] = -lambdas_x[:-1, k-1] / (x_grid[1:-1] - x_grid[:-2])**2
            T_field_halfnext[:, k] = TDMA_solve(main_diag_x, upper_diag_x, lower_diag_x, b_x)

        # step in Y direction
        for i in range(1, X_DENSITY-1):
            #b_y[1:-1] = T_field_halfnext[i, 1:-1] * 2/TAU - s_c[i, 1:-1] + \
                        #(lambdas_x[i, :]/(x_grid[i] - x_grid[i+1])**2 * (T_field_halfnext[i+1, 2:] - T_field_halfnext[i, 1:-1])) - \
                        #(lambdas_x[i-1, :]/(x_grid[i-1] - x_grid[i])**2 * (T_field_halfnext[i, 1:-1] - T_field_halfnext[i-1, :-2]))
            b_y[1:-1] = T_field_halfnext[i, 1:-1] * 2/TAU - s_c[i, 1:-1]
            b_y[0], b_y[-1] = T_field_halfnext[i, 0], T_field_halfnext[i, -1]
            main_diag_y[0], main_diag_y[-1] = 1, 1
            main_diag_y[1:-1] = 2 / TAU + s_p[i, 1:-1] + \
                                lambdas_y[i-1, :-1]/(y_grid[1:-1] - y_grid[:-2])**2 + \
                                lambdas_y[i-1, 1:]/(y_grid[2:] - y_grid[1:-1])**2
            upper_diag_y[0] = 0
            upper_diag_y[1:] = -lambdas_y[i-1, 1:] / (y_grid[2:] - y_grid[1:-1])**2
            lower_diag_y[-1] = 0
            lower_diag_y[:-1] = -lambdas_y[i-1, :-1] / (y_grid[1:-1] - y_grid[:-2])**2
            T_field_next[i, :] = TDMA_solve(main_diag_y, upper_diag_y, lower_diag_y, b_y)

        T_field_prev = T_field_next.copy()

    #print(T_field_next)
    print('Min/max temperature: {}, {}'.format(np.min(T_field_next.ravel()),
                                               np.max(T_field_next.ravel())))
    pretty_plot(T_field_next)


if __name__ == '__main__':
    solver(1)

# vim: set tw=0:
