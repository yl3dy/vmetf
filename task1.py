#!/usr/bin/env python3

import sys
from numpy import empty, linspace, float64
from math import floor
import matplotlib.pyplot as plt
from os import system

# Constants:
A = 1
X0 = 0.1
F1 = 0
F2 = 1
# size
L = 1
H = 0.01
# time
T = 1
TAU = 0.001

PLOT_NUM = 20    # number of iters to plot

GRID_X_SIZE = floor(L / H) + 1
GRID_T_SIZE = floor(T / TAU) + 1
X_VALUES = linspace(0, L, GRID_X_SIZE)
T_VALUES = linspace(0, T, GRID_T_SIZE)

def plot_result(t, data1, data2, data3):
    """Output results neatly."""
    plt.axis([0., L, -0.1, F2*1.1])
    plt.plot(X_VALUES, data1, X_VALUES, data2, X_VALUES, data3)
    plt.savefig('task1-out/{}.png'.format(t))
    plt.clf()

def exact_solution(t):
    """Analytical view at some t."""
    f_grid = empty([GRID_X_SIZE])
    x_n = int((X0 + A*t) / H)
    f_grid[:x_n], f_grid[x_n:] = F2, F1
    return f_grid

def simple_1st_order():
    """Simple 1st order method."""
    f_linear, f_linear_old = empty([GRID_X_SIZE]), empty([GRID_X_SIZE])
    f_nonlin, f_nonlin_old = empty([GRID_X_SIZE]), empty([GRID_X_SIZE])
    f_linear[0] = f_nonlin[0] = F2
    f_linear_old = exact_solution(0.)
    f_nonlin_old = exact_solution(0.)
    for n in range(GRID_T_SIZE):
        f_exact = exact_solution(T_VALUES[n])
        if n == 0:
            plot_result(n, f_exact, f_exact, f_exact)
            continue
        for i in range(1, GRID_X_SIZE):
            f_linear[i] = f_linear_old[i] - (A*TAU / H) * (f_linear_old[i] -
                          f_linear_old[i-1])
            f_nonlin[i] = f_nonlin_old[i] - (TAU / H) * (f_nonlin_old[i]**2 - f_nonlin_old[i-1]**2)
        f_linear, f_linear_old = f_linear_old, f_linear
        f_nonlin, f_nonlin_old = f_nonlin_old, f_nonlin
        # do not plot too often
        if n % int(floor(GRID_T_SIZE / PLOT_NUM)) == 0:
            plot_result(n, f_exact, f_linear, f_nonlin)

def lax_wendroff():
    f_linear, f_linear_old = empty([GRID_X_SIZE]), empty([GRID_X_SIZE])
    f_nonlin, f_nonlin_old = empty([GRID_X_SIZE]), empty([GRID_X_SIZE])
    f_linear[0], f_nonlin[0] = F2, F2
    f_linear_old, f_nonlin_old = exact_solution(0.), exact_solution(0.)
    for n in range(GRID_T_SIZE):
        f_exact = exact_solution(T_VALUES[n])
        if n == 0:
            plot_result(n, f_exact, f_exact, f_exact)
            continue
        f_plus_lin, f_minus_lin = F2, F2
        f_plus_nonlin, f_minus_nonlin = F2, F2
        for i in range(1, GRID_X_SIZE-1):
            f_plus_lin = 0.5 * (f_linear_old[i] + f_linear_old[i+1]) - \
                         (A*TAU)/(2*H) * (f_linear_old[i+1] -
                                          f_linear_old[i])
            f_linear[i] = f_linear_old[i] - (A*TAU/H) * (f_plus_lin -
                                                         f_minus_lin)
            f_plus_nonlin = 0.5 * (f_nonlin_old[i] + f_nonlin_old[i+1]) - \
                            (TAU)/(2*H) * (f_nonlin_old[i+1]**2 -
                                           f_nonlin_old[i]**2)
            f_nonlin[i] = f_nonlin_old[i] - (TAU/H) * (f_plus_nonlin**2 -
                                                              f_minus_nonlin**2)
            f_plus_lin, f_minus_lin = f_minus_lin, f_plus_lin
            f_plus_nonlin, f_minus_nonlin = f_minus_nonlin, f_plus_nonlin
        f_linear[-1] = f_linear[-2]
        f_linear, f_linear_old = f_linear_old, f_linear
        f_nonlin[-1] = f_nonlin[-2]
        f_nonlin, f_nonlin_old = f_nonlin_old, f_nonlin
        # do not plot too often
        if n % int(floor(GRID_T_SIZE / PLOT_NUM)) == 0:
            plot_result(n, f_exact, f_linear, f_nonlin)

def clean_output():
    system('rm task1-out/*')

def main():
    """Entry point."""
    if len(sys.argv) != 2 or int(sys.argv[1]) not in range(1,6):
        print('Bad arguments. Should specify only method number (1-5)')
        quit()
    arg = sys.argv[1]
    prompt = lambda desc: 'Using method {}: {}'.format(arg, desc)
    clean_output()
    if arg == '1':
        print(prompt('simple 1st order'))
        simple_1st_order()
    elif arg == '2':
        print(prompt(' Lax-Wendroff'))
        lax_wendroff()
    elif arg == '3':
        print(prompt(''))
    elif arg == '4':
        print(prompt(''))
    elif arg == '5':
        print(prompt(''))

if __name__ == '__main__':
    main()
