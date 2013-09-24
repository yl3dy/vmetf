#!/usr/bin/env python3

import sys
from numpy import empty, linspace, float64
from math import floor
import matplotlib.pyplot as plt
from os import system

# Constants:
A = 1
X0 = 1
F1 = 0
F2 = 1
# size
L = 100
H = 10
# time
T = 100
TAU = 10

GRID_X_SIZE = floor(L / H) + 1
GRID_T_SIZE = floor(T / TAU) + 1
X_VALUES = linspace(0, L, GRID_X_SIZE)
T_VALUES = linspace(0, T, GRID_T_SIZE)

def plot_result(t, data1, data2=None):
    """Output results neatly."""
    plt.axis([0., L, F1*1.1, F2*1.1])
    if data2:
        plt.plot(X_VALUES, data1, X_VALUES, data2)
    else:
        plt.plot(X_VALUES, data1)
    plt.savefig('task1-out/{}.png'.format(t))
    plt.clf()

def exact_solution(t):
    """Analytical view at some t."""
    f_grid = empty([GRID_X_SIZE])
    x_n = int((X0 + A*t) / H)
    f_grid[:x_n], f_grid[x_n:] = F2, F1
    return f_grid

def simple_1st_order():
    for t in T_VALUES:
        f_exact = exact_solution(t)
        plot_result(t, f_exact)

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
        print(prompt(''))
        simple_1st_order()
    elif arg == '2':
        print(prompt(''))
    elif arg == '3':
        print(prompt(''))
    elif arg == '4':
        print(prompt(''))
    elif arg == '5':
        print(prompt(''))

if __name__ == '__main__':
    main()
