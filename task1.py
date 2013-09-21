#!/usr/bin/env python3

import sys
from numpy import empty, linspace, float64
from math import floor
from os import system, remove

# Constants:
A = 1
X0 = 1
F1 = 0
F2 = 1
# size
L = 100
H = 0.1
# time
T = 100
TAU = 10.

GRID_X_SIZE = floor(L / H)
GNUPLOT_CMD = "gnuplot -e data1=\\'{dat}\\' \
                       -e pngfile=\\'task1-out/{output}.png\\' \
                       -e xmax=\\'" + str(L) + "\\' \
                       task1.gpt"

def plot_result(t, data1, data2=None):
    """Output results neatly."""
    tmp_paths = 'tmp1.dat'
    data1.tofile(tmp_paths)
    system(GNUPLOT_CMD.format(dat=tmp_paths, output=int(t*TAU)))
    remove(tmp_paths)

def analytical(t):
    """Analytical view at some t."""
    f_grid = empty([GRID_X_SIZE, 2], dtype=float64)
    x_n = int((X0 + A*t) / H)
    f_grid[:, 0] = linspace(0, L, GRID_X_SIZE)
    f_grid[:x_n, 1], f_grid[x_n:, 1] = F2, F1
    return f_grid

def simple_1st_order():
    for t in linspace(0, T, floor(T / TAU) + 1):
        f1 = analytical(t)
        plot_result(t, f1)

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
