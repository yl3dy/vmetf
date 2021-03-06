#!/usr/bin/env python3

import sys
from numpy import empty, linspace, float64
from math import floor
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from os import system

# Constants:
A = 0.5
X0 = 0.1
F1 = 0
F2 = 1
Q = 0.1
# size
L = 1
H = 0.002
# time
T = 1
TAU = 0.001

PLOT_NUM = 250    # number of iters to plot

GRID_X_SIZE = floor(L / H) + 1
GRID_T_SIZE = floor(T / TAU) + 1
X_VALUES = linspace(0, L, GRID_X_SIZE)
T_VALUES = linspace(0, T, GRID_T_SIZE)

def plot_result(t, data1, data2, data3):
    """Output results in image files."""
    plt.axis([0., L, -0.1, F2*1.1])
    plt.plot(X_VALUES, data1, X_VALUES, data2, X_VALUES, data3)
    plt.savefig('task1-out/{}.png'.format(t))
    plt.clf()

class AnimatedPlot:
    """For animated result output using matplotlib.animate."""
    sequence = []
    fig = plt.figure()
    def __init__(self):
        plt.axis([0., L*0.5, -0.1, 1.1])
    def add_frame(self, t, data1, data2, data3=None):
        if not data3 == None:
            self.sequence.append(plt.plot(X_VALUES, data1, 'r',
                                          X_VALUES, data2, 'b',
                                          X_VALUES, data3, 'g'))
        else:
            self.sequence.append(plt.plot(X_VALUES, data1, 'r',
                                          X_VALUES, data2, 'b'))
    def finalize(self):
        animated_seq = ArtistAnimation(self.fig, self.sequence,
                                       interval=100, blit=True)
        plt.show()

def exact_solution(t):
    """Analytical solution at some t."""
    f_grid = empty([GRID_X_SIZE])
    x_n = int((X0 + A*t) / H)
    f_grid[:x_n], f_grid[x_n:] = F2, F1
    return f_grid

def smoothing(f_next):
    f_tmp = f_next.copy()
    for i in range(2, GRID_X_SIZE-2):
        D_m = f_next[i] - f_next[i-1]
        D_mm = f_next[i-1] - f_next[i-2]
        D_p = f_next[i+1] - f_next[i]
        D_pp = f_next[i+2] - f_next[i+1]

        if D_p*D_m <= 0 or D_p*D_pp <= 0:
            Q_plus = D_p
        else:
            Q_plus = 0
        if D_p*D_m <= 0 or D_m*D_mm <= 0:
            Q_minus = D_m
        else:
            Q_minus = 0

        f_tmp[i] = f_next[i] + Q * (Q_plus - Q_minus)
    return f_tmp

def linear_methods():
    """Solution of linear equation using simple and Lax-Wendroff methods."""
    anim = AnimatedPlot()
    f_simple, f_simple_old = empty([GRID_X_SIZE]), empty([GRID_X_SIZE])
    f_lw, f_lw_old = empty([GRID_X_SIZE]), empty([GRID_X_SIZE])
    f_simple[0], f_lw[0] = F2, F2
    f_simple_old, f_lw_old = exact_solution(0.), exact_solution(0.)
    for n in range(GRID_T_SIZE):
        f_exact = exact_solution(T_VALUES[n])
        if n == 0:
            anim.add_frame(n, f_exact, f_exact, f_exact)
            continue
        f_plus, f_minus = F2, F2
        for i in range(1, GRID_X_SIZE):
            # Simple 1st order
            f_simple[i] = f_simple_old[i] - (A*TAU / H) * (f_simple_old[i] -
                          f_simple_old[i-1])
            # Lax-Wendroff
            if i < (GRID_X_SIZE-1):
                f_plus = 0.5 * (f_lw_old[i] + f_lw_old[i+1]) - \
                        (A*TAU)/(2*H) * (f_lw_old[i+1] - f_lw_old[i])
                f_lw[i] = f_lw_old[i] - (A*TAU/H) * (f_plus - f_minus)
                f_plus, f_minus = f_minus, f_plus
        f_simple, f_simple_old = f_simple_old, f_simple
        f_lw[-1] = f_lw[-2]
        f_lw = smoothing(f_lw)
        f_lw, f_lw_old = f_lw_old, f_lw
        # do not plot too often
        if n % int(floor(GRID_T_SIZE / PLOT_NUM)) == 0:
            anim.add_frame(n, f_exact, f_simple, f_lw)
    anim.finalize()

def nonlinear_methods():
    """Solution of nonlinear equation (same methods as above)."""
    anim = AnimatedPlot()
    f_simple, f_simple_old = empty([GRID_X_SIZE]), empty([GRID_X_SIZE])
    f_lw, f_lw_old = empty([GRID_X_SIZE]), empty([GRID_X_SIZE])
    f_simple[0], f_lw[0] = F2, F2
    f_simple_old, f_lw_old = exact_solution(0.), exact_solution(0.)
    for n in range(GRID_T_SIZE):
        f_exact = exact_solution(T_VALUES[n])
        if n == 0:
            anim.add_frame(n, f_exact, f_exact, f_exact)
            continue
        f_plus, f_minus = F2, F2
        for i in range(1, GRID_X_SIZE):
            # Simple 1st order
            f_simple[i] = f_simple_old[i] - 0.5 * (TAU / H) * (f_simple_old[i]**2 -
                                                               f_simple_old[i-1]**2)
            # Lax-Wendroff
            if i < (GRID_X_SIZE-1):
                f_plus = 0.5 * (f_lw_old[i] + f_lw_old[i+1]) - \
                         (TAU)/(4*H) * (f_lw_old[i+1]**2 - f_lw_old[i]**2)
                f_lw[i] = f_lw_old[i] - 0.5*(TAU/H) * (f_plus**2 - f_minus**2)
                f_plus, f_minus = f_minus, f_plus
        f_simple, f_simple_old = f_simple_old, f_simple
        f_lw[-1] = f_lw[-2]
        f_lw = smoothing(f_lw)
        f_lw, f_lw_old = f_lw_old, f_lw
        # do not plot too often
        if n % int(floor(GRID_T_SIZE / PLOT_NUM)) == 0:
            anim.add_frame(n, f_simple, f_lw)
    anim.finalize()

def clean_output():
    """Clean image output."""
    system('rm task1-out/* 2> /dev/null')

def main():
    """Entry point."""
    if len(sys.argv) != 2 or int(sys.argv[1]) not in range(1,6):
        print('Bad arguments. Should specify only method number (1-5)')
        quit()
    arg = sys.argv[1]
    prompt = lambda desc: 'Using method {}: {}'.format(arg, desc)
    clean_output()
    if arg == '1':
        print(prompt('linear equation'))
        linear_methods()
    elif arg == '2':
        print(prompt('Burgers equation'))
        nonlinear_methods()

if __name__ == '__main__':
    main()

# vim:set tw=0:
