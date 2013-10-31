#!/usr/bin/env python3

import sys
from numpy import empty, linspace
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from scipy.special import erfc
from math import sqrt

### SIZES ###
# spacial
H = 0.001
L = 1
# time
TAU = 0.01
T = 1
### Initial parameters ###
U0 = 0      # initial ("cold") temperature
U1 = 1      # heater temperature
# Linear equation parameters
NU = 0.5
# Nonlinear equation parameters
K = 1
SIGMA = 0.5
C = sqrt(U1**SIGMA * K / SIGMA)

X_VALUES = linspace(0, L, L / H)
T_VALUES = linspace(0, T, T / TAU)

class AnimatedPlot:
    """For animated result output using matplotlib.animate."""
    sequence = []
    fig = plt.figure()
    def __init__(self):
        plt.axis([0., L, -0.5, 1.5])
    def add_frame(self, t, data1, data2=None, data3=None):
        if data2 == None:
            self.sequence.append(plt.plot(X_VALUES, data1, 'r'))
        elif data3 == None:
            self.sequence.append(plt.plot(X_VALUES, data1, 'r',
                                          X_VALUES, data2, 'b'))
        else:
            self.sequence.append(plt.plot(X_VALUES, data1, 'r',
                                          X_VALUES, data2, 'b',
                                          X_VALUES, data3, 'g'))
    def finalize(self):
        animated_seq = ArtistAnimation(self.fig, self.sequence,
                                       interval=100, blit=True)
        plt.show()

def exact_linear(t):
    """Exact solution of linear equation."""
    if t > 0:
        return [U1 * erfc(0.5*x / sqrt(NU*t)) for x in X_VALUES]
    else:
        return [U0 for x in X_VALUES]

def exact_nonlinear(t):
    """Exact solution of nonlinear equation."""
    u = []
    for x in X_VALUES:
        if x > C*t:
            u.append(0.)
        else:
            u.append((SIGMA*C/K) * (C*t - x)**(1/SIGMA))
    return u

def solve_3_layer(equation_num):
    amplot = AnimatedPlot()
    if equation_num == '1':
        for t in T_VALUES:
            u_exact = exact_linear(t)
            amplot.add_frame(t, u_exact)
        amplot.finalize()
    else:
        for t in T_VALUES:
            u_exact = exact_nonlinear(t)
            amplot.add_frame(t, u_exact)
        amplot.finalize()

if __name__ == '__main__':
    solve_3_layer(sys.argv[1])
