#!/usr/bin/env python3

import sys
from numpy import empty, linspace, array
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from scipy.special import erfc
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from math import sqrt

### SIZES ###
# spacial
H = 0.005
L = 1
# time
TAU = 0.002
T = 0.6
### Initial parameters ###
U0 = 0.      # initial ("cold") temperature
U1 = 1.      # heater temperature
### Equation parameters ###
# Linear
NU = 1
# Nonlinear
K = 1
SIGMA = 0.5
CS = sqrt(U1**SIGMA * K / SIGMA)
### 3-layer method parameters ###
KSI = 0.
#KSI = 0.5 + H**2 / (12*NU*TAU)

X_SIZE = int(L/H - 1)
X_VALUES = linspace(0, L, X_SIZE+1)[1:]
T_VALUES = linspace(0, T, T / TAU)

class AnimatedPlot:
    """For animated result output using matplotlib.animate."""
    sequence = []
    fig = plt.figure()
    INTERVAL = 50
    def __init__(self, legend=None):
        """Initialize plot (optionally with a legend)."""
        plt.axis([0., L, -0.5, 1.5])
        if legend: plt.legend(legend)
    def add_frame(self, *args):
        """Add a frame to animation."""
        colors = ['b', 'r', 'g']
        if len(args) > len(colors):
            print('Too many arguments to AnimatedPlot.add_frame')
        plot_args = sum([[X_VALUES, args[i], colors[i]]
                         for i in range(len(args))], [])
        self.sequence.append(plt.plot(*plot_args))
    def finalize(self):
        """Show animated plot."""
        _ = ArtistAnimation(self.fig, self.sequence, interval=self.INTERVAL,
                            blit=True)
        plt.show()

def check_parabolic_courant():
    """Check if Courant number is in the bounds of stability."""
    parabolic_courant = 0.5*TAU*NU / H**2
    if parabolic_courant <= 1.0:
        print('Warning! Courant number is bad: ' + str(parabolic_courant))

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
        if x > CS*t:
            u.append(U0)
        else:
            u.append((SIGMA*CS/K) * (CS*t - x)**(1/SIGMA))
    return u

def solve_3_layer(equation_mode):
    """General solver for (non)linear equation."""
    if equation_mode == 'linear':
        exact = exact_linear
        A, B, C = -2*NU / H**2, NU / H**2, NU / H**2
        u0 = lambda t: U1
        check_parabolic_courant()
    elif equation_mode == 'nonlinear':
        exact = exact_nonlinear
        u0 = lambda t: U1 * t**(1/SIGMA)
        # TODO: nonlinear
    else:
        return False

    amplot = AnimatedPlot()
    u_prev, u_prev_2 = array(exact(0)), array(exact(0))  # initial conditions
    implicit_matrix, = lil_matrix((X_SIZE, X_SIZE))
    implicit_b = empty((1, X_SIZE))
    main_diag = empty([X_SIZE])
    lower_diag, upper_diag = empty([X_SIZE-1]), empty([X_SIZE-1])

    for t in T_VALUES:
        u_exact = exact(t)
        if t == 0:
            amplot.add_frame(u_exact, u_prev)
            continue

        # Fill A matrix for given t
        if equation_mode == 'linear':
            main_diag[:] = (1+KSI)/TAU - A
            upper_diag[:], lower_diag[:] = -B, -C
        else:
            # TODO: nonlinear
            quit()
        implicit_matrix.setdiag(main_diag)
        implicit_matrix.setdiag(upper_diag, k=1)
        implicit_matrix.setdiag(lower_diag, k=-1)
        # Fill b matrix
        implicit_b = (1+2*KSI)/TAU * u_prev - KSI/TAU * u_prev_2
        # Set boundary conditions
        implicit_b[0] += C*u0(t)        # left
        implicit_matrix[-1, -1] -= B    # right #1
        #implicit_matrix[-1, -1] -= 2*B    # right #2
        #implicit_matrix[-1, -2] += B      # --"--

        # Solve Ax=b and unpack
        u_prev_2, u_prev = u_prev, u_prev_2
        u_prev = spsolve(implicit_matrix.tocsr(), implicit_b)

        amplot.add_frame(u_exact, u_prev)
    amplot.finalize()

def main():
    """Entry point of programm."""
    equations = {'1': 'linear', '2': 'nonlinear'}
    try:
        solve_3_layer(equations[sys.argv[1]])
    except (KeyError, IndexError):
        print('Specify equation. 1 - linear, 2 - nonlinear')

if __name__ == '__main__':
    main()
