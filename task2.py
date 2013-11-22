#!/usr/bin/env python3

import sys
from numpy import empty, linspace, array, copyto, zeros
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from scipy.special import erfc
from math import sqrt, floor

### SIZES ###
# spatial
H = 0.025
L = 1
# time
TAU = 0.0001
T = 0.5
### Initial parameters ###
U0 = 1.
### Equation parameters ###
# Linear
NU = 1.
# Nonlinear
K = 1.0
SIGMA = 0.8
CS = sqrt(U0**SIGMA * K / SIGMA)
lambda_nonlin = lambda u: K * u**SIGMA
### 3-layer method parameters ###
KSI = -0.5      # should be < 0 (lazy to correct coeffs in formulae)

# very small number
EPS = 1e-20
SKIP = 10

X_SIZE = floor(L/H - 1)
X_VALUES = linspace(0, L, X_SIZE + 2)[1:-1]
T_VALUES = linspace(0, T, floor(T / TAU + 1))

class AnimatedPlot:
    """For animated result output using matplotlib.animate."""
    sequence = []
    fig = plt.figure()
    INTERVAL = 200
    def __init__(self, legend=None):
        """Initialize plot (optionally with a legend)."""
        plt.axis([0., 1.0, -0.01, 1.2])
        self.legend = legend
    def add_frame(self, *args):
        """Add a frame to animation."""
        colors = ['b', 'r', 'g']
        if len(args) > len(colors):
            print('Too many arguments to AnimatedPlot.add_frame')
        plot_args = sum([[X_VALUES, args[i], colors[i]]
                         for i in range(len(args))], [])
        if self.legend: plt.legend(self.legend)
        self.sequence.append(plt.plot(*plot_args))
    def finalize(self):
        """Show animated plot."""
        _ = ArtistAnimation(self.fig, self.sequence, interval=self.INTERVAL,
                            blit=True)
        plt.show()

def check_parabolic_courant():
    """Check if Courant number is in the bounds of stability."""
    parabolic_courant = 0.5*TAU*NU / H**2
    if parabolic_courant > 1.0:
        print('Warning! Courant number is bad: ' + str(parabolic_courant))

def exact_linear(t):
    """Exact solution of linear equation."""
    if t > 0:
        return array([U0 * erfc(0.5*x / sqrt(NU*t)) for x in X_VALUES])
    else:
        return zeros(X_SIZE)

def exact_nonlinear(t):
    """Exact solution of nonlinear equation."""
    u = zeros(X_SIZE)
    front = floor(CS*t / H)
    for i in range(front):
        u[i] = ((SIGMA*CS/K) * (CS*t - (i+1)*H))**(1/SIGMA)
    return u

def TDM_solve(main_diag, upper_diag, lower_diag, b):
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

def get_lambdas(u_extended, lambdas_new, u_prev, u0):
    """Calculate \lambda_+, \lambda_- and u_extended."""
    u_extended[1:-1] = u_prev
    u_extended[0], u_extended[-1] = u0, u_extended[-2]
    lambdas = lambda_nonlin(u_extended)
    #lambdas_new = (2 * lambdas[:-1] * lambdas[1:] + EPS) / (lambdas[:-1] + lambdas[1:] + EPS)
    lambdas_new = 0.5 * (lambdas[:-1] + lambdas[1:])
    return lambdas_new, u_extended

def solve_3_layer(equation_mode):
    """General solver for (non)linear equation."""
    if equation_mode == 'linear':
        exact = exact_linear
        A, B, C = -2*NU / H**2, NU / H**2, NU / H**2
        u0 = lambda t: U0
        check_parabolic_courant()
    elif equation_mode == 'nonlinear':
        exact = exact_nonlinear
        u0 = lambda t: U0 * t**(1/SIGMA)
        lambdas_new = empty(X_SIZE+1)
        lambdas_new_expl = empty(X_SIZE+1)
    else:
        return False

    amplot = AnimatedPlot(('Exact solution', 'Implicit solver', 'Explicit solver'))

    # Implicit parameters
    u_prev, u_prev_2 = exact(0), exact(0)  # initial conditions
    u_extended = empty(X_SIZE+2)
    implicit_b = empty(X_SIZE)
    main_diag = empty(X_SIZE)
    lower_diag, upper_diag = empty(X_SIZE-1), empty(X_SIZE-1)

    # Explicit parameters
    u_prev_expl = exact(0)   # initial conditions
    u_extended_expl = empty(X_SIZE+2)

    for t in T_VALUES:
        if t == 0:
            amplot.add_frame(exact(t), u_prev, u_prev_expl)
            continue

        # Auxiliary array for explicit method
        u_extended_expl[1:-1] = u_prev_expl
        u_extended_expl[0], u_extended_expl[-1] = u0(t), u_extended_expl[-2]
        # Fill A matrix for given t
        if equation_mode == 'linear':
            main_diag[:] = (1+KSI)/TAU - A
            upper_diag[:], lower_diag[:] = -B, -C
        else:       # nonlinear
            lambdas_new_expl, u_extended_expl = get_lambdas(u_extended_expl, lambdas_new_expl, u_prev_expl, u0(t))
            lambdas_new, u_extended = get_lambdas(u_extended, lambdas_new, u_prev, u0(t))

            # Implicit-only part
            main_diag = (1 + KSI)/TAU + (lambdas_new[1:] + lambdas_new[:-1])/H**2
            upper_diag = - lambdas_new[1:-1] / H**2
            lower_diag = - lambdas_new[1:-1] / H**2
            # set B, C for boundary points for compatibility
            C = lambdas_new[0] / H**2
            B = lambdas_new[-1] / H**2

        # Fill b vector
        implicit_b = (1 + 2*KSI)/TAU * u_prev - KSI/TAU * u_prev_2
        # Set boundary conditions
        implicit_b[0] += C*u0(t)   # left
        main_diag[-1] -= B         # right

        # Solve implicit
        copyto(u_prev_2, u_prev)
        u_prev = TDM_solve(main_diag, upper_diag, lower_diag, implicit_b)

        # Solve explicit
        if equation_mode == 'linear':
            u_prev_expl = TAU * ((1/TAU + A) * u_extended_expl[1:-1] +
                                 B * (u_extended_expl[2:] + u_extended_expl[:-2]))
        else:
            u_prev_expl = TAU * ((1/TAU - (lambdas_new_expl[1:] + lambdas_new_expl[:-1])/H**2) * u_extended_expl[1:-1] +
                                 lambdas_new_expl[1:] * u_extended_expl[2:] / H**2 +
                                 lambdas_new_expl[:-1] * u_extended_expl[:-2] / H**2)

        if (t / TAU) % SKIP == 0:
            amplot.add_frame(exact(t), u_prev, u_prev_expl)
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
