#!/usr/bin/env python3

import sys
from numpy import empty, linspace, array, copy
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from scipy.special import erfc
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from math import sqrt

### SIZES ###
# spacial
H = 0.01
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
K = 1.0
SIGMA = 0.7
CS = sqrt(U1**SIGMA * K / SIGMA)
lambda_nonlin = lambda u: K * u**SIGMA
### 3-layer method parameters ###
KSI = 0.5
#KSI = 0.5 + H**2 / (12*NU*TAU)

# very small number
EPS = 1e-20

X_SIZE = int(L/H - 1)
X_VALUES = linspace(0, L, X_SIZE+1)[1:]
T_VALUES = linspace(0, T, T / TAU)

class AnimatedPlot:
    """For animated result output using matplotlib.animate."""
    sequence = []
    fig = plt.figure()
    INTERVAL = 25
    def __init__(self, legend=None):
        """Initialize plot (optionally with a legend)."""
        plt.axis([0., L, -0.01, 0.3])
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
            u.append(0)
        else:
            u.append(((SIGMA*CS/K) * (CS*t - x))**(1/SIGMA))
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
    else:
        return False

    amplot = AnimatedPlot(('Exact solution', 'Implicit solver'))
    u_prev, u_prev_2 = array(exact(0)), array(exact(0))  # initial conditions
    implicit_b = empty((1, X_SIZE))
    main_diag = empty(X_SIZE)
    lower_diag, upper_diag = empty(X_SIZE-1), empty(X_SIZE-1)
    lambdas, lambdas_new = empty(X_SIZE), empty(X_SIZE+1)
    u_extended = empty(X_SIZE+2)

    for t in T_VALUES:
        u_exact = exact(t)
        if t == 0:
            amplot.add_frame(u_exact, u_prev)
            continue

        # Fill A matrix for given t
        if equation_mode == 'linear':
            main_diag[:] = (1+KSI)/TAU - A
            upper_diag[:], lower_diag[:] = -B, -C
        else:       # nonlinear
            u_extended[1:-1] = u_prev
            u_extended[0], u_extended[-1] = u0(t), u_extended[-2]
            lambdas = lambda_nonlin(u_extended)

            lambdas_new = (2 * lambdas[:-1] * lambdas[1:] + EPS) / (lambdas[:-1] + lambdas[1:] + EPS)

            main_diag = (1 + KSI)/TAU + (lambdas_new[1:] + lambdas_new[:-1]) / H**2
            upper_diag = - lambdas_new[1:-1] / H**2
            lower_diag = - lambdas_new[1:-1] / H**2
            # set B, C for boundary points for compatibility
            C = lambdas_new[0] / H**2
            B = lambdas_new[-1] / H**2

        # Fill b matrix
        implicit_b = (1+2*KSI)/TAU * u_prev - KSI/TAU * u_prev_2
        # Set boundary conditions
        implicit_b[0] += C*u0(t)   # left
        main_diag[-1] -= B         # right

        # Solve Ax=b and unpack
        u_prev_2, u_prev = u_prev, u_prev_2
        u_prev = TDM_solve(main_diag, upper_diag, lower_diag, implicit_b)

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
