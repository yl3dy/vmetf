#!/usr/bin/env python3

import sys
from numpy import empty, linspace, array, nan_to_num
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
T = 0.3
### Initial parameters ###
U0 = 0.      # initial ("cold") temperature
U1 = 1.      # heater temperature
### Equation parameters ###
# Linear
NU = 1
# Nonlinear
K = 2.0
SIGMA = 0.7
CS = sqrt(U1**SIGMA * K / SIGMA)
lambda_nonlin = lambda u: K * u**(SIGMA)
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
    INTERVAL = 100
    def __init__(self, legend=None):
        """Initialize plot (optionally with a legend)."""
        plt.axis([0., L, -0.1, 0.1])
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
        u1 = (SIGMA * CS**2 / K)**(1/SIGMA)
        u0 = lambda t: u1 * t**(1/SIGMA)
        # TODO: nonlinear
    else:
        return False

    amplot = AnimatedPlot(('Exact solution', 'Implicit solver'))
    u_prev, u_prev_2 = array(exact(0)), array(exact(0))  # initial conditions
    implicit_matrix = lil_matrix((X_SIZE, X_SIZE))
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
            u_extended[0], u_extended[-1] = u1, u_extended[-2]
            lambdas = lambda_nonlin(u_extended)

            #for i in range(X_SIZE+1):
                #if lambdas[i+1] > 0:
                    #lambdas_new[i] = 2*lambdas[i]*lambdas[i+1] / (lambdas[i]+lambdas[i+1])
                #elif lambdas[i] > 0:
                    #lambdas_new[i] = 0.5 * lambdas[i]
                #else:
                    #lambdas_new[i] = 0

            #lambdas[1:], lambdas[:-1] = lambda_prev, lambda_next
            #lambdas_new[lambda_next > 0] = 2*lambda_next[lambda_next > 0]*lambda_prev[lambda_next > 0]
            #lambdas_new[lambda_next == 0] = 0.5 * lambda_prev[lambda_next == 0]
            #lambdas_new[lambda_prev == 0] = 0

            #lambdas_new = 2 * lambdas[:-1] * lambdas[1:] / (lambdas[:-1] + lambdas[1:])
            #lambdas_new[lambdas_new == 0] = 0.5 * lambdas[lambdas_new == 0]
            #lambdas_new = nan_to_num(lambdas_new)

            lambdas_new = 0.5 * (lambdas[:-1] + lambdas[1:])

            main_diag = (1 + KSI)/TAU + (lambdas_new[1:] + lambdas_new[:-1]) / H**2
            upper_diag = - lambdas_new[1:-1] / H**2
            lower_diag = - lambdas_new[1:-1] / H**2
            # set B, C for boundary points for compatibility
            C = lambdas_new[0] / H**2
            B = lambdas_new[-1] / H**2

            #if t > 0.1:
                #print(lambdas_new)
                ##print(main_diag)
                #quit()
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
