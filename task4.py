#!/usr/bin/python3

from numpy import empty, linspace, empty_like
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from math import floor
import os

### Physical parameters
# Left
RHO_LEFT = 2
U_LEFT = 2
E_LEFT = 2
# Right
RHO_RIGHT = 1
U_RIGHT = 1
E_RIGHT = 1
GAMMA = 3.5
### Model parameters
# Coordinates
X_LIMITS = [-10., 10.]
NODE_NUM = 100    # including borders
DX = abs(X_LIMITS[0] - X_LIMITS[1]) / (NODE_NUM - 1)
X_VALUES = linspace(min(X_LIMITS), max(X_LIMITS), NODE_NUM)
# Time
TAU = 0.0001
T = 0.5
T_VALUES = linspace(0, T, floor(T / TAU + 1))
# Other
SKIP = 200

def draw_picture(i, rho, u, e, P):
    plt.axis([-1, 10, 0, 5])
    plt.plot(X_VALUES, rho, 'r', X_VALUES, u, 'b', X_VALUES, e, 'g', X_VALUES, P, 'm')
    plt.legend(['rho', 'u', 'e', 'P'])
    plt.savefig('/tmp/vmetf/{:0>5d}.png'.format(i))
    plt.clf()

def P_state(rho, u, e):
    """State equation."""
    return rho*(e - 0.5*u**2)*(GAMMA - 1)

def intermediate_vals(f):
    """Calculate f_+ and f_- array."""
    return 0.5 * (f[1:] + f[:-1])

def initial_conditions(harlow=True):
    """Set initial conditions and arrays."""
    rho, u, e = empty(NODE_NUM), empty(NODE_NUM), empty(NODE_NUM)
    rho[X_VALUES < 0] = RHO_LEFT
    rho[X_VALUES >= 0] = RHO_RIGHT
    u[X_VALUES < 0] = U_LEFT
    u[X_VALUES >= 0] = U_RIGHT
    e[X_VALUES < 0] = E_LEFT
    e[X_VALUES >= 0] = E_RIGHT

    return rho, u, e

def harlow():
    """Harlow method implementation."""

    rho, u, e = initial_conditions()
    rho_next, u_next, e_next = initial_conditions()
    u_temp, e_temp = empty_like(u), empty_like(e)
    P_aver = empty(NODE_NUM-1)
    dM, D_minus, D_plus = empty(NODE_NUM-1), empty(NODE_NUM-1), empty(NODE_NUM-1)

    for t in T_VALUES:
        i = int(t/TAU)

        # Euler step
        P = P_state(rho, u, e)
        P_aver = intermediate_vals(P)
        u_aver = intermediate_vals(u)

        u_temp[1:-1] = u[1:-1] - TAU/(DX*rho[1:-1]) * (P_aver[1:] - P_aver[:-1])
        u_temp[0] = u_temp[1]
        u_temp[-1] = u_temp[-2]

        e_temp[1:-1] = e[1:-1] - TAU/(DX*rho[1:-1]) * (P_aver[1:]*u_aver[1:] - P_aver[:-1]*u_aver[:-1])
        e_temp[0] = e_temp[1]
        e_temp[-1] = e_temp[-2]


        # Lagrange step
        u_temp_aver = intermediate_vals(u_temp)
        dM[u_temp_aver > 0] = (rho[:-1])[u_temp_aver > 0]
        dM[u_temp_aver <= 0] = (rho[1:])[u_temp_aver <= 0]
        dM = dM * u_temp_aver * TAU
        D_plus[u_temp_aver < 0] = 1
        D_plus[u_temp_aver >= 0] = 0
        D_minus[u_temp_aver < 0] = 0
        D_minus[u_temp_aver >= 0] = 1

        rho_next[1:-1] = rho[1:-1] + (dM[:-1] - dM[1:]) / DX
        rho_next[0] = rho_next[1]
        rho_next[-1] = rho_next[-2]

        def new_X(X, X_new, rho, rho_new):
            X_new[1:-1] = (D_minus[:-1]*X[:-2]*np.abs(dM[:-1]) + D_plus[1:]*X[2:]*np.abs(dM[1:]) +
                           X[1:-1]*(rho[1:-1]*DX - (1 - D_minus[:-1])*np.abs(dM[:-1]) -
                                    (1 - D_plus[1:])*np.abs(dM[1:]))) / (rho_new[1:-1] * DX)
            X_new[0] = X_new[1]
            X_new[-1] = X_new[-2]
            return X_new

        u_next = new_X(u_temp, u_next, rho, rho_next)
        e_next = new_X(e_temp, e_next, rho, rho_next)

        if i % SKIP == 0:
            draw_picture(i, rho, u, e, P)

        u, u_next = u_next, u
        e, e_next = e_next, e
        rho, rho_next = rho_next, rho

def smoothing():
    pass

def lax_wendroff():
    """Lax-Wendroff method implementation."""
    get_f = lambda rho, u, e: np.vstack([rho, rho*u, rho*e]).T
    get_F = lambda rho, u, e: np.vstack([rho*u, (rho*u)**2 / rho + P_state(rho, u, e),
                                         u * (rho*e + P_state(rho, u, e))]).T
    get_3 = lambda x: (x[:, 0], x[:, 1]/x[:, 0], x[:, 2]/x[:, 0])

    rho, u, e = initial_conditions()
    f_old, F_old = get_f(rho, u, e), get_F(rho, u, e)
    f_temp, F_temp = empty([f_old.shape[0]-1, f_old.shape[1]]), empty([F_old.shape[0]-1, F_old.shape[1]])
    f_new = empty_like(f_old)

    for t in T_VALUES:
        i = floor(t/TAU)
        # Predictor
        f_temp = 0.5*(f_old[1:] + f_old[:-1]) - TAU/DX * (F_old[1:] - F_old[:-1])
        F_temp = get_F(*[intermediate_vals(param) for param in get_3(f_old)])
        # Corrector
        f_new[1:-1] = f_old[1:-1] - TAU/DX * (F_temp[1:] - F_temp[:-1])
        f_new[0] = f_new[1]
        f_new[-1] = f_new[-2]

        rho, u, e = get_3(f_new)
        f_old, f_new = f_new, f_old
        F_old = get_F(rho, u, e)
        if i % SKIP == 0:
            draw_picture(i, rho, u, e, P_state(rho, u, e))



def main():
    os.system('rm /tmp/vmetf/*')
    #harlow()
    lax_wendroff()

if __name__ == '__main__':
    main()

# vim: set tw=100:
