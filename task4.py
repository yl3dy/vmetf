#!/usr/bin/python3

from numpy import empty, linspace, empty_like, zeros
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from math import floor, sqrt
import os
try:
    import cython
    import pyximport; pyximport.install()
    import task4_aux as aux
except ImportError:
    import task4_aux_simple as aux

### Physical parameters
# Left
RHO_LEFT = 1
U_LEFT = 0
E_LEFT = 1
# Right
RHO_RIGHT = 0.3
U_RIGHT = 0
E_RIGHT = 1
GAMMA = 1.4
### Model parameters
# Coordinates
X_LIMITS = [-1., 1.]
NODE_NUM = 200    # including borders
DX = abs(X_LIMITS[0] - X_LIMITS[1]) / (NODE_NUM - 1)
X_VALUES = linspace(min(X_LIMITS), max(X_LIMITS), NODE_NUM)
# Time
TAU = 0.001
T = 0.5
T_VALUES = linspace(0, T, floor(T / TAU + 1))
# Other
SKIP = 20
Q = 0.04

def draw_picture(i, rho, u, e, P):
    plt.axis([-1, 10, 0, 5])
    plt.plot(X_VALUES, rho, 'r', X_VALUES, u, 'b', X_VALUES, e, 'g', X_VALUES, P, 'm')
    plt.legend(['rho', 'u', 'e', 'P'])
    plt.savefig('/tmp/vmetf/{:0>5d}.png'.format(i))
    plt.clf()

def draw(result_exact, result_harlow, result_lw):
    if not len(result_lw) == len(result_harlow) == len(result_exact):
        print(len(result_lw), len(result_harlow), len(result_exact))
    for i in range(len(result_exact)):
        # density
        exact_cur = result_exact[i][0]
        harlow_cur = result_harlow[i][0]
        lw_cur = result_lw[i][0]
        plt.axis([-1.2, 1.2, 0, 1.5])
        plt.plot(X_VALUES, exact_cur, 'r', X_VALUES, harlow_cur, 'b', X_VALUES, lw_cur, 'g')
        plt.legend(['Exact', 'Harlow', 'Lax-Wendroff'])
        plt.savefig('/tmp/vmetf/density_{:0>5d}.png'.format(i))
        plt.clf()
        # pressure
        exact_cur = result_exact[i][1]
        harlow_cur = result_harlow[i][1]
        lw_cur = result_lw[i][1]
        plt.axis([-1.2, 1.2, 0, 0.6])
        plt.plot(X_VALUES, exact_cur, 'r', X_VALUES, harlow_cur, 'b', X_VALUES, lw_cur, 'g')
        plt.legend(['Exact', 'Harlow', 'Lax-Wendroff'])
        plt.savefig('/tmp/vmetf/pressure_{:0>5d}.png'.format(i))
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

    result_list = []
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
            result_list.append([rho.copy(), P.copy()])
            #draw_picture(i, rho, u, e, P)

        u, u_next = u_next, u
        e, e_next = e_next, e
        rho, rho_next = rho_next, rho

    return result_list

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

    result_list = []
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
        rho, u, e = aux.smoothing(rho, u, e, Q)
        f_old = get_f(rho, u, e)
        F_old = get_F(rho, u, e)
        if i % SKIP == 0:
            result_list.append([rho, P_state(rho, u, e)])
            #draw_picture(i, rho, u, e, P_state(rho, u, e))

    return result_list

def exact():
    """Exact iterative solution."""
    P1 = P_state(RHO_LEFT, U_LEFT, E_LEFT)
    P2 = P_state(RHO_RIGHT, U_RIGHT, E_RIGHT)
    P = 0.5 * (P1 + P2)

    # Determine params in iterative process
    for it in range(200):
        if P >= P1:
            a1 = sqrt(RHO_LEFT * (0.5*(GAMMA+1)*P + 0.5*(GAMMA-1)*P1))
        else:
            c1 = sqrt(GAMMA * P1 / RHO_LEFT)
            a1 = (GAMMA - 1) * 0.5 * RHO_LEFT * c1 * (1 - P/P1) / \
                 (GAMMA*(1 - (P/P1)**((GAMMA-1)*0.5/GAMMA)))

        if P >= P2:
            a2 = sqrt(RHO_RIGHT*( 0.5*(GAMMA+1)*P + 0.5*(GAMMA-1)*P2 ))
        else:
            c2 = sqrt(GAMMA * P2 / RHO_RIGHT)
            a2 = (GAMMA - 1) * 0.5 * RHO_RIGHT * c2 * (1 - P/P2) / \
                 (GAMMA*(1 - (P/P2)**((GAMMA-1)*0.5/GAMMA)))

        z = P / (P1 + P2)
        a = (GAMMA - 1) * (1 - z) / \
            (3 * GAMMA * z**((GAMMA+1)*0.5/GAMMA) * (1 - z**((GAMMA-1)*0.5/GAMMA))) - 1
        if a <= 0: a = 0
        fi = (a2*P1 + a1*P2 + a1*a2*(U_LEFT - U_RIGHT)) / (a1 + a2)
        P = (a*P + fi) / (1 + a)

    U = (a1*U_LEFT + a2*U_RIGHT + P1 - P2) / (a1 + a2)
    # Левая ударная волна
    if P >= P1:
        D1 = U_LEFT - a1 / RHO_LEFT
        R1 = RHO_LEFT * a1 / (a1 - RHO_LEFT*(U_LEFT - U))
    else:    # Левая волна разряжения
        v1 = U_LEFT - c1
        c11 = c1 + 0.5*(GAMMA - 1)*(U_LEFT - U)
        R1 = GAMMA * P / c11**2
        v11 = U - c11

    if P >= P2:
        D2 = U_RIGHT + a2 / RHO_RIGHT
        R2 = RHO_RIGHT * a2 / (a2 + RHO_RIGHT*(U_RIGHT - U))
    else:
        v2 = U_RIGHT + c2
        c22 = c2 - 0.5*(GAMMA-1)*(U_RIGHT - U)
        R2 = GAMMA * P / c22**2
        v22 = U + c22
    ############################################

    Rho, uuu, eee = initial_conditions()
    P_arr = P_state(Rho, uuu, eee)
    result_list = []
    for t in T_VALUES:
        ii = float(t / TAU)

        if P1 > P:
            # s - approximation for velocity
            for s in range(1, 1001):
                v = v11 + (s - 1)*(v1 - v11) / 1000.
                c = (GAMMA-1)*(U_LEFT - v) / (GAMMA+1) + 2*c1 / (GAMMA+1)
                i = round(1 + (t*v + 1) / DX)
                P_arr[i] = P1 * (c/c1)**((2*GAMMA) / (GAMMA-1))
                Rho[i] = RHO_LEFT * (c/c1)**(2/(GAMMA-1))
        if P2 > P:
            for s in range(1, 1001):
                v = v2 + (s - 1) * (v22 - v2) / 1000.
                c = - (GAMMA-1)*(U_RIGHT - v) / (GAMMA+1) + 2*c2 / (GAMMA+1)
                i = round(1 + (v*t + 1) / DX)
                P_arr[i] = P2 * (c / c2)**((2*GAMMA) / (GAMMA-1))
                Rho[i] = RHO_RIGHT * (c/c2)**(2/(GAMMA-1))
        if P2 < P:
            i_min = round(1 + (D2*(ii-1)*TAU + 1) / DX)
            i_max = round(1 + (D2*ii*TAU + 1) / DX)
            P_arr[i_min:i_max+1] = P
            Rho[i_min:i_max+1] = R2
        if P1 < P:
            i_max = round(1 + (D1*ii*TAU + 1) / DX)
            i_min = round(1 + (D1*(ii-1)*TAU + 1) / DX)
            P_arr[i_min:i_max+1] = P
            Rho[i_min:i_max+1] = R1
        if U > 0:
            i_min = round(1 + (U*(ii-1)*TAU + 1) / DX)
            i_max = round(1 + (U*ii*TAU + 1) / DX)
            P_arr[i_min:i_max+1] = P
            Rho[i_min:i_max+1] = R1
        if U < 0:
            i_min = round(1 + (U*ii*TAU + 1) / DX)
            i_max = round(1 + (U*(ii-1)*TAU + 1) / DX)
            P_arr[i_min:i_max+1] = P
            Rho[i_min:i_max+1] = R2

        if ii % SKIP == 0:
            result_list.append([Rho.copy(), P_arr.copy()])
    return result_list


def main():
    os.system('rm /tmp/vmetf/*')
    print('Calculating exact')
    result_exact = exact()
    print('Calculating Harlow')
    result_harlow = harlow()
    print('Calculating Lax-Wendroff')
    result_lw = lax_wendroff()
    print('Drawing results')
    draw(result_exact, result_harlow, result_lw)

if __name__ == '__main__':
    main()

# vim: set tw=100:
