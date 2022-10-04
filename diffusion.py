import numpy as np
from scipy.linalg import inv as spinv
import matplotlib.pyplot as plt
import time 
from numba import jit, njit
from collections import Counter
"constants"
m = 5.68
hbar = 0.6582

dt = 0.001
t = np.arange(0, 1, dt)
dx = 0.001
x = np.arange(0, 5, dx)
#dx = 0.01
p0 = 1
x0 = 2.5
sigma = 0.05
N = len(x) 
psi = np.zeros((len(t), len(x) ), dtype=complex)
psi[0] = 1/(np.pi**0.25 * sigma**0.5 ) * np.exp(-(x-x0)**2/2/sigma**2) * np.exp(1j  * p0 * x/ hbar  )

t0 = 0.05
psi_xt = 1 / (np.pi ** 0.25 * (sigma * (1 + 1j * hbar * t0 / m / sigma**2))**0.5) * np.exp(- (x - (x0 + p0 * t0/ m))**2 / ( 2*sigma**2*(1+1j*hbar*t0/m/sigma**2) ) ) * np.exp(1j * (p0 * x - p0**2 / 2/ m * t0) / hbar ) 
#psi[0] = (1/N) ** 0.5  * np.exp(1j  * p0 * x/ hbar  )
#psi[:, 0], psi[:, -1] =  0, 0  # boundary condition -- infinite potential well

# matrix method
length = len(x)
A = np.zeros((length, length), dtype=complex)
for i in range(length):
    A[i, i-1] = 1
    A[i, (i+1) % length] = 1
    A[0, -1] = 0 # 1: indication of periodical boundary condition 
    A[-1, 0] = 0
AA = -2 + (4j * m * dx **2) /  (hbar * dt)
M = A.copy()
np.fill_diagonal(M, AA)
#M_inv = np.linalg.inv(M)
M_inv = spinv(M, check_finite=False)
B = -A.copy()
np.fill_diagonal(B,  2 + (4j * m * dx **2) /  (hbar * dt) )
MB = M_inv.dot(B)

"""
alpha = 1
k = np.real(np.sum(A, 0))
MB = alpha * dt * A.copy()
np.fill_diagonal(MB, 1 - alpha * dt * k)
"""

def time_evolution(MB, psi):
    for ti in range(0, len(t)-1):
        #B = - np.hstack(( psi[ti, 1:], psi[ti, 0] )) - np.hstack((psi[ti, -1], psi[ti, :-1] )) + psi[ti] * (2+ (4j * m * dx **2) /  (hbar * dt) ) 
        #psi[ti+1] = M_inv.dot(B)
        psi[ti+1]  = MB.dot(psi[ti])
    return psi

def time_evolution_step(psi):
    A = np.zeros((len(t), len(x) ), dtype=complex)
    B = np.zeros((len(t), len(x) ), dtype=complex)
    R = np.zeros((len(t), len(x) ), dtype=complex)
    U = np.zeros((len(t), len(x) ), dtype=complex)
    for ti in range(0, len(t)-1):
        psi[ti, 0] = 0
        psi[ti, -1] = 0
        A[ti, 1:-1] =  -2 + (4j * m * dx **2) /  (hbar * dt)
        B[ti, 1:-1] = - psi[ti, 2:] - psi[ti, :-2]  + psi[ti, 1:-1] * (2+ (4j * m * dx **2) /  (hbar * dt) ) 
        U[ti, 1] = 1/ A[ti, 1]
        R[ti, 1] = B[ti, 1] * U[ti, 1]
        for xi in range(1, len(x)-1):
            U[ti, xi] = 1/(A[ti, xi] - U[ti, xi-1])
            R[ti, xi] = (B[ti, xi] - R[ti, xi-1]) * U[ti, xi]
        psi[ti+1, -2] = R[ti, -2]  # periodic boundary condition
        xi = len(x)-3
        while xi >= 1:
            psi[ti+1, xi] = R[ti, xi] - U[ti, xi] * psi[ti+1, xi+1]
            xi -= 1
    return psi




def dpp(psi, quantum=True):
    length = len(psi[0])
    if quantum:
        rho = np.abs(psi) ** 2
        diff = rho - np.mean(rho[0])
    else:
        diff = np.real(psi) -  np.real( np.mean(psi[0]) ) 
    sign = np.heaviside(diff[:-1] * diff[1:], 1)
    index_list = []
    for i in range(length):
        sign_i = sign[:, i] 
        index = np.where(sign_i ==0)[0]
        if len(index) :
            index_list.append(np.round((index[0] + 1) * dt, 5))
        else:
            index_list.append(round(t[-1], 5))
    diffusion_change_sign = Counter(index_list )
    return diffusion_change_sign
        

"""
diffusion_change_sign = Counter()
diffusion_persistence_dict = dict()
for _ in range(3):
    t1 = time.time()
    psi = np.zeros((len(t), len(x) ), dtype=complex)
    psi[0] = (1/N) ** 0.5  * np.exp(1j  * p0 * x/ hbar  )
    #psi[0] = np.random.normal(size=len(x))
    psi = time_evolution(MB, psi)
    diffusion_change_sign += dpp(psi)
    t2 = time.time()
    print(t2 - t1)

num_trans = 0
for i in sorted(diffusion_change_sign.keys()):
    num_trans += diffusion_change_sign[i]
    diffusion_persistence_dict[i] = 1- num_trans  / length / 3


t_start = 0.5
t_end = 29
fit_dict = {}
for ti in diffusion_persistence_dict.keys():
    if t_start < ti < t_end:
        fit_dict[ti] = diffusion_persistence_dict[ti]

times = np.log(list(fit_dict.keys()))
prob = np.log(list(fit_dict.values()))

plt.loglog(diffusion_persistence_dict.keys(), diffusion_persistence_dict.values(), 'o')
"""
