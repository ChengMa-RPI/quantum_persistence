import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'
import sys
sys.path.insert(1, '/home/mac/RPI/research/')
from mutual_framework import network_generate

import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import multiprocessing as mp
import pandas as pd 
from scipy.linalg import inv as spinv
import networkx as nx




class diffusionPersistence:
    def __init__(self, quantum_or_not, network_type, N, d, seed, alpha, t, dt, initial_setup, reference_line):
        """TODO: Docstring for __init__.

        :quantum_not: TODO
        :network_type: TODO
        :N: TODO
        :d: TODO
        :seed: TODO
        :alpha: diffusion ratio
        :: TODO
        :returns: TODO

        """
        self.quantum_or_not = quantum_or_not
        self.network_type = network_type
        self.N = N
        self.d = d
        self.seed = seed
        self.alpha = alpha
        self.t = t
        self.dt = dt
        self.initial_setup = initial_setup
        self.reference_line = reference_line
        self.seed_initial_condition = None

        self.save_A_M()
        self.degree = np.sum(self.A, 0)
        self.N_actual = len(self.A)
        self.L = -self.A + np.diag(self.degree)

    def save_A_M(self):
        save_des = '../data/matrix_save/'
        topology_des = save_des + 'topology/'
        operator_des = save_des + 'quan_operator/'
        for des in [topology_des, operator_des]:
            if not os.path.exists(des):
                os.makedirs(des)

        file_topology = topology_des + 'network_type={self.network_type}_N={self.N}_d={self.d}_seed={self.seed}.npy'
        file_operator = operator_des + 'network_type={self.network_type}_N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}.npy'
        if os.path.exists(file_topology):
            pass

        else:
            self.A, self.A_interaction, self.index_i, self.index_j, self.cum_index = network_generate(network_type, N, 1, 0, seed, d) 

        if os.path.exists(file_operator):
            pass

        else:
            t = self.t
            dt = self.dt
            dx = self.alpha
            degree = np.sum(self.A, 0)
            a = -degree + (4j * m * dx **2) /  (hbar * dt)
            b = degree + (4j * m * dx **2) /  (hbar * dt)
            A1 = self.A + np.diag(a)
            B1 = - self.A  + np.diag(b)
            A1_inv = spinv(A1, check_finite=False)
            M = A1_inv.dot(B1)
            np.save(file_operator, M) 



    def get_initial_condition(self):
        """TODO: Docstring for get_initial_condition.
        :returns: TODO

        """
        seed_initial_condition = self.seed_initial_condition
        initial_setup = self.initial_setup
        average = 1 / self.N_actual
        if quantum_or_not == 0:
            if initial_setup == 'uniform_random':
                initial_condition = np.random.RandomState(seed_initial_condition).uniform(0, 1, size = self.N_actual)
                initial_condition = initial_condition / np.sum(initial_condition)
        else:
            if initial_setup == 'rho_uniform_phase_uniform':
                initial_rho = np.random.RandomState(seed_initial_condition).uniform(0, 1, size=self.N_actual)
                initial_phase = np.random.RandomState(seed_initial_condition).uniform(0, 2 * np.pi, size = self.N_actual)
            elif initial_setup == 'rho_const_phase_uniform':
                initial_phase = np.random.RandomState(seed_initial_condition).uniform(0, 2 * np.pi, size = self.N_actual)
                initial_rho = np.ones(self.N_actual) 
            elif initial_setup == 'rho_uniform_phase_const_pi':
                initial_rho = np.random.RandomState(seed_initial_condition).uniform(0, 1, size=self.N_actual)
                initial_phase = np.pi
            elif initial_setup == 'rho_uniform_phase_const_pi_half':
                initial_rho = np.random.RandomState(seed_initial_condition).uniform(0, 1, size=self.N_actual)
                initial_phase = np.pi/2
            elif initial_setup == 'rho_uniform_phase_const_pi_quater':
                initial_rho = np.random.RandomState(seed_initial_condition).uniform(0, 1, size=self.N_actual)
                initial_phase = np.pi/4
            elif initial_setup == 'gaussian_wave':
                sigma = 1
                p0 = 1
                x = np.arange(0, 10, 1/self.N)
                x0 = np.round(x.mean(), 5)
                initial_rho = (1/(np.pi**0.25 * sigma**0.5 ) * np.exp(-(x-x0)**2/2/sigma**2)  ) ** 2
                initial_phase =  p0 * x/ hbar
            else:
                print('Please input initial setup!')
            initial_rho = initial_rho / np.sum(initial_rho)
            initial_A = np.sqrt( initial_rho)
            initial_condition = initial_A * np.exp(1j * initial_phase)
        return initial_condition

    def classic_diffusion(self):
        """TODO: Docstring for classic_diffusion.

        :arg1: TODO

        :arg1: TODO
        :returns: TODO

        """
        initial_condition = self.get_initial_condition()
        L = self.L
        t = self.t
        dt = self.dt
        alpha = self.alpha
        A_interaction, index_j, cum_index, degree = self.A_interaction, self.index_j, self.cum_index, self.degree
        phi_state = np.zeros((len(t), len(self.A)))
        phi_state[0] = initial_condition
        #M = (-L * alpha * dt)   # very slow for large sparse matrix
        for i in range(len(t)-1):
            phi_state[i+1] = phi_state[i] * (1 - degree * alpha * dt) + np.add.reduceat(A_interaction * phi_state[i][index_j], cum_index[:-1]) * alpha * dt
            #phi_state[i+1] = M.dot(phi_state[i]) + phi_state[i]
        return phi_state

    def quantum_diffusion(self):
        "For large network, the matrix M should be saved"
        initial_condition = self.get_initial_condition()
        t = self.t
        dt = self.dt
        dx = self.alpha
        degree = self.degree
        phi_state = np.zeros((len(t), len(self.A)), dtype=complex)
        phi_state[0] = initial_condition
        a = -degree + (4j * m * dx **2) /  (hbar * dt)
        b = degree + (4j * m * dx **2) /  (hbar * dt)
        A1 = self.A + np.diag(a)
        B1 = - self.A  + np.diag(b)
        A1_inv = spinv(A1, check_finite=False)
        M = A1_inv.dot(B1)
        for i in range(len(t)-1):
            phi_state[i+1] = M.dot(phi_state[i]) 
        rho = np.abs(phi_state) ** 2
        return rho

    def test_qd_gausswave(self, dx, dt):
        x0 = 50
        x = np.arange(0, 100, dx)
        t = np.arange(0, int(5000*dt), dt)
        p0 = 0
        sigma = 10
        phi_state = np.zeros((len(t), len(x) ), dtype=complex)
        phi_state[0] = 1/(np.pi**0.25 * sigma**0.5 ) * np.exp(-(x-x0)**2/2/sigma**2) * np.exp(1j  * p0 * x/ hbar  )
        psi_xt = np.zeros((len(t), len(x) ), dtype=complex)
        for i, t0 in enumerate(t):
            psi_xt[i] = 1 / (np.pi ** 0.25 * (sigma * (1 + 1j * hbar * t0 / m / sigma**2))**0.5) * np.exp(- (x - (x0 + p0 * t0/ m))**2 / ( 2*sigma**2*(1+1j*hbar*t0/m/sigma**2) ) ) * np.exp(1j * (p0 * x - p0**2 / 2/ m * t0) / hbar ) 

        a = -2 + (4j * m * dx **2) /  (hbar * dt)
        b = 2 + (4j * m * dx **2) /  (hbar * dt)
        A = nx.to_numpy_array(nx.grid_graph(dim=[len(x)], periodic=False))
        A1 = A + np.identity(len(x)) * a
        B1 = -A  + np.identity(len(x)) * b
        A1_inv = spinv(A1, check_finite=False)
        M = A1_inv.dot(B1)
        for i in range(len(t)-1):
            phi_state[i+1] = M.dot(phi_state[i]) 
        return phi_state, psi_xt

    def test_qd_dtdx(self, dt_list):
        self.seed_initial_condition = 0
        rho_list = []
        for dt in dt_list:
            self.dt = dt
            self.t = np.arange(0, 100, dt)
            rho = self.quantum_diffusion()
            rho_list.append(rho)
        return rho_list


    def get_phi_state(self, seed_initial_condition, save_des=None):
        self.seed_initial_condition = seed_initial_condition
        if quantum_or_not:
            phi_state = self.quantum_diffusion()
        else:
            phi_state = self.classic_diffusion()
        if save_des:
            data = np.hstack(( self.t.reshape(len(self.t), 1), phi_state )) 
            np.save(save_des, data) 
        return phi_state
        
    def diffusive_persistence_prob(self, seed_initial_condition):
        t, dt = self.t, self.dt
        phi_state = self.get_phi_state(seed_initial_condition)
        reference_value = 1/ self.N_actual
        if reference_line != 'average':
            reference_value *= float(reference_line) 
        mask_above = np.heaviside(phi_state - reference_value, 0)
        mask_below = mask_above == 0
        first_above = np.where(mask_above.any(axis=0), mask_above.argmax(axis=0), -1) * dt
        first_below = np.where(mask_below.any(axis=0), mask_below.argmax(axis=0), -1) * dt
        always_above = np.sum(first_below < 0)
        always_below = np.sum(first_above < 0)
        na = np.zeros(( len(t) ))
        nb = np.zeros(( len(t) ))
        for i, t_i in enumerate(t):
            na[i] = np.sum(first_below > t_i)
            nb[i] = np.sum(first_above > t_i)
        na += always_above
        nb += always_below
        pa = na / self.N_actual
        pb = nb / self.N_actual
        return pa, pb

    def save_dpp(self, seed_initial_condition, des):
        """TODO: Docstring for save_dpp.

        :arg1: TODO
        :returns: TODO

        """
        pa, pb = self.diffusive_persistence_prob(seed_initial_condition)
        des_file = des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_seed_initial={seed_initial_condition}_setup={self.initial_setup}_reference={self.reference_line}.csv'
        df = pd.DataFrame(np.vstack(( self.t, pa, pb )).transpose()) 
        df.to_csv(des_file, index=None, header=None)

    def get_dpp_parallel(self, cpu_number, seed_initial_condition_list):
        if self.quantum_or_not:
            des = '../data/quantum/' + self.network_type + '/' 
        else:
            des = '../data/classical/' + self.network_type + '/' 

        if not os.path.exists(des):
            os.makedirs(des)
        p = mp.Pool(cpu_number)
        p.starmap_async(self.save_dpp,  [(seed_initial_condition, des) for seed_initial_condition in seed_initial_condition_list]).get()
        p.close()
        p.join()
        return None

    def save_phi_parallel(self, cpu_number, seed_initial_condition_list):
        if self.quantum_or_not:
            des = '../data/quantum/state/' + self.network_type + '/' 
        else:
            des = '../data/classical/state/' + self.network_type + '/' 
        if not os.path.exists(des):
            os.makedirs(des)
        save_file = des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_setup={self.initial_setup}_seed_initial='
        p = mp.Pool(cpu_number)
        p.starmap_async(self.get_phi_state,  [(seed_initial_condition, save_file + f'{seed_initial_condition}') for seed_initial_condition in seed_initial_condition_list]).get()
        p.close()
        p.join()
        return None
    

    

cpu_number = 4


m = 5.68
hbar = 0.6582


if __name__ == '__main__':
    quantum_or_not = True
    network_type = '1D'
    N = 100**2
    d = 4
    seed = 0
    alpha = 1
    dt = 0.1
    seed = 0
    t = np.arange(0, 1000, dt)

    initial_setup = 'uniform_random'
    reference_line = 'average'
    reference_line = 0.5
    reference_lines = ['average']
    seed_initial_condition_list = np.arange(10)
    L_list = np.arange(10, 40, 10)
    N_list = np.power(L_list, 2)
    N_list = np.arange(100, 200, 200)
    initial_setup = 'rho_uniform_phase_uniform'
    for reference_line in reference_lines:
        for N in N_list:
            dp = diffusionPersistence(quantum_or_not, network_type, N, d, seed, alpha, t, dt, initial_setup, reference_line)
            #dp.get_dpp_parallel(cpu_number, seed_initial_condition_list)
            #dp.save_phi_parallel(cpu_number, seed_initial_condition_list)
            pass

