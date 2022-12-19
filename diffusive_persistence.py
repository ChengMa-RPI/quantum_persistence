import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
sys.path.insert(1, '/home/mac/RPI/research/')
from mutual_framework import network_generate

import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
import pandas as pd 
from scipy.linalg import inv as spinv
import networkx as nx




class diffusionPersistence:
    def __init__(self, quantum_or_not, network_type, N, d, seed, alpha, t, dt, initial_setup, distribution_params, reference_line):
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
        self.distribution_params = distribution_params
        self.reference_line = reference_line
        self.seed_initial_condition = None

        self.save_read_A_M()  # save A and M to the disk if not yet; otherwise, read them from the disk. 
        self.degree = np.sum(self.A, 0)
        self.N_actual = len(self.A)
        #self.L = -self.A + np.diag(self.degree)

    def save_read_A_M(self):
        save_des = '../data/matrix_save/'
        topology_des = save_des + 'topology/'
        operator_des = save_des + 'quan_operator/'
        for des in [topology_des, operator_des]:
            if not os.path.exists(des):
                os.makedirs(des)

        file_topology = topology_des + f'network_type={self.network_type}_N={self.N}_d={self.d}_seed={self.seed}.npy'
        file_operator = operator_des + f'network_type={self.network_type}_N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}.npy'
        if os.path.exists(file_topology):
            A = np.load(file_topology)
            A_index = np.where(A>0)
            A_interaction = A[A_index]
            index_i = A_index[0] 
            index_j = A_index[1] 
            degree = np.sum(A>0, 1)
            cum_index = np.hstack((0, np.cumsum(degree)))
            self.A, self.A_interaction, self.index_i, self.index_j, self.cum_index = A, A_interaction, index_i, index_j, cum_index
        else:
            self.A, self.A_interaction, self.index_i, self.index_j, self.cum_index = network_generate(network_type, N, 1, 0, seed, d) 
            np.save(file_topology, self.A) 

        if os.path.exists(file_operator):
            self.M = np.load(file_operator)
        else:
            dt = self.dt
            dx = self.alpha
            degree = np.sum(self.A, 0)
            a = -degree + (4j * m * dx **2) /  (hbar * dt)
            b = degree + (4j * m * dx **2) /  (hbar * dt)
            A1 = self.A + np.diag(a)
            B1 = - self.A  + np.diag(b)
            A1_inv = spinv(A1, check_finite=False)
            M = A1_inv.dot(B1)
            self.M = M
            np.save(file_operator, M) 


    def get_initial_condition(self):
        """TODO: Docstring for get_initial_condition.
        :returns: TODO

        """
        seed_initial_condition = self.seed_initial_condition
        t = time.time()
        modifer = int((t - int(t)) * 1e9) 
        np.random.seed(seed_initial_condition + modifer)  # set the seed in the beginning

        initial_setup = self.initial_setup
        N_actual = self.N_actual
        average = 1 / N_actual
        if quantum_or_not == 0:
            if initial_setup == 'uniform_random':
                rho_start, rho_end = self.distribution_params
                #initial_condition = np.random.RandomState(seed_initial_condition).uniform(rho_start, rho_end, size=N_actual)
                initial_condition = np.random.uniform(rho_start, rho_end, size=N_actual)
                initial_condition = initial_condition / np.sum(initial_condition)
        else:
            if initial_setup == 'uniform_random':
                rho_start, rho_end, phase_start, phase_end = self.distribution_params
                if rho_start == rho_end:
                    initial_rho = np.ones(N_actual) * rho_start
                else:
                    #initial_rho = np.random.RandomState(seed_initial_condition).uniform(rho_start, rho_end, size=N_actual)
                    initial_rho = np.random.uniform(rho_start, rho_end, size=N_actual)
                if phase_start == phase_end:
                    initial_phase = np.ones(N_actual) * phase_start * np.pi
                else:
                    #initial_phase = np.random.RandomState(seed_initial_condition).uniform(phase_start * np.pi, phase_end * np.pi, size = N_actual)
                    initial_phase = np.random.uniform(phase_start * np.pi, phase_end * np.pi, size=N_actual)

            elif initial_setup == 'gaussian_wave':
                sigma = 1
                p0 = 1
                x = np.arange(0, 10, 10/self.N)
                x0 = np.round(x.mean(), 5)
                initial_rho = (1/(np.pi**0.25 * sigma**0.5 ) * np.exp(-(x-x0)**2/2/sigma**2)  ) ** 2
                initial_phase =  p0 * x/ hbar
            elif initial_setup == 'sum_sin' or initial_setup == 'sum_sin_inphase':
                # 10 sin function with random amplitude, frequency, and phase 
                amplitude = np.random.RandomState(seed_initial_condition).random(10)
                frequency = np.random.RandomState(seed_initial_condition + 10).exponential(size=10)
                # minimal resolution dx = 0.01
                dx_min = 0.01
                phase = np.random.RandomState(seed_initial_condition + 100).random(int(self.N_actual * self.alpha/ dx_min))
                if initial_setup == 'sum_sin_inphase':
                    phase = 0
                    #frequency = np.random.RandomState(seed_initial_condition + 10).random(size=10) # less up and down
                x = np.arange(0, int(self.N_actual * self.alpha), dx_min)
                y = 0
                for Ai, fi in zip(amplitude, frequency):
                    y += Ai * np.sin(x * fi + phase) 
                # normalize
                rho_positive = y - y.min() + np.abs(np.mean(y) / 10)
                interval = int(self.alpha / dx_min)
                initial_rho = rho_positive[::interval]
                initial_phase = y[::interval]

            else:
                print('Please input initial setup!')
            initial_rho = initial_rho / np.sum(initial_rho)
            initial_A = np.sqrt(initial_rho)
            initial_condition = initial_A * np.exp(1j * initial_phase)
        return initial_condition

    def classic_diffusion(self):
        """TODO: Docstring for classic_diffusion.

        :arg1: TODO

        :arg1: TODO
        :returns: TODO

        """
        initial_condition = self.get_initial_condition()
        #L = self.L
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
        M = self.M
        degree = self.degree
        phi_state = np.zeros((len(t), len(self.A)), dtype=complex)
        phi_state[0] = initial_condition
        """
        a = -degree + (4j * m * dx **2) /  (hbar * dt)
        b = degree + (4j * m * dx **2) /  (hbar * dt)
        A1 = self.A + np.diag(a)
        B1 = - self.A  + np.diag(b)
        A1_inv = spinv(A1, check_finite=False)
        M = A1_inv.dot(B1)
        """
        for i in range(len(t)-1):
            phi_state[i+1] = M.dot(phi_state[i]) 
        phase = np.angle(phi_state) 
        rho = np.abs(phi_state) ** 2
        return rho, phase

    def test_qd_gausswave(self, dx, dt):
        x0 = 50
        x = np.arange(0, 100, dx)
        t = np.arange(0, 500, dt)
        p0 = 1
        sigma = 15
        phi_state = np.zeros((len(t), len(x) ), dtype=complex)
        phi_state[0] = 1/(np.pi**0.25 * sigma**0.5 ) * np.exp(-(x-x0)**2/2/sigma**2) * np.exp(1j  * p0 * x/ hbar  )
        psi_xt = np.zeros((len(t), len(x) ), dtype=complex)
        for i, t0 in enumerate(t):
            psi_xt[i] = 1 / (np.pi ** 0.25 * (sigma * (1 + 1j * hbar * t0 / m / sigma**2))**0.5) * np.exp(- (x - (x0 + p0 * t0/ m))**2 / ( 2*sigma**2*(1+1j*hbar*t0/m/sigma**2) ) ) * np.exp(1j * (p0 * x - p0**2 / 2/ m * t0) / hbar ) 

        a = -2 + (4j * m * dx **2) /  (hbar * dt)
        b = 2 + (4j * m * dx **2) /  (hbar * dt)
        A = nx.to_numpy_array(nx.grid_graph(dim=[len(x)], periodic=True))
        A1 = A + np.identity(len(x)) * a
        B1 = -A  + np.identity(len(x)) * b
        A1_inv = spinv(A1, check_finite=False)
        M = A1_inv.dot(B1)
        for i in range(len(t)-1):
            phi_state[i+1] = M.dot(phi_state[i]) 
        return phi_state, psi_xt

    def test_random_function(self, dx, dt):
        t = np.arange(0, 100, dt)
        A = np.random.RandomState(seed=1).random(10)
        frequency = np.random.RandomState(seed=0).exponential(size=10)
        frequency = np.random.RandomState(seed=0).random(size=10)
        x = np.arange(0, 100, 0.01)
        phase = np.random.RandomState(seed=2).random(10000)
        phase = 0
        y = 0
        for Ai, fi in zip(A, frequency):
            y += Ai * np.sin(x * fi + phase) 
        # normalize
        amplitude = y - y.min() + np.abs(np.mean(y) / 10)

        interval = int(dx / 0.01)
        x = np.arange(0, 100, dx)
        N = len(x)
        select_amplitude = amplitude[::interval]
        initial_rho = select_amplitude/ np.sum(select_amplitude)
        initial_phase = y[::interval]

        phi_state = np.zeros((len(t), len(x)), dtype=complex)
        phi_state[0] = np.sqrt(initial_rho) * np.exp(1j * initial_phase)
        
        a = -2 + (4j * m * dx **2) /  (hbar * dt)
        b = 2 + (4j * m * dx **2) /  (hbar * dt)
        A = nx.to_numpy_array(nx.grid_graph(dim=[len(x)], periodic=False))
        A1 = A + np.identity(len(x)) * a
        B1 = -A  + np.identity(len(x)) * b
        A1_inv = spinv(A1, check_finite=False)
        M = A1_inv.dot(B1)
        for i in range(len(t)-1):
            phi_state[i+1] = M.dot(phi_state[i]) 
        return phi_state

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
            data = self.quantum_diffusion()
            #data = [rho, phase]
        else:
            phi_state = self.classic_diffusion()
            data = [phi_state]
        for file_i, data_i in zip(save_des, data):
            data_save = np.hstack(( self.t.reshape(len(self.t), 1), data_i )) 
            np.save(file_i, data_save) 
        return data
        

    def save_phi_parallel(self, cpu_number, seed_initial_condition_list):
        if self.quantum_or_not:
            des_state = '../data/quantum/state/' + self.network_type + '/' 
            des_phase = '../data/quantum/phase/' + self.network_type + '/' 
            des_list = [des_state, des_phase]
        else:
            des = '../data/classical/state/' + self.network_type + '/' 
            des_list = [des]
        save_file = []
        for des in des_list:
            if not os.path.exists(des):
                os.makedirs(des)
            save_file.append(des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_seed_initial=' )

        p = mp.Pool(cpu_number)
        p.starmap_async(self.get_phi_state,  [(seed_initial_condition, [file_i + f'{seed_initial_condition}' for file_i in save_file]) for seed_initial_condition in seed_initial_condition_list]).get()
        p.close()
        p.join()
        return None
    

    

cpu_number = 4


m = 5.68
hbar = 0.6582


if __name__ == '__main__':
    quantum_or_not = False
    quantum_or_not = True
    network_type = '2D_disorder'
    N = 100
    alpha = 1
    dt = 1
    t = np.arange(0, 1000, dt)

    initial_setup = 'uniform_random'
    reference_line = 'average'
    reference_line = 0.5
    reference_lines = ['average']
    seed_initial_condition_list = np.arange(10)
    initial_setup = 'rho_uniform_phase_uniform'
    initial_setup = 'uniform_random'
    rho_list = [[0, 1], [1/4, 3/4], [3/8, 5/8], [1, 1]]
    phase_list = [[-1, 1], [-1/2, 1/2], [-1/4, 1/4], [0, 0]]
    rho_list = [[1, 1]]
    phase_list = [[-1, 1], [-1/2, 1/2], [-1/4, 1/4]]
    distribution_params_raw = [rho + phase for rho in rho_list for phase in phase_list]
    #distribution_params_raw = [rho for rho in rho_list]

    distribution_params_list = []
    for i in distribution_params_raw:
        distribution_params_list.append( [round(j, 3) for j in i])


    N_list = [100]
    alpha_list = [1]
    dt_list = [0.1]
    num_realization_list = [10]
    
    d_list = [0.6, 0.7, 0.8, 0.9]
    seed = 0
    for d in d_list:
        for N, alpha, dt, num_realization in zip(N_list, alpha_list, dt_list, num_realization_list):
            seed_initial_condition_list = np.arange(num_realization)
            t = np.arange(0, 10000*dt, dt)
            for distribution_params in distribution_params_list:
                t1 = time.time()
                dp = diffusionPersistence(quantum_or_not, network_type, N, d, seed, alpha, t, dt, initial_setup, distribution_params, reference_line)
                dp.save_phi_parallel(cpu_number, seed_initial_condition_list)
                t2 = time.time()
                print(t2 - t1)



    """
    ### test gaussian wave evolution
    dx_list = [5, 1, 1, 0.5, 0.1, 0.1]
    dt_list = [5, 1, 0.1, 0.1, 0.1, 0.01]
    phi_state_list = []
    for dx, dt in zip(dx_list, dt_list):
        phi_state, psi_xt = dp.test_qd_gausswave(dx, dt)
        phi_state_list.append(phi_state)
    t0 = 0
    linestyles = ['-', '-', '--', '-', '-', '--']
    for (i, dx), dt, phi_state in zip(enumerate(dx_list), dt_list, phi_state_list):
        x = np.arange(0, 100, dx)
        index = int(t0/dt)
        plt.plot(x, np.abs(phi_state[index])**2, label=f'dx={dx} dt={dt}', linestyle = linestyles[i])
    plt.plot(x, np.abs(psi_xt[index])**2, label='theory', linestyle='-.')
    plt.xlabel('$x$', fontsize=17)
    plt.ylabel('$\\rho$', fontsize=17)
    plt.legend()
    plt.title(f't={t0}')
    plt.show()
    """

    """
    ### generate a random continuous function
    dx_list = [5, 1, 1, 0.5, 0.1, 0.1]
    dt_list = [5, 1, 0.1, 0.1, 0.1, 0.01]
    phi_state_list = []
    for dx, dt in zip(dx_list, dt_list):
        phi_state = dp.test_random_function(dx, dt)
        phi_state_list.append(phi_state)
    t0 = 10
    linestyles = ['-', '-', '--', '-', '-', '--']
    for (i, dx), dt, phi_state in zip(enumerate(dx_list), dt_list, phi_state_list):
        x = np.arange(0, 100, dx)
        index = int(t0/dt)
        plt.plot(x, np.abs(phi_state[index])**2/ dx, label=f'dx={dx} dt={dt}', linestyle = linestyles[i])
    plt.xlabel('$x$', fontsize=17)
    plt.ylabel('$\\rho$', fontsize=17)
    plt.legend()
    plt.title(f't={t0}')
    plt.show()

    """
