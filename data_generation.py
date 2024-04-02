import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
from helper_function import network_generate
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
import pandas as pd 
from scipy.linalg import inv as spinv
import networkx as nx
from scipy.stats import chi2 
import scipy.stats as stats
from scipy.fft import fft, fftn




class diffusionPersistence:
    def __init__(self, quantum_or_not, network_type, m, N, d, seed, alpha, t, dt, initial_setup, distribution_params):
        """TODO: Class for implementing time-dependent simulation for diffusion systems.

        :quantum_not: quantum or classical, quantum if True, else classical
        :network_type: network topology, regular lattice: '1D', '2D', '3D'; disordered lattice: '2D_disorder', '3D_disorder'
        :m: mass of particle
        :N: the number of grids in the system
        :d: the parameter for network generation, d = 4 for regular lattice, and d = 1 - edge_removal_rate for disordered system
        :seed: the random seed to generate network, no impact for regular lattice
        :alpha: delta x for quantum, and diffusion ratio for classical 
        :t: the timestamps to simulate
        :dt: the timestamp interval
        :initial_setup: the initial condition 
        :distribution_params: parameters for the initial condition generation function
        :returns: 

        """
        self.quantum_or_not = quantum_or_not
        self.network_type = network_type
        self.m = m
        self.N = N
        self.d = d
        self.seed = seed
        self.alpha = alpha
        self.t = t
        self.dt = dt
        self.initial_setup = initial_setup
        self.distribution_params = distribution_params
        self.seed_initial_condition = None  # set as None as parameters will pass to multiprocessing task

        self.save_read_A_M()  # save matrices A and M to the disk if not yet; otherwise, read them from the disk. 
        self.degree = np.sum(self.A, 0)  # an array of degrees for each node
        self.N_actual = len(self.A)  # the number of actual nodes, N = N_actual for regular lattice

    def save_read_A_M(self):
        """TODO: generate adjacency matrix A and operator matrix M and save it to disk if not 
        -----\psi(t+dt) = M \psi(t)

        """

        # define file name and destination to save
        save_des = '../data/matrix_save/'
        topology_des = save_des + 'topology/'
        operator_des = save_des + 'quan_operator/'
        for des in [topology_des, operator_des]:
            if not os.path.exists(des):
                os.makedirs(des)
        file_topology = topology_des + f'network_type={self.network_type}_N={self.N}_d={self.d}_seed={self.seed}.npy'
        if self.m == m_e:
            file_operator = operator_des + f'network_type={self.network_type}_N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}.npy'
        else:
            file_operator = operator_des + f'network_type={self.network_type}_m={self.m}_N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}.npy'

        if os.path.exists(file_topology):
            # read A if it has already been saved
            A = np.load(file_topology)
            A_index = np.where(A>0)
            A_interaction = A[A_index]
            index_i = A_index[0] 
            index_j = A_index[1] 
            degree = np.sum(A>0, 1)
            cum_index = np.hstack((0, np.cumsum(degree)))
            self.A, self.A_interaction, self.index_i, self.index_j, self.cum_index = A, A_interaction, index_i, index_j, cum_index
        else:
            # generate A
            self.A, self.A_interaction, self.index_i, self.index_j, self.cum_index = network_generate(network_type, N, 1, 0, seed, d) 
            np.save(file_topology, self.A) 

        if os.path.exists(file_operator):
            self.M = np.load(file_operator)
        else:
            # calculate M and save it
            m = self.m
            dt = self.dt
            if type(self.alpha) == int:
                dx = self.alpha
            else:
                dx = self.alpha[0]
            degree = np.sum(self.A, 0)
            a = -degree + (4j * m * dx **2) /  (hbar * dt)
            b = degree + (4j * m * dx **2) /  (hbar * dt)
            A1 = self.A + np.diag(a)
            B1 = - self.A  + np.diag(b)
            A1_inv = spinv(A1, check_finite=False)
            M = A1_inv.dot(B1)
            self.M = M
            np.save(file_operator, M) 

    def cutoff_limit(self, N, random_data, lambda_ratio):
        """TODO: high frequency mode cutoff
        :N: the number of nodes
        :random_data: data (phase or amplitude) to be processed
        :lambda_ratio: 1/a, a is the length scale, or the percentage of low frequency modes been preserved 
        :return: data after cutoff processing
        """
        
        if self.network_type == '1D':
            fft_k = fft(random_data)
            index = int(N * lambda_ratio / 2) + 1
            pos_n_select = np.arange(1, index, 1)
            k_select = 2 * np.pi / N * pos_n_select
            x = np.arange(0, N, 1)
            exponential = np.exp(1j * k_select * x.reshape(len(x), 1))
            data_select = np.sum( np.real(fft_k[1:index] * exponential), axis=1) * 2 / N
            data_select /= np.sqrt(lambda_ratio)
        elif self.network_type == '2D':
            N_x = round(N ** (1/2))
            index = int(N_x * lambda_ratio // 2)
            nx_pos = np.arange(0, N_x//2, 1)
            nx_neg = np.arange(-N_x//2, 0, 1)
            nx = np.hstack(( nx_pos, nx_neg ))
            invalid_index = np.hstack(([0], np.where((nx > index) | (nx < -index))[0]))
            kx = 2 * np.pi * nx / N_x
            data_2D = random_data.reshape(N_x, N_x)
            fft_k = fftn(data_2D)
            fft_k[invalid_index] = 0
            fft_k[:, invalid_index] = 0
            data_select = np.zeros((N_x, N_x))
            ### can be optimized, but may take some time to figure out the multi-dimension multiplication.
            for i, x in enumerate(range(N_x)):
                for j, y in enumerate(range(N_x)):
                    kr_prod = (kx * x).reshape(len(kx), 1) + kx * y
                    u_fft = np.real(np.sum(fft_k * np.exp(1j * kr_prod))) / N_x ** 2
                    data_select[i, j] = u_fft
            data_select = data_select.ravel()
            data_select /= lambda_ratio
        elif self.network_type == '3D':
            N_x = round(N ** (1/3))
            index = int(N_x * lambda_ratio // 2)
            nx_pos = np.arange(0, N_x//2, 1)
            nx_neg = np.arange(-N_x//2, 0, 1)
            nx = np.hstack(( nx_pos, nx_neg ))
            invalid_index = np.hstack(([0], np.where((nx > index) | (nx < -index))[0]))
            kx = 2 * np.pi * nx / N_x
            data_3D = random_data.reshape(N_x, N_x, N_x)
            fft_k = fftn(data_3D)
            fft_k[invalid_index] = 0
            fft_k[:, invalid_index, :] = 0
            fft_k[:, :, invalid_index] = 0
            data_select = np.zeros((N_x, N_x, N_x))
            ### can be optimized, but may take some time to figure out the multi-dimension multiplication.
            for i, x in enumerate(range(N_x)):
                for j, y in enumerate(range(N_x)):
                    for l, z in enumerate(range(N_x)):
                        kr_prod = (kx * x).reshape(len(kx), 1, 1) + (kx * y).reshape(1, len(kx), 1) + (kx * z).reshape(1, 1, len(kx))
                        u_fft = np.real(np.sum(fft_k * np.exp(1j * kr_prod))) / N_x ** 3
                        data_select[i, j, l] = u_fft
            data_select = data_select.ravel()
            data_select /= np.sqrt(lambda_ratio) ** 3
        return data_select

    def get_initial_condition(self):
        """TODO: implementations for different initial conditions
        :returns: 

        """
        seed_initial_condition = self.seed_initial_condition  # seed to control random generator for initial condition
        modifier = 0  # no modifier, to reproduce the results!!!
        np.random.seed(seed_initial_condition + modifier)  # set the seed in the beginning

        initial_setup = self.initial_setup
        N_actual = self.N_actual
        average = 1 / N_actual
        if not quantum_or_not:
            # for classical
            if initial_setup == 'uniform_random':
                rho_range = self.distribution_params
                initial_condition = np.random.uniform(-rho_range, rho_range, size=N_actual)
            elif initial_setup == 'normal_random':
                rho_std = self.distribution_params
                initial_condition = np.random.normal(0, rho_std, size=N_actual)
            elif initial_setup == 'gaussian_wave':
                sigma = self.distribution_params
                x = np.arange(0, int(self.N  * self.alpha[0]), self.alpha[0])
                x0 = np.round(x.mean(), 5)
                initial_condition = (1/(np.pi**0.25 * sigma**0.5) * np.exp(-(x-x0)**2/2/sigma**2)  ) ** 2


        else:
            # for quantum
            "u is relative size of the amplitude fluctuations (compared to r0), see paper"
            if initial_setup == 'u_uniform_random':
                # uniform random for u and theta
                u_std, phase_std = self.distribution_params  # standard deviation for u and phase
                u_std = u_std / np.sqrt(1-u_std**2)
                r0 = 1/np.sqrt(N_actual)  
                u_std *= r0
                u_start, u_end = -u_std * np.sqrt(3), u_std * np.sqrt(3)
                phase_start, phase_end = -phase_std * np.sqrt(3), phase_std * np.sqrt(3)
                initial_u = np.random.uniform(u_start, u_end, size=N_actual)
                initial_phase = np.random.uniform(phase_start, phase_end, size=N_actual)
                initial_rho = (initial_u + r0) ** 2
                
            elif initial_setup == 'u_uniform_random_cutoff':
                # uniform random for u and theta with high frequency mode cutoffs
                u_std, u_cutoff, phase_std, phase_cutoff = self.distribution_params
                r0 = 1/np.sqrt(N_actual)

                u_start, u_end = -u_std * np.sqrt(3), u_std * np.sqrt(3)
                phase_start, phase_end = -phase_std * np.sqrt(3), phase_std * np.sqrt(3)
                initial_u = np.random.uniform(u_start, u_end, size=N_actual)
                initial_phase = np.random.uniform(phase_start, phase_end, size=N_actual)

                u_select = self.cutoff_limit(N, initial_u, u_cutoff)
                u_select = u_select / np.sqrt(1-u_std**2)
                u_norm = (u_select + 1) / np.sqrt(np.sum(( u_select+1) **2)) - r0
                initial_rho = (u_norm + r0) ** 2

                phase_select = self.cutoff_limit(N, initial_phase, phase_cutoff)
                initial_phase = phase_select 


            elif initial_setup in ['u_normal_random', 'u_normal_phase_uniform_random']:
                u_std, phase_std = self.distribution_params
                u_std = u_std / np.sqrt(1-u_std**2)
                r0 = 1/np.sqrt(N_actual)
                u_std *= r0
                u_lower, u_upper = -r0, r0
                if u_std == 0:
                    u_a, u_b = u_lower/0.1, u_upper / 0.1
                else:
                    u_a, u_b = u_lower / u_std, u_upper / u_std
                initial_u = stats.truncnorm(u_a, u_b, loc=0, scale=u_std).rvs(N_actual)
                initial_rho = (initial_u + r0) ** 2
                phase_lower, phase_upper = -np.pi, np.pi
                if initial_setup == 'u_normal_random':
                    # (Truncated) Gaussian random for u and phase, truncated to avoid invalid u (<0) or phase (outside of [-pi, pi])
                    if phase_std == 0:
                        phase_a, phase_b = phase_lower / 0.1, phase_upper / 0.1
                    else:
                        phase_a, phase_b = phase_lower / phase_std, phase_upper / phase_std
                    initial_phase = stats.truncnorm(phase_a, phase_b, loc=0, scale=phase_std).rvs(N_actual)
                else:
                    # (Truncated) Gaussian random fro u and uniform random for phase
                    initial_phase = np.random.uniform(-phase_std * np.pi, phase_std * np.pi, size=N_actual)  # please note phase_std is not real standard dev, it is actually the range! 
            elif initial_setup == 'u_normal_random_cutoff':
                # (Truncated) Gaussian random with cutoffs
                u_std, u_cutoff, phase_std, phase_cutoff = self.distribution_params
                r0 = 1/np.sqrt(N_actual)
                u_std *= r0

                u_lower, u_upper = -r0, r0
                if u_std == 0:
                    u_a, u_b = u_lower/0.1, u_upper / 0.1
                else:
                    u_a, u_b = u_lower / u_std, u_upper / u_std
                initial_u = stats.truncnorm(u_a, u_b, loc=0, scale=u_std).rvs(N_actual)
                phase_lower, phase_upper = -np.pi, np.pi
                if phase_std == 0:
                    phase_a, phase_b = phase_lower / 0.1, phase_upper / 0.1
                else:
                    phase_a, phase_b = phase_lower / phase_std, phase_upper / phase_std
                initial_phase = stats.truncnorm(phase_a, phase_b, loc=0, scale=phase_std).rvs(N_actual)

                u_select = self.cutoff_limit(N, initial_u, u_cutoff)
                u_select = u_select / np.sqrt(1-u_std**2)  # offset the selected k mode for u_cutoff
                u_norm = (u_select + r0) / np.sqrt(np.sum(( u_select+r0) **2)) - r0
                initial_rho = (u_norm + r0) ** 2

                phase_select = self.cutoff_limit(N, initial_phase, phase_cutoff)
                initial_phase = phase_select 


            elif initial_setup == 'uniform_random':
                # uniform random for amplitude and phase
                rho_start, rho_end, phase_start, phase_end = self.distribution_params
                if rho_start == rho_end:
                    initial_rho = np.ones(N_actual) * rho_start
                else:
                    initial_rho = np.random.uniform(rho_start, rho_end, size=N_actual)
                if phase_start == phase_end:
                    initial_phase = np.ones(N_actual) * phase_start * np.pi
                else:
                    initial_phase = np.random.uniform(phase_start * np.pi, phase_end * np.pi, size=N_actual)

            elif initial_setup == 'chi2_uniform':
                # seems no longer under experiments, chi square distribution for u and uniform for phase
                df, phase_start, phase_end = self.distribution_params
                initial_rho = chi2.rvs(df, size=N_actual)
                if phase_start == phase_end:
                    initial_phase = np.ones(N_actual) * phase_start * np.pi
                else:
                    initial_phase = np.random.uniform(phase_start * np.pi, phase_end * np.pi, size=N_actual)

            elif initial_setup == 'full_local':
                # start from fully localized state, rho is nonzero only for a single node
                _, phase_start, phase_end = self.distribution_params
                initial_rho = np.zeros(N_actual)
                if self.network_type == '1D':
                    initial_rho[N_actual//2] = 1.0
                elif self.network_type in ['2D', '2D_disorder', '3D', '3D_disorder']:
                    initial_rho[int(N_actual//2 + np.sqrt(N_actual) // 2)] = 1.0
                if phase_start == phase_end:
                    initial_phase = np.ones(N_actual) * phase_start * np.pi
                else:
                    initial_phase = np.random.uniform(phase_start * np.pi, phase_end * np.pi, size=N_actual)

            elif initial_setup == 'phase_multi_locals':
                # phase localized, equal density at all sites, but with multiple nodes with nonzero phase
                u_std, phase_local_num, phase_local_value = self.distribution_params
                u_std = u_std / np.sqrt(1-u_std**2)
                r0 = 1/np.sqrt(N_actual)
                u_std *= r0
                u_lower, u_upper = -r0, r0
                if u_std == 0:
                    u_a, u_b = u_lower/0.1, u_upper / 0.1
                else:
                    u_a, u_b = u_lower / u_std, u_upper / u_std
                initial_u = stats.truncnorm(u_a, u_b, loc=0, scale=u_std).rvs(N_actual)
                initial_rho = (initial_u + r0) ** 2
                initial_phase = np.zeros(N_actual)
                if phase_local_num == 1:
                    if self.network_type == '1D':
                        initial_phase[N_actual//2] = phase_local_value * np.pi
                    elif self.network_type in ['2D', '2D_disorder', '3D', '3D_disorder']:
                        initial_phase[int(N_actual//2 + np.sqrt(N_actual) // 2)] = phase_local_value * np.pi

                elif phase_local_num == 2: 
                    if self.network_type == '1D':
                        initial_phase[N_actual//2] = phase_local_value * np.pi
                        initial_phase[N_actual//2+1] = -phase_local_value * np.pi
                    elif self.network_type in ['2D', '2D_disorder', '3D', '3D_disorder']:
                        initial_phase[int(N_actual//2 + np.sqrt(N_actual) // 2)] = phase_local_value * np.pi
                        initial_phase[int(N_actual//2 + np.sqrt(N_actual) // 2)+1] = -phase_local_value * np.pi
                else:
                    # under experiments! For now, multiple local sites are randomly selected
                    local_site = np.random.choice(N_actual, phase_local_num, replace=False)
                    initial_phase[local_site] = phase_local_value * np.pi


            elif initial_setup == 'gaussian_wave':
                # start from Gaussian wave packet
                sigma, p0 = self.distribution_params
                x = np.arange(0, int(self.N  * self.alpha), self.alpha)
                x0 = np.round(x.mean(), 5)
                initial_rho = (1/(np.pi**0.25 * sigma**0.5) * np.exp(-(x-x0)**2/2/sigma**2)  ) ** 2
                initial_phase =  p0 * x/ hbar

            elif initial_setup == 'sum_sin' or initial_setup == 'sum_sin_inphase':
                # seems no longer under experiments, k_mode sin function with random amplitude, frequency, and phase 
                k_mode, ampl_strength, freq_strength = self.distribution_params
                amplitude = (np.random.RandomState(seed_initial_condition).random(k_mode) * 0.5 + 0.5) * ampl_strength
                #frequency = np.random.RandomState(seed_initial_condition + 10).exponential(size=k_mode)  # use integer, not float
                frequency = np.random.RandomState(seed_initial_condition + 10).randint(low=int(self.N_actual/20 * freq_strength), high=int(self.N_actual/2 * freq_strength), size=k_mode) * 2 * np.pi / self.N_actual
                phase = np.random.RandomState(seed_initial_condition + 100).random(self.N_actual)
                if initial_setup == 'sum_sin_inphase':
                    phase = 0
                    #frequency = np.random.RandomState(seed_initial_condition + 10).random(size=10) # less up and down
                x = np.arange(0, self.N_actual, 1)
                y = 0
                for Ai, fi in zip(amplitude, frequency):
                    y += Ai * np.sin(x * fi + phase) 
                # normalize
                initial_rho = (y + 1/np.sqrt(self.N)) ** 2

                initial_phase = y

            else:
                print('Please input initial setup!')
            # normalizations for rho  
            initial_rho = initial_rho / np.sum(initial_rho)
            initial_A = np.sqrt(initial_rho)
            initial_condition = initial_A * np.exp(1j * initial_phase)

        return initial_condition

    def classic_diffusion(self):
        """TODO: implementation of classical diffusion

        :returns: state, \phi 

        """
        initial_condition = self.get_initial_condition()
        t = self.t
        dt = self.dt
        alpha, diff_alpha = self.alpha
        A_interaction, index_j, cum_index, degree = self.A_interaction, self.index_j, self.cum_index, self.degree
        phi_state = np.zeros((len(t), len(self.A)))
        phi_state[0] = initial_condition
        for i in range(len(t)-1):
            phi_state[i+1] = phi_state[i] * (1 - degree * dt / alpha ** 2 * diff_alpha) + np.add.reduceat(A_interaction * phi_state[i][index_j], cum_index[:-1]) * dt / alpha ** 2  * diff_alpha

        return phi_state

    def quantum_diffusion(self):
        """TODO: implementation of Crank-Nicholson method for quantum diffusion

        :returns: state, rho and phase 

        """
        initial_condition = self.get_initial_condition()
        t = self.t
        dt = self.dt
        dx = self.alpha
        M = self.M
        degree = self.degree
        phi_state = np.zeros((len(t), len(self.A)), dtype=complex)
        phi_state[0] = initial_condition
        for i in range(len(t)-1):
            phi_state[i+1] = M.dot(phi_state[i]) 
        phase = np.angle(phi_state) 
        rho = np.abs(phi_state) ** 2
        return rho, phase

    def eigen_fourier(self):
        """TODO: implementation of continuum limit Fourier transformation method for quantum dynamics

        :returns: state, rho and phase 

        """
        t_list = self.t 
        psi_0 = self.get_initial_condition()
        if self.network_type == '1D':
            L = self.alpha * self.N
            grid_list = np.arange(0, self.N, 1)
            k_list = 2 * np.pi  * grid_list / self.N / self.alpha
            psi_k = np.sum(np.exp(-1j * k_list.reshape(len(k_list), 1) * grid_list * self.alpha) * psi_0, axis=1)
            E_k = hbar ** 2 * (1 - np.cos(k_list * self.alpha)) / m / self.alpha ** 2
            x_list = grid_list * self.alpha
            psi_xt = 1 / L  * psi_k * np.exp(-1j * E_k.reshape(len(k_list), 1) * t_list / hbar).transpose()@(np.exp(1j * k_list.reshape(len(k_list), 1) * x_list))
        elif self.network_type == '2D':
            ### more challenging to deal with 2D matrices, double summation.
            N_x = int(round(self.N ** (1/2)))
            L_x = self.alpha * N_x
            grid_x = np.tile(np.arange(0, N_x, 1), (N_x))
            grid_y = np.tile(np.arange(0, N_x, 1), (N_x, 1)).transpose().ravel()
            kx_list = 2 * np.pi  * grid_x / N_x / self.alpha
            ky_list = 2 * np.pi  * grid_y / N_x / self.alpha
            psi_k = np.sum(np.exp(-1j * (kx_list.reshape(len(kx_list), 1) * grid_x + ky_list.reshape(len(ky_list), 1) * grid_y) * self.alpha) * psi_0, axis=1)
            E_k = hbar ** 2 * ((1 - np.cos(kx_list * self.alpha)) +  (1 - np.cos(ky_list * self.alpha))) / m / self.alpha ** 2
            x_list = grid_x * self.alpha
            y_list = grid_y * self.alpha
            psi_xt = 1 / L_x ** 2  * psi_k * np.exp(-1j * E_k.reshape(len(E_k), 1) * t_list / hbar).transpose()@(np.exp(1j * (kx_list.reshape(len(kx_list), 1) * x_list +  ky_list.reshape(len(ky_list), 1) * y_list)))
        elif self.network_type == '3D':
            N_x = int(round(self.N ** (1/3)))
            L_x = self.alpha * N_x
            grid_x = np.tile(np.arange(0, N_x, 1), (N_x, N_x, 1)).ravel()
            grid_y = np.tile(np.arange(0, N_x, 1), (N_x, N_x, 1)).transpose(1, 2, 0).ravel()
            grid_z = np.tile(np.arange(0, N_x, 1), (N_x, N_x, 1)).transpose(2, 1, 0).ravel()
            kx_list = 2 * np.pi  * grid_x / N_x / self.alpha
            ky_list = 2 * np.pi  * grid_y / N_x / self.alpha
            kz_list = 2 * np.pi  * grid_z / N_x / self.alpha
            psi_k = np.sum(np.exp(-1j * (kx_list.reshape(len(kx_list), 1) * grid_x + ky_list.reshape(len(ky_list), 1) * grid_y + kz_list.reshape(len(kz_list), 1) * grid_z) * self.alpha) * psi_0, axis=1)
            E_k = hbar ** 2 * ((1 - np.cos(kx_list * self.alpha)) +  (1 - np.cos(ky_list * self.alpha)) + +  (1 - np.cos(kz_list * self.alpha))) / m / self.alpha ** 2
            x_list = grid_x * self.alpha
            y_list = grid_y * self.alpha
            z_list = grid_z * self.alpha
            psi_xt = 1 / L_x ** 3  * psi_k * np.exp(-1j * E_k.reshape(len(E_k), 1) * t_list / hbar).transpose()@(np.exp(1j * (kx_list.reshape(len(kx_list), 1) * x_list +  ky_list.reshape(len(ky_list), 1) * y_list +  kz_list.reshape(len(kz_list), 1) * z_list)))

        phase = np.angle(psi_xt) 
        rho = np.abs(psi_xt) ** 2
        return rho, phase 

    def test_qd_gausswave(self, dx, dt):
        """TODO: For test purpose, quantum dynamics for Gaussian wave packet

        :returns: state, psi for Crank-Nicholson and analytical solution 

        """
        L = 10
        x = np.arange(0, L, dx)
        x0 = int(L//2)
        t = np.arange(0, 500, dt)
        p0 = 0
        sigma = 1
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
        """TODO: For test purpose, quantum dynamics for random fluctuations

        :returns: state, psi 

        """
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

    def get_phi_state(self, seed_initial_condition, save_des=None):
        """TODO: run simulation to generate time-dependent state for t 

        :seed_initial_condition: seed to generate initial condition
        :returns: data, (t, state)

        """
        self.seed_initial_condition = seed_initial_condition
        if quantum_or_not:
            data = self.quantum_diffusion()

        else:
            phi_state = self.classic_diffusion()
            data = [phi_state]

        if len(self.t) > 5000:
            print('---save partial data---')
            t = np.hstack(( self.t[:100], self.t[100:10000][::10], self.t[10000:][::100] ))  # save space
            for file_i, data_i in zip(save_des, data):
                data_i = np.vstack((data_i[:100], data_i[100:10000][::10], data_i[10000:][::100]))
                data_save = np.hstack(( t.reshape(len(t), 1), data_i )) 
                np.save(file_i, data_save) 
        else:
            print('---save ALL data---')
            t = self.t
            for file_i, data_i in zip(save_des, data):
                data_save = np.hstack(( t.reshape(len(t), 1), data_i )) 
                np.save(file_i + '_full', data_save ) 
        return data
        
    def save_phi_parallel(self, cpu_number, seed_initial_condition_list):
        """TODO: call parallel function to implement Crank-Nicholson method to generate data

        :cpu_number: the number of cpus to use for multiprocessing
        :seed_initial_condition_list: seed list to generate initial conditions
        :returns: None

        """
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
            if self.m == m_e:
                save_file.append(des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_seed_initial=' )
            else:
                save_file.append(des + f'm={self.m}_N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_seed_initial=' )

        p = mp.Pool(cpu_number)
        p.starmap_async(self.get_phi_state,  [(seed_initial_condition, [file_i + f'{seed_initial_condition}' for file_i in save_file]) for seed_initial_condition in seed_initial_condition_list]).get()
        p.close()
        p.join()
        return None

    def get_phi_eigen_fourier(self, seed_initial_condition, save_des=None):
        """TODO: implement Fourier transformation to generate time-dependent state for t 

        :seed_initial_condition: seed to generate initial condition
        :returns: data, (t, state)

        """
        self.seed_initial_condition = seed_initial_condition
        data = self.eigen_fourier()
        if len(self.t) > 5000:
            t = np.hstack(( self.t[:100], self.t[100:10000][::10], self.t[10000:][::100] ))  # save space
            for file_i, data_i in zip(save_des, data):
                data_i = np.vstack((data_i[:100], data_i[100:10000][::10], data_i[10000:][::100]))
                data_save = np.hstack(( t.reshape(len(t), 1), data_i )) 
                np.save(file_i, data_save) 
        else:
            print('save ALL data')
            t = self.t
            for file_i, data_i in zip(save_des, data):
                data_save = np.hstack(( t.reshape(len(t), 1), data_i )) 
                np.save(file_i + '_full', data_save ) 
        return data
        
    def save_phi_eigen_fourier_parallel(self, cpu_number, seed_initial_condition_list):
        """TODO: call parallel function to implement Fourier transformation to generate data

        :cpu_number: the number of cpus to use for multiprocessing
        :seed_initial_condition_list: seed list to generate initial conditions
        :returns: None

        """
        des_state = '../data/quantum/state_ectfp/' + self.network_type + '/' 
        des_phase = '../data/quantum/phase_ectfp/' + self.network_type + '/' 
        des_list = [des_state, des_phase]
        save_file = []
        for des in des_list:
            if not os.path.exists(des):
                os.makedirs(des)
            if self.m == m_e:
                save_file.append(des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_seed_initial=' )
            else:
                save_file.append(des + f'm={self.m}_N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_seed_initial=' )

        p = mp.Pool(cpu_number)
        p.starmap_async(self.get_phi_eigen_fourier,  [(seed_initial_condition, [file_i + f'{seed_initial_condition}' for file_i in save_file]) for seed_initial_condition in seed_initial_condition_list]).get()
        p.close()
        p.join()
        return None
    


    

cpu_number = 40


m_e = 5.68
hbar = 0.6582


if __name__ == '__main__':

    "quantum or classical"
    quantum_or_not = False
    quantum_or_not = True


    "initial setup"
    #chi2 for rho and uniform for phase
    initial_setup = 'chi2_uniform'
    rho_list = [[1e-4], [1e-2], [1], [10]]
    phase_list = [[-1, 1], [-1/2, 1/2], [-1/4, 1/4], [0, 0]]

    # pseudo sinfunction
    initial_setup = 'sum_sin_inphase'
    rho_list = [[1], [5], [10]]
    rho_list = [[5], [10]]
    phase_list = [[1e-3, 1], [1e-3, 0.1], [1e-3, 0.03]]
    phase_list = [[1e-2, 1], [1e-2, 0.1], [1e-2, 0.03]]

    # uniform random distribution for rho and phase
    initial_setup = 'uniform_random'
    rho_list = [[0, 1], [1/4, 3/4], [3/8, 5/8], [1, 1]]
    phase_list = [[-1, 1], [-1/2, 1/2], [-1/4, 1/4], [0, 0]]

    # uniform random for u and phase
    initial_setup = 'u_uniform_random'
    rho_list = [[0], [0.05], [0.1], [0.2]]
    phase_list = [[0], [0.05], [0.1], [0.2]]

    #fully localized density and uniform phase
    initial_setup = 'full_local'
    rho_list = [[0]]
    phase_list = [[0, 0]]

    # normal random for u and uniform random for phase 
    initial_setup = 'u_normal_phase_uniform_random'
    phase_list = [[1]]
    rho_list = [[0], [0.05], [0.1], [0.2]]

    # normal random for u and phase
    initial_setup = 'u_normal_random'
    rho_list = [[0], [0.05], [0.1], [0.2]]
    phase_list = [[0], [0.05], [0.1], [0.2]]

    # normal random for u and phase with cutoffs
    initial_setup = 'u_normal_random_cutoff'
    rho_list = [[0, 0.2], [0.05, 0.2], [0.1, 0.2], [0.2, 0.2]]
    phase_list = [[0, 0.2], [0.05, 0.2], [0.1, 0.2], [0.2, 0.2]]

    # multiple localized phase, uniform density
    initial_setup = 'phase_multi_locals'
    rho_list = [[0]]
    phase_list = [[1, 1], [10, 0.5]]

    "prepare distribution parameters"
    distribution_params_raw = [rho + phase for phase in phase_list for rho in rho_list]
    distribution_params_list = []
    for i in distribution_params_raw:
        distribution_params_list.append( [round(j, 10) for j in i])




    "network topology"
    ####### disordered lattice ######################
    network_type_list = ['2D_disorder', '3D_disorder'] 
    d_list = [0.7, 0.4]
    N_list_list = [[10000], [8000]]
    alpha_list = [1]
    seed_list = [0, 1, 2]

    ########## regular lattice #####################
    network_type_list = ['1D', '2D', '3D']
    d_list = [4, 4, 4]
    N_list_list = [[10000], [10000], [8000]]
    alpha_list = [1]
    seed_list = [0]

    "for test purpose, set small N first"
    network_type = ['1D']
    d_list = [4]
    N_list_list = [[1000]]
    alpha_list = [1]
    seed_list = [0]


    num_realization_list = [1]* len(alpha_list)
    dt_list = [1] * len(alpha_list)
    m_list = [m_e]  * len(alpha_list)

    #############################################################
    # start simulation
    for seed in seed_list:
        for network_type, d, N_list in zip(network_type_list, d_list, N_list_list):
            for m, N, alpha, dt, num_realization in zip(m_list, N_list, alpha_list, dt_list, num_realization_list):
                seed_initial_condition_list = np.arange(num_realization) 
                t = np.arange(0, 2000*dt, dt)
                t = np.arange(0, 10000*dt, dt)
                for distribution_params in distribution_params_list:
                    t1 = time.time()
                    dp = diffusionPersistence(quantum_or_not, network_type, m, N, d, seed, alpha, t, dt, initial_setup, distribution_params)
                    dp.save_phi_parallel(cpu_number, seed_initial_condition_list)
                    #dp.save_phi_eigen_fourier_parallel(cpu_number, seed_initial_condition_list)
                    t2 = time.time()
                    print(f'time for running on network={network_type}, N={N}, initial distribution={distribution_params} is {round(t2 - t1, 2)}.')

