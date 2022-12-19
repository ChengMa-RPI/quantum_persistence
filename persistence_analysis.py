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
import json




class persistenceAnalysis:
    def __init__(self, quantum_or_not, network_type, N, d, seed, alpha, dt, initial_setup, distribution_params, reference_line):
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
        self.dt = dt
        self.initial_setup = initial_setup
        self.distribution_params = distribution_params
        self.reference_line = reference_line
        self.seed_initial_condition = None

    def read_phi(self, seed_initial_condition, rho_or_phase='rho'):
        if self.quantum_or_not:
            if rho_or_phase == 'rho':
                des = '../data/quantum/state/' + self.network_type + '/' 
            elif rho_or_phase == 'phase':
                des = '../data/quantum/phase/' + self.network_type + '/' 

        else:
            des = '../data/classical/state/' + self.network_type + '/' 
        save_file = des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_seed_initial={seed_initial_condition}.npy'
        data = np.load(save_file)
        t, state = data[:, 0], data[:, 1:]
        return t, state

    def read_meta_data(self, seed_initial_condition):
        if self.quantum_or_not:
            des = '../data/quantum/meta_data/' + self.network_type + '/' 
        else:
            des = '../data/classical/meta_data/' + self.network_type + '/' 
        filename = des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_reference={self.reference_line}_seed_initial={seed_initial_condition}.json'
        f = open(filename)
        meta_data = json.load(f)
        return meta_data

    def diffusive_persistence_prob(self, seed_initial_condition, save_des=None):
        """OLD ONE"""
        t, phi_state = self.read_phi(seed_initial_condition)
        N_actual = len(phi_state[0])
        dt = np.round(np.mean(np.diff(t)), 5)
        reference_value = 1/ N_actual
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
        pa = na / N_actual
        pb = nb / N_actual
        df = pd.DataFrame(np.vstack(( t, pa, pb )).transpose()) 
        if save_des:
            df.to_csv(save_des, index=None, header=None)
        return df

    def get_meta_data(self, seed_initial_condition, save_des=None):
        t, phi_state = self.read_phi(seed_initial_condition)
        t = t[2:]
        phi_state = phi_state[2:]
        t_length = len(t)
        N_actual = len(phi_state[0])
        dt = np.round(np.mean(np.diff(t)), 5)
        reference_value = 1/ N_actual
        if reference_line != 'average':
            reference_value *= float(reference_line)
        mask_above = np.heaviside(phi_state - reference_value, 0)
        cross_meta_data = {}
        for node, node_mask in enumerate(mask_above.transpose()):
            transition_index = np.where(np.diff(node_mask) != 0)[0] + 1
            padded_index = np.round(np.r_[0, transition_index * dt, t[-1]], 3)
            even = padded_index[::2].tolist()
            odd = padded_index[1::2].tolist()
            if node_mask[0] > 0:
                above_start = even
                below_start = odd
            else:
                above_start = odd
                below_start = even
            cross_meta_data[node] = {'above_start' : above_start, 'below_start' : below_start}
        meta_data = {'t':[t[0], t[-1], dt], 'meta_data':cross_meta_data}
        if save_des:
            with open(save_des, 'w') as fp:
                json.dump(meta_data, fp)
        return meta_data

    def diffusive_persistence(self, seed_initial_condition, save_des=None):
        """
        needs to be simplified... na, nb should be generalized
        """
        meta_data = self.read_meta_data(seed_initial_condition)
        t_start, t_end, dt = meta_data['t']
        cross_meta_data = meta_data['meta_data']
        nodes = cross_meta_data.keys()
        N_actual = len(nodes)
        all_first_below = []
        all_first_above = []
        for node in nodes:
            above_start = cross_meta_data[node]['above_start']
            below_start = cross_meta_data[node]['below_start']
            if above_start[0] == 0:
                if len(above_start) > 1:
                    first_above = above_start[1]
                else:
                    first_above = -1
                first_below = below_start[0]
            elif below_start[0] == 0:
                if len(below_start) > 1:
                    first_below = below_start[1]
                else:
                    first_below = -1
                first_above = above_start[0]
            else:
                print('where is 0 in starts')
            all_first_above.append(first_above)
            all_first_below.append(first_below)

        all_first_above = np.array(all_first_above)
        all_first_below = np.array(all_first_below)
        always_above = np.sum((all_first_above < 0) & (np.abs(all_first_below - t_end) < 1e-5) )
        always_below = np.sum((all_first_below < 0) & (np.abs(all_first_above - t_end) < 1e-5) )
        t = np.arange(t_start, t_end, dt)
        na = np.zeros(( len(t) ))
        nb = np.zeros(( len(t) ))
        is_above = (all_first_above > all_first_below)
        is_below = (all_first_above < all_first_below)
        na_candidate = all_first_below[is_above]
        nb_candidate = all_first_above[is_below]
        for i, t_i in enumerate(t):
            na[i] = np.sum(na_candidate > t_i)
            nb[i] = np.sum(nb_candidate > t_i)
        # this can be improved: sort na_candidate and save cross time.
        na += always_above
        nb += always_below
        pa = na / N_actual
        pb = nb / N_actual
        df = pd.DataFrame(np.vstack(( t, pa, pb )).transpose()) 
        if save_des:
            df.to_csv(save_des, index=None, header=None)
        return df
            
    def get_dpp_parallel(self, cpu_number, seed_initial_condition_list):
        if self.quantum_or_not:
            des = '../data/quantum/persistence/' + self.network_type + '/' 
        else:
            des = '../data/classical/persistence/' + self.network_type + '/' 
        save_file = des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_reference={self.reference_line}_seed_initial='
        if not os.path.exists(des):
            os.makedirs(des)
        p = mp.Pool(cpu_number)
        p.starmap_async(self.diffusive_persistence,  [(seed_initial_condition, save_file + f'{seed_initial_condition}.csv') for seed_initial_condition in seed_initial_condition_list]).get()
        p.close()
        p.join()
        return None

    def get_meta_data_parallel(self, cpu_number, seed_initial_condition_list):
        if self.quantum_or_not:
            des = '../data/quantum/meta_data/' + self.network_type + '/' 
        else:
            des = '../data/classical/meta_data/' + self.network_type + '/' 
        save_file = des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_reference={self.reference_line}_seed_initial='
        if not os.path.exists(des):
            os.makedirs(des)
        p = mp.Pool(cpu_number)
        p.starmap_async(self.get_meta_data,  [(seed_initial_condition, save_file + f'{seed_initial_condition}.json') for seed_initial_condition in seed_initial_condition_list]).get()
        p.close()
        p.join()
        return None

    def get_state_distribution(self, seed_initial_condition, t_list, rho_or_phase, bin_num = 100):
        if self.quantum_or_not:
            if rho_or_phase == 'rho':
                des = '../data/quantum/state_distribution/' + self.network_type + '/' 
            elif rho_or_phase == 'phase':
                des = '../data/quantum/phase_distribution/' + self.network_type + '/' 
            else:
                print('Please specify which quantity to look at!')
                return 
        else:
            des = '../data/classical/state_distribution/' + self.network_type + '/' 
        save_file = des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_reference={self.reference_line}_seed_initial={seed_initial_condition}.json'
        if not os.path.exists(des):
            os.makedirs(des)
        t, state = self.read_phi(seed_initial_condition, rho_or_phase)
        dt = np.round(t[1] - t[0], 3)
        p_state = {} 
        for t_i in t_list:
            p, bins = np.histogram(state[int(t_i/dt)], bin_num)
            p_state[t_i]  = {'p':p.tolist(), 'bins':(bins[:-1] + (bins[2]-bins[1])/2).tolist()}
        state_distribution = {'t': t[-1], 't_list': t_list, 'bin_num': bin_num, 'p_state': p_state}
        
        with open(save_file, 'w') as fp:
            json.dump(state_distribution, fp)
        return None




    

cpu_number = 4


m = 5.68
hbar = 0.6582


if __name__ == '__main__':
    network_type = '2D'
    network_type = '2D_disorder'
    quantum_or_not = False
    initial_setup = 'uniform_random'
    quantum_or_not = True
    initial_setup = 'gaussian_wave'
    initial_setup = 'sum_sin_inphase'
    initial_setup = 'uniform_random'
    N = 100
    d = 4
    seed = 0
    alpha = 1
    reference_line = 0.5
    reference_line = 'average'
    reference_lines = ['average']
    seed_initial_condition_list = np.arange(10)


    alpha_list = [1]
    N_list = [100]
    dt_list = [0.1]
    num_realization_list = [10] 
    rho_list = [[0, 1], [1/4, 3/4], [3/8, 5/8], [1, 1]]
    phase_list = [[-1, 1], [-1/2, 1/2], [-1/4, 1/4], [0, 0]]

    rho_list = [[1, 1]]
    phase_list = [[-1, 1], [-1/2, 1/2], [-1/4, 1/4]]
    distribution_params_raw = [rho + phase for rho in rho_list for phase in phase_list]


    distribution_params_list = []
    for i in distribution_params_raw:
        distribution_params_list.append( [round(j, 3) for j in i])


    d_list = [0.6, 0.7, 0.8, 0.9]
    for d in d_list:
        for N, alpha, dt, num_realization in zip(N_list, alpha_list, dt_list, num_realization_list):
            seed_initial_condition_list = np.arange(num_realization)
            for distribution_params in distribution_params_list:
                pA = persistenceAnalysis(quantum_or_not, network_type, N, d, seed, alpha, dt, initial_setup, distribution_params, reference_line)
                pA.get_meta_data_parallel(cpu_number, seed_initial_condition_list)
                pA.get_dpp_parallel(cpu_number, seed_initial_condition_list)
                t_list = np.round(np.arange(0.0, 100, 1), 1).tolist()
                for seed_initial_condition in seed_initial_condition_list:
                    pA.get_state_distribution(seed_initial_condition, t_list, 'rho')
                    pA.get_state_distribution(seed_initial_condition, t_list, 'phase')
                    pass
                pass

