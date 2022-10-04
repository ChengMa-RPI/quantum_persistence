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
import json




class persistenceAnalysis:
    def __init__(self, quantum_or_not, network_type, N, d, seed, alpha, initial_setup, reference_line):
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
        self.initial_setup = initial_setup
        self.reference_line = reference_line
        self.seed_initial_condition = None

    def read_phi(self, seed_initial_condition):
        if self.quantum_or_not:
            des = '../data/quantum/state/' + self.network_type + '/' 
        else:
            des = '../data/classical/state/' + self.network_type + '/' 
        save_file = des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_setup={self.initial_setup}_seed_initial={seed_initial_condition}.npy'
        data = np.load(save_file)
        t, phi = data[:, 0], data[:, 1:]
        return t, phi

    def read_meta_data(self, seed_initial_condition):
        if self.quantum_or_not:
            des = '../data/quantum/meta_data/' + self.network_type + '/' 
        else:
            des = '../data/classical/meta_data/' + self.network_type + '/' 
        filename = des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_setup={self.initial_setup}_reference={self.reference_line}_seed_initial={seed_initial_condition}.json'
        f = open(filename)
        meta_data = json.load(f)
        return meta_data

    def diffusive_persistence_prob(self, seed_initial_condition, save_des=None):
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
        save_file = des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_setup={self.initial_setup}_reference={self.reference_line}_seed_initial='
        if not os.path.exists(des):
            os.makedirs(des)
        p = mp.Pool(cpu_number)
        p.starmap_async(self.diffusive_persistence_prob,  [(seed_initial_condition, save_file + f'{seed_initial_condition}.csv') for seed_initial_condition in seed_initial_condition_list]).get()
        p.close()
        p.join()
        return None

    def get_meta_dat_parallel(self, cpu_number, seed_initial_condition_list):
        if self.quantum_or_not:
            des = '../data/quantum/meta_data/' + self.network_type + '/' 
        else:
            des = '../data/classical/meta_data/' + self.network_type + '/' 
        save_file = des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_setup={self.initial_setup}_reference={self.reference_line}_seed_initial='
        if not os.path.exists(des):
            os.makedirs(des)
        p = mp.Pool(cpu_number)
        p.starmap_async(self.get_meta_data,  [(seed_initial_condition, save_file + f'{seed_initial_condition}.json') for seed_initial_condition in seed_initial_condition_list]).get()
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
    seed = 0
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
            pA = persistenceAnalysis(quantum_or_not, network_type, N, d, seed, alpha, initial_setup, reference_line)
            #pA.get_dpp_parallel(cpu_number, seed_initial_condition_list)
            pA.get_meta_dat_parallel(cpu_number, seed_initial_condition_list)
            pass

