import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
sys.path.insert(1, '/home/mac/RPI/research/')

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd 
from cycler import cycler
import matplotlib as mpl
import itertools
import seaborn as sns
import multiprocessing as mp

from collections import Counter
from scipy.integrate import odeint
from mutual_framework import network_generate, betaspace
import scipy.stats as stats
import time
from netgraph import Graph
import matplotlib.image as mpimg
from collections import defaultdict
from matplotlib import patches 
import json

fs = 24
ticksize = 20
labelsize = 35
anno_size = 18
subtitlesize = 15
legendsize= 20
alpha = 0.8
lw = 3
marksize = 8


mpl.rcParams['axes.prop_cycle'] = (cycler(color=['#fc8d62',  '#66c2a5', '#e78ac3','#a6d854',  '#8da0cb', '#ffd92f','#b3b3b3', '#e5c494', '#7fc97f', '#beaed4', '#ffff99' ])  * cycler(linestyle=['-']))
color1 = ['#fc8d62',  '#66c2a5', '#e78ac3', '#a6d854',  '#8da0cb', '#ffd92f','#b3b3b3', '#e5c494', '#7fc97f', '#beaed4', '#ffff99']
color2 = ['brown', 'orange', 'lightgreen', 'steelblue','slategrey', 'violet']

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

class plotStateDistribution():
    def __init__(self, quantum_or_not, network_type, N, d, seed, alpha, initial_setup, seed_initial_condition_list, reference_line):
        self.quantum_or_not = quantum_or_not
        self.network_type = network_type
        self.N = N
        self.d = d
        self.seed = seed
        self.alpha = alpha
        self.initial_setup = initial_setup
        self.seed_initial_condition_list = seed_initial_condition_list
        self.reference_line = reference_line

    def read_state_distribution(self, seed_initial_condition):
        if self.quantum_or_not:
            des = '../data/quantum/state_distribution/' + self.network_type + '/' 
        else:
            des = '../data/classical/state_distribution/' + self.network_type + '/' 
        filename = des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_setup={self.initial_setup}_reference={self.reference_line}_seed_initial={seed_initial_condition}.json'
        f = open(filename)
        state_distribution = json.load(f)
        return state_distribution

    
    def plot_state_distribution(self, ax, seed_initial_condition_list, t_list):
        data_list = defaultdict(list)
        for seed_initial_condition in seed_initial_condition_list:
            state_distribution = self.read_state_distribution(seed_initial_condition)
            p_state = state_distribution['p_state']
            bins_num = state_distribution['bin_num']
            #t_list = p_state['t_list']
            for t_i in t_list:
                data_i = p_state[str(t_i)]
                p, bins = data_i['p'], data_i['bins']
                replicate = np.repeat(bins, p)
                data_list[t_i].append(replicate)
        for t_i in t_list:
            p, bins = np.histogram(data_list[t_i], bins=bins_num)
            p = p / self.N / len(seed_initial_condition_list)
            
            ax.semilogy(bins[:-1] - 1/self.N , p, '-', label='t=' + str(t_i) )
        if self.quantum_or_not == True:
            state = '$\\rho$'
        else:
            state = '$\\psi$'

        #ax.xlabel(state, fontsize=labelsize * 0.5)
        #ax.ylabel('$P$(' + state +')', fontsize=labelsize * 0.5)
        plt.legend(frameon=False)
        plt.locator_params(axis='x', nbins=4)
        #plt.savefig('../report/report100422/' + 'F4.png')
        #plt.close()
        return state_distribution

    def plot_statedis_initial_setup(self, initial_setup_list, seed_initial_condition_list, t_list):
        cols = len(initial_setup_list)
        rows = 1
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(4 * cols, 3.5 * rows))
        for (i, initial_setup), title in zip(enumerate(initial_setup_list), titles):
            ax = axes[i]
            simpleaxis(ax)
            self.initial_setup = initial_setup
            self.plot_state_distribution(ax, seed_initial_condition_list, t_list)
            self.plot_state_distribution(ax, seed_initial_condition_list, t_list)
            ax.tick_params(axis='both', which='major', labelsize=13)
            ax.set_title(title, size=labelsize*0.5)


        #ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.23, 0.09) ) 

        if self.quantum_or_not == True:
            state = '$\\rho$'
        else:
            state = '$\\psi$'
        ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.19, -0.09) ) 
        fig.text(x=0.02, y=0.5, horizontalalignment='center', s='$P$(' + state +')', size=labelsize*0.6, rotation=90)
        fig.text(x=0.5, y=0.03, horizontalalignment='center', s=state, size=labelsize*0.6)
        fig.subplots_adjust(left=0.1, right=0.95, wspace=0.25, hspace=0.25, bottom=0.21, top=0.90)
        #save_des = '../manuscript/dimension_reduction_v3_072422/' + self.dynamics + '_' + self.network_type + f'_tau_c_m.png'
        #plt.savefig(save_des, format='png')
        #plt.close()


        




if __name__ == '__main__':
    quantum_or_not = False
    initial_setup = 'uniform_random'
    quantum_or_not = True
    initial_setup = 'rho_uniform_phase_uniform'
    initial_setup = 'rho_const_phase_uniform'
    initial_setup = 'rho_uniform_phase_const_pi'
    initial_setup = 'rho_uniform_phase_const_pi_quater'
    initial_setup = 'rho_uniform_phase_const_pi_half'
    initial_setup = 'gaussian_wave'
    network_type = '1D'
    N = 10000
    d = 4
    seed = 0
    alpha = 1
    reference_line = 0.8
    reference_line = 'average'
    seed_initial_condition_list = np.arange(0, 10, 1)
    psd = plotStateDistribution(quantum_or_not, network_type, N, d, seed, alpha, initial_setup, seed_initial_condition_list, reference_line)
    #pdpp.plot_dpp_t()

    #pdpp.plot_dpp_scaling(N_list)
    reference_lines = ['average']
    t_list = np.round(np.arange(0.0, 50, 5), 1).tolist()
    #seed_initial_condition_list = [0]
    initial_setup_list = ['rho_uniform_phase_uniform', 'rho_const_phase_uniform', 'rho_uniform_phase_const_pi']
    titles = ['uniform random', 'const $\\rho$', 'const $\\theta$']

    collector = psd.plot_statedis_initial_setup(initial_setup_list, seed_initial_condition_list, t_list)


