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

def title_name(params):
    rho_start, rho_end, phase_start, phase_end = params
    if rho_start == rho_end:
        rho_title = '$\\rho = C$'
    else:
        rho_title = '$\\rho \sim \\mathcal{U}$' + f'({rho_start * 2}/N, {rho_end * 2}/N)'

    if phase_start == phase_end:
        phase_title = '$\\theta = C$'
    else:
        phase_title = '$\\theta \sim \\mathcal{U}$' + f'({phase_start } $\\pi$, {phase_end } $\\pi$)'
    return rho_title  + ', ' + phase_title



class plotStateDistribution():
    def __init__(self, quantum_or_not, network_type, N, d, seed, alpha, dt, initial_setup, distribution_params, seed_initial_condition_list, reference_line, rho_or_phase):
        self.quantum_or_not = quantum_or_not
        self.network_type = network_type
        self.N = N
        self.d = d
        self.seed = seed
        self.alpha = alpha
        self.dt = dt
        self.initial_setup = initial_setup
        self.distribution_params = distribution_params
        self.seed_initial_condition_list = seed_initial_condition_list
        self.reference_line = reference_line
        self.rho_or_phase = rho_or_phase

    def read_state_distribution(self, seed_initial_condition):
        if self.quantum_or_not:
            if self.rho_or_phase == 'rho':
                des = '../data/quantum/state_distribution/' + self.network_type + '/' 
            elif self.rho_or_phase == 'phase':
                des = '../data/quantum/phase_distribution/' + self.network_type + '/' 
            else:
                print('Please specify the state type')
                return 
        else:
            des = '../data/classical/state_distribution/' + self.network_type + '/' 
        filename = des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_reference={self.reference_line}_seed_initial={seed_initial_condition}.json'
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
            
            if self.rho_or_phase == 'phase':
                ax.semilogy(bins[:-1] / np.pi , p, '-', label='t=' + str(t_i) )
            else:
                ax.semilogy(bins[:-1] - 1/self.N , p, '-', label='t=' + str(t_i) )
        if self.quantum_or_not == True:
            state = '$\\rho$'
        else:
            state = '$\\psi$'

        plt.legend(frameon=False)
        plt.locator_params(axis='x', nbins=4)
        return state_distribution

    def plot_statedis_initial_setup(self, distribution_params_list, seed_initial_condition_list, t_list):
        cols = len(distribution_params_list) // 2
        rows = 2
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(4 * cols, 3.5 * rows))
        for i, distribution_params in enumerate(distribution_params_list):
            title = title_name(distribution_params)
            #ax = axes[i]
            ax = axes[i // 2, i % 2]
            simpleaxis(ax)
            self.distribution_params = distribution_params
            self.plot_state_distribution(ax, seed_initial_condition_list, t_list)
            ax.tick_params(axis='both', which='major', labelsize=13)
            ax.set_title(title, size=labelsize*0.5)

        #ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.23, 0.09) ) 

        if self.quantum_or_not == True:
            if self.rho_or_phase == 'rho':
                xlabel = '$\\rho - \\langle \\rho \\rangle$'
                ylabel =  '$P(\\rho)$'
            elif self.rho_or_phase == 'phase':
                xlabel = '$\\theta$   ($\\times \pi$)'
                ylabel = '$P(\\theta)$'
        else:
            xlabel = '$\\psi$'
            ylabel = '$P(\\psi)$'
        ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.19, -0.09) ) 
        fig.text(x=0.02, y=0.5, horizontalalignment='center', s=ylabel, size=labelsize*0.6, rotation=90)
        fig.text(x=0.5, y=0.03, horizontalalignment='center', s=xlabel, size=labelsize*0.6)
        fig.subplots_adjust(left=0.1, right=0.95, wspace=0.25, hspace=0.25, bottom=0.21, top=0.90)
        filename = f'N={self.N}_{rho_or_phase}_distribution.png'
        save_des = '../transfer_figure/' + filename
        plt.savefig(save_des, format='png')
        plt.close()



if __name__ == '__main__':
    quantum_or_not = False
    initial_setup = 'uniform_random'
    quantum_or_not = True
    initial_setup = 'gaussian_wave'
    initial_setup = 'uniform_random'
    network_type = '1D'
    N = 100
    d = 4
    seed = 0
    alpha = 1
    dt = 1
    reference_line = 'average'
    seed_initial_condition_list = np.arange(0, 100, 1)
    distribution_params = [1, 1, 1, 1]
    rho_or_phase = 'phase'
    rho_or_phase = 'rho'
    psd = plotStateDistribution(quantum_or_not, network_type, N, d, seed, alpha, dt, initial_setup, distribution_params, seed_initial_condition_list, reference_line, rho_or_phase)
    #pdpp.plot_dpp_t()

    #pdpp.plot_dpp_scaling(N_list)
    reference_lines = ['average']
    t_list = np.round(np.arange(0.0, 10, 2), 1).tolist()
    #seed_initial_condition_list = [0]

    distribution_params_raw = [[0, 1, 1, 1], [1, 1, -1, 1], [1/4, 3/4, 0, 0], [3/8, 5/8, 0, 0] ]

    distribution_params_list = []
    for i in distribution_params_raw:
        distribution_params_list.append( [round(j, 3) for j in i])


    collector = psd.plot_statedis_initial_setup(distribution_params_list, seed_initial_condition_list, t_list)


