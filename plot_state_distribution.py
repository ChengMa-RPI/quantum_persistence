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

def title_name(params, quantum_or_not, d=None):

    if quantum_or_not:
        rho_start, rho_end, phase_start, phase_end = params
        if phase_start == phase_end:
            phase_title = '$\\theta = C$'
        else:
            phase_title = '$\\theta \sim $' + f'({phase_start } $\\pi$, {phase_end } $\\pi$)'

    else:
        rho_start, rho_end = params

    if rho_start == rho_end:
        rho_title = '$\\rho = C$'
    else:
        rho_title = '$\\rho \sim $' + f'({rho_start * 2}/N, {rho_end * 2}/N)'

    if quantum_or_not:
        if d:
            return f'$d={d}$\n' + phase_title
        else:
            return rho_title  + '\n' + phase_title
    else:
        if d:
            return f'$d={d}$' 
        else:
            return rho_title 




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
        N_total = 0
        for seed_initial_condition in seed_initial_condition_list:
            state_distribution = self.read_state_distribution(seed_initial_condition)
            p_state = state_distribution['p_state']
            N_actual = sum(p_state[str(t_list[0])]['p'])
            N_total += N_actual
            bins_num = state_distribution['bin_num']
            #t_list = p_state['t_list']
            for t_i in t_list:
                data_i = p_state[str(t_i)]
                p, bins = data_i['p'], data_i['bins']
                replicate = np.repeat(bins, p)
                data_list[t_i].append(replicate)
        for t_i in t_list:
            p, bins = np.histogram(data_list[t_i], bins=bins_num)
            p = p / N_total  # normalize by the total number of nodes in all realizations
            if self.rho_or_phase == 'rho':
                p = p[5:-5]
                bins = bins[5:-5]  # remove edge effects, which is caused by multiple realizations

            if self.rho_or_phase == 'phase':
                ax.semilogy(bins[:-1] / np.pi , p, '-', label='t=' + str(t_i) )
            else:
                ax.semilogy(bins[:-1] - 1/self.N , p, '-', label='t=' + str(t_i) )
        if self.quantum_or_not == True:
            state = '$\\rho$'
        else:
            state = '$\\psi$'

        if self.rho_or_phase == 'rho':
            ax.set_xlim(-1.2/self.N, 6/self.N)
        ax.locator_params(axis='x', nbins=4)
        return state_distribution

    def plot_statedis_initial_setup(self, distribution_params_list, seed_initial_condition_list, t_list):
        rows = 4
        cols = len(distribution_params_list) // rows
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(4 * cols, 3.5 * rows))
        for i, distribution_params in enumerate(distribution_params_list):
            title = title_name(distribution_params, self.quantum_or_not)
            #ax = axes[i]
            ax = axes[i // cols, i % cols]
            simpleaxis(ax)
            self.distribution_params = distribution_params
            self.plot_state_distribution(ax, seed_initial_condition_list, t_list)
            ax.tick_params(axis='both', which='major', labelsize=13)
            ax.annotate(f'({letters[i]})', xy=(0, 0), xytext=(-0.05, 1.02), xycoords='axes fraction', size=22)
            ax.set_title(title, size=labelsize*0.5, y = 0.92)

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

    def plot_statedis_disorder(self, d_list, distribution_params_list, seed_initial_condition_list, t_list):
        rows = len(d_list)
        cols = len(distribution_params_list)
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(4 * cols, 3.5 * rows))
        for i, d in enumerate(d_list):
            for j, distribution_params in enumerate(distribution_params_list):
                ax = axes[i, j]
                simpleaxis(ax)
                self.distribution_params = distribution_params
                self.d = d
                title = title_name(distribution_params, self.quantum_or_not, d)
                self.plot_state_distribution(ax, seed_initial_condition_list, t_list)
                ax.tick_params(axis='both', which='major', labelsize=13)
                ax.annotate(f'({letters[i]})', xy=(0, 0), xytext=(-0.05, 1.02), xycoords='axes fraction', size=22)
                ax.set_title(title, size=labelsize*0.5, y = 0.92)

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
        fig.subplots_adjust(left=0.1, right=0.95, wspace=0.25, hspace=0.25, bottom=0.11, top=0.90)
        filename = f'quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}_seed={seed}_{rho_or_phase}_distribution.png'
        save_des = '../transfer_figure/' + filename
        plt.savefig(save_des, format='png')
        plt.close()


letters = 'abcdefghijklmnopqrstuvwxyz'

if __name__ == '__main__':
    quantum_or_not = False
    initial_setup = 'uniform_random'
    quantum_or_not = True
    initial_setup = 'gaussian_wave'
    initial_setup = 'uniform_random'
    network_type = '2D'
    network_type = '2D_disorder'
    N = 100
    d = 4
    seed = 0
    alpha = 1
    dt = 0.1
    reference_line = 'average'
    reference_line = 0.5
    seed_initial_condition_list = np.arange(0, 10, 1)
    distribution_params = [1, 1, 1, 1]
    rho_or_phase = 'phase'
    rho_or_phase = 'rho'
    psd = plotStateDistribution(quantum_or_not, network_type, N, d, seed, alpha, dt, initial_setup, distribution_params, seed_initial_condition_list, reference_line, rho_or_phase)
    #pdpp.plot_dpp_t()

    #pdpp.plot_dpp_scaling(N_list)
    reference_lines = ['average']
    t_list = np.round(np.arange(0.0, 100, 20), 1).tolist()
    #seed_initial_condition_list = [0]

    rho_list = [[0, 1], [1/4, 3/4], [3/8, 5/8], [1, 1]]
    phase_list = [[-1, 1], [-1/2, 1/2], [-1/4, 1/4], [0, 0]]
    rho_list = [[1, 1]]
    phase_list = [[-1, 1], [-1/2, 1/2], [-1/4, 1/4]]
    distribution_params_raw = [rho + phase for rho in rho_list for phase in phase_list]



    distribution_params_list = []
    for i in distribution_params_raw:
        distribution_params_list.append( [round(j, 3) for j in i])


    #psd.plot_statedis_initial_setup(distribution_params_list, seed_initial_condition_list, t_list)

    d_list = [0.6, 0.7, 0.8, 0.9]
    psd.plot_statedis_disorder(d_list, distribution_params_list, seed_initial_condition_list, t_list)
