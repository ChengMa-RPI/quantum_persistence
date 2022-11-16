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

class plotSignTime():
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

    def read_meta_data(self, seed_initial_condition):
        if self.quantum_or_not:
            des = '../data/quantum/meta_data/' + self.network_type + '/' 
        else:
            des = '../data/classical/meta_data/' + self.network_type + '/' 
        filename = des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_setup={self.initial_setup}_reference={self.reference_line}_seed_initial={seed_initial_condition}.json'
        f = open(filename)
        meta_data = json.load(f)
        return meta_data

    def get_above_interval_time(self, seed_initial_condition, tau):
        meta_data = self.read_meta_data(seed_initial_condition)
        t_start, t_end, dt = meta_data['t']
        cross_meta_data = meta_data['meta_data']
        nodes = cross_meta_data.keys()
        N_actual = len(nodes)
        all_above_intervals = []
        all_above_time = []
        for node in nodes:
            above_start = cross_meta_data[node]['above_start']
            below_start = cross_meta_data[node]['below_start']
            above_interval = []
            above_time = 0
            if above_start[0] == 0 and len(above_start) > len(below_start):
                below_start = below_start[:-1]
            elif below_start[0] == 0:
                if len(above_start) ==  len(below_start):
                    above_start = above_start[:-1] 
                below_start = below_start[1:]

            for i, j in zip(above_start, below_start):
                if j > tau and i < tau:
                    above_interval.append([i, tau])
                    above_time +=  tau - i
                elif j < tau:
                    above_interval.append([i, j - dt])
                    above_time += j - dt - i
            all_above_intervals.append(above_interval)
            all_above_time.append(above_time)
        return all_above_intervals, all_above_time
    
    def plot_sign_time_distribution(self, ax, tau, above_or_below):
        collector = []
        for seed_initial_condition in self.seed_initial_condition_list:
            all_above_intervals, all_above_time = self.get_above_interval_time(seed_initial_condition, tau)
            all_below_time = tau - np.array(all_above_time)
            if above_or_below == 'above':
                label = 'above'
                collector.extend(all_above_time)
            elif above_or_below == 'below':
                collector.extend(all_below_time)
                label = 'below'

        collector = np.array(collector)
        collector /= tau
        #collector = collector / tau * (1 - collector / tau)
        count, bins = np.histogram(collector, bins = 100)
        y = count / self.N / len(self.seed_initial_condition_list)
        x = bins[:-1]
        ax.semilogy(x, y, '.-', label=label)
        #ax.xlabel('$\\tau / t$', fontsize=labelsize * 0.5)
        #ax.ylabel('$P(\\tau / t)$', fontsize=labelsize * 0.5)
        return collector

    def plot_signtime_initial_setup(self, tau, initial_setup_list, titles):
        cols = len(initial_setup_list)
        rows = 1
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(4 * cols, 3.5 * rows))
        for (i, initial_setup), title in zip(enumerate(initial_setup_list), titles):
            ax = axes[i]
            simpleaxis(ax)
            self.initial_setup = initial_setup
            self.plot_sign_time_distribution(ax, tau, 'above')
            self.plot_sign_time_distribution(ax, tau, 'below')
            ax.tick_params(axis='both', which='major', labelsize=13)
            ax.set_title(title, size=labelsize*0.5)


        #ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.23, 0.09) ) 
        ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.19, 0.79) ) 
        fig.text(x=0.02, y=0.5, horizontalalignment='center', s="$P(\\tau/t)$", size=labelsize*0.6, rotation=90)
        fig.text(x=0.5, y=0.01, horizontalalignment='center', s="$\\tau/t$", size=labelsize*0.6)
        fig.subplots_adjust(left=0.1, right=0.95, wspace=0.25, hspace=0.25, bottom=0.2, top=0.90)
        #save_des = '../manuscript/dimension_reduction_v3_072422/' + self.dynamics + '_' + self.network_type + f'_tau_c_m.png'
        #plt.savefig(save_des, format='png')
        #plt.close()

    def plot_signtime_dxdt(self, tau, N_list, alpha_list, initial_setup, titles):
        cols = len(initial_setup_list)
        rows = 1
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(4 * cols, 3.5 * rows))
        for (i, initial_setup), title in zip(enumerate(initial_setup_list), titles):
            ax = axes[i]
            simpleaxis(ax)
            self.initial_setup = initial_setup
            self.plot_sign_time_distribution(ax, tau, 'above')
            self.plot_sign_time_distribution(ax, tau, 'below')
            ax.tick_params(axis='both', which='major', labelsize=13)
            ax.set_title(title, size=labelsize*0.5)


        #ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.23, 0.09) ) 
        ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.19, 0.79) ) 
        fig.text(x=0.02, y=0.5, horizontalalignment='center', s="$P(\\tau/t)$", size=labelsize*0.6, rotation=90)
        fig.text(x=0.5, y=0.01, horizontalalignment='center', s="$\\tau/t$", size=labelsize*0.6)
        fig.subplots_adjust(left=0.1, right=0.95, wspace=0.25, hspace=0.25, bottom=0.2, top=0.90)
        #save_des = '../manuscript/dimension_reduction_v3_072422/' + self.dynamics + '_' + self.network_type + f'_tau_c_m.png'
        #plt.savefig(save_des, format='png')
        #plt.close()


        




if __name__ == '__main__':
    quantum_or_not = False
    initial_setup = 'uniform_random'
    quantum_or_not = True
    initial_setup = 'rho_uniform_phase_uniform'
    initial_setup = 'rho_uniform_phase_const'
    initial_setup = 'rho_const_phase_uniform'
    network_type = '1D'
    N = 10000
    d = 4
    seed = 0
    alpha = 1
    reference_line = 0.8
    reference_line = 'average'
    seed_initial_condition_list = np.arange(0, 10, 1)
    pst = plotSignTime(quantum_or_not, network_type, N, d, seed, alpha, initial_setup, seed_initial_condition_list, reference_line)
    #pdpp.plot_dpp_t()
    L_list = np.arange(100, 60, 50)
    N_list = np.power(L_list, 2)
    N_list = [10000]

    #pdpp.plot_dpp_scaling(N_list)
    reference_lines = ['average']
    tau = 1000
    above_or_below = 'below'
    above_or_below = 'above'
    initial_setup_list = ['rho_uniform_phase_uniform', 'rho_const_phase_uniform', 'rho_uniform_phase_const_pi']
    titles = ['uniform random', 'const $\\rho$', 'const $\\theta$']

    pst.plot_signtime_initial_setup(tau, initial_setup_list, titles)


