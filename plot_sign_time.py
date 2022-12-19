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
            return f'$d={d}$' + '\n' + phase_title
        else:
            return rho_title  + '\n' + phase_title
    else:
        if d:
            return f'$d={d}$' 
        else:
            return rho_title  





class plotSignTime():
    def __init__(self, quantum_or_not, network_type, N, d, seed, alpha, dt, initial_setup, distribution_params, seed_initial_condition_list, reference_line):
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

    def read_meta_data(self, seed_initial_condition):
        if self.quantum_or_not:
            des = '../data/quantum/meta_data/' + self.network_type + '/' 
        else:
            des = '../data/classical/meta_data/' + self.network_type + '/' 
        filename = des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_reference={self.reference_line}_seed_initial={seed_initial_condition}.json'
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
        N_total = 0
        for seed_initial_condition in self.seed_initial_condition_list:
            all_above_intervals, all_above_time = self.get_above_interval_time(seed_initial_condition, tau)
            N_actual = len(all_above_time)
            N_total += N_actual
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
        y = count / N_total
        x = bins[:-1]
        ax.semilogy(x, y, '.-', label=label)
        #ax.xlabel('$\\tau / t$', fontsize=labelsize * 0.5)
        #ax.ylabel('$P(\\tau / t)$', fontsize=labelsize * 0.5)
        return collector

    def plot_signtime_initial_setup(self, tau, distribution_params_list):
        rows = 4
        cols = len(distribution_params_list) // 4
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(4 * cols, 3.5 * rows))
        for i, distribution_params in enumerate(distribution_params_list):
            ax = axes[i // cols, i % cols]
            simpleaxis(ax)
            self.distribution_params = distribution_params
            self.plot_sign_time_distribution(ax, tau, 'above')
            self.plot_sign_time_distribution(ax, tau, 'below')
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.set_title(title_name(distribution_params, self.quantum_or_not), size=labelsize*0.55, y=0.92)


        fig.delaxes(axes[-1, -1] )
        axes[-1, -2].legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.19, 0.19) ) 
        fig.text(x=0.03, y=0.5, horizontalalignment='center', s="$P(\\tau/t)$", size=labelsize*0.6, rotation=90)
        fig.text(x=0.5, y=0.04, horizontalalignment='center', s="$\\tau/t$", size=labelsize*0.6)
        fig.subplots_adjust(left=0.1, right=0.95, wspace=0.25, hspace=0.25, bottom=0.11, top=0.95)
        save_des = f'../transfer_figure/sign_time_quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}.png'
        plt.savefig(save_des, format='png')
        plt.close()

    def plot_signtime_disorder(self, tau, d_list, distribution_params_list):
        rows = len(d_list)
        cols = len(distribution_params_list) 
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(4 * cols, 3.5 * rows))
        for i, d in enumerate(d_list):
            for j, distribution_params in enumerate(distribution_params_list):
                ax = axes[i, j]
                simpleaxis(ax)
                self.distribution_params = distribution_params
                self.d = d
                self.plot_sign_time_distribution(ax, tau, 'above')
                self.plot_sign_time_distribution(ax, tau, 'below')
                ax.tick_params(axis='both', which='major', labelsize=15)
                ax.set_title(title_name(distribution_params, self.quantum_or_not, d), size=labelsize*0.55, y=0.92)


        axes[-1, -2].legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.19, 0.19) ) 
        fig.text(x=0.03, y=0.5, horizontalalignment='center', s="$P(\\tau/t)$", size=labelsize*0.6, rotation=90)
        fig.text(x=0.5, y=0.04, horizontalalignment='center', s="$\\tau/t$", size=labelsize*0.6)
        fig.subplots_adjust(left=0.1, right=0.95, wspace=0.25, hspace=0.25, bottom=0.11, top=0.95)
        save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}_seed={self.seed}_sign_time.png'
        plt.savefig(save_des, format='png')
        plt.close()
        




if __name__ == '__main__':
    quantum_or_not = False
    initial_setup = 'uniform_random'
    quantum_or_not = True
    network_type = '2D'
    network_type = '2D_disorder'
    N = 100
    d = 4
    seed = 0
    alpha = 1
    dt = 0.1
    reference_line = 'average'
    seed_initial_condition_list = np.arange(0, 10, 1)
    distribution_params = []
    pst = plotSignTime(quantum_or_not, network_type, N, d, seed, alpha, dt, initial_setup, distribution_params, seed_initial_condition_list, reference_line)
    tau = 1000
    rho_list = [[0, 1], [1/4, 3/4], [3/8, 5/8], [1, 1]]
    phase_list = [[-1, 1], [-1/2, 1/2], [-1/4, 1/4], [0, 0]]

    rho_list = [[1, 1]]
    phase_list = [[-1, 1], [-1/2, 1/2], [-1/4, 1/4]]
    distribution_params_raw = [rho + phase for rho in rho_list for phase in phase_list]
    distribution_params_list = []
    for i in distribution_params_raw:
        distribution_params_list.append([round(j, 3) for  j in i])


    #pst.plot_signtime_initial_setup(tau, distribution_params_list)
    d_list = [0.6, 0.7, 0.8, 0.9]
    pst.plot_signtime_disorder(tau, d_list, distribution_params_list)


