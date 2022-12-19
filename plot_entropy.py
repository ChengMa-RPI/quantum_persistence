import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
sys.path.insert(1, '/home/mac6/RPI/research/')

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

def title_name(params, quantum_or_not):

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
        return rho_title  + '\n' + phase_title
    else:
        return rho_title  



class plotEntropy():
    def __init__(self, quantum_or_not, network_type, N, d, seed, alpha, dt, initial_setup, distribution_params, seed_initial_condition_list, reference_line, rho_or_phase, relative):
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
        self.relative = relative

    def read_phi(self, seed_initial_condition):
        if self.quantum_or_not:
            if self.rho_or_phase == 'rho':
                des = '../data/quantum/state/' + self.network_type + '/' 
            elif self.rho_or_phase == 'phase':
                des = '../data/quantum/phase/' + self.network_type + '/' 

        else:
            des = '../data/classical/state/' + self.network_type + '/' 
        save_file = des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_seed_initial={seed_initial_condition}.npy'
        data = np.load(save_file)
        t, state = data[:, 0], data[:, 1:]
        return t, state
    
    def plot_entropy_t(self, ax):
        entropy_list = []
        for seed_initial_condition in self.seed_initial_condition_list:
            t, state = self.read_phi(seed_initial_condition)
            if relative:
                entropy = -np.sum(state * np.log(state/ state[0]), 1)
            else:
                entropy = -np.sum(state * np.log(state), 1)
            entropy_list.append(entropy)
        ax.semilogx(t, np.mean(np.vstack((entropy_list)), 0))
        #ax.set_yscale('symlog')
        #plt.locator_params(axis='x', nbins=4)
        return None


    def plot_entropy_initial_setup(self, distribution_params_list):
        rows = 4
        cols = len(distribution_params_list) // rows
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(4 * cols, 3.5 * rows))
        for i, distribution_params in enumerate(distribution_params_list):
            title = title_name(distribution_params, self.quantum_or_not)
            #ax = axes[i]
            ax = axes[i // cols, i % cols]
            simpleaxis(ax)
            self.distribution_params = distribution_params
            self.plot_entropy_t(ax)
            ax.tick_params(axis='both', which='major', labelsize=13)
            ax.set_title(title, size=labelsize*0.5, y=0.92)

        #ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.23, 0.09) ) 

        if self.relative == True:
            ylabel =  '$S_{rel}$'
        else:
            ylabel = '$S_{abs}$'
        xlabel = '$t$'
        ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.19, -0.09) ) 
        fig.text(x=0.02, y=0.5, horizontalalignment='center', s=ylabel, size=labelsize*0.6, rotation=90)
        fig.text(x=0.5, y=0.04, horizontalalignment='center', s=xlabel, size=labelsize*0.6)
        fig.subplots_adjust(left=0.1, right=0.95, wspace=0.25, hspace=0.25, bottom=0.13, top=0.90)
        filename = f'quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}_{rho_or_phase}_entropy_{self.relative}.png'
        save_des = '../transfer_figure/' + filename
        plt.savefig(save_des, format='png')
        plt.close()



    def plot_correlation_t(self, ax, y):
        correlation = []
        for seed_initial_condition in self.seed_initial_condition_list:
            t, state = self.read_phi(seed_initial_condition)
            N_actual = len(state[0])
            cor_i = np.sum(state * np.roll(state, shift=y, axis=1), axis=1) / N_actual  - np.mean(state[0]) ** 2
            correlation.append(cor_i)
        ax.plot(t, np.mean(np.vstack((correlation)), 0))
        #ax.set_yscale('symlog')
        #plt.locator_params(axis='x', nbins=4)
        return None



    def plot_correlation_initial_setup(self, func, norm, distribution_params_list, t_list, y_list, direction):
        rows = 4
        cols = len(distribution_params_list) // rows
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(4 * cols, 3.5 * rows))
        for i, distribution_params in enumerate(distribution_params_list):
            title = title_name(distribution_params, self.quantum_or_not)
            #ax = axes[i]
            ax = axes[i // cols, i % cols]
            simpleaxis(ax)
            self.distribution_params = distribution_params
            self.plot_correlation_y_t(func, norm, ax, y_list, t_list, direction)

            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.set_title(title, size=labelsize*0.5, y=0.92)

        if norm == True:
            if self.rho_or_phase == 'phase':
                ylabel = '$g_{\\theta}'
            else:
                ylabel = '$g_{\\rho}'

        else:
            if self.rho_or_phase == 'phase':
                ylabel = '$G_{\\theta}'
            else:
                ylabel = '$G_{\\rho}'

        if func == 'y':
            xlabel = '$r$'
            ylabel += '(r)$'
        elif func == 't':
            xlabel = '$t$'
            ylabel += '(t)$'
        ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.19, -0.09) ) 
        fig.text(x=0.02, y=0.5, horizontalalignment='center', s=ylabel, size=labelsize*0.8, rotation=90)
        fig.text(x=0.5, y=0.03, horizontalalignment='center', s=xlabel, size=labelsize*0.8)
        fig.subplots_adjust(left=0.1, right=0.95, wspace=0.25, hspace=0.25, bottom=0.1, top=0.95)
        if func == 'y':
            filename = f'quantum={self.quantum_or_not}_network_={self.network_type}_N={self.N}_{rho_or_phase}_correlation_t={t_list}_norm={norm}'
        elif func == 't':
            filename = f'quantum={self.quantum_or_not}_network_={self.network_type}_N={self.N}_{rho_or_phase}_correlation_y={y_list}_norm={norm}'
        if self.network_type == '2D':
            filename += f'_direction={direction}'

        save_des = '../transfer_figure/' + filename + '.png'
        plt.savefig(save_des, format='png')
        plt.close()

    def plot_correlation_y_t(self, func, norm, ax, y_list, t_list, direction='x=0'):
        correlation = []
        for seed_initial_condition in self.seed_initial_condition_list:
            t, state = self.read_phi(seed_initial_condition)
            N_actual = len(state[0])
            if func == 'y':
                if self.network_type == '1D':
                    y_list = [i for i in range(10, N_actual, 10)]
                elif self.network_type == '2D':
                    y_list = [i for i in range(2, int(np.sqrt(N_actual)), 2)]
            elif func == 't':
                t_list = t[::10]
                
            t_index = [np.where(np.abs(np.array(t) - t_i) < 1e-5)[0][0] for t_i in t_list] 
            state = state[t_index]
            state_mean = np.mean(state, axis=1).reshape(len(state), 1)
            state_var = np.mean(state ** 2, axis=1).reshape(len(state), 1) - state_mean ** 2
            state = state - state_mean
            if self.network_type == '1D':
                state_stack = np.stack(( [np.roll(state, shift=y, axis=1) for y in y_list ] ))  # shift, t, N
                cor_i = np.mean(state * state_stack, axis=-1) 
            if self.network_type == '2D':
                state_2D = state.reshape( len(state), int(np.sqrt(self.N)), int(np.sqrt(self.N)) )
                if direction == 'x=0':
                    state_shift1 = np.stack(( [np.roll(state_2D, shift=y, axis=1) for y in y_list ] ))  # shift, t, sqrt(N), sqrt(N)
                    state_shift2 = np.stack(( [np.roll(state_2D, shift=y, axis=2) for y in y_list ] ))  # shift, t, sqrt(N), sqrt(N)
                    cor_i = 1/2*(np.mean(state_2D * state_shift1, axis=(-1, -2))  + np.mean(state_2D * state_shift2, axis=(-1, -2)) )
                elif direction == 'x=y' :
                    state_shift1 = np.stack(( [np.roll(np.roll(state_2D, shift=y, axis=1), shift=-y, axis=2) for y in y_list ] ))  # shift, t, sqrt(N), sqrt(N)
                    state_shift2 = np.stack(( [np.roll(np.roll(state_2D, shift=-y, axis=1), shift=-y, axis=2) for y in y_list ] ))  # shift, t, sqrt(N), sqrt(N)
                    cor_i = 1/2*(np.mean(state_2D * state_shift1, axis=(-1, -2))  + np.mean(state_2D * state_shift2, axis=(-1, -2)) )


            if norm == True:
                state_var = np.clip(state_var, 1e-14, None) 
                cor_i = cor_i / state_var.transpose()
            correlation.append(cor_i)
        cor_all = np.stack((correlation))
        cor_mean = np.mean(cor_all, axis=0)
        if func == 'y':
            for i, t_i in enumerate(t_list):
                ax.plot(y_list, cor_mean[:, i], label=f't={t_i}')
        if func == 't':
            for i, y_i in enumerate(y_list):
                ax.plot(t_list, cor_mean[i], label=f'r={y_i}')
        #ax.set_yscale('symlog')
        #plt.locator_params(axis='x', nbins=4)
        return None








if __name__ == '__main__':
    initial_setup = 'uniform_random'
    quantum_or_not = False
    quantum_or_not = True
    initial_setup = 'gaussian_wave'
    initial_setup = 'uniform_random'
    network_type = '2D'
    N = 10000
    d = 4
    seed = 0
    alpha = 1
    dt = 0.1
    reference_line = 'average'
    seed_initial_condition_list = np.arange(0, 10, 1)
    distribution_params = [1, 1, -1, 1]
    rho_or_phase = 'phase'
    rho_or_phase = 'rho'
    relative = True
    pent = plotEntropy(quantum_or_not, network_type, N, d, seed, alpha, dt, initial_setup, distribution_params, seed_initial_condition_list, reference_line, rho_or_phase, relative)

    rho_list = [[0, 1], [1/4, 3/4], [3/8, 5/8], [1, 1]]
    phase_list = [[-1, 1], [-1/2, 1/2], [-1/4, 1/4], [0, 0]]
    distribution_params_raw = [rho + phase for rho in rho_list for phase in phase_list]
    #distribution_params_raw = [rho for rho in rho_list]


    distribution_params_list = []
    for i in distribution_params_raw:
        distribution_params_list.append( [round(j, 3) for j in i])
    #pent.plot_entropy_initial_setup(distribution_params_list)


    y = 500


    t_lists = [[0], [5], [10], [50], [100], [500]]
    y_lists = [[0], [5], [10], [50], [100], [500]]

    t_lists = [[5, 10, 100, 500]]
    y_lists = [[2, 5, 10, 50]]
    norm = True
    direction = 'x=y'
    direction = 'x=0'
    for t_list in t_lists:
        pent.plot_correlation_initial_setup('y', norm, distribution_params_list, t_list, y_lists, direction)
        pass

    for y_list in y_lists:
        pent.plot_correlation_initial_setup('t', norm, distribution_params_list, t_lists, y_list, direction)
        pass
