from functools import wraps
import inspect

def initializer(func):
    """
    Automatically assigns the parameters.

    >>> class process:
    ...     @initializer
    ...     def __init__(self, cmd, reachable=False, user='root'):
    ...         pass
    >>> p = process('halt', True)
    >>> p.cmd, p.reachable, p.user
    ('halt', True, 'root')
    """
    names, varargs, keywords, defaults = inspect.getargspec(func)

    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(names), reversed(defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kargs)

    return wrapper

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

class Plot_Dpp():
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


    def read_dpp(self):
        """TODO: Docstring for read_dpp.

        :arg1: TODO
        :returns: TODO

        """
        if self.quantum_or_not:
            des = '../data/quantum/persistence/' + self.network_type + '/' 
        else:
            des = '../data/classical/persistence/' + self.network_type + '/' 
        PA = []
        PB = []
        for seed_initial_condition in self.seed_initial_condition_list:
            filename = f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_setup={self.initial_setup}_reference={self.reference_line}_seed_initial={seed_initial_condition}.csv'
            data = np.array(pd.read_csv(des + filename, header=None))
            t, pa, pb = data[1:, 0], data[1:, 1], data[1:, 2]
            PA.append(pa)
            PB.append(pb)
        PA = np.vstack(( PA ))
        PB = np.vstack(( PB ))
        PA_ave = np.mean(PA, 0)
        PB_ave = np.mean(PB, 0)
        return t, PA_ave, PB_ave, PA_ave + PB_ave
        

    def plot_dpp_t(self):
        t, PA_ave, PB_ave, P_ave = self.read_dpp()
        plt.semilogy(t, PA_ave)
        plt.semilogy(t, PB_ave)
        return None

    def plot_dpp_t_N_list(self, ax, N_list, pa_or_pb):
        for N in N_list:
            self.N = N
            t, PA_ave, PB_ave, P_ave = self.read_dpp()
            index = np.where(t < 500)[0]
            t = t[index]
            PA_ave = PA_ave[index]
            PB_ave = PB_ave[index]
            P_ave = P_ave[index]
            if pa_or_pb == 'pa':
                #ax.loglog(t, PA_ave, label=f'N={N}')
                ax.semilogy(t, PA_ave, label=f'N={N}')
            elif pa_or_pb == 'pb':
                #ax.loglog(t, PB_ave, label=f'N={N}')
                ax.semilogy(t, PB_ave, label=f'N={N}')
            else:
                #ax.loglog(t, P_ave, label=f'N={N}')
                ax.semilogy(t, P_ave, label=f'N={N}')

            #plt.semilogx(t, PA_ave, label=f'N={N}')

    def plot_dpp_scaling(self, N_list, theta=0.186, z=1.99):
        fig, ax = plt.subplots(1, 1)
        simpleaxis(ax)
        for N in N_list:
            self.N = N
            t, PA_ave, PB_ave, P_ave = self.read_dpp()
            L = np.sqrt(N)
            lzt = L ** (theta * z)
            lz = L ** z
            y =  lzt * P_ave
            x = t / lz
            ax.loglog(x, y, label=f'N={N}')
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.03, 0.15) ) 
        fig.text(x=0.5, y=0.01, horizontalalignment='center', s="$t/L^{z}$", size=labelsize*0.5)
        fig.text(x=0.05, y=0.45, horizontalalignment='center', s="$L^{z\\theta}P(t, L)$", size=labelsize*0.5, rotation=90)
        fig.subplots_adjust(left=0.2, right=0.95, wspace=0.25, hspace=0.25, bottom=0.2, top=0.95)


    def plot_pa_pb_reference(self, N_list, reference_lines):
        cols = len(reference_lines)
        rows = 2
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(4 * cols, 3.5 * rows))
        for i in range(cols):
            self.reference_line = reference_lines[i]
            for j, pa_or_pb in enumerate(['pa', 'pb']):
                if cols == 1:
                    ax = axes[j]
                else:
                    ax = axes[j, i]
                simpleaxis(ax)
                self.plot_dpp_t_N_list(ax, N_list, pa_or_pb)
                ax.tick_params(axis='both', which='major', labelsize=13)
                #ax.set_ylim(7e-2, 1)
                if j == 0:
                    if self.reference_line == 'average':
                        title = '$r = \\frac{1}{N}$'
                    else:
                        title = f'$r = {self.reference_line}\\times$' + '$\\frac{1}{N}$'

                    ax.set_title(title, size=labelsize*0.5)
        #ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.23, 0.09) ) 
        ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.03, 0.09) ) 
        fig.text(x=0.5, y=0.01, horizontalalignment='center', s="$t$", size=labelsize*0.6)
        fig.text(x=0.05, y=0.75, horizontalalignment='center', s="$P_a$", size=labelsize*0.6, rotation=90)
        fig.text(x=0.05, y=0.3, horizontalalignment='center', s="$P_b$", size=labelsize*0.6, rotation=90)
        fig.subplots_adjust(left=0.1, right=0.95, wspace=0.25, hspace=0.25, bottom=0.1, top=0.95)
        #save_des = '../manuscript/dimension_reduction_v3_072422/' + self.dynamics + '_' + self.network_type + f'_tau_c_m.png'
        #plt.savefig(save_des, format='png')
        #plt.close()


    def plot_dpp_t_initial_setup(self, ax, N, pa_or_pb, label):
        t, PA_ave, PB_ave, P_ave = self.read_dpp()
        #t = t/ self.alpha **2
        #index = np.where(t < 500)[0]
        index = np.where(t <1e5)[0]
        t = t[index]
        PA_ave = PA_ave[index]
        PB_ave = PB_ave[index]
        P_ave = P_ave[index]
        if pa_or_pb == 'pa':
            #ax.semilogy(t, PA_ave, label=label)
            ax.loglog(t, PA_ave, label=label)
        elif pa_or_pb == 'pb':
            #ax.semilogy(t, PB_ave, label=label)
            ax.loglog(t, PB_ave, label=label)

        else:
            #ax.semilogy(t, P_ave, label=label)
            ax.loglog(t, P, label=label)


    def plot_pa_pb_initial_setup(self, N, initial_setup_list, titles):
        cols = 2
        rows = 1
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(4 * cols, 3.5 * rows))
        for j, pa_or_pb in enumerate(['pa', 'pb']):
            ax = axes[j]
            simpleaxis(ax)
            for initial_setup, label in zip(initial_setup_list, titles):
                self.initial_setup = initial_setup
                self.plot_dpp_t_initial_setup(ax, N, pa_or_pb, label)
                ax.tick_params(axis='both', which='major', labelsize=13)
                #ax.set_ylim(7e-2, 1)


        #ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.23, 0.09) ) 
        ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.09, 0.59) ) 
        fig.text(x=0.02, y=0.5, horizontalalignment='center', s="$P_a$", size=labelsize*0.6, rotation=90)
        fig.text(x=0.52, y=0.5, horizontalalignment='center', s="$P_b$", size=labelsize*0.6, rotation=90)
        fig.text(x=0.5, y=0.01, horizontalalignment='center', s="$t$", size=labelsize*0.6)
        fig.subplots_adjust(left=0.1, right=0.95, wspace=0.25, hspace=0.25, bottom=0.1, top=0.95)
        #save_des = '../manuscript/dimension_reduction_v3_072422/' + self.dynamics + '_' + self.network_type + f'_tau_c_m.png'
        #plt.savefig(save_des, format='png')
        #plt.close()

    def plot_pa_pb_alpha(self, N_list, alpha_list, num_realization_list, initial_setup_list):
        cols = 2
        rows = 1
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(4 * cols, 3.5 * rows))
        for j, pa_or_pb in enumerate(['pa', 'pb']):
            ax = axes[j]
            simpleaxis(ax)
            for N, alpha, num_realization in zip(N_list, alpha_list, num_realization_list):
                self.alpha = alpha
                self.seed_initial_condition_list = np.arange(num_realization)
                self.N = N
                label = f'N={N}'
                self.plot_dpp_t_initial_setup(ax, N, pa_or_pb, label)
                ax.tick_params(axis='both', which='major', labelsize=13)
                #ax.set_ylim(7e-2, 1)


        #ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.23, 0.09) ) 
        ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.09, 0.59) ) 
        fig.text(x=0.02, y=0.5, horizontalalignment='center', s="$P_a$", size=labelsize*0.6, rotation=90)
        fig.text(x=0.52, y=0.5, horizontalalignment='center', s="$P_b$", size=labelsize*0.6, rotation=90)
        #fig.text(x=0.5, y=0.01, horizontalalignment='center', s="$t / (\\Delta x) ^2$", size=labelsize*0.6)
        fig.text(x=0.5, y=0.01, horizontalalignment='center', s="$t$", size=labelsize*0.6)
        fig.subplots_adjust(left=0.1, right=0.95, wspace=0.25, hspace=0.25, bottom=0.2, top=0.95)
        #save_des = '../manuscript/dimension_reduction_v3_072422/' + self.dynamics + '_' + self.network_type + f'_tau_c_m.png'
        #plt.savefig(save_des, format='png')
        #plt.close()


        


if __name__ == '__main__':
    quantum_or_not = True
    initial_setup = 'rho_uniform_phase_const_pi'
    initial_setup = 'rho_const_phase_uniform'
    initial_setup = 'rho_uniform_phase_uniform'
    network_type = '1D'
    N = 10000
    d = 4
    seed = 0
    alpha = 1
    reference_line = 0.8
    reference_line = 'average'
    seed_initial_condition_list = np.arange(0, 10, 1)
    pdpp = Plot_Dpp(quantum_or_not, network_type, N, d, seed, alpha, initial_setup, seed_initial_condition_list, reference_line)
    #pdpp.plot_dpp_t()
    L_list = np.arange(10, 40, 10)
    N_list = np.power(L_list, 2)
    N_list = [100, 300, 500]

    #pdpp.plot_dpp_scaling(N_list)
    N_list = [10000]
    reference_lines = ['average']
    #pdpp.plot_pa_pb_reference(N_list, reference_lines)

    N = 10000
    initial_setup_list = ['rho_uniform_phase_uniform', 'rho_const_phase_uniform', 'rho_uniform_phase_const_pi']
    titles = ['uniform random', 'const $\\rho$', 'const $\\theta$']
    #pdpp.plot_pa_pb_initial_setup(N, initial_setup_list, titles)


    N_list = [100, 1000, 10000]
    alpha_list = [10, 1, 0.1]
    num_realization_list = [1000, 100, 10]
    pdpp.plot_pa_pb_alpha(N_list, alpha_list, num_realization_list, initial_setup)


