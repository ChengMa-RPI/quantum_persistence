import numpy as np 
from scipy.integrate import odeint 
import matplotlib.pyplot as plt
import time 
import os
import pandas as pd
import multiprocessing as mp
from scipy.optimize import fsolve, root
import networkx as nx
import scipy.integrate as sin
import seaborn as sns
import sympy as sp
import itertools
import scipy.io

import imageio



def network_generate(network_type, N, beta, betaeffect, seed, d=None):
    """TODO: Docstring for network_generate.

    :arg1: TODO
    :returns: TODO

    """
    if network_type == 'complete':
        G = nx.complete_graph(N)
    elif network_type == '1D':
        G = nx.grid_graph(dim=[N], periodic=True)
    elif network_type == '2D':
        G = nx.grid_graph(dim=[int(np.sqrt(N)),int(np.sqrt(N))], periodic=True)
    elif network_type == '3D':
        G = nx.grid_graph(dim=[round(N**(1/3)), round(N**(1/3)), round(N**(1/3))], periodic=True)
    elif network_type == '2D_disorder' or network_type == '3D_disorder':
        if network_type == '2D_disorder':
            G = nx.grid_graph(dim=[int(np.sqrt(N)),int(np.sqrt(N))], periodic=True)
        if network_type == '3D_disorder':
            G = nx.grid_graph(dim=[round(N**(1/3)), round(N**(1/3)), round(N**(1/3))], periodic=True)
        A = nx.to_numpy_array(G)
        #t = time.time()
        #modifer = int((t-int(t)) * 1e9)
        modifer = 1000
        A = A * np.random.RandomState(seed + modifer).uniform(0, 1,  (N, N))  
        A = np.triu(A, 0) + np.triu(A, 1).transpose()
        A = np.array(A > (1-d), dtype=int)  # 1-d is the edge remove rate
        G = nx.from_numpy_matrix(A)
    elif network_type == 'RR':
        G = nx.random_regular_graph(d, N, seed)
    elif network_type == 'ER':
        #G = nx.fast_gnp_random_graph(N, d, seed)
        m = d
        G = nx.gnm_random_graph(N, m, seed)
    elif network_type == 'BA':
        m = d
        G = nx.barabasi_albert_graph(N, m, seed)
    elif network_type == 'SF':
        
        gamma, kmax, kmin = d
        G = generate_SF(N, seed, gamma, kmax, kmin)
        '''
        kmax = int(kmin * N ** (1/(gamma-1))) 
        probability = lambda k: (gamma - 1) * kmin**(gamma-1) * k**(-gamma)
        k_list = np.arange(kmin, 10 *kmax, 0.001)
        p_list = probability(k_list)
        p_list = p_list/np.sum(p_list)
        degree_seq = np.array(np.round(np.random.RandomState(seed=seed[0]).choice(k_list, size=N, p=p_list)), int)
        kmin, gamma, kmax = d
        degree_seq = np.array(((np.random.RandomState(seed[0]).pareto(gamma-1, N) + 1) * kmin), int)
        degree_max = np.max(degree_seq)
        if degree_max > kmax:
            degree_seq[degree_seq>kmax] = kmax
        else:
            degree_seq[degree_seq == degree_max] = kmax
        i = 0
        while np.sum(degree_seq)%2:
            i+=1
            #degree_seq[-1] = int(np.round(np.random.RandomState(seed=i).choice(k_list, size=1, p=p_list))) 
            degree_seq[-1] = int((np.random.RandomState(seed=N+i).pareto(gamma-1, 1) + 1) * kmin)
            degree_max = np.max(degree_seq)
            if degree_max > kmax:
                degree_seq[degree_seq>kmax] = kmax
            else:
                degree_seq[degree_seq == degree_max] = kmax

        G = nx.configuration_model(degree_seq, seed=seed[1])
        G = nx.Graph(G)  # remove parallel edges
        G.remove_edges_from(list(nx.selfloop_edges(G)))  # remove self loops (networkx version is not the newest one)
        '''
    elif network_type == 'star':
        G = nx.star_graph(N-1)
    elif network_type == 'RGG':
        G = nx.generators.geometric.random_geometric_graph(N, d, seed=seed)

    elif network_type == 'real':
        A, M, N = load_data(seed)
        #A = A_from_data(seed%2, M)
        A = np.heaviside(A, 0) # unweighted network
        G = nx.from_numpy_matrix(A)
    elif network_type == 'SBM_ER':
        N_group = N
        p = d
        G = nx.stochastic_block_model(N_group, p, seed=seed)

    elif network_type == 'degree_seq':
        G = nx.configuration_model(d, seed=seed)
 

    if nx.is_connected(G) == False:
        print('more than one component')
        G = G.subgraph(max(nx.connected_components(G), key=len))
    #A = np.array(nx.adjacency_matrix(G).todense()) 
    A = nx.to_numpy_array(G)
    if betaeffect:
        beta_eff, _ = betaspace(A, [0])
        weight = beta/ beta_eff
    else:
        weight = beta
    A = A * weight
    A_index = np.where(A>0)
    A_interaction = A[A_index]
    index_i = A_index[0] 
    index_j = A_index[1] 
    degree = np.sum(A>0, 1)
    cum_index = np.hstack((0, np.cumsum(degree)))
    return A, A_interaction, index_i, index_j, cum_index

def gif(data_des, file_type, file_range, save_des):
    """TODO: Docstring for gif.

    :des_data: TODO
    :des_save: TODO
    :returns: TODO

    """
    with imageio.get_writer(save_des + 'evolution.gif', mode='I') as writer:
        for i in file_range:

            filename = data_des + str(i) + file_type 
            image = imageio.imread(filename)
            writer.append_data(image)

def generate_powerlaw(N, seed, gamma, kmax, ktry, kmin):
    """generate scale free network with fixed N, kmin, kmax, kave, gamma

    :N: TODO
    :seed: TODO
    :d: TODO
    :returns: TODO

    """
    "generate degree sequence "
    N_predefine = N + 1
    while N_predefine > N:
        k = np.arange(kmin, ktry+1, 1)
        degree_hist = np.array(np.round((k/ktry) ** (-gamma)), dtype=int)
        degree_seq1 = np.hstack(([degree_hist[i] * [i+kmin] for i in range(np.size(degree_hist))]))
        N_predefine = int(np.size(degree_seq1))
        ktry = ktry - 1
    degree_seq2 = np.array(((np.random.RandomState(seed[0]).pareto(gamma-1, N-N_predefine - 1) + 1) * kmin), int)

    i= 0
    while np.size(degree_seq2) and np.max(degree_seq2) > ktry:
        i+=1
        large_index = np.where(degree_seq2>ktry)[0]
        degree_seq2[large_index] = np.array(((np.random.RandomState(seed[0]+i).pareto(gamma-1, np.size(large_index)) + 1) * kmin), int)

    degree_seq = np.hstack((degree_seq1, degree_seq2, kmax))
    i = 0
    while np.sum(degree_seq)%2:
        i+=1
        substitution = int((np.random.RandomState(seed=seed[0]+N+i).pareto(gamma-1, 1) + 1) * kmin)
        if substitution <= ktry:
            degree_seq[-2] = substitution

    degree_original = degree_seq.copy()

    G = nx.empty_graph(N)
    "generate scale free network using configuration model"
    #stublist = list(itertools.chain.from_iterable([n] * d for n, d in enumerate(degree_sequence))) 
    no_add = 0
    degree_change = 1
    j = 0
    while np.sum(degree_seq) and no_add < 10:

        stublist = nx.generators.degree_seq._to_stublist(degree_seq)
        M = len(stublist)//2  # the number of edges

        random_state = np.random.RandomState(seed[1] + j)
        random_state.shuffle(stublist)
        out_stublist, in_stublist = stublist[:M], stublist[M:]
        if degree_change == 0:
            no_add += 1
        else:
            no_add = 0
        G.add_edges_from(zip(out_stublist, in_stublist))

        G = nx.Graph(G)  # remove parallel edges
        G.remove_edges_from(list(nx.selfloop_edges(G)))  # remove self loops (networkx version is not the newest one)
        if nx.is_connected(G) == False:
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        degree_alive = np.array([G.degree[i] if i in G.nodes() else 0 for i in range(N)])
        degree_former = np.sum(degree_seq)
        degree_seq = degree_original - degree_alive
        degree_now = np.sum(degree_seq)
        degree_change = degree_now-degree_former
        j += 1
    return G

def generate_SF(N, seed, gamma, kmax, kmin):
    """generate scale-free network using configuration model with given gamma, kmin, kmax. 

    :N: TODO
    :seed: TODO
    :gamma: TODO
    :kmin: TODO
    :kmax: TODO
    :returns: TODO

    """
    p = lambda k: k ** (float(-gamma))
    k = np.arange(kmin, N, 1)
    pk = p(k) / np.sum(p(k))
    random_state = np.random.RandomState(seed[0])
    if kmax == N-1 or kmax == N-2:
        degree_seq = random_state.choice(k, size=N, p=pk)
    elif kmax == 0 or kmax == 1:
        degree_try = random_state.choice(k, size=1000000, p=pk)
        k_upper = int(np.sqrt(N * np.mean(degree_try)))
        k = np.arange(kmin, k_upper+1, 1)
        pk = p(k) /np.sum(p(k))
        degree_seq = random_state.choice(k, size=N, p=pk)

    i = 0
    while np.sum(degree_seq)%2:
        i+=1
        degree_seq[-1] = np.random.RandomState(seed=seed[0]+N+i).choice(k, size=1, p=pk)

    degree_original = degree_seq.copy()

    G = nx.empty_graph(N)
    "generate scale free network using configuration model"
    no_add = 0
    degree_change = 1
    j = 0
    while np.sum(degree_seq) and no_add < 10:

        stublist = nx.generators.degree_seq._to_stublist(degree_seq)
        M = len(stublist)//2  # the number of edges

        random_state = np.random.RandomState(seed[1] + j)
        random_state.shuffle(stublist)
        out_stublist, in_stublist = stublist[:M], stublist[M:]
        if degree_change == 0:
            no_add += 1
        else:
            no_add = 0
        G.add_edges_from(zip(out_stublist, in_stublist))

        G = nx.Graph(G)  # remove parallel edges
        G.remove_edges_from(list(nx.selfloop_edges(G)))  # remove self loops (networkx version is not the newest one)
        if nx.is_connected(G) == False:
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        degree_alive = np.array([G.degree[i] if i in G.nodes() else 0 for i in range(N)])
        degree_former = np.sum(degree_seq)
        degree_seq = degree_original - degree_alive
        degree_now = np.sum(degree_seq)
        degree_change = degree_now-degree_former
        j += 1
        if kmax == 1 or kmax == N-2:
            break
    return G

def disorder_lattice_clusters(network_type, N, seed, d):
    if network_type == '2D_disorder':
        G = nx.grid_graph(dim=[int(np.sqrt(N)),int(np.sqrt(N))], periodic=True)
    elif network_type == '3D_disorder':
        G = nx.grid_graph(dim=[round(N**(1/3)), round(N**(1/3)), round(N**(1/3))], periodic=True)
    A = nx.to_numpy_array(G)
    modifer = 1000
    A = A * np.random.RandomState(seed + modifer).uniform(0, 1,  (N, N))  
    A = np.triu(A, 0) + np.triu(A, 1).transpose()
    A = np.array(A > (1-d), dtype=int)  # 1-d is the edge remove rate
    G = nx.from_numpy_matrix(A)
    cluster = max(nx.connected_components(G), key=len)
    return cluster
