import matplotlib.pyplot as plt
import numpy as np

quantum_type = 'quantum'
network_type = '1D'
d = 4
seed = 0
N = 10000
alpha = 1
initial_setup = 'rho_const_phase_uniform'
initial_setup = 'rho_uniform_phase_const_pi'
initial_setup = 'gaussian_wave'
initial_setup = 'rho_uniform_phase_uniform'
initial_setup = 'sum_sin_inphase'
initial_setup = 'sum_sin'
seed_initial = 0

def plot_samestep(quantum_type, network_type, d, seed, N, alpha, initial_setup, seed_initial, t_list, dt):
    des = '../data/' + quantum_type + '/state/' + network_type + '/' 
    data_file = des + f'N={N}_d={d}_seed={seed}_alpha={alpha}_dt={dt}_setup={initial_setup}_seed_initial={seed_initial}.npy'
    data = np.load(data_file)
    t, rho = data[:, 0], data[:, 1:]
    rho_ave = np.mean(rho)

    space = int(1/alpha)
    for ti in t_list:
        index = np.where( np.abs(t - ti) < 1e-5 )[0][0]
        plt.plot(np.arange(N)[::space] * alpha, rho[index][::space] - rho_ave, '.-', linewidth = 0.5, label=f't={ti}')
    plt.legend()
    plt.xlabel('x', fontsize=20)
    plt.ylabel('$\\rho(t)$', fontsize=20)

    des = f'../transfer_figure/quantum_type_N={N}_alpha={alpha}_dt={dt}_setup={initial_setup}_seed_initial_{seed_initial}_samestep'
    plt.title(f'N={N}_$\\Delta x$={alpha}', fontsize=20)
    plt.tight_layout()
    plt.savefig(des + '.png')
    plt.close()

def plot_sametime(quantum_type, network_type, d, seed, N_list, alpha_list, dt_list, initial_setup, seed_initial, t_plot):
    des = '../data/' + quantum_type + '/state/' + network_type + '/' 
    data_list = []
    for N, alpha, dt in zip(N_list, alpha_list, dt_list):

        data_file = des + f'N={N}_d={d}_seed={seed}_alpha={alpha}_dt={dt}_setup={initial_setup}_seed_initial={seed_initial}.npy'
        data = np.load(data_file)
        t, rho = data[:, 0], data[:, 1:]
        rho_ave = np.mean(rho)
        space = int(0.1/alpha)

        index = np.where( np.abs(t - t_plot) < 1e-5 )[0][0]
        data_list.append(data)
        plt.plot(np.arange(N)[::space] * alpha, (rho[index][::space] - rho_ave)/alpha, '.-', linewidth = 0.5, label=f'N={N}_dt={dt}')
        #plt.plot(rho[index][::space] / alpha , '.-', linewidth = 0.5, label=f'N={N}_dt={dt}')
    plt.legend()
    plt.xlabel('x', fontsize=20)
    plt.ylabel('$\\rho(t) - \\langle \\rho \\rangle $', fontsize=20)

    des = f'../transfer_figure/quantum_type_setup={initial_setup}_seed_initial_{seed_initial}_t={t_plot}_L={int(N*alpha)}_sametime'
    plt.title(f't={t_plot}', fontsize=20)
    plt.tight_layout()
    plt.savefig(des + '.png')
    plt.close()
    return data_list



N_list = [100, 1000, 10000]
alpha_list = [10, 1, 0.1]
t_list_list = [[100, 200, 500, 1000, 5000], [1, 2, 5, 10, 50], [0.01, 0.02, 0.05, 0.1, 0.5]]
dt_list = [100, 1, 0.01]
for N, alpha, t_list, dt in zip(N_list, alpha_list, t_list_list, dt_list):
    #plot_samestep(quantum_type, network_type, d, seed, N, alpha, initial_setup, seed_initial, t_list, dt)
    pass

N_list = [100, 1000, 10000]
alpha_list = [10, 1, 0.1]
dt_list = [10, 1, 0.01]

N_list = [100, 100, 1000]
alpha_list = [0.1, 0.1, 0.01]
dt_list = [0.1, 0.01, 0.01]

for t_plot in [0, 10, 50]:
    data_list = plot_sametime(quantum_type, network_type, d, seed, N_list, alpha_list, dt_list, initial_setup, seed_initial, t_plot)
