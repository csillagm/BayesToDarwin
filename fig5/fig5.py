import numpy as np
from pathlib import Path
import seaborn as sns
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter


if __name__ == '__main__':

    N = 100
    tau = 1
    K = 1

    data_dir = Path(f'../results/fig5/fig5_N{N}_K{K}')
    figs_dir = Path(f'../results/fig5/{data_dir.name}_figs')
    figs_dir.mkdir(exist_ok=True, parents=True)

    do_smoothen = True

    mus = np.logspace(np.log10(1e-3), np.log10(0.5), 2)
    betas = np.logspace(np.log10(0.5), np.log10(50.1), 2)

    ks_matrix = np.zeros([len(mus), len(betas)])
    meanf_matrix = np.zeros([len(mus), len(betas)])
    Ts = np.zeros([len(mus), len(betas)])

    for mu_idx, mu in tqdm(enumerate(mus), total=len(mus)):
        for beta_idx, beta in enumerate(betas):

            files = glob(str(data_dir)+"/"+str(round(mu, 4))+"_"+str(round(beta, 4))+"_*.npz")
            data = np.load(files[0], allow_pickle=True)

            taus = data['taus']
            tau_ind = int(np.argwhere(np.array(taus)==tau))
            T_taus = data['T_taus']
            T = T_taus[tau_ind]

            ks_matrix[mu_idx, beta_idx] = data['ks_measure'][tau_ind]
            meanf_matrix[mu_idx, beta_idx] = data['f_l_maxes_info'][tau_ind]
            Ts[mu_idx, beta_idx] = T
    Ts = Ts[0]

    if do_smoothen:
        ks_matrix = gaussian_filter(ks_matrix, sigma=2)
        meanf_matrix = gaussian_filter(meanf_matrix, sigma=2)

    n_mu_ticks = 6
    mu_ticks = np.linspace(0, len(mus)-1, n_mu_ticks, dtype=int)
    mu_ticklabels = [round(mus[idx], 3) for idx in mu_ticks]
    mu_ticks = mu_ticks.astype(np.float) + 0.5

    n_beta_ticks = 5
    beta_ticks = np.linspace(0, len(betas)-1, n_beta_ticks, dtype=int)
    beta_ticklabels = [round(betas[idx]) for idx in beta_ticks]
    beta_ticks = beta_ticks.astype(np.float) + 0.5

    n_T_ticks = 5
    T_ticks = np.linspace(0, len(Ts)-1, n_T_ticks, dtype=int)
    T_ticklabels = [round(Ts[idx]) for idx in T_ticks]
    T_ticks = T_ticks.astype(np.float) + 0.5

    ax = sns.heatmap(ks_matrix[::-1, :], cmap='PiYG', vmin=0, vmax=1)
    ax2 = ax.twiny()
    ax2.set_xlim([0, ax.get_xlim()[1]])
    ax2.set_xticks(beta_ticks)
    ax2.set_xticklabels(beta_ticklabels, fontsize=16)
    ax2.tick_params(top=True)
    ax2.set_xlabel(r'selection strength $s$', fontsize=20, labelpad=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.set_yticks(mu_ticks)
    ax.set_yticklabels(mu_ticklabels[::-1], fontsize=16, rotation='horizontal')
    ax.set_xticks(T_ticks)
    ax.set_xticklabels(T_ticklabels, fontsize=16)
    ax.set_xlabel(r'number of generations $T$', fontsize=20)
    ax.set_ylabel(r'mutation rate $m$', fontsize=20)
    plt.tight_layout()

    if tau == 1:
        plt.savefig(figs_dir /f'fig5_ks_N{N}_tau{tau}_k{K}.png', format='png')
    plt.close()

    ax = sns.heatmap(meanf_matrix[::-1, :], cmap='coolwarm', vmax=0, vmin=17)
    ax.set_yticks(mu_ticks)
    ax.set_yticklabels(mu_ticklabels[::-1], fontsize=16, rotation='horizontal')
    ax2 = ax.twiny()
    ax2.set_xlim([0, ax.get_xlim()[1]])
    ax2.set_xticks(beta_ticks)
    ax2.set_xticklabels(beta_ticklabels, fontsize=16)
    ax2.tick_params(top=True)
    ax2.set_xlabel(r'selection strength $s$', fontsize=20, labelpad=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.set_xticks(T_ticks)
    ax.set_xticklabels(T_ticklabels, fontsize=16)
    ax.set_xlabel(r'number of generations $T$', fontsize=20)
    ax.set_ylabel(r'mutation rate $m$', fontsize=20)
    plt.tight_layout()

    if tau == 10:
        plt.savefig(figs_dir /f'fig5_optim_N{N}_tau{tau}_k{K}.png', format='png')
    plt.close()
