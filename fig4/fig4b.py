import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import cm

from base.infomeasure_fitness import information_measure
from base.utils import ideal_posterior_vector, normalize

def get_information_measure_weighted_by_likelihood(landscape_id, alpha, beta, t):
    [n_nk, k_nk, nk_id] = landscape_id.replace('_', '-').split('-')
    all_likelihood_values = np.load(f'../landscape_data/landscape_values/evaluated_on_likelihood_{n_nk}-{k_nk}_likelihood{nk_id}.npy')
    all_prior_values = np.load(f'../landscape_data/landscape_values/evaluated_on_prior_{n_nk}-{k_nk}_prior{nk_id}.npy')

    sort_indices = np.argsort(all_likelihood_values)
    all_likelihood_values_sorted = all_likelihood_values[sort_indices]
    all_prior_values_sorted = all_prior_values[sort_indices]

    exact_values = ideal_posterior_vector([], all_prior_values_sorted, all_likelihood_values_sorted, n_nk, alpha, beta, t)
    exact_values = np.exp(exact_values)
    exact_values_normalized = normalize(exact_values)

    likelihood_i = information_measure(all_likelihood_values_sorted, all_likelihood_values_sorted)

    exact_mean = np.average(likelihood_i, weights=exact_values_normalized)
    variance = np.average((likelihood_i-exact_mean)**2, weights=exact_values_normalized)
    exact_std = np.sqrt(variance)

    return exact_mean, exact_std


if __name__ == '__main__':

    mus = np.logspace(np.log10(1e-3), np.log10(0.5), 2)
    betas = np.logspace(np.log10(0.5), np.log10(50.1), 2)
    tau = 1
    xmax = 21

    landscape_id = '20-1_1'

    runs_dir = Path(f'../results/fig4_{landscape_id}')
    figs_dir = Path(f'../results/{runs_dir.name}_figs')
    figs_dir.mkdir(parents=True, exist_ok=True)
    all_filepaths = list(runs_dir.rglob('*.npz'))
    filepaths = []

    means = []
    stds = []
    kss = []
    exact_means = []
    exact_stds = []
    all_mus = []
    all_betas = []

    p_exacts_prev = None
    for beta in tqdm(betas):
        for mu in mus:

            paths = list(runs_dir.rglob(f"{str(round(mu, 4))}_{str(round(beta, 4))}_*.npz"))
            paths = np.sort(paths)

            mb_means = []
            mb_stds = []
            mb_kss = []

            for filepath in paths:
                data = np.load(filepath, allow_pickle=True)
                taus = data['taus']
                T_taus = data['T_taus']
                tau_ind = int(np.argwhere(np.array(taus) == tau))

                fitness = data['f_l_info_as']
                mb_means.append(np.average(fitness, axis=1)[tau_ind])
                mb_stds.append(np.std(fitness, axis=1)[tau_ind])
                ks = data['ks_measure']
                mb_kss.append(ks[tau_ind])

            means.append(np.average(mb_means))
            stds.append(np.average(mb_stds))
            kss.append(np.average(mb_kss))
            all_mus.append(mu)
            all_betas.append(beta)

        exact_mean, exact_std = get_information_measure_weighted_by_likelihood(landscape_id, 100, beta, T_taus[tau_ind])
        exact_means.append(exact_mean)
        exact_stds.append(exact_std)

    fig = plt.figure()
    ax = plt.gca()
    sc = ax.scatter(means, stds, c=kss, cmap='PiYG', norm=cm.colors.Normalize(0.15, 1))
    ax.scatter(exact_means, exact_stds, c='cyan', marker="*", s=600, edgecolor='black', linewidth=2)
    ax.set_ylabel(r'std of fitness $\sigma$', fontsize=20)
    ax.set_xlabel(r'mean fitness $\mu$', fontsize=20)
    ax.set_xlim([0, xmax])
    ax.set_yticks([0, 3, 6])
    xticks = np.arange(0, xmax, step=5)
    ax.set_xticks(xticks)
    ax.set_yticklabels([0, 3, 6], fontsize=15)
    ax.set_xticklabels(xticks, fontsize=15)
    cbar = plt.colorbar(sc)
    cbar.set_ticks([0.3, 0.6, 0.9])
    cbar.ax.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig(figs_dir / f'fig4b_tau{tau}.png', format='png', bbox_inches='tight')
    #plt.show()