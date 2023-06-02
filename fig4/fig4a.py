import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import cm



if __name__ == '__main__':

    mus = np.logspace(np.log10(1e-3), np.log10(0.5), 2)
    betas = np.logspace(np.log10(0.5), np.log10(50.1), 2)
    tau = 1
    size_scaler = 1.2 # tau1 N2: 2.3, tau10 N2: 1.5, tau1 N100: 1.5, tau10 N100: 1.15
    landscape_id = '20-1_1'

    runs_dir = Path(f'../results/fig4_{landscape_id}')
    figs_dir = Path(f'../results/{runs_dir.name}_figs')
    figs_dir.mkdir(parents=True, exist_ok=True)
    all_filepaths = list(runs_dir.rglob('*.npz'))
    filepaths = []

    mu_points = []
    beta_points = []
    means = []
    stds = []

    # drop filepaths - keep only 1 for each beta, mu combo
    for beta in tqdm(betas):
        for mu in mus:
            paths = list(runs_dir.rglob(f"{str(round(mu, 4))}_{str(round(beta, 4))}_*.npz"))
            paths = np.sort(paths)

            mu_points.append(float(mu))
            beta_points.append(float(beta))
            mb_means = []
            mb_stds = []

            for filepath in paths:
                data = np.load(filepath, allow_pickle=True)
                taus = data['taus']
                T_taus = data['T_taus']
                tau_ind = int(np.argwhere(np.array(taus) == tau))

                fitness = data['f_l_info_as']
                mb_means.append(np.average(fitness, axis=1)[tau_ind])
                mb_stds.append(np.std(fitness, axis=1)[tau_ind])

            means.append(np.average(mb_means))
            stds.append(np.average(mb_stds))

    ss = [np.exp(size_scaler*e) for e in stds]
    fig = plt.figure()
    ax = plt.gca()
    sc = ax.scatter(beta_points, mu_points, s=ss, c=means, cmap='coolwarm', norm=cm.colors.Normalize(0, 17))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'mutation rate $m$', fontsize=20)
    ax.set_xlabel(r'selection strength $s$', fontsize=20)
    cbar = plt.colorbar(sc)
    cbar.set_ticks([1, 5, 10, 15, 20])
    cbar.ax.set_ylim([0, 17])
    cbar.ax.tick_params(labelsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    sorted_es = np.sort(stds)
    sorted_sizes = [np.exp(size_scaler*e) for e in sorted_es]
    legends = [1, 2, 3, 4]
    legend_sizes = [np.exp(size_scaler*l) for l in legends]

    marker1 = plt.scatter([],[], s=legend_sizes[0], color='black')
    marker2 = plt.scatter([],[], s=legend_sizes[1], color='black')
    marker3 = plt.scatter([],[], s=legend_sizes[2], color='black')
    marker4 = plt.scatter([],[], s=legend_sizes[3], color='black')

    legend_markers = [marker1, marker2, marker3, marker4]

    labels = [str(round(legends[0])),
              str(round(legends[1])),
              str(round(legends[2])),
              str(round(legends[3]))]

    leg = plt.legend(legend_markers, labels, ncol=1, frameon=True, fontsize=14,
                     handlelength=2, loc='center right',
                     scatterpoints=1, bbox_to_anchor=[1.5, 0.5])

    plt.savefig(figs_dir / f'fig4a_tau{tau}.png', format='png', bbox_extra_artists = (leg,), bbox_inches='tight')
    plt.show()
