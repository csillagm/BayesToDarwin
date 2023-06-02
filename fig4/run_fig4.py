import numpy as np
from pathlib import Path
from tqdm import tqdm

from base.bayesian_integration_actionselection import eval_param

if __name__ == '__main__':
    mus = np.logspace(np.log10(1e-3), np.log10(0.5), 2)
    betas = np.logspace(np.log10(0.5), np.log10(50.1), 2)

    alpha = 100

    n_repeats = 5

    for rep in range(1, n_repeats+1):
        N = 100
        taus = [1, 10]
        n_runs = 100
        landscape_id = "20-1_1"
        save_dir = f"../results/fig4_{landscape_id}"
        Path(save_dir).mkdir(exist_ok=True, parents=True)

        for mu in tqdm(mus):
            for beta in tqdm(betas):
                ks_measure, f_l_maxes, f_l_info_population, f_l_info_as, T_taus = \
                    eval_param(N, mu, beta, taus, n_runs, landscape_id, alpha)
                out_path = save_dir/Path(f'{np.round(mu,4)}_{np.round(beta,4)}_{landscape_id.replace("_","-")}_{rep}.npz')
                np.savez(out_path,
                         ks_measure=ks_measure,
                         f_l_info_population=f_l_info_population,
                         f_l_info_as=f_l_info_as,
                         T_taus=T_taus,
                         taus=taus)
