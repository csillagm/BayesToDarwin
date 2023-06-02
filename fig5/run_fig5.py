import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from base.bayesian_integration_actionselection import eval_param

if __name__ == '__main__':
    mus = np.logspace(np.log10(1e-3), np.log10(0.5), 2)
    betas = np.logspace(np.log10(0.5), np.log10(50.1), 2)

    alpha = 100

    with open('run_config_example.jsonnet') as config_file:
        config = json.load(config_file)

    for k in config.keys():
        N = config[k]['N']
        taus = config[k]['taus']
        n_runs = config[k]['n_runs']
        landscape_id = config[k]['landscape_id']
        save_dir = f'{config[k]["save_dir"]}/fig5_N{N}_K{landscape_id.split("-")[1][0]}'
        Path(save_dir).mkdir(exist_ok=True, parents=True)

        for mu in tqdm(mus):
            for beta in tqdm(betas):
                ks_measure, f_l_maxes, final_fitnesses_info, f_l_info_as, T_taus = eval_param(N, mu, beta, taus, n_runs, landscape_id, alpha)
                out_path = save_dir/Path(f'{np.round(mu,4)}_{np.round(beta,4)}_{landscape_id.replace("_","-")}.npz')

                np.savez(out_path,
		                 ks_measure=ks_measure,
		                 f_l_maxes_info=f_l_maxes,
		                 final_fitnesses_info=final_fitnesses_info,
		                 T_taus=T_taus,
		                 taus=taus)
