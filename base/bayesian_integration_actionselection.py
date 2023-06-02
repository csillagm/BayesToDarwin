import numpy as np
from pathlib import Path
from tqdm import tqdm

from base.nk_landscape import neighbourmap, bitstring_to_fitness
from base.infomeasure_fitness import information_measure
from base.utils import get_all_binaries, particle_filter, normalize, mutate_binary_bitwise, ideal_posterior_vector, \
    get_sampled_weights, get_cdf_values
from base.prior_sampling import mcmc

def eval_param(N, mu, beta, taus, n_runs, landscape_id, alpha):

    final_strings = np.empty([len(taus), n_runs], dtype='U20')
    final_maxfitnesses = np.zeros([len(taus), n_runs], dtype=float)
    final_fitnesses_as = np.zeros([len(taus), n_runs], dtype=float)
    final_fitnesses_population = np.zeros([len(taus), n_runs, N], dtype=float)
    sampled_values = np.zeros([len(taus), n_runs], dtype=float)

    ks_measure = np.zeros(len(taus), dtype=float)

    prior_landscape_id = '_prior'.join(landscape_id.split('_'))
    prior_values_path = Path(f'../landscape_data/prior_files/Prior_N-{N}_alpha-{alpha}_NK-{prior_landscape_id}.npz')
    if prior_values_path.exists():
        prior_file = np.load(prior_values_path, allow_pickle=True)
        prior_n_nk = int(prior_file["params"][3])
        prior_k_nk = int(prior_file["params"][4])
        mcmc_binaries = prior_file["binaries"]

    else:
        _, mcmc_binaries, _ = mcmc(N=N, m=1000, alpha=alpha, load_landscape=f'../landscape_data/landscape_files/{prior_landscape_id}.npz')
        mcmc_binaries = mcmc_binaries[:, -1]
        prior_n_nk, prior_k_nk = list(map(int, (landscape_id.split('_')[0].split('-'))))

    likelihood_landscape_id = '_likelihood'.join(landscape_id.split('_'))
    likelihood_landscape_file = np.load(f'../landscape_data/landscape_files/{likelihood_landscape_id}.npz', allow_pickle=True)
    likelihood_landscape = likelihood_landscape_file["nk_landscape"]
    n_nk = int(likelihood_landscape_file["N"])
    k_nk = int(likelihood_landscape_file["K"])

    if n_nk != prior_n_nk or k_nk != prior_k_nk:
        raise AssertionError("N_nk and K_nk values must match across the two landscapes")
    nk_map = neighbourmap(n_nk, k_nk)

    # set T
    T_taus = [int(tau * alpha / beta) for tau in taus]
    T = T_taus[-1]

    all_bitstrings = np.sort(get_all_binaries(n_nk, ["0", "1"]))

    # Evaluation of all binary strings on both landscapes and save results for future runs
    # If there is a saved npy file containing the values, the file is loaded and the evaluation is skipped.
    prior_values_path = Path(f'../landscape_data/landscape_values/evaluated_on_prior_{prior_landscape_id}.npy')
    likelihood_values_path = Path(f'../landscape_data/landscape_values/evaluated_on_likelihood_{likelihood_landscape_id}.npy')
    if prior_values_path.exists():
        prior_values = np.load(prior_values_path)
    else:
        prior_landscape = prior_file["nk_landscape"]
        prior_values = np.zeros(2**n_nk, dtype=float)
        print("Evaluation of all binaries on prior landscape...")
        for i, bitstring in tqdm(enumerate(all_bitstrings), total=len(all_bitstrings)):
            prior_values[i] = bitstring_to_fitness(bitstring, prior_landscape, nk_map, n_nk, k_nk, -1)
        np.save(prior_values_path, prior_values)

    if likelihood_values_path.exists():
        likelihood_values = np.load(likelihood_values_path)
    else:
        likelihood_values = np.zeros(2**n_nk, dtype=float)
        print("Evaluation of all binaries on prior landscape...")
        for i, bitstring in tqdm(enumerate(all_bitstrings), total=len(all_bitstrings)):
            likelihood_values[i] = bitstring_to_fitness(bitstring, likelihood_landscape, nk_map, n_nk, k_nk, -1)
        np.save(likelihood_values_path, likelihood_values)


    for run_i in range(n_runs):
        # Initialize arrays
        f_likelihood = np.zeros([N, T+1], dtype=float)
        binaries = np.empty([N, T+1], dtype=list)

        # Initialize binaries from prior MCMC
        binaries[:, 0] = mcmc_binaries
        # Initial evaluation
        for particle in range(N):
            f_likelihood[particle, 0] = bitstring_to_fitness(''.join(map(str, binaries[particle, 0])),
                                                             likelihood_landscape, nk_map, n_nk, k_nk, -1)

        # Run particle filter
        for generation in range(1, T+1):

            offspring = particle_filter(normalize(np.exp(f_likelihood[:, generation-1] * beta)))
            for i, ind in enumerate(offspring):
                binaries[i, generation] = mutate_binary_bitwise(binaries[ind, generation-1], mu)

            for particle in range(N):
                f_likelihood[particle, generation] = bitstring_to_fitness(
                    ''.join(map(str, binaries[particle, generation])),
                    likelihood_landscape, nk_map, n_nk, k_nk, -1)

            if generation in T_taus:
                tau_ind = int(np.argwhere(np.array(T_taus) == generation))

                final_maxfitnesses[tau_ind, run_i] = np.max(f_likelihood[:, generation])
                final_fitnesses_population[tau_ind, run_i, :] = f_likelihood[:, generation]
                sampled_i = np.random.choice(N, size=1)[0]
                sampled_binary = binaries[sampled_i, generation]
                final_strings[tau_ind, run_i] = ''.join([str(b) for b in sampled_binary])
                final_fitnesses_as[tau_ind, run_i] = f_likelihood[sampled_i, generation]

    likelihood_sort_idxs = np.argsort(likelihood_values)
    likelihood_values_sorted = likelihood_values[likelihood_sort_idxs]
    f_l_info_all = information_measure(likelihood_values_sorted, likelihood_values_sorted)
    f_l_info_population = information_measure(final_fitnesses_population, likelihood_values_sorted)
    f_l_info_as = information_measure(final_fitnesses_as, likelihood_values_sorted)
    f_l_info_max = information_measure(final_maxfitnesses, likelihood_values_sorted)
    f_l_maxes = np.mean(f_l_info_max, axis=1)

    # sort all arrays based on likelihood sorting
    all_bitstrings_sorted = all_bitstrings[likelihood_sort_idxs]
    prior_values_sorted = prior_values[likelihood_sort_idxs]

    # compute KS measure
    for tau_ind in range(len(taus)):
        exact_values = f_l_info_all
        exact_weights = ideal_posterior_vector(all_bitstrings_sorted, prior_values_sorted,
                                               likelihood_values_sorted,
                                               n_nk, alpha, beta, t=T_taus[tau_ind])
        exact_weights = normalize(np.exp(exact_weights))

        sampled_values[tau_ind] = np.sort(f_l_info_as[tau_ind, :])
        sampled_weights = get_sampled_weights(sampled_values[tau_ind])

        exact_cdf_values = get_cdf_values(exact_weights)
        sampled_cdf_values = get_cdf_values(sampled_weights)
        sampled_interp_cdf = np.interp(exact_values.astype(np.float64), sampled_values[tau_ind], sampled_cdf_values)

        ks_measure[tau_ind] = np.amax(np.abs(exact_cdf_values - sampled_interp_cdf))

    return ks_measure, f_l_maxes, f_l_info_population, f_l_info_as, T_taus



if __name__ == '__main__':

    N = 100
    mu = 0.01
    beta = 5
    alpha = 100
    taus = [1, 10]
    n_runs = 100
    landscape_id = '20-1_1'

    ks_measure, f_l_maxes, final_fitnesses_info, f_l_info_as, T_taus = \
        eval_param(N, mu, beta, taus, n_runs, landscape_id, alpha)
