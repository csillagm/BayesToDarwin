import numpy as np
from tqdm import tqdm

from base.nk_landscape import neighbourmap, bitstring_to_fitness
from base.utils import mutate_binary


def mcmc(N, m, alpha, load_landscape, save_samples=True):

    if isinstance(load_landscape, str):
        landscape_file = np.load(load_landscape, allow_pickle=True)
        landscape = landscape_file["nk_landscape"]
        n_nk = landscape_file["N"]
        k_nk = landscape_file["K"]

    else:
        landscape = load_landscape
        n_nk = len(landscape)
        k_nk = int(np.log2(len(landscape[0])))

    nk_map = neighbourmap(n_nk, k_nk)

    binaries = np.empty([N, m + 1], dtype=list)
    fitness = np.zeros([N, m + 1], dtype=float)
    acceptance_rate = np.empty(m, dtype=float)

    for particle in range(N):
        binaries[particle, 0] = np.random.randint(0, 2, n_nk)
        fitness[particle, 0] = bitstring_to_fitness(''.join(map(str, binaries[particle, 0])),
                                                    landscape, nk_map, n_nk, k_nk, alpha)

    n_accepted = 0

    print("Running MCMC sampling...")
    for step in range(1, m+1):

        if step % 100000 == 0:
            print(step)

        for particle in range(N):
            last_binary = binaries[particle, step - 1]
            last_fitness = fitness[particle, step - 1]
            candidate = mutate_binary(last_binary, mutation_rate=1)
            candidate_fitness = bitstring_to_fitness(''.join(map(str, candidate)),
                                                        landscape, nk_map, n_nk, k_nk, alpha)

            acceptance_ratio = candidate_fitness / last_fitness
            threshold = np.random.random()

            if acceptance_ratio > threshold:
                binaries[particle, step] = candidate
                fitness[particle, step] = candidate_fitness
                n_accepted += 1

            else:
                binaries[particle, step] = last_binary
                fitness[particle, step] = last_fitness

        acceptance_rate[step-1] = n_accepted / (step * N)

    if save_samples:
        landscape_id = load_landscape.split('/')[-1].strip('.npz').split('_')[-1]
        save_path = f'../landscape_data/prior_files/Prior_N-{N}_alpha-{alpha}_NK-{n_nk}-{k_nk}_{landscape_id}'

        np.savez(save_path,
                 params=[N, m, alpha, n_nk, k_nk, load_landscape],
                 nk_landscape=landscape,
                 binaries=binaries[:, -1])
        print(f'Prior sampling result saved at location {save_path}.npz')

    return fitness, binaries, acceptance_rate


if __name__ == '__main__':
    landscape_id = '20-1_prior1'
    fitness, binaries, acceptance_rate = mcmc(N = 100, m = 1000, alpha=100, load_landscape=f'../landscape_data/landscape_files/{landscape_id}.npz')
