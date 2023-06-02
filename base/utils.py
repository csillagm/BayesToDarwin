import numpy as np
import random


def normalize(x):
    if sum(x)==0:
        return 0
    else:
        return x/sum(x)


def particle_filter(weights):

    '''
    Given normalized weights, returns offspring indices according to particle filter process.

    :param weights: Normalized array of fitness/weights
    :return: array of sampled indices
    '''

    sample = np.random.multinomial(len(weights), weights)
    indices = []

    for ind, x in enumerate(sample):
        for i in range(x):
            indices.append(ind)

    return np.array(indices)


def mutate_binary(binary, mutation_rate):

    if np.random.random() <= mutation_rate:
        idx_to_change = np.random.choice(np.arange(len(binary)))
        out_binary = binary.copy()
        out_binary[idx_to_change] = abs(binary[idx_to_change]-1)
    else:
        out_binary = binary

    return out_binary


def mutate_binary_bitwise(binary, mutation_rate):

    out_binary = binary.copy()

    for idx, bit in enumerate(out_binary):
        if random.random() <= mutation_rate:
            out_binary[idx] = abs(bit - 1)
    return out_binary #, len(indices_to_change)


def get_all_binaries(N, l):
    """
    Generates all possible strings of length n, containing characters in l

    Parameters
    ----------
    N - length of strings
    l - array of strings, a character which are to be used to combine,
        use ['0','1'] for binary strings

    Returns
    -------
    list of all possible strings
    """

    if N == 1:
        return l
    else:
        if len(l) == 0:
            return get_all_binaries(N - 1, ["0", "1"])
        else:
            return get_all_binaries(N - 1, [i + "0" for i in l] + [i + "1" for i in l])



def ideal_posterior_vector(all_bitstrings, evaluated_on_prior, evaluated_on_likelihood, n_nk, alpha, beta, t):

    ideal_posterior_all = evaluated_on_prior*alpha + evaluated_on_likelihood*beta*t

    return ideal_posterior_all.astype(np.float128)


def get_sampled_weights(sampled_values):

    vals, counts = np.unique(sampled_values, return_counts=True)
    sampled_count_dict = dict(zip(vals, counts))
    sampled_weights = []
    for sampled_v in sampled_values:
        sampled_weights.append(sampled_count_dict[sampled_v])

    return normalize(sampled_weights)


def get_cdf_values(weights):

    cdf_values = np.zeros(len(weights), dtype=np.float)
    for i in range(len(weights)):
        if i == 0:
            cdf_values[i] = weights[i]
        else:
            cdf_values[i] = cdf_values[i-1] + weights[i]

    return cdf_values

