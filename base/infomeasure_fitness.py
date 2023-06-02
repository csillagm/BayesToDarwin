import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from base.nk_landscape import bitstring_to_fitness, neighbourmap

def generate_binary(n, l):
    """
    Generates all possible strings of length n, containing characters in l

    Parameters
    ----------
    n - length of strings
    l - array of strings, a character which are to be used to combine,
        use ['0','1'] for binary strings

    Returns
    -------
    list of all possible strings
    """

    if n == 1:
        return l
    else:
        if len(l) == 0:
            return generate_binary(n-1, ["0", "1"])
        else:
            return generate_binary(n-1, [i + "0" for i in l] + [i + "1" for i in l])



def all_values_of_landscape(landscape_filename, return_sorted=False):
    """
    Calculates each possible value of a given NK-landscape

    Parameters
    ----------
    landscape - array containing the values of component functions in NK landscape
    return_sorted - Boolean, whether to return sorted array of values

    Returns
    -------
    array of all the values
    """

    # if loads file
    landscape_file = np.load(landscape_filename, allow_pickle=True)

    N = landscape_file['N']
    K = landscape_file['K']
    landscape = landscape_file['nk_landscape']

    #N = len(landscape)
    #K = int(np.log2(len(landscape[0])))

    print(N)
    print(K)

    nkmap = neighbourmap(N, K)

    all_bitstrings = generate_binary(N, ['0', '1'])
    all_values = np.empty(2**N, dtype=float)

    for ind, bitsring in tqdm(enumerate(all_bitstrings), total=len(all_bitstrings)):
        all_values[ind] = bitstring_to_fitness(bitsring, landscape, nkmap, N, K, -1)

    #np.save(str(landscape_filename)+"_all_values", all_values)

    if return_sorted == True:
        all_values.sort()
        return all_values

    elif return_sorted == False:
        return all_values


def information_measure(fitness, landscape_values):
    '''
    Transform absolute fitness values to information measure.
    Where information(f) = - log_2(p(f)).
    Here p(f) is the probability of randomly sampling a k point on the landscape such that k >= f.

    Parameters
    ----------
    fitness - matrix of absolute fitness values
    landscape_values - array of all values of NK landscape

    Returns
    -------
    information measure matrix, same shape az input fitness matrix
    '''

    landscape_size = len(landscape_values)
    original_shape = fitness.shape
    fitness = fitness.flatten()

    p_f = np.empty(len(fitness), dtype=float)

    for ind, f in enumerate(fitness):
        p_f[ind] = (landscape_size - np.searchsorted(landscape_values,f)) / landscape_size


    info_measure = -np.log2(p_f).reshape(original_shape)

    return info_measure


def attach_info_measure_to_data(filename):

    data = np.load(filename, allow_pickle=True)

    params = data["params"]
    binaries = data["binaries"]
    f_prior = data["f_prior"]
    f_posterior_sampled = data["f_posterior_sampled"]
    f_posterior_ideal = data["f_posterior_ideal"]
    f_likelihood = data['f_likelihood']

    landscape_file = params[-1]

    landscape_values = all_values_of_landscape(landscape_file, return_sorted=True)
    fitness_measure = information_measure(f_likelihood, landscape_values)

    outfile = filename.replace(".npz", "_mod.npz")

    np.savez(outfile, params=params,
             f_prior=f_prior,
             f_likelihood=f_likelihood,
             f_posterior_sampled=f_posterior_sampled,
             f_posterior_ideal=f_posterior_ideal,
             binaries=binaries,
             f_likelihood_infomeasure=fitness_measure)

    print(outfile, "saved")


def attach_info_measure_to_data_heatmap(filename, landscape_values=None):

    data = np.load(filename, allow_pickle=True)

    params = data["params"]
    f_likelihood = data['f_likelihood']
    n_binaries = data['n_binaries']
    ks_measure_es = data['ks_measure_es']
    taus = data['taus']
    T_taus = data['T_taus']

    if landscape_values is None:
        landscape_file_path = params[-1]
        landscape_values = all_values_of_landscape(landscape_file_path, return_sorted=True)
    fitness_measure = information_measure(f_likelihood, landscape_values)

    parts = filename.split('/')
    #parts[-1] = '3' + parts[-1]
    outfilename = '/'.join(parts)

    np.savez(outfilename,
             params=params,
             f_likelihood=f_likelihood,
             n_binaries=n_binaries,
             ks_measure_es=ks_measure_es,
             f_likelihood_infomeasure=fitness_measure,
             taus=taus,
             T_taus=T_taus)

    #print(outfile, "saved")

#landscape_values = all_values_of_landscape("20-3_landscape_prior1.npz")
#
# fitness = np.random.choice(landscape_values, [3,3], replace=False)
# print(fitness)
#
# info_test = information_measure(fitness, landscape_values)
#
# print(info_test)

# attach_info_measure_to_data("100-20-10-3-0523-0051.npz")
