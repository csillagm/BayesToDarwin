import numpy as np
import datetime
from random import random


def neighbourmap(N, K):
    nkmap = []
    for x in range(N - K):
        nkmap.append(list(range(x, x + K)))
    for x in range(N - K, N):
        nkmap.append(list(range(x, N)) + list(range(K - N + x)))
    return nkmap


def generate_nklandscape(N_nk, K_nk, landscape_id, save_to_npz=False):
    landscape = []
    for x in range(N_nk):
        landscape.append({})
        for y in range(pow(2, K_nk)):
            landscape[x][y] = random()

    if save_to_npz:
        save_path = f'../landscape_data/landscape_files/{N_nk}-{K_nk}_{landscape_id}'
        np.savez(save_path, N=N_nk,
                 K=K_nk,
                 nk_landscape=landscape)
        print(f'NK landscape with parameters N={N_nk} and K={K_nk} saved to location {save_path}.npz')
    return landscape


def bitstring_to_fitness(bitstring, landscape, nkmap, N_nk, K_nk, alpha):
    if len(bitstring) != N_nk:
        assert False, "bitsrting should be %i-long" % N_nk

    s = ''
    summa = 0
    for x in range(N_nk):
        for y in range(K_nk):
            s += bitstring[nkmap[x][y]]
        summa += landscape[x][int(s, 2)]
        s = ''
    if alpha == -1:
        return summa / N_nk

    else:
        return np.exp(alpha*summa / N_nk)

if __name__ == '__main__':
    generate_nklandscape(20, 1, 'likelihood1', True)