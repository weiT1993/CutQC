import numpy as np
import argparse


def cz_layer(nrow, ncol, starting_idx):
    # Store the qubit locations for this layer's CZs as an array of indices
    cz_list = []
    if starting_idx == 0 or starting_idx == 1:
        for row in range(0, nrow, 2):
            for col in range(0, ncol, 4):
                if col + starting_idx + 1 < ncol:
                    cz_list += [([row,col+starting_idx], [row,col+starting_idx+1])]
                if col + starting_idx + 2 + 1 < ncol and row+1<nrow:
                    cz_list += [([row+1,col+starting_idx+2], [row+1,col+starting_idx+3])]

    elif starting_idx == 2 or starting_idx == 3:
        for row in range(0, nrow, 2):
            for col in range(0, ncol, 4):
                if col + starting_idx - 2 + 1 < ncol and row+1<nrow:
                    cz_list += [([row+1,col+starting_idx-2], [row+1,col+starting_idx-1])]
                if col + starting_idx + 1 < ncol:
                    cz_list += [([row,col+starting_idx], [row,col+starting_idx+1])]

    elif starting_idx == 4 or starting_idx == 5:
        for col in range(0, ncol, 2):
            for row in range(0, nrow, 4):
                if row + starting_idx - 4 + 1 < nrow:
                    cz_list += [([row+starting_idx-4,col], [row+starting_idx-3,col])]
                if row + starting_idx - 4 + 2 + 1 < nrow and col+1<ncol:
                    cz_list += [([row+starting_idx-2,col+1], [row+starting_idx-1,col+1])]

    elif starting_idx == 6 or starting_idx == 7:
        for col in range(0, ncol, 2):
            for row in range(0, nrow, 4):
                if row + starting_idx - 4 - 2 + 1 < nrow and col+1<ncol:
                    cz_list += [([row+starting_idx-6,col+1], [row+starting_idx-5,col+1])]
                if row + starting_idx - 4 + 1 < nrow:
                    cz_list += [([row+starting_idx-4,col], [row+starting_idx-3,col])]

    return cz_list


def get_layers(n,m):
    """
    With the given n, m denoting a (n x m) matrix,
    return the corresponding layers of CZ gates that
    will enact a two qubit interaction between
    each pair of neighbors.
    """

    layers = []
    for i in range(8):
        cur_layer = cz_layer(n, m, i)
        layers += [cur_layer]

    return layers


def get_row_major(n, m, loc):
    return loc[0]*m + loc[1]


def main():

    n = 4
    m = 6
    layers = get_layers(n,m)
    print(layers)
    #for i, l in enumerate(layers):
    #    for cz in l:
    #        print('{} cz {} {}'.format(i,get_row_major(n,m,cz[0]),
    #                                   get_row_major(n,m,cz[1])))
    #    print('')

if __name__ == '__main__':
    main()
