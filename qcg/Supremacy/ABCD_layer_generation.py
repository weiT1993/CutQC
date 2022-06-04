import numpy as np
import argparse


def get_pattern_A(n,m):
    pattern = []
    for row in range(n):
        if row % 2 == 0:
            start_idx = 0
        else:
            start_idx = 1
        for col in np.arange(start_idx,m,2):
            if col != m-1:
                pattern += [([row,col],[row,col+1])]
    return pattern


def get_pattern_B(n,m):
    pattern = []
    for row in range(n):
        if row % 2 == 0:
            start_idx = 1
        else:
            start_idx = 0
        for col in np.arange(start_idx,m,2):
            if col != m-1:
                pattern += [([row,col],[row,col+1])]
    return pattern


def get_pattern_C(n,m):
    pattern = []
    for col in range(m):
        if col % 2 == 0:
            start_idx = 0
        else:
            start_idx = 1
        for row in np.arange(start_idx,n,2):
            if row != n-1:
                pattern += [([row,col],[row+1,col])]
    return pattern


def get_pattern_D(n,m):
    pattern = []
    for col in range(m):
        if col % 2 == 0:
            start_idx = 1
        else:
            start_idx = 0
        for row in np.arange(start_idx,n,2):
            if row != n-1:
                pattern += [([row,col],[row+1,col])]
    return pattern


def get_layers(n,m):
    """
    With the given n, m denoting a (n x m) matrix,
    return the corresponding A, B, C, and D layers of 2qb-gates that
    will enact a two qubit interaction between each pair of neighbors.

    A, B, C, and D layers are defined in Google's Supp. Info:
    (https://www.nature.com/articles/s41586-019-1666-5#Sec9)
    """
    return [get_pattern_A(n,m), get_pattern_B(n,m), get_pattern_C(n,m),
            get_pattern_D(n,m)]


def get_row_major(n, m, loc):
    return loc[0]*m + loc[1]


def main():

    n = 4
    m = 6
    layers = get_layers(n,m)
    for i, l in enumerate(layers):
        for cz in l:
            print('{} cz {} {}'.format(i,get_row_major(n,m,cz[0]),
                                       get_row_major(n,m,cz[1])))
        print('')

if __name__ == '__main__':
    main()
