#!/usr/bin/env python3

import numpy as np


def main():
    x = [1, 10, 2, -3]
    N = len(x)
    assert N & (N - 1) == 0
    M = np.empty((N, N), dtype=np.complex)
    for r in range(0, N):
        for c in range(0, N):
            M[r, c] = np.exp(-1j * 2 * np.pi * r * c / N)

    y = np.matmul(M, x)
    print(y)
    '''
    [10.+0.0000000e+00j -1.-1.3000000e+01j -4.+3.6739404e-16j
     -1.+1.3000000e+01j]
    '''

    y2 = np.fft.fft(x, N)
    print(y2)
    '''
    [10. +0.j     -1.-13.j     -4. +0.j     -1.+13.j]
    '''

    x = np.array([x, x])
    print(x)
    '''
    [[ 1 10  2 -3]
     [ 1 10  2 -3]]
    '''

    # compute multiple 1-d fft simultaneously
    y3 = np.fft.fft(x, N, axis=-1)
    print(y3)
    '''
    [[10. +0.j -1.-13.j -4. +0.j -1.+13.j]
     [10. +0.j -1.-13.j -4. +0.j -1.+13.j]]
    '''


if __name__ == '__main__':
    main()
