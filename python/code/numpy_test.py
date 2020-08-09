#!/usr/bin/env python3

import numpy as np


def test_repeat():
    a = np.array([
        [1],
        [2],
        [3],
    ])
    b = np.repeat(a, 3, axis=-1)
    print(b)
    np.testing.assert_array_equal(b, [
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
    ])


def test_pad():
    a = np.arange(48).reshape(4, 3, 4)
    b = np.pad(a, [[0, 2], [0, 0], [0, 0]])
    print(a)
    print(b)
    print(a.shape, b.shape)
    c = b.reshape((2, 3) + b.shape[1:])

    print(c.shape)
    print(c)


def test_tile():
    a = np.zeros((2, 1, 2))
    print(a.shape)
    print(a)

    b = np.tile(a, (1, 3, 1))
    print(b.shape)
    print(b)


def test_concatenate():
    a = np.arange(6).reshape(2, 3)

    b = np.concatenate([a, a], axis=0)
    assert b.shape == (4, 3)

    c = np.concatenate([a, a], axis=1)
    assert c.shape == (2, 6)


def test_sum():
    a = np.array([
        [1, 2, 3],
        [4, 5, 6],
    ])
    d = np.sum([a, a], axis=0)
    c = a / d
    print(c)


def main():
    #  test_repeat()
    #  test_pad()
    #  test_sum()
    #  test_tile()
    test_concatenate()


if __name__ == '__main__':
    main()
