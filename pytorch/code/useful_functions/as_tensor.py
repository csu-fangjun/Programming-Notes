#!/usr/bin/env python3

import torch


def test1():
    # memory is shared
    a = torch.tensor([1, 2])
    b = torch.as_tensor(a)
    a[0] = 10
    assert b[0] == 10


def test2():
    # memory is copied if torch.tensor() is used
    a = torch.tensor([1, 2])
    b = torch.tensor(a)  # Print a UserWarning
    a[0] = 10
    assert b[0] == 1


def test3():
    a = [1, 2]
    # there is memory copy since `a` is a list
    b = torch.as_tensor(a)
    a[0] = 10
    assert b[0] == 1


def test4():
    import numpy as np
    # memory is shared!
    a = np.array([1, 2])
    b = torch.as_tensor(a)
    a[0] = 10
    assert b[0] == 10


def main():
    test1()
    test2()
    test3()
    test4()


if __name__ == '__main__':
    main()
