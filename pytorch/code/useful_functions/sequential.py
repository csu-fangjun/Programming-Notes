#!/usr/bin/env python3

import torch
import torch.nn as nn


def test1():
    # from an iterable
    a = [nn.ReLU(), nn.Linear(2, 3), nn.ReLU()]
    model = nn.Sequential(*a)
    print(model)
    '''
    Sequential(
      (0): ReLU()
      (1): Linear(in_features=2, out_features=3, bias=True)
      (2): ReLU()
    )
    '''


def test2():
    from collections import OrderedDict
    a = OrderedDict()
    a['a'] = nn.ReLU()
    a['b'] = nn.Linear(2, 3)
    a['c'] = nn.ReLU()
    model = nn.Sequential(a)
    print(model)
    '''
    Sequential(
      (a): ReLU()
      (b): Linear(in_features=2, out_features=3, bias=True)
      (c): ReLU()
    )
    '''


def main():
    test1()
    test2()


if __name__ == '__main__':
    main()
