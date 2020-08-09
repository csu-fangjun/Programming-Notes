#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from torch.nn.utils.weight_norm import remove_weight_norm


def main():
    linear = nn.Linear(in_features=2, out_features=3)
    state_dict = linear.state_dict()
    print(state_dict)

    weight = torch.tensor([
        [1, 0],
        [1, 1],
        [0, -1],
    ], dtype=torch.float32)
    bias = torch.tensor([10, 20, 30], dtype=torch.float32)

    state_dict['weight'] = weight
    state_dict['bias'] = bias
    linear.load_state_dict(state_dict)
    print('before weight norm', linear.state_dict())

    x = torch.tensor([1, 2], dtype=torch.float32)
    weight_norm(linear, 'weight')
    weight_norm(linear, 'bias')
    y = linear(x)
    print('after weight norm', linear.state_dict())
    print(y)

    pass


if __name__ == '__main__':
    torch.manual_seed(202004)
    main()
