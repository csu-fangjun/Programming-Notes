#!/usr/bin/env python3

# Author: Fangjun Kuang <csukuangfj@gmail.com>

import torch
import torch.nn as nn


@torch.no_grad()
def main():
    # with bias
    in_dim = 2
    out_dim = 3
    linear = nn.Linear(in_dim, out_dim, bias=True)
    state_dict = linear.state_dict()
    weight = state_dict['weight']
    bias = state_dict['bias']
    # Note that the weight shape is (out_dim, in_dim)
    assert weight.shape == (3, 2)
    assert bias.shape == (3,)

    # The input for linear is (N, *, in_dim)
    x = torch.rand(3, 4, 5, in_dim)
    y = linear(x)

    expected_y = torch.matmul(x, weight.t()) + bias
    assert torch.allclose(y, expected_y)


if __name__ == '__main__':
    torch.manual_seed(20210407)
    main()
