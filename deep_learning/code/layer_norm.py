#!/usr/bin/env python3

# Author: Fangjun Kuang <csukuangfj@gmail.com>

import torch


@torch.no_grad()
def case1():
    # input shape [x1, x2, x3, ...]
    #  input_shape = torch.Size([2, 3, 4, 5, 6])
    input_shape = torch.Size([2, 3, 4])
    num = 2
    eps = 1e-5
    ln = torch.nn.LayerNorm(input_shape[-num:],
                            eps=eps,
                            elementwise_affine=True)
    state_dict = ln.state_dict()

    assert state_dict['weight'].ndim == num
    assert state_dict['weight'].shape == input_shape[-num:]

    assert state_dict['bias'].ndim == num
    assert state_dict['bias'].shape == input_shape[-num:]

    weight = torch.randn(input_shape[-num:])
    bias = torch.randn(input_shape[-num:])

    state_dict['weight'] = weight
    state_dict['bias'] = bias
    ln.load_state_dict(state_dict)

    x = torch.rand(input_shape)
    y = ln(x)
    assert x.shape == y.shape

    d = x.view(-1, *input_shape[-num:])
    expected_y = torch.empty_like(d)
    for i, row in enumerate(d):
        mu = row.mean()
        var = (row - mu).square().mean()
        stddev = torch.sqrt(var + eps)
        expected_y[i] = (row - mu) / stddev * weight + bias
    expected_y = expected_y.reshape(y.shape)
    assert torch.allclose(y, expected_y, atol=1e-6)


if __name__ == '__main__':
    torch.manual_seed(20210402)
    case1()
