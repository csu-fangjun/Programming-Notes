#!/usr/bin/env python3

"""
See https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html

See also
https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/cells.py

The output of the cell is also used as the hidden state for the next time step.
Input shape is (N, input_size). Also, it is not bidirectional.

"""

import torch


@torch.no_grad()
def main():
    input_size = 3
    hidden_size = 5
    cell = torch.nn.RNNCell(
        input_size=input_size, hidden_size=hidden_size, bias=True, nonlinearity="tanh"
    )

    # input-hidden weight
    assert cell.weight_ih.shape == (hidden_size, input_size)

    # hidden-hidden weight
    assert cell.weight_hh.shape == (hidden_size, hidden_size)

    assert cell.bias_ih.shape == (hidden_size,)
    assert cell.bias_hh.shape == (hidden_size,)

    N = 3
    x = torch.rand(N, input_size)
    prev_h = torch.rand(N, hidden_size)
    h = cell(x, prev_h)
    assert h.shape == (N, hidden_size)

    igates = torch.mm(x, cell.weight_ih.t()) + cell.bias_ih
    hgates = torch.mm(prev_h, cell.weight_hh.t()) + cell.bias_hh
    expected_h1 = (igates + hgates).tanh()
    expected_h2 = torch.tanh(igates + hgates)

    assert torch.allclose(h, expected_h1)
    assert torch.allclose(h, expected_h2)
    print(cell)
    print(str(cell))


if __name__ == "__main__":
    torch.manual_seed(20211201)
    main()
