#!/usr/bin/env python3

"""
See https://pytorch.org/docs/stable/generated/torch.nn.RNN.html#torch.nn.RNN

See also
https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/RNN.cpp#L910

It returns only the hidden states of the last steps of all layers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def test_single_layer():
    input_size = 2
    hidden_size = 3
    rnn = nn.RNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        nonlinearity="tanh",
        bias=True,
        batch_first=True,
        dropout=0,
        bidirectional=False,
    )
    assert rnn.weight_ih_l0.shape == (hidden_size, input_size)
    assert rnn.bias_ih_l0.shape == (hidden_size,)

    assert rnn.weight_hh_l0.shape == (hidden_size, hidden_size)
    assert rnn.bias_hh_l0.shape == (hidden_size,)

    N = 2
    T = 3
    x = torch.rand(N, T, input_size)
    hx = torch.rand(1, N, hidden_size)  # 1 means 1 layer

    y, hy = rnn(x, hx)
    assert y.shape == (N, T, hidden_size)
    assert hy.shape == (1, N, hidden_size)

    hyy = hx[0]
    yy = []
    for i in range(T):
        input_x = x[:, i]
        t = F.linear(input_x, rnn.weight_ih_l0, rnn.bias_ih_l0) + F.linear(
            hyy, rnn.weight_hh_l0, rnn.bias_hh_l0
        )
        hyy = t.tanh()
        yy.append(hyy)
    hyy = hyy.unsqueeze(0)

    yy = torch.stack(yy, dim=1)
    assert torch.allclose(y, yy)
    assert torch.allclose(hy, hyy)


def test_multiple_layers():
    input_size = 2
    hidden_size = 3
    num_layers = 4
    rnn = nn.RNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        nonlinearity="tanh",
        bias=True,
        batch_first=True,
        dropout=0,
        bidirectional=False,
    )
    assert rnn.weight_ih_l0.shape == (hidden_size, input_size)
    assert rnn.bias_ih_l0.shape == (hidden_size,)

    assert rnn.weight_hh_l0.shape == (hidden_size, hidden_size)
    assert rnn.bias_hh_l0.shape == (hidden_size,)
    for i in range(1, num_layers):
        assert getattr(rnn, f"weight_ih_l{i}").shape == (hidden_size, hidden_size)
        assert getattr(rnn, f"weight_hh_l{i}").shape == (hidden_size, hidden_size)
        assert getattr(rnn, f"bias_ih_l{i}").shape == (hidden_size,)
        assert getattr(rnn, f"bias_hh_l{i}").shape == (hidden_size,)

    N = 2
    T = 3
    x = torch.rand(N, T, input_size)
    hx = torch.rand(num_layers, N, hidden_size)
    y, hy = rnn(x, hx)
    assert y.shape == (N, T, hidden_size)
    assert hy.shape == (num_layers, N, hidden_size)

    hyy = []
    yy = []

    prev_h = hx
    for i in range(T):
        input_x = x[:, i]
        for layer in range(num_layers):
            input_h = prev_h[layer]

            weight_ih = getattr(rnn, f"weight_ih_l{layer}")
            bias_ih = getattr(rnn, f"bias_ih_l{layer}")
            weight_hh = getattr(rnn, f"weight_hh_l{layer}")
            bias_hh = getattr(rnn, f"bias_hh_l{layer}")
            t = F.linear(input_x, weight_ih, bias_ih) + F.linear(
                input_h, weight_hh, bias_hh
            )
            t = t.tanh()

            prev_h[layer] = t
            input_x = t
        yy.append(prev_h[-1].clone())

    yy = torch.stack(yy, dim=1)
    assert y.shape == yy.shape
    assert torch.allclose(yy, y)

    assert hy.shape == prev_h.shape
    assert torch.allclose(hy, prev_h)


def test_single_layer_bidirectional():
    input_size = 2
    hidden_size = 3
    rnn = nn.RNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        nonlinearity="tanh",
        bias=True,
        batch_first=True,
        dropout=0,
        bidirectional=True,
    )
    assert rnn.weight_ih_l0.shape == (hidden_size, input_size)
    assert rnn.bias_ih_l0.shape == (hidden_size,)

    assert rnn.weight_hh_l0.shape == (hidden_size, hidden_size)
    assert rnn.bias_hh_l0.shape == (hidden_size,)

    assert rnn.weight_ih_l0_reverse.shape == (hidden_size, input_size)
    assert rnn.bias_ih_l0_reverse.shape == (hidden_size,)

    assert rnn.weight_hh_l0_reverse.shape == (hidden_size, hidden_size)
    assert rnn.bias_hh_l0_reverse.shape == (hidden_size,)

    N = 2
    T = 3
    x = torch.rand(N, T, input_size)
    num_layers = 1
    direction = 2
    hx = torch.rand(direction, num_layers, N, hidden_size).reshape(
        direction * num_layers, N, hidden_size
    )

    y, hy = rnn(x, hx)
    assert y.shape == (N, T, direction * hidden_size)
    assert hy.shape == (direction * num_layers, N, hidden_size)

    # forward
    prev_h = hx.reshape(direction, num_layers, N, hidden_size)[0][0]
    hyy = []
    yy = []
    for i in range(T):
        input_x = x[:, i]
        weight_ih = rnn.weight_ih_l0
        bias_ih = rnn.bias_ih_l0

        weight_hh = rnn.weight_hh_l0
        bias_hh = rnn.bias_hh_l0

        t = F.linear(input_x, weight_ih, bias_ih) + F.linear(prev_h, weight_hh, bias_hh)
        t = t.tanh()
        prev_h = t
        yy.append(t)

    # y is of shape (N, T, direction*hidden_size)
    # forward direction is 0
    # backward direction is 1
    forward, reverse = torch.chunk(y, chunks=2, dim=-1)
    forward2 = y.reshape(N, T, direction, hidden_size)[:, :, 0, :]
    reverse2 = y.reshape(N, T, direction, hidden_size)[:, :, 1, :]
    assert torch.allclose(forward, forward2)
    assert torch.allclose(reverse, reverse2)

    yy = torch.stack(yy, dim=1)
    assert torch.allclose(yy, forward)

    # now for the reverse direction
    yy2 = []
    prev_h_reverse = hx.reshape(direction, num_layers, N, hidden_size)[1][0]
    # x is of shape (N, T, input_size), we reverse the T axis here
    x_reverse = torch.flip(x, dims=[1])
    for i in range(T):
        input_x = x_reverse[:, i]
        weight_ih = rnn.weight_ih_l0_reverse
        bias_ih = rnn.bias_ih_l0_reverse

        weight_hh = rnn.weight_hh_l0_reverse
        bias_hh = rnn.bias_hh_l0_reverse

        t = F.linear(input_x, weight_ih, bias_ih) + F.linear(
            prev_h_reverse, weight_hh, bias_hh
        )
        t = t.tanh()
        prev_h_reverse = t
        yy2.append(t)

    yy2 = torch.stack(yy2, dim=1)
    # Note: We need to reverse yy2 again
    yy2 = torch.flip(yy2, dims=[1])
    assert torch.allclose(reverse, yy2)

    # Use torch.cat() to combine forward/reverse
    expeted_y = torch.cat((yy, yy2), dim=-1)
    assert torch.allclose(y, expeted_y)
    # prev_h is of shape (N, hidden_size)
    assert prev_h.shape == (N, hidden_size)
    expected_hy = torch.stack((prev_h, prev_h_reverse), dim=0)
    assert expected_hy.shape == (direction, N, hidden_size)

    torch.allclose(hy, expected_hy)


@torch.no_grad()
def main():
    #  test_single_layer()
    #  test_multiple_layers()
    test_single_layer_bidirectional()


if __name__ == "__main__":
    torch.manual_seed(20211202)
    main()
