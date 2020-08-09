#!/usr/bin/env python3

import numpy as np

import torch
import torch.nn as nn


@torch.no_grad()
def test_rnn():
    input_size = 20
    hidden_size = 50
    w_ih = np.random.rand(hidden_size, input_size).astype(np.float32)
    bias_ih = np.random.rand(hidden_size).astype(np.float32)

    w_hh = np.random.rand(hidden_size, hidden_size).astype(np.float32)
    bias_hh = np.random.rand(hidden_size).astype(np.float32)

    seq_len = 30
    batch = 10

    in_data = np.random.randn(seq_len, batch, input_size).astype(np.float32)
    h_data = np.random.randn(batch, hidden_size).astype(np.float32)

    old_h = h_data

    h_list = []
    y_list = []

    for i in range(seq_len):
        x = in_data[i]
        #  x_gate = np.matmul(x, w_ih.T, x.T) + bias_ih
        x_gate = np.matmul(x, w_ih.T) + bias_ih
        h_gate = np.matmul(h_data, w_hh.T) + bias_hh

        h_data = np.tanh(x_gate + h_gate)
        h_list.append(h_data)

        y_data = h_data
        y_list.append(y_data)
    y = np.stack(y_list)

    old_h = np.expand_dims(old_h, axis=0)
    rnn = nn.RNN(input_size=input_size,
                 hidden_size=hidden_size,
                 num_layers=1,
                 nonlinearity='tanh',
                 bias=True,
                 batch_first=False,
                 dropout=0,
                 bidirectional=False)

    state_dict = rnn.state_dict()
    state_dict['weight_ih_l0'] = torch.from_numpy(w_ih)
    state_dict['weight_hh_l0'] = torch.from_numpy(w_hh)
    state_dict['bias_ih_l0'] = torch.from_numpy(bias_ih).squeeze()
    state_dict['bias_hh_l0'] = torch.from_numpy(bias_hh).squeeze()
    rnn.load_state_dict(state_dict)
    out_y, out_h = rnn(torch.from_numpy(in_data), torch.from_numpy(old_h))
    torch.testing.assert_allclose(out_y.numpy(), y)


@torch.no_grad()
def test_birnn():
    # bidirectional rnn
    input_size = 2
    hidden_size = 5
    w_ih = np.random.randn(hidden_size, input_size).astype(np.float32) * 1
    w_ih_reverse = np.random.randn(hidden_size, input_size).astype(np.float32) * 1

    b_ih = np.random.randn(hidden_size).astype(np.float32) * 1
    b_ih_reverse = np.random.randn(hidden_size).astype(np.float32) * 1

    w_hh = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 1
    w_hh_reverse = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 1

    b_hh = np.random.randn(hidden_size).astype(np.float32) * 1
    b_hh_reverse = np.random.randn(hidden_size).astype(np.float32) * 1

    seq_len = 2
    batch = 3

    in_data = np.random.randn(seq_len, batch, input_size).astype(np.float32) * 1
    h_data = np.random.randn(2, batch, hidden_size).astype(np.float32) * 1

    old_h = h_data.copy()

    h_list = []
    h_list_reverse = []
    y_list = []
    y_list_reverse = []
    for i in range(seq_len):
        x_gate = np.matmul(in_data[i], w_ih.T) + b_ih
        h_gate = np.matmul(h_data[0], w_hh.T) + b_hh

        h_data[0] = np.tanh(x_gate + h_gate)
        h_list.append(h_data[0].copy())
        y_list.append(h_data[0].copy())
    y = np.stack(y_list)
    h = np.stack(h_list)

    # for the reverse
    for i in range(seq_len):
        x_gate = np.matmul(in_data[seq_len - 1 - i], w_ih_reverse.T) + b_ih_reverse
        h_gate = np.matmul(h_data[1], w_hh_reverse.T) + b_hh_reverse
        h_data[1] = np.tanh(x_gate + h_gate)
        h_list_reverse.append(h_data[1].copy())
        y_list_reverse.append(h_data[1].copy())
    h_list_reverse.reverse()
    y_list_reverse.reverse()
    y_reverse = np.stack(y_list_reverse)
    h_reverse = np.stack(h_list_reverse)

    h = np.stack([h, h_reverse])

    y = np.concatenate([y, y_reverse], axis=-1)

    rnn = nn.RNN(input_size=input_size,
                 hidden_size=hidden_size,
                 num_layers=1,
                 nonlinearity='tanh',
                 bias=True,
                 batch_first=False,
                 dropout=0,
                 bidirectional=True)

    state_dict = rnn.state_dict()
    state_dict['weight_ih_l0'] = torch.from_numpy(w_ih)
    state_dict['weight_hh_l0'] = torch.from_numpy(w_hh)
    state_dict['bias_ih_l0'] = torch.from_numpy(b_ih)
    state_dict['bias_hh_l0'] = torch.from_numpy(b_hh)

    state_dict['weight_ih_l0_reverse'] = torch.from_numpy(w_ih_reverse)
    state_dict['weight_hh_l0_reverse'] = torch.from_numpy(w_hh_reverse)
    state_dict['bias_ih_l0_reverse'] = torch.from_numpy(b_ih_reverse).squeeze()
    state_dict['bias_hh_l0_reverse'] = torch.from_numpy(b_hh_reverse).squeeze()

    rnn.load_state_dict(state_dict)

    out_y, out_h = rnn(torch.from_numpy(in_data), torch.from_numpy(old_h))
    torch.testing.assert_allclose(out_y, torch.from_numpy(y))
    # for the forward rnn, H is the last time step T-1
    # for the backward rnn, H is the 0th time step 0
    torch.testing.assert_allclose(out_h[0], torch.from_numpy(h[0][-1]))
    torch.testing.assert_allclose(out_h[1], torch.from_numpy(h[1][0]))


if __name__ == '__main__':
    np.random.seed(20200807)
    test_rnn()
    test_birnn()
