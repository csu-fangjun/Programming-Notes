#!/usr/bin/env python3

# Author: Fangjun Kuang <csukuangfj@gmail.com>

import torch


def case1():
    log_probs = torch.rand(2, 3).log_softmax(dim=-1)

    target = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32)

    loss_func = torch.nn.KLDivLoss(reduction='none')

    loss = loss_func(input=log_probs, target=target)
    eps = torch.finfo(torch.float32).eps
    expected_loss = target * ((target + eps).log() - log_probs)

    assert loss.shape == log_probs.shape
    assert torch.allclose(loss, expected_loss)

    # ---

    target = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
    loss = loss_func(input=log_probs, target=target)
    expected_loss = target * ((target + eps).log() - log_probs)

    assert loss.shape == log_probs.shape
    assert torch.allclose(loss, expected_loss)


if __name__ == '__main__':
    torch.manual_seed(20210427)
    case1()
