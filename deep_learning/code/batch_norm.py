#!/usr/bin/env python3

# Author: Fangjun Kuang <csukuangfj@gmail.com>

import torch

from typing import Tuple


def set_weights(bn: torch.nn.Module) -> None:
    '''Initialize the scale and bias.

    Note:
      Scale is also known as weight.

    Args:
      bn:
        An instance of `torch.nn.BatchNorm1d`. It is modified in-place.
    Returns:
      Return None. The input argument is modified in-place.
    '''
    params = bn.state_dict()

    C = params['weight'].shape[0]

    assert params['weight'].shape == (C,)
    assert params['bias'].shape == (C,)
    assert params['running_mean'].shape == (C,)
    assert params['running_var'].shape == (C,)

    params['weight'] = torch.randn(C)
    params['bias'] = torch.randn(C)

    assert torch.allclose(params['running_mean'], torch.zeros(C))
    assert torch.allclose(params['running_var'], torch.ones(C))
    assert torch.allclose(params['num_batches_tracked'], torch.tensor([0]))

    bn.load_state_dict(params)


def manual_forward(scale: torch.Tensor, bias: torch.Tensor, x: torch.Tensor,
                   eps: float):
    '''
    Args:
      scale:
        A 1-d torch.Tensor of dimension (C,)
      bias:
        A 1-d torch.Tensor of dimension (C,)
      x:
        Of shape (N, C, L)
      eps:
        A float number of avoid division by zero. Its default value is usually
        1e-5.
    Returns:
      Return a torch that has the same shape as `x`.
    '''
    assert x.ndim == 3
    assert scale.shape[0] == x.shape[1] == bias.shape[0]
    C = scale.shape[0]
    ans = torch.empty_like(x)
    for c in range(C):
        mu = x[:, c, :].mean()
        var = (x[:, c, :] - mu).square().mean()
        stddev = (var + eps).sqrt()
        ans[:, c, :] = (x[:, c, :] - mu) / stddev * scale[c] + bias[c]
    return ans


def manual_statistics(running_mean: torch.Tensor, running_var: torch.Tensor,
                      momentum: float,
                      x: torch.Tensor) -> Tuple[torch.tensor, torch.tensor]:
    '''Update the running_mean and running_var using the current batch.

    Note:
      ans_mean = running_mean * (1 - momentum) + this_batch_mean * momentum
      ans_var = running_var * (1 - momentum) + this_batch_var * momentum

    Caution:
      running_mean and running_var are used only in the reference mode.
      They are not updated in the inference mode. Their value is computed
      in the training mode. The `this_batch_var` is an unbiased estimator.

    Args:
      running_mean:
        The previous running mean.
      running_var:
        The previous running var.
      momentum:
        A float number.
      x:
        The current input batch of shape (N, C, L)
    '''
    assert running_mean.ndim == running_var.ndim == 1
    assert running_mean.shape[0] == running_var.shape[0] == x.shape[1]
    assert x.ndim == 3

    C = running_mean.shape[0]
    ans_mean = torch.empty_like(running_mean)
    ans_var = torch.empty_like(running_var)
    num = x.shape[0] * x.shape[2]
    for c in range(C):
        mean = x[:, c, :].mean()
        biased_var = (x[:, c, :] - mean).square().mean()
        unbiased_var = biased_var * num / (num - 1)

        #  unbiased_var = torch.var(x[:, c, :], unbiased=True)

        ans_mean[c] = running_mean[c] * (1 - momentum) + mean * momentum
        ans_var[c] = running_var[c] * (1 - momentum) + unbiased_var * momentum

    return ans_mean, ans_var


@torch.no_grad()
def main():
    momentum = 0.1
    eps = 1e-5
    C = torch.randint(low=10, high=30, size=(1,)).item()
    bn = torch.nn.BatchNorm1d(num_features=C,
                              eps=eps,
                              momentum=momentum,
                              affine=True,
                              track_running_stats=True)
    set_weights(bn)

    running_mean = torch.zeros_like(bn.running_mean)
    running_var = torch.ones_like(bn.running_mean)
    for i in range(10):
        N = torch.randint(low=1, high=100, size=(1,))
        L = torch.randint(low=2, high=100, size=(1,))
        x = torch.rand(N, C, L)
        y = bn(x)

        assert y.shape == (N, C, L)

        expected_y = manual_forward(scale=bn.weight,
                                    bias=bn.bias,
                                    x=x,
                                    eps=eps)
        assert torch.allclose(y, expected_y, atol=1e-6)
        running_mean, running_var = manual_statistics(
            running_mean=running_mean,
            running_var=running_var,
            momentum=momentum,
            x=x)
        assert torch.allclose(running_mean, bn.running_mean)
        assert torch.allclose(running_var, bn.running_var)


if __name__ == '__main__':
    torch.manual_seed(20210329)
    main()
