#!/usr/bin/env python3

# Author: Fangjun Kuang <csukuangfj@gmail.com>

import torch
import torch.nn as nn
import torch.nn.functional as F


def main():
    # number of classes
    C = 3
    x = torch.rand(4, C)
    y = F.log_softmax(x, dim=1)
    # C is 3, so valid values are [0, 1, 2]
    # There are 4 batches, so len(target) == 4
    target = torch.tensor([1, 0, 1, 2])
    loss = nn.NLLLoss()(y, target)

    # The first item y[0], the class is 1, so it uses y[0][1]
    # The second item y[1], the class is 0, so it uses y[1][0]
    # The third item y[2], the class is 1, so it uses y[2][1]
    # The fourth item y[3], the class is 2, so it uses y[3][2]
    # divide by 4 to get its mean
    # multiplied by (-1), because it contains negative in its name
    expected_loss = (y[0][1] + y[1][0] + y[2][1] + y[3][2]) / 4 * (-1)
    assert torch.allclose(loss, expected_loss)

    # every class has a weigth, len(weight) == C
    w = torch.tensor([10, 20, 30.])
    loss = nn.NLLLoss(weight=w)(y, target)

    # Note that the denominator in the mean is `w[target].sum()` !!!
    expected_loss = \
            (y[0][1] *w[1] + y[1][0] * w[0] + y[2][1]*w[1] + y[3][2]*w[2]) / w[target].sum() * (-1)
    assert torch.allclose(loss, expected_loss)

    # Now with ignore index

    loss = nn.NLLLoss(ignore_index=1)(y, target)
    # The class 1 is ignored
    expected_loss = (y[1][0] + y[3][2]) / 2 * (-1)
    assert torch.allclose(loss, expected_loss)

    # class 2 is ignored
    loss = nn.NLLLoss(weight=w, ignore_index=2)(y, target)
    expected_loss = \
            (y[0][1] *w[1] + y[1][0] * w[0] + y[2][1]*w[1]) / w[torch.tensor([1, 0, 1])].sum() * (-1)
    assert torch.allclose(loss, expected_loss)


if __name__ == '__main__':
    main()
