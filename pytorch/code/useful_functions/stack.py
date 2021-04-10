#!/usr/bin/env python3
#
# Author: Fangjun Kuang <csukuangfj@gmail.com>
#
import torch


def main():
    a = torch.tensor([1, 2])
    b = torch.tensor([3, 4])
    # default dim=0
    c = torch.stack([a, b])
    assert c.ndim == 2
    expected_c = torch.tensor([
        [1, 2],
        [3, 4],
    ])
    assert torch.all(torch.eq(c, expected_c))

    c = torch.stack([a, b], dim=1)
    assert c.ndim == 2
    expected_c = torch.tensor([
        [1, 3],
        [2, 4],
    ])
    assert torch.all(torch.eq(c, expected_c))

    # now for 2-dim
    a = torch.tensor([
        [1, 2],
        [3, 4],
    ])
    b = torch.tensor([
        [5, 6],
        [7, 8],
    ])

    c = torch.stack([a, b])
    expected_c = torch.tensor([[
        [1, 2],
        [3, 4],
    ], [
        [5, 6],
        [7, 8],
    ]])
    assert torch.all(torch.eq(c, expected_c))

    c = torch.stack([a, b], dim=1)
    expected_c = torch.tensor([[
        [1, 2],
        [5, 6],
    ], [
        [3, 4],
        [7, 8],
    ]])
    assert torch.all(torch.eq(c, expected_c))


if __name__ == '__main__':
    main()
