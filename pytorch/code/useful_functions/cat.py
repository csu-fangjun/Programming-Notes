#!/usr/bin/env python3
#
# Author: Fangjun Kuang <csukuangfj@gmail.com>
#
import torch


def main():
    a = torch.tensor([1, 2])
    b = torch.tensor([3, 4, 5])
    c = torch.cat([a, b])
    assert torch.all(torch.eq(c, torch.tensor([1, 2, 3, 4, 5])))

    # for 2-d
    a = torch.tensor([
        [1, 2],
        [3, 4],
    ])

    b = torch.tensor([
        [5, 6],
        [7, 8],
    ])

    # default dim=0
    c = torch.cat([a, b])
    expected_c = torch.tensor([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
    ])
    assert torch.all(torch.eq(c, expected_c))

    c = torch.cat([a, b], dim=1)
    expected_c = torch.tensor([
        [1, 2, 5, 6],
        [3, 4, 7, 8],
    ])
    assert torch.all(torch.eq(c, expected_c))


if __name__ == '__main__':
    main()
