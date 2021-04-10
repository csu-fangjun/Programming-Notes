#!/usr/bin/env python3
#
# Author: Fangjun Kuang <csukuangfj@gmail.com>
#

import torch


def main():
    a = torch.tensor([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
    ])

    b = a.narrow(dim=0, start=1, length=2)
    expected = a[1:3]

    assert torch.all(torch.eq(b, expected))
    b[0, 0] = 10
    assert a[1, 0] == 10, 'Memory is shared between a and b!'


if __name__ == '__main__':
    main()
