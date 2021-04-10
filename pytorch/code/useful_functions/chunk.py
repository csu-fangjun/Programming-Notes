#!/usr/bin/env python3
#
# Author: Fangjun Kuang <csukuangfj@gmail.com>
#

import torch


def main():
    a = torch.tensor([
        [1, 2, 30, 4],
        [3, 4, 50, 6],
        [5, 6, 70, 8],
        [7, 8, 90, 10],
    ])

    b = a.chunk(chunks=2, dim=-1)

    assert isinstance(b, tuple)
    expected_b0 = torch.tensor([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
    ])
    assert torch.all(torch.eq(b[0], expected_b0))

    expected_b1 = torch.tensor([
        [30, 4],
        [50, 6],
        [70, 8],
        [90, 10],
    ])
    assert torch.all(torch.eq(b[1], expected_b1))


if __name__ == '__main__':
    main()
