#!/usr/bin/env python3
#
# Author: Fangjun Kuang <csukuangfj@gmail.com>
#

import torch


def test_1dim():
    a = torch.tensor([1, 2])
    b = torch.nn.functional.pad(a, pad=(0, 1))  # padding right
    assert torch.all(torch.eq(b, torch.tensor([1, 2, 0])))

    b = torch.nn.functional.pad(a, pad=(0, 3))
    assert torch.all(torch.eq(b, torch.tensor([1, 2, 0, 0, 0])))

    b = torch.nn.functional.pad(a, pad=(1, 0))  # padding left
    assert torch.all(torch.eq(b, torch.tensor([0, 1, 2])))

    b = torch.nn.functional.pad(a, pad=(3, 0))
    assert torch.all(torch.eq(b, torch.tensor([0, 0, 0, 1, 2])))

    # padding with a given value

    b = torch.nn.functional.pad(a, pad=(1, 2), value=-1)
    assert torch.all(torch.eq(b, torch.tensor([-1, 1, 2, -1, -1])))


def test_2dim():
    a = torch.tensor([
        [1, 2],
        [3, 4],
        [5, 6],
    ])
    b = torch.nn.functional.pad(a, pad=(0, 1))  # padding right
    expected = torch.tensor([
        [1, 2, 0],
        [3, 4, 0],
        [5, 6, 0],
    ])
    assert torch.all(torch.eq(b, expected))

    b = torch.nn.functional.pad(a, pad=(2, 0))  # padding left
    expected = torch.tensor([
        [0, 0, 1, 2],
        [0, 0, 3, 4],
        [0, 0, 5, 6],
    ])
    assert torch.all(torch.eq(b, expected))

    b = torch.nn.functional.pad(a, pad=(0, 0, 1, 0))  # padding top
    expected = torch.tensor([
        [0, 0],
        [1, 2],
        [3, 4],
        [5, 6],
    ])
    assert torch.all(torch.eq(b, expected))

    # (left, right, top, bottom)
    b = torch.nn.functional.pad(a, pad=(0, 0, 0, 1))  # padding bottom
    expected = torch.tensor([
        [1, 2],
        [3, 4],
        [5, 6],
        [0, 0],
    ])
    assert torch.all(torch.eq(b, expected))

    b = torch.nn.functional.pad(a, pad=(1, 2, 3,
                                        4))  # padding left/right/top.bottom
    expected = torch.tensor([
        #1        1  2
        [0, 0, 0, 0, 0],  # 3
        [0, 0, 0, 0, 0],  # 2
        [0, 0, 0, 0, 0],  # 1
        [0, 1, 2, 0, 0],
        [0, 3, 4, 0, 0],
        [0, 5, 6, 0, 0],
        [0, 0, 0, 0, 0],  # 1
        [0, 0, 0, 0, 0],  # 2
        [0, 0, 0, 0, 0],  # 3
        [0, 0, 0, 0, 0],  # 4
    ])
    assert torch.all(torch.eq(b, expected))


def main():
    test_1dim()
    test_2dim()


if __name__ == '__main__':
    main()
