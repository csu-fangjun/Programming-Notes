#!/usr/bin/env python3
#
# Author: Fangjun Kuang <csukuangfj@gmail.com>
#

import torch


def test_1dim():
    #                 0, 1, 2, 3
    a = torch.tensor([3, 2, 0, 10])
    b = torch.topk(a, k=1)
    # b is a subclass of named tuple
    assert isinstance(b, tuple)
    assert hasattr(b, 'values')
    assert hasattr(b, 'indices')

    assert torch.all(torch.eq(b.values, torch.tensor([10])))
    assert torch.all(torch.eq(b.indices, torch.tensor([3])))

    values, indices = torch.topk(a, k=1)
    assert torch.all(torch.eq(values, torch.tensor([10])))
    assert torch.all(torch.eq(indices, torch.tensor([3])))

    b = a.topk(k=1)

    assert torch.all(torch.eq(b.values, torch.tensor([10])))
    assert torch.all(torch.eq(b.indices, torch.tensor([3])))

    # Now for the least
    b = torch.topk(a, k=1, largest=False)
    assert torch.all(torch.eq(b.values, torch.tensor([0])))
    assert torch.all(torch.eq(b.indices, torch.tensor([2])))

    b = a.topk(k=1, largest=False)
    assert torch.all(torch.eq(b.values, torch.tensor([0])))
    assert torch.all(torch.eq(b.indices, torch.tensor([2])))

    # for k = 2
    b = torch.topk(a, k=2)
    assert torch.all(torch.eq(b.values, torch.tensor([10, 3])))
    assert torch.all(torch.eq(b.indices, torch.tensor([3, 0])))

    b = a.topk(k=2)
    assert torch.all(torch.eq(b.values, torch.tensor([10, 3])))
    assert torch.all(torch.eq(b.indices, torch.tensor([3, 0])))


def test_two_dim():
    a = torch.tensor([
        [1, 5, 3, 2],
        [6, 2, 9, 8],
        [1, 3, 6, 7],
    ])
    b = torch.topk(a, k=1)
    # for each row, take it topk
    assert torch.all(torch.eq(b.values, torch.tensor([[5], [9], [7]])))
    assert torch.all(torch.eq(b.indices, torch.tensor([[1], [2], [3]])))

    b = torch.topk(a, k=1, dim=0)
    #  For each column, take it topk
    assert torch.all(torch.eq(b.values, torch.tensor([[6, 5, 9, 8]])))
    assert torch.all(torch.eq(b.indices, torch.tensor([[1, 0, 1, 1]])))


def main():
    test_1dim()
    test_two_dim()


if __name__ == '__main__':
    main()
