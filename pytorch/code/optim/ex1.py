#!/usr/bin/env python3

import torch
import torch.nn as nn


class Foo(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(2, 3)
        self.b = nn.Linear(3, 4)
        self.c = nn.Linear(3, 4)
        self.d = nn.Linear(3, 4)


def test1():
    f = Foo()
    params = f.parameters()
    assert str(type(params)) == "<class 'generator'>"
    for p in params:
        assert isinstance(p, torch.nn.parameter.Parameter)


def test_constructor():
    # From torch/optim/optimizer.py
    # The constructor accepts
    #  (1) A list of tensors, e.g., params=[t1, t2]
    #  (2) A list of dict, i.e,
    #        params = [{ "params": [tensor1, tensor2]},
    #                  {"params": [tensor3], lr=1e-3}
    #                  {"params": tensor4, lr=1e-3}
    #                 ]
    #  each dict is a parameter group.
    #  each parameter group is must contain the key "params".
    # Optional, it can contain some extra `keys`. These
    # keys are the same as the constructor arguments
    # of the given optimizer.
    #
    # If the value of "params" contains only a single tensor,
    # it does not need to be a list.

    foo = Foo()
    opt = torch.optim.SGD(
        params=[
            {"params": [foo.a.weight, foo.a.bias]},
            {"params": foo.c.weight, "lr": 0.5},
            {"params": [foo.d.weight, foo.d.bias]},
        ],
        lr=0.2,
    )
    # Look at the string representation of `opt`
    print(opt)
    # Note that:
    #  param group 0 and 2 have the default lr=0.2
    #  param group 1 has the provided lr=0.5
    print(opt.state_dict())


def main():
    test1()
    test_constructor()


if __name__ == "__main__":
    main()
