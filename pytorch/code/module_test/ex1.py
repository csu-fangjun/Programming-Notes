#!/usr/bin/env python3

import torch
import torch.nn as nn


class Ex1(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.register_buffer('buf', torch.tensor([10]))


def main():
    ex1 = Ex1()
    #  w = ex1.get_parameter('linear.weight') # available only in master (2021-04-16)
    w = ex1.linear.weight
    print(w)

    # For each submodule m in ex1:
    # call init(m)
    #
    def init(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
        elif isinstance(m, Ex1):
            nn.init.ones_(m.buf)

    ex1.apply(init)
    print(w)
    print(ex1.buf)


if __name__ == '__main__':
    main()
