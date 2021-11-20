#!/usr/bin/env python3

import torch

a = torch.arange(3)
torch.save(a, "bb.pt")
torch.save(a, "bb2.pt")
torch.save(a, "bb3.pt", _use_new_zipfile_serialization=False)

import sys

sys.exit(0)

torch.classes.load_library("build/libmy_stack.so")
assert isinstance(torch.classes.loaded_libraries, set)
print(torch.classes.loaded_libraries)

stack = torch.classes.foo.MyStack([10, 20])

#  torch.save(stack, "ab.pt")


class My(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = torch.classes.foo.MyStack([10, 200])


my = torch.jit.script(My())
my.save("ab2.pt")
