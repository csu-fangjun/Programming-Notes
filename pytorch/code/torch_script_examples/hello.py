#!/usr/bin/env python3
#
# Author: Fangjun Kuang <csukuangfj@gmail.com>
#

import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([1, 2.0, 3]), requires_grad=False)

    def forward(self, x):
        if x.sum() < 0:
            return 10 * x * self.w
        else:
            return 2 * x * self.w

    @torch.jit.export
    def hello(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


def test_trace():
    model = MyModel()
    x = torch.tensor([10, 20, 30])
    traced_model = torch.jit.trace(model, x)
    traced_model.eval()
    print(traced_model, type(traced_model))
    print(str(type(traced_model)))
    assert (
        str(type(traced_model)) == "<class 'torch.jit._script.RecursiveScriptModule'>"
    )
    print(traced_model.graph)
    print(traced_model.code)
    print(traced_model(x))


def test_script():
    model = MyModel()
    # a script model can capture if/else, while a traced model cannot.
    scripted_model = torch.jit.script(model)
    assert (
        str(type(scripted_model)) == "<class 'torch.jit._script.RecursiveScriptModule'>"
    )
    print(scripted_model)
    print(type(scripted_model))
    print(scripted_model.run_method)

    x = torch.tensor([10, 20, 30])
    traced_model = torch.jit.trace(scripted_model, x)
    print(traced_model.graph)
    print(traced_model.code)


def main():
    test_trace()
    test_script()


if __name__ == "__main__":
    main()
