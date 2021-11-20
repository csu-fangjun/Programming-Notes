#!/usr/bin/env python3
import torch
from typing import Final


class Foo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.i = 10
        self.s = "hello"
        self.f = 0.5

        self.k = 100

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 3

    @torch.jit.export
    def test2(self, y: torch.Tensor, k: int) -> torch.Tensor:
        return y + k


def main():
    m = torch.jit.script(Foo())
    print(m)
    print(m.i)
    print(m.s)
    print(m.f)
    print(m.k)
    print(m.graph)
    print(m.test2.graph)
    m.save("foo.pt")


if __name__ == "__main__":
    main()
