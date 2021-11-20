#!/usr/bin/env python3

from collections import OrderedDict


def main():
    d = OrderedDict()
    d["a"] = 10
    d["c"] = 20
    d["b"] = 30

    # The order of the key is determined by the insertion order
    assert list(d.keys()) == ["a", "c", "b"]

    d.move_to_end("a")
    assert list(d.keys()) == ["c", "b", "a"]

    # re-assign the key 'c' does not change its order
    d["c"] = 100

    assert list(d.keys()) == ["c", "b", "a"]


# In PyTorch, nn.modules._module is an instance
# of OrderedDict.
#
# The dict is used by nn.Sequence()

if __name__ == "__main__":
    main()
