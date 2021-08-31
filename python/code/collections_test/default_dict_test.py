#!/usr/bin/env python3
from collections import defaultdict


def main():
    a = defaultdict(int)
    assert a['hello'] == 0
    a.default_factory == int
    assert int() == 0
    assert a['foo'] == int()
    assert a['foo'] == a.default_factory()


if __name__ == '__main__':
    main()
