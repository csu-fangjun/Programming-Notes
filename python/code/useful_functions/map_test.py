#!/usr/bin/env python3

from collections.abc import Iterable


def main():
    a = [1, 2, 3]

    def my_func(i):
        return i + 1

    b = map(my_func, a)
    assert isinstance(b, Iterable)
    assert list(b) == [2, 3, 4]

    c = map(lambda i: i + 2, a)
    assert list(c) == [3, 4, 5]


if __name__ == '__main__':
    main()
