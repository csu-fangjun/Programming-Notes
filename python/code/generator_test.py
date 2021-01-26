#!/usr/bin/env python3

import contextlib


def f():
    i = 3
    yield i
    print('end')


def test():
    a = f()

    assert str(type(a)) == "<class 'generator'>"
    assert next(a) == 3


@contextlib.contextmanager
def g():
    i = 10
    yield
    print('end g')


def test2():
    a = g()
    with a:
        print('end test2')


if __name__ == '__main__':
    test()
    test2()
