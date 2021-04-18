#!/usr/bin/env python3

import contextlib


def f():
    i = 3
    yield i
    print('end')


def test():
    a = f()
    assert a.__class__.__name__ == 'generator'
    assert str(type(a)) == "<class 'generator'>"

    assert next(a) == 3


@contextlib.contextmanager
def g():
    i = 10
    print('in g')
    yield
    print('end g')


def test2():
    a = g()
    with a:
        print('end test2')
    # It prints:
    #
    # in g
    # end test2
    # end g


def foo():
    for i in range(3):
        yield i


def test3():
    for i in foo():
        print(i)
    # It prints:
    # 0
    # 1
    # 2


if __name__ == '__main__':
    test()
    test2()
    test3()
