#!/usr/bin/env python3

# See https://docs.python.org/3/library/functions.html#max


def test1():
    # accepts two argument
    a = '321'
    b = '0123'
    c = max(a, b, key=len)
    assert c == b

    def my_compare(x):
        print('x is', x, type(x))
        return int(x)

    c = max(a, b, key=my_compare)
    assert c == a

    # accepts three argument

    c = '00042'
    d = max(a, b, c, key=len)
    assert d == c

    print('-' * 3)
    d = max(a, b, c, key=my_compare)
    assert d == a
    print('-- end test1 --')


def test2():
    # use a iterable
    a = '0a12'
    b = 'aa'
    c = 'b'
    d = max((a, b, c), key=len)
    assert d == a

    # two args with an iterable
    # Note that we have to use *(a,), not (a,)
    # If we use (a,), then len((a,)) is 1
    d = max(b, b, *(a,), key=len)
    assert d == a


def main():
    test1()
    test2()


if __name__ == '__main__':
    main()
