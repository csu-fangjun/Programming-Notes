#!/usr/bin/env python3


def main():
    s = 'a.b.c.d'
    a = s.rpartition('.')
    assert isinstance(a, tuple)
    assert a[0] == 'a.b.c'
    assert a[1] == '.'
    assert a[2] == 'd'

    a = s.partition('.')
    assert isinstance(a, tuple)
    assert a[0] == 'a'
    assert a[1] == '.'
    assert a[2] == 'b.c.d'


if __name__ == '__main__':
    main()
