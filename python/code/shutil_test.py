#!/usr/bin/env python3

import shutil


def test1():
    a = shutil.which('ls')
    assert isinstance(a, str)
    assert a == '/bin/ls'

    a = shutil.which('ls2')
    assert a is None


def main():
    test1()


if __name__ == '__main__':
    main()
