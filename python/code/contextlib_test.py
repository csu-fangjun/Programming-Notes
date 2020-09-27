#!/usr/bin/env python3

import contextlib


class Foo:

    def __enter__(self):
        print(self)
        print("entered!")
        return 10
        pass

    def __exit__(self, *args):
        print(self)
        print("exited!")
        pass
        return True

    pass


def main():
    print(contextlib.__file__)
    with Foo() as f:
        print(f)
        raise ValueError('abc')
        pass
    print('here')


if __name__ == '__main__':
    main()
