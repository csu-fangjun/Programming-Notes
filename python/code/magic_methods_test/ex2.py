#!/usr/bin/env python3


class Foo(object):
    # note: __new__ is a classmethod by default
    # so its first argument is the class
    def __new__(cls, *args, **kwargs):
        print(f'called Foo __new__: args = {args}, kwargs = {kwargs}')
        ans = object.__new__(cls)  # ok
        print(id(ans))
        return ans
        #  return super(Foo, cls).__new__(cls)  # ok
        #  return super(Foo, cls).__new__(cls)  # ok

    def __init__(self, *args, **kwargs):
        print(f'In init: {id(self)}')
        pass


def main():
    f1 = Foo()
    print(f'f1: {id(f1)}')

    f2 = Foo(1, 2, a=3, b=4)
    print(f'f2: {id(f2)}')


if __name__ == '__main__':
    main()
