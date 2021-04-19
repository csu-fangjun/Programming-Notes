#!/usr/bin/env python3
'''
def foo(sources):
    for src in sources:
        for item in src:
            yield item

is equivalent to:

def foo(sources):
    for src in sources:
        yield from src
'''


def foo():
    for i in range(3):
        yield from list(range(i + 1))


def main():
    f = foo()
    for i in f:
        print(i, type(i))
    #  print(next(f))
    #  print(list(f))


if __name__ == '__main__':
    main()
