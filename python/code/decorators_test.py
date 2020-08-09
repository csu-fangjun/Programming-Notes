#!/usr/bin/env python3

from functools import wraps


def decorator(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        f(*args, **kwargs)

    return wrapper


@decorator
def func(arg1, arg2):
    pass


def main():
    assert func.__name__ == 'wrapper'


if __name__ == '__main__':
    main()
