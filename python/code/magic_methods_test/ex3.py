#!/usr/bin/env python3


class Float(int):

    def __new__(cls, *args):
        print(f'args is: {args}')
        return super().__new__(cls, *args)


def main():
    f = Float(3)
    assert f == 3
    f = Float(3, 4)  # TypeError: float expected at most 1 argument, got 2
    print(f)


if __name__ == '__main__':
    main()
