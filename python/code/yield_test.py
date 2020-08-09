#!/usr/bin/env python3


def my_range():
    k = 0
    for i in range(3):
        yield k
        k += 2


def main():
    for i in my_range():
        print(i)


if __name__ == '__main__':
    main()
