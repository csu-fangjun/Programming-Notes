#!/usr/bin/env python3

import hello

print(hello.add(1, 2))
print(dir(hello))


def test_vector():
    v = hello.create_int_vec()
    s = hello.print_int_vec(v)
    print(s)
    print(type(v))

    hello.append_int_vec(v, 10)
    s = hello.print_int_vec(v)
    print(s)


def test_pointers():
    hello.inc(1000)  # print the same value for pointers
    hello.inc(2000)
    hello.inc(3000)

    print(help(hello.Hi))
    print(help(hello.inc))


def main():
    test_vector()
    test_pointers()


if __name__ == '__main__':
    main()
