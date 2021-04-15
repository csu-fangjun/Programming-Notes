#!/usr/bin/env python3

import decimal


def main():
    a = 1.1
    b = 1.1 * 2
    c = a + b
    print(c)

    a = decimal.Decimal('1.1')
    b = decimal.Decimal('1.1') * 2
    c = a + b
    print(c)


if __name__ == '__main__':
    main()
