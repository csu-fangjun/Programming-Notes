#!/usr/bin/env python3

import sys


def main():
    # major=3, minor=5, micro=2
    print(sys.version_info)  # it is a subclass of a named tuple
    print(sys.version[:5])  # 3.5.2, it is a str

    # now for args
    assert isinstance(sys.argv, list)
    print(sys.argv[0])  # the program name, ./sys_test.py


if __name__ == '__main__':
    main()
