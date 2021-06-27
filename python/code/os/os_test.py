#!/usr/bin/env python3

import os


def main():
    print(type(os.environ))  # <class os._Environ>, which inherits from collections.abc.MutableMapping
    print(help(os.environ))


if __name__ == '__main__':
    main()
