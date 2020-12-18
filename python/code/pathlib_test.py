#!/usr/bin/env python3

import os
from pathlib import Path


def main():
    assert str(Path('.')) == '.'
    assert str(Path('.').resolve()) == os.getcwd()
    assert __file__ == './pathlib_test.py'

    assert str(Path(__file__).parent) == '.'
    assert str(Path(__file__).parent.resolve()) == os.getcwd()
    assert str(Path(__file__).resolve().parent) == os.getcwd()

    assert str(Path(__file__).resolve()) == os.path.realpath(__file__)

    assert str(Path('a') / 'b' / 'c') == 'a/b/c'


if __name__ == '__main__':
    main()
