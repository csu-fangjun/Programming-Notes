#!/usr/bin/env python3

import os
from pathlib import Path


def test1():
    p = Path('/a/b/c.txt')
    assert p.name == 'c.txt'
    assert p.suffix == '.txt'
    assert p.root == '/'
    assert p.parts == ('/', 'a', 'b', 'c.txt')
    assert p.is_file() is False
    assert p.exists() is False
    assert p.stem == 'c'
    assert p.is_dir() is False

    p = Path('a/b/c')
    assert p.name == 'c'
    assert p.suffix == ''
    assert p.root == ''
    assert p.parts == ('a', 'b', 'c')
    assert p.is_file() is False
    assert p.exists() is False
    assert p.stem == 'c'
    assert p.is_dir() is False

    p = Path('a.txt')
    assert p.name == 'a.txt'
    assert p.suffix == '.txt'
    assert p.root == ''
    assert p.parts == ('a.txt',)
    assert p.is_file() is False
    assert p.exists() is False
    assert p.stem == 'a'
    assert p.is_dir() is False

    p = Path('a', 'b')
    assert str(p) == 'a/b'

    p = Path('a') / 'b'
    assert str(p) == 'a/b'
    q = p.with_name('c.txt')
    assert str(q) == 'a/c.txt'
    p = q.with_suffix('.md')
    assert str(p) == 'a/c.md'


def test2():
    p = Path('..')

    # no-recursive, iterate over all files and directories in that directory
    #
    # CAUTION: It is no recursive and it displays both directories and regular files.
    for files_and_dirs in p.iterdir():
        assert isinstance(files_and_dirs, Path)

    # to get all files
    files = [f for f in p.iterdir() if f.is_file()]

    # to list all files matching a given patthern
    #
    #  non-recursive
    p = Path('..')
    files = p.glob('*.py')
    # files is a generato
    print(files)


def main():
    test1()
    test2()

    #  print(os.getcwd()) # /xxx/xxx/xxx/python/code/pathlib_test

    assert str(Path.cwd()) == os.getcwd()

    assert str(Path('.')) == '.'

    # resolve -> make a absolute path
    assert str(Path('.').resolve()) == os.getcwd()
    assert __file__ == './ex1.py'

    assert str(Path(__file__).parent) == '.'
    assert str(Path(__file__).parent.resolve()) == os.getcwd()
    assert str(Path(__file__).resolve().parent) == os.getcwd()

    assert str(Path(__file__).resolve()) == os.path.realpath(__file__)

    assert str(Path('a') / 'b' / 'c') == 'a/b/c'

    assert Path('abc/def.txt').stem == 'def'


if __name__ == '__main__':
    main()
