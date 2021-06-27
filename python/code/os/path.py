#!/usr/bin/env python3
import os

# See
# https://docs.python.org/3/library/os.path.html

def main():
    d = os.path.dirname('./a/b/c.py')
    assert d == './a/b'

    d = os.path.join('a', 'b')
    assert d == 'a/b'

    assert os.path.exists('/tmp')
    assert os.path.exists('./to-delete') is False
    os.mkdir('./to-delete')
    assert os.path.exists('./to-delete') is True
    os.rmdir('./to-delete')

    files = os.listdir()
    assert isinstance(files, list)

    d = os.path.abspath('/tmp/a/..')
    assert d == '/tmp'

    d = os.path.basename('./a/b.txt')
    assert d == 'b.txt'

    d = os.path.dirname('./a/b.txt')
    assert d == './a'

    assert os.path.exists('./a.txt') is False
    assert os.path.exists('./a/b/c') is False

    assert os.path.isfile('./a.txt') is False
    assert os.path.isdir('./a/b') is False
    assert os.path.isdir('../') is True

    a, b = os.path.split('./a/b/c.txt')
    assert a == './a/b'
    assert b == 'c.txt'

    a, b = os.path.splitext('./a/b/c.txt')
    assert a == './a/b/c'
    assert b == '.txt'

    a, b = os.path.splitext('c.txt')
    assert a == 'c'
    assert b == '.txt'

    a, b = os.path.splitext('c.b.a.txt')
    assert a == 'c.b.a'
    assert b == '.txt'



if __name__ == '__main__':
    main()
