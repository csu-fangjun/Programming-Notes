#!/usr/bin/env python3

import weakref


class Foo:
    pass


def test1():

    def mycalback(f: weakref):
        print(f'callback: {hex(id(f))}, {f}')
        # Note the object is already dead
        assert f() is None

    f = Foo()
    print(hex(id(f)))
    r = weakref.ref(f, mycalback)
    assert r.__class__.__name__ == 'weakref'

    assert weakref.getweakrefcount(f) == 1
    refs = weakref.getweakrefs(f)
    assert len(refs) == 1
    assert id(refs[0]) == id(r)

    print(r.__callback__)  # Not None
    print(r)

    # if the object is still alive, r() returns that object
    print('hex(id(r()))', hex(id(r())))
    assert id(r()) == id(f)
    del f  # will invoke the callback

    # the object is dead, so r() returns None and its callback is also None
    assert r() is None
    print('r()', r())
    print(r)
    print(r.__callback__)  # None
    '''
    0x7f31d172fb20
    <function test1.<locals>.mycalback at 0x7f31cfa6d8b0>
    <weakref at 0x7f31d1719d10; to 'Foo' at 0x7f31d172fb20>
    callback: 0x7f31d1719d10, <weakref at 0x7f31d1719d10; dead>
    <weakref at 0x7f31d1719d10; dead>
    None
    '''


def main():
    test1()


if __name__ == '__main__':
    main()
