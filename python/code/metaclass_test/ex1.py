#!/usr/bin/env python3


class Foo(object):
    pass


#  print(help(type))
assert Foo.__name__ == "Foo"
assert Foo.__class__ == type
assert Foo.__bases__ == (object,)
assert Foo.__dict__["__module__"] == "__main__"
assert Foo.__module__ == "__main__"

Tom = type("Tom2", (), {})
t = Tom()
#  assert t.__name__ == "Tom2"
assert not hasattr(t, "__name__")
assert t.__dict__ == {}
assert str(t.__class__) == "<class '__main__.Tom2'>"


class Bar:
    a = 10


b1 = Bar()
b2 = Bar()
assert b1.a == b2.a == 10

b1.a = 100
assert b1.a == 100
assert b2.a == 10

Bar.a = 20
assert b1.a == 100
assert b2.a == 20

b3 = Bar()
assert b3.a == 20
