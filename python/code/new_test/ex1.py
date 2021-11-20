#!/usr/bin/env python3

# this file test __new__


class Foo:
    def __new__(cls, *args, **kwargs):
        print("cls", cls)
        print("args", args)
        print("kwargs", kwargs)
        # Note: super().__new__() takes only one argument!
        ans = super().__new__(cls)
        print("ans", id(ans), type(ans))
        return ans

    def __init__(self, *args, **kwargs):
        # this function is invoked automatically by ``__new__``,
        # as long as __new__ returns an instance of type `Foo`.
        print("id(self)", id(self))
        print("args", args)
        print("kwargs", kwargs)


f1 = Foo()
print("-" * 10)
f2 = Foo("a")
print("-" * 10)
f3 = Foo("a", "b")
print("-" * 10)
f4 = Foo("a", "b", c=1)
